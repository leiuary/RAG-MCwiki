"""Embedding 后端实现 —— HuggingFace / ONNX / GGUF，均遵循 LangChain Embeddings 协议。"""
import logging
from typing import List

logger = logging.getLogger(__name__)


# ── HuggingFace 后端（原版 Qwen3 / text2vec 等） ──

class HuggingFaceEmbeddingBackend:
    def __init__(self, preset):
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f"加载 HuggingFace 嵌入模型: {preset.model_name_or_path}")
        self._model_id = preset.id
        self._progress_cb = None       # 外部可设置，用于批次进度回调
        self._hf = HuggingFaceEmbeddings(
            model_name=preset.model_name_or_path,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 2},
        )
        try:
            import torch
            self._device = "CUDA" if torch.cuda.is_available() else "CPU"
        except ImportError:
            self._device = "CPU"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._hf.embed_documents(texts)
        if self._progress_cb:
            self._progress_cb(len(texts), len(texts))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._hf.embed_query(text)


# ── ONNX 后端（gte-small-zh INT8 量化） ──

class ONNXEmbeddingBackend:
    def __init__(self, preset):
        import onnxruntime as ort
        import huggingface_hub
        from transformers import AutoTokenizer

        logger.info(f"加载 ONNX 嵌入模型: {preset.model_name_or_path}/{preset.onnx_filename}")

        onnx_path = huggingface_hub.hf_hub_download(
            repo_id=preset.model_name_or_path,
            filename=preset.onnx_filename,
        )
        self._model_id = preset.id
        self._progress_cb = None       # 外部可设置，用于微批次进度回调

        # 优先 GPU → CPU
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]
        self._device = providers[0].replace("ExecutionProvider", "") if providers else "CPU"
        logger.info(f"ONNX Runtime providers: {available} → 选用 {providers}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(preset.model_name_or_path)
        self._dim = preset.dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        micro_batch = 64
        all_embeddings: List[List[float]] = []
        total = len(texts)

        for start in range(0, total, micro_batch):
            chunk = texts[start : start + micro_batch]
            inputs = self._tokenizer(
                chunk, padding=True, truncation=True, return_tensors="np", max_length=512
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"]
            model_input_names = [i.name for i in self._session.get_inputs()]
            ort_inputs = {k: v for k, v in ort_inputs.items() if k in model_input_names}

            outputs = self._session.run(None, ort_inputs)
            token_embeddings = outputs[0]

            # Mean pooling
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, axis=-1)
            mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)
            summed = np.sum(token_embeddings * mask_expanded, axis=1)
            counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = summed / counts

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            all_embeddings.extend(embeddings.tolist())

            # 微批次进度回调
            if self._progress_cb:
                self._progress_cb(start + len(chunk), total)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


# ── GGUF 后端（Qwen3 Q3_K_M 量化） ──

class GGUFEmbeddingBackend:
    def __init__(self, preset):
        from llama_cpp import Llama
        import huggingface_hub

        self._model_id = preset.id
        self._progress_cb = None       # 外部可设置，用于逐条进度回调
        local_path = huggingface_hub.hf_hub_download(
            repo_id=preset.model_name_or_path,
            filename=preset.gguf_filename,
        )
        logger.info(f"加载 GGUF 嵌入模型: {local_path}")
        self._llm = Llama(
            model_path=local_path,
            embedding=True,
            n_ctx=512,
            n_gpu_layers=-1,  # -1 = 全部 offload 到 GPU，0 = 纯 CPU
            verbose=False,
        )
        try:
            import torch
            self._device = "CUDA" if torch.cuda.is_available() else "CPU"
        except ImportError:
            self._device = "CPU"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            resp = self._llm.create_embedding(text)
            results.append(resp["data"][0]["embedding"])
            if self._progress_cb and i % 10 == 0:
                self._progress_cb(i + 1, total)
        return results

    def embed_query(self, text: str) -> List[float]:
        resp = self._llm.create_embedding(text)
        return resp["data"][0]["embedding"]


# ── 工厂方法 ──

def create_embedding_backend(preset):
    if preset.backend_type == "huggingface":
        return HuggingFaceEmbeddingBackend(preset)
    elif preset.backend_type == "onnx":
        return ONNXEmbeddingBackend(preset)
    elif preset.backend_type == "gguf":
        return GGUFEmbeddingBackend(preset)
    else:
        raise ValueError(f"未知的后端类型: {preset.backend_type}")
