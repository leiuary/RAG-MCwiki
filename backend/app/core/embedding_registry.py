"""Embedding 模型预设注册表 —— 定义所有可用模型及其元数据。"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EmbeddingModelPreset:
    id: str                      # 机器键，如 "qwen3-hf"
    display_name: str            # 显示标签
    backend_type: str            # "huggingface" | "onnx" | "gguf"
    model_name_or_path: str      # HF repo ID 或本地路径
    persist_subdir: str          # chroma_db/ 下的子目录名
    dim: int                     # 向量维度
    is_default: bool = False
    description: str = ""
    gguf_filename: Optional[str] = None   # GGUF 文件名（仅 backend_type="gguf"）
    onnx_filename: str = "model_quantized.onnx"  # ONNX 文件名（仅 backend_type="onnx"）


EMBEDDING_MODEL_PRESETS: List[EmbeddingModelPreset] = [
    EmbeddingModelPreset(
        id="qwen3-hf",
        display_name="Qwen3-Embedding (原版 HF)",
        backend_type="huggingface",
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        persist_subdir="qwen3",
        dim=1024,
        is_default=True,
        description="原版 HuggingFace 模型，效果最优，内存占用约 2GB",
    ),
    EmbeddingModelPreset(
        id="qwen3-gguf",
        display_name="Qwen3-Embedding (GGUF Q3_K_M)",
        backend_type="gguf",
        model_name_or_path="PeterAM4/Qwen3-Embedding-0.6B-GGUF",
        persist_subdir="qwen3-gguf",
        dim=1024,
        description="GGUF 量化版，内存占用约 450MB，效果接近原版",
        gguf_filename="Qwen3-Embedding-0.6B-Q3_K_M-imat.gguf",
    ),
    EmbeddingModelPreset(
        id="gte-quant",
        display_name="gte-small-zh (ONNX INT8)",
        backend_type="onnx",
        model_name_or_path="kanewang/gte-small-zh",
        persist_subdir="gte-quant",
        dim=512,
        description="ONNX INT8 量化版，内存占用约 150MB，速度最快",
        onnx_filename="onnx/model_quantized.onnx",
    ),
]

EMBEDDING_MODEL_MAP: Dict[str, EmbeddingModelPreset] = {
    p.id: p for p in EMBEDDING_MODEL_PRESETS
}

DEFAULT_MODEL_ID = "qwen3-hf"


def get_model_persist_dir(base_persist_dir: str, model_id: str) -> str:
    preset = EMBEDDING_MODEL_MAP[model_id]
    return os.path.join(base_persist_dir, preset.persist_subdir)
