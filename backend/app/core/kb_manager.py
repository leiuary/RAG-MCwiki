import os
import json
import re
import shutil
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from langchain_core.documents import Document
# 仅 _build_vectorstore 需要，延迟导入以避免启动时加载 transformers/torch/sympy 链
from langchain_community.vectorstores import Chroma

from backend.app.core.config import settings
from backend.app.core.embedding_registry import (
    EMBEDDING_MODEL_PRESETS,
    EMBEDDING_MODEL_MAP,
    DEFAULT_MODEL_ID,
    get_model_persist_dir,
)
from backend.app.core.embedding_backends import create_embedding_backend

logger = logging.getLogger(__name__)


def compute_data_signature(folder_path: str) -> str:
    """计算数据目录的 SHA-256 签名，用于检测源数据是否变更。"""
    if not os.path.isdir(folder_path):
        return ""
    hasher = hashlib.sha256()
    json_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        try:
            stat = os.stat(file_path)
            hasher.update(filename.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("ascii"))
            hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        except OSError:
            continue
    return hasher.hexdigest()


class KnowledgeBaseManager:
    """多模型知识库管理器，支持 HuggingFace / ONNX / GGUF 嵌入后端。"""

    def __init__(self):
        self.embeddings = None
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.current_model_id = settings.EMBEDDING_MODEL_ID
        self.vectorstore = None
        self.retriever = None
        self._lock = threading.RLock()
        self._title_index: Dict[str, List[dict]] = {}       # title → [{text, meta}, ...]

    # ── 公开接口 ──

    def connect(self):
        """启动时连接向量库，不触发构建。自动迁移旧布局。"""
        with self._lock:
            self._migrate_old_layout_if_needed()
            self._init_embeddings(self.current_model_id)

            persist_dir = self._get_persist_dir()
            data_signature = compute_data_signature(settings.DATA_DIR)

            if self._is_vectorstore_ready(persist_dir, data_signature):
                logger.info(f"连接到现有向量库 ({persist_dir})...")
                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": settings.RETRIEVAL_K}
                )
                self._build_indices()
            else:
                logger.warning(f"向量库未就绪 ({persist_dir})，请通过 manage_kb.py build 构建")
                self.retriever = None

    def rebuild(self, model_id: Optional[str] = None, progress_cb=None):
        """按指定模型重建向量库。model_id 默认使用当前活跃模型。"""
        with self._lock:
            target = model_id or self.current_model_id
            if target not in EMBEDDING_MODEL_MAP:
                raise ValueError(f"未知的模型 ID: {target}")

            preset = EMBEDDING_MODEL_MAP[target]
            logger.info(f"使用 {preset.display_name} 重建向量库")
            self._init_embeddings(target)

            persist_dir = self._get_persist_dir()
            data_signature = compute_data_signature(settings.DATA_DIR)
            self._build_vectorstore(persist_dir, data_signature, progress_cb=progress_cb)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}
            )

    def switch_model(self, model_id: str) -> dict:
        """运行时切换到另一个已构建的嵌入模型，失败时自动回滚。"""
        with self._lock:
            if model_id == self.current_model_id:
                return self.get_status()

            if model_id not in EMBEDDING_MODEL_MAP:
                raise ValueError(f"未知的模型 ID: {model_id}")

            preset = EMBEDDING_MODEL_MAP[model_id]
            prev_id = self.current_model_id
            prev_embeddings = self.embeddings

            try:
                self._init_embeddings(model_id)
                persist_dir = self._get_persist_dir()
                data_signature = compute_data_signature(settings.DATA_DIR)

                if not self._is_vectorstore_ready(persist_dir, data_signature):
                    raise RuntimeError(
                        f"模型「{preset.display_name}」的向量库尚未构建，"
                        f"请先运行: python scripts/manage_kb.py build --model-id {model_id}"
                    )

                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": settings.RETRIEVAL_K}
                )
                logger.info(f"模型已切换到: {preset.display_name}")
            except Exception:
                self.current_model_id = prev_id
                self.embeddings = prev_embeddings
                raise

            return self.get_status()

    def clean(self, model_id: Optional[str] = None):
        """清理向量库。不传 model_id 则清理当前模型。"""
        with self._lock:
            target = model_id or self.current_model_id
            persist_dir = self._get_persist_dir(target)
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            if target == self.current_model_id:
                self.vectorstore = None
                self.retriever = None
            logger.info(f"向量库已清理: {persist_dir}")

    def unload_model(self):
        """卸载当前的 Embedding 模型并强制释放显存/内存"""
        with self._lock:
            if self.embeddings:
                del self.embeddings
                self.embeddings = None
                self.vectorstore = None
                self.retriever = None
                
                import gc
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except ImportError:
                    pass
                
                logger.info("已卸载 Embedding 模型并尝试释放显存")

    def get_status(self) -> Dict[str, Any]:
        """返回当前模型状态 + 所有模型概览。"""
        with self._lock:
            preset = EMBEDDING_MODEL_MAP.get(self.current_model_id)
            persist_dir = self._get_persist_dir()
            state_path = os.path.join(persist_dir, "build_state.json")
            version = "unknown"
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    version = state.get("built_at") or "unknown"
                except Exception:
                    version = "unknown"

            return {
                "ready": self.retriever is not None,
                "active_model_id": self.current_model_id,
                "active_model_name": preset.display_name if preset else self.current_model_id,
                "embedding_model": self.embedding_model_name,
                "version": version,
                "available_models": self._get_available_models_sync(),
            }

    # ── 关键词索引（辅助向量检索，零额外依赖） ──

    def _build_indices(self):
        """构建标题→chunks 内存索引（仅标题，极快）。"""
        self._title_index.clear()
        try:
            all_data = self.vectorstore.get()
        except Exception:
            logger.warning("无法从向量库加载全部数据，跳过标题索引构建")
            return
        metas = all_data.get("metadatas", [])
        docs = all_data.get("documents", [])
        for meta, doc in zip(metas, docs):
            title = meta.get("title", "")
            if title:
                self._title_index.setdefault(title, []).append(
                    {"text": doc, "meta": meta}
                )
        logger.info(f"标题索引构建完成: {len(self._title_index)} 个页面")

    def lookup_by_keyword(self, token: str, max_titles: int = 80) -> List[dict]:
        """返回标题（优先）或内容中包含 token 的 chunk 列表。泛词返回空。"""
        results = []
        seen_titles = set()
        # 第一遍：标题匹配
        for title, chunks in self._title_index.items():
            if token in title:
                seen_titles.add(title)
                if len(seen_titles) > max_titles:
                    return []
                results.extend(chunks)
        # 第二遍：标题没命中时，搜索内容（仅 token 不泛时触发）
        if not results:
            for title, chunks in self._title_index.items():
                for chunk in chunks:
                    if token in chunk["text"]:
                        seen_titles.add(title)
                        if len(seen_titles) > max_titles:
                            return []
                        results.append(chunk)
        return results

    # ── 内部方法 ──

    def _init_embeddings(self, model_id: str):
        preset = EMBEDDING_MODEL_MAP[model_id]
        self.embeddings = create_embedding_backend(preset)
        self.current_model_id = model_id
        self.embedding_model_name = preset.model_name_or_path

    def _get_persist_dir(self, model_id: Optional[str] = None) -> str:
        return get_model_persist_dir(settings.PERSIST_DIR, model_id or self.current_model_id)

    def _get_available_models_sync(self) -> list:
        data_sig = compute_data_signature(settings.DATA_DIR)
        results = []
        for preset in EMBEDDING_MODEL_PRESETS:
            persist_dir = get_model_persist_dir(settings.PERSIST_DIR, preset.id)
            built = self._is_vectorstore_ready(persist_dir, data_sig)
            built_at = ""
            if built:
                try:
                    with open(os.path.join(persist_dir, "build_state.json"), "r", encoding="utf-8") as f:
                        built_at = json.load(f).get("built_at", "")
                except Exception:
                    pass
            results.append({
                "id": preset.id,
                "name": preset.display_name,
                "description": preset.description,
                "dim": preset.dim,
                "is_built": built,
                "built_at": built_at,
                "is_default": preset.is_default,
                "is_active": preset.id == self.current_model_id,
            })
        return results

    def _is_vectorstore_ready(self, persist_dir: str, data_signature: str) -> bool:
        state_path = os.path.join(persist_dir, "build_state.json")
        if not os.path.exists(state_path):
            return False
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if (
                state.get("status") == "complete"
                and state.get("data_signature") == data_signature
            ):
                sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
                return os.path.exists(sqlite_path)
        except Exception:
            return False
        return False

    def _build_vectorstore(self, persist_dir: str, data_signature: str, progress_cb=None):
        progress_path = os.path.join(persist_dir, "build_progress.json")

        resume_from = 0
        if os.path.exists(progress_path) and os.path.exists(
            os.path.join(persist_dir, "chroma.sqlite3")
        ):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    p = json.load(f)
                if (
                    p.get("data_signature") == data_signature
                    and p.get("embedding_model") == self.embedding_model_name
                ):
                    resume_from = p.get("completed_docs", 0)
            except Exception:
                pass

        if resume_from == 0:
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            os.makedirs(persist_dir, exist_ok=True)

        documents = []
        json_files = sorted(f for f in os.listdir(settings.DATA_DIR) if f.endswith(".json"))
        total_files = len(json_files)
        logger.info(f"开始加载 {total_files} 个知识文件...")

        # 噪声过滤器
        _NOISE_PREFIXES = ("原主机版", "Nintendo Switch版", "Xbox", "PlayStation")
        _NOISE_SNAPSHOT = re.compile(
            r"^(Java版|基岩版|携带版|教育版)"
            r"(Alpha\s*v?|Beta\s*|Preview\s*|pre\s*|rc\s*|Pre\-release\s*|"
            r"Infdev\s*|Classic\s*|Indev\s*)?"
            r"[\d]+[wW].*"
        )
        _DISAMBIG_PATTERN = re.compile(r"（消歧义）")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for idx, filename in enumerate(json_files):
            if idx % 100 == 0:
                logger.info(f"加载进度: {idx}/{total_files}...")
            if progress_cb and idx % 10 == 0:
                progress_cb("load", idx + 1, total_files)
            filepath = os.path.join(settings.DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "未知标题")
            source_url = data.get("source_url", "")

            # ── 噪声过滤 ──
            if title.startswith(_NOISE_PREFIXES):
                continue
            if _NOISE_SNAPSHOT.match(title):
                continue
            if _DISAMBIG_PATTERN.search(title):
                sc = data.get("structured_content", {})
                total_len = sum(
                    len(v) for section in sc.values()
                    for v in (section.values() if isinstance(section, dict) else [])
                )
                if total_len < 200:
                    continue

            # ── 按 legacy 方式构建每文件一个 Document，让分句器自然切分 ──
            content_lines = []
            if "structured_content" in data:
                for section_title, entities in data["structured_content"].items():
                    content_lines.append(f"## {section_title}")
                    for entity_name, values in entities.items():
                        content_lines.append(entity_name)
                        if isinstance(values, list):
                            for v in values:
                                if v and str(v).strip():
                                    content_lines.append(str(v).strip())
                    content_lines.append("")
            else:
                text = data.get("text", "")
                if text:
                    content_lines.append(text)

            full_text = "\n".join(line for line in content_lines if line)
            if not full_text.strip():
                continue

            doc = Document(
                page_content=full_text,
                metadata={"title": title, "source_url": source_url},
            )
            documents.append(doc)

        logger.info(f"原始文档加载完成: {len(documents)} 篇，开始切分...")
        splits = text_splitter.split_documents(documents)
        total = len(splits)
        logger.info(f"结构化分片完成，共生成 {total} 个片段")

        if resume_from > 0:
            logger.info(
                f"检测到未完成构建，从第 {resume_from} 个片段继续（已完成 {resume_from}/{total}）"
            )
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            splits = splits[resume_from:]

        batch_size = 200

        def _set_micro_cb(offset: int):
            """将外层 progress_cb 注入到支持微批次回调的嵌入后端（如 ONNX）。"""
            if progress_cb and hasattr(self.embeddings, '_progress_cb'):
                self.embeddings._progress_cb = lambda done, _: progress_cb(
                    "embed", offset + done, total
                )

        if resume_from == 0:
            first_batch = splits[:batch_size]
            logger.info(f"正在向量化并写入批次: 1~{len(first_batch)}/{total}...")
            _set_micro_cb(0)
            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=persist_dir,
            )
            if hasattr(self.embeddings, '_progress_cb'):
                self.embeddings._progress_cb = None
            if progress_cb:
                progress_cb("embed", len(first_batch), total)
            self._save_progress(progress_path, data_signature, total, len(first_batch))
            start_offset = batch_size
        else:
            start_offset = 0

        for i in range(start_offset, len(splits), batch_size):
            end = min(i + batch_size, len(splits))
            batch = splits[i:end]
            actual_end = resume_from + end
            logger.info(f"正在向量化并写入批次: {resume_from + i + 1}~{actual_end}/{total}...")
            _set_micro_cb(resume_from + i)
            self.vectorstore.add_documents(batch)
            if hasattr(self.embeddings, '_progress_cb'):
                self.embeddings._progress_cb = None
            if progress_cb:
                progress_cb("embed", actual_end, total)
            self._save_progress(progress_path, data_signature, total, actual_end)

        with open(
            os.path.join(persist_dir, "build_state.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "status": "complete",
                    "data_signature": data_signature,
                    "embedding_model": self.embedding_model_name,
                    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                },
                f,
            )
        try:
            os.remove(progress_path)
        except OSError:
            pass

    def _save_progress(self, path: str, data_signature: str, total: int, completed: int):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "data_signature": data_signature,
                        "embedding_model": self.embedding_model_name,
                        "total_docs": total,
                        "completed_docs": completed,
                    },
                    f,
                )
        except OSError:
            pass

    def _migrate_old_layout_if_needed(self):
        """一次性将旧 chroma_db/ 根目录内容迁移到 chroma_db/qwen3/"""
        old_state = os.path.join(settings.PERSIST_DIR, "build_state.json")
        persist_dir = get_model_persist_dir(settings.PERSIST_DIR, "qwen3-hf")

        if os.path.exists(old_state) and not os.path.exists(
            os.path.join(persist_dir, "build_state.json")
        ):
            try:
                with open(old_state, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("embedding_model") == "Qwen/Qwen3-Embedding-0.6B":
                    logger.info("检测到旧版向量库布局，正在迁移到 chroma_db/qwen3/...")
                    os.makedirs(persist_dir, exist_ok=True)
                    # 排除其他模型的子目录，避免误搬
                    model_subdirs = {p.persist_subdir for p in EMBEDDING_MODEL_PRESETS}
                    for item in os.listdir(settings.PERSIST_DIR):
                        if item in model_subdirs:
                            continue  # 跳过其他模型的子目录
                        item_path = os.path.join(settings.PERSIST_DIR, item)
                        if os.path.isfile(item_path) or os.path.isdir(item_path):
                            dest = os.path.join(persist_dir, item)
                            shutil.move(item_path, dest)
                    logger.info("迁移完成")
            except Exception as e:
                logger.warning(f"迁移旧布局失败 (非致命): {e}")


kb_manager = KnowledgeBaseManager()
