import os
import json
import shutil
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from backend.app.core.config import settings

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
    """独立的知识库管理器，负责 Embedding 模型加载与向量库构建/清理/状态。"""

    def __init__(self):
        self.embeddings = None
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.vectorstore = None
        self.retriever = None
        self._lock = threading.RLock()

    # ── 公开接口 ──

    def initialize(self):
        """初始化 Embedding 模型并加载/构建向量库。"""
        with self._lock:
            logger.info("正在初始化 Embedding 模型...")
            self._init_embeddings(settings.EMBEDDING_MODEL_NAME)

            data_signature = compute_data_signature(settings.DATA_DIR)

            if self._is_vectorstore_ready(settings.PERSIST_DIR, data_signature):
                logger.info("连接到现有向量库...")
                self.vectorstore = Chroma(
                    persist_directory=settings.PERSIST_DIR,
                    embedding_function=self.embeddings,
                )
            else:
                logger.info("向量库不存在或已过期，开始构建...")
                self._build_vectorstore(data_signature)

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}
            )

    def rebuild(self, model_name: Optional[str] = None):
        """按指定 embedding 模型重建向量库。"""
        with self._lock:
            selected = model_name or self.embedding_model_name or settings.EMBEDDING_MODEL_NAME
            logger.info(f"使用 Embedding 模型重建向量库: {selected}")
            self._init_embeddings(selected)
            data_signature = compute_data_signature(settings.DATA_DIR)
            self._build_vectorstore(data_signature)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}
            )

    def clean(self):
        """清理向量库。"""
        with self._lock:
            if os.path.exists(settings.PERSIST_DIR):
                shutil.rmtree(settings.PERSIST_DIR)
            self.vectorstore = None
            self.retriever = None
            logger.info("向量库已清理")

    def get_status(self) -> Dict[str, Any]:
        """返回知识库运行状态。"""
        with self._lock:
            state_path = os.path.join(settings.PERSIST_DIR, "build_state.json")
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
                "embedding_model": self.embedding_model_name,
                "version": version,
            }

    # ── 内部方法 ──

    def _init_embeddings(self, model_name: str):
        self.embedding_model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

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
                stored_model = state.get("embedding_model")
                if stored_model is not None and stored_model != self.embedding_model_name:
                    logger.info(
                        f"Embedding 模型已变更 ({stored_model} → {self.embedding_model_name})，需重建向量库"
                    )
                    return False
                sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
                return os.path.exists(sqlite_path)
        except Exception:
            return False
        return False

    def _build_vectorstore(self, data_signature: str):
        if os.path.exists(settings.PERSIST_DIR):
            shutil.rmtree(settings.PERSIST_DIR)
        os.makedirs(settings.PERSIST_DIR, exist_ok=True)

        documents = []
        json_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith(".json")]
        total_files = len(json_files)
        logger.info(f"开始加载 {total_files} 个知识文件...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        for idx, filename in enumerate(json_files):
            if idx % 100 == 0:
                logger.info(f"加载进度: {idx}/{total_files}...")
            filepath = os.path.join(settings.DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "未知标题")
            source_url = data.get("source_url", "")

            if "structured_content" in data:
                for section_title, text_list in data["structured_content"].items():
                    section_text = "\n".join([t for t in text_list if t.strip()])
                    if not section_text:
                        continue
                    full_content = f"## {title} - {section_title} ##\n{section_text}"
                    if len(full_content) < 1000:
                        documents.append(
                            Document(
                                page_content=full_content,
                                metadata={
                                    "title": title,
                                    "section": section_title,
                                    "source_url": source_url,
                                },
                            )
                        )
                    else:
                        sub_docs = text_splitter.create_documents(
                            [full_content],
                            metadatas=[
                                {
                                    "title": title,
                                    "section": section_title,
                                    "source_url": source_url,
                                }
                            ],
                        )
                        documents.extend(sub_docs)
            else:
                text = data.get("text", "")
                if text:
                    documents.extend(
                        text_splitter.create_documents(
                            [text],
                            metadatas=[{"title": title, "source_url": source_url}],
                        )
                    )

        logger.info(f"结构化分片完成，共生成 {len(documents)} 个片段")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=settings.PERSIST_DIR,
        )

        with open(
            os.path.join(settings.PERSIST_DIR, "build_state.json"), "w", encoding="utf-8"
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


kb_manager = KnowledgeBaseManager()

