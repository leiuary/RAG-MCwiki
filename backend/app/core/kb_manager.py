import os
import sys
import json
import re
import shutil
import hashlib
import logging
import threading
import time
import pickle
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import jieba

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
    json_files = sorted(f for f in os.listdir(folder_path) if f.endswith((".json", ".md")))
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
        self._token_to_titles: Dict[str, set] = {}          # n-gram → {title, ...}
        self._bm25_corpus: List[List[str]] = []             # jieba 分词后的文档列表
        self._bm25_doc_ids: List[str] = []                  # 文档 ID，与 corpus 同序
        self._bm25_docs: List[dict] = []                    # 文档内容 [{text, meta}, ...]，与 corpus 同序
        self._bm25 = None                                   # BM25Okapi 实例

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

    def rebuild(self, model_id: Optional[str] = None, progress_cb=None) -> dict:
        """按指定模型重建向量库（双缓冲：构建阶段不持锁，完成后原子替换）。返回统计信息。"""
        target = model_id or self.current_model_id
        if target not in EMBEDDING_MODEL_MAP:
            raise ValueError(f"未知的模型 ID: {target}")

        preset = EMBEDDING_MODEL_MAP[target]
        persist_dir = get_model_persist_dir(settings.PERSIST_DIR, target)

        # 检测是首次建立还是更新
        state_path = os.path.join(persist_dir, "build_state.json")
        is_update = os.path.exists(state_path)
        action = "更新" if is_update else "建立"
        logger.info(f"使用 {preset.display_name} {action}向量库")

        # 扫描文件
        md_files = sorted(f for f in os.listdir(settings.DATA_DIR) if f.endswith(".md"))
        total_files = len(md_files)
        logger.info(f"扫描完成: 发现 {total_files} 个文件")
        if progress_cb:
            progress_cb("scan", total_files, total_files, f"扫描完成: {total_files} 个文件")

        # 读取旧向量库信息（如果有）
        old_chunks = 0
        if is_update:
            try:
                old_state_path = os.path.join(persist_dir, "build_state.json")
                with open(old_state_path, "r", encoding="utf-8") as f:
                    old_state = json.load(f)
                # 尝试从 chunks_cache_meta.json 获取旧的 chunk 数量
                chunks_meta_path = os.path.join(persist_dir, "chunks_cache_meta.json")
                if os.path.exists(chunks_meta_path):
                    with open(chunks_meta_path, "r", encoding="utf-8") as f:
                        old_chunks = json.load(f).get("total_chunks", 0)
            except Exception:
                pass

        # 阶段 1：构建新向量库（不持锁，耗时操作）
        temp_embeddings = create_embedding_backend(preset, batch_size=settings.EMBEDDING_BATCH_SIZE)
        data_signature = compute_data_signature(settings.DATA_DIR)
        effective_chunk_size = settings.EMBEDDING_CHUNK_SIZE or preset.chunk_size
        self._build_vectorstore(
            persist_dir, data_signature, progress_cb=progress_cb,
            embeddings=temp_embeddings, model_name=preset.model_name_or_path,
            chunk_size=effective_chunk_size, chunk_overlap=preset.chunk_overlap,
            max_tokens=preset.max_tokens,
            target_tokens=settings.EMBEDDING_TARGET_TOKENS,
            target_utilization=settings.EMBEDDING_TOKEN_TARGET_UTILIZATION,
        )

        # 读取新构建的 chunk 数量
        new_chunks = 0
        try:
            chunks_meta_path = os.path.join(persist_dir, "chunks_cache_meta.json")
            if os.path.exists(chunks_meta_path):
                with open(chunks_meta_path, "r", encoding="utf-8") as f:
                    new_chunks = json.load(f).get("total_chunks", 0)
        except Exception:
            pass

        # 阶段 2：原子替换引用（持锁，瞬间完成）
        with self._lock:
            self.embeddings = temp_embeddings
            self.current_model_id = target
            self.embedding_model_name = preset.model_name_or_path
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVAL_K}
            )
            self._build_indices()

        # 计算变更统计
        chunks_added = max(0, new_chunks - old_chunks) if is_update else new_chunks
        chunks_removed = max(0, old_chunks - new_chunks) if is_update else 0

        stats = {
            "action": action,
            "model_name": preset.display_name,
            "files_scanned": total_files,
            "old_chunks": old_chunks,
            "new_chunks": new_chunks,
            "chunks_added": chunks_added,
            "chunks_removed": chunks_removed,
        }
        logger.info(
            f"Embedding 向量库{action}完成: {new_chunks} 个 chunks"
        )
        return stats

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

    def _get_indices_cache_path(self) -> str:
        """返回索引缓存文件路径（所有模型共用，存放在 chroma_db 根目录）。"""
        return os.path.join(settings.PERSIST_DIR, "indices_cache.pkl")

    def _build_indices(self):
        """构建标题索引 + n-gram 倒排索引 + BM25 索引。优先从缓存加载。"""
        self._title_index.clear()
        self._token_to_titles.clear()
        self._bm25_corpus.clear()
        self._bm25_doc_ids.clear()
        self._bm25_docs.clear()
        self._bm25 = None

        # 尝试从缓存加载
        cache_path = self._get_indices_cache_path()
        data_signature = compute_data_signature(settings.DATA_DIR)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("data_signature") == data_signature and cache.get("cache_version") == 2:
                    self._title_index = cache["title_index"]
                    self._token_to_titles = cache["token_to_titles"]
                    self._bm25_corpus = cache["bm25_corpus"]
                    self._bm25_doc_ids = cache.get("bm25_doc_ids", [])
                    self._bm25_docs = cache.get("bm25_docs", [])
                    if self._bm25_corpus:
                        from rank_bm25 import BM25Okapi
                        self._bm25 = BM25Okapi(
                            self._bm25_corpus,
                            k1=settings.BM25_K1,
                            b=settings.BM25_B,
                        )
                    logger.info(
                        f"从缓存加载索引: {len(self._title_index)} 个页面, "
                        f"{len(self._token_to_titles)} 个 n-gram, "
                        f"{len(self._bm25_corpus)} 个 BM25 文档"
                    )
                    return
                else:
                    logger.info("索引缓存签名不匹配，重新构建")
            except Exception as e:
                logger.warning(f"索引缓存加载失败: {e}，重新构建")

        # 从 ChromaDB 构建
        batch_size = 2000
        offset = 0
        try:
            while True:
                all_data = self.vectorstore.get(limit=batch_size, offset=offset, include=["metadatas", "documents"])
                metas = all_data.get("metadatas", [])
                docs = all_data.get("documents", [])
                ids = all_data.get("ids", [])
                if not metas:
                    break
                for meta, doc, doc_id in zip(metas, docs, ids):
                    title = meta.get("title", "")
                    if title:
                        self._title_index.setdefault(title, []).append(
                            {"text": doc, "meta": meta}
                        )
                    # BM25: 存分词结果、ID 和文档内容
                    tokens = [sys.intern(t) for t in jieba.cut(doc) if len(t.strip()) > 0]
                    self._bm25_corpus.append(tokens)
                    self._bm25_doc_ids.append(doc_id)
                    self._bm25_docs.append({"text": doc, "meta": meta})
                offset += batch_size
        except Exception as e:
            logger.warning(f"索引构建出错: {e}")

        # 构建 n-gram 倒排索引（2-5 字符滑动窗口）
        for title in self._title_index:
            for i in range(len(title)):
                for length in range(2, min(6, len(title) - i + 1)):
                    token = title[i:i + length]
                    self._token_to_titles.setdefault(token, set()).add(title)

        # 构建 BM25 索引
        if self._bm25_corpus:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(
                self._bm25_corpus,
                k1=settings.BM25_K1,
                b=settings.BM25_B,
            )
            logger.info(
                f"索引构建完成: {len(self._title_index)} 个页面, "
                f"{len(self._token_to_titles)} 个 n-gram, "
                f"{len(self._bm25_corpus)} 个 BM25 文档"
            )
        else:
            logger.warning("BM25 语料为空，跳过构建")

        # 持久化缓存
        try:
            cache = {
                "cache_version": 2,
                "data_signature": data_signature,
                "title_index": self._title_index,
                "token_to_titles": self._token_to_titles,
                "bm25_corpus": self._bm25_corpus,
                "bm25_doc_ids": self._bm25_doc_ids,
                "bm25_docs": self._bm25_docs,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            logger.info(f"索引缓存已保存: {cache_path}")
        except Exception as e:
            logger.warning(f"索引缓存保存失败: {e}")

    def lookup_by_keyword(self, token: str, max_titles: int = 80, max_chunks_per_title: int = 5, max_results: int = 200) -> List[dict]:
        """返回标题（优先）或内容中包含 token 的 chunk 列表。泛词返回空。"""
        results = []
        seen_titles = set()

        # 快速路径：倒排索引定位候选标题
        candidate_titles = self._token_to_titles.get(token, set())
        if candidate_titles:
            for title in candidate_titles:
                if len(seen_titles) >= max_titles:
                    return results[:max_results]
                seen_titles.add(title)
                chunks = self._title_index.get(title, [])
                results.extend(chunks[:max_chunks_per_title])
                if len(results) >= max_results:
                    return results[:max_results]
            return results

        # 回退：线性扫描标题（长短语可能不在倒排索引中）
        for title, chunks in self._title_index.items():
            if token in title:
                seen_titles.add(title)
                if len(seen_titles) > max_titles:
                    return results[:max_results]
                results.extend(chunks[:max_chunks_per_title])
                if len(results) >= max_results:
                    return results[:max_results]

        # 第二遍：标题没命中时，搜索内容
        if not results:
            for title, chunks in self._title_index.items():
                for chunk in chunks:
                    if token in chunk["text"]:
                        seen_titles.add(title)
                        if len(seen_titles) > max_titles:
                            return results[:max_results]
                        results.append(chunk)
                        if len(results) >= max_results:
                            return results[:max_results]
        return results

    def bm25_search(self, query_tokens: List[str], top_k: int = 20, boost_phrases: Optional[List[str]] = None) -> List[dict]:
        """BM25 检索，返回 top_k 个最相关的 chunk。

        Args:
            query_tokens: jieba 分词后的查询 token 列表
            top_k: 返回数量
            boost_phrases: 原始搜索短语列表，包含这些短语的文档获得额外加分
        """
        if not self._bm25 or not query_tokens or not self._bm25_docs:
            return []
        scores = self._bm25.get_scores(query_tokens)

        # 短语匹配加成：原始搜索短语作为子串出现在文档中 → 加分
        if boost_phrases:
            _PHRASE_BOOST = 5.0
            for i in range(len(scores)):
                if scores[i] <= 0:
                    continue
                text = self._bm25_docs[i]["text"]
                title = self._bm25_docs[i]["meta"].get("title", "")
                combined = title + " " + text
                for phrase in boost_phrases:
                    if len(phrase) >= 2 and phrase in combined:
                        scores[i] += _PHRASE_BOOST

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                break
            doc_info = self._bm25_docs[idx]
            results.append({
                "text": doc_info["text"],
                "meta": doc_info["meta"],
                "_bm25_score": float(scores[idx]),
                "_bm25_rank": rank,
            })
        return results

    def rebuild_bm25(self, progress_cb=None) -> dict:
        """重建 BM25 索引（标题索引 + n-gram + BM25）。
        直接从 data/markdown/ 读取文件，不依赖 Embedding 向量库。
        返回统计信息。
        """
        with self._lock:
            # 检测是首次建立还是更新
            cache_path = self._get_indices_cache_path()
            is_update = os.path.exists(cache_path)
            action = "更新" if is_update else "建立"
            logger.info(f"开始{action} BM25 索引...")
            if progress_cb:
                progress_cb("init", 0, 0, f"正在扫描文件（{action}模式）...")

            # 清空现有索引
            self._title_index.clear()
            self._token_to_titles.clear()
            self._bm25_corpus.clear()
            self._bm25_doc_ids.clear()
            self._bm25_docs.clear()
            self._bm25 = None

            _SOURCE_RE = re.compile(r"^Source:\s*(\S+)", re.MULTILINE)

            # 扫描文件并统计
            md_files = sorted(f for f in os.listdir(settings.DATA_DIR) if f.endswith(".md"))
            total_files = len(md_files)

            # 读取旧缓存（如果有）
            old_pages = 0
            old_docs = 0
            if is_update:
                try:
                    with open(cache_path, "rb") as f:
                        old_cache = pickle.load(f)
                    old_pages = len(old_cache.get("title_index", {}))
                    old_docs = len(old_cache.get("bm25_corpus", []))
                except Exception:
                    pass

            scan_msg = f"扫描完成: 发现 {total_files} 个文件"
            if is_update:
                scan_msg += f"（旧索引: {old_pages} 页面, {old_docs} 文档）"
            logger.info(scan_msg)
            if progress_cb:
                progress_cb("scan", total_files, total_files, scan_msg)

            logger.info(f"开始加载 {total_files} 个 Markdown 知识文件...")

            doc_id_counter = 0
            for idx, filename in enumerate(md_files):
                if idx % 100 == 0:
                    logger.info(f"加载进度: {idx}/{total_files}...")
                if progress_cb and idx % 10 == 0:
                    progress_cb("load", idx + 1, total_files)

                filepath = os.path.join(settings.DATA_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                # 提取标题和来源
                title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else filename.replace(".md", "")
                source_match = _SOURCE_RE.search(content)
                source_url = source_match.group(1) if source_match else ""

                if not content.strip():
                    continue

                # 按章节切分（简单的标题分割，不依赖 LangChain）
                sections = self._split_markdown_sections(content)

                for section_title, section_text in sections:
                    if not section_text.strip():
                        continue

                    # 构建元数据
                    meta = {
                        "title": title,
                        "source_url": source_url,
                    }
                    if section_title:
                        meta["section"] = section_title

                    # 添加到标题索引
                    self._title_index.setdefault(title, []).append(
                        {"text": section_text, "meta": meta}
                    )

                    # BM25 分词
                    tokens = [sys.intern(t) for t in jieba.cut(section_text) if len(t.strip()) > 0]
                    self._bm25_corpus.append(tokens)
                    self._bm25_doc_ids.append(f"bm25_{doc_id_counter}")
                    self._bm25_docs.append({"text": section_text, "meta": meta})
                    doc_id_counter += 1

            # 构建 n-gram 倒排索引
            for title in self._title_index:
                for i in range(len(title)):
                    for length in range(2, min(6, len(title) - i + 1)):
                        token = title[i:i + length]
                        self._token_to_titles.setdefault(token, set()).add(title)

            # 构建 BM25
            if self._bm25_corpus:
                from rank_bm25 import BM25Okapi
                if progress_cb:
                    progress_cb("bm25", 0, 0, "正在构建 BM25 索引...")
                self._bm25 = BM25Okapi(
                    self._bm25_corpus,
                    k1=settings.BM25_K1,
                    b=settings.BM25_B,
                )

            # 持久化缓存
            cache_path = self._get_indices_cache_path()
            data_signature = compute_data_signature(settings.DATA_DIR)
            try:
                cache = {
                    "cache_version": 2,
                    "data_signature": data_signature,
                    "title_index": self._title_index,
                    "token_to_titles": self._token_to_titles,
                    "bm25_corpus": self._bm25_corpus,
                    "bm25_doc_ids": self._bm25_doc_ids,
                    "bm25_docs": self._bm25_docs,
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
                logger.info(f"索引缓存已保存: {cache_path}")
            except Exception as e:
                logger.warning(f"索引缓存保存失败: {e}")

            # 计算变更统计
            new_pages = len(self._title_index)
            new_docs = len(self._bm25_corpus)
            pages_added = max(0, new_pages - old_pages) if is_update else new_pages
            pages_removed = max(0, old_pages - new_pages) if is_update else 0
            docs_added = max(0, new_docs - old_docs) if is_update else new_docs
            docs_removed = max(0, old_docs - new_docs) if is_update else 0

            stats = {
                "pages": new_pages,
                "ngrams": len(self._token_to_titles),
                "bm25_docs": new_docs,
                "action": action,
                "files_scanned": total_files,
                "pages_added": pages_added,
                "pages_removed": pages_removed,
                "docs_added": docs_added,
                "docs_removed": docs_removed,
                "old_pages": old_pages,
                "old_docs": old_docs,
            }
            logger.info(
                f"BM25 索引{action}完成: {stats['pages']} 个页面, "
                f"{stats['ngrams']} 个 n-gram, {stats['bm25_docs']} 个文档"
            )
            if progress_cb:
                progress_cb("done", 1, 1, "完成")
            return stats

    def _split_markdown_sections(self, content: str) -> list:
        """简单的 Markdown 章节切分，返回 [(section_title, text), ...]"""
        sections = []
        current_title = ""
        current_lines = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # 保存当前章节
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                # 新章节
                current_title = line.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 保存最后一个章节
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return sections

    # ── 内部方法 ──

    def _init_embeddings(self, model_id: str):
        preset = EMBEDDING_MODEL_MAP[model_id]
        self.embeddings = create_embedding_backend(preset, batch_size=settings.EMBEDDING_BATCH_SIZE)
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

    def _build_vectorstore(self, persist_dir: str, data_signature: str, progress_cb=None, embeddings=None, model_name=None, chunk_size: int = 400, chunk_overlap: int = 50, max_tokens: int = 512, target_tokens: Optional[int] = None, target_utilization: float = 0.95):
        embeddings = embeddings or self.embeddings
        model_name = model_name or self.embedding_model_name
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
                    and p.get("embedding_model") == model_name
                ):
                    resume_from = p.get("completed_docs", 0)
            except Exception:
                pass

        # ── 分块缓存：相同数据签名 + 分块参数时跳过加载和分块 ──
        cache_path = os.path.join(persist_dir, "chunks_cache.pkl")
        cache_meta_path = os.path.join(persist_dir, "chunks_cache_meta.json")
        cache_hit = False
        cache_meta = {
            "data_signature": data_signature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        if os.path.exists(cache_path) and os.path.exists(cache_meta_path):
            try:
                with open(cache_meta_path, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                if (
                    saved_meta.get("data_signature") == data_signature
                    and saved_meta.get("chunk_size") == chunk_size
                    and saved_meta.get("chunk_overlap") == chunk_overlap
                ):
                    logger.info("检测到分块缓存，正在加载...")
                    with open(cache_path, "rb") as f:
                        splits = pickle.load(f)
                    total = len(splits)
                    logger.info(f"从缓存加载 {total} 个分块，跳过文件加载和分块步骤")
                    cache_hit = True
            except Exception as e:
                logger.warning(f"分块缓存加载失败，将重新处理: {e}")

        if not cache_hit:
            documents = []
            md_files = sorted(f for f in os.listdir(settings.DATA_DIR) if f.endswith(".md"))
            total_files = len(md_files)
            logger.info(f"开始加载 {total_files} 个 Markdown 知识文件...")

            # 噪声过滤器
            _NOISE_PREFIXES = ("原主机版", "Nintendo Switch版", "Xbox", "PlayStation")
            _NOISE_SNAPSHOT = re.compile(
                r"^(Java版|基岩版|携带版|教育版)"
                r"(Alpha\s*v?|Beta\s*|Preview\s*|pre\s*|rc\s*|Pre\-release\s*|"
                r"Infdev\s*|Classic\s*|Indev\s*)?"
                r"[\d]+[wW].*"
            )
            _DISAMBIG_PATTERN = re.compile(r"（消歧义）")
            _SOURCE_RE = re.compile(r"^Source:\s*(\S+)", re.MULTILINE)

            from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for idx, filename in enumerate(md_files):
                if idx % 100 == 0:
                    logger.info(f"加载进度: {idx}/{total_files}...")
                if progress_cb and idx % 10 == 0:
                    progress_cb("load", idx + 1, total_files)
                filepath = os.path.join(settings.DATA_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # 从 Markdown 内容提取标题和来源
                title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else filename.replace(".md", "")
                source_match = _SOURCE_RE.search(content)
                source_url = source_match.group(1) if source_match else ""

                # ── 噪声过滤 ──
                if title.startswith(_NOISE_PREFIXES):
                    continue
                if _NOISE_SNAPSHOT.match(title):
                    continue
                if _DISAMBIG_PATTERN.search(title):
                    if len(content) < 200:
                        continue

                if not content.strip():
                    continue

                # 使用 MarkdownHeaderTextSplitter 进行结构化切分
                md_docs = markdown_splitter.split_text(content)
                for md_doc in md_docs:
                    md_doc.metadata["title"] = title
                    md_doc.metadata["source_url"] = source_url
                    documents.append(md_doc)

            logger.info(f"原始文档加载完成: {len(documents)} 段 Markdown 分块，开始按字符长度细分...")
            splits = text_splitter.split_documents(documents)
            total = len(splits)
            logger.info(f"结构化分片完成，共生成 {total} 个片段")

            # ── 动态 chunk_size 调整：逼近目标 token 利用率 ──
            # 如果用户在 .env 中显式设置了 EMBEDDING_CHUNK_SIZE > 0，则视为“强制手动模式”，跳过优化
            if settings.EMBEDDING_CHUNK_SIZE > 0:
                logger.info(f"检测到手动设置 EMBEDDING_CHUNK_SIZE={settings.EMBEDDING_CHUNK_SIZE}，跳过动态优化")
            else:
                try:
                    from transformers import AutoTokenizer as _AT
                    _tok = _AT.from_pretrained(model_name, trust_remote_code=True)
                    
                    # 确定目标 Token 数：优先使用 target_tokens (如 500)，否则使用 max_tokens
                    _effective_limit = target_tokens if target_tokens and target_tokens < max_tokens else max_tokens
                    _safe_limit = _effective_limit - 10
                    _target_tokens = int(_safe_limit * target_utilization)

                    if progress_cb:
                        progress_cb("load", 0, 1, f"检查 token 利用率 (目标 {target_utilization*100:.0f}%)...")

                    for _round in range(3):
                        _sample = sorted(splits, key=lambda d: len(d.page_content), reverse=True)[:200]
                        _max_tok = max(len(_tok.encode(d.page_content, add_special_tokens=True)) for d in _sample)
                        _max_chars = max(len(d.page_content) for d in _sample)
                        _ratio = _max_tok / _max_chars if _max_chars > 0 else 1.5
                        _utilization = _max_tok / _effective_limit

                        if _max_tok > _safe_limit:
                            _new_size = int(_target_tokens / _ratio)
                            _new_size = max(_new_size, 100)
                            _msg = f"Token 超限 {_max_tok}/{_effective_limit}，缩小 {chunk_size}→{_new_size}"
                            logger.warning(_msg)
                            if progress_cb:
                                progress_cb("load", 0, 1, _msg)
                        elif abs(_utilization - target_utilization) < 0.05:
                            _msg = f"Token {_max_tok}/{_effective_limit} ({_utilization*100:.0f}%) chunk={chunk_size}"
                            logger.info(_msg)
                            if progress_cb:
                                progress_cb("load", 1, 1, _msg)
                            break
                        else:
                            _new_size = int(chunk_size * _target_tokens / _max_tok)
                            _new_size = max(min(_new_size, 3000), 100)
                            if _new_size == chunk_size:
                                _msg = f"Token {_max_tok}/{_effective_limit} ({_utilization*100:.0f}%) chunk={chunk_size}"
                                logger.info(_msg)
                                if progress_cb:
                                    progress_cb("load", 1, 1, _msg)
                                break
                            _direction = "增大" if _new_size > chunk_size else "缩小"
                            _msg = f"Token {_max_tok}/{_effective_limit} ({_utilization*100:.0f}%) {_direction} {chunk_size}→{_new_size}"
                            logger.info(_msg)
                            if progress_cb:
                                progress_cb("load", 0, 1, _msg)

                        chunk_size = _new_size
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        splits = text_splitter.split_documents(documents)
                        total = len(splits)

                    # 发送最终结果
                    if progress_cb:
                        _final = f"chunk={chunk_size} tok≈{_max_tok}/{_effective_limit} ({_utilization*100:.0f}%)"
                        progress_cb("load", 1, 1, final_msg=_final)
                except Exception as e:
                    logger.warning(f"Token 安全检查跳过（不影响构建）: {e}")

            # 保存分块缓存（chunk_size 可能已被动态调整）
            cache_meta["chunk_size"] = chunk_size
            try:
                os.makedirs(persist_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(cache_meta_path, "w", encoding="utf-8") as f:
                    json.dump({**cache_meta, "total_chunks": total}, f)
                logger.info(f"分块缓存已保存 ({os.path.getsize(cache_path) // 1024 // 1024}MB)")
            except Exception as e:
                logger.warning(f"分块缓存保存失败（不影响构建）: {e}")

        # 首次构建：清理旧的 ChromaDB 文件，保留分块缓存
        if resume_from == 0 and os.path.exists(persist_dir):
            for item in os.listdir(persist_dir):
                if item.startswith("chunks_cache"):
                    continue
                item_path = os.path.join(persist_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        os.makedirs(persist_dir, exist_ok=True)

        if resume_from > 0:
            if resume_from >= total:
                logger.warning(f"续传进度 ({resume_from}) >= 总分块数 ({total})，数据可能已变更，从头构建")
                resume_from = 0
                # 清理旧 ChromaDB，保留分块缓存
                if os.path.exists(persist_dir):
                    for item in os.listdir(persist_dir):
                        if item.startswith("chunks_cache"):
                            continue
                        item_path = os.path.join(persist_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                os.makedirs(persist_dir, exist_ok=True)
            else:
                logger.info(
                    f"检测到未完成构建，从第 {resume_from} 个片段继续（已完成 {resume_from}/{total}）"
                )
                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings,
                )
                splits = splits[resume_from:]

        batch_size = 200

        def _set_micro_cb(offset: int):
            """将外层 progress_cb 注入到支持微批次回调的嵌入后端（如 ONNX）。"""
            if progress_cb and hasattr(embeddings, '_progress_cb'):
                embeddings._progress_cb = lambda done, _: progress_cb(
                    "embed", offset + done, total
                )

        embed_start_time = time.time()
        done_at_start = resume_from  # 用于计算 ETA 的已完成数基准

        def _fmt_eta(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            m, s = divmod(int(seconds), 60)
            if m < 60:
                return f"{m}m{s:02d}s"
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m"

        def _log_speed(done: int):
            elapsed = time.time() - embed_start_time
            speed = done / elapsed if elapsed > 0 else 0
            remaining = total - done_at_start - done
            eta = remaining / speed if speed > 0 else 0
            logger.info(
                f"  ↳ 速度: {speed:.1f} chunks/s | ETA: {_fmt_eta(eta)}"
            )

        if resume_from == 0:
            first_batch = splits[:batch_size]
            logger.info(f"正在向量化并写入批次: 1~{len(first_batch)}/{total}...")
            _set_micro_cb(0)
            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
            if hasattr(embeddings, '_progress_cb'):
                embeddings._progress_cb = None
            if progress_cb:
                progress_cb("embed", len(first_batch), total)
            self._save_progress(progress_path, data_signature, total, len(first_batch), model_name)
            _log_speed(len(first_batch))
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
            if hasattr(embeddings, '_progress_cb'):
                embeddings._progress_cb = None
            if progress_cb:
                progress_cb("embed", actual_end, total)
            self._save_progress(progress_path, data_signature, total, actual_end, model_name)
            _log_speed(actual_end - done_at_start)

        embed_total_time = time.time() - embed_start_time
        avg_speed = (total - done_at_start) / embed_total_time if embed_total_time > 0 else 0
        logger.info(
            f"向量化完成: {total - done_at_start} 个片段, "
            f"耗时 {_fmt_eta(embed_total_time)}, "
            f"平均 {avg_speed:.1f} chunks/s"
        )

        with open(
            os.path.join(persist_dir, "build_state.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "status": "complete",
                    "data_signature": data_signature,
                    "embedding_model": model_name,
                    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                },
                f,
            )
        try:
            os.remove(progress_path)
        except OSError:
            pass

    def _save_progress(self, path: str, data_signature: str, total: int, completed: int, model_name: str = ""):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "data_signature": data_signature,
                        "embedding_model": model_name or self.embedding_model_name,
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
