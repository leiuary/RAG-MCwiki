import os
import json
import shutil
import sqlite3
import time
import hashlib
import logging
import threading
from datetime import datetime, timezone
import jieba
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.vectorstore = None
        self.retriever = None
        self._lock = threading.RLock()
        self.stopwords = {"的", "是", "了", "在", "怎么", "什么", "和", "与", "如何", "帮我", "查一下", "告诉我", "有哪些"}

    def _init_embeddings(self, model_name: str):
        self.embedding_model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
        
    def init_models(self):
        """初始化 Embedding 模型和向量库"""
        with self._lock:
            logger.info("正在初始化 Embedding 模型...")
            self._init_embeddings(settings.EMBEDDING_MODEL_NAME)

            data_signature = self._compute_data_signature(settings.DATA_DIR)

            if self._is_vectorstore_ready(settings.PERSIST_DIR, data_signature):
                logger.info("连接到现有向量库...")
                self.vectorstore = Chroma(
                    persist_directory=settings.PERSIST_DIR,
                    embedding_function=self.embeddings
                )
            else:
                logger.info("向量库不存在或已过期，开始构建...")
                self._build_vectorstore(data_signature)

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})

    def rebuild_vectorstore(self, model_name: Optional[str] = None):
        """按指定 embedding 模型重建向量库"""
        with self._lock:
            selected_model = model_name or self.embedding_model_name or settings.EMBEDDING_MODEL_NAME
            logger.info(f"使用 Embedding 模型重建向量库: {selected_model}")
            self._init_embeddings(selected_model)
            data_signature = self._compute_data_signature(settings.DATA_DIR)
            self._build_vectorstore(data_signature)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})

    def clean_vectorstore(self):
        """清理向量库"""
        with self._lock:
            if os.path.exists(settings.PERSIST_DIR):
                shutil.rmtree(settings.PERSIST_DIR)
            self.vectorstore = None
            self.retriever = None
            logger.info("向量库已清理")

    def get_runtime_status(self) -> Dict[str, Any]:
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

    def _compute_data_signature(self, folder_path: str) -> str:
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

    def _is_vectorstore_ready(self, persist_dir: str, data_signature: str) -> bool:
        state_path = os.path.join(persist_dir, "build_state.json")
        if not os.path.exists(state_path):
            return False
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("status") == "complete" and state.get("data_signature") == data_signature:
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
        
        # 引入分词器作为兜底，防止单个章节过长
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        for idx, filename in enumerate(json_files):
            if idx % 100 == 0:
                logger.info(f"加载进度: {idx}/{total_files}...")
            filepath = os.path.join(settings.DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            title = data.get('title', '未知标题')
            source_url = data.get('source_url', '')
            
            # 2.1 结构化分片：以章节 (Section) 为基本单位
            if 'structured_content' in data:
                for section_title, text_list in data['structured_content'].items():
                    section_text = "\n".join([t for t in text_list if t.strip()])
                    if not section_text: continue
                    
                    full_content = f"## {title} - {section_title} ##\n{section_text}"
                    
                    # 如果章节内容适中，直接作为一个 Doc；如果太长则二次切分
                    if len(full_content) < 1000:
                        documents.append(Document(
                            page_content=full_content,
                            metadata={'title': title, 'section': section_title, 'source_url': source_url}
                        ))
                    else:
                        sub_docs = text_splitter.create_documents(
                            [full_content], 
                            metadatas=[{'title': title, 'section': section_title, 'source_url': source_url}]
                        )
                        documents.append(sub_docs)
            else:
                text = data.get('text', '')
                if text:
                    documents.append(text_splitter.create_documents(
                        [text], 
                        metadatas=[{'title': title, 'source_url': source_url}]
                    ))
        
        # 处理嵌套列表
        flattened_docs = []
        for d in documents:
            if isinstance(d, list): flattened_docs.extend(d)
            else: flattened_docs.append(d)

        logger.info(f"结构化分片完成，共生成 {len(flattened_docs)} 个片段")
        self.vectorstore = Chroma.from_documents(
            documents=flattened_docs,
            embedding=self.embeddings,
            persist_directory=settings.PERSIST_DIR
        )
        
        with open(os.path.join(settings.PERSIST_DIR, "build_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "complete",
                    "data_signature": data_signature,
                    "embedding_model": self.embedding_model_name,
                    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                },
                f,
            )

    async def _transform_query(self, query: str, llm) -> List[str]:
        """2.2 Query 改写：将用户口语转化为精准的搜索短语"""
        prompt = (
            "你是一个Minecraft搜索专家。请将用户的提问改写为3个用于知识库检索的短语。\n"
            "要求：\n"
            "1. 包含核心实体（如 Java版, 红石, 26.1等）\n"
            "2. 使用维基百科风格的术语\n"
            "3. 只输出短语，每行一个，不要有其他解释。\n\n"
            f"用户问题：{query}"
        )
        try:
            # 简单调用模型生成，这里不使用流式
            response = await llm.ainvoke(prompt)
            lines = response.content.strip().split("\n")
            # 清洗结果
            transformed = [l.strip("- ").strip() for l in lines if l.strip()]
            logger.info(f"Query 改写结果: {transformed}")
            return transformed[:3]
        except Exception as e:
            logger.error(f"Query 改写失败: {e}")
            return [query]

    async def retrieve(self, query: str, llm=None) -> tuple[List[Document], List[str]]:
        """整合改写后的查询逻辑"""
        search_terms = [query]
        if llm:
            transformed = await self._transform_query(query, llm)
            search_terms.extend(transformed)

        all_docs = []
        with self._lock:
            current_retriever = self.retriever
            if not current_retriever:
                return [], search_terms

            # 移除原有的 jieba 逻辑，改为基于改写后的多路向量检索
            for term in set(search_terms):
                all_docs.extend(current_retriever.invoke(term))
            
        # 语义去重
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            # 简单的内容 hash 去重
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        return unique_docs[:8], search_terms

    def get_chat_model(self, model_choice: str, api_key: Optional[str] = None):
        if model_choice == "local":
            return ChatOpenAI(
                base_url=settings.LOCAL_LLM_URL,
                api_key="lm-studio",
                model="Qwen/Qwen3.5-9B",
                temperature=0.7,
                streaming=True
            )
        else:
            return ChatOpenAI(
                base_url=settings.DEEPSEEK_API_URL,
                api_key=api_key,
                model="deepseek-chat",
                temperature=0.1,
                streaming=True
            )

    def get_qa_chain(self, llm, answer_detail="标准"):
        detail_reqs = {
            "简洁": "回答控制在3到5句，聚焦关键结论。",
            "标准": "先给结论，再给步骤，步骤不少于5条，并补充1到2条注意事项。",
            "详细": "按“结论 -> 具体步骤 -> 材料清单 -> 常见错误 -> 进阶优化”输出，步骤不少于8条。"
        }
        
        system_prompt = (
            "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。\n"
            "如果你不知道答案，就说你不知道，不要编造。\n"
            f"回答详细度要求：{detail_reqs.get(answer_detail, detail_reqs['标准'])}\n\n"
            "背景信息：\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(llm, prompt)

rag_engine = RAGEngine()
