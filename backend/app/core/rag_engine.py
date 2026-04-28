import hashlib
import logging
import asyncio
import ipaddress
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
]


def _validate_llm_url(url: str, provider: str) -> None:
    """校验 LLM base_url 防止 SSRF。local 提供者允许 localhost；api 提供者只允许已知外部域名。"""
    parsed = urlparse(url)
    host = parsed.hostname or ""

    if provider == "local":
        if host in ("localhost", "127.0.0.1", "::1"):
            return
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return
        except ValueError:
            pass

    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"不支持的协议: {parsed.scheme}")

    if provider != "local":
        if parsed.scheme != "https":
            raise ValueError("云端 API 必须使用 HTTPS")

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return

    for r in _PRIVATE_IP_RANGES:
        if addr in r:
            raise ValueError(f"不允许访问内网地址: {host}")


class RAGEngine:
    """对话检索引擎，依赖外部注入的 KnowledgeBaseManager 提供向量库。"""

    def __init__(self, kb_manager=None):
        self.kb = kb_manager
        self.sessions: Dict[str, List[Dict]] = {}
        self.max_history_window = 10

    @property
    def retriever(self):
        return self.kb.retriever if self.kb else None

    # ── 检索 ──

    async def _transform_query(self, query: str, llm) -> List[str]:
        """Query 改写：将用户口语转化为精准的搜索短语。"""
        prompt = (
            "你是一个Minecraft搜索专家。请将用户的提问改写为3个用于知识库检索的短语。\n"
            "要求：1. 包含核心实体；2. 使用维基风格术语；3. 只输出短语，每行一个。\n\n"
            f"用户问题：{query}"
        )
        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=10.0)
            lines = response.content.strip().split("\n")
            transformed = [l.strip("- ").strip() for l in lines if l.strip()]
            return transformed[:3]
        except asyncio.TimeoutError:
            logger.warning("Query 改写超时，使用原句检索")
            return [query]
        except Exception as e:
            logger.error(f"Query 改写失败: {e}")
            return [query]

    async def retrieve(self, query: str, llm=None) -> tuple[List[Document], List[str]]:
        """整合改写后的多路向量检索。"""
        search_terms = [query]
        if llm:
            transformed = await self._transform_query(query, llm)
            search_terms.extend(transformed)

        all_docs = []
        with self.kb._lock:
            current_retriever = self.retriever
            if not current_retriever:
                return [], search_terms

            for term in set(search_terms):
                all_docs.extend(current_retriever.invoke(term))

        # 语义去重
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)

        return unique_docs[:8], search_terms

    # ── 会话管理 ──

    def get_session_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_to_session(self, session_id: str, role: str, content: str):
        history = self.get_session_history(session_id)
        history.append({"role": role, "content": content})
        if len(history) > self.max_history_window * 2:
            self.sessions[session_id] = history[-(self.max_history_window * 2):]

    def format_history_for_chain(self, history: List[Dict]) -> List:
        result = []
        for msg in history:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                result.append(AIMessage(content=msg["content"]))
        return result

    # ── 共享对话逻辑 ──

    async def chat(
        self,
        message: str,
        model_choice: str = "api",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        answer_detail: str = "标准",
        session_id: Optional[str] = None,
        streaming: bool = True,
    ) -> dict:
        if not self.retriever:
            raise RuntimeError("RAG Engine not initialized")

        llm, llm_config = self.get_chat_model(
            model_choice, api_key, base_url, model_name,
            streaming=streaming,
        )
        system_prompt = self.build_system_prompt(answer_detail)

        docs, search_terms = await self.retrieve(message, llm=llm)

        include_history = session_id is not None
        qa_chain = self.get_qa_chain(llm, answer_detail, include_history=include_history)

        chain_input: dict = {"input": message, "context": docs}
        if include_history:
            history = self.get_session_history(session_id)
            chain_input["chat_history"] = self.format_history_for_chain(history)

        if streaming:
            return {
                "qa_chain": qa_chain,
                "chain_input": chain_input,
                "docs": docs,
                "search_terms": search_terms,
                "llm_config": llm_config,
                "system_prompt": system_prompt,
            }

        try:
            result = await qa_chain.ainvoke(chain_input)
            content = result if isinstance(result, str) else str(result)
        except Exception:
            logger.warning("ainvoke 失败，尝试不带历史重试")
            chain_input["chat_history"] = []
            try:
                result = await qa_chain.ainvoke(chain_input)
                content = result if isinstance(result, str) else str(result)
            except Exception as e2:
                raise RuntimeError(f"对话调用失败: {e2}") from e2

        if session_id is not None:
            self.add_to_session(session_id, "user", message)
            self.add_to_session(session_id, "assistant", content)

        return {
            "content": content,
            "docs": docs,
            "search_terms": search_terms,
            "llm_config": llm_config,
            "system_prompt": system_prompt,
        }

    async def list_models(self, provider: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> List[str]:
        """从指定供应商获取可用模型列表。"""
        url = (base_url or (settings.LOCAL_LLM_URL if provider == "local" else settings.DEEPSEEK_API_URL)).rstrip('/')
        _validate_llm_url(url, provider)
        key = "lm-studio" if provider == "local" else api_key

        test_urls = [f"{url}/models"]
        if "/v1" not in url:
            test_urls.insert(0, f"{url}/v1/models")

        import httpx
        errors = []

        async with httpx.AsyncClient(timeout=5.0) as client:
            for target_url in test_urls:
                try:
                    response = await client.get(
                        target_url,
                        headers={"Authorization": f"Bearer {key}"} if key else {}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and isinstance(data["data"], list):
                            return [m["id"] for m in data["data"]]
                    else:
                        errors.append(f"{target_url} 返回状态码 {response.status_code}")
                except Exception as e:
                    errors.append(f"请求 {target_url} 失败: {str(e)}")

        error_msg = " | ".join(errors[-2:])
        logger.error(f"获取模型列表失败: {error_msg}")
        raise Exception(f"模型列表拉取失败: {error_msg}")

    def get_chat_model(self, model_choice: str, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None, streaming: bool = True):
        url = base_url
        key = api_key
        name = model_name

        if model_choice == "local":
            url = url or settings.LOCAL_LLM_URL
            key = key or "lm-studio"
            name = name or "Qwen/Qwen3.5-9B"
            backend = "LM Studio"
        else:
            url = url or settings.DEEPSEEK_API_URL
            name = name or "deepseek-chat"
            backend = "DeepSeek API"

        _validate_llm_url(url, model_choice)

        config = {
            "backend": backend,
            "base_url": url,
            "model": name,
            "temperature": 1.0,
            "streaming": streaming,
        }

        return ChatOpenAI(
            base_url=url,
            api_key=key,
            model=name,
            temperature=1.0,
            streaming=streaming,
        ), config

    @staticmethod
    def build_system_prompt(answer_detail="标准"):
        detail_reqs = {
            "简洁": "回答控制在3到5句，聚焦关键结论。",
            "标准": "先给结论，再给步骤，步骤不少于5条，并补充1到2条注意事项。",
            "详细": "按「结论 → 具体步骤 → 材料清单 → 常见错误 → 进阶优化」输出，步骤不少于8条。",
        }
        return (
            "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。\n"
            "如果你不知道答案，就说你不知道，不要编造。\n"
            f"回答详细度要求：{detail_reqs.get(answer_detail, detail_reqs['标准'])}\n\n"
            "背景信息：\n{context}"
        )

    def get_qa_chain(self, llm, answer_detail="标准", include_history=False):
        system_prompt = self.build_system_prompt(answer_detail)
        messages = [("system", system_prompt)]
        if include_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
        messages.append(("human", "{input}"))
        prompt = ChatPromptTemplate.from_messages(messages)
        return create_stuff_documents_chain(llm, prompt)


rag_engine = RAGEngine()  # kb_manager 由 main.py 在启动时注入
