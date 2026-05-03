import hashlib
import logging
import asyncio
import time
import re
import ipaddress
import jieba
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

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
        # host 是域名而非 IP，放行（由 DNS 解析后的 TLS 证书校验保护）
        return

    if addr.is_private or addr.is_loopback or addr.is_link_local:
        raise ValueError(f"不允许访问内网地址: {host}")


# 标题中包含以下模式 → 细节页（快照/预发布/RC/子页面），不包含 → 概述页
_DETAIL_TITLE_PATTERNS = [
    re.compile(r'-snapshot-\d+', re.IGNORECASE),
    re.compile(r'-pre-\d+', re.IGNORECASE),
    re.compile(r'-rc-\d+', re.IGNORECASE),
    re.compile(r'/'),
    re.compile(r'测试版本'),
    re.compile(r'开发版本'),
]



def _is_overview_page(title: str) -> bool:
    for pat in _DETAIL_TITLE_PATTERNS:
        if pat.search(title):
            return False
    return True


class _TokenCounter(BaseCallbackHandler):
    """抓取 LLM 调用结束时的 token 用量。"""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        try:
            usage = None
            # DeepSeek / OpenAI 格式的 usage 在不同 LangChain 版本中位置不同
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage')
            if not usage:
                generations = getattr(response, 'generations', [])
                for gen_list in generations:
                    for g in gen_list:
                        msg = getattr(g, 'message', None)
                        if msg and hasattr(msg, 'usage_metadata'):
                            usage = msg.usage_metadata
                            break
            if usage:
                # 兼容两种格式：旧版 prompt_tokens/completion_tokens，新版 input_tokens/output_tokens
                if isinstance(usage, dict):
                    self.prompt_tokens = usage.get('prompt_tokens') or usage.get('input_tokens', 0) or 0
                    self.completion_tokens = usage.get('completion_tokens') or usage.get('output_tokens', 0) or 0
                    self.total_tokens = usage.get('total_tokens', 0) or 0
                else:
                    # UsageMetadata dataclass / TypedDict
                    self.prompt_tokens = getattr(usage, 'input_tokens', 0) or 0
                    self.completion_tokens = getattr(usage, 'output_tokens', 0) or 0
                    self.total_tokens = getattr(usage, 'total_tokens', 0) or 0
        except Exception:
            pass


SESSION_TTL_SECONDS = 1800  # 30 分钟


class RAGEngine:
    """对话检索引擎，依赖外部注入的 KnowledgeBaseManager 提供向量库。"""

    def __init__(self, kb_manager=None):
        self.kb = kb_manager
        self.sessions: Dict[str, List[Dict]] = {}
        self._session_last_access: Dict[str, float] = {}
        self.max_history_window = 10

    @property
    def retriever(self):
        return self.kb.retriever if self.kb else None

    # ── 检索 ──

    async def _transform_query(self, query: str, llm) -> List[str]:
        """Query 改写：版本号走规则（零延迟），其余走 LLM。"""
        # ── 规则扩展：版本号 → 跳过 LLM，直接返回确定好用的搜索词 ──
        ver_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', query)
        if ver_match:
            v = ver_match.group(1)
            terms = [v, f"Java版{v}"]
            if "基岩" in query or "携带" in query or "bedrock" in query.lower():
                terms.append(f"基岩版{v}")
            if query not in terms:
                terms.append(query)
            return terms[:4]

        # ── LLM 改写 ──
        prompt = (
            "你是Minecraft Wiki搜索专家。将以下问题改写为不同的搜索短语，每个短语一行。\n"
            "策略指南：\n"
            "- 提取问题中的核心游戏实体（物品、方块、生物、结构等）作为独立搜索词\n"
            "- 使用Minecraft Wiki上真实存在的条目名称，不要编造\n"
            "- 如果问题涉及游戏机制（如生成、繁殖、合成、更新），将机制名称作为一个短语\n"
            "- 每条短语应当与其它短语有显著差异，不要重复\n\n"
            f"用户问题：{query}"
        )
        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=settings.QUERY_REWRITE_TIMEOUT)
            lines = response.content.strip().split("\n")
            llm_terms = [l.strip("- ").strip() for l in lines if l.strip()]
            llm_terms.insert(0, query)
            return list(dict.fromkeys(llm_terms))[:settings.QUERY_REWRITE_COUNT + 1]
        except (asyncio.TimeoutError, Exception) as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.warning("Query 改写超时")
            else:
                logger.error(f"Query 改写失败: {e}")
            return [query]

    async def retrieve(self, query: str, llm=None, use_bm25: Optional[bool] = None) -> tuple[List[Document], List[str]]:
        """双路径检索：BM25+RRF 混合检索 或 原有关键词启发式检索。"""
        effective_use_bm25 = use_bm25 if use_bm25 is not None else settings.BM25_ENABLED

        search_terms = await self._transform_query(query, llm) if llm else [query]
        if not search_terms:
            search_terms = [query]

        # ── 向量检索（并行） ──
        with self.kb._lock:
            current_retriever = self.retriever
            if not current_retriever:
                return [], search_terms

        async def _invoke(term):
            return await asyncio.to_thread(current_retriever.invoke, term)

        tasks = [_invoke(term) for term in set(search_terms)]
        results = await asyncio.gather(*tasks)

        # ── 公共：jieba 分词 + 完整搜索词 ──
        all_tokens: set = set()      # 完整搜索词 + jieba token，用于标题匹配
        bm25_tokens: set = set()     # 仅 jieba token，用于 BM25 查询
        for term in set(search_terms):
            all_tokens.add(term)
            for t in jieba.cut(term):
                if len(t) > 1:
                    all_tokens.add(t)
                    bm25_tokens.add(t)

        if effective_use_bm25:
            # ══════════ BM25 + RRF 混合检索 ══════════

            # 向量检索：记录每个文档的最佳排名
            vector_rank: Dict[str, int] = {}
            doc_cache: Dict[str, Document] = {}
            for docs_batch in results:
                for rank, doc in enumerate(docs_batch, start=1):
                    h = hashlib.md5(doc.page_content.encode()).hexdigest()
                    doc_cache[h] = doc
                    if h not in vector_rank or rank < vector_rank[h]:
                        vector_rank[h] = rank

            # BM25 检索（仅用 jieba token，完整搜索词留给标题加成）
            bm25_rank: Dict[str, int] = {}
            boost_phrases: List[str] = []
            if bm25_tokens and self.kb:
                # 短语加成：原始查询整体 + jieba 实义 token（过滤停用词）
                stop_words = set(settings.PHRASE_STOP_WORDS.split(","))
                raw_tokens = [t for t in jieba.cut(query) if len(t.strip()) >= 2 and t not in stop_words]
                boost_phrases = [query] + raw_tokens
                bm25_results = self.kb.bm25_search(list(bm25_tokens), top_k=settings.BM25_TOP_K, boost_phrases=boost_phrases)
                for entry in bm25_results:
                    h = hashlib.md5(entry["text"].encode()).hexdigest()
                    if h not in bm25_rank or entry["_bm25_rank"] < bm25_rank[h]:
                        bm25_rank[h] = entry["_bm25_rank"]
                    if h not in doc_cache:
                        doc_cache[h] = Document(
                            page_content=entry["text"],
                            metadata=entry["meta"],
                        )

            # Reciprocal Rank Fusion
            rrf_k = settings.RRF_K
            all_hashes = set(vector_rank.keys()) | set(bm25_rank.keys())
            fused = []
            for h in all_hashes:
                score = 0.0
                in_vector = h in vector_rank
                in_bm25 = h in bm25_rank
                if in_vector:
                    score += 1.0 / (rrf_k + vector_rank[h])
                if in_bm25:
                    score += 1.0 / (rrf_k + bm25_rank[h])
                # 标注来源
                source = "both" if (in_vector and in_bm25) else ("vector" if in_vector else "bm25")
                doc_cache[h].metadata["retrieval_source"] = source
                fused.append((h, score))

            # 标题匹配 + 短语匹配加成（叠加在 RRF 分数上）
            for i, (h, score) in enumerate(fused):
                doc = doc_cache[h]
                title = doc.metadata.get("title", "")
                content = doc.page_content
                combined = title + " " + content
                bonus = 0.0
                # 标题 token 匹配
                for t in all_tokens:
                    if t == title:
                        bonus += settings.TITLE_EXACT_BOOST
                        break
                    elif t in title:
                        bonus += settings.TITLE_SUBSTR_BOOST
                # 概述页加成
                if _is_overview_page(title):
                    bonus += settings.OVERVIEW_BOOST_RRF
                # 原始查询短语匹配（弥补 BM25-only 文档在 RRF 中的劣势）
                for phrase in boost_phrases:
                    if len(phrase) >= 2 and phrase in combined:
                        bonus += settings.QUERY_PHRASE_BOOST
                        break  # 只加一次
                fused[i] = (h, score + bonus)

            fused.sort(key=lambda x: -x[1])
            unique_docs = [doc_cache[h] for h, _ in fused]

        else:
            # ══════════ 原有关键词启发式检索 ══════════
            all_docs = []
            for docs_batch in results:
                all_docs.extend(docs_batch)

            if all_tokens and self.kb:
                for token in all_tokens:
                    entries = self.kb.lookup_by_keyword(token)
                    for entry in entries:
                        all_docs.append(Document(
                            page_content=entry["text"],
                            metadata=entry["meta"],
                        ))

            # 语义去重
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)

            # 关键词 + 概述页加成重排序
            if all_tokens:
                def _keyword_score(doc):
                    title = doc.metadata.get("title", "")
                    content = doc.page_content
                    score = 0
                    for t in all_tokens:
                        if t == title:
                            score += settings.KEYWORD_TITLE_EXACT_BOOST
                        elif t in title:
                            score += settings.KEYWORD_TITLE_SUBSTR_BOOST
                        elif t in content:
                            score += settings.KEYWORD_CONTENT_BOOST
                    if _is_overview_page(title):
                        score += settings.OVERVIEW_BOOST_KEYWORD
                    return score

                scored = [(d, _keyword_score(d), i) for i, d in enumerate(unique_docs)]
                scored.sort(key=lambda x: (-x[1], x[2]))
                unique_docs = [d for d, _, _ in scored]

        # ── 跨文档多样性：每个标题最多 N 条（共用） ──
        title_counts = {}
        diverse_docs = []
        for doc in unique_docs:
            t = doc.metadata.get("title", "__unknown__")
            if title_counts.get(t, 0) < settings.PER_TITLE_CAP:
                diverse_docs.append(doc)
                title_counts[t] = title_counts.get(t, 0) + 1

        # ── 按总字数截断：累计到 MAX_CONTEXT_CHARS 为止 ──
        max_chars = settings.MAX_CONTEXT_CHARS
        selected = []
        total_chars = 0
        for doc in diverse_docs:
            doc_len = len(doc.page_content)
            if total_chars + doc_len > max_chars and selected:
                break
            selected.append(doc)
            total_chars += doc_len

        return selected, search_terms

    # ── 会话管理 ──

    def _evict_expired_sessions(self):
        now = time.monotonic()
        expired = [
            sid for sid, ts in self._session_last_access.items()
            if now - ts > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            self._session_last_access.pop(sid, None)
        if expired:
            logger.info(f"清理过期会话 {len(expired)} 个")

    def get_session_history(self, session_id: str) -> List[Dict]:
        self._evict_expired_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self._session_last_access[session_id] = time.monotonic()
        return self.sessions[session_id]

    def add_to_session(self, session_id: str, role: str, content: str):
        history = self.get_session_history(session_id)
        history.append({"role": role, "content": content})
        if len(history) > self.max_history_window * 2:
            self.sessions[session_id] = history[-(self.max_history_window * 2):]
        self._session_last_access[session_id] = time.monotonic()

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
        use_bm25: Optional[bool] = None,
    ) -> dict:
        if not self.retriever:
            raise RuntimeError("RAG Engine not initialized")

        llm, llm_config = self.get_chat_model(
            model_choice, api_key, base_url, model_name,
            streaming=streaming,
        )
        system_prompt = self.build_system_prompt(answer_detail)

        retrieve_start = time.time()
        docs, search_terms = await self.retrieve(message, llm=llm, use_bm25=use_bm25)
        retrieve_time_ms = (time.time() - retrieve_start) * 1000

        include_history = session_id is not None
        qa_chain = self.get_qa_chain(llm, answer_detail, include_history=include_history)
        token_counter = _TokenCounter()

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
                "retrieve_time_ms": retrieve_time_ms,
                "token_counter": token_counter,
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

        # 已知域名直接用正确的 URL，跳过探测
        _KNOWN_MODELS_URL = {
            "api.deepseek.com": "https://api.deepseek.com/v1/models",
        }
        host = urlparse(url).hostname or ""
        if host in _KNOWN_MODELS_URL:
            test_urls = [_KNOWN_MODELS_URL[host]]
        else:
            test_urls = [f"{url}/models"]
            if "/v1" not in url:
                test_urls.insert(0, f"{url}/v1/models")

        import httpx
        errors = []

        async with httpx.AsyncClient(timeout=3.0) as client:
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
            "temperature": 0.3,
            "streaming": streaming,
        }

        model_kwargs = {}
        if model_choice == "api":
            model_kwargs["stream_options"] = {"include_usage": True}

        return ChatOpenAI(
            base_url=url,
            api_key=key,
            model=name,
            temperature=0.3,
            streaming=streaming,
            model_kwargs=model_kwargs,
        ), config

    @staticmethod
    def build_system_prompt(answer_detail="标准"):
        detail_reqs = {
            "简洁": "回答控制在3到5句，聚焦关键结论。",
            "标准": "先给结论，再展开解释。涉及操作流程时给出步骤（不少于5条），并补充注意事项。",
            "详细": (
                "先给出准确结论，再展开详细解释。\n"
                "请根据问题类型灵活选择回答结构：\n"
                "- 如果问题涉及操作流程或合成配方：给出步骤和材料\n"
                "- 如果问题是事实查询：直接列出数据和要点\n"
                "- 如果涉及多个变体或版本：用表格或列表对比\n"
                "- 内容充分展开，确保覆盖所有相关细节\n"
                "- 不要套用固定的章节模板，让内容决定结构"
            ),
        }
        return (
            "你是一个Minecraft知识库智能助手。优先使用以下检索到的背景信息回答问题。\n"
            "如果背景信息不足以回答，可以结合你的知识补充，但必须明确标注「以下为推测内容，可能不准确」。\n"
            "重要约定：游戏默认指Java版，除非明确说了基岩版。\n"
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
