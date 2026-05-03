from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "RAG-MCwiki"

    # 路径配置
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: str = ""
    PERSIST_DIR: str = ""

    # 模型配置
    EMBEDDING_MODEL_NAME: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Deprecated: 向后兼容，请使用 EMBEDDING_MODEL_ID",
    )
    EMBEDDING_MODEL_ID: str = "yuan2-hf"
    LOCAL_LLM_URL: str = "http://localhost:1234/v1"
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"

    # ── 检索配置 ──

    # 向量检索：每个搜索词从 ChromaDB 返回的最大条数
    # LLM 改写约 3-4 个搜索词，总召回 ≈ RETRIEVAL_K × 词数
    RETRIEVAL_K: int = 20

    # 是否启用 BM25 关键词检索（与向量检索组成混合检索）
    BM25_ENABLED: bool = True
    # BM25 参数 k1：词频饱和度，越大越信任高频词（典型范围 1.2-2.0）
    BM25_K1: float = 1.5
    # BM25 参数 b：文档长度归一化强度，0=不归一化，1=完全归一化
    BM25_B: float = 0.75
    # BM25 检索返回的最大条数
    BM25_TOP_K: int = 20

    # RRF（Reciprocal Rank Fusion）融合常数 k
    # score = Σ 1/(k + rank_i)，k 越大排名差异越小（典型范围 30-100）
    RRF_K: int = 60

    # 召回片段总字数上限，按相关性排序逐条累加直到此值
    MAX_CONTEXT_CHARS: int = 7000

    # LLM 查询改写的搜索词数量（不含原始查询，实际总数 = 此值 + 1）
    QUERY_REWRITE_COUNT: int = 3
    # 查询改写 LLM 超时（秒），超时则退回原始查询
    QUERY_REWRITE_TIMEOUT: float = 5.0

    # 同一标题最多保留的 chunk 条数，防止单页面占满结果
    PER_TITLE_CAP: int = 3

    # ── 重排序加成系数 ──

    # 标题精确匹配加成：搜索词 == 标题时加分
    TITLE_EXACT_BOOST: float = 0.01
    # 标题子串匹配加成：搜索词 ⊂ 标题时加分
    TITLE_SUBSTR_BOOST: float = 0.005
    # 概述页加成（RRF 模式）：非快照/预发布/RC/子页面时加分（概述页通常更完整）
    OVERVIEW_BOOST_RRF: float = 0.003
    # 概述页加成（关键词模式）：同上，但用于非 BM25 的关键词打分（整数分制）
    OVERVIEW_BOOST_KEYWORD: int = 6
    # 关键词模式标题匹配加成：搜索词 == 标题
    KEYWORD_TITLE_EXACT_BOOST: int = 10
    # 关键词模式标题子串匹配加成：搜索词 ⊂ 标题
    KEYWORD_TITLE_SUBSTR_BOOST: int = 5
    # 关键词模式内容匹配加成：搜索词 ⊂ 正文
    KEYWORD_CONTENT_BOOST: int = 1
    # 原始查询短语匹配加成：文档包含原始查询关键词时加分
    # 用于弥补 BM25-only 文档在 RRF 融合中相对"双重来源"文档的分数劣势
    QUERY_PHRASE_BOOST: float = 0.025

    # 短语匹配停用词列表（逗号分隔），这些词不参与短语加成
    # 避免"版本""失效"等泛词匹配到大量无关文档
    PHRASE_STOP_WORDS: str = "版本,什么,哪个,哪些,怎么,如何,为什么,可以,不能,失效,更新,获取,使用"

    # ── Embedding 构建配置 ──

    EMBEDDING_BATCH_SIZE: int = 32
    # chunk 字符数，0 = 使用模型预设值
    EMBEDDING_CHUNK_SIZE: int = 0
    # 目标 Token 数（RAG 建议 500 左右）
    EMBEDDING_TARGET_TOKENS: int = 500
    # 目标 token 利用率，动态调整 chunk_size 使最大 token 数逼近此比例
    EMBEDDING_TOKEN_TARGET_UTILIZATION: float = 0.95

    # 认证
    BOT_API_KEY: Optional[str] = None

    def model_post_init(self, __context):
        if not self.DATA_DIR:
            object.__setattr__(self, "DATA_DIR", str(Path(self.BASE_DIR).parent / "data" / "markdown"))
        if not self.PERSIST_DIR:
            object.__setattr__(self, "PERSIST_DIR", str(Path(self.BASE_DIR).parent / "data" / "chroma_db"))


settings = Settings()
