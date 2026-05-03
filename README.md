# Minecraft Wiki 知识库智能助手 (RAG-MCwiki)

基于 RAG（检索增强生成）的 Minecraft Wiki 知识问答系统。采集中文 Minecraft Wiki 数据，通过 ChromaDB 建立向量索引，结合本地或云端 LLM 生成有据可依的回答。

## 核心特性

- **混合检索**：BM25 + 向量 RRF 融合检索，短语匹配加成 + 标题/概述页重排序，覆盖专有名词和社区黑话
- **多模型嵌入**：支持 HuggingFace / ONNX / GGUF 三种嵌入后端，运行时一键切换
- **流式对话**：SSE 实时推送，首字响应 < 1s，支持中途停止生成
- **透明溯源**：每条回答附带参考来源，支持查看原始 Wiki 链接
- **专业调试**：内置全链路追踪面板，展示查询改写、检索片段、Token 统计、各阶段耗时
- **Bot API**：`/chat/bot` 端点支持 QQ Bot 等外部接入，自带会话管理和 Bearer Token 认证

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | Next.js 16 + React 19 + Tailwind CSS + shadcn/ui |
| 后端 | FastAPI + LangChain + ChromaDB + rank_bm25 |
| 嵌入模型 | Yuan-embedding-2.0-zh / Qwen3-Embedding-0.6B / gte-small-zh (ONNX) |
| 分词 | Jieba（中文分词 + BM25 索引） |
| LLM | LM Studio (本地) / DeepSeek API (云端) — OpenAI 兼容协议 |
| 数据源 | 中文 Minecraft Wiki ZIM 离线包 + MediaWiki API 增量更新 |

## 项目结构

```text
.
├── backend/
│   ├── main.py                    # FastAPI 入口
│   ├── app/
│   │   ├── api/endpoints.py       # REST API 路由
│   │   ├── core/
│   │   │   ├── config.py          # 配置管理（pydantic-settings，支持 .env）
│   │   │   ├── rag_engine.py      # RAG 对话引擎（查询改写 + 检索 + 生成）
│   │   │   ├── kb_manager.py      # 知识库管理器（向量库 CRUD + 倒排索引）
│   │   │   ├── embedding_registry.py  # 嵌入模型预设注册表
│   │   │   └── embedding_backends.py  # 嵌入后端工厂（HF / ONNX / GGUF）
│   │   └── schemas/chat.py        # Pydantic 请求/响应模型
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # 主聊天页（SSE 流式 + Markdown 渲染）
│   │   └── services/api.ts        # API 客户端
│   ├── components/
│   │   ├── LeftSidebar.tsx        # 知识库状态 + 嵌入模型选择
│   │   ├── RightSidebar.tsx       # 推理配置面板
│   │   └── ProfessionalTrace.tsx  # 全链路追踪面板
│   └── package.json
├── scripts/
│   ├── wiki_manager.py            # Wiki 数据管理（ZIM 导出 + API 增量更新）
│   ├── wiki_cleaner.py            # HTML → Markdown 清洗
│   ├── manage_kb.py               # 知识库管理 TUI（Embedding 构建/切换/清理 + BM25 构建）
│   ├── dev-up.ps1                 # 开发环境启动脚本
│   └── dev-down.ps1               # 开发环境停止脚本
├── data/
│   ├── markdown/                  # 清洗后的 Markdown 文件（8300+ 篇）
│   ├── chroma_db/                 # 向量数据库 + BM25 索引缓存（按模型分目录）
│   ├── html_cache/                # HTML 缓存（ZIM + API 双源）
│   └── page_metadata.json         # 页面元数据（增量同步用）
├── .env.example                   # 环境变量模板
├── CLAUDE.md                      # Claude Code 项目指南
└── start-dev.bat / stop-dev.bat   # Windows 一键启停
```

## 快速开始

### 一键启动（Windows）

```powershell
# 同时启动前后端
powershell -ExecutionPolicy Bypass -File .\scripts\dev-up.ps1

# 仅启动后端
powershell -ExecutionPolicy Bypass -File .\scripts\dev-up.ps1 -SkipFrontend

# 仅启动前端
powershell -ExecutionPolicy Bypass -File .\scripts\dev-up.ps1 -SkipBackend

# 停止全部
powershell -ExecutionPolicy Bypass -File .\scripts\dev-down.ps1
```

或直接双击 `start-dev.bat` / `stop-dev.bat`。

### 手动启动

**1. 后端**

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖（如需 GPU 加速，先安装 CUDA 版 PyTorch）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt

# 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 设置 BOT_API_KEY、LLM 地址等

# 启动
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. 前端**

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:3000`。

**3. 构建知识库**

```bash
# 交互式 TUI（含 Embedding 构建/切换/清理 + BM25 构建）
python scripts/manage_kb.py

# 或直接构建 Embedding（使用默认模型）
python scripts/manage_kb.py build

# 构建 BM25 索引（独立于 Embedding，可单独运行）
python scripts/manage_kb.py bm25
```

### 配置

所有配置项均可通过环境变量或 `.env` 文件覆盖，参见 `.env.example`：

**模型与地址**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL_ID` | `yuan2-hf` | 嵌入模型（yuan2-hf / qwen3-hf / qwen3-gguf / gte-quant） |
| `LOCAL_LLM_URL` | `http://localhost:1234/v1` | 本地 LLM 地址 |
| `DEEPSEEK_API_URL` | `https://api.deepseek.com` | 云端 LLM 地址 |
| `BOT_API_KEY` | _(空)_ | Bot API 认证密钥（未设置则跳过认证） |

**检索参数**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVAL_K` | `20` | 向量检索每个搜索词返回条数 |
| `BM25_ENABLED` | `true` | 是否启用 BM25+RRF 混合检索 |
| `BM25_K1` | `1.5` | BM25 词频饱和度（典型 1.2-2.0） |
| `BM25_B` | `0.75` | BM25 文档长度归一化（0=不归一化，1=完全归一化） |
| `BM25_TOP_K` | `20` | BM25 返回条数 |
| `RRF_K` | `60` | RRF 融合常数（典型 30-100） |
| `MAX_CONTEXT_CHARS` | `7000` | 召回片段总字数上限 |
| `QUERY_REWRITE_COUNT` | `3` | LLM 改写搜索词数量 |
| `QUERY_REWRITE_TIMEOUT` | `5.0` | 查询改写超时秒数 |
| `PER_TITLE_CAP` | `3` | 同一标题最多保留条数 |

**重排序加成**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TITLE_EXACT_BOOST` | `0.01` | 标题精确匹配加成（RRF 模式） |
| `TITLE_SUBSTR_BOOST` | `0.005` | 标题子串匹配加成（RRF 模式） |
| `OVERVIEW_BOOST_RRF` | `0.003` | 概述页加成（RRF 模式） |
| `QUERY_PHRASE_BOOST` | `0.025` | 原始查询短语匹配加成 |
| `PHRASE_STOP_WORDS` | `版本,什么,...` | 短语匹配停用词（逗号分隔） |
| `OVERVIEW_BOOST_KEYWORD` | `6` | 概述页加成（关键词模式） |
| `KEYWORD_TITLE_EXACT_BOOST` | `10` | 标题精确匹配加成（关键词模式） |
| `KEYWORD_TITLE_SUBSTR_BOOST` | `5` | 标题子串匹配加成（关键词模式） |
| `KEYWORD_CONTENT_BOOST` | `1` | 内容匹配加成（关键词模式） |

## RAG 流程

```
用户提问
  → 查询改写（版本号走规则扩展，其余走 LLM 生成 3 个搜索短语）
  → 双路并行检索
      ├─ 向量检索：每个搜索词 → ChromaDB top-K（默认 20）
      └─ BM25 检索：jieba 分词 → BM25Okapi top-K（默认 20）
  → RRF 融合：score = Σ 1/(k + rank_i)，叠加标题/概述页/短语匹配加成
  → 多样性过滤：同标题最多 3 条
  → 字数截断：按相关性排序累加到 MAX_CONTEXT_CHARS（默认 7000 字）
  → 文档送入 LangChain stuff chain
  → LLM 流式生成 → SSE 推送前端（每条来源标注 vector / bm25 / both）
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | SSE 流式对话（前端用） |
| `POST` | `/api/v1/chat/bot` | JSON 对话（Bot 接入，需 Bearer Token） |
| `POST` | `/api/v1/models` | 获取可用模型列表 |
| `GET` | `/api/v1/knowledge_base/status` | 知识库状态（含 Embedding 模型信息） |
| `GET` | `/api/v1/knowledge_base/models` | 所有嵌入模型列表及构建状态 |
| `POST` | `/api/v1/knowledge_base/switch_model` | 运行时切换嵌入模型 |
| `GET` | `/api/v1/health` | 健康检查 |

## 数据更新

```bash
# 1. 从 MediaWiki API 获取全站页面元数据
python scripts/wiki_manager.py meta

# 2. 从 ZIM 离线包导出 HTML（首次全量）
python scripts/wiki_manager.py export

# 3. 增量更新变更页面
python scripts/wiki_manager.py update

# 4. 构建/更新 Embedding 向量库
python scripts/manage_kb.py build

# 5. 构建/更新 BM25 索引（独立于 Embedding，直接读取 markdown 文件）
python scripts/manage_kb.py bm25
```

## 已知限制

- **社区黑话**：Jieba 分词对"刷线机"等社区俗称有效（整词保留），但 LLM 改写可能丢失原始关键词，依赖短语匹配加成兜底
- **版本号区分**：向量检索无法区分 `26.1` 与 `26.10`，已通过规则扩展缓解
- **知识库覆盖**：部分问题（如 mod 生态历史）知识库中无对应文章，依赖 LLM 自身知识
- **短 chunk**：知识库中存在大量 stub 页面（< 50 字），向量检索可能召回这些低价值片段

## License

本项目仅供学习和个人使用。Wiki 内容遵循 [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) 协议。
