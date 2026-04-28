import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router as api_router
from backend.app.core.rag_engine import rag_engine
from backend.app.core.kb_manager import kb_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-MCwiki API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_rag_bg():
    """后台初始化知识库和 RAG 引擎，避免阻塞服务器启动"""
    logger.info("知识库正在后台进行初始化...")
    try:
        await asyncio.to_thread(kb_manager.initialize)
        rag_engine.kb = kb_manager
        logger.info("知识库后台初始化成功")
    except Exception as e:
        logger.error(f"知识库后台初始化失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("正在启动 RAG-MCwiki 接口服务...")
    # 立即启动后台初始化任务
    asyncio.create_task(initialize_rag_bg())

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0 确保局域网可访问，并启用较短的超时
    uvicorn.run(app, host="0.0.0.0", port=8000)
