import json
import asyncio
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.app.schemas.chat import ChatRequest
from backend.app.core.rag_engine import rag_engine

class BuildRequest(BaseModel):
    model: str = "shibing624/text2vec-base-chinese"


class SyncRequest(BaseModel):
    model: Optional[str] = None

router = APIRouter()

@router.post("/knowledge_base/build")
async def build_kb(req: BuildRequest):
    await asyncio.to_thread(rag_engine.rebuild_vectorstore, req.model)
    return {
        "status": "success",
        "message": "知识库构建完成",
        "embedding_model": rag_engine.embedding_model_name,
    }

@router.post("/knowledge_base/clean")
async def clean_kb():
    await asyncio.to_thread(rag_engine.clean_vectorstore)
    return {"status": "success", "message": "知识库已清理"}


@router.post("/knowledge_base/sync")
async def sync_kb(req: SyncRequest):
    await asyncio.to_thread(rag_engine.rebuild_vectorstore, req.model)
    return {
        "status": "success",
        "message": "知识库同步完成",
        "embedding_model": rag_engine.embedding_model_name,
    }

@router.get("/knowledge_base/status")
async def kb_status():
    runtime_status = rag_engine.get_runtime_status()
    return {
        "status": "ready" if runtime_status["ready"] else "empty",
        "version": runtime_status["version"],
        "embedding_model": runtime_status["embedding_model"],
    }

@router.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    if not rag_engine.retriever:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    
    # 0. 获取 LLM 实例 (用于 Query 改写和最终回答)
    llm = rag_engine.get_chat_model(request.model_choice, request.api_key)
    
    # 1. 检索上下文 (传入 llm 以支持 Query 改写)
    retrieve_start = time.time()
    docs, search_terms = await rag_engine.retrieve(request.message, llm=llm)
    retrieve_time = (time.time() - retrieve_start) * 1000
    
    # 2. 准备引用来源
    sources = [
        {
            "title": d.metadata.get("title"), 
            "section": d.metadata.get("section"),
            "source_url": d.metadata.get("source_url"), 
            "content": d.page_content
        }
        for d in docs
    ]
    
    # 3. 获取对话链
    qa_chain = rag_engine.get_qa_chain(llm, request.answer_detail)
    # 获取原始 system_prompt 用于展示
    detail_reqs = {
        "简洁": "回答控制在3到5句，聚焦关键结论。",
        "标准": "先给结论，再给步骤，步骤不少于5条，并补充1到2条注意事项。",
        "详细": "按“结论 -> 具体步骤 -> 材料清单 -> 常见错误 -> 进阶优化”输出，步骤不少于8条。"
    }
    current_detail = detail_reqs.get(request.answer_detail, detail_reqs["标准"])
    system_prompt = (
        "你是一个Minecraft知识库智能助手。请使用以下检索到的背景信息来回答问题。\n"
        "如果你不知道答案，就说你不知道，不要编造。\n"
        f"回答详细度要求：{current_detail}\n\n"
        "背景信息：\n{context}"
    )
    
    trace_data = {
        "user_input": request.message,
        "model_choice": request.model_choice,
        "answer_detail": request.answer_detail,
        "search_terms": search_terms,
        "retrieved_chunk_count": len(docs),
        "retrieve_time_ms": round(retrieve_time, 2),
        "system_prompt": system_prompt
    }
    
    # 4. 流式输出
    async def event_generator():
        # 发送 trace 数据
        yield f"data: {json.dumps({'type': 'trace', 'data': trace_data})}\n\n"
        
        # 先发送引用来源
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        first_token_time = None
        total_chars = 0
        
        # 迭代流式文本
        # 注意：create_stuff_documents_chain 返回的是一个 Runnable
        # 使用 astream 处理异步流
        async for chunk in qa_chain.astream({"input": request.message, "context": docs}):
            if first_token_time is None:
                first_token_time = time.time()
            if chunk:
                total_chars += len(chunk)
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                
        # 发送性能 trace
        total_time = (time.time() - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        yield f"data: {json.dumps({'type': 'trace_perf', 'data': {'ttft_ms': round(ttft, 2), 'total_time_ms': round(total_time, 2), 'output_chars': total_chars}})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/health")
async def health_check():
    return {"status": "ok", "retriever_ready": rag_engine.retriever is not None}
