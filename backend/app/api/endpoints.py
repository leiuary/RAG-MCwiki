import json
import asyncio
import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from backend.app.schemas.chat import ChatRequest, BotChatRequest, BotChatResponse
from backend.app.core.rag_engine import rag_engine
from backend.app.core.kb_manager import kb_manager
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class BuildRequest(BaseModel):
    model: str = "shibing624/text2vec-base-chinese"

class SyncRequest(BaseModel):
    model: Optional[str] = None

class ModelsRequest(BaseModel):
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

router = APIRouter()
_security = HTTPBearer(auto_error=False)


async def verify_bot_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> None:
    if settings.BOT_API_KEY:
        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        if credentials.credentials != settings.BOT_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")


# ── Knowledge Base ──

@router.post("/knowledge_base/build")
async def build_kb(req: BuildRequest):
    await asyncio.to_thread(kb_manager.rebuild, req.model)
    return {
        "status": "success",
        "message": "知识库构建完成",
        "embedding_model": kb_manager.embedding_model_name,
    }

@router.post("/knowledge_base/clean")
async def clean_kb():
    await asyncio.to_thread(kb_manager.clean)
    return {"status": "success", "message": "知识库已清理"}

@router.post("/knowledge_base/sync")
async def sync_kb(req: SyncRequest):
    await asyncio.to_thread(kb_manager.rebuild, req.model)
    return {
        "status": "success",
        "message": "知识库同步完成",
        "embedding_model": kb_manager.embedding_model_name,
    }

@router.get("/knowledge_base/status")
async def kb_status():
    runtime_status = kb_manager.get_status()
    return {
        "status": "ready" if runtime_status["ready"] else "empty",
        "version": runtime_status["version"],
        "embedding_model": runtime_status["embedding_model"],
    }

# ── Models ──

@router.post("/models")
async def list_models(req: ModelsRequest):
    models = await rag_engine.list_models(req.provider, req.base_url, req.api_key)
    return {"models": models}

# ── SSE Chat (frontend) ──

@router.post("/chat")
async def chat(request: ChatRequest):
    if not rag_engine.retriever:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    async def event_generator():
        start_time = time.time()
        step_durations = {}
        execution_mode = "stream"
        fallback_used = False
        fallback_reason = ""
        first_context_ms = None
        first_token_time = None
        total_chars = 0

        try:
            chain_build_start = time.time()
            result = await rag_engine.chat(
                message=request.message,
                model_choice=request.model_choice,
                api_key=request.api_key,
                base_url=request.base_url,
                model_name=request.model_name,
                answer_detail=request.answer_detail,
                session_id=None,
                streaming=True,
            )
            qa_chain = result["qa_chain"]
            chain_input = result["chain_input"]
            docs = result["docs"]
            search_terms = result["search_terms"]
            llm_config = result["llm_config"]
            system_prompt = result["system_prompt"]
            step_durations["链与Prompt构建"] = round((time.time() - chain_build_start) * 1000, 2)

            # trace_config
            yield f"data: {json.dumps({'type': 'trace_config', 'data': {'system_prompt': system_prompt, 'llm_config': llm_config}})}\n\n"

            # processing
            yield f"data: {json.dumps({'type': 'trace', 'data': {'status': 'processing', 'model_choice': request.model_choice, 'answer_detail': request.answer_detail}})}\n\n"

            # trace (retrieval already done by chat())
            retrieve_time = round(step_durations.get("检索", 0), 2)
            # Re-measure retrieval time: rag_engine.chat already did retrieval, but we can approximate
            step_durations["检索"] = round((time.time() - chain_build_start - step_durations["链与Prompt构建"]) * 1000, 2)
            if docs:
                first_context_ms = round((time.time() - start_time) * 1000, 2)

            context_total_chars = sum(len(d.page_content) for d in docs)

            trace_data = {
                "search_terms": search_terms,
                "retrieved_chunk_count": len(docs),
                "retrieve_time_ms": step_durations["检索"],
                "context_total_chars": context_total_chars,
                "first_context_ms": first_context_ms,
                "step_durations_ms": step_durations,
            }
            yield f"data: {json.dumps({'type': 'trace', 'data': trace_data})}\n\n"

            # sources
            sources = [
                {
                    "title": d.metadata.get("title"),
                    "section": d.metadata.get("section"),
                    "source_url": d.metadata.get("source_url"),
                    "content_length": len(d.page_content),
                    "content_preview": d.page_content,
                }
                for d in docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            # stream generation
            llm_call_start = time.time()

            try:
                async for chunk in qa_chain.astream(chain_input):
                    if first_token_time is None:
                        first_token_time = time.time()
                    if chunk:
                        total_chars += len(chunk)
                        yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                step_durations["LLM流式调用"] = round((time.time() - llm_call_start) * 1000, 2)
            except Exception as stream_error:
                execution_mode = "invoke_fallback"
                fallback_used = True
                fallback_reason = str(stream_error)
                step_durations["LLM流式调用"] = round((time.time() - llm_call_start) * 1000, 2)
                logger.warning(f"流式调用失败，降级到 invoke: {stream_error}")

                fallback_start = time.time()
                try:
                    content = await qa_chain.ainvoke(chain_input)
                    content = content if isinstance(content, str) else str(content)
                    total_chars = len(content)
                    if first_token_time is None:
                        first_token_time = time.time()
                    yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
                except Exception as invoke_error:
                    logger.error(f"invoke 也失败: {invoke_error}")
                    raise invoke_error
                step_durations["LLM兜底调用"] = round((time.time() - fallback_start) * 1000, 2)

            # perf trace
            total_time = (time.time() - start_time) * 1000
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            yield f"data: {json.dumps({'type': 'trace_perf', 'data': {'ttft_ms': round(ttft, 2), 'total_time_ms': round(total_time, 2), 'output_chars': total_chars, 'step_durations_ms': step_durations, 'execution_mode': execution_mode, 'fallback_used': fallback_used, 'fallback_reason': fallback_reason}})}\n\n"

        except Exception as e:
            logger.error(f"推理流异常: {str(e)}")
            yield f"data: {json.dumps({'type': 'content', 'data': f'❌ 推理失败: {str(e)}'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ── Bot Chat (JSON) ──

@router.post("/chat/bot", dependencies=[Depends(verify_bot_token)])
async def chat_bot(request: BotChatRequest):
    if not rag_engine.retriever:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")

    try:
        result = await rag_engine.chat(
            message=request.message,
            model_choice=request.model_choice,
            api_key=request.api_key,
            base_url=request.base_url,
            model_name=request.model_name,
            answer_detail=request.answer_detail,
            session_id=request.session_id,
            streaming=False,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        {
            "title": d.metadata.get("title"),
            "section": d.metadata.get("section"),
            "source_url": d.metadata.get("source_url"),
        }
        for d in result["docs"]
    ]

    return BotChatResponse(
        session_id=request.session_id,
        reply=result["content"],
        sources=sources,
    )

@router.get("/health")
async def health_check():
    return {"status": "ok", "retriever_ready": rag_engine.retriever is not None}
