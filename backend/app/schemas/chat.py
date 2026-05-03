from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的问题")
    model_choice: str = Field("api", description="选择的模型供应商：'local' 或 'api'")
    base_url: Optional[str] = Field(None, description="模型服务的基础 URL")
    model_name: Optional[str] = Field(None, description="具体的模型名称")
    api_key: Optional[str] = Field(None, description="API Key")
    answer_detail: str = Field("标准", description="回答详细度：'简洁'、'标准'、'详细'")
    use_bm25: Optional[bool] = Field(None, description="是否启用 BM25 混合检索（None=使用全局默认值）")

class BotChatRequest(ChatRequest):
    """QQ Bot 请求，继承 ChatRequest 增加会话标识"""
    session_id: str = Field(..., description="QQ 用户 ID 作为会话标识")

class BotChatResponse(BaseModel):
    """Bot 端点的 JSON 响应"""
    session_id: str
    reply: str
    sources: list[dict] = []
