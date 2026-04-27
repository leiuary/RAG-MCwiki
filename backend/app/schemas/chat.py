from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的问题")
    model_choice: str = Field("local", description="选择的模型：'local' 或 'deepseek'")
    api_key: Optional[str] = Field(None, description="DeepSeek API Key")
    answer_detail: str = Field("标准", description="回答详细度：'简洁'、'标准'、'详细'")

class SourceDocument(BaseModel):
    title: str
    source_url: str
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
