import os

class Settings:
    PROJECT_NAME: str = "RAG-MCwiki"
    
    # 路径配置
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "..", "data", "structured_output")
    PERSIST_DIR: str = os.path.join(BASE_DIR, "..", "data", "chroma_db")
    
    # 模型配置
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    LOCAL_LLM_URL: str = "http://localhost:1234/v1"
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"
    
    # 检索配置
    RETRIEVAL_K: int = 3

settings = Settings()
