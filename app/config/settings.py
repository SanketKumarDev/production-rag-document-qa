from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AskMyTechDocs"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
