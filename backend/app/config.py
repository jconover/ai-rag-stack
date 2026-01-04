"""Configuration management"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_default_model: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")
    
    # Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "devops_docs")
    
    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_db: int = int(os.getenv("REDIS_DB", 0))
    
    # RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", 5))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", 4096))

    # Embeddings - set to 'cuda' for GPU acceleration (5-10x faster)
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # CORS - comma-separated list of allowed origins
    # Use "*" only for development; in production, specify exact origins
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    @property
    def cors_origins_list(self) -> list:
        """Parse CORS_ORIGINS into a list of origins"""
        if not self.cors_origins:
            return ["http://localhost:3000"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    class Config:
        env_file = ".env"


settings = Settings()
