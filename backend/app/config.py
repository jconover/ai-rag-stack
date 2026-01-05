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

    # Reranker - Cross-encoder for improved retrieval quality
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", 5))  # Final results after reranking
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 20))  # Initial retrieval before reranking

    # Score thresholds for filtering low-quality results
    min_similarity_score: float = float(os.getenv("MIN_SIMILARITY_SCORE", 0.3))
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", 0.01))

    # Hybrid Search - BM25 (sparse) + Vector (dense) with RRF fusion
    hybrid_search_enabled: bool = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
    hybrid_search_alpha: float = float(os.getenv("HYBRID_SEARCH_ALPHA", 0.5))  # Weight for dense vs sparse (0=sparse only, 1=dense only)
    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", 60))  # RRF constant (higher = more emphasis on top ranks)
    sparse_encoder_model: str = os.getenv("SPARSE_ENCODER_MODEL", "Qdrant/bm25")

    # HyDE (Hypothetical Document Embeddings) query expansion
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "false").lower() == "true"
    hyde_model: str = os.getenv("HYDE_MODEL", "llama3.1:8b")
    hyde_temperature: float = float(os.getenv("HYDE_TEMPERATURE", "0.3"))
    hyde_max_tokens: int = int(os.getenv("HYDE_MAX_TOKENS", "256"))
    hyde_min_query_length: int = int(os.getenv("HYDE_MIN_QUERY_LENGTH", "10"))
    hyde_max_query_length: int = int(os.getenv("HYDE_MAX_QUERY_LENGTH", "500"))
    hyde_timeout_seconds: float = float(os.getenv("HYDE_TIMEOUT_SECONDS", "10.0"))

    # Web Search Fallback (Tavily) - triggers when local retrieval scores are low
    web_search_enabled: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
    web_search_api_key: str = os.getenv("TAVILY_API_KEY", "")
    web_search_min_score_threshold: float = float(os.getenv("WEB_SEARCH_MIN_SCORE_THRESHOLD", "0.4"))
    web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 5))
    web_search_timeout_seconds: float = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10.0"))
    web_search_include_domains: str = os.getenv("WEB_SEARCH_INCLUDE_DOMAINS", "")  # Comma-separated
    web_search_exclude_domains: str = os.getenv("WEB_SEARCH_EXCLUDE_DOMAINS", "")  # Comma-separated

    # Metrics and logging
    enable_retrieval_metrics: bool = os.getenv("ENABLE_RETRIEVAL_METRICS", "true").lower() == "true"
    log_retrieval_details: bool = os.getenv("LOG_RETRIEVAL_DETAILS", "false").lower() == "true"
    
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
