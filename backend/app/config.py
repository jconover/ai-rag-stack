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

    # Redis Connection Pool
    redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", 50))
    redis_socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", 5.0))
    redis_socket_connect_timeout: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 5.0))
    
    # RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", 5))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", 4096))

    # Embeddings - device for running embedding model
    # Options: "auto" (recommended), "cuda" (NVIDIA GPU), "mps" (Apple Silicon), "cpu"
    # "auto" will detect and use the best available GPU, falling back to CPU
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    # Embedding model - BAAI/bge-base-en-v1.5 offers +10-15% retrieval quality over all-MiniLM-L6-v2
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    # Embedding dimension - must match the model (bge-base-en-v1.5: 768, all-MiniLM-L6-v2: 384)
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", 768))

    # Embedding Cache - cache embeddings to reduce computation
    embedding_cache_enabled: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    embedding_cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", 3600))  # TTL in seconds (default: 1 hour)

    # Reranker - Cross-encoder for improved retrieval quality
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Reranker device - Options: "auto" (recommended), "cuda", "mps", "cpu"
    reranker_device: str = os.getenv("RERANKER_DEVICE", "auto")
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

    # Conversation Context - use conversation history to improve retrieval for follow-up questions
    conversation_context_enabled: bool = os.getenv("CONVERSATION_CONTEXT_ENABLED", "true").lower() == "true"
    conversation_context_history_limit: int = int(os.getenv("CONVERSATION_CONTEXT_HISTORY_LIMIT", "3"))
    conversation_context_min_query_length: int = int(os.getenv("CONVERSATION_CONTEXT_MIN_QUERY_LENGTH", "5"))
    conversation_context_max_terms: int = int(os.getenv("CONVERSATION_CONTEXT_MAX_TERMS", "10"))

    # Few-shot learning - include domain-specific examples to improve output consistency and formatting
    few_shot_enabled: bool = os.getenv("FEW_SHOT_ENABLED", "true").lower() == "true"

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

    # Output validation and hallucination detection
    output_validation_enabled: bool = os.getenv("OUTPUT_VALIDATION_ENABLED", "true").lower() == "true"
    output_validation_min_confidence: float = float(os.getenv("OUTPUT_VALIDATION_MIN_CONFIDENCE", "0.5"))
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # CORS - comma-separated list of allowed origins
    # Use "*" only for development; in production, specify exact origins
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    # PostgreSQL Database
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", 5432))
    postgres_user: str = os.getenv("POSTGRES_USER", "devops_assistant")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "devops_password")
    postgres_db: str = os.getenv("POSTGRES_DB", "devops_assistant")
    postgres_pool_size: int = int(os.getenv("POSTGRES_POOL_SIZE", 10))
    postgres_max_overflow: int = int(os.getenv("POSTGRES_MAX_OVERFLOW", 20))
    postgres_pool_timeout: int = int(os.getenv("POSTGRES_POOL_TIMEOUT", 30))
    postgres_pool_recycle: int = int(os.getenv("POSTGRES_POOL_RECYCLE", 3600))  # Recycle connections after 1 hour
    postgres_echo_sql: bool = os.getenv("POSTGRES_ECHO_SQL", "false").lower() == "true"

    # Query logging - enable/disable PostgreSQL query logging
    query_logging_enabled: bool = os.getenv("QUERY_LOGGING_ENABLED", "true").lower() == "true"

    # A/B Testing Configuration
    ab_testing_enabled: bool = os.getenv("AB_TESTING_ENABLED", "true").lower() == "true"
    ab_testing_auto_record_metrics: bool = os.getenv("AB_TESTING_AUTO_RECORD_METRICS", "true").lower() == "true"
    ab_testing_default_experiment: str | None = os.getenv("AB_TESTING_DEFAULT_EXPERIMENT") or None

    # Authentication Configuration
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    session_expire_hours: int = int(os.getenv("SESSION_EXPIRE_HOURS", 24))
    api_key_prefix: str = os.getenv("API_KEY_PREFIX", "rag_")
    require_email_verification: bool = os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"

    # OpenTelemetry Distributed Tracing Configuration
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    # Exporter type: "otlp" for OTLP/gRPC (Jaeger, etc.), "console" for stdout
    tracing_exporter: str = os.getenv("TRACING_EXPORTER", "console")
    # OTLP endpoint for sending traces (default: Jaeger OTLP gRPC port)
    tracing_otlp_endpoint: str = os.getenv("TRACING_OTLP_ENDPOINT", "http://localhost:4317")
    # Service name for trace identification
    tracing_service_name: str = os.getenv("TRACING_SERVICE_NAME", "devops-ai-assistant")
    # Sampling ratio (0.0 to 1.0) - 1.0 means trace all requests
    tracing_sample_rate: float = float(os.getenv("TRACING_SAMPLE_RATE", "1.0"))

    # Real-time Analytics Configuration
    analytics_enabled: bool = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
    analytics_short_window_seconds: int = int(os.getenv("ANALYTICS_SHORT_WINDOW_SECONDS", 300))  # 5 minutes
    analytics_long_window_seconds: int = int(os.getenv("ANALYTICS_LONG_WINDOW_SECONDS", 3600))  # 1 hour
    analytics_endpoint_protected: bool = os.getenv("ANALYTICS_ENDPOINT_PROTECTED", "false").lower() == "true"
    analytics_api_key: str = os.getenv("ANALYTICS_API_KEY", "")

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL async connection URL for asyncpg driver"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list:
        """Parse CORS_ORIGINS into a list of origins"""
        if not self.cors_origins:
            return ["http://localhost:3000"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    class Config:
        env_file = ".env"


settings = Settings()
