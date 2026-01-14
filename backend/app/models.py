"""Pydantic models for API requests and responses"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

# Maximum query length to prevent OOM in embedding/LLM
# Used by ChatRequest validator and main.py's validate_query_length function
MAX_QUERY_LENGTH = 8000


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    model: Optional[str] = Field(None, description="Ollama model to use")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    use_rag: Optional[bool] = Field(True, description="Whether to use RAG for context")

    @field_validator('message')
    @classmethod
    def validate_message_length(cls, v: str) -> str:
        """Validate message length to prevent resource exhaustion attacks."""
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f'Message too long. Maximum {MAX_QUERY_LENGTH} characters, got {len(v)}.')
        return v


class SourceDocument(BaseModel):
    """Source document with retrieval metadata"""
    source: str = Field(..., description="Document source path/name")
    source_type: str = Field(..., description="Type of documentation source")
    content_preview: str = Field(..., description="Preview of document content")
    similarity_score: Optional[float] = Field(None, description="Vector similarity score (0-1)")
    rerank_score: Optional[float] = Field(None, description="Reranker relevance score")
    rank: Optional[int] = Field(None, description="Final rank position")


class ValidationIssue(BaseModel):
    """A single validation issue detected in the response"""
    code: str = Field(..., description="Machine-readable issue code")
    message: str = Field(..., description="Human-readable description")
    severity: str = Field(..., description="Issue severity: info, warning, error")
    snippet: Optional[str] = Field(None, description="Relevant text snippet")


class OutputValidation(BaseModel):
    """Output validation and hallucination detection results"""
    is_valid: bool = Field(True, description="Whether response passed validation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    issues_count: int = Field(0, description="Number of issues detected")
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of validation issues")
    source_citation_count: int = Field(0, description="Number of source citations found")
    unsupported_claims_count: int = Field(0, description="Number of unsupported claims detected")
    hallucination_markers_found: List[str] = Field(default_factory=list, description="Detected hallucination markers")
    validation_time_ms: float = Field(0.0, description="Time taken for validation in ms")


class RetrievalMetrics(BaseModel):
    """Metrics about the retrieval process"""
    initial_candidates: int = Field(..., description="Number of initial vector search results")
    after_reranking: Optional[int] = Field(None, description="Number of results after reranking")
    reranker_used: bool = Field(False, description="Whether reranker was applied")
    reranker_model: Optional[str] = Field(None, description="Reranker model used")
    hybrid_search_used: bool = Field(False, description="Whether hybrid search (BM25 + vector) was used")
    hyde_used: bool = Field(False, description="Whether HyDE query expansion was applied")
    hyde_time_ms: Optional[float] = Field(None, description="Time for HyDE generation in ms")
    web_search_used: bool = Field(False, description="Whether web search fallback was triggered")
    web_search_reason: Optional[str] = Field(None, description="Reason web search was triggered")
    web_search_results: Optional[int] = Field(None, description="Number of web search results")
    web_search_time_ms: Optional[float] = Field(None, description="Time for web search in ms")
    avg_similarity_score: Optional[float] = Field(None, description="Average similarity score")
    avg_rerank_score: Optional[float] = Field(None, description="Average rerank score")
    retrieval_time_ms: Optional[float] = Field(None, description="Time for vector retrieval in ms")
    rerank_time_ms: Optional[float] = Field(None, description="Time for reranking in ms")
    embedding_cache_hit: Optional[bool] = Field(None, description="Whether query embedding was served from cache")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    model: str = Field(..., description="Model used for generation")
    context_used: bool = Field(..., description="Whether RAG context was used")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    session_id: str = Field(..., description="Session ID")
    retrieval_metrics: Optional[RetrievalMetrics] = Field(None, description="Retrieval performance metrics")
    output_validation: Optional[OutputValidation] = Field(None, description="Output validation and hallucination detection results")


class ComponentStatus(BaseModel):
    """Status of an individual component"""
    connected: bool = Field(..., description="Whether component is connected/available")
    model: Optional[str] = Field(None, description="Model name if applicable")
    error: Optional[str] = Field(None, description="Error message if not connected")


class RedisPoolStats(BaseModel):
    """Redis connection pool statistics (verbose mode - contains sensitive info)"""
    max_connections: int = Field(..., description="Maximum pool connections configured")
    current_connections: Optional[int] = Field(None, description="Connections currently in use")
    available_connections: Optional[int] = Field(None, description="Connections available in pool")
    host: Optional[str] = Field(None, description="Redis host")
    port: Optional[int] = Field(None, description="Redis port")
    db: Optional[int] = Field(None, description="Redis database number")


class RedisPoolStatsSafe(BaseModel):
    """Redis connection pool statistics (safe mode - no sensitive info)"""
    max_connections: int = Field(..., description="Maximum pool connections configured")
    current_connections: Optional[int] = Field(None, description="Connections currently in use")
    available_connections: Optional[int] = Field(None, description="Connections available in pool")


class PostgresPoolStats(BaseModel):
    """PostgreSQL connection pool statistics (verbose mode - contains sensitive info)"""
    host: Optional[str] = Field(None, description="PostgreSQL host")
    port: Optional[int] = Field(None, description="PostgreSQL port")
    database: Optional[str] = Field(None, description="Database name")
    pool_size: int = Field(..., description="Configured pool size")
    max_overflow: int = Field(..., description="Maximum overflow connections")
    checked_in: Optional[int] = Field(None, description="Connections available in pool")
    checked_out: Optional[int] = Field(None, description="Connections currently in use")
    overflow: Optional[int] = Field(None, description="Current overflow connections")
    pool_timeout: Optional[float] = Field(None, description="Pool connection timeout in seconds")


class PostgresPoolStatsSafe(BaseModel):
    """PostgreSQL connection pool statistics (safe mode - no sensitive info)"""
    pool_size: int = Field(..., description="Configured pool size")
    max_overflow: int = Field(..., description="Maximum overflow connections")
    checked_in: Optional[int] = Field(None, description="Connections available in pool")
    checked_out: Optional[int] = Field(None, description="Connections currently in use")
    overflow: Optional[int] = Field(None, description="Current overflow connections")


class CircuitBreakerStatus(BaseModel):
    """Status of a single circuit breaker"""
    name: str = Field(..., description="Circuit breaker name (ollama, qdrant, tavily)")
    state: str = Field(..., description="Current state: closed, open, half_open")
    total_calls: int = Field(0, description="Total number of calls through this breaker")
    successful_calls: int = Field(0, description="Number of successful calls")
    failed_calls: int = Field(0, description="Number of failed calls")
    rejected_calls: int = Field(0, description="Calls rejected due to open circuit")
    success_rate: Optional[float] = Field(None, description="Success rate as decimal (0-1)")
    time_until_retry: Optional[float] = Field(None, description="Seconds until circuit may close (when open)")
    failure_threshold: int = Field(..., description="Failures needed to open circuit")
    reset_timeout_seconds: float = Field(..., description="Time before testing recovery")


class CircuitBreakersStatus(BaseModel):
    """Status of all circuit breakers"""
    healthy: bool = Field(..., description="True if all breakers are closed or half-open")
    ollama: CircuitBreakerStatus = Field(..., description="Ollama LLM circuit breaker")
    qdrant: CircuitBreakerStatus = Field(..., description="Qdrant vector DB circuit breaker")
    tavily: CircuitBreakerStatus = Field(..., description="Tavily web search circuit breaker")


class DeviceInfo(BaseModel):
    """ML device configuration and availability"""
    embedding_device: str = Field(..., description="Actual device used for embeddings (cuda/mps/cpu)")
    reranker_device: Optional[str] = Field(None, description="Actual device used for reranker (cuda/mps/cpu)")
    cuda_available: bool = Field(False, description="Whether NVIDIA CUDA is available")
    cuda_device_name: Optional[str] = Field(None, description="NVIDIA GPU name if available")
    mps_available: bool = Field(False, description="Whether Apple Silicon MPS is available")
    pytorch_available: bool = Field(True, description="Whether PyTorch is installed")


class HealthResponse(BaseModel):
    """Health response with full details (verbose mode - contains sensitive info)"""
    status: str
    ollama_connected: bool
    qdrant_connected: bool
    redis_connected: bool
    postgres_connected: bool = Field(False, description="Whether PostgreSQL is connected")
    reranker_enabled: bool = Field(False, description="Whether reranker is enabled")
    reranker_loaded: bool = Field(False, description="Whether reranker model is loaded")
    reranker_model: Optional[str] = Field(None, description="Reranker model name")
    device_info: Optional[DeviceInfo] = Field(None, description="ML device configuration and GPU availability")
    redis_pool: Optional[RedisPoolStats] = Field(None, description="Redis connection pool statistics")
    postgres_pool: Optional[PostgresPoolStats] = Field(None, description="PostgreSQL connection pool statistics")
    circuit_breakers: Optional[CircuitBreakersStatus] = Field(None, description="Circuit breaker status for resilience")
    components: Optional[Dict[str, ComponentStatus]] = Field(None, description="Detailed component status")


class ServiceStatus(BaseModel):
    """Safe service status without internal details"""
    status: str = Field(..., description="Service status: connected, disconnected, degraded")
    models_available: Optional[int] = Field(None, description="Number of models available (for Ollama)")
    points_count: Optional[int] = Field(None, description="Number of indexed vectors (for Qdrant)")
    pool_utilization: Optional[float] = Field(None, description="Connection pool utilization percentage")


class HealthResponseSafe(BaseModel):
    """Health response without sensitive internal details (safe for production)"""
    status: str = Field(..., description="Overall health status: healthy, degraded, unhealthy")
    services: Dict[str, ServiceStatus] = Field(..., description="Individual service status")
    reranker: Optional[Dict[str, Any]] = Field(None, description="Reranker status (enabled/loaded)")
    device_info: Optional[DeviceInfo] = Field(None, description="ML device info (no sensitive data)")
    circuit_breakers: Optional[Dict[str, str]] = Field(None, description="Circuit breaker states (state only)")


class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class StatsResponse(BaseModel):
    collection_name: str
    vectors_count: int
    indexed_documents: int


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to documents to ingest")
    source_name: str = Field(..., description="Name/label for this document source")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int


class FeedbackRequest(BaseModel):
    """User feedback on a response"""
    session_id: str = Field(..., description="Session ID for the conversation")
    message_index: Optional[int] = Field(None, description="Index of the response in session (0-based)")
    helpful: bool = Field(..., description="Whether the response was helpful (True=thumbs up, False=thumbs down)")
    query_hash: Optional[str] = Field(None, description="Query hash for correlation with retrieval metrics")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback"""
    status: str = Field(..., description="Status of feedback submission")
    feedback_id: str = Field(..., description="Unique ID for this feedback")
    timestamp: str = Field(..., description="ISO timestamp when feedback was recorded")


# Analytics models for query logs endpoint

class QueryLogEntry(BaseModel):
    """Single query log entry for analytics"""
    id: str = Field(..., description="Unique query identifier")
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="User query text")
    model: str = Field(..., description="LLM model used")
    timestamp: str = Field(..., description="Query timestamp (ISO format)")
    latency_ms: Optional[float] = Field(None, description="Total latency in milliseconds")
    token_count: Optional[int] = Field(None, description="Tokens generated")
    response_length: Optional[int] = Field(None, description="Response length in characters")
    context_used: bool = Field(True, description="Whether RAG context was used")
    sources_count: Optional[int] = Field(None, description="Number of sources retrieved")
    sources_returned: Optional[List[str]] = Field(None, description="Source document paths")
    retrieval_scores: Optional[Dict[str, Any]] = Field(None, description="Retrieval score metrics")
    hybrid_search_used: bool = Field(False, description="Whether hybrid search was used")
    hyde_used: bool = Field(False, description="Whether HyDE was used")
    reranker_used: bool = Field(False, description="Whether reranker was used")
    web_search_used: bool = Field(False, description="Whether web search was triggered")


class QueryLogsResponse(BaseModel):
    """Response for query logs analytics endpoint"""
    total: int = Field(..., description="Total number of matching records")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of records per page")
    total_pages: int = Field(..., description="Total number of pages")
    queries: List[QueryLogEntry] = Field(..., description="Query log entries")


class QueryAnalyticsSummary(BaseModel):
    """Summary statistics for query analytics"""
    total_queries: int = Field(..., description="Total queries in time range")
    unique_sessions: int = Field(..., description="Unique session count")
    avg_latency_ms: Optional[float] = Field(None, description="Average latency in ms")
    p50_latency_ms: Optional[float] = Field(None, description="50th percentile latency")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile latency")
    p99_latency_ms: Optional[float] = Field(None, description="99th percentile latency")
    model_distribution: Dict[str, int] = Field(default_factory=dict, description="Query count by model")
    feature_usage: Dict[str, int] = Field(default_factory=dict, description="Feature usage counts")
    queries_per_day: Dict[str, int] = Field(default_factory=dict, description="Query count by date")


class QueryLogsFilter(BaseModel):
    """Filter parameters for query logs"""
    start_date: Optional[str] = Field(None, description="Start date (ISO format or YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (ISO format or YYYY-MM-DD)")
    model: Optional[str] = Field(None, description="Filter by model name")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    min_latency_ms: Optional[float] = Field(None, description="Minimum latency threshold")
    max_latency_ms: Optional[float] = Field(None, description="Maximum latency threshold")
    hybrid_search_used: Optional[bool] = Field(None, description="Filter by hybrid search usage")
    web_search_used: Optional[bool] = Field(None, description="Filter by web search usage")
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=500, description="Records per page")


# =====================
# A/B Testing Models
# =====================

class ExperimentVariant(BaseModel):
    """Configuration for an experiment variant"""
    name: str = Field(..., description="Variant name (e.g., 'control', 'treatment_a')")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Traffic weight for this variant (0.0-1.0)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Variant-specific configuration (model, temperature, etc.)")


class ExperimentCreate(BaseModel):
    """Request to create a new A/B experiment"""
    name: str = Field(..., min_length=1, max_length=200, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description and hypothesis")
    experiment_type: str = Field(
        "model",
        description="Type of experiment: 'model' (LLM model), 'config' (RAG config), 'prompt' (prompt template)"
    )
    variants: List[ExperimentVariant] = Field(
        ...,
        min_length=2,
        description="List of variants (minimum 2: control and at least one treatment)"
    )
    traffic_percentage: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of total traffic to include in experiment (0-100)"
    )
    start_at: Optional[str] = Field(None, description="Scheduled start time (ISO format)")
    end_at: Optional[str] = Field(None, description="Scheduled end time (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExperimentUpdate(BaseModel):
    """Request to update an existing experiment"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Updated experiment name")
    description: Optional[str] = Field(None, description="Updated description")
    status: Optional[str] = Field(
        None,
        description="New status: 'draft', 'running', 'paused', 'completed', 'archived'"
    )
    variants: Optional[List[ExperimentVariant]] = Field(
        None,
        description="Updated variants (cannot change while running)"
    )
    traffic_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Updated traffic percentage"
    )
    end_at: Optional[str] = Field(None, description="Updated end time (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class ExperimentResponse(BaseModel):
    """Response with experiment details"""
    id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    experiment_type: str = Field(..., description="Type of experiment")
    status: str = Field(..., description="Current status: draft, running, paused, completed, archived")
    variants: List[ExperimentVariant] = Field(..., description="Experiment variants")
    traffic_percentage: float = Field(..., description="Traffic percentage in experiment")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")
    start_at: Optional[str] = Field(None, description="Scheduled start time")
    end_at: Optional[str] = Field(None, description="Scheduled end time")
    started_at: Optional[str] = Field(None, description="Actual start time")
    ended_at: Optional[str] = Field(None, description="Actual end time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExperimentListResponse(BaseModel):
    """Response for listing experiments"""
    total: int = Field(..., description="Total number of experiments")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Records per page")
    experiments: List[ExperimentResponse] = Field(..., description="List of experiments")


class VariantStats(BaseModel):
    """Statistics for a single variant"""
    variant_name: str = Field(..., description="Variant name")
    sample_size: int = Field(..., description="Number of observations")
    conversions: int = Field(0, description="Number of conversions (positive outcomes)")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate (conversions/sample_size)")
    avg_latency_ms: Optional[float] = Field(None, description="Average latency in milliseconds")
    p50_latency_ms: Optional[float] = Field(None, description="50th percentile latency")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile latency")
    avg_metric_value: Optional[float] = Field(None, description="Average primary metric value")
    std_metric_value: Optional[float] = Field(None, description="Standard deviation of metric")
    positive_feedback: int = Field(0, description="Thumbs up count")
    negative_feedback: int = Field(0, description="Thumbs down count")
    feedback_rate: Optional[float] = Field(None, description="Positive feedback rate")


class StatisticalSignificance(BaseModel):
    """Statistical significance testing results"""
    control_variant: str = Field(..., description="Name of control variant")
    treatment_variant: str = Field(..., description="Name of treatment variant")
    metric_name: str = Field(..., description="Metric being compared")
    control_mean: float = Field(..., description="Control group mean")
    treatment_mean: float = Field(..., description="Treatment group mean")
    relative_difference: float = Field(..., description="Relative difference ((treatment-control)/control)")
    p_value: Optional[float] = Field(None, description="P-value from statistical test")
    confidence_interval_lower: Optional[float] = Field(None, description="95% CI lower bound")
    confidence_interval_upper: Optional[float] = Field(None, description="95% CI upper bound")
    is_significant: bool = Field(False, description="Whether result is statistically significant (p < 0.05)")
    test_type: str = Field("t-test", description="Statistical test used")
    sample_size_adequate: bool = Field(False, description="Whether sample size is adequate for reliable results")
    minimum_detectable_effect: Optional[float] = Field(None, description="MDE with current sample size")


class ExperimentStatsResponse(BaseModel):
    """Response with experiment statistics and significance"""
    experiment_id: str = Field(..., description="Experiment identifier")
    experiment_name: str = Field(..., description="Experiment name")
    status: str = Field(..., description="Current experiment status")
    duration_hours: Optional[float] = Field(None, description="Experiment duration in hours")
    total_observations: int = Field(..., description="Total observations across all variants")
    variant_stats: List[VariantStats] = Field(..., description="Per-variant statistics")
    significance_tests: List[StatisticalSignificance] = Field(
        default_factory=list,
        description="Statistical significance tests between variants"
    )
    winning_variant: Optional[str] = Field(None, description="Variant with best performance (if significant)")
    recommendation: Optional[str] = Field(None, description="Action recommendation based on results")
    computed_at: str = Field(..., description="When statistics were computed (ISO format)")


class VariantAssignmentResponse(BaseModel):
    """Response with variant assignment for a session"""
    experiment_id: str = Field(..., description="Experiment identifier")
    experiment_name: str = Field(..., description="Experiment name")
    variant_name: str = Field(..., description="Assigned variant name")
    variant_config: Dict[str, Any] = Field(..., description="Configuration to apply for this variant")
    session_id: str = Field(..., description="Session identifier used for assignment")
    is_control: bool = Field(False, description="Whether this is the control variant")


class ExperimentResultRecord(BaseModel):
    """Request to record a result/metric for an experiment"""
    session_id: str = Field(..., description="Session identifier")
    variant_name: str = Field(..., description="Variant the session was assigned to")
    metric_name: str = Field("latency_ms", description="Name of the metric being recorded")
    metric_value: float = Field(..., description="Metric value")
    is_conversion: bool = Field(False, description="Whether this is a conversion event")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional result metadata")


class ExperimentResultResponse(BaseModel):
    """Response after recording an experiment result"""
    id: str = Field(..., description="Result record identifier")
    experiment_id: str = Field(..., description="Experiment identifier")
    variant_name: str = Field(..., description="Variant name")
    recorded_at: str = Field(..., description="Recording timestamp (ISO format)")


# =====================
# Authentication Models
# =====================

class UserCreate(BaseModel):
    """Request to create a new user account"""
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=8, max_length=128, description="Password (minimum 8 characters)")


class UserLogin(BaseModel):
    """Request to authenticate a user"""
    email_or_username: str = Field(..., description="Email address or username")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    """User information response"""
    id: str = Field(..., description="User unique identifier")
    email: str = Field(..., description="User email address")
    username: str = Field(..., description="Username")
    is_active: bool = Field(True, description="Whether user account is active")
    created_at: str = Field(..., description="Account creation timestamp (ISO format)")


class UserUpdate(BaseModel):
    """Request to update user profile"""
    email: Optional[str] = Field(None, description="New email address")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="New username")
    password: Optional[str] = Field(None, min_length=8, max_length=128, description="New password")


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_at: str = Field(..., description="Token expiration timestamp (ISO format)")


class APIKeyCreate(BaseModel):
    """Request to create a new API key"""
    name: str = Field(..., min_length=1, max_length=100, description="API key name/description")
    permissions: List[str] = Field(
        default_factory=lambda: ["read", "chat"],
        description="List of permissions: read, chat, write, admin"
    )
    expires_at: Optional[str] = Field(None, description="Optional expiration timestamp (ISO format)")


class APIKeyResponse(BaseModel):
    """API key information response"""
    id: str = Field(..., description="API key unique identifier")
    name: str = Field(..., description="API key name/description")
    permissions: List[str] = Field(..., description="List of granted permissions")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp (ISO format)")
    last_used_at: Optional[str] = Field(None, description="Last usage timestamp (ISO format)")
    key: Optional[str] = Field(None, description="API key value (only returned on creation)")
    key_prefix: Optional[str] = Field(None, description="First 8 characters of API key for identification")


# =====================
# Kubernetes Readiness Probe Models
# =====================

class ReadinessCheck(BaseModel):
    """Individual readiness check result"""
    ready: bool = Field(..., description="Whether this component is ready to serve traffic")
    message: str = Field(..., description="Human-readable status message")
    latency_ms: float = Field(..., description="Time taken to perform this check in milliseconds")


class ReadinessResponse(BaseModel):
    """Deep readiness probe response for Kubernetes deployments"""
    ready: bool = Field(..., description="Overall readiness - true only if all critical checks pass")
    checks: Dict[str, ReadinessCheck] = Field(..., description="Individual component check results")


class LivenessResponse(BaseModel):
    """Simple liveness probe response"""
    alive: bool = Field(True, description="Whether the application process is running")
    timestamp: str = Field(..., description="Current server timestamp (ISO format)")


# =====================
# Prompt Template Models
# =====================

class TemplateVariable(BaseModel):
    """Definition of a variable in a prompt template"""
    name: str = Field(..., description="Variable name used in the template (e.g., 'pod_name')")
    type: str = Field("string", description="Variable type: string, number, boolean, select")
    description: Optional[str] = Field(None, description="Human-readable description of what this variable represents")
    required: bool = Field(False, description="Whether this variable must be provided")
    default: Optional[Any] = Field(None, description="Default value if not provided")
    options: Optional[List[str]] = Field(None, description="Allowed values for 'select' type variables")
    placeholder: Optional[str] = Field(None, description="Placeholder text to show in UI")


class PromptTemplate(BaseModel):
    """Prompt template with variable support"""
    id: str = Field(..., description="Unique template identifier")
    category: str = Field(..., description="Template category (e.g., 'Kubernetes', 'Docker')")
    title: str = Field(..., description="Human-readable template title")
    description: str = Field(..., description="Description of what this template does")
    prompt: str = Field(..., description="Template prompt with {variable} placeholders")
    variables: List[TemplateVariable] = Field(
        default_factory=list,
        description="List of variables that can be customized in this template"
    )


class RenderTemplateRequest(BaseModel):
    """Request to render a template with variable substitutions"""
    template_id: str = Field(..., description="ID of the template to render")
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variable name to value mapping for substitution"
    )


class RenderTemplateResponse(BaseModel):
    """Response with rendered template prompt"""
    template_id: str = Field(..., description="ID of the rendered template")
    original_prompt: str = Field(..., description="Original template prompt with placeholders")
    rendered_prompt: str = Field(..., description="Rendered prompt with variables substituted")
    variables_used: Dict[str, Any] = Field(..., description="Variables that were substituted")
    missing_required: List[str] = Field(
        default_factory=list,
        description="Required variables that were not provided (rendered with defaults/placeholders)"
    )


class TemplatesResponse(BaseModel):
    """Response for listing templates"""
    templates: List[PromptTemplate] = Field(..., description="List of prompt templates")
    categories: List[str] = Field(..., description="All available categories")


# =====================
# Real-time Analytics Models
# =====================

class LatencyMetrics(BaseModel):
    """Latency statistics over a time window"""
    avg_ms: Optional[float] = Field(None, description="Average latency in milliseconds")
    median_ms: Optional[float] = Field(None, description="Median (p50) latency in milliseconds")
    p95_ms: Optional[float] = Field(None, description="95th percentile latency in milliseconds")
    p99_ms: Optional[float] = Field(None, description="99th percentile latency in milliseconds")
    min_ms: Optional[float] = Field(None, description="Minimum latency in milliseconds")
    max_ms: Optional[float] = Field(None, description="Maximum latency in milliseconds")


class RequestMetrics(BaseModel):
    """Request rate and error metrics"""
    requests_per_minute: float = Field(..., description="Request rate per minute")
    total_requests_in_window: int = Field(..., description="Total requests in measurement window")
    error_rate_percent: float = Field(..., description="Error rate as percentage")


class CacheMetrics(BaseModel):
    """Embedding cache performance metrics"""
    hit_rate_percent: float = Field(..., description="Cache hit rate as percentage")
    hits_in_window: int = Field(..., description="Cache hits in measurement window")
    misses_in_window: int = Field(..., description="Cache misses in measurement window")


class RetrievalQualityMetrics(BaseModel):
    """Retrieval quality score statistics"""
    avg_score: Optional[float] = Field(None, description="Average similarity score")
    median_score: Optional[float] = Field(None, description="Median similarity score")
    min_score: Optional[float] = Field(None, description="Minimum similarity score")
    max_score: Optional[float] = Field(None, description="Maximum similarity score")


class ModelUsageStats(BaseModel):
    """Usage statistics for a single model"""
    count: int = Field(..., description="Number of requests using this model")
    percentage: float = Field(..., description="Percentage of total requests")


class TopQueryEntry(BaseModel):
    """Anonymized top query entry"""
    query_hash: str = Field(..., description="SHA256 hash of query (first 12 chars)")
    count: int = Field(..., description="Number of times this query was made")
    last_seen: str = Field(..., description="Last time this query was seen (ISO format)")


class RealtimeAnalyticsResponse(BaseModel):
    """Real-time analytics snapshot for operational visibility"""
    timestamp: str = Field(..., description="Snapshot timestamp (ISO format)")
    window_seconds: int = Field(..., description="Measurement window in seconds")
    request_metrics: RequestMetrics = Field(..., description="Request rate and error metrics")
    latency_metrics: LatencyMetrics = Field(..., description="Response latency statistics")
    cache_metrics: CacheMetrics = Field(..., description="Embedding cache performance")
    retrieval_quality: RetrievalQualityMetrics = Field(..., description="Retrieval similarity score metrics")
    model_usage: Dict[str, ModelUsageStats] = Field(default_factory=dict, description="Model usage distribution")
    top_queries: List[TopQueryEntry] = Field(default_factory=list, description="Top queries (anonymized, last hour)")


# =====================
# Model Drift Detection Models
# =====================

class DriftMetricsResponse(BaseModel):
    """Statistical metrics for a score distribution window"""
    mean_score: float = Field(..., description="Mean similarity score")
    std_score: float = Field(..., description="Standard deviation of scores")
    p25: float = Field(..., description="25th percentile score")
    p50: float = Field(..., description="50th percentile (median) score")
    p75: float = Field(..., description="75th percentile score")
    min_score: float = Field(..., description="Minimum score in window")
    max_score: float = Field(..., description="Maximum score in window")
    sample_count: int = Field(..., description="Number of samples in window")
    timestamp: str = Field(..., description="When metrics were computed (ISO format)")


class DriftCheckResponse(BaseModel):
    """Response from drift detection check"""
    status: str = Field(
        ...,
        description="Drift status: stable, drift_detected, warning, insufficient_data, no_baseline, error"
    )
    message: str = Field(..., description="Human-readable status message")
    checked_at: str = Field(..., description="When check was performed (ISO format)")
    mean_shift_pct: Optional[float] = Field(None, description="Percentage shift in mean score from baseline")
    std_shift_pct: Optional[float] = Field(None, description="Percentage shift in standard deviation")
    median_shift_pct: Optional[float] = Field(None, description="Percentage shift in median score")
    current: Optional[DriftMetricsResponse] = Field(None, description="Current window metrics")
    baseline: Optional[DriftMetricsResponse] = Field(None, description="Baseline metrics for comparison")


class DriftStatusResponse(BaseModel):
    """Drift detector configuration and status"""
    enabled: bool = Field(True, description="Whether drift detection is enabled")
    window_hours: int = Field(..., description="Time window for collecting scores (hours)")
    drift_threshold_pct: float = Field(..., description="Threshold to trigger drift alert (percentage)")
    warning_threshold_pct: float = Field(..., description="Threshold to trigger warning (percentage)")
    min_samples_required: int = Field(..., description="Minimum samples needed for valid metrics")
    current_sample_count: int = Field(..., description="Current number of samples in buffer")
    buffer_size: int = Field(..., description="In-memory buffer size")
    redis_connected: bool = Field(..., description="Whether Redis is available for storage")


class DriftHistoryEntry(BaseModel):
    """Daily drift metrics history entry"""
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    mean_score: float = Field(..., description="Average mean score for the day")
    std_score: float = Field(..., description="Average standard deviation for the day")
    total_samples: int = Field(..., description="Total samples collected that day")
    measurement_count: int = Field(..., description="Number of metric measurements that day")


class DriftHistoryResponse(BaseModel):
    """Response with drift metrics history"""
    history: List[DriftHistoryEntry] = Field(..., description="Daily metrics history")
    days_requested: int = Field(..., description="Number of days requested")
    days_returned: int = Field(..., description="Number of days with data")


class SetBaselineResponse(BaseModel):
    """Response after setting drift baseline"""
    success: bool = Field(..., description="Whether baseline was set successfully")
    message: str = Field(..., description="Status message")
    baseline: Optional[DriftMetricsResponse] = Field(None, description="The baseline metrics that were set")


# Documentation Freshness Models

class SourceFreshnessModel(BaseModel):
    """Freshness status for a documentation source"""
    source_type: str = Field(..., description="Documentation source type (e.g., kubernetes, terraform)")
    last_download: Optional[str] = Field(None, description="ISO timestamp of last download")
    last_commit_date: Optional[str] = Field(None, description="ISO timestamp of last git commit")
    days_since_update: int = Field(..., description="Days since last update (-1 if unknown)")
    staleness_risk: str = Field(..., description="Risk level: fresh, low, medium, high, or unknown")
    recommended_action: str = Field(..., description="Recommended action to take")


class FreshnessReportResponse(BaseModel):
    """Response containing documentation freshness report"""
    sources: List[SourceFreshnessModel] = Field(..., description="Freshness status for all tracked sources")
    total_sources: int = Field(..., description="Total number of tracked sources")
    stale_count: int = Field(..., description="Number of sources with medium or higher staleness")
    fresh_count: int = Field(..., description="Number of fresh sources")
    generated_at: str = Field(..., description="ISO timestamp when report was generated")
