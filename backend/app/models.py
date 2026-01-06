"""Pydantic models for API requests and responses"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    model: Optional[str] = Field(None, description="Ollama model to use")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    use_rag: Optional[bool] = Field(True, description="Whether to use RAG for context")


class SourceDocument(BaseModel):
    """Source document with retrieval metadata"""
    source: str = Field(..., description="Document source path/name")
    source_type: str = Field(..., description="Type of documentation source")
    content_preview: str = Field(..., description="Preview of document content")
    similarity_score: Optional[float] = Field(None, description="Vector similarity score (0-1)")
    rerank_score: Optional[float] = Field(None, description="Reranker relevance score")
    rank: Optional[int] = Field(None, description="Final rank position")


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


class ComponentStatus(BaseModel):
    """Status of an individual component"""
    connected: bool = Field(..., description="Whether component is connected/available")
    model: Optional[str] = Field(None, description="Model name if applicable")
    error: Optional[str] = Field(None, description="Error message if not connected")


class RedisPoolStats(BaseModel):
    """Redis connection pool statistics"""
    max_connections: int = Field(..., description="Maximum pool connections configured")
    current_connections: Optional[int] = Field(None, description="Connections currently in use")
    available_connections: Optional[int] = Field(None, description="Connections available in pool")
    host: Optional[str] = Field(None, description="Redis host")
    port: Optional[int] = Field(None, description="Redis port")
    db: Optional[int] = Field(None, description="Redis database number")


class PostgresPoolStats(BaseModel):
    """PostgreSQL connection pool statistics"""
    host: Optional[str] = Field(None, description="PostgreSQL host")
    port: Optional[int] = Field(None, description="PostgreSQL port")
    database: Optional[str] = Field(None, description="Database name")
    pool_size: int = Field(..., description="Configured pool size")
    max_overflow: int = Field(..., description="Maximum overflow connections")
    checked_in: Optional[int] = Field(None, description="Connections available in pool")
    checked_out: Optional[int] = Field(None, description="Connections currently in use")
    overflow: Optional[int] = Field(None, description="Current overflow connections")
    pool_timeout: Optional[float] = Field(None, description="Pool connection timeout in seconds")


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    qdrant_connected: bool
    redis_connected: bool
    postgres_connected: bool = Field(False, description="Whether PostgreSQL is connected")
    reranker_enabled: bool = Field(False, description="Whether reranker is enabled")
    reranker_loaded: bool = Field(False, description="Whether reranker model is loaded")
    reranker_model: Optional[str] = Field(None, description="Reranker model name")
    redis_pool: Optional[RedisPoolStats] = Field(None, description="Redis connection pool statistics")
    postgres_pool: Optional[PostgresPoolStats] = Field(None, description="PostgreSQL connection pool statistics")
    components: Optional[Dict[str, ComponentStatus]] = Field(None, description="Detailed component status")


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
