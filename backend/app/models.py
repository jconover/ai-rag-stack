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


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    qdrant_connected: bool
    redis_connected: bool
    reranker_enabled: bool = Field(False, description="Whether reranker is enabled")
    reranker_loaded: bool = Field(False, description="Whether reranker model is loaded")
    reranker_model: Optional[str] = Field(None, description="Reranker model name")
    redis_pool: Optional[RedisPoolStats] = Field(None, description="Redis connection pool statistics")
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
