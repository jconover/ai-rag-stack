"""SQLAlchemy ORM models for PostgreSQL persistence.

This module defines the database schema for:
- QueryLog: Chat query history with metrics and metadata
- Feedback: User feedback on AI responses
- IngestionRegistry: Track ingested documents for incremental updates

Tables use proper indexing for common query patterns:
- Date range queries for analytics
- Session-based lookups
- Model filtering
- File path lookups for ingestion tracking
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    Float,
    Integer,
    Boolean,
    DateTime,
    Index,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from app.database import Base


class QueryLog(Base):
    """Log of all chat queries for analytics and debugging.

    Stores each query with its context, response metadata, and performance metrics.
    Useful for:
    - Query volume analytics
    - Model usage tracking
    - Performance monitoring
    - Response quality analysis
    """

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique query identifier",
    )

    # Request information
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Session ID for conversation grouping",
    )
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="User's query text",
    )
    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="LLM model used for response",
    )

    # Timestamp (primary time reference)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Query timestamp (alias for created_at for compatibility)",
    )

    # Performance metrics
    latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total end-to-end latency in milliseconds",
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens generated in response",
    )

    # Retrieval metrics as JSONB for flexible score storage
    retrieval_scores: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="JSONB with similarity, rerank, and other retrieval scores",
    )

    # Sources returned as PostgreSQL ARRAY
    sources_returned: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String(500)),
        nullable=True,
        comment="Array of source document paths/identifiers returned",
    )

    # Response information
    response_length: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Length of generated response in characters",
    )
    context_used: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        comment="Whether RAG context was used",
    )
    sources_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of source documents retrieved",
    )

    # RAG pipeline metrics
    retrieval_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time for vector retrieval in milliseconds",
    )
    rerank_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time for reranking in milliseconds",
    )
    total_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total response time in milliseconds",
    )
    avg_similarity_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average similarity score of retrieved documents",
    )
    avg_rerank_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average rerank score of final documents",
    )

    # Feature flags used
    hybrid_search_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether hybrid search was used",
    )
    hyde_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether HyDE query expansion was used",
    )
    reranker_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether reranker was applied",
    )
    web_search_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether web search fallback was triggered",
    )

    # Additional metadata as JSONB for flexibility
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional metadata (retrieval metrics, etc.)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Query timestamp",
    )

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("ix_query_logs_model_created", "model", "created_at"),
        Index("ix_query_logs_session_created", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, session={self.session_id[:8]}..., model={self.model})>"


class Feedback(Base):
    """User feedback on AI responses.

    Stores thumbs up/down feedback with optional correlation to
    specific queries via query_hash or query_log_id.
    """

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique feedback identifier",
    )

    # Session and query context
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Session ID for the conversation",
    )
    message_index: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Index of the response in session (0-based)",
    )
    query_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Hash for correlation with retrieval metrics",
    )
    query_log_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Reference to query_logs.id if available",
    )

    # Feedback data
    helpful: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        comment="True=thumbs up, False=thumbs down",
    )
    rating: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Numeric rating 1-5 (optional granular feedback)",
    )
    feedback_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Type of feedback: accuracy, relevance, completeness, etc.",
    )
    comment: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional user comment/explanation",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Feedback submission timestamp",
    )

    # Composite index for analytics
    __table_args__ = (
        Index("ix_feedback_helpful_created", "helpful", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, helpful={self.helpful}, session={self.session_id[:8]}...)>"


class AnalyticsCache(Base):
    """Cache for pre-computed analytics to avoid expensive queries.

    Stores aggregated statistics that can be refreshed periodically
    rather than computed on every request.
    """

    __tablename__ = "analytics_cache"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    cache_key: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Cache key identifier",
    )
    cache_value: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Cached analytics data",
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When this cache entry was computed",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Cache expiration time",
    )

    def __repr__(self) -> str:
        return f"<AnalyticsCache(key={self.cache_key})>"


class IngestionRegistry(Base):
    """Registry for tracking ingested documents.

    Enables incremental ingestion by tracking:
    - Which files have been ingested
    - Content hashes for change detection
    - Chunk counts for statistics
    - Source type classification

    The file_path is the primary key to ensure uniqueness and
    enable efficient lookups when checking if a file needs re-ingestion.
    """

    __tablename__ = "ingestion_registry"

    # Primary key: file path (unique identifier for documents)
    file_path: Mapped[str] = mapped_column(
        String(1000),
        primary_key=True,
        comment="Absolute or relative path to the ingested file",
    )

    # Content tracking for change detection
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of file content for change detection",
    )

    # Ingestion statistics
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of chunks created from this document",
    )

    # Source classification
    source_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Document source type: kubernetes, terraform, ansible, custom, etc.",
    )

    # File metadata
    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="File size in bytes at time of ingestion",
    )
    file_modified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="File modification timestamp from filesystem",
    )

    # Ingestion metadata
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When this file was last ingested",
    )
    ingestion_duration_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time taken to ingest this file in milliseconds",
    )

    # Processing metadata as JSONB for flexibility
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional metadata: embedding model, chunk settings, etc.",
    )

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("ix_ingestion_registry_source_ingested", "source_type", "ingested_at"),
        Index("ix_ingestion_registry_hash_source", "content_hash", "source_type"),
    )

    def __repr__(self) -> str:
        return f"<IngestionRegistry(path={self.file_path[:50]}..., source={self.source_type}, chunks={self.chunk_count})>"
