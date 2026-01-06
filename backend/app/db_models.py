"""SQLAlchemy ORM models for PostgreSQL persistence.

This module defines the database schema for:
- User: User accounts with authentication
- UserSession: User session management for web authentication
- APIKey: Programmatic API access keys
- QueryLog: Chat query history with metrics and metadata
- Feedback: User feedback on AI responses
- IngestionRegistry: Track ingested documents for incremental updates
- Experiment: A/B testing experiment configuration
- ExperimentAssignment: User assignments to experiment variants
- ExperimentResult: Metrics collected from experiment variants

Tables use proper indexing for common query patterns:
- Date range queries for analytics
- Session-based lookups
- Model filtering
- File path lookups for ingestion tracking
- Experiment variant analysis
- User authentication and authorization
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
    ForeignKey,
    Enum as SQLEnum,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import enum

from app.database import Base


# Enums for A/B testing
class ExperimentType(enum.Enum):
    """Types of A/B testing experiments."""
    MODEL = "model"
    PROMPT = "prompt"
    RAG_CONFIG = "rag_config"
    TEMPERATURE = "temperature"


class ExperimentStatus(enum.Enum):
    """Lifecycle status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


# =============================================================================
# User Account Models
# =============================================================================


class User(Base):
    """User account for authentication and authorization.

    Stores user credentials and profile information. Passwords are stored
    as bcrypt hashes. Supports email verification and account activation.

    Related models:
    - UserSession: Active login sessions
    - APIKey: Programmatic API access keys
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique user identifier",
    )

    # Authentication credentials
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address (unique, used for login)",
    )
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="User display name (unique)",
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Bcrypt hashed password",
    )

    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the account is active (can login)",
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether the email has been verified",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Account creation timestamp",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last account update timestamp",
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful login timestamp",
    )

    # Relationships
    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("ix_users_active_verified", "is_active", "is_verified"),
        Index("ix_users_email_active", "email", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class UserSession(Base):
    """User session for web authentication.

    Tracks active login sessions with hashed tokens. Supports tracking
    of client information (IP, user agent) for security auditing.

    Session tokens should be generated securely and hashed before storage.
    The actual token is returned to the client only once at creation.
    """

    __tablename__ = "user_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique session identifier",
    )

    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the user who owns this session",
    )

    # Session token (hashed)
    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="SHA-256 hash of the session token",
    )

    # Session lifecycle
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Session expiration timestamp",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Session creation timestamp",
    )

    # Client information for security auditing
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # Supports IPv6 addresses
        nullable=True,
        comment="Client IP address at session creation",
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Client user agent string",
    )

    # Relationship back to user
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions",
    )

    # Composite indexes for session management
    __table_args__ = (
        Index("ix_user_sessions_user_expires", "user_id", "expires_at"),
        Index("ix_user_sessions_token_expires", "token_hash", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, expires_at={self.expires_at})>"


class APIKey(Base):
    """API key for programmatic access.

    Allows users to create API keys for automated/programmatic access
    to the system. Keys have:
    - Hashed storage (original key shown only once at creation)
    - Optional expiration
    - Granular permissions via JSONB
    - Usage tracking

    Permissions are stored as JSONB for flexibility, e.g.:
    {"actions": ["chat", "upload"], "rate_limit": 1000}
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique API key identifier",
    )

    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the user who owns this API key",
    )

    # Key identification and storage
    key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="SHA-256 hash of the API key",
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="User-friendly name for the API key",
    )

    # Permissions as JSONB for flexibility
    permissions: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="JSONB with allowed actions and constraints, e.g., {actions: ['chat', 'upload'], rate_limit: 1000}",
    )

    # Lifecycle management
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Optional expiration timestamp (null = never expires)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="API key creation timestamp",
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time this API key was used",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the API key is active (can be used)",
    )

    # Relationship back to user
    user: Mapped["User"] = relationship(
        "User",
        back_populates="api_keys",
    )

    # Composite indexes for API key management
    __table_args__ = (
        Index("ix_api_keys_user_active", "user_id", "is_active"),
        Index("ix_api_keys_hash_active", "key_hash", "is_active"),
        Index("ix_api_keys_user_name", "user_id", "name", unique=True),
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, user_id={self.user_id}, name={self.name}, is_active={self.is_active})>"


# =============================================================================
# Query and Feedback Models
# =============================================================================


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
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="User ID if authenticated (optional)",
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


class Experiment(Base):
    """A/B testing experiment configuration.

    Defines experiments to compare different configurations:
    - Model variants (llama3.1:8b vs mistral:7b)
    - Prompt templates
    - RAG configuration parameters
    - Temperature settings

    Variants and traffic splits are stored as JSONB for flexibility.
    """

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique experiment identifier",
    )

    # Experiment identification
    name: Mapped[str] = mapped_column(
        String(200),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique experiment name",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Detailed description of experiment hypothesis and goals",
    )

    # Experiment configuration
    experiment_type: Mapped[ExperimentType] = mapped_column(
        SQLEnum(ExperimentType, name="experiment_type_enum", create_type=True),
        nullable=False,
        index=True,
        comment="Type of experiment: model, prompt, rag_config, temperature",
    )
    status: Mapped[ExperimentStatus] = mapped_column(
        SQLEnum(ExperimentStatus, name="experiment_status_enum", create_type=True),
        nullable=False,
        default=ExperimentStatus.DRAFT,
        index=True,
        comment="Experiment lifecycle status: draft, running, paused, completed",
    )

    # Variant configuration as JSONB
    variants: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of variant configs, e.g., [{id: 'control', model: 'llama3.1:8b'}, {id: 'treatment', model: 'mistral:7b'}]",
    )
    traffic_split: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Percentage allocation per variant, e.g., {control: 50, treatment: 50}",
    )

    # Success criteria
    success_metric: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Primary metric to measure: rating, latency, relevance, etc.",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="Experiment creation timestamp",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp",
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When experiment started running",
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When experiment ended",
    )

    # Relationships
    assignments: Mapped[List["ExperimentAssignment"]] = relationship(
        "ExperimentAssignment",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )
    results: Mapped[List["ExperimentResult"]] = relationship(
        "ExperimentResult",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    # Composite indexes
    __table_args__ = (
        Index("ix_experiments_status_created", "status", "created_at"),
        Index("ix_experiments_type_status", "experiment_type", "status"),
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status.value})>"


class ExperimentAssignment(Base):
    """Assignment of sessions to experiment variants.

    Tracks which variant each session is assigned to for consistent
    experience throughout the session and for analytics attribution.
    """

    __tablename__ = "experiment_assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique assignment identifier",
    )

    # Foreign key to experiment
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the experiment",
    )

    # Assignment details
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Session ID assigned to this variant",
    )
    variant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Variant identifier from experiment.variants",
    )

    # Timestamp
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When the assignment was made",
    )

    # Relationship back to experiment
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="assignments",
    )

    # Composite indexes for lookups
    __table_args__ = (
        Index("ix_exp_assign_experiment_session", "experiment_id", "session_id", unique=True),
        Index("ix_exp_assign_experiment_variant", "experiment_id", "variant_id"),
    )

    def __repr__(self) -> str:
        return f"<ExperimentAssignment(experiment={self.experiment_id}, session={self.session_id[:8]}..., variant={self.variant_id})>"


class ExperimentResult(Base):
    """Metrics collected from experiment variants.

    Stores individual metric observations for statistical analysis.
    Supports multiple metric types per experiment for comprehensive
    analysis (latency, ratings, relevance scores, etc.).
    """

    __tablename__ = "experiment_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique result identifier",
    )

    # Foreign key to experiment
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the experiment",
    )

    # Variant and session tracking
    variant_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Variant identifier for this result",
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Session that generated this result",
    )

    # Optional link to specific query
    query_log_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_logs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Reference to specific query if applicable",
    )

    # Metric data
    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Name of the metric: latency, rating, relevance, etc.",
    )
    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Numeric value of the metric",
    )

    # Timestamp
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When this result was recorded",
    )

    # Relationship back to experiment
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="results",
    )

    # Composite indexes for analytics queries
    __table_args__ = (
        Index("ix_exp_results_experiment_variant", "experiment_id", "variant_id"),
        Index("ix_exp_results_experiment_metric", "experiment_id", "metric_name"),
        Index("ix_exp_results_variant_recorded", "variant_id", "recorded_at"),
    )

    def __repr__(self) -> str:
        return f"<ExperimentResult(experiment={self.experiment_id}, variant={self.variant_id}, metric={self.metric_name}={self.metric_value})>"
