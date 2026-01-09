"""FastAPI application for DevOps AI Assistant"""
import uuid
import os
import subprocess
import time
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from sqlalchemy.sql import text
import redis
import json

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, HealthResponse,
    ModelsResponse, ModelInfo, StatsResponse,
    FeedbackRequest, FeedbackResponse, RedisPoolStats,
    PostgresPoolStats, QueryLogEntry, QueryLogsResponse,
    QueryAnalyticsSummary,
    # A/B Testing models
    ExperimentCreate, ExperimentUpdate, ExperimentResponse,
    ExperimentListResponse, ExperimentStatsResponse,
    VariantAssignmentResponse, ExperimentResultRecord,
    ExperimentResultResponse, ExperimentVariant,
    VariantStats, StatisticalSignificance,
    # Authentication models
    UserCreate, UserLogin, UserResponse, UserUpdate,
    TokenResponse, APIKeyCreate, APIKeyResponse,
)
from app.rag import rag_pipeline
from app.vectorstore import vector_store
from app.templates import get_templates, get_template_by_id, get_categories
from app.metrics import get_metrics_summary, ENABLE_PROMETHEUS
from app.feedback import feedback_log, get_feedback_summary
from app.database import (
    init_db, close_db, get_db, get_db_context,
    check_postgres_connection, get_postgres_pool_stats
)
from app.db_models import (
    QueryLog, Feedback,
    Experiment, ExperimentAssignment, ExperimentResult,
    ExperimentStatus, ExperimentType,
    User, UserSession, APIKey,
)
from app.auth import (
    auth_service, hash_password, verify_password,
    generate_api_key, hash_token,
    get_current_user, get_optional_user,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Initializing DevOps AI Assistant API...")

    # Initialize PostgreSQL database tables
    try:
        await init_db()
        logger.info("PostgreSQL database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}")
        # Continue without PostgreSQL - degrade gracefully

    yield

    # Shutdown
    logger.info("Shutting down DevOps AI Assistant API...")
    try:
        await close_db()
        logger.info("PostgreSQL connections closed")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL connections: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DevOps AI Assistant API",
    description="RAG-powered AI assistant for DevOps documentation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - origins configurable via CORS_ORIGINS env var
# Default: http://localhost:3000 (development)
# Production: Set CORS_ORIGINS to comma-separated list of allowed origins
# Warning: Using "*" allows all origins and should only be used for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection pool for conversation memory
# Connection pooling improves performance by reusing connections instead of
# creating new ones for each request. This reduces connection overhead and
# prevents connection exhaustion under high load.
redis_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    max_connections=settings.redis_max_connections,
    socket_timeout=settings.redis_socket_timeout,
    socket_connect_timeout=settings.redis_socket_connect_timeout,
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=redis_pool)


def get_redis_pool_stats() -> dict:
    """Get Redis connection pool statistics for monitoring.

    Returns:
        Dictionary with pool configuration and current usage stats
    """
    try:
        pool_info = {
            "max_connections": redis_pool.max_connections,
            "current_connections": len(redis_pool._in_use_connections),
            "available_connections": len(redis_pool._available_connections),
            "host": settings.redis_host,
            "port": settings.redis_port,
            "db": settings.redis_db,
        }
        return pool_info
    except Exception:
        return {
            "max_connections": settings.redis_max_connections,
            "error": "Unable to retrieve pool stats"
        }


def get_conversation_history(session_id: str, limit: int = 5) -> list:
    """Get conversation history from Redis"""
    try:
        history_key = f"chat:{session_id}"
        messages = redis_client.lrange(history_key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    except:
        return []


def save_message(session_id: str, role: str, content: str):
    """Save message to conversation history using pipeline for efficiency."""
    try:
        history_key = f"chat:{session_id}"
        message = json.dumps({"role": role, "content": content})
        # Use pipeline to batch rpush and expire into single round-trip
        pipe = redis_client.pipeline()
        pipe.rpush(history_key, message)
        pipe.expire(history_key, 86400)  # 24 hour expiry
        pipe.execute()
    except Exception as e:
        print(f"Error saving message: {e}")


@app.get("/")
async def root():
    return {
        "message": "DevOps AI Assistant API",
        "docs": "/docs",
        "health": "/api/health"
    }


async def log_query_to_postgres(
    session_id: str,
    query: str,
    model: str,
    result: dict,
    total_time_ms: float,
    user_id: Optional[str] = None,
) -> None:
    """Log a chat query to PostgreSQL for analytics.

    This function runs asynchronously and fails silently to avoid
    impacting the main request flow.

    Args:
        session_id: Session identifier
        query: User's query text
        model: LLM model used
        result: Response result dictionary
        total_time_ms: Total request latency in milliseconds
        user_id: Optional authenticated user ID
    """
    try:
        async with get_db_context() as db:
            # Extract metrics from result
            retrieval_metrics = result.get('retrieval_metrics', {})
            sources = result.get('sources', [])

            # Build retrieval scores dict
            retrieval_scores = None
            if retrieval_metrics:
                retrieval_scores = {
                    "avg_similarity_score": retrieval_metrics.get('avg_similarity_score'),
                    "avg_rerank_score": retrieval_metrics.get('avg_rerank_score'),
                    "initial_candidates": retrieval_metrics.get('initial_candidates'),
                    "after_reranking": retrieval_metrics.get('after_reranking'),
                }

            # Extract source paths
            sources_returned = None
            if sources:
                sources_returned = [s.get('source', '') for s in sources if s.get('source')]

            # Create query log entry
            query_log = QueryLog(
                session_id=session_id,
                user_id=uuid.UUID(user_id) if user_id else None,
                query=query,
                model=model,
                response_length=len(result.get('response', '')),
                context_used=result.get('context_used', True),
                sources_count=len(sources) if sources else 0,
                sources_returned=sources_returned,
                retrieval_scores=retrieval_scores,
                latency_ms=total_time_ms,
                total_time_ms=total_time_ms,
                retrieval_time_ms=retrieval_metrics.get('retrieval_time_ms') if retrieval_metrics else None,
                rerank_time_ms=retrieval_metrics.get('rerank_time_ms') if retrieval_metrics else None,
                avg_similarity_score=retrieval_metrics.get('avg_similarity_score') if retrieval_metrics else None,
                avg_rerank_score=retrieval_metrics.get('avg_rerank_score') if retrieval_metrics else None,
                hybrid_search_used=retrieval_metrics.get('hybrid_search_used', False) if retrieval_metrics else False,
                hyde_used=retrieval_metrics.get('hyde_used', False) if retrieval_metrics else False,
                reranker_used=retrieval_metrics.get('reranker_used', False) if retrieval_metrics else False,
                web_search_used=retrieval_metrics.get('web_search_used', False) if retrieval_metrics else False,
                metadata={
                    "temperature": result.get('temperature'),
                    "web_search_reason": retrieval_metrics.get('web_search_reason') if retrieval_metrics else None,
                }
            )

            db.add(query_log)
            await db.commit()

    except Exception as e:
        # Log error but don't raise - query logging should not break the main flow
        logger.warning(f"Failed to log query to PostgreSQL: {e}")


async def save_feedback_to_postgres(
    session_id: str,
    helpful: bool,
    message_index: Optional[int] = None,
    query_hash: Optional[str] = None,
) -> Optional[str]:
    """Save feedback to PostgreSQL.

    Args:
        session_id: Session identifier
        helpful: Whether the response was helpful
        message_index: Index of the response in session
        query_hash: Hash for correlation with retrieval metrics

    Returns:
        Feedback ID if successful, None otherwise
    """
    try:
        async with get_db_context() as db:
            feedback = Feedback(
                session_id=session_id,
                helpful=helpful,
                message_index=message_index,
                query_hash=query_hash,
            )
            db.add(feedback)
            await db.commit()
            await db.refresh(feedback)
            return str(feedback.id)
    except Exception as e:
        logger.warning(f"Failed to save feedback to PostgreSQL: {e}")
        return None


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status including reranker, Redis pool, and PostgreSQL."""
    ollama_connected = rag_pipeline.is_ollama_connected()
    qdrant_connected = vector_store.is_connected()

    try:
        redis_client.ping()
        redis_connected = True
    except:
        redis_connected = False

    # Check PostgreSQL connection
    postgres_connected = await check_postgres_connection()

    # Get reranker status
    reranker_status = rag_pipeline.get_reranker_status()

    # Get Redis pool statistics
    pool_stats = get_redis_pool_stats()
    redis_pool_stats = RedisPoolStats(
        max_connections=pool_stats.get("max_connections", settings.redis_max_connections),
        current_connections=pool_stats.get("current_connections"),
        available_connections=pool_stats.get("available_connections"),
        host=pool_stats.get("host"),
        port=pool_stats.get("port"),
        db=pool_stats.get("db"),
    )

    # Get PostgreSQL pool statistics
    pg_pool_stats = await get_postgres_pool_stats()
    postgres_pool_stats = PostgresPoolStats(
        host=pg_pool_stats.get("host"),
        port=pg_pool_stats.get("port"),
        database=pg_pool_stats.get("database"),
        pool_size=pg_pool_stats.get("pool_size", settings.postgres_pool_size),
        max_overflow=pg_pool_stats.get("max_overflow", settings.postgres_max_overflow),
        checked_in=pg_pool_stats.get("checked_in"),
        checked_out=pg_pool_stats.get("checked_out"),
        overflow=pg_pool_stats.get("overflow"),
        pool_timeout=pg_pool_stats.get("pool_timeout"),
    )

    # Core services must be connected for healthy status
    # PostgreSQL is optional - not required for core functionality
    core_healthy = all([ollama_connected, qdrant_connected, redis_connected])

    # If reranker is enabled but not loaded, status is degraded
    reranker_healthy = (
        not reranker_status.get('enabled') or
        reranker_status.get('loaded', False)
    )

    # Determine overall status
    if core_healthy and reranker_healthy and postgres_connected:
        status = "healthy"
    elif core_healthy and reranker_healthy:
        status = "degraded"  # PostgreSQL down but core services OK
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        ollama_connected=ollama_connected,
        qdrant_connected=qdrant_connected,
        redis_connected=redis_connected,
        postgres_connected=postgres_connected,
        reranker_enabled=reranker_status.get('enabled', False),
        reranker_loaded=reranker_status.get('loaded', False),
        reranker_model=reranker_status.get('model_name'),
        redis_pool=redis_pool_stats,
        postgres_pool=postgres_pool_stats,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    req: Request,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """Chat with the AI assistant.

    Optionally authenticates the user to associate queries with their account.
    Works with or without authentication.
    """

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Track user_id for authenticated requests
    user_id = str(current_user.id) if current_user else None

    # Save user message
    save_message(session_id, "user", request.message)

    # Track timing for query logging
    start_time = time.time()

    # Check for active A/B experiment
    experiment_config = None
    experiment_id = None
    variant_name = None

    try:
        experiment_config = await get_active_experiment_config(session_id, "model")
        if experiment_config:
            experiment_id = experiment_config.get("experiment_id")
            variant_name = experiment_config.get("variant_name")
            variant_cfg = experiment_config.get("config", {})

            # Apply experiment variant settings (only if not explicitly set in request)
            if request.model is None and "model" in variant_cfg:
                request.model = variant_cfg["model"]
            if request.temperature is None and "temperature" in variant_cfg:
                request.temperature = variant_cfg["temperature"]
    except Exception as e:
        logger.warning(f"Failed to get experiment config: {e}")
        # Continue without experiment - graceful degradation

    try:
        # Generate response
        result = await rag_pipeline.generate_response(
            query=request.message,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_rag=request.use_rag,
        )

        # Calculate total latency
        total_time_ms = (time.time() - start_time) * 1000

        # Save assistant response
        save_message(session_id, "assistant", result['response'])

        # Log query to PostgreSQL (async, non-blocking)
        if settings.query_logging_enabled:
            await log_query_to_postgres(
                session_id=session_id,
                query=request.message,
                model=result['model'],
                result=result,
                total_time_ms=total_time_ms,
                user_id=user_id,
            )

        # Record experiment metrics if in an active experiment
        if experiment_id and variant_name:
            await record_experiment_metrics(
                experiment_id=experiment_id,
                session_id=session_id,
                variant_name=variant_name,
                latency_ms=total_time_ms,
            )

        return ChatResponse(
            response=result['response'],
            model=result['model'],
            context_used=result['context_used'],
            sources=result['sources'],
            session_id=session_id,
            retrieval_metrics=result.get('retrieval_metrics'),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses from the AI assistant"""

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Save user message
    save_message(session_id, "user", request.message)

    async def generate():
        full_response = ""
        try:
            async for chunk in rag_pipeline.generate_response_stream(
                query=request.message,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_rag=request.use_rag,
            ):
                # Accumulate response content
                if chunk.get('type') == 'content':
                    full_response += chunk.get('content', '')

                # Add session_id to metadata
                if chunk.get('type') == 'metadata':
                    chunk['session_id'] = session_id

                # Stream as Server-Sent Events format
                yield f"data: {json.dumps(chunk)}\n\n"

            # Save complete assistant response
            if full_response:
                save_message(session_id, "assistant", full_response)

        except Exception as e:
            error_chunk = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """List available Ollama models"""
    try:
        models = rag_pipeline.list_models()
        model_infos = [
            ModelInfo(
                name=m.get('name', ''),
                size=str(m.get('size', '')),
                modified=m.get('modified_at', '')
            )
            for m in models
        ]
        return ModelsResponse(models=model_infos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector database statistics"""
    try:
        stats = vector_store.get_stats()
        return StatsResponse(
            collection_name=stats['collection_name'],
            vectors_count=stats.get('vectors_count', 0),
            indexed_documents=stats.get('points_count', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """Get conversation history for a session"""
    history = get_conversation_history(session_id, limit)
    return {"session_id": session_id, "messages": history}


@app.get("/api/templates")
async def list_templates(category: Optional[str] = None):
    """Get prompt templates, optionally filtered by category"""
    templates = get_templates()
    if category:
        templates = [t for t in templates if t['category'] == category]
    return {
        "templates": templates,
        "categories": get_categories()
    }


@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific prompt template by ID"""
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    auto_ingest: bool = True
):
    """Upload documentation files and optionally trigger ingestion"""

    # Create custom docs directory if it doesn't exist
    custom_docs_dir = Path("/data/custom")
    custom_docs_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    failed_files = []

    # Save uploaded files
    for file in files:
        try:
            # Validate file type
            if not (file.filename.endswith('.md') or
                    file.filename.endswith('.txt') or
                    file.filename.endswith('.markdown')):
                failed_files.append({
                    "filename": file.filename,
                    "error": "Only .md, .txt, and .markdown files are supported"
                })
                continue

            # Save file
            file_path = custom_docs_dir / file.filename
            content = await file.read()

            with open(file_path, 'wb') as f:
                f.write(content)

            uploaded_files.append({
                "filename": file.filename,
                "size": len(content),
                "path": str(file_path)
            })

        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })

    result = {
        "uploaded": len(uploaded_files),
        "failed": len(failed_files),
        "files": uploaded_files,
        "errors": failed_files if failed_files else None,
    }

    # Trigger ingestion if requested and files were uploaded
    if auto_ingest and uploaded_files:
        try:
            # Run ingestion script in background
            ingestion_result = subprocess.run(
                ["python", "/scripts/ingest_docs.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            result["ingestion"] = {
                "status": "success" if ingestion_result.returncode == 0 else "failed",
                "output": ingestion_result.stdout,
                "error": ingestion_result.stderr if ingestion_result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            result["ingestion"] = {
                "status": "timeout",
                "error": "Ingestion took longer than 5 minutes"
            }
        except Exception as e:
            result["ingestion"] = {
                "status": "error",
                "error": str(e)
            }

    return result


@app.get("/api/metrics/retrieval")
async def get_retrieval_metrics(last_n: int = 100):
    """Get summary of recent retrieval metrics.

    Returns aggregated statistics from the last N retrieval operations,
    including score distributions, latency percentiles, and query success rates.

    Args:
        last_n: Number of recent entries to analyze (default 100)

    Returns:
        Summary statistics for retrieval quality monitoring
    """
    if not settings.enable_retrieval_metrics:
        raise HTTPException(
            status_code=503,
            detail="Retrieval metrics are disabled. Set ENABLE_RETRIEVAL_METRICS=true"
        )

    try:
        summary = get_metrics_summary(last_n=last_n)
        return {
            "status": "ok",
            "metrics_enabled": True,
            "prometheus_enabled": ENABLE_PROMETHEUS,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a response (thumbs up/down).

    Logs user feedback to track response quality. Feedback is stored in both
    the file-based log (for backward compatibility) and PostgreSQL (for analytics).
    Feedback can be correlated with retrieval metrics using the query_hash field.

    Args:
        request: Feedback request with session_id, helpful flag, and optional metadata

    Returns:
        FeedbackResponse with feedback_id and timestamp
    """
    try:
        # Log to file-based system (existing behavior)
        entry = feedback_log.log_feedback(
            session_id=request.session_id,
            helpful=request.helpful,
            message_index=request.message_index,
            query_hash=request.query_hash
        )

        # Also save to PostgreSQL for analytics
        pg_feedback_id = await save_feedback_to_postgres(
            session_id=request.session_id,
            helpful=request.helpful,
            message_index=request.message_index,
            query_hash=request.query_hash,
        )

        # Use PostgreSQL ID if available, otherwise use file-based ID
        feedback_id = pg_feedback_id or entry.feedback_id

        return FeedbackResponse(
            status="saved",
            feedback_id=feedback_id,
            timestamp=entry.timestamp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/summary")
async def get_feedback_stats(last_n: int = 100):
    """Get summary of recent user feedback.

    Returns aggregated statistics from the last N feedback entries,
    including helpful rate and unique session count.

    Args:
        last_n: Number of recent entries to analyze (default 100)

    Returns:
        Summary statistics for feedback quality monitoring
    """
    try:
        summary = get_feedback_summary(last_n=last_n)
        return {
            "status": "ok",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/queries", response_model=QueryLogsResponse)
async def get_query_logs(
    start_date: Optional[str] = Query(None, description="Start date (ISO format or YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format or YYYY-MM-DD)"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    min_latency_ms: Optional[float] = Query(None, description="Minimum latency threshold"),
    max_latency_ms: Optional[float] = Query(None, description="Maximum latency threshold"),
    hybrid_search_used: Optional[bool] = Query(None, description="Filter by hybrid search usage"),
    web_search_used: Optional[bool] = Query(None, description="Filter by web search usage"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=500, description="Records per page"),
):
    """Retrieve query logs with optional filters for analytics.

    Provides paginated access to stored query logs with various filter options
    including date range, model, latency thresholds, and feature usage flags.

    Args:
        start_date: Filter queries after this date
        end_date: Filter queries before this date
        model: Filter by specific LLM model
        session_id: Filter by session ID
        min_latency_ms: Minimum latency in milliseconds
        max_latency_ms: Maximum latency in milliseconds
        hybrid_search_used: Filter by hybrid search usage
        web_search_used: Filter by web search fallback usage
        page: Page number (1-indexed)
        page_size: Number of records per page

    Returns:
        QueryLogsResponse with paginated query logs
    """
    try:
        async with get_db_context() as db:
            # Build query with filters
            query = select(QueryLog)
            conditions = []

            # Date range filters
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    conditions.append(QueryLog.timestamp >= start_dt)
                except ValueError:
                    # Try parsing as date only
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    conditions.append(QueryLog.timestamp >= start_dt)

            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    conditions.append(QueryLog.timestamp <= end_dt)
                except ValueError:
                    # Try parsing as date only (end of day)
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59, tzinfo=timezone.utc
                    )
                    conditions.append(QueryLog.timestamp <= end_dt)

            # Model filter
            if model:
                conditions.append(QueryLog.model == model)

            # Session filter
            if session_id:
                conditions.append(QueryLog.session_id == session_id)

            # Latency filters
            if min_latency_ms is not None:
                conditions.append(QueryLog.latency_ms >= min_latency_ms)
            if max_latency_ms is not None:
                conditions.append(QueryLog.latency_ms <= max_latency_ms)

            # Feature usage filters
            if hybrid_search_used is not None:
                conditions.append(QueryLog.hybrid_search_used == hybrid_search_used)
            if web_search_used is not None:
                conditions.append(QueryLog.web_search_used == web_search_used)

            # Apply conditions
            if conditions:
                query = query.where(and_(*conditions))

            # Get total count
            count_query = select(func.count()).select_from(QueryLog)
            if conditions:
                count_query = count_query.where(and_(*conditions))
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0

            # Apply pagination and ordering
            offset = (page - 1) * page_size
            query = query.order_by(desc(QueryLog.timestamp)).offset(offset).limit(page_size)

            # Execute query
            result = await db.execute(query)
            logs = result.scalars().all()

            # Convert to response format
            queries = []
            for log in logs:
                queries.append(QueryLogEntry(
                    id=str(log.id),
                    session_id=log.session_id,
                    query=log.query,
                    model=log.model,
                    timestamp=log.timestamp.isoformat() if log.timestamp else "",
                    latency_ms=log.latency_ms,
                    token_count=log.token_count,
                    response_length=log.response_length,
                    context_used=log.context_used,
                    sources_count=log.sources_count,
                    sources_returned=log.sources_returned,
                    retrieval_scores=log.retrieval_scores,
                    hybrid_search_used=log.hybrid_search_used,
                    hyde_used=log.hyde_used,
                    reranker_used=log.reranker_used,
                    web_search_used=log.web_search_used,
                ))

            # Calculate total pages
            total_pages = (total + page_size - 1) // page_size if total > 0 else 1

            return QueryLogsResponse(
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                queries=queries,
            )

    except Exception as e:
        logger.error(f"Failed to retrieve query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/queries/summary", response_model=QueryAnalyticsSummary)
async def get_query_analytics_summary(
    start_date: Optional[str] = Query(None, description="Start date (ISO format or YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format or YYYY-MM-DD)"),
):
    """Get summary analytics for query logs.

    Provides aggregated statistics including total queries, latency percentiles,
    model distribution, and feature usage counts.

    Args:
        start_date: Filter queries after this date
        end_date: Filter queries before this date

    Returns:
        QueryAnalyticsSummary with aggregated statistics
    """
    try:
        async with get_db_context() as db:
            conditions = []

            # Date range filters
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                except ValueError:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                conditions.append(QueryLog.timestamp >= start_dt)

            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59, tzinfo=timezone.utc
                    )
                conditions.append(QueryLog.timestamp <= end_dt)

            # Base query with conditions
            base_condition = and_(*conditions) if conditions else True

            # Total queries
            total_query = select(func.count()).select_from(QueryLog).where(base_condition)
            total_result = await db.execute(total_query)
            total_queries = total_result.scalar() or 0

            # Unique sessions
            sessions_query = select(func.count(func.distinct(QueryLog.session_id))).where(base_condition)
            sessions_result = await db.execute(sessions_query)
            unique_sessions = sessions_result.scalar() or 0

            # Average latency
            avg_latency_query = select(func.avg(QueryLog.latency_ms)).where(base_condition)
            avg_result = await db.execute(avg_latency_query)
            avg_latency = avg_result.scalar()

            # Latency percentiles (using PostgreSQL percentile_cont)
            # Build date filter clause for raw SQL
            date_conditions = ["latency_ms IS NOT NULL"]
            query_params = {}

            if start_date:
                date_conditions.append("timestamp >= :start_dt")
                query_params["start_dt"] = start_dt

            if end_date:
                date_conditions.append("timestamp <= :end_dt")
                query_params["end_dt"] = end_dt

            where_clause = " AND ".join(date_conditions)

            percentile_query = text(f"""
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
                    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
                FROM query_logs
                WHERE {where_clause}
            """)

            percentile_result = await db.execute(percentile_query, query_params)
            percentiles = percentile_result.fetchone()
            p50 = percentiles[0] if percentiles else None
            p95 = percentiles[1] if percentiles else None
            p99 = percentiles[2] if percentiles else None

            # Model distribution
            model_query = (
                select(QueryLog.model, func.count().label('count'))
                .where(base_condition)
                .group_by(QueryLog.model)
            )
            model_result = await db.execute(model_query)
            model_distribution = {row.model: row.count for row in model_result}

            # Feature usage counts
            feature_usage = {}

            # Hybrid search usage
            hybrid_query = select(func.count()).select_from(QueryLog).where(
                and_(base_condition, QueryLog.hybrid_search_used == True)
            )
            hybrid_result = await db.execute(hybrid_query)
            feature_usage["hybrid_search"] = hybrid_result.scalar() or 0

            # HyDE usage
            hyde_query = select(func.count()).select_from(QueryLog).where(
                and_(base_condition, QueryLog.hyde_used == True)
            )
            hyde_result = await db.execute(hyde_query)
            feature_usage["hyde"] = hyde_result.scalar() or 0

            # Reranker usage
            reranker_query = select(func.count()).select_from(QueryLog).where(
                and_(base_condition, QueryLog.reranker_used == True)
            )
            reranker_result = await db.execute(reranker_query)
            feature_usage["reranker"] = reranker_result.scalar() or 0

            # Web search usage
            web_query = select(func.count()).select_from(QueryLog).where(
                and_(base_condition, QueryLog.web_search_used == True)
            )
            web_result = await db.execute(web_query)
            feature_usage["web_search"] = web_result.scalar() or 0

            # Queries per day
            daily_query = (
                select(
                    func.date_trunc('day', QueryLog.timestamp).label('day'),
                    func.count().label('count')
                )
                .where(base_condition)
                .group_by(func.date_trunc('day', QueryLog.timestamp))
                .order_by(func.date_trunc('day', QueryLog.timestamp))
            )
            daily_result = await db.execute(daily_query)
            queries_per_day = {
                row.day.strftime("%Y-%m-%d") if row.day else "unknown": row.count
                for row in daily_result
            }

            return QueryAnalyticsSummary(
                total_queries=total_queries,
                unique_sessions=unique_sessions,
                avg_latency_ms=round(avg_latency, 2) if avg_latency else None,
                p50_latency_ms=round(p50, 2) if p50 else None,
                p95_latency_ms=round(p95, 2) if p95 else None,
                p99_latency_ms=round(p99, 2) if p99 else None,
                model_distribution=model_distribution,
                feature_usage=feature_usage,
                queries_per_day=queries_per_day,
            )

    except Exception as e:
        logger.error(f"Failed to compute query analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# A/B Testing Endpoints
# =====================

def _experiment_to_response(exp: Experiment) -> ExperimentResponse:
    """Convert SQLAlchemy Experiment model to Pydantic response."""
    # Convert variants from DB format to Pydantic models
    variants_list = exp.variants if isinstance(exp.variants, list) else exp.variants.get('variants', [])
    variants = [
        ExperimentVariant(
            name=v.get('id', v.get('name', '')),
            weight=exp.traffic_split.get(v.get('id', v.get('name', '')), 0.5) if exp.traffic_split else 0.5,
            config={k: v for k, v in v.items() if k not in ('id', 'name')},
        )
        for v in variants_list
    ]

    # Calculate traffic percentage from weights
    total_weight = sum(v.weight for v in variants)
    traffic_percentage = total_weight * 100 if total_weight <= 1 else 100.0

    return ExperimentResponse(
        id=str(exp.id),
        name=exp.name,
        description=exp.description,
        experiment_type=exp.experiment_type.value if hasattr(exp.experiment_type, 'value') else str(exp.experiment_type),
        status=exp.status.value if hasattr(exp.status, 'value') else str(exp.status),
        variants=variants,
        traffic_percentage=traffic_percentage,
        created_at=exp.created_at.isoformat() if exp.created_at else "",
        updated_at=exp.updated_at.isoformat() if exp.updated_at else "",
        start_at=None,  # Not in current schema
        end_at=None,  # Not in current schema
        started_at=exp.started_at.isoformat() if exp.started_at else None,
        ended_at=exp.ended_at.isoformat() if exp.ended_at else None,
        metadata=None,
    )


def _get_experiment_assignment_key(experiment_id: str, session_id: str) -> str:
    """Generate Redis key for experiment assignment."""
    return f"exp_assign:{experiment_id}:{session_id}"


def _assign_variant(variants: list, traffic_split: dict, session_id: str) -> str:
    """Deterministically assign a session to a variant based on hash.

    Uses consistent hashing to ensure the same session always gets the same variant.
    """
    import hashlib

    # Create deterministic hash from session_id
    hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    bucket = (hash_val % 100) / 100.0  # 0.0 to 0.99

    # Walk through variants according to traffic split
    cumulative = 0.0
    for variant in variants:
        variant_id = variant.get('id', variant.get('name', ''))
        weight = traffic_split.get(variant_id, 0.0) / 100.0  # Convert percentage to decimal
        cumulative += weight
        if bucket < cumulative:
            return variant_id

    # Fallback to first variant
    return variants[0].get('id', variants[0].get('name', '')) if variants else 'control'


@app.post("/api/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(request: ExperimentCreate):
    """Create a new A/B testing experiment.

    Creates an experiment with the specified variants and traffic allocation.
    The experiment starts in 'draft' status and must be explicitly started.

    Args:
        request: Experiment configuration including name, variants, and traffic split

    Returns:
        ExperimentResponse with the created experiment details
    """
    try:
        async with get_db_context() as db:
            # Check for duplicate name
            existing = await db.execute(
                select(Experiment).where(Experiment.name == request.name)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail=f"Experiment with name '{request.name}' already exists"
                )

            # Convert variants to DB format
            variants_db = [
                {"id": v.name, **v.config}
                for v in request.variants
            ]

            # Normalize weights to percentages
            total_weight = sum(v.weight for v in request.variants)
            traffic_split = {
                v.name: (v.weight / total_weight * 100) if total_weight > 0 else (100 / len(request.variants))
                for v in request.variants
            }

            # Map experiment type string to enum
            exp_type_map = {
                "model": ExperimentType.MODEL,
                "prompt": ExperimentType.PROMPT,
                "config": ExperimentType.RAG_CONFIG,
                "rag_config": ExperimentType.RAG_CONFIG,
                "temperature": ExperimentType.TEMPERATURE,
            }
            experiment_type = exp_type_map.get(request.experiment_type.lower(), ExperimentType.MODEL)

            # Create experiment
            experiment = Experiment(
                name=request.name,
                description=request.description,
                experiment_type=experiment_type,
                status=ExperimentStatus.DRAFT,
                variants=variants_db,
                traffic_split=traffic_split,
                success_metric="latency_ms",  # Default metric
            )

            db.add(experiment)
            await db.commit()
            await db.refresh(experiment)

            return _experiment_to_response(experiment)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status: draft, running, paused, completed"),
    experiment_type: Optional[str] = Query(None, description="Filter by type: model, prompt, rag_config, temperature"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Records per page"),
):
    """List all experiments with optional filtering.

    Args:
        status: Filter by experiment status
        experiment_type: Filter by experiment type
        page: Page number (1-indexed)
        page_size: Number of records per page

    Returns:
        ExperimentListResponse with paginated experiments
    """
    try:
        async with get_db_context() as db:
            query = select(Experiment)
            conditions = []

            # Status filter
            if status:
                status_map = {
                    "draft": ExperimentStatus.DRAFT,
                    "running": ExperimentStatus.RUNNING,
                    "paused": ExperimentStatus.PAUSED,
                    "completed": ExperimentStatus.COMPLETED,
                }
                if status.lower() in status_map:
                    conditions.append(Experiment.status == status_map[status.lower()])

            # Type filter
            if experiment_type:
                type_map = {
                    "model": ExperimentType.MODEL,
                    "prompt": ExperimentType.PROMPT,
                    "config": ExperimentType.RAG_CONFIG,
                    "rag_config": ExperimentType.RAG_CONFIG,
                    "temperature": ExperimentType.TEMPERATURE,
                }
                if experiment_type.lower() in type_map:
                    conditions.append(Experiment.experiment_type == type_map[experiment_type.lower()])

            if conditions:
                query = query.where(and_(*conditions))

            # Get total count
            count_query = select(func.count()).select_from(Experiment)
            if conditions:
                count_query = count_query.where(and_(*conditions))
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0

            # Apply pagination and ordering
            offset = (page - 1) * page_size
            query = query.order_by(desc(Experiment.created_at)).offset(offset).limit(page_size)

            result = await db.execute(query)
            experiments = result.scalars().all()

            return ExperimentListResponse(
                total=total,
                page=page,
                page_size=page_size,
                experiments=[_experiment_to_response(exp) for exp in experiments],
            )

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get details of a specific experiment.

    Args:
        experiment_id: UUID of the experiment

    Returns:
        ExperimentResponse with experiment details
    """
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.id == uuid.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            return _experiment_to_response(experiment)

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to get experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(experiment_id: str, request: ExperimentUpdate):
    """Update an existing experiment.

    Only certain fields can be updated based on experiment status:
    - draft: All fields can be updated
    - running: Only status (to pause/complete) and end_at can be updated
    - paused: Can resume (status to running) or complete
    - completed: No updates allowed

    Args:
        experiment_id: UUID of the experiment
        request: Fields to update

    Returns:
        ExperimentResponse with updated experiment details
    """
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.id == uuid.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            current_status = experiment.status

            # Handle status transitions
            if request.status:
                new_status_map = {
                    "draft": ExperimentStatus.DRAFT,
                    "running": ExperimentStatus.RUNNING,
                    "paused": ExperimentStatus.PAUSED,
                    "completed": ExperimentStatus.COMPLETED,
                }
                new_status = new_status_map.get(request.status.lower())

                if not new_status:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid status: {request.status}"
                    )

                # Validate state transitions
                valid_transitions = {
                    ExperimentStatus.DRAFT: [ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED],
                    ExperimentStatus.RUNNING: [ExperimentStatus.PAUSED, ExperimentStatus.COMPLETED],
                    ExperimentStatus.PAUSED: [ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED],
                    ExperimentStatus.COMPLETED: [],  # No transitions allowed
                }

                if new_status not in valid_transitions.get(current_status, []):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot transition from {current_status.value} to {new_status.value}"
                    )

                experiment.status = new_status

                # Record start/end times
                if new_status == ExperimentStatus.RUNNING and not experiment.started_at:
                    experiment.started_at = datetime.now(timezone.utc)
                elif new_status == ExperimentStatus.COMPLETED:
                    experiment.ended_at = datetime.now(timezone.utc)

            # Only allow other updates if draft
            if current_status == ExperimentStatus.DRAFT:
                if request.name:
                    # Check for duplicate name
                    existing = await db.execute(
                        select(Experiment).where(
                            and_(
                                Experiment.name == request.name,
                                Experiment.id != uuid.UUID(experiment_id)
                            )
                        )
                    )
                    if existing.scalar_one_or_none():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Experiment with name '{request.name}' already exists"
                        )
                    experiment.name = request.name

                if request.description is not None:
                    experiment.description = request.description

                if request.variants:
                    variants_db = [
                        {"id": v.name, **v.config}
                        for v in request.variants
                    ]
                    experiment.variants = variants_db

                    # Update traffic split
                    total_weight = sum(v.weight for v in request.variants)
                    experiment.traffic_split = {
                        v.name: (v.weight / total_weight * 100) if total_weight > 0 else (100 / len(request.variants))
                        for v in request.variants
                    }

                if request.traffic_percentage is not None:
                    # Scale all weights proportionally
                    scale_factor = request.traffic_percentage / 100.0
                    experiment.traffic_split = {
                        k: v * scale_factor
                        for k, v in experiment.traffic_split.items()
                    }

            await db.commit()
            await db.refresh(experiment)

            return _experiment_to_response(experiment)

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to update experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/experiments/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str):
    """Delete an experiment.

    Running experiments cannot be deleted - they must be completed first.
    Deleting an experiment also removes all associated assignments and results.

    Args:
        experiment_id: UUID of the experiment
    """
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.id == uuid.UUID(experiment_id))
            )
            experiment = result.scalar_one_or_none()

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            if experiment.status == ExperimentStatus.RUNNING:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete a running experiment. Complete or pause it first."
                )

            await db.delete(experiment)
            await db.commit()

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiments/{experiment_id}/stats", response_model=ExperimentStatsResponse)
async def get_experiment_stats(experiment_id: str):
    """Get statistics and significance tests for an experiment.

    Computes per-variant statistics and runs statistical significance tests
    comparing each treatment variant to the control (first variant).

    Args:
        experiment_id: UUID of the experiment

    Returns:
        ExperimentStatsResponse with variant statistics and significance tests
    """
    try:
        async with get_db_context() as db:
            # Get experiment
            exp_result = await db.execute(
                select(Experiment).where(Experiment.id == uuid.UUID(experiment_id))
            )
            experiment = exp_result.scalar_one_or_none()

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            # Get all results for this experiment in a single query
            # This eliminates the N+1 query pattern where we previously fetched
            # feedback data separately for each variant in a loop
            results_query = select(ExperimentResult).where(
                ExperimentResult.experiment_id == uuid.UUID(experiment_id)
            )
            results_result = await db.execute(results_query)
            results = results_result.scalars().all()

            # Group results by variant and process all metrics including feedback
            # in a single pass through the results (no additional queries needed)
            variant_data = {}
            for r in results:
                if r.variant_id not in variant_data:
                    variant_data[r.variant_id] = {
                        'metrics': [],
                        'conversions': 0,
                        'count': 0,
                        'positive_feedback': 0,
                        'negative_feedback': 0,
                    }

                # Track feedback metrics separately from other metrics
                if r.metric_name == 'feedback_positive' and r.metric_value > 0:
                    variant_data[r.variant_id]['positive_feedback'] += 1
                elif r.metric_name == 'feedback_negative' and r.metric_value > 0:
                    variant_data[r.variant_id]['negative_feedback'] += 1
                else:
                    # Non-feedback metrics contribute to the metrics list
                    variant_data[r.variant_id]['metrics'].append(r.metric_value)
                    variant_data[r.variant_id]['count'] += 1

                # Track conversions regardless of metric type
                if r.metric_name == 'conversion' or (hasattr(r, 'is_conversion') and r.is_conversion):
                    variant_data[r.variant_id]['conversions'] += 1

            # Build variant stats
            variant_stats = []
            for variant_id, data in variant_data.items():
                metrics = data['metrics']
                count = data['count']

                if metrics:
                    import statistics
                    avg_metric = statistics.mean(metrics)
                    std_metric = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
                    sorted_metrics = sorted(metrics)
                    p50 = sorted_metrics[len(sorted_metrics) // 2] if sorted_metrics else None
                    p95_idx = int(len(sorted_metrics) * 0.95)
                    p95 = sorted_metrics[min(p95_idx, len(sorted_metrics) - 1)] if sorted_metrics else None
                else:
                    avg_metric = std_metric = p50 = p95 = None

                positive_fb = data.get('positive_feedback', 0)
                negative_fb = data.get('negative_feedback', 0)
                total_fb = positive_fb + negative_fb

                variant_stats.append(VariantStats(
                    variant_name=variant_id,
                    sample_size=count,
                    conversions=data['conversions'],
                    conversion_rate=data['conversions'] / count if count > 0 else None,
                    avg_latency_ms=avg_metric if experiment.success_metric == 'latency_ms' else None,
                    p50_latency_ms=p50 if experiment.success_metric == 'latency_ms' else None,
                    p95_latency_ms=p95 if experiment.success_metric == 'latency_ms' else None,
                    avg_metric_value=avg_metric,
                    std_metric_value=std_metric,
                    positive_feedback=positive_fb,
                    negative_feedback=negative_fb,
                    feedback_rate=positive_fb / total_fb if total_fb > 0 else None,
                ))

            # Statistical significance tests
            significance_tests = []
            if len(variant_stats) >= 2:
                control = variant_stats[0]
                control_metrics = variant_data.get(control.variant_name, {}).get('metrics', [])

                for treatment in variant_stats[1:]:
                    treatment_metrics = variant_data.get(treatment.variant_name, {}).get('metrics', [])

                    if len(control_metrics) >= 2 and len(treatment_metrics) >= 2:
                        try:
                            from scipy import stats as scipy_stats

                            # Perform t-test
                            t_stat, p_value = scipy_stats.ttest_ind(control_metrics, treatment_metrics)

                            # Calculate confidence interval for difference of means
                            control_mean = statistics.mean(control_metrics)
                            treatment_mean = statistics.mean(treatment_metrics)
                            diff = treatment_mean - control_mean

                            pooled_std = (
                                (statistics.stdev(control_metrics) ** 2 / len(control_metrics) +
                                 statistics.stdev(treatment_metrics) ** 2 / len(treatment_metrics)) ** 0.5
                            )
                            ci_margin = 1.96 * pooled_std  # 95% CI

                            relative_diff = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0

                            # Sample size adequacy check (rule of thumb: n >= 30 per group)
                            adequate_sample = len(control_metrics) >= 30 and len(treatment_metrics) >= 30

                            significance_tests.append(StatisticalSignificance(
                                control_variant=control.variant_name,
                                treatment_variant=treatment.variant_name,
                                metric_name=experiment.success_metric,
                                control_mean=round(control_mean, 4),
                                treatment_mean=round(treatment_mean, 4),
                                relative_difference=round(relative_diff, 4),
                                p_value=round(p_value, 4) if p_value else None,
                                confidence_interval_lower=round(diff - ci_margin, 4),
                                confidence_interval_upper=round(diff + ci_margin, 4),
                                is_significant=p_value < 0.05 if p_value else False,
                                test_type="two-sample t-test",
                                sample_size_adequate=adequate_sample,
                                minimum_detectable_effect=None,
                            ))
                        except ImportError:
                            # scipy not available, skip significance testing
                            significance_tests.append(StatisticalSignificance(
                                control_variant=control.variant_name,
                                treatment_variant=treatment.variant_name,
                                metric_name=experiment.success_metric,
                                control_mean=control.avg_metric_value or 0,
                                treatment_mean=treatment.avg_metric_value or 0,
                                relative_difference=0,
                                p_value=None,
                                confidence_interval_lower=None,
                                confidence_interval_upper=None,
                                is_significant=False,
                                test_type="unavailable (scipy required)",
                                sample_size_adequate=False,
                                minimum_detectable_effect=None,
                            ))

            # Determine winning variant
            winning_variant = None
            recommendation = None
            significant_results = [t for t in significance_tests if t.is_significant]
            if significant_results:
                # Find best performing significant result
                best = max(significant_results, key=lambda x: x.relative_difference)
                if best.relative_difference > 0:
                    winning_variant = best.treatment_variant
                    recommendation = f"Consider adopting {best.treatment_variant} - shows {abs(best.relative_difference)*100:.1f}% improvement with p-value {best.p_value}"
                else:
                    winning_variant = best.control_variant
                    recommendation = f"Keep {best.control_variant} - treatment {best.treatment_variant} shows {abs(best.relative_difference)*100:.1f}% worse performance"
            elif variant_stats:
                total_samples = sum(v.sample_size for v in variant_stats)
                if total_samples < 100:
                    recommendation = f"Insufficient data ({total_samples} samples). Continue collecting data for reliable results."
                else:
                    recommendation = "No statistically significant difference detected. Consider extending the experiment or increasing traffic."

            # Calculate duration
            duration_hours = None
            if experiment.started_at:
                end_time = experiment.ended_at or datetime.now(timezone.utc)
                duration_hours = (end_time - experiment.started_at).total_seconds() / 3600

            return ExperimentStatsResponse(
                experiment_id=str(experiment.id),
                experiment_name=experiment.name,
                status=experiment.status.value,
                duration_hours=round(duration_hours, 2) if duration_hours else None,
                total_observations=sum(v.sample_size for v in variant_stats),
                variant_stats=variant_stats,
                significance_tests=significance_tests,
                winning_variant=winning_variant,
                recommendation=recommendation,
                computed_at=datetime.now(timezone.utc).isoformat(),
            )

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to compute experiment stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/{experiment_id}/record", response_model=ExperimentResultResponse)
async def record_experiment_result(experiment_id: str, request: ExperimentResultRecord):
    """Record a metric/result for an experiment.

    Records individual observations for statistical analysis. Typically called
    after each chat interaction to record latency, feedback, or other metrics.

    Args:
        experiment_id: UUID of the experiment
        request: Result data including session_id, variant, metric name/value

    Returns:
        ExperimentResultResponse confirming the recorded result
    """
    try:
        async with get_db_context() as db:
            # Verify experiment exists and is running
            exp_result = await db.execute(
                select(Experiment).where(Experiment.id == uuid.UUID(experiment_id))
            )
            experiment = exp_result.scalar_one_or_none()

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            if experiment.status != ExperimentStatus.RUNNING:
                raise HTTPException(
                    status_code=400,
                    detail=f"Experiment is not running (status: {experiment.status.value})"
                )

            # Create result record
            result = ExperimentResult(
                experiment_id=uuid.UUID(experiment_id),
                variant_id=request.variant_name,
                session_id=request.session_id,
                metric_name=request.metric_name,
                metric_value=request.metric_value,
            )

            db.add(result)
            await db.commit()
            await db.refresh(result)

            return ExperimentResultResponse(
                id=str(result.id),
                experiment_id=experiment_id,
                variant_name=request.variant_name,
                recorded_at=result.recorded_at.isoformat(),
            )

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to record experiment result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiments/assignment", response_model=VariantAssignmentResponse)
async def get_variant_assignment(
    session_id: str = Query(..., description="Session ID to get assignment for"),
    experiment_type: Optional[str] = Query(None, description="Filter by experiment type: model, prompt, rag_config"),
):
    """Get the variant assignment for a session.

    Returns the variant configuration that should be applied for the given session.
    If the session hasn't been assigned yet, assigns it to a variant based on
    traffic allocation and stores the assignment for consistency.

    Args:
        session_id: Session identifier
        experiment_type: Optionally filter to specific experiment type

    Returns:
        VariantAssignmentResponse with variant details and configuration
    """
    try:
        async with get_db_context() as db:
            # Find active (running) experiments
            query = select(Experiment).where(Experiment.status == ExperimentStatus.RUNNING)

            if experiment_type:
                type_map = {
                    "model": ExperimentType.MODEL,
                    "prompt": ExperimentType.PROMPT,
                    "config": ExperimentType.RAG_CONFIG,
                    "rag_config": ExperimentType.RAG_CONFIG,
                    "temperature": ExperimentType.TEMPERATURE,
                }
                if experiment_type.lower() in type_map:
                    query = query.where(Experiment.experiment_type == type_map[experiment_type.lower()])

            # Order by created_at to get oldest (most established) experiment first
            query = query.order_by(Experiment.created_at)
            result = await db.execute(query)
            experiments = result.scalars().all()

            if not experiments:
                raise HTTPException(
                    status_code=404,
                    detail="No active experiments found"
                )

            # Use first matching experiment
            experiment = experiments[0]

            # Check if session already has an assignment
            assignment_result = await db.execute(
                select(ExperimentAssignment).where(
                    and_(
                        ExperimentAssignment.experiment_id == experiment.id,
                        ExperimentAssignment.session_id == session_id
                    )
                )
            )
            assignment = assignment_result.scalar_one_or_none()

            if assignment:
                # Return existing assignment
                variant_id = assignment.variant_id
            else:
                # Assign new variant
                variants = experiment.variants if isinstance(experiment.variants, list) else experiment.variants.get('variants', [])
                variant_id = _assign_variant(variants, experiment.traffic_split, session_id)

                # Store assignment
                new_assignment = ExperimentAssignment(
                    experiment_id=experiment.id,
                    session_id=session_id,
                    variant_id=variant_id,
                )
                db.add(new_assignment)
                await db.commit()

            # Get variant config
            variants = experiment.variants if isinstance(experiment.variants, list) else experiment.variants.get('variants', [])
            variant_config = {}
            is_control = False
            for i, v in enumerate(variants):
                v_id = v.get('id', v.get('name', ''))
                if v_id == variant_id:
                    variant_config = {k: val for k, val in v.items() if k not in ('id', 'name')}
                    is_control = (i == 0)  # First variant is control
                    break

            return VariantAssignmentResponse(
                experiment_id=str(experiment.id),
                experiment_name=experiment.name,
                variant_name=variant_id,
                variant_config=variant_config,
                session_id=session_id,
                is_control=is_control,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get variant assignment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_active_experiment_config(session_id: str, experiment_type: str = "model") -> Optional[dict]:
    """Get active experiment configuration for a session.

    Helper function to be used by chat endpoints to check for active experiments
    and get the appropriate configuration to apply.

    Args:
        session_id: Session identifier
        experiment_type: Type of experiment to check for

    Returns:
        Dictionary with experiment_id, variant_name, and config if active experiment exists
        None if no active experiment
    """
    try:
        async with get_db_context() as db:
            type_map = {
                "model": ExperimentType.MODEL,
                "prompt": ExperimentType.PROMPT,
                "config": ExperimentType.RAG_CONFIG,
                "rag_config": ExperimentType.RAG_CONFIG,
                "temperature": ExperimentType.TEMPERATURE,
            }
            exp_type = type_map.get(experiment_type.lower(), ExperimentType.MODEL)

            # Find running experiment of this type
            result = await db.execute(
                select(Experiment).where(
                    and_(
                        Experiment.status == ExperimentStatus.RUNNING,
                        Experiment.experiment_type == exp_type
                    )
                ).order_by(Experiment.created_at)
            )
            experiment = result.scalar_one_or_none()

            if not experiment:
                return None

            # Check for existing assignment
            assignment_result = await db.execute(
                select(ExperimentAssignment).where(
                    and_(
                        ExperimentAssignment.experiment_id == experiment.id,
                        ExperimentAssignment.session_id == session_id
                    )
                )
            )
            assignment = assignment_result.scalar_one_or_none()

            if assignment:
                variant_id = assignment.variant_id
            else:
                # Assign variant
                variants = experiment.variants if isinstance(experiment.variants, list) else experiment.variants.get('variants', [])
                variant_id = _assign_variant(variants, experiment.traffic_split, session_id)

                # Store assignment
                new_assignment = ExperimentAssignment(
                    experiment_id=experiment.id,
                    session_id=session_id,
                    variant_id=variant_id,
                )
                db.add(new_assignment)
                await db.commit()

            # Get variant config
            variants = experiment.variants if isinstance(experiment.variants, list) else experiment.variants.get('variants', [])
            variant_config = {}
            for v in variants:
                v_id = v.get('id', v.get('name', ''))
                if v_id == variant_id:
                    variant_config = {k: val for k, val in v.items() if k not in ('id', 'name')}
                    break

            return {
                "experiment_id": str(experiment.id),
                "variant_name": variant_id,
                "config": variant_config,
            }

    except Exception as e:
        logger.warning(f"Failed to get active experiment config: {e}")
        return None


async def record_experiment_metrics(
    experiment_id: str,
    session_id: str,
    variant_name: str,
    latency_ms: float,
    feedback_positive: Optional[bool] = None,
) -> None:
    """Record metrics for an experiment asynchronously.

    Helper function to record metrics after a chat interaction.
    Fails silently to not impact the main request flow.

    Args:
        experiment_id: UUID of the experiment
        session_id: Session identifier
        variant_name: Name of the variant used
        latency_ms: Response latency in milliseconds
        feedback_positive: Optional feedback (True=positive, False=negative, None=no feedback)
    """
    try:
        async with get_db_context() as db:
            # Record latency
            latency_result = ExperimentResult(
                experiment_id=uuid.UUID(experiment_id),
                variant_id=variant_name,
                session_id=session_id,
                metric_name="latency_ms",
                metric_value=latency_ms,
            )
            db.add(latency_result)

            # Record feedback if provided
            if feedback_positive is not None:
                feedback_result = ExperimentResult(
                    experiment_id=uuid.UUID(experiment_id),
                    variant_id=variant_name,
                    session_id=session_id,
                    metric_name="feedback_positive" if feedback_positive else "feedback_negative",
                    metric_value=1.0,
                )
                db.add(feedback_result)

            await db.commit()

    except Exception as e:
        logger.warning(f"Failed to record experiment metrics: {e}")


# =====================
# Authentication Endpoints
# =====================

@app.post("/api/auth/register", response_model=UserResponse, status_code=201)
async def register_user(
    request: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account.

    Creates a new user with the provided email, username, and password.
    Password is hashed using bcrypt before storage.

    Args:
        request: UserCreate with email, username, and password
        db: Database session

    Returns:
        UserResponse with created user details (no password)

    Raises:
        HTTPException: 400 if email/username already exists
        HTTPException: 503 if auth is disabled
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not enabled. Set AUTH_ENABLED=true to enable."
        )

    try:
        user = await auth_service.register_user(
            db=db,
            email=request.email,
            username=request.username,
            password=request.password,
        )
        await db.commit()

        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/login", response_model=TokenResponse)
async def login_user(
    request: UserLogin,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate user and create session.

    Validates credentials and creates a new session token.
    Supports login with either email or username.

    Args:
        request: UserLogin with email_or_username and password
        req: FastAPI Request object for client info
        db: Database session

    Returns:
        TokenResponse with session token and expiration

    Raises:
        HTTPException: 401 if credentials are invalid
        HTTPException: 503 if auth is disabled
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not enabled. Set AUTH_ENABLED=true to enable."
        )

    try:
        # Authenticate user
        user = await auth_service.authenticate(
            db=db,
            email_or_username=request.email_or_username,
            password=request.password,
        )

        # Get client info
        client_ip = req.client.host if req.client else None
        user_agent = req.headers.get("user-agent")

        # Create session
        token = await auth_service.create_session(
            db=db,
            user_id=user.id,
            ip_address=client_ip,
            user_agent=user_agent,
            duration_hours=settings.session_expire_hours,
        )
        await db.commit()

        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.session_expire_hours)

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_at=expires_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to login user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/logout", status_code=204)
async def logout_user(
    authorization: str = Header(..., description="Bearer token"),
    db: AsyncSession = Depends(get_db),
):
    """Logout and invalidate session.

    Invalidates the current session token.

    Args:
        authorization: Bearer token header
        db: Database session

    Returns:
        204 No Content on success

    Raises:
        HTTPException: 401 if not authenticated
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not enabled"
        )

    # Extract token from header
    if authorization.lower().startswith("bearer "):
        token = authorization[7:]
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format"
        )

    # Invalidate session
    success = await auth_service.invalidate_session(db, token)
    await db.commit()

    if not success:
        raise HTTPException(status_code=401, detail="Invalid session")

    return None


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get current user information.

    Returns the profile of the currently authenticated user.

    Args:
        current_user: Authenticated user (injected via dependency)

    Returns:
        UserResponse with user details

    Raises:
        HTTPException: 401 if not authenticated
    """
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )


@app.put("/api/auth/me", response_model=UserResponse)
async def update_current_user(
    request: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user profile.

    Updates the profile of the currently authenticated user.
    Only provided fields are updated.

    Args:
        request: UserUpdate with optional email, username, password
        current_user: Authenticated user (injected via dependency)
        db: Database session

    Returns:
        UserResponse with updated user details

    Raises:
        HTTPException: 400 if email/username conflict
        HTTPException: 401 if not authenticated
    """
    try:
        # Re-fetch user in this session
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update email if provided
        if request.email and request.email.lower() != user.email:
            existing_result = await db.execute(
                select(User).where(User.email == request.email.lower())
            )
            existing = existing_result.scalar_one_or_none()
            if existing and existing.id != user.id:
                raise HTTPException(
                    status_code=400,
                    detail="Email address already in use"
                )
            user.email = request.email.lower()

        # Update username if provided
        if request.username and request.username != user.username:
            existing_result = await db.execute(
                select(User).where(User.username == request.username)
            )
            existing = existing_result.scalar_one_or_none()
            if existing and existing.id != user.id:
                raise HTTPException(
                    status_code=400,
                    detail="Username already taken"
                )
            user.username = request.username

        # Update password if provided
        if request.password:
            user.password_hash = hash_password(request.password)

        await db.commit()
        await db.refresh(user)

        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key_endpoint(
    request: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key for the current user.

    Generates a new API key with the specified name and permissions.
    The plain key is returned only once - store it securely.

    Args:
        request: APIKeyCreate with name, permissions, optional expiration
        current_user: Authenticated user (injected via dependency)
        db: Database session

    Returns:
        APIKeyResponse with key details including the plain key (only on creation)

    Raises:
        HTTPException: 400 if key name already exists for user
        HTTPException: 401 if not authenticated
    """
    try:
        # Parse expiration if provided
        expires_at = None
        if request.expires_at:
            try:
                expires_at = datetime.fromisoformat(request.expires_at.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid expires_at format. Use ISO format."
                )

        # Create API key using auth service
        plain_key, api_key = await auth_service.create_api_key(
            db=db,
            user_id=current_user.id,
            name=request.name,
            permissions={"actions": request.permissions},
            expires_at=expires_at,
        )
        await db.commit()

        return APIKeyResponse(
            id=str(api_key.id),
            name=api_key.name,
            permissions=request.permissions,
            created_at=api_key.created_at.isoformat(),
            expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
            last_used_at=None,
            key=plain_key,  # Only returned on creation
            key_prefix=plain_key[:8],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the current user.

    Returns all active API keys owned by the authenticated user.
    Note: The actual key values are not returned, only metadata.

    Args:
        current_user: Authenticated user (injected via dependency)
        db: Database session

    Returns:
        List of APIKeyResponse with key metadata (no key values)

    Raises:
        HTTPException: 401 if not authenticated
    """
    try:
        result = await db.execute(
            select(APIKey)
            .where(
                and_(
                    APIKey.user_id == current_user.id,
                    APIKey.is_active == True,
                )
            )
            .order_by(desc(APIKey.created_at))
        )
        api_keys = result.scalars().all()

        return [
            APIKeyResponse(
                id=str(key.id),
                name=key.name,
                permissions=key.permissions.get("actions", []) if key.permissions else [],
                created_at=key.created_at.isoformat(),
                expires_at=key.expires_at.isoformat() if key.expires_at else None,
                last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
                key=None,  # Never return the actual key
                key_prefix=key.key_hash[:8] if key.key_hash else None,  # Use hash prefix as identifier
            )
            for key in api_keys
        ]

    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/api-keys/{key_id}", status_code=204)
async def revoke_api_key_endpoint(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke (delete) an API key.

    Permanently deactivates the specified API key.
    Only the key owner can revoke their own keys.

    Args:
        key_id: UUID of the API key to revoke
        current_user: Authenticated user (injected via dependency)
        db: Database session

    Returns:
        204 No Content on success

    Raises:
        HTTPException: 404 if key not found
        HTTPException: 403 if key belongs to another user
        HTTPException: 401 if not authenticated
    """
    try:
        # Use auth service to revoke key (verifies ownership)
        await auth_service.revoke_api_key(
            db=db,
            key_id=uuid.UUID(key_id),
            user_id=current_user.id,
        )
        await db.commit()
        return None

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid API key ID format")
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional Prometheus metrics endpoint
if ENABLE_PROMETHEUS:
    try:
        from prometheus_client import make_asgi_app

        # Mount Prometheus metrics endpoint
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    except ImportError:
        pass  # prometheus_client not installed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
