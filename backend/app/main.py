"""FastAPI application for DevOps AI Assistant"""
import uuid
import os
import subprocess
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
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
    QueryAnalyticsSummary
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
from app.db_models import QueryLog, Feedback

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
    total_time_ms: float
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
async def chat(request: ChatRequest):
    """Chat with the AI assistant"""

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Save user message
    save_message(session_id, "user", request.message)

    # Track timing for query logging
    start_time = time.time()

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
            percentile_query = text("""
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
                    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
                FROM query_logs
                WHERE latency_ms IS NOT NULL
            """)
            if conditions:
                # For simplicity, re-run with date filter in raw SQL if needed
                pass

            percentile_result = await db.execute(percentile_query)
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
