"""OpenTelemetry distributed tracing for the RAG pipeline.

This module provides distributed tracing capabilities using OpenTelemetry,
enabling end-to-end visibility into the RAG pipeline performance.

Features:
- Automatic span creation for key RAG operations
- Configurable exporters (OTLP/Jaeger, console)
- Span attributes for query analysis and debugging
- Request context propagation across services

Usage:
    from app.tracing import get_tracer, trace_rag_operation

    # Get tracer for creating spans
    tracer = get_tracer()

    # Create a span for a custom operation
    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("custom.attribute", "value")
        # ... do work ...

    # Or use the decorator
    @trace_rag_operation("custom.operation")
    def my_function():
        pass

Environment Variables:
    TRACING_ENABLED: Enable/disable tracing (default: false)
    TRACING_EXPORTER: "otlp" or "console" (default: console)
    TRACING_OTLP_ENDPOINT: OTLP gRPC endpoint (default: http://localhost:4317)
    TRACING_SERVICE_NAME: Service name in traces (default: devops-ai-assistant)
    TRACING_SAMPLE_RATE: Sampling ratio 0.0-1.0 (default: 1.0)
"""
import logging
import functools
from contextlib import contextmanager
from typing import Optional, Any, Dict, Callable, TypeVar, ParamSpec

from app.config import settings

logger = logging.getLogger(__name__)

# Type variables for decorator
P = ParamSpec('P')
T = TypeVar('T')

# Global tracer instance (initialized lazily)
_tracer = None
_tracer_provider = None
_initialized = False


def init_tracing(app=None) -> bool:
    """Initialize OpenTelemetry tracing.

    This function sets up the tracer provider, configures the exporter,
    and optionally instruments FastAPI if an app instance is provided.

    Args:
        app: Optional FastAPI application instance for auto-instrumentation

    Returns:
        True if tracing was initialized successfully, False otherwise
    """
    global _tracer, _tracer_provider, _initialized

    if _initialized:
        logger.debug("Tracing already initialized")
        return True

    if not settings.tracing_enabled:
        logger.info("Tracing is disabled (TRACING_ENABLED=false)")
        _initialized = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        # Create resource with service information
        resource = Resource.create({
            SERVICE_NAME: settings.tracing_service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production" if not settings.log_level == "debug" else "development",
        })

        # Create sampler based on configured rate
        sampler = TraceIdRatioBased(settings.tracing_sample_rate)

        # Create tracer provider with resource and sampler
        _tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler,
        )

        # Configure exporter based on settings
        if settings.tracing_exporter == "otlp":
            _configure_otlp_exporter(_tracer_provider)
        else:
            _configure_console_exporter(_tracer_provider)

        # Set the global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Get tracer instance
        _tracer = trace.get_tracer(
            instrumenting_module_name="app.rag",
            tracer_provider=_tracer_provider,
        )

        # Instrument FastAPI if app is provided
        if app is not None:
            _instrument_fastapi(app)

        # Instrument other libraries
        _instrument_libraries()

        _initialized = True
        logger.info(
            f"OpenTelemetry tracing initialized: "
            f"exporter={settings.tracing_exporter}, "
            f"endpoint={settings.tracing_otlp_endpoint if settings.tracing_exporter == 'otlp' else 'stdout'}, "
            f"sample_rate={settings.tracing_sample_rate}"
        )
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry not installed, tracing disabled: {e}")
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _initialized = True
        return False


def _configure_otlp_exporter(provider) -> None:
    """Configure OTLP gRPC exporter for sending traces to collectors like Jaeger."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(
        endpoint=settings.tracing_otlp_endpoint,
        insecure=True,  # Use insecure for local development
    )
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    logger.debug(f"OTLP exporter configured: {settings.tracing_otlp_endpoint}")


def _configure_console_exporter(provider) -> None:
    """Configure console exporter for development/debugging."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    exporter = ConsoleSpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    logger.debug("Console exporter configured")


def _instrument_fastapi(app) -> None:
    """Instrument FastAPI for automatic HTTP span creation."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,health/live,health/ready,metrics",
        )
        logger.debug("FastAPI instrumentation enabled")
    except ImportError:
        logger.warning("FastAPI instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")


def _instrument_libraries() -> None:
    """Instrument common libraries used by the RAG pipeline."""
    # Instrument httpx for outgoing HTTP calls (Ollama, Tavily)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.debug("httpx instrumentation enabled")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to instrument httpx: {e}")

    # Instrument Redis for cache operations
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
        logger.debug("Redis instrumentation enabled")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")


def get_tracer():
    """Get the global tracer instance.

    Returns a tracer that creates spans for tracking operations.
    If tracing is disabled, returns a no-op tracer.

    Returns:
        OpenTelemetry Tracer instance or NoOpTracer if tracing disabled
    """
    global _tracer, _initialized

    if not _initialized:
        init_tracing()

    if _tracer is None:
        # Return a no-op tracer if tracing is disabled
        return NoOpTracer()

    return _tracer


class NoOpSpan:
    """No-operation span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NoOpTracer:
    """No-operation tracer for when tracing is disabled."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
):
    """Create a traced span for an operation.

    This is a convenience context manager for creating spans with
    automatic exception recording and attribute setting.

    Args:
        name: Name of the span (e.g., "rag.retrieval")
        attributes: Optional dictionary of span attributes
        record_exception: Whether to record exceptions in the span

    Yields:
        The span object for adding additional attributes/events

    Example:
        with create_span("rag.retrieval", {"query_length": len(query)}) as span:
            results = vector_store.search(query)
            span.set_attribute("result_count", len(results))
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            if record_exception and hasattr(span, 'record_exception'):
                span.record_exception(e)
            raise


def trace_rag_operation(
    operation_name: str,
    extract_attributes: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for tracing RAG pipeline operations.

    Automatically creates a span around the decorated function with
    optional attribute extraction from function arguments.

    Args:
        operation_name: Name for the span (e.g., "rag.retrieval")
        extract_attributes: Optional function to extract attributes from args/kwargs

    Returns:
        Decorated function with tracing

    Example:
        @trace_rag_operation("rag.rerank", lambda docs, query: {"doc_count": len(docs)})
        def rerank_documents(docs, query):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()

            # Extract attributes if function provided
            attributes = {}
            if extract_attributes:
                try:
                    attributes = extract_attributes(*args, **kwargs)
                except Exception:
                    pass

            with tracer.start_as_current_span(operation_name) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if hasattr(span, 'record_exception'):
                        span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()

            attributes = {}
            if extract_attributes:
                try:
                    attributes = extract_attributes(*args, **kwargs)
                except Exception:
                    pass

            with tracer.start_as_current_span(operation_name) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    if hasattr(span, 'record_exception'):
                        span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def shutdown_tracing() -> None:
    """Shutdown the tracer provider gracefully.

    Should be called during application shutdown to ensure
    all pending spans are exported.
    """
    global _tracer_provider

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("Tracing shutdown complete")
        except Exception as e:
            logger.warning(f"Error during tracing shutdown: {e}")


# Span attribute constants for consistent naming
class SpanAttributes:
    """Constants for span attribute names following OpenTelemetry semantic conventions."""

    # Query attributes
    QUERY_TEXT = "rag.query.text"
    QUERY_LENGTH = "rag.query.length"
    QUERY_HASH = "rag.query.hash"

    # Retrieval attributes
    RETRIEVAL_TOP_K = "rag.retrieval.top_k"
    RETRIEVAL_RESULT_COUNT = "rag.retrieval.result_count"
    RETRIEVAL_MIN_SCORE = "rag.retrieval.min_score"
    RETRIEVAL_AVG_SCORE = "rag.retrieval.avg_score"
    RETRIEVAL_MAX_SCORE = "rag.retrieval.max_score"
    RETRIEVAL_TIME_MS = "rag.retrieval.time_ms"
    RETRIEVAL_HYBRID = "rag.retrieval.hybrid_search"
    RETRIEVAL_CACHE_HIT = "rag.retrieval.cache_hit"

    # Reranker attributes
    RERANK_ENABLED = "rag.rerank.enabled"
    RERANK_MODEL = "rag.rerank.model"
    RERANK_INPUT_COUNT = "rag.rerank.input_count"
    RERANK_OUTPUT_COUNT = "rag.rerank.output_count"
    RERANK_TIME_MS = "rag.rerank.time_ms"
    RERANK_AVG_SCORE = "rag.rerank.avg_score"

    # HyDE attributes
    HYDE_ENABLED = "rag.hyde.enabled"
    HYDE_USED = "rag.hyde.used"
    HYDE_TIME_MS = "rag.hyde.time_ms"
    HYDE_SKIP_REASON = "rag.hyde.skip_reason"

    # Web search attributes
    WEB_SEARCH_ENABLED = "rag.web_search.enabled"
    WEB_SEARCH_TRIGGERED = "rag.web_search.triggered"
    WEB_SEARCH_REASON = "rag.web_search.trigger_reason"
    WEB_SEARCH_RESULT_COUNT = "rag.web_search.result_count"
    WEB_SEARCH_TIME_MS = "rag.web_search.time_ms"

    # LLM generation attributes
    LLM_MODEL = "rag.llm.model"
    LLM_TEMPERATURE = "rag.llm.temperature"
    LLM_MAX_TOKENS = "rag.llm.max_tokens"
    LLM_CONTEXT_LENGTH = "rag.llm.context_length"
    LLM_RESPONSE_LENGTH = "rag.llm.response_length"
    LLM_TIME_MS = "rag.llm.time_ms"

    # Overall pipeline attributes
    PIPELINE_TOTAL_TIME_MS = "rag.pipeline.total_time_ms"
    PIPELINE_CONTEXT_USED = "rag.pipeline.context_used"
    PIPELINE_SOURCE_COUNT = "rag.pipeline.source_count"
