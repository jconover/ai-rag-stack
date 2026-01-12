"""Real-time analytics collector for operational visibility.

Provides in-memory metrics collection with sliding window aggregation for:
- Request rate and throughput
- Response latency distribution
- Error rate tracking
- Cache hit rate monitoring
- Top queries (anonymized)
- Model usage distribution
- Retrieval quality metrics

All metrics are stored in memory with configurable time windows.
No external database required.
"""

import hashlib
import time
import threading
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Deque, Tuple
import logging

logger = logging.getLogger(__name__)

# Default time windows in seconds
DEFAULT_SHORT_WINDOW = 300  # 5 minutes
DEFAULT_LONG_WINDOW = 3600  # 1 hour
MAX_TOP_QUERIES = 50  # Maximum number of top queries to track


@dataclass
class RequestRecord:
    """Record of a single request for metrics aggregation."""
    timestamp: float
    latency_ms: float
    model: str
    status_code: int
    endpoint: str
    is_error: bool = False
    cache_hit: Optional[bool] = None
    avg_similarity_score: Optional[float] = None
    query_hash: Optional[str] = None


@dataclass
class QueryRecord:
    """Anonymized query record for top queries tracking."""
    query_hash: str
    timestamp: float
    count: int = 1
    last_seen: float = 0


class SlidingWindowCounter:
    """Thread-safe sliding window counter for rate calculations."""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._records: Deque[float] = deque()
        self._lock = threading.Lock()

    def add(self, timestamp: Optional[float] = None):
        """Add a timestamp to the counter."""
        ts = timestamp or time.time()
        with self._lock:
            self._records.append(ts)
            self._prune()

    def _prune(self):
        """Remove timestamps outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._records and self._records[0] < cutoff:
            self._records.popleft()

    def count(self) -> int:
        """Get count of events in the window."""
        with self._lock:
            self._prune()
            return len(self._records)

    def rate_per_minute(self) -> float:
        """Get rate per minute based on current window."""
        with self._lock:
            self._prune()
            if not self._records:
                return 0.0
            # Calculate actual time span covered
            if len(self._records) < 2:
                return len(self._records)  # 1 request in window
            time_span = self._records[-1] - self._records[0]
            if time_span <= 0:
                return len(self._records)
            # Limit time_span to window size
            time_span = min(time_span, self.window_seconds)
            return (len(self._records) / time_span) * 60


class MetricsCollector:
    """Collects and aggregates real-time metrics for the API.

    Thread-safe implementation using locks for concurrent access.
    All data is stored in memory with automatic pruning based on time windows.
    """

    def __init__(
        self,
        short_window_seconds: int = DEFAULT_SHORT_WINDOW,
        long_window_seconds: int = DEFAULT_LONG_WINDOW,
    ):
        self.short_window = short_window_seconds
        self.long_window = long_window_seconds

        # Request records for detailed analysis
        self._requests: Deque[RequestRecord] = deque()
        self._requests_lock = threading.Lock()

        # Counters for quick rate calculations
        self._request_counter = SlidingWindowCounter(short_window_seconds)
        self._error_counter = SlidingWindowCounter(short_window_seconds)
        self._cache_hit_counter = SlidingWindowCounter(short_window_seconds)
        self._cache_miss_counter = SlidingWindowCounter(short_window_seconds)

        # Top queries tracking (anonymized)
        self._query_counts: Dict[str, QueryRecord] = {}
        self._query_lock = threading.Lock()

        # Model usage tracking
        self._model_usage: Dict[str, int] = {}
        self._model_lock = threading.Lock()

        # Latency tracking for percentile calculations
        self._latencies: Deque[Tuple[float, float]] = deque()  # (timestamp, latency_ms)
        self._latencies_lock = threading.Lock()

        # Retrieval quality scores
        self._similarity_scores: Deque[Tuple[float, float]] = deque()  # (timestamp, score)
        self._scores_lock = threading.Lock()

        logger.info(
            f"MetricsCollector initialized with windows: "
            f"short={short_window_seconds}s, long={long_window_seconds}s"
        )

    def record_request(
        self,
        latency_ms: float,
        model: str,
        status_code: int,
        endpoint: str,
        cache_hit: Optional[bool] = None,
        avg_similarity_score: Optional[float] = None,
        query: Optional[str] = None,
    ):
        """Record a completed request with its metrics.

        Args:
            latency_ms: Request latency in milliseconds
            model: LLM model used
            status_code: HTTP status code
            endpoint: API endpoint path
            cache_hit: Whether embedding cache was hit
            avg_similarity_score: Average retrieval similarity score
            query: Original query (will be hashed/anonymized)
        """
        now = time.time()
        is_error = status_code >= 400

        # Generate query hash for anonymization
        query_hash = None
        if query:
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]

        record = RequestRecord(
            timestamp=now,
            latency_ms=latency_ms,
            model=model,
            status_code=status_code,
            endpoint=endpoint,
            is_error=is_error,
            cache_hit=cache_hit,
            avg_similarity_score=avg_similarity_score,
            query_hash=query_hash,
        )

        # Update request records
        with self._requests_lock:
            self._requests.append(record)
            self._prune_requests()

        # Update counters
        self._request_counter.add(now)
        if is_error:
            self._error_counter.add(now)

        if cache_hit is not None:
            if cache_hit:
                self._cache_hit_counter.add(now)
            else:
                self._cache_miss_counter.add(now)

        # Update latency tracking
        with self._latencies_lock:
            self._latencies.append((now, latency_ms))
            self._prune_latencies()

        # Update similarity scores
        if avg_similarity_score is not None:
            with self._scores_lock:
                self._similarity_scores.append((now, avg_similarity_score))
                self._prune_scores()

        # Update model usage
        with self._model_lock:
            self._model_usage[model] = self._model_usage.get(model, 0) + 1

        # Track query for top queries
        if query_hash:
            self._track_query(query_hash, now)

    def _prune_requests(self):
        """Remove old request records outside the long window."""
        cutoff = time.time() - self.long_window
        while self._requests and self._requests[0].timestamp < cutoff:
            self._requests.popleft()

    def _prune_latencies(self):
        """Remove old latency records outside the short window."""
        cutoff = time.time() - self.short_window
        while self._latencies and self._latencies[0][0] < cutoff:
            self._latencies.popleft()

    def _prune_scores(self):
        """Remove old score records outside the short window."""
        cutoff = time.time() - self.short_window
        while self._similarity_scores and self._similarity_scores[0][0] < cutoff:
            self._similarity_scores.popleft()

    def _track_query(self, query_hash: str, timestamp: float):
        """Track query for top queries (with automatic pruning)."""
        with self._query_lock:
            if query_hash in self._query_counts:
                self._query_counts[query_hash].count += 1
                self._query_counts[query_hash].last_seen = timestamp
            else:
                self._query_counts[query_hash] = QueryRecord(
                    query_hash=query_hash,
                    timestamp=timestamp,
                    count=1,
                    last_seen=timestamp,
                )

            # Prune old queries outside long window
            cutoff = time.time() - self.long_window
            expired = [
                k for k, v in self._query_counts.items()
                if v.last_seen < cutoff
            ]
            for k in expired:
                del self._query_counts[k]

            # Limit total tracked queries
            if len(self._query_counts) > MAX_TOP_QUERIES * 2:
                # Keep only top queries by count
                sorted_queries = sorted(
                    self._query_counts.items(),
                    key=lambda x: x[1].count,
                    reverse=True
                )
                self._query_counts = dict(sorted_queries[:MAX_TOP_QUERIES])

    def get_request_rate(self) -> float:
        """Get requests per minute over the short window."""
        return self._request_counter.rate_per_minute()

    def get_error_rate(self) -> float:
        """Get error rate as percentage over the short window."""
        total = self._request_counter.count()
        if total == 0:
            return 0.0
        errors = self._error_counter.count()
        return (errors / total) * 100

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage over the short window."""
        hits = self._cache_hit_counter.count()
        misses = self._cache_miss_counter.count()
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100

    def get_latency_stats(self) -> Dict[str, Optional[float]]:
        """Get latency statistics over the short window."""
        with self._latencies_lock:
            self._prune_latencies()
            if not self._latencies:
                return {
                    "avg_ms": None,
                    "median_ms": None,
                    "p95_ms": None,
                    "p99_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                }

            latencies = [l[1] for l in self._latencies]
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            return {
                "avg_ms": round(statistics.mean(latencies), 2),
                "median_ms": round(statistics.median(latencies), 2),
                "p95_ms": round(sorted_latencies[int(n * 0.95)] if n > 0 else 0, 2),
                "p99_ms": round(sorted_latencies[int(n * 0.99)] if n > 0 else 0, 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
            }

    def get_similarity_score_stats(self) -> Dict[str, Optional[float]]:
        """Get retrieval similarity score statistics over the short window."""
        with self._scores_lock:
            self._prune_scores()
            if not self._similarity_scores:
                return {
                    "avg_score": None,
                    "median_score": None,
                    "min_score": None,
                    "max_score": None,
                }

            scores = [s[1] for s in self._similarity_scores]

            return {
                "avg_score": round(statistics.mean(scores), 4),
                "median_score": round(statistics.median(scores), 4),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
            }

    def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top queries by count (anonymized) over the long window."""
        with self._query_lock:
            # Prune old queries
            cutoff = time.time() - self.long_window
            self._query_counts = {
                k: v for k, v in self._query_counts.items()
                if v.last_seen >= cutoff
            }

            # Sort by count and return top N
            sorted_queries = sorted(
                self._query_counts.values(),
                key=lambda x: x.count,
                reverse=True
            )[:limit]

            return [
                {
                    "query_hash": q.query_hash,
                    "count": q.count,
                    "last_seen": datetime.fromtimestamp(
                        q.last_seen, tz=timezone.utc
                    ).isoformat(),
                }
                for q in sorted_queries
            ]

    def get_model_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get model usage distribution."""
        with self._model_lock:
            total = sum(self._model_usage.values())
            if total == 0:
                return {}

            return {
                model: {
                    "count": count,
                    "percentage": round((count / total) * 100, 2),
                }
                for model, count in sorted(
                    self._model_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }

    def get_realtime_analytics(self) -> Dict[str, Any]:
        """Get comprehensive real-time analytics snapshot.

        Returns all metrics in a single call for the analytics endpoint.
        """
        now = datetime.now(timezone.utc)

        latency_stats = self.get_latency_stats()
        score_stats = self.get_similarity_score_stats()

        return {
            "timestamp": now.isoformat(),
            "window_seconds": self.short_window,
            "request_metrics": {
                "requests_per_minute": round(self.get_request_rate(), 2),
                "total_requests_in_window": self._request_counter.count(),
                "error_rate_percent": round(self.get_error_rate(), 2),
            },
            "latency_metrics": latency_stats,
            "cache_metrics": {
                "hit_rate_percent": round(self.get_cache_hit_rate(), 2),
                "hits_in_window": self._cache_hit_counter.count(),
                "misses_in_window": self._cache_miss_counter.count(),
            },
            "retrieval_quality": score_stats,
            "model_usage": self.get_model_usage(),
            "top_queries": self.get_top_queries(limit=10),
        }

    def reset(self):
        """Reset all metrics. Useful for testing."""
        with self._requests_lock:
            self._requests.clear()
        with self._latencies_lock:
            self._latencies.clear()
        with self._scores_lock:
            self._similarity_scores.clear()
        with self._query_lock:
            self._query_counts.clear()
        with self._model_lock:
            self._model_usage.clear()

        self._request_counter = SlidingWindowCounter(self.short_window)
        self._error_counter = SlidingWindowCounter(self.short_window)
        self._cache_hit_counter = SlidingWindowCounter(self.short_window)
        self._cache_miss_counter = SlidingWindowCounter(self.short_window)

        logger.info("MetricsCollector reset")


# Global metrics collector singleton
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector():
    """Reset the global metrics collector. Useful for testing."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.reset()
