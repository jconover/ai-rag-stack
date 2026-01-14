"""Retrieval metrics logging and Prometheus instrumentation for RAG observability.

This module provides structured logging for retrieval metrics and optional
Prometheus metrics for monitoring retrieval quality and performance.

Metrics logged:
- Retrieval scores (similarity scores from vector search)
- Query latency
- Score distribution statistics
- Top-k configuration

Log format: JSON lines for easy parsing and aggregation.
"""

import json
import hashlib
import time
import statistics
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import os

# Configure structured logger for retrieval metrics
METRICS_LOG_PATH = os.getenv("RETRIEVAL_METRICS_LOG", "/data/logs/retrieval_metrics.jsonl")
ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS_METRICS", "false").lower() == "true"

# Setup file handler for metrics logging
metrics_logger = logging.getLogger("retrieval_metrics")
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False  # Prevent duplicate logging

# Try to set up file handler, fall back to /tmp if permissions fail
try:
    Path(METRICS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(METRICS_LOG_PATH, mode='a')
except (PermissionError, OSError):
    # Fall back to /tmp if /data/logs is not writable
    METRICS_LOG_PATH = "/tmp/retrieval_metrics.jsonl"
    file_handler = logging.FileHandler(METRICS_LOG_PATH, mode='a')
    logging.warning(f"Could not write to original path, using fallback: {METRICS_LOG_PATH}")

file_handler.setFormatter(logging.Formatter('%(message)s'))  # Raw JSON output
metrics_logger.addHandler(file_handler)

# Console handler for development visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[RETRIEVAL] %(message)s'))
console_handler.setLevel(logging.DEBUG)
metrics_logger.addHandler(console_handler)


@dataclass
class IRQualityMetrics:
    """Information retrieval quality metrics for evaluating ranking.

    These metrics measure how well the retrieval system ranks relevant
    documents. They are computed from similarity/rerank scores and provide
    insight into ranking quality without requiring ground-truth relevance labels.

    Attributes:
        ndcg_at_5: Normalized Discounted Cumulative Gain at K=5
        mrr: Mean Reciprocal Rank (position of first highly relevant result)
        recall_at_k: Proportion of results with positive relevance signal
        precision_at_k: Proportion of results above quality threshold
    """
    ndcg_at_5: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0

    @staticmethod
    def compute_dcg(scores: List[float], k: int = 5) -> float:
        """Compute Discounted Cumulative Gain using scores as relevance grades.

        Args:
            scores: List of relevance scores (similarity or rerank scores)
            k: Number of top results to consider

        Returns:
            DCG value where higher is better
        """
        dcg = 0.0
        for i, score in enumerate(scores[:k]):
            # Normalize score to [0, 1] range as relevance grade
            # Handle both positive [0, 1] scores and potentially negative scores
            rel = max(0.0, min(1.0, score)) if score >= 0 else max(0.0, (score + 5) / 10)
            dcg += rel / math.log2(i + 2)  # i+2 because position is 1-indexed
        return dcg

    @staticmethod
    def compute_ndcg(scores: List[float], k: int = 5) -> float:
        """Compute Normalized DCG comparing actual vs ideal ranking.

        NDCG measures ranking quality by comparing the actual ranking to
        the ideal ranking (sorted by score). A value of 1.0 means the
        results are in perfect order by relevance.

        Args:
            scores: List of relevance scores in retrieval order
            k: Number of top results to consider

        Returns:
            NDCG value in [0, 1] where 1.0 is perfect ranking
        """
        dcg = IRQualityMetrics.compute_dcg(scores, k)
        ideal_scores = sorted(scores, reverse=True)
        idcg = IRQualityMetrics.compute_dcg(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_mrr(scores: List[float], threshold: float = 0.5) -> float:
        """Compute Mean Reciprocal Rank (position of first relevant result).

        MRR measures how quickly the system returns a highly relevant result.
        A value of 1.0 means the first result is relevant; 0.5 means the
        second result is the first relevant one, etc.

        Args:
            scores: List of relevance scores in retrieval order
            threshold: Minimum score to consider a result "relevant"

        Returns:
            MRR value in (0, 1] or 0.0 if no relevant results
        """
        for i, score in enumerate(scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    @classmethod
    def from_scores(
        cls,
        similarity_scores: List[float],
        rerank_scores: Optional[List[float]] = None,
        k: int = 5
    ) -> "IRQualityMetrics":
        """Compute all IR metrics from retrieval scores.

        Uses rerank scores if available, otherwise falls back to similarity scores.
        This allows evaluation both before and after reranking.

        Args:
            similarity_scores: Raw similarity scores from vector search
            rerank_scores: Optional reranked scores (preferred if available)
            k: Number of top results to evaluate

        Returns:
            IRQualityMetrics instance with all metrics computed
        """
        scores = rerank_scores if rerank_scores else similarity_scores
        if not scores:
            return cls()

        return cls(
            ndcg_at_5=cls.compute_ndcg(scores, k),
            mrr=cls.compute_mrr(scores),
            recall_at_k=sum(1 for s in scores[:k] if s > 0) / k if scores else 0.0,
            precision_at_k=sum(1 for s in scores[:k] if s > 0.3) / k if scores else 0.0,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RetrievalMetrics:
    """Structured retrieval metrics data class."""
    timestamp: str
    query_hash: str
    query_preview: str  # First 100 chars of query for debugging
    model: str
    top_k: int
    num_results: int
    scores: List[float]
    latency_ms: float
    score_threshold: Optional[float]
    filtered_count: int  # Number filtered by threshold

    # Score distribution statistics
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_mean: Optional[float] = None
    score_std: Optional[float] = None
    score_median: Optional[float] = None

    # Score buckets for quick analysis
    scores_above_0_8: int = 0
    scores_0_6_to_0_8: int = 0
    scores_0_4_to_0_6: int = 0
    scores_below_0_4: int = 0

    # IR quality metrics
    ir_metrics: Optional[Dict[str, float]] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_scores(
        cls,
        query: str,
        model: str,
        scores: List[float],
        top_k: int,
        latency_ms: float,
        score_threshold: Optional[float] = None,
        filtered_count: int = 0,
        rerank_scores: Optional[List[float]] = None
    ) -> "RetrievalMetrics":
        """Create metrics from raw score data with computed statistics.

        Args:
            query: The search query
            model: LLM model being used
            scores: List of similarity scores from retrieval
            top_k: Number of results requested
            latency_ms: Retrieval latency in milliseconds
            score_threshold: Optional threshold used for filtering
            filtered_count: Number of results filtered by threshold
            rerank_scores: Optional reranked scores for IR metrics computation

        Returns:
            RetrievalMetrics object with computed statistics and IR metrics
        """

        # Generate deterministic query hash for correlation
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        # Compute score statistics
        score_stats = {}
        if scores:
            score_stats = {
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": statistics.mean(scores),
                "score_median": statistics.median(scores),
            }
            if len(scores) > 1:
                score_stats["score_std"] = statistics.stdev(scores)
            else:
                score_stats["score_std"] = 0.0

        # Compute score buckets
        buckets = {
            "scores_above_0_8": sum(1 for s in scores if s >= 0.8),
            "scores_0_6_to_0_8": sum(1 for s in scores if 0.6 <= s < 0.8),
            "scores_0_4_to_0_6": sum(1 for s in scores if 0.4 <= s < 0.6),
            "scores_below_0_4": sum(1 for s in scores if s < 0.4),
        }

        # Compute IR quality metrics
        ir_quality = IRQualityMetrics.from_scores(
            similarity_scores=scores,
            rerank_scores=rerank_scores,
            k=top_k
        )

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_hash=query_hash,
            query_preview=query[:100] + "..." if len(query) > 100 else query,
            model=model,
            top_k=top_k,
            num_results=len(scores),
            scores=scores,
            latency_ms=round(latency_ms, 2),
            score_threshold=score_threshold,
            filtered_count=filtered_count,
            ir_metrics=ir_quality.to_dict(),
            **score_stats,
            **buckets
        )


class RetrievalMetricsLogger:
    """Logger for retrieval metrics with optional Prometheus integration."""

    def __init__(self):
        self.prometheus_enabled = ENABLE_PROMETHEUS
        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics if enabled."""
        if not self.prometheus_enabled:
            self.retrieval_score_histogram = None
            self.retrieval_latency_histogram = None
            self.query_counter = None
            self.score_bucket_counter = None
            self.ndcg_histogram = None
            self.mrr_histogram = None
            self.precision_histogram = None
            self.recall_histogram = None
            return

        try:
            from prometheus_client import Histogram, Counter, REGISTRY

            # Check if metrics already exist (for hot-reload scenarios)
            try:
                # Retrieval score histogram
                self.retrieval_score_histogram = Histogram(
                    'rag_retrieval_score',
                    'Distribution of retrieval similarity scores',
                    ['model', 'top_k'],
                    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                )

                # Retrieval latency histogram
                self.retrieval_latency_histogram = Histogram(
                    'rag_retrieval_latency_ms',
                    'Retrieval latency in milliseconds',
                    ['model'],
                    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
                )

                # Query counter
                self.query_counter = Counter(
                    'rag_queries_total',
                    'Total number of RAG queries',
                    ['model', 'has_results']
                )

                # Score bucket counter for quick analysis
                self.score_bucket_counter = Counter(
                    'rag_score_bucket_total',
                    'Count of scores by quality bucket',
                    ['bucket']
                )

                # IR quality metrics histograms
                self.ndcg_histogram = Histogram(
                    'rag_ir_ndcg',
                    'Normalized Discounted Cumulative Gain at K',
                    ['model'],
                    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                )

                self.mrr_histogram = Histogram(
                    'rag_ir_mrr',
                    'Mean Reciprocal Rank',
                    ['model'],
                    buckets=[0.0, 0.1, 0.2, 0.25, 0.33, 0.5, 1.0]
                )

                self.precision_histogram = Histogram(
                    'rag_ir_precision_at_k',
                    'Precision at K',
                    ['model'],
                    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                )

                self.recall_histogram = Histogram(
                    'rag_ir_recall_at_k',
                    'Recall at K',
                    ['model'],
                    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                )

            except ValueError:
                # Metrics already registered, get existing ones
                self.retrieval_score_histogram = REGISTRY._names_to_collectors.get(
                    'rag_retrieval_score'
                )
                self.retrieval_latency_histogram = REGISTRY._names_to_collectors.get(
                    'rag_retrieval_latency_ms'
                )
                self.query_counter = REGISTRY._names_to_collectors.get(
                    'rag_queries_total'
                )
                self.score_bucket_counter = REGISTRY._names_to_collectors.get(
                    'rag_score_bucket_total'
                )
                self.ndcg_histogram = REGISTRY._names_to_collectors.get(
                    'rag_ir_ndcg'
                )
                self.mrr_histogram = REGISTRY._names_to_collectors.get(
                    'rag_ir_mrr'
                )
                self.precision_histogram = REGISTRY._names_to_collectors.get(
                    'rag_ir_precision_at_k'
                )
                self.recall_histogram = REGISTRY._names_to_collectors.get(
                    'rag_ir_recall_at_k'
                )

        except ImportError:
            metrics_logger.warning("prometheus_client not installed. Prometheus metrics disabled.")
            self.prometheus_enabled = False
            self.retrieval_score_histogram = None
            self.retrieval_latency_histogram = None
            self.query_counter = None
            self.score_bucket_counter = None
            self.ndcg_histogram = None
            self.mrr_histogram = None
            self.precision_histogram = None
            self.recall_histogram = None

    def log_retrieval(
        self,
        query: str,
        model: str,
        scores: List[float],
        top_k: int,
        latency_ms: float,
        score_threshold: Optional[float] = None,
        filtered_count: int = 0,
        rerank_scores: Optional[List[float]] = None
    ) -> RetrievalMetrics:
        """Log retrieval metrics to file and optionally Prometheus.

        Args:
            query: The search query
            model: LLM model being used
            scores: List of similarity scores from retrieval
            top_k: Number of results requested
            latency_ms: Retrieval latency in milliseconds
            score_threshold: Optional threshold used for filtering
            filtered_count: Number of results filtered by threshold
            rerank_scores: Optional reranked scores for IR metrics computation

        Returns:
            RetrievalMetrics object with computed statistics and IR quality metrics
        """

        # Create structured metrics
        metrics = RetrievalMetrics.from_scores(
            query=query,
            model=model,
            scores=scores,
            top_k=top_k,
            latency_ms=latency_ms,
            score_threshold=score_threshold,
            filtered_count=filtered_count,
            rerank_scores=rerank_scores
        )

        # Log to JSON lines file
        metrics_logger.info(metrics.to_json())

        # Record Prometheus metrics if enabled
        if self.prometheus_enabled:
            self._record_prometheus_metrics(metrics)

        return metrics

    def _record_prometheus_metrics(self, metrics: RetrievalMetrics):
        """Record metrics to Prometheus."""
        if not self.prometheus_enabled:
            return

        top_k_label = str(metrics.top_k)

        # Record each score in histogram
        for score in metrics.scores:
            if self.retrieval_score_histogram:
                self.retrieval_score_histogram.labels(
                    model=metrics.model,
                    top_k=top_k_label
                ).observe(score)

        # Record latency
        if self.retrieval_latency_histogram:
            self.retrieval_latency_histogram.labels(
                model=metrics.model
            ).observe(metrics.latency_ms)

        # Increment query counter
        if self.query_counter:
            self.query_counter.labels(
                model=metrics.model,
                has_results=str(metrics.num_results > 0).lower()
            ).inc()

        # Record score bucket counts
        if self.score_bucket_counter:
            self.score_bucket_counter.labels(bucket="above_0.8").inc(metrics.scores_above_0_8)
            self.score_bucket_counter.labels(bucket="0.6_to_0.8").inc(metrics.scores_0_6_to_0_8)
            self.score_bucket_counter.labels(bucket="0.4_to_0.6").inc(metrics.scores_0_4_to_0_6)
            self.score_bucket_counter.labels(bucket="below_0.4").inc(metrics.scores_below_0_4)

        # Record IR quality metrics
        if metrics.ir_metrics:
            if self.ndcg_histogram:
                self.ndcg_histogram.labels(
                    model=metrics.model
                ).observe(metrics.ir_metrics.get('ndcg_at_5', 0.0))

            if self.mrr_histogram:
                self.mrr_histogram.labels(
                    model=metrics.model
                ).observe(metrics.ir_metrics.get('mrr', 0.0))

            if self.precision_histogram:
                self.precision_histogram.labels(
                    model=metrics.model
                ).observe(metrics.ir_metrics.get('precision_at_k', 0.0))

            if self.recall_histogram:
                self.recall_histogram.labels(
                    model=metrics.model
                ).observe(metrics.ir_metrics.get('recall_at_k', 0.0))


class RetrievalTimer:
    """Context manager for timing retrieval operations."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "RetrievalTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000


# Singleton instance for use across the application
retrieval_metrics_logger = RetrievalMetricsLogger()


def get_metrics_summary(log_path: str = METRICS_LOG_PATH, last_n: int = 100) -> Dict[str, Any]:
    """Read recent metrics and compute summary statistics.

    Useful for debugging and monitoring dashboard data.

    Args:
        log_path: Path to metrics log file
        last_n: Number of recent entries to analyze

    Returns:
        Summary statistics dictionary
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Get last N entries
        recent_lines = lines[-last_n:] if len(lines) > last_n else lines

        all_scores = []
        all_latencies = []
        queries_with_results = 0
        queries_without_results = 0

        # IR quality metrics aggregation
        all_ndcg = []
        all_mrr = []
        all_precision = []
        all_recall = []

        for line in recent_lines:
            try:
                entry = json.loads(line.strip())
                all_scores.extend(entry.get('scores', []))
                all_latencies.append(entry.get('latency_ms', 0))

                if entry.get('num_results', 0) > 0:
                    queries_with_results += 1
                else:
                    queries_without_results += 1

                # Collect IR metrics if present
                ir_metrics = entry.get('ir_metrics')
                if ir_metrics:
                    all_ndcg.append(ir_metrics.get('ndcg_at_5', 0.0))
                    all_mrr.append(ir_metrics.get('mrr', 0.0))
                    all_precision.append(ir_metrics.get('precision_at_k', 0.0))
                    all_recall.append(ir_metrics.get('recall_at_k', 0.0))

            except json.JSONDecodeError:
                continue

        summary = {
            "analyzed_queries": len(recent_lines),
            "queries_with_results": queries_with_results,
            "queries_without_results": queries_without_results,
            "total_scores_analyzed": len(all_scores),
        }

        if all_scores:
            summary.update({
                "score_mean": round(statistics.mean(all_scores), 4),
                "score_median": round(statistics.median(all_scores), 4),
                "score_min": round(min(all_scores), 4),
                "score_max": round(max(all_scores), 4),
                "score_std": round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0,
            })

        if all_latencies:
            summary.update({
                "latency_mean_ms": round(statistics.mean(all_latencies), 2),
                "latency_median_ms": round(statistics.median(all_latencies), 2),
                "latency_p95_ms": round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 2),
                "latency_max_ms": round(max(all_latencies), 2),
            })

        # Add IR quality metrics summary
        if all_ndcg:
            summary.update({
                "ir_metrics": {
                    "ndcg_mean": round(statistics.mean(all_ndcg), 4),
                    "ndcg_median": round(statistics.median(all_ndcg), 4),
                    "mrr_mean": round(statistics.mean(all_mrr), 4),
                    "mrr_median": round(statistics.median(all_mrr), 4),
                    "precision_at_k_mean": round(statistics.mean(all_precision), 4),
                    "precision_at_k_median": round(statistics.median(all_precision), 4),
                    "recall_at_k_mean": round(statistics.mean(all_recall), 4),
                    "recall_at_k_median": round(statistics.median(all_recall), 4),
                }
            })

        return summary

    except FileNotFoundError:
        return {"error": "Metrics log file not found", "path": log_path}
    except Exception as e:
        return {"error": str(e)}
