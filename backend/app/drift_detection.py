"""Model drift detection for embedding and retrieval quality.

This module tracks embedding similarity score distributions over time to detect
model drift that could degrade RAG performance. It uses Redis for storing
historical statistics and provides statistical tests to identify significant
distribution shifts.

Drift detection is critical for production ML systems because:
- Embedding models can degrade silently over time
- Data distribution changes affect retrieval quality
- Early detection enables proactive retraining or model updates

Usage:
    from app.drift_detection import drift_detector

    # Record scores during retrieval
    drift_detector.record_scores([0.85, 0.72, 0.68, 0.55, 0.42])

    # Check for drift periodically
    status = await drift_detector.check_drift()
"""
import logging
import json
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DriftStatus(str, Enum):
    """Drift detection status states."""
    STABLE = "stable"
    DRIFT_DETECTED = "drift_detected"
    WARNING = "warning"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_BASELINE = "no_baseline"
    ERROR = "error"


@dataclass
class DriftMetrics:
    """Statistical metrics for a score distribution window.

    Captures the key statistical properties of similarity scores
    for comparison between time windows.
    """
    mean_score: float
    std_score: float
    p25: float  # 25th percentile
    p50: float  # 50th percentile (median)
    p75: float  # 75th percentile
    min_score: float
    max_score: float
    sample_count: int
    timestamp: str
    window_hours: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftMetrics":
        """Create from dictionary."""
        return cls(
            mean_score=data["mean_score"],
            std_score=data["std_score"],
            p25=data["p25"],
            p50=data["p50"],
            p75=data["p75"],
            min_score=data["min_score"],
            max_score=data["max_score"],
            sample_count=data["sample_count"],
            timestamp=data["timestamp"],
            window_hours=data.get("window_hours", 24),
        )


@dataclass
class DriftCheckResult:
    """Result of a drift detection check.

    Provides detailed information about detected drift including
    the magnitude of changes across different statistical measures.
    """
    status: DriftStatus
    mean_shift_pct: Optional[float] = None
    std_shift_pct: Optional[float] = None
    median_shift_pct: Optional[float] = None
    current_metrics: Optional[DriftMetrics] = None
    baseline_metrics: Optional[DriftMetrics] = None
    message: str = ""
    checked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        result = {
            "status": self.status.value,
            "message": self.message,
            "checked_at": self.checked_at,
        }

        if self.mean_shift_pct is not None:
            result["mean_shift_pct"] = round(self.mean_shift_pct, 2)
        if self.std_shift_pct is not None:
            result["std_shift_pct"] = round(self.std_shift_pct, 2)
        if self.median_shift_pct is not None:
            result["median_shift_pct"] = round(self.median_shift_pct, 2)

        if self.current_metrics:
            result["current"] = {
                "mean_score": round(self.current_metrics.mean_score, 4),
                "std_score": round(self.current_metrics.std_score, 4),
                "p25": round(self.current_metrics.p25, 4),
                "p50": round(self.current_metrics.p50, 4),
                "p75": round(self.current_metrics.p75, 4),
                "min_score": round(self.current_metrics.min_score, 4),
                "max_score": round(self.current_metrics.max_score, 4),
                "sample_count": self.current_metrics.sample_count,
                "timestamp": self.current_metrics.timestamp,
            }

        if self.baseline_metrics:
            result["baseline"] = {
                "mean_score": round(self.baseline_metrics.mean_score, 4),
                "std_score": round(self.baseline_metrics.std_score, 4),
                "p25": round(self.baseline_metrics.p25, 4),
                "p50": round(self.baseline_metrics.p50, 4),
                "p75": round(self.baseline_metrics.p75, 4),
                "min_score": round(self.baseline_metrics.min_score, 4),
                "max_score": round(self.baseline_metrics.max_score, 4),
                "sample_count": self.baseline_metrics.sample_count,
                "timestamp": self.baseline_metrics.timestamp,
            }

        return result


class DriftDetector:
    """Detects model drift by tracking embedding score distributions.

    Uses a sliding window approach to compare recent score distributions
    against established baselines. Supports configurable thresholds and
    multiple detection strategies.

    Redis Keys:
    - drift:scores:current - List of recent scores (rolling window)
    - drift:baseline - Stored baseline metrics for comparison
    - drift:history:{date} - Historical metrics by date

    Attributes:
        REDIS_KEY_PREFIX: Prefix for all drift-related Redis keys
        WINDOW_HOURS: Time window for collecting current scores
        DRIFT_THRESHOLD: Percentage shift in mean to trigger drift alert
        WARNING_THRESHOLD: Percentage shift to trigger warning
        MIN_SAMPLES: Minimum samples needed for valid metrics
        SCORE_TTL_HOURS: How long to keep raw scores in Redis
    """

    REDIS_KEY_PREFIX = "drift:"
    WINDOW_HOURS = 24
    DRIFT_THRESHOLD = 0.15  # 15% shift in mean triggers drift alert
    WARNING_THRESHOLD = 0.08  # 8% shift triggers warning
    MIN_SAMPLES = 10  # Minimum samples for valid metrics
    SCORE_TTL_HOURS = 48  # Keep raw scores for 48 hours
    MAX_SCORES_STORED = 10000  # Maximum scores to store in Redis list

    def __init__(self, redis_client=None):
        """Initialize drift detector.

        Args:
            redis_client: Redis client for persistent storage.
                         If None, will be lazy-loaded from main module.
        """
        self._redis = redis_client
        self._current_scores: List[float] = []  # In-memory buffer
        self._buffer_size = 100  # Flush to Redis every N scores

    @property
    def redis(self):
        """Lazy-load Redis client if not provided."""
        if self._redis is None:
            try:
                from app.redis_client import get_redis_string_client
                self._redis = get_redis_string_client()
            except ImportError:
                logger.warning("Redis client not available for drift detection")
                return None
        return self._redis

    def _get_key(self, suffix: str) -> str:
        """Get fully qualified Redis key."""
        return f"{self.REDIS_KEY_PREFIX}{suffix}"

    def record_score(self, score: float) -> None:
        """Record a single retrieval score for drift detection.

        Scores are buffered in memory and periodically flushed to Redis
        for efficiency. This is safe to call on every retrieval.

        Args:
            score: Similarity score from vector retrieval (0.0 to 1.0)
        """
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            return  # Silently ignore invalid scores

        self._current_scores.append(float(score))

        # Flush to Redis when buffer is full
        if len(self._current_scores) >= self._buffer_size:
            self._flush_scores()

    def record_scores(self, scores: List[float]) -> None:
        """Record multiple retrieval scores at once.

        Convenient for batch recording after a retrieval operation
        returns multiple documents with scores.

        Args:
            scores: List of similarity scores (0.0 to 1.0)
        """
        for score in scores:
            self.record_score(score)

    def _flush_scores(self) -> None:
        """Flush buffered scores to Redis."""
        if not self._current_scores or not self.redis:
            return

        try:
            key = self._get_key("scores:current")
            timestamp = datetime.now(timezone.utc).isoformat()

            # Store each score with timestamp for time-windowed queries
            pipe = self.redis.pipeline()
            for score in self._current_scores:
                entry = json.dumps({"score": score, "ts": timestamp})
                pipe.lpush(key, entry)

            # Trim to max size to prevent unbounded growth
            pipe.ltrim(key, 0, self.MAX_SCORES_STORED - 1)

            # Set TTL on the key
            pipe.expire(key, self.SCORE_TTL_HOURS * 3600)

            pipe.execute()

            logger.debug(f"Flushed {len(self._current_scores)} scores to Redis")
            self._current_scores = []

        except Exception as e:
            logger.warning(f"Failed to flush scores to Redis: {e}")

    def _get_scores_from_redis(self, hours: int = None) -> List[float]:
        """Retrieve scores from Redis within time window.

        Args:
            hours: Time window in hours (default: WINDOW_HOURS)

        Returns:
            List of scores within the time window
        """
        if not self.redis:
            return self._current_scores.copy()

        hours = hours or self.WINDOW_HOURS
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            key = self._get_key("scores:current")
            entries = self.redis.lrange(key, 0, -1)

            scores = []
            for entry in entries:
                try:
                    data = json.loads(entry)
                    ts = datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
                    if ts >= cutoff:
                        scores.append(data["score"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            # Include any buffered scores not yet flushed
            scores.extend(self._current_scores)

            return scores

        except Exception as e:
            logger.warning(f"Failed to retrieve scores from Redis: {e}")
            return self._current_scores.copy()

    def compute_metrics(self, scores: List[float] = None) -> Optional[DriftMetrics]:
        """Compute distribution metrics from scores.

        Calculates statistical measures including percentiles for
        robust drift detection even with outliers.

        Args:
            scores: List of scores to analyze. If None, uses current window.

        Returns:
            DriftMetrics if sufficient samples, None otherwise
        """
        if scores is None:
            scores = self._get_scores_from_redis()

        if len(scores) < self.MIN_SAMPLES:
            return None

        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        # Calculate percentiles using linear interpolation
        p25_idx = int(n * 0.25)
        p50_idx = int(n * 0.5)
        p75_idx = int(n * 0.75)

        return DriftMetrics(
            mean_score=statistics.mean(sorted_scores),
            std_score=statistics.stdev(sorted_scores) if n > 1 else 0.0,
            p25=sorted_scores[p25_idx],
            p50=sorted_scores[p50_idx],
            p75=sorted_scores[p75_idx],
            min_score=sorted_scores[0],
            max_score=sorted_scores[-1],
            sample_count=n,
            timestamp=datetime.now(timezone.utc).isoformat(),
            window_hours=self.WINDOW_HOURS,
        )

    async def _get_baseline(self) -> Optional[DriftMetrics]:
        """Retrieve stored baseline metrics from Redis.

        Returns:
            DriftMetrics if baseline exists, None otherwise
        """
        if not self.redis:
            return None

        try:
            key = self._get_key("baseline")
            data = self.redis.get(key)

            if data:
                return DriftMetrics.from_dict(json.loads(data))
            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve baseline: {e}")
            return None

    async def set_baseline(self, metrics: DriftMetrics = None) -> bool:
        """Store current metrics as the baseline for drift detection.

        Call this after confirming the system is performing well
        to establish a reference point for future comparisons.

        Args:
            metrics: Metrics to use as baseline. If None, computes from current.

        Returns:
            True if baseline was set successfully
        """
        if not self.redis:
            return False

        if metrics is None:
            metrics = self.compute_metrics()

        if metrics is None:
            logger.warning("Cannot set baseline: insufficient data")
            return False

        try:
            key = self._get_key("baseline")
            self.redis.set(key, json.dumps(metrics.to_dict()))
            logger.info(
                f"Drift detection baseline set: mean={metrics.mean_score:.4f}, "
                f"std={metrics.std_score:.4f}, samples={metrics.sample_count}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to set baseline: {e}")
            return False

    async def check_drift(self) -> DriftCheckResult:
        """Check if drift has occurred compared to baseline.

        Compares current score distribution against the stored baseline
        using multiple statistical measures for robust detection.

        Returns:
            DriftCheckResult with status, metrics, and shift percentages
        """
        checked_at = datetime.now(timezone.utc).isoformat()

        # Flush any buffered scores first
        self._flush_scores()

        # Compute current metrics
        current = self.compute_metrics()
        if current is None:
            return DriftCheckResult(
                status=DriftStatus.INSUFFICIENT_DATA,
                message=f"Need at least {self.MIN_SAMPLES} samples for drift detection",
                checked_at=checked_at,
            )

        # Get historical baseline
        baseline = await self._get_baseline()
        if baseline is None:
            return DriftCheckResult(
                status=DriftStatus.NO_BASELINE,
                current_metrics=current,
                message="No baseline set. Use POST /api/metrics/drift/baseline to establish one.",
                checked_at=checked_at,
            )

        # Calculate shifts (relative change as percentage)
        def safe_pct_change(current_val: float, baseline_val: float) -> float:
            if abs(baseline_val) < 0.001:
                return 0.0 if abs(current_val) < 0.001 else 100.0
            return abs(current_val - baseline_val) / baseline_val * 100

        mean_shift = safe_pct_change(current.mean_score, baseline.mean_score)
        std_shift = safe_pct_change(current.std_score, baseline.std_score)
        median_shift = safe_pct_change(current.p50, baseline.p50)

        # Determine drift status based on thresholds
        # Convert thresholds to percentages for comparison
        drift_pct = self.DRIFT_THRESHOLD * 100
        warning_pct = self.WARNING_THRESHOLD * 100

        if mean_shift >= drift_pct:
            status = DriftStatus.DRIFT_DETECTED
            message = (
                f"Significant drift detected: mean score shifted {mean_shift:.1f}% "
                f"(threshold: {drift_pct:.0f}%). Current: {current.mean_score:.4f}, "
                f"Baseline: {baseline.mean_score:.4f}"
            )
        elif mean_shift >= warning_pct:
            status = DriftStatus.WARNING
            message = (
                f"Warning: mean score shifted {mean_shift:.1f}% "
                f"(warning threshold: {warning_pct:.0f}%). Monitor closely."
            )
        else:
            status = DriftStatus.STABLE
            message = (
                f"Distribution stable. Mean shift: {mean_shift:.1f}%, "
                f"samples: {current.sample_count}"
            )

        # Store current metrics for historical tracking
        await self._store_history(current)

        return DriftCheckResult(
            status=status,
            mean_shift_pct=mean_shift,
            std_shift_pct=std_shift,
            median_shift_pct=median_shift,
            current_metrics=current,
            baseline_metrics=baseline,
            message=message,
            checked_at=checked_at,
        )

    async def _store_history(self, metrics: DriftMetrics) -> None:
        """Store metrics in daily history for trend analysis.

        Args:
            metrics: Metrics to store
        """
        if not self.redis:
            return

        try:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = self._get_key(f"history:{date_str}")

            # Append to daily history list
            self.redis.rpush(key, json.dumps(metrics.to_dict()))

            # Keep history for 30 days
            self.redis.expire(key, 30 * 24 * 3600)

        except Exception as e:
            logger.warning(f"Failed to store metrics history: {e}")

    async def get_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve historical metrics for trend analysis.

        Args:
            days: Number of days of history to retrieve

        Returns:
            List of daily metrics summaries
        """
        if not self.redis:
            return []

        history = []

        try:
            for i in range(days):
                date = datetime.now(timezone.utc) - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                key = self._get_key(f"history:{date_str}")

                entries = self.redis.lrange(key, 0, -1)
                if entries:
                    # Compute daily summary from entries
                    daily_metrics = []
                    for entry in entries:
                        try:
                            data = json.loads(entry)
                            daily_metrics.append(data)
                        except json.JSONDecodeError:
                            continue

                    if daily_metrics:
                        # Average the metrics for the day
                        avg_mean = statistics.mean(m["mean_score"] for m in daily_metrics)
                        avg_std = statistics.mean(m["std_score"] for m in daily_metrics)
                        total_samples = sum(m["sample_count"] for m in daily_metrics)

                        history.append({
                            "date": date_str,
                            "mean_score": round(avg_mean, 4),
                            "std_score": round(avg_std, 4),
                            "total_samples": total_samples,
                            "measurement_count": len(daily_metrics),
                        })

        except Exception as e:
            logger.warning(f"Failed to retrieve history: {e}")

        return history

    async def reset(self) -> bool:
        """Reset drift detection state.

        Clears all stored scores and baseline. Use with caution.

        Returns:
            True if reset was successful
        """
        if not self.redis:
            self._current_scores = []
            return True

        try:
            # Clear current scores
            self.redis.delete(self._get_key("scores:current"))

            # Clear baseline
            self.redis.delete(self._get_key("baseline"))

            # Clear in-memory buffer
            self._current_scores = []

            logger.info("Drift detection state reset")
            return True

        except Exception as e:
            logger.error(f"Failed to reset drift detection: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current drift detector status.

        Returns:
            Dictionary with detector configuration and state
        """
        scores = self._get_scores_from_redis()

        return {
            "enabled": True,
            "window_hours": self.WINDOW_HOURS,
            "drift_threshold_pct": self.DRIFT_THRESHOLD * 100,
            "warning_threshold_pct": self.WARNING_THRESHOLD * 100,
            "min_samples_required": self.MIN_SAMPLES,
            "current_sample_count": len(scores),
            "buffer_size": len(self._current_scores),
            "redis_connected": self.redis is not None,
        }


# Singleton instance for use across the application
drift_detector = DriftDetector()
