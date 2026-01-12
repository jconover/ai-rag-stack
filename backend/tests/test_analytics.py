"""Tests for the real-time analytics module.

Tests the MetricsCollector class for:
- Request tracking and rate calculations
- Latency statistics computation
- Error rate tracking
- Cache hit rate tracking
- Top queries (anonymized)
- Model usage distribution
- Retrieval quality metrics
"""

import time
import pytest
from unittest.mock import patch

# Import the analytics module directly (doesn't depend on external services)
from app.analytics import (
    MetricsCollector,
    SlidingWindowCounter,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestSlidingWindowCounter:
    """Tests for the SlidingWindowCounter class."""

    def test_basic_counting(self):
        """Test basic event counting within window."""
        counter = SlidingWindowCounter(window_seconds=60)

        counter.add()
        counter.add()
        counter.add()

        assert counter.count() == 3

    def test_window_expiry(self):
        """Test that events outside window are pruned."""
        counter = SlidingWindowCounter(window_seconds=1)

        # Add events
        counter.add()
        counter.add()

        # Wait for window to expire
        time.sleep(1.1)

        assert counter.count() == 0

    def test_rate_per_minute(self):
        """Test rate calculation."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Add 10 events at roughly the same time
        now = time.time()
        for i in range(10):
            counter.add(now + i * 0.1)  # Spread over 1 second

        # Rate should be approximately 10 events in ~1 second = 600/min
        rate = counter.rate_per_minute()
        assert rate > 0

    def test_empty_counter(self):
        """Test empty counter returns zero rate."""
        counter = SlidingWindowCounter(window_seconds=60)
        assert counter.count() == 0
        assert counter.rate_per_minute() == 0.0


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a fresh collector for each test."""
        return MetricsCollector(
            short_window_seconds=60,
            long_window_seconds=300,
        )

    def test_record_request(self, collector):
        """Test recording a single request."""
        collector.record_request(
            latency_ms=100.5,
            model="llama3.1:8b",
            status_code=200,
            endpoint="/api/chat",
        )

        # Check request was recorded
        assert collector._request_counter.count() == 1
        assert collector.get_error_rate() == 0.0

    def test_error_tracking(self, collector):
        """Test error rate calculation."""
        # Record 3 successes and 1 error
        for _ in range(3):
            collector.record_request(
                latency_ms=100,
                model="llama3.1:8b",
                status_code=200,
                endpoint="/api/chat",
            )

        collector.record_request(
            latency_ms=50,
            model="llama3.1:8b",
            status_code=500,
            endpoint="/api/chat",
        )

        # Error rate should be 25%
        assert collector.get_error_rate() == pytest.approx(25.0, rel=0.1)

    def test_cache_hit_rate(self, collector):
        """Test cache hit rate calculation."""
        # Record 2 cache hits and 2 cache misses
        collector.record_request(
            latency_ms=50, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat", cache_hit=True,
        )
        collector.record_request(
            latency_ms=100, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat", cache_hit=True,
        )
        collector.record_request(
            latency_ms=200, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat", cache_hit=False,
        )
        collector.record_request(
            latency_ms=180, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat", cache_hit=False,
        )

        # Hit rate should be 50%
        assert collector.get_cache_hit_rate() == pytest.approx(50.0, rel=0.1)

    def test_latency_stats(self, collector):
        """Test latency statistics calculation."""
        latencies = [100, 200, 150, 300, 250]
        for lat in latencies:
            collector.record_request(
                latency_ms=lat,
                model="llama3.1:8b",
                status_code=200,
                endpoint="/api/chat",
            )

        stats = collector.get_latency_stats()

        assert stats["avg_ms"] == pytest.approx(200.0, rel=0.1)
        assert stats["min_ms"] == 100
        assert stats["max_ms"] == 300
        assert stats["median_ms"] == 200  # Middle value of sorted [100,150,200,250,300]

    def test_similarity_score_stats(self, collector):
        """Test retrieval quality score statistics."""
        scores = [0.8, 0.9, 0.85, 0.7, 0.95]
        for score in scores:
            collector.record_request(
                latency_ms=100,
                model="llama3.1:8b",
                status_code=200,
                endpoint="/api/chat",
                avg_similarity_score=score,
            )

        stats = collector.get_similarity_score_stats()

        assert stats["avg_score"] == pytest.approx(0.84, rel=0.01)
        assert stats["min_score"] == 0.7
        assert stats["max_score"] == 0.95

    def test_model_usage_distribution(self, collector):
        """Test model usage tracking."""
        # Record requests with different models
        for _ in range(3):
            collector.record_request(
                latency_ms=100, model="llama3.1:8b", status_code=200,
                endpoint="/api/chat",
            )
        for _ in range(2):
            collector.record_request(
                latency_ms=100, model="mistral:7b", status_code=200,
                endpoint="/api/chat",
            )

        usage = collector.get_model_usage()

        assert "llama3.1:8b" in usage
        assert "mistral:7b" in usage
        assert usage["llama3.1:8b"]["count"] == 3
        assert usage["llama3.1:8b"]["percentage"] == pytest.approx(60.0, rel=0.1)
        assert usage["mistral:7b"]["count"] == 2
        assert usage["mistral:7b"]["percentage"] == pytest.approx(40.0, rel=0.1)

    def test_top_queries_anonymized(self, collector):
        """Test top queries are properly anonymized via hash."""
        # Record same query multiple times
        for _ in range(5):
            collector.record_request(
                latency_ms=100, model="llama3.1:8b", status_code=200,
                endpoint="/api/chat", query="How do I create a Kubernetes deployment?",
            )
        for _ in range(3):
            collector.record_request(
                latency_ms=100, model="llama3.1:8b", status_code=200,
                endpoint="/api/chat", query="What is Docker?",
            )

        top_queries = collector.get_top_queries(limit=10)

        assert len(top_queries) == 2
        # Queries should be hashed, not plaintext
        for query in top_queries:
            assert "query_hash" in query
            assert len(query["query_hash"]) == 12  # SHA256 first 12 chars
            assert "kubernetes" not in query["query_hash"].lower()
            assert "docker" not in query["query_hash"].lower()

        # First query should have count 5
        assert top_queries[0]["count"] == 5
        assert top_queries[1]["count"] == 3

    def test_get_realtime_analytics(self, collector):
        """Test comprehensive analytics snapshot."""
        # Record some requests
        collector.record_request(
            latency_ms=100, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat", cache_hit=True, avg_similarity_score=0.85,
            query="Test query",
        )
        collector.record_request(
            latency_ms=200, model="llama3.1:8b", status_code=500,
            endpoint="/api/chat", cache_hit=False, avg_similarity_score=0.75,
            query="Another query",
        )

        analytics = collector.get_realtime_analytics()

        # Check structure
        assert "timestamp" in analytics
        assert "window_seconds" in analytics
        assert "request_metrics" in analytics
        assert "latency_metrics" in analytics
        assert "cache_metrics" in analytics
        assert "retrieval_quality" in analytics
        assert "model_usage" in analytics
        assert "top_queries" in analytics

        # Check values
        assert analytics["request_metrics"]["total_requests_in_window"] == 2
        assert analytics["request_metrics"]["error_rate_percent"] == pytest.approx(50.0, rel=0.1)
        assert analytics["cache_metrics"]["hit_rate_percent"] == pytest.approx(50.0, rel=0.1)

    def test_reset(self, collector):
        """Test that reset clears all metrics."""
        # Record some data
        collector.record_request(
            latency_ms=100, model="llama3.1:8b", status_code=200,
            endpoint="/api/chat",
        )

        assert collector._request_counter.count() == 1

        # Reset
        collector.reset()

        assert collector._request_counter.count() == 0
        assert collector.get_model_usage() == {}
        assert collector.get_top_queries() == []

    def test_empty_latency_stats(self, collector):
        """Test latency stats with no data."""
        stats = collector.get_latency_stats()

        assert stats["avg_ms"] is None
        assert stats["min_ms"] is None
        assert stats["max_ms"] is None

    def test_empty_similarity_stats(self, collector):
        """Test similarity stats with no data."""
        stats = collector.get_similarity_score_stats()

        assert stats["avg_score"] is None
        assert stats["min_score"] is None


class TestGlobalMetricsCollector:
    """Tests for the global metrics collector singleton."""

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_reset_metrics_collector(self):
        """Test resetting the global collector."""
        collector = get_metrics_collector()

        # Record some data
        collector.record_request(
            latency_ms=100, model="test", status_code=200, endpoint="/api/test",
        )

        # Reset
        reset_metrics_collector()

        # Data should be cleared
        collector = get_metrics_collector()
        assert collector._request_counter.count() == 0
