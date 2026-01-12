"""Circuit breaker implementation for external service resilience.

This module provides circuit breaker patterns for external service calls to prevent
cascade failures when Ollama, Qdrant, or Tavily services are unavailable.

Circuit breaker states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests fail fast without attempting
- HALF_OPEN: Testing if service has recovered

Usage:
    from app.circuit_breaker import (
        ollama_circuit_breaker,
        qdrant_circuit_breaker,
        tavily_circuit_breaker,
        with_ollama_circuit_breaker,
        with_qdrant_circuit_breaker,
        with_tavily_circuit_breaker,
    )

    # Use as decorator
    @with_ollama_circuit_breaker
    def call_ollama(...):
        ...

    # Or use the circuit breaker directly
    result = ollama_circuit_breaker.call(some_function, arg1, arg2)

    # Get circuit breaker states for health checks
    from app.circuit_breaker import get_circuit_breaker_states
    states = get_circuit_breaker_states()
"""

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        name: Unique name for the circuit breaker (used in logging)
        failure_threshold: Number of failures before circuit opens
        success_threshold: Number of successes in half-open to close circuit
        reset_timeout_seconds: Time before attempting recovery (half-open)
        retry_attempts: Number of retry attempts per call
        retry_min_wait: Minimum wait between retries (seconds)
        retry_max_wait: Maximum wait between retries (seconds)
        retry_multiplier: Exponential backoff multiplier
        exceptions: Exception types to catch and retry
    """
    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    retry_multiplier: float = 2.0
    exceptions: tuple = field(default_factory=lambda: (Exception,))


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    current_failures: int = 0
    current_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, breaker_name: str, time_until_retry: float):
        self.breaker_name = breaker_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN. "
            f"Retry in {time_until_retry:.1f} seconds."
        )


class CircuitBreaker:
    """Thread-safe circuit breaker implementation with retry logic.

    The circuit breaker prevents cascade failures by:
    1. Allowing calls through when CLOSED (normal operation)
    2. Failing fast when OPEN (service known to be failing)
    3. Testing recovery when HALF_OPEN (after reset timeout)

    Combines with tenacity retry logic for:
    - Exponential backoff between retry attempts
    - Configurable retry counts per service
    - Exception-specific retry behavior
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker with configuration.

        Args:
            config: CircuitBreakerConfig with breaker settings
        """
        self.config = config
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._last_state_change_time = time.time()
        self._fallback: Optional[Callable] = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic half-open transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if reset timeout has passed
                elapsed = time.time() - self._last_state_change_time
                if elapsed >= self.config.reset_timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                current_failures=self._stats.current_failures,
                current_successes=self._stats.current_successes,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_changes=self._stats.state_changes,
            )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.time()
        self._stats.state_changes += 1

        logger.warning(
            f"Circuit breaker '{self.config.name}' state changed: "
            f"{old_state.value} -> {new_state.value}"
        )

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            self._stats.current_failures = 0
            self._stats.current_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.current_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._stats.current_successes = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._stats.current_failures += 1
            self._stats.current_successes = 0

            logger.warning(
                f"Circuit breaker '{self.config.name}' recorded failure "
                f"({self._stats.current_failures}/{self.config.failure_threshold}): {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.current_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _check_state(self) -> None:
        """Check if circuit allows calls, raise if open."""
        with self._lock:
            current_state = self.state  # This may transition to half-open
            self._stats.total_calls += 1

            if current_state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                elapsed = time.time() - self._last_state_change_time
                time_until_retry = max(0, self.config.reset_timeout_seconds - elapsed)
                raise CircuitBreakerOpen(self.config.name, time_until_retry)

    def set_fallback(self, fallback: Callable) -> "CircuitBreaker":
        """Set a fallback function to call when circuit is open.

        Args:
            fallback: Function to call as fallback

        Returns:
            self for chaining
        """
        self._fallback = fallback
        return self

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback
            Exception: Any exception from func after retries exhausted
        """
        try:
            self._check_state()
        except CircuitBreakerOpen:
            if self._fallback:
                logger.info(
                    f"Circuit breaker '{self.config.name}' using fallback"
                )
                return self._fallback(*args, **kwargs)
            raise

        # Execute with retries
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(self.config.retry_attempts),
                wait=wait_exponential(
                    multiplier=self.config.retry_multiplier,
                    min=self.config.retry_min_wait,
                    max=self.config.retry_max_wait,
                ),
                retry=retry_if_exception_type(self.config.exceptions),
                reraise=True,
            ):
                with attempt:
                    result = func(*args, **kwargs)
                    self._record_success()
                    return result
        except RetryError as e:
            self._record_failure(e.last_attempt.exception())
            raise e.last_attempt.exception() from e
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback
            Exception: Any exception from func after retries exhausted
        """
        try:
            self._check_state()
        except CircuitBreakerOpen:
            if self._fallback:
                logger.info(
                    f"Circuit breaker '{self.config.name}' using fallback"
                )
                if asyncio.iscoroutinefunction(self._fallback):
                    return await self._fallback(*args, **kwargs)
                return self._fallback(*args, **kwargs)
            raise

        # Execute with retries
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.config.retry_attempts),
                wait=wait_exponential(
                    multiplier=self.config.retry_multiplier,
                    min=self.config.retry_min_wait,
                    max=self.config.retry_max_wait,
                ),
                retry=retry_if_exception_type(self.config.exceptions),
                reraise=True,
            ):
                with attempt:
                    result = await func(*args, **kwargs)
                    self._record_success()
                    return result
        except RetryError as e:
            self._record_failure(e.last_attempt.exception())
            raise e.last_attempt.exception() from e
        except Exception as e:
            self._record_failure(e)
            raise

    def reset(self) -> None:
        """Reset circuit breaker to closed state and clear statistics."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitBreakerStats()
            self._last_state_change_time = time.time()
            logger.info(f"Circuit breaker '{self.config.name}' reset to CLOSED")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status for health checks.

        Returns:
            Dictionary with circuit breaker status and statistics
        """
        stats = self.stats
        current_state = self.state

        # Calculate time until retry if open
        time_until_retry = None
        if current_state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change_time
            time_until_retry = max(0, self.config.reset_timeout_seconds - elapsed)

        return {
            "name": self.config.name,
            "state": current_state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "reset_timeout_seconds": self.config.reset_timeout_seconds,
                "retry_attempts": self.config.retry_attempts,
            },
            "stats": {
                "total_calls": stats.total_calls,
                "successful_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "rejected_calls": stats.rejected_calls,
                "current_failures": stats.current_failures,
                "current_successes": stats.current_successes,
                "success_rate": (
                    stats.successful_calls / stats.total_calls
                    if stats.total_calls > 0 else None
                ),
            },
            "time_until_retry": time_until_retry,
            "state_changes": stats.state_changes,
        }


# =============================================================================
# Pre-configured Circuit Breakers for External Services
# =============================================================================

# Ollama circuit breaker configuration
# - Higher retry count (3) for transient failures
# - Exponential backoff: 1s, 2s, 4s
# - Opens after 5 consecutive failures
# - 30 second reset timeout
OLLAMA_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Includes connection refused
)

ollama_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="ollama",
        failure_threshold=5,
        success_threshold=2,
        reset_timeout_seconds=30.0,
        retry_attempts=3,
        retry_min_wait=1.0,
        retry_max_wait=10.0,
        retry_multiplier=2.0,
        exceptions=OLLAMA_EXCEPTIONS,
    )
)

# Qdrant circuit breaker configuration
# - Similar to Ollama for internal service
# - 3 retries with exponential backoff
# - Opens after 5 consecutive failures
# - 30 second reset timeout
QDRANT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

qdrant_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="qdrant",
        failure_threshold=5,
        success_threshold=2,
        reset_timeout_seconds=30.0,
        retry_attempts=3,
        retry_min_wait=1.0,
        retry_max_wait=10.0,
        retry_multiplier=2.0,
        exceptions=QDRANT_EXCEPTIONS,
    )
)

# Tavily circuit breaker configuration
# - Shorter retries (2) for external API
# - Shorter backoff for faster feedback
# - Opens after 3 consecutive failures (external APIs can be rate limited)
# - Longer reset timeout (60s) for external API recovery
TAVILY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

tavily_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="tavily",
        failure_threshold=3,
        success_threshold=2,
        reset_timeout_seconds=60.0,
        retry_attempts=2,
        retry_min_wait=0.5,
        retry_max_wait=5.0,
        retry_multiplier=2.0,
        exceptions=TAVILY_EXCEPTIONS,
    )
)


# =============================================================================
# Decorator Factories
# =============================================================================

def with_circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Callable] = None,
) -> Callable:
    """Create a decorator that wraps a function with circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use
        fallback: Optional fallback function when circuit is open

    Returns:
        Decorator function

    Usage:
        @with_circuit_breaker(ollama_circuit_breaker)
        def call_ollama(...):
            ...

        @with_circuit_breaker(qdrant_circuit_breaker, fallback=lambda: [])
        def search_qdrant(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if fallback:
                    breaker.set_fallback(fallback)
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if fallback:
                    breaker.set_fallback(fallback)
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    return decorator


def with_ollama_circuit_breaker(func: Callable = None, *, fallback: Optional[Callable] = None):
    """Decorator for Ollama circuit breaker protection.

    Usage:
        @with_ollama_circuit_breaker
        def call_ollama(...):
            ...

        @with_ollama_circuit_breaker(fallback=lambda: "Service unavailable")
        def call_ollama(...):
            ...
    """
    if func is None:
        return with_circuit_breaker(ollama_circuit_breaker, fallback=fallback)
    return with_circuit_breaker(ollama_circuit_breaker)(func)


def with_qdrant_circuit_breaker(func: Callable = None, *, fallback: Optional[Callable] = None):
    """Decorator for Qdrant circuit breaker protection.

    Usage:
        @with_qdrant_circuit_breaker
        def search_qdrant(...):
            ...

        @with_qdrant_circuit_breaker(fallback=lambda *args, **kwargs: [])
        def search_qdrant(...):
            ...
    """
    if func is None:
        return with_circuit_breaker(qdrant_circuit_breaker, fallback=fallback)
    return with_circuit_breaker(qdrant_circuit_breaker)(func)


def with_tavily_circuit_breaker(func: Callable = None, *, fallback: Optional[Callable] = None):
    """Decorator for Tavily circuit breaker protection.

    Usage:
        @with_tavily_circuit_breaker
        async def search_web(...):
            ...

        @with_tavily_circuit_breaker(fallback=lambda *args: WebSearchResponse(...))
        async def search_web(...):
            ...
    """
    if func is None:
        return with_circuit_breaker(tavily_circuit_breaker, fallback=fallback)
    return with_circuit_breaker(tavily_circuit_breaker)(func)


# =============================================================================
# Health Check Utilities
# =============================================================================

def get_circuit_breaker_states() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers for health checks.

    Returns:
        Dictionary mapping breaker name to status
    """
    return {
        "ollama": ollama_circuit_breaker.get_status(),
        "qdrant": qdrant_circuit_breaker.get_status(),
        "tavily": tavily_circuit_breaker.get_status(),
    }


def get_circuit_breakers_healthy() -> bool:
    """Check if all circuit breakers are in healthy state (not OPEN).

    Returns:
        True if all breakers are CLOSED or HALF_OPEN
    """
    states = get_circuit_breaker_states()
    for name, status in states.items():
        if status["state"] == CircuitState.OPEN.value:
            return False
    return True


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state.

    Use with caution - only for testing or manual recovery.
    """
    ollama_circuit_breaker.reset()
    qdrant_circuit_breaker.reset()
    tavily_circuit_breaker.reset()
    logger.info("All circuit breakers reset to CLOSED state")


# =============================================================================
# Exception Helpers
# =============================================================================

def is_circuit_breaker_exception(exc: Exception) -> bool:
    """Check if an exception is a circuit breaker exception.

    Args:
        exc: Exception to check

    Returns:
        True if exception is CircuitBreakerOpen
    """
    return isinstance(exc, CircuitBreakerOpen)


def get_service_unavailable_message(exc: CircuitBreakerOpen) -> str:
    """Get a user-friendly message for circuit breaker open exception.

    Args:
        exc: CircuitBreakerOpen exception

    Returns:
        User-friendly error message
    """
    service_names = {
        "ollama": "LLM service",
        "qdrant": "Vector database",
        "tavily": "Web search service",
    }
    service_name = service_names.get(exc.breaker_name, exc.breaker_name)
    return (
        f"{service_name} is temporarily unavailable. "
        f"Please try again in {exc.time_until_retry:.0f} seconds."
    )
