"""
Enhanced error handling and resilience patterns for Crew-Camufox
Provides circuit breakers, retry mechanisms, and graceful degradation
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Any, Optional, Type, Union, List, Dict
from dataclasses import dataclass, field
from enum import Enum
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .logging import get_logger

logger = get_logger("resilience")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: int = 60
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None

    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker"""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)

    def _async_wrapper(self, func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._execute_async(func, *args, **kwargs)

        return wrapper

    def _sync_wrapper(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_sync(func, *args, **kwargs)

        return wrapper

    async def _execute_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {func.__name__} moved to HALF_OPEN")
            else:
                logger.warning(f"Circuit breaker for {func.__name__} is OPEN, rejecting call")
                raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            self._record_success(func.__name__)
            return result
        except self.config.expected_exception as e:
            self._record_failure(func.__name__, e)
            raise

    def _execute_sync(self, func: Callable, *args, **kwargs):
        """Execute sync function with circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {func.__name__} moved to HALF_OPEN")
            else:
                logger.warning(f"Circuit breaker for {func.__name__} is OPEN, rejecting call")
                raise CircuitBreakerOpenError(f"Circuit breaker is OPEN for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._record_success(func.__name__)
            return result
        except self.config.expected_exception as e:
            self._record_failure(func.__name__, e)
            raise

    def _record_success(self, func_name: str):
        """Record successful execution"""
        self.success_count += 1
        self.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker for {func_name} moved to CLOSED")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _record_failure(self, func_name: str, exception: Exception):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.error(f"Circuit breaker recorded failure for {func_name}: {exception}")

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker for {func_name} moved to OPEN")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.timeout


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""

    pass


class RetryConfig:
    """Configuration for retry mechanisms"""

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        multiplier: float = 2.0,
        exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.multiplier = multiplier
        self.exceptions = exceptions or [Exception]


def resilient_operation(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable] = None,
):
    """
    Decorator combining retry, circuit breaker, and fallback patterns
    """

    def decorator(func: Callable) -> Callable:
        wrapped_func = func

        # Apply circuit breaker if configured
        if circuit_breaker_config:
            circuit_breaker = CircuitBreaker(circuit_breaker_config)
            wrapped_func = circuit_breaker(wrapped_func)

        # Apply retry if configured
        if retry_config:
            retry_decorator = retry(
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.multiplier,
                    min=retry_config.min_wait,
                    max=retry_config.max_wait,
                ),
                retry=retry_if_exception_type(tuple(retry_config.exceptions)),
                before_sleep=before_sleep_log(logger.logger, logging.WARNING),
            )
            wrapped_func = retry_decorator(wrapped_func)

        # Apply fallback wrapper
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(wrapped_func)
            async def async_fallback_wrapper(*args, **kwargs):
                try:
                    return await wrapped_func(*args, **kwargs)
                except Exception as e:
                    if fallback:
                        logger.warning(f"Operation {func.__name__} failed, using fallback: {e}")
                        if asyncio.iscoroutinefunction(fallback):
                            return await fallback(*args, **kwargs)
                        else:
                            return fallback(*args, **kwargs)
                    raise

            return async_fallback_wrapper
        else:

            @functools.wraps(wrapped_func)
            def sync_fallback_wrapper(*args, **kwargs):
                try:
                    return wrapped_func(*args, **kwargs)
                except Exception as e:
                    if fallback:
                        logger.warning(f"Operation {func.__name__} failed, using fallback: {e}")
                        return fallback(*args, **kwargs)
                    raise

            return sync_fallback_wrapper

    return decorator


# Pre-configured resilient decorators for common use cases


def resilient_web_request(max_retries: int = 3):
    """Resilient decorator for web requests"""
    return resilient_operation(
        retry_config=RetryConfig(
            max_attempts=max_retries,
            min_wait=2.0,
            max_wait=30.0,
            exceptions=[ConnectionError, TimeoutError, OSError],
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5, timeout=60, expected_exception=Exception
        ),
    )


def resilient_llm_call(max_retries: int = 3):
    """Resilient decorator for LLM API calls"""
    return resilient_operation(
        retry_config=RetryConfig(
            max_attempts=max_retries,
            min_wait=5.0,
            max_wait=60.0,
            exceptions=[ConnectionError, TimeoutError, ValueError],  # For API errors
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3, timeout=120, expected_exception=Exception
        ),
    )


def resilient_browser_operation(max_retries: int = 2):
    """Resilient decorator for browser operations"""
    return resilient_operation(
        retry_config=RetryConfig(
            max_attempts=max_retries,
            min_wait=3.0,
            max_wait=15.0,
            exceptions=[TimeoutError, ConnectionError],
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3, timeout=180, expected_exception=Exception  # 3 minutes
        ),
    )


class HealthChecker:
    """
    Health checking system for external dependencies
    """

    def __init__(self):
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("health_checker")

    async def check_ollama_health(self, base_url: str) -> bool:
        """Check Ollama service health"""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    healthy = response.status == 200
                    self.health_status["ollama"] = {
                        "healthy": healthy,
                        "last_check": time.time(),
                        "url": base_url,
                        "status_code": response.status,
                    }
                    return healthy
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            self.health_status["ollama"] = {
                "healthy": False,
                "last_check": time.time(),
                "url": base_url,
                "error": str(e),
            }
            return False

    async def check_redis_health(self, redis_url: Optional[str]) -> bool:
        """Check Redis health"""
        if not redis_url:
            return True  # Redis is optional

        try:
            # Try to import redis, handle gracefully if not available
            try:
                import redis.asyncio as redis_async
            except ImportError:
                self.logger.warning("Redis not installed, skipping health check")
                return True

            r = redis_async.from_url(redis_url)
            await r.ping()
            await r.close()

            self.health_status["redis"] = {
                "healthy": True,
                "last_check": time.time(),
                "url": redis_url,
            }
            return True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            self.health_status["redis"] = {
                "healthy": False,
                "last_check": time.time(),
                "url": redis_url,
                "error": str(e),
            }
            return False

    async def check_all_dependencies(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Check all system dependencies"""
        results = {}

        # Check Ollama
        ollama_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
        results["ollama"] = await self.check_ollama_health(ollama_url)

        # Check Redis if configured
        redis_url = config.get("cache", {}).get("redis_url")
        results["redis"] = await self.check_redis_health(redis_url)

        return results

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current health status"""
        return self.health_status.copy()


# Global health checker instance
health_checker = HealthChecker()


class GracefulDegradation:
    """
    Graceful degradation strategies for system components
    """

    @staticmethod
    def fallback_search_results() -> List[Dict[str, Any]]:
        """Fallback when search engines fail"""
        return [
            {
                "title": "Search Service Temporarily Unavailable",
                "url": "https://example.com",
                "snippet": "The search service is temporarily unavailable. Please try again later.",
                "source": "fallback",
            }
        ]

    @staticmethod
    def fallback_llm_response(query: str) -> str:
        """Fallback when LLM is unavailable"""
        return f"""
# Research Report: {query}

## Status
The AI research system is temporarily unavailable. This is a fallback response.

## Recommendation
Please try again later when the system has recovered.

## Manual Research Suggestions
1. Use search engines directly: Google, Bing, DuckDuckGo
2. Check academic databases: Google Scholar, JSTOR, PubMed
3. Review official documentation and websites
4. Consult expert sources and publications

## System Information
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Status: Fallback mode active
"""

    @staticmethod
    async def graceful_shutdown(cleanup_functions: List[Callable]):
        """Perform graceful shutdown with cleanup"""
        logger.info("Starting graceful shutdown...")

        for cleanup_func in cleanup_functions:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
                logger.info(f"Cleanup function {cleanup_func.__name__} completed")
            except Exception as e:
                logger.error(f"Error during cleanup {cleanup_func.__name__}: {e}")

        logger.info("Graceful shutdown completed")
