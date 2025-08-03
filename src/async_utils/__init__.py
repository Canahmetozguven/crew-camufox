"""
Async optimization utilities for crew-camufox
Provides connection pooling, concurrent execution, and async utilities
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable, TypeVar, Awaitable
from datetime import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionPool:
    """HTTP connection pool manager for async requests"""

    def __init__(
        self, connector_limit: int = 100, connector_limit_per_host: int = 10, timeout: int = 30
    ):
        self.connector_limit = connector_limit
        self.connector_limit_per_host = connector_limit_per_host
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def start(self):
        """Initialize connection pool"""
        if self._session is None:
            self._connector = aiohttp.TCPConnector(
                limit=self.connector_limit,
                limit_per_host=self.connector_limit_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            self._session = aiohttp.ClientSession(connector=self._connector, timeout=timeout)

            logger.info("HTTP connection pool started with %d connections", self.connector_limit)

    async def close(self):
        """Close connection pool"""
        if self._session:
            await self._session.close()
            self._session = None
            self._connector = None
            logger.info("HTTP connection pool closed")

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get HTTP session"""
        if self._session is None:
            raise RuntimeError(
                "Connection pool not started. Use async context manager or call start()"
            )
        return self._session


class AsyncBatch:
    """Utility for batching async operations with concurrency control"""

    def __init__(self, max_concurrent: int = 10, batch_size: int = 50):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_batch(
        self, tasks: List[Callable[..., Awaitable[T]]], *args, **kwargs
    ) -> List[Optional[T]]:
        """Execute a batch of async tasks with concurrency control"""

        async def _execute_with_semaphore(task_func):
            async with self.semaphore:
                try:
                    return await task_func(*args, **kwargs)
                except Exception as e:
                    logger.error("Task execution failed: %s", e)
                    return None

        # Split into chunks if needed
        results = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i : i + self.batch_size]
            batch_results = await asyncio.gather(
                *[_execute_with_semaphore(task) for task in batch], return_exceptions=True
            )
            results.extend(batch_results)

        return results

    async def map_async(self, func: Callable[[T], Awaitable[Any]], items: List[T]) -> List[Any]:
        """Map async function over list of items with concurrency control"""

        async def _execute_item(item):
            async with self.semaphore:
                try:
                    return await func(item)
                except Exception as e:
                    logger.error("Item processing failed for %s: %s", item, e)
                    return None

        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_results = await asyncio.gather(
                *[_execute_item(item) for item in batch], return_exceptions=True
            )
            results.extend(batch_results)

        return results


class AsyncTimer:
    """Async timing utilities for performance monitoring"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    async def __aenter__(self):
        """Start timing"""
        self.start_time = time.time()
        logger.debug("Started timing %s", self.name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End timing"""
        self.end_time = time.time()
        duration = self.duration
        logger.info("Completed %s in %.2fs", self.name, duration)

    @property
    def duration(self) -> float:
        """Get operation duration in seconds"""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class SearchExecutor:
    """Optimized search execution with caching and concurrency"""

    def __init__(
        self,
        connection_pool: ConnectionPool,
        max_concurrent_searches: int = 5,
        default_timeout: int = 30,
    ):
        self.connection_pool = connection_pool
        self.max_concurrent = max_concurrent_searches
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent_searches)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def execute_search(
        self, search_func: Callable, query: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Execute a single search with timeout and error handling"""
        async with self.semaphore:
            try:
                async with AsyncTimer(f"search_{search_func.__name__}") as timer:
                    # Use asyncio timeout for the search
                    result = await asyncio.wait_for(
                        search_func(query, **kwargs),
                        timeout=kwargs.get("timeout", self.default_timeout),
                    )

                    if result:
                        result["execution_time"] = timer.duration
                        result["timestamp"] = datetime.now().isoformat()

                    return result

            except asyncio.TimeoutError:
                logger.warning("Search timeout for query: %s", query[:50])
                return None
            except Exception as e:
                logger.error("Search failed for query %s: %s", query[:50], e)
                return None

    async def execute_parallel_searches(
        self, searches: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Execute multiple searches in parallel with concurrency control"""
        tasks = []

        for search_config in searches:
            search_func = search_config["function"]
            query = search_config["query"]
            kwargs = search_config.get("kwargs", {})

            task = self.execute_search(search_func, query, **kwargs)
            tasks.append(task)

        logger.info("Executing %d searches in parallel", len(tasks))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Search %d failed: %s", i, result)
                processed_results.append(None)
            else:
                processed_results.append(result)

        successful_searches = sum(1 for r in processed_results if r is not None)
        logger.info("Completed %d/%d searches successfully", successful_searches, len(searches))

        return processed_results

    async def run_in_thread(self, func: Callable, *args, **kwargs):
        """Run a synchronous function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)

    def close(self):
        """Clean up thread pool"""
        self.thread_pool.shutdown(wait=True)


class AsyncQueue:
    """Async queue for processing items with backpressure control"""

    def __init__(self, maxsize: int = 100):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.processing = False
        self.processed_count = 0
        self.error_count = 0

    async def put(self, item: Any, timeout: Optional[float] = None):
        """Add item to queue with optional timeout"""
        if timeout:
            await asyncio.wait_for(self.queue.put(item), timeout=timeout)
        else:
            await self.queue.put(item)

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue with optional timeout"""
        if timeout:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        else:
            return await self.queue.get()

    async def process_queue(self, processor: Callable[[Any], Awaitable[Any]], max_workers: int = 5):
        """Process queue items with multiple workers"""
        if self.processing:
            logger.warning("Queue processing already started")
            return

        self.processing = True
        self.processed_count = 0
        self.error_count = 0

        async def worker(worker_id: int):
            logger.debug("Worker %d started", worker_id)
            while self.processing:
                try:
                    # Use timeout to allow checking processing flag
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                    try:
                        result = await processor(item)
                        self.processed_count += 1
                        logger.debug("Worker %d processed item: %s", worker_id, result)
                    except Exception as e:
                        self.error_count += 1
                        logger.error("Worker %d processing error: %s", worker_id, e)
                    finally:
                        self.queue.task_done()

                except asyncio.TimeoutError:
                    # Check if we should continue processing
                    continue
                except Exception as e:
                    logger.error("Worker %d error: %s", worker_id, e)
                    break

            logger.debug("Worker %d stopped", worker_id)

        # Start workers
        workers = [asyncio.create_task(worker(i)) for i in range(max_workers)]

        try:
            # Wait for all current items to be processed
            await self.queue.join()
        finally:
            # Stop processing and cleanup workers
            self.processing = False
            await asyncio.gather(*workers, return_exceptions=True)

        logger.info(
            "Queue processing completed: %d processed, %d errors",
            self.processed_count,
            self.error_count,
        )

    def stop_processing(self):
        """Stop queue processing"""
        self.processing = False

    @property
    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()


@asynccontextmanager
async def async_resource_manager(*resources):
    """Context manager for multiple async resources"""
    try:
        # Start all resources
        for resource in resources:
            if hasattr(resource, "start"):
                await resource.start()

        yield resources

    finally:
        # Clean up all resources
        for resource in resources:
            try:
                if hasattr(resource, "close"):
                    await resource.close()
                elif hasattr(resource, "stop_processing"):
                    resource.stop_processing()
            except Exception as e:
                logger.error("Error closing resource %s: %s", resource, e)


class AsyncBatchProcessor:
    """Generic async batch processor with error handling and progress tracking"""

    def __init__(
        self, batch_size: int = 10, max_concurrent: int = 5, timeout_per_item: float = 30.0
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.timeout_per_item = timeout_per_item
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_items(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Process items in batches with progress tracking"""
        results = []
        total_items = len(items)
        processed_items = 0

        async def process_single_item(item):
            async with self.semaphore:
                try:
                    result = await asyncio.wait_for(processor(item), timeout=self.timeout_per_item)
                    return result
                except asyncio.TimeoutError:
                    logger.warning("Item processing timeout: %s", str(item)[:100])
                    return None
                except Exception as e:
                    logger.error("Item processing error: %s", e)
                    return None

        # Process in batches
        for i in range(0, total_items, self.batch_size):
            batch = items[i : i + self.batch_size]

            logger.info(
                "Processing batch %d/%d (%d items)",
                i // self.batch_size + 1,
                (total_items + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[process_single_item(item) for item in batch], return_exceptions=True
            )

            results.extend(batch_results)
            processed_items += len(batch)

            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(processed_items, total_items)
                except Exception as e:
                    logger.error("Progress callback error: %s", e)

        successful_results = sum(1 for r in results if r is not None)
        logger.info("Batch processing completed: %d/%d successful", successful_results, total_items)

        return results


# Global instances for easy access
_connection_pool: Optional[ConnectionPool] = None
_search_executor: Optional[SearchExecutor] = None


async def get_connection_pool() -> ConnectionPool:
    """Get or create global connection pool"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool()
        await _connection_pool.start()
    return _connection_pool


async def get_search_executor() -> SearchExecutor:
    """Get or create global search executor"""
    global _search_executor
    if _search_executor is None:
        pool = await get_connection_pool()
        _search_executor = SearchExecutor(pool)
    return _search_executor


async def cleanup_global_resources():
    """Clean up global async resources"""
    global _connection_pool, _search_executor

    if _search_executor:
        _search_executor.close()
        _search_executor = None

    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None

    logger.info("Global async resources cleaned up")
