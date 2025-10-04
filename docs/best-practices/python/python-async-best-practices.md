# Python Async Best Practices: Write Coroutines That Don't Betray You

**Objective**: Master Python async programming with a focus on production reliability, performance, and maintainability. When you need to handle thousands of concurrent connections, when you want to keep UIs fluid, when you're building high-performance I/O systems—async becomes your weapon of choice.

Async is a power tool. Use it well and you'll saturate sockets, keep UIs fluid, and squeeze latency out of I/O. Use it badly and you'll summon deadlocks, zombie tasks, and cancellation bugs that only appear at 3 a.m. This guide shows you how to wield async with the precision of a battle-tested backend engineer.

## 0) Prerequisites (Read Once, Live by Them)

### The Seven Commandments

1. **Understand async fundamentals**
   - Event loop and cooperative scheduling
   - I/O-bound vs CPU-bound workloads
   - When async works vs when it doesn't
   - Cancellation and exception handling

2. **Master structured concurrency**
   - TaskGroup for bounded parallelism
   - Proper task lifecycle management
   - Exception propagation and cleanup
   - Resource management patterns

3. **Know your performance boundaries**
   - Async for I/O, not CPU
   - Bounded concurrency and backpressure
   - Timeout management and cancellation
   - Resource pooling and reuse

4. **Validate everything**
   - Test async code properly
   - Handle cancellation gracefully
   - Monitor resource usage
   - Profile performance bottlenecks

5. **Plan for production**
   - Graceful shutdown patterns
   - Error handling and recovery
   - Resource cleanup and lifecycle
   - Monitoring and debugging

**Why These Principles**: Async mastery is the foundation of high-performance Python applications. Understanding these patterns prevents deadlocks, resource leaks, and performance bottlenecks.

## 1) The Three Deadly Sins (and Their Penance)

### Sin #1: Blocking the Event Loop

```python
# ❌ BAD: blocks event loop; every task starves
import requests

async def bad_fetch(url: str) -> str:
    # This blocks the entire event loop
    response = requests.get(url)  # DON'T DO THIS
    return response.text

# ✅ GOOD: offload blocking I/O
import asyncio
import aiohttp

async def good_fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()

# ✅ ALTERNATIVE: offload to thread pool
async def alternative_fetch(url: str) -> str:
    response = await asyncio.to_thread(requests.get, url)
    return response.text
```

### Sin #2: Fire-and-Forget Tasks

```python
# ❌ BAD: loses exceptions; lifetime unmanaged
async def bad_worker():
    asyncio.create_task(do_work())  # DON'T DO THIS
    # Task is orphaned, exceptions are lost

# ✅ GOOD: own the task lifecycle
async def good_worker():
    task = asyncio.create_task(do_work())
    try:
        result = await task
        return result
    except Exception as e:
        task.cancel()
        raise

# ✅ BETTER: use TaskGroup (Python 3.11+)
async def better_worker():
    async with asyncio.TaskGroup() as tg:
        task = tg.create_task(do_work())
        return await task
```

### Sin #3: Swallowing Cancellation

```python
# ❌ BAD: eats CancelledError
async def bad_handler():
    try:
        await something()
    except Exception:
        pass  # DON'T DO THIS - eats CancelledError!

# ✅ GOOD: handle cancellation properly
async def good_handler():
    try:
        await something()
    except asyncio.CancelledError:
        raise  # always re-raise CancelledError
    except Exception as e:
        logger.error("Operation failed", exc_info=e)
        raise
```

**Why These Sins Matter**: These patterns break async's cooperative nature and lead to deadlocks, resource leaks, and lost exceptions. Understanding these pitfalls prevents production failures.

## 2) Idiomatic Async Function Anatomy

### Proper Async Function Structure

```python
import asyncio
from typing import Any
import aiohttp

async def fetch_resource(
    session: aiohttp.ClientSession, 
    url: str, 
    *, 
    timeout: float = 5.0
) -> dict[str, Any]:
    """GET a JSON resource with deadline and cancellation-safe cleanup."""
    async with asyncio.timeout(timeout):  # Python 3.11+
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

# Usage with proper resource management
async def main():
    async with aiohttp.ClientSession() as session:
        data = await fetch_resource(session, "https://api.example.com/data")
        return data
```

### Key Principles

```python
# 1. Type hints and keyword-only parameters
async def process_data(
    data: list[str], 
    *, 
    batch_size: int = 100,
    timeout: float = 30.0
) -> list[dict]:
    """Process data with explicit parameters."""
    pass

# 2. Use async context managers for resource cleanup
class AsyncResource:
    async def __aenter__(self):
        self.conn = await open_connection()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()

# 3. Use asyncio.timeout() for deadlines
async def with_deadline(coro, seconds: float):
    async with asyncio.timeout(seconds):
        return await coro
```

**Why This Structure Matters**: Proper async function design prevents resource leaks and makes cancellation handling explicit. Understanding these patterns enables reliable async code.

## 3) Structured Concurrency with TaskGroup

### Stop Hand-Rolling Task Orchestration

```python
import asyncio
import aiohttp

async def crawl_all(session: aiohttp.ClientSession, urls: list[str]) -> list[bytes]:
    """Crawl URLs with structured concurrency."""
    results: list[bytes] = []
    
    async with asyncio.TaskGroup() as tg:
        tasks = {tg.create_task(fetch_url(session, url)): url for url in urls}
        
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                print(f"Failed to fetch {tasks[task]}: {e}")
    
    return results

async def fetch_url(session: aiohttp.ClientSession, url: str) -> bytes:
    """Fetch a single URL."""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()
```

### Why TaskGroup is Better

```python
# ❌ OLD WAY: manual task management
async def old_way():
    tasks = [asyncio.create_task(work(i)) for i in range(10)]
    results = []
    for task in tasks:
        try:
            result = await task
            results.append(result)
        except Exception as e:
            print(f"Task failed: {e}")
    return results

# ✅ NEW WAY: structured concurrency
async def new_way():
    results = []
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(work(i)) for i in range(10)]
        for task in tasks:
            result = await task
            results.append(result)
    return results
```

**Why Structured Concurrency Matters**: TaskGroup ensures tasks can't outlive their scope and properly handles exception propagation. Understanding these patterns prevents orphaned tasks and lost exceptions.

## 4) Limit Concurrency (Or the Kernel Will Rate-Limit Your Dreams)

### Bounded Concurrency with Semaphores

```python
import asyncio
import aiohttp

class BoundedFetcher:
    """HTTP client with bounded concurrency."""
    
    def __init__(self, session: aiohttp.ClientSession, limit: int = 100):
        self.session = session
        self.sem = asyncio.Semaphore(limit)
    
    async def get(self, url: str) -> bytes:
        """Fetch URL with concurrency limit."""
        async with self.sem:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.read()

# Usage
async def main():
    async with aiohttp.ClientSession() as session:
        fetcher = BoundedFetcher(session, limit=50)
        
        urls = [f"https://api.example.com/data/{i}" for i in range(1000)]
        tasks = [fetcher.get(url) for url in urls]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### Rule of Thumb for Concurrency Limits

```python
# Concurrency limits by use case
concurrency_limits = {
    "http_requests": "min(100, 10 * cpu_count)",
    "database_queries": "min(50, 5 * cpu_count)", 
    "file_operations": "min(20, 2 * cpu_count)",
    "external_apis": "min(10, cpu_count)"
}

# Example: adaptive concurrency
import os

def get_optimal_concurrency(base_limit: int = 10) -> int:
    """Calculate optimal concurrency based on system resources."""
    cpu_count = os.cpu_count() or 1
    return min(base_limit, 10 * cpu_count)
```

**Why Concurrency Limits Matter**: Unbounded concurrency leads to resource exhaustion and rate limiting. Understanding these patterns prevents system overload and enables sustainable performance.

## 5) Backpressure with asyncio.Queue

### Producer-Consumer with Backpressure

```python
import asyncio
from typing import Any

async def producer(q: asyncio.Queue[Any], data_source):
    """Producer that respects queue capacity."""
    for item in data_source:
        await q.put(item)  # blocks when queue is full
        print(f"Produced: {item}")

async def consumer(q: asyncio.Queue[Any], consumer_id: int):
    """Consumer that processes items."""
    while True:
        try:
            item = await q.get()
            if item is None:  # Shutdown signal
                break
            
            # Process item
            await process_item(item, consumer_id)
            
        except Exception as e:
            print(f"Consumer {consumer_id} error: {e}")
        finally:
            q.task_done()

async def process_item(item: Any, consumer_id: int):
    """Process individual item."""
    await asyncio.sleep(0.1)  # Simulate work
    print(f"Consumer {consumer_id} processed: {item}")

async def main():
    """Main coordination function."""
    # Create bounded queue
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)
    
    # Start consumers
    num_consumers = 4
    consumers = [
        asyncio.create_task(consumer(q, i)) 
        for i in range(num_consumers)
    ]
    
    # Start producer
    data_source = range(1000)
    producer_task = asyncio.create_task(producer(q, data_source))
    
    # Wait for producer to finish
    await producer_task
    
    # Shutdown consumers
    for _ in range(num_consumers):
        await q.put(None)
    
    # Wait for all work to complete
    await q.join()
    
    # Cancel and wait for consumers
    for consumer_task in consumers:
        consumer_task.cancel()
    
    await asyncio.gather(*consumers, return_exceptions=True)

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

**Why Backpressure Matters**: Unbounded queues lead to memory exhaustion. Understanding these patterns prevents system overload and enables sustainable data processing.

## 6) Timeouts at Boundaries, Not Everywhere

### Layered Timeout Strategy

```python
import asyncio
from typing import Any

async def with_deadline(coro, seconds: float):
    """Apply timeout to any coroutine."""
    async with asyncio.timeout(seconds):
        return await coro

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    base_timeout: float = 5.0
) -> Any:
    """Fetch with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            # Timeout at the I/O boundary
            async with asyncio.timeout(base_timeout * (2 ** attempt)):
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))
```

### Timeout Best Practices

```python
# ✅ GOOD: timeout at I/O boundaries
async def good_timeout_pattern():
    async with asyncio.timeout(30.0):  # Overall deadline
        async with asyncio.timeout(5.0):  # Per-request timeout
            result = await fetch_data()
        return result

# ❌ BAD: timeout everywhere
async def bad_timeout_pattern():
    result1 = await asyncio.wait_for(step1(), timeout=1.0)
    result2 = await asyncio.wait_for(step2(), timeout=1.0)
    result3 = await asyncio.wait_for(step3(), timeout=1.0)
    return result1, result2, result3
```

**Why Timeout Strategy Matters**: Proper timeout placement prevents cascading failures while maintaining responsiveness. Understanding these patterns enables robust error handling.

## 7) Cancellation: Design for It Like You Mean It

### Proper Cancellation Handling

```python
import asyncio
from typing import Any

async def cancellable_operation():
    """Operation that handles cancellation properly."""
    try:
        # Do work that can be cancelled
        await asyncio.sleep(1.0)
        return "completed"
    except asyncio.CancelledError:
        # Cleanup before re-raising
        print("Operation cancelled, cleaning up...")
        await cleanup()
        raise  # Always re-raise CancelledError
    except Exception as e:
        # Handle other exceptions
        print(f"Operation failed: {e}")
        raise

async def cleanup():
    """Cleanup function."""
    await asyncio.sleep(0.1)  # Simulate cleanup

# Critical sections that must complete
async def critical_operation():
    """Operation that must complete or fail cleanly."""
    try:
        # Do work
        await do_work()
    except asyncio.CancelledError:
        # Use shield for critical sections
        await asyncio.shield(commit_transaction())
        raise

async def do_work():
    """Simulate work."""
    await asyncio.sleep(0.5)

async def commit_transaction():
    """Commit that must complete."""
    await asyncio.sleep(0.1)
    print("Transaction committed")
```

### Cancellation Patterns

```python
# Pattern 1: Cleanup in finally
async def with_cleanup():
    resource = None
    try:
        resource = await acquire_resource()
        await use_resource(resource)
    finally:
        if resource:
            await release_resource(resource)

# Pattern 2: Shield critical sections
async def with_shield():
    try:
        await do_work()
    except asyncio.CancelledError:
        # This must complete
        await asyncio.shield(critical_cleanup())
        raise

# Pattern 3: Graceful degradation
async def with_fallback():
    try:
        return await primary_operation()
    except asyncio.CancelledError:
        return await fallback_operation()
```

**Why Cancellation Matters**: Proper cancellation handling prevents resource leaks and ensures clean shutdowns. Understanding these patterns enables robust async applications.

## 8) Don't Do CPU Work Here

### Offload CPU-Bound Work

```python
import asyncio
import math
from concurrent.futures import ProcessPoolExecutor

def cpu_heavy(n: int) -> float:
    """CPU-intensive computation."""
    return sum(math.sqrt(i) for i in range(n))

async def async_cpu_work():
    """Offload CPU work to process pool."""
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_heavy, 10_000_000)
    
    return result

# For GIL-releasing C libraries
async def numpy_work():
    """Use to_thread for NumPy operations that release GIL."""
    import numpy as np
    
    def numpy_computation():
        arr = np.random.random((1000, 1000))
        return np.dot(arr, arr.T).sum()
    
    result = await asyncio.to_thread(numpy_computation)
    return result
```

### CPU vs I/O Decision Matrix

```python
# Decision matrix for CPU work
cpu_decision_matrix = {
    "pure_python": "ProcessPoolExecutor",
    "numpy_scipy": "asyncio.to_thread (if GIL released)",
    "c_extensions": "asyncio.to_thread (if GIL released)",
    "blocking_io": "asyncio.to_thread",
    "network_io": "native async (aiohttp, etc.)"
}

# Example: hybrid approach
async def hybrid_processing(data):
    """Process data with both I/O and CPU work."""
    # I/O: fetch data
    async with aiohttp.ClientSession() as session:
        raw_data = await fetch_data(session)
    
    # CPU: process data
    processed_data = await asyncio.to_thread(process_data, raw_data)
    
    # I/O: save results
    await save_results(processed_data)
    
    return processed_data
```

**Why CPU Offloading Matters**: Async won't make CPU-bound Python faster due to the GIL. Understanding these patterns prevents performance bottlenecks and enables optimal resource utilization.

## 9) Async Context Managers & Iterators

### Resource Management Patterns

```python
import asyncio
from typing import AsyncIterator

class AsyncResource:
    """Async resource with proper lifecycle management."""
    
    async def __aenter__(self):
        self.conn = await self.open_connection()
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            await self.conn.close()
    
    async def open_connection(self):
        """Open connection."""
        await asyncio.sleep(0.1)  # Simulate connection
        return Connection()
    
    async def close_connection(self):
        """Close connection."""
        if self.conn:
            await self.conn.close()

class Connection:
    """Mock connection."""
    async def close(self):
        await asyncio.sleep(0.1)

# Async iterators
async def consume_stream(stream: AsyncIterator[str]):
    """Consume async iterator."""
    async for chunk in stream:
        await process_chunk(chunk)

async def process_chunk(chunk: str):
    """Process individual chunk."""
    await asyncio.sleep(0.01)
    print(f"Processed: {chunk}")
```

### Context Manager Best Practices

```python
# Pattern 1: Database connections
class DatabasePool:
    async def __aenter__(self):
        self.pool = await create_pool()
        return self.pool
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.pool.close()

# Pattern 2: HTTP sessions
class HTTPSession:
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

# Pattern 3: File operations
class AsyncFile:
    async def __aenter__(self):
        self.file = await aiofiles.open(self.path, 'r')
        return self.file
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.file.close()
```

**Why Context Managers Matter**: Proper resource management prevents leaks and ensures clean shutdowns. Understanding these patterns enables reliable async applications.

## 10) Logging, Tracing, and Context Variables

### Request Context with Context Variables

```python
import contextvars
import logging
from typing import Optional

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

def get_logger(name: str = "app") -> logging.Logger:
    """Get logger with context information."""
    logger = logging.getLogger(name)
    
    # Add context to log records
    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.request_id = request_id_var.get()
            record.user_id = user_id_var.get()
            return True
    
    logger.addFilter(ContextFilter())
    return logger

async def handle_request(request_id: str, user_id: str):
    """Handle request with context."""
    # Set context variables
    request_token = request_id_var.set(request_id)
    user_token = user_id_var.set(user_id)
    
    try:
        logger = get_logger()
        logger.info("Request started")
        
        # Process request
        result = await process_request()
        
        logger.info("Request completed")
        return result
        
    finally:
        # Reset context
        request_id_var.reset(request_token)
        user_id_var.reset(user_token)

async def process_request():
    """Process request with context preserved."""
    logger = get_logger()
    logger.info("Processing request")
    
    # Context is preserved across awaits
    await asyncio.sleep(0.1)
    
    logger.info("Request processed")
    return "success"
```

### Structured Logging

```python
import json
import logging
from typing import Dict, Any

class StructuredLogger:
    """Logger that outputs structured JSON."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data."""
        data = {
            "level": "INFO",
            "message": message,
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            **kwargs
        }
        self.logger.info(json.dumps(data))
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error with structured data."""
        data = {
            "level": "ERROR",
            "message": message,
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            **kwargs
        }
        if exc_info:
            data["exception"] = str(kwargs.get("exception"))
        self.logger.error(json.dumps(data))
```

**Why Context Variables Matter**: Request context flows through async calls without explicit parameter passing. Understanding these patterns enables proper tracing and debugging.

## 11) Testing Async Code (Without Lies)

### Proper Async Testing

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_worker_drains_queue():
    """Test worker that drains queue."""
    q = asyncio.Queue()
    await q.put(1)
    await q.put(2)
    await q.put(None)  # Shutdown signal
    
    out = []
    
    async def worker():
        while (item := await q.get()) is not None:
            out.append(item)
            q.task_done()
    
    task = asyncio.create_task(worker())
    await q.join()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    
    assert out == [1, 2]

@pytest.mark.asyncio
async def test_http_client():
    """Test HTTP client with mocked session."""
    with patch('aiohttp.ClientSession') as mock_session:
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.__aenter__.return_value = mock_response
        
        mock_session.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Test the function
        result = await fetch_data("https://api.example.com")
        assert result == {"status": "ok"}

@pytest.mark.asyncio
async def test_timeout_handling():
    """Test timeout handling."""
    async def slow_operation():
        await asyncio.sleep(10)  # This will timeout
        return "completed"
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.1)
```

### Testing Patterns

```python
# Pattern 1: Mock external dependencies
@pytest.mark.asyncio
async def test_with_mocks():
    with patch('aiohttp.ClientSession') as mock_session:
        # Setup mocks
        pass

# Pattern 2: Use asyncio.Event for synchronization
@pytest.mark.asyncio
async def test_with_events():
    event = asyncio.Event()
    
    async def worker():
        await event.wait()
        return "done"
    
    task = asyncio.create_task(worker())
    event.set()
    result = await task
    assert result == "done"

# Pattern 3: Test cancellation
@pytest.mark.asyncio
async def test_cancellation():
    async def cancellable_task():
        try:
            await asyncio.sleep(1)
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"
    
    task = asyncio.create_task(cancellable_task())
    task.cancel()
    result = await task
    assert result == "cancelled"
```

**Why Proper Testing Matters**: Async code has different failure modes than sync code. Understanding these patterns prevents flaky tests and enables reliable async applications.

## 12) Resource Management Checklist

### Production Resource Management

```python
import asyncio
import aiohttp
import aiofiles
from typing import Dict, Any

class ResourceManager:
    """Centralized resource management."""
    
    def __init__(self):
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.pools: Dict[str, Any] = {}
        self.files: Dict[str, Any] = {}
    
    async def get_session(self, service: str) -> aiohttp.ClientSession:
        """Get or create HTTP session for service."""
        if service not in self.sessions:
            self.sessions[service] = aiohttp.ClientSession()
        return self.sessions[service]
    
    async def get_pool(self, database: str) -> Any:
        """Get or create database pool."""
        if database not in self.pools:
            self.pools[database] = await create_pool(database)
        return self.pools[database]
    
    async def cleanup(self):
        """Cleanup all resources."""
        # Close HTTP sessions
        for session in self.sessions.values():
            await session.close()
        
        # Close database pools
        for pool in self.pools.values():
            await pool.close()
        
        # Close files
        for file in self.files.values():
            await file.close()

# Usage with proper cleanup
async def main():
    manager = ResourceManager()
    try:
        # Use resources
        session = await manager.get_session("api")
        pool = await manager.get_pool("main")
        
        # Do work
        await do_work(session, pool)
        
    finally:
        # Always cleanup
        await manager.cleanup()
```

### Resource Management Best Practices

```python
# ✅ GOOD: One session per service
class HTTPService:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, path: str):
        async with self.session.get(f"{self.base_url}{path}") as resp:
            return await resp.json()

# ✅ GOOD: Database pool with limits
class DatabaseService:
    def __init__(self, connection_string: str, max_size: int = 10):
        self.connection_string = connection_string
        self.max_size = max_size
        self.pool = None
    
    async def __aenter__(self):
        self.pool = await create_pool(
            self.connection_string,
            max_size=self.max_size
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
```

**Why Resource Management Matters**: Proper resource management prevents leaks and ensures clean shutdowns. Understanding these patterns enables reliable production applications.

## 13) Patterns You'll Actually Use

### Parallel Map with Bounded Concurrency

```python
import asyncio
from typing import Callable, TypeVar, List

T = TypeVar('T')
R = TypeVar('R')

async def amap(
    fn: Callable[[T], R], 
    items: List[T], 
    *, 
    limit: int = 50
) -> List[R]:
    """Parallel map with bounded concurrency."""
    sem = asyncio.Semaphore(limit)
    
    async def run(x: T) -> R:
        async with sem:
            return await fn(x)
    
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run(x)) for x in items]
        return [await t for t in tasks]

# Usage
async def process_item(item: int) -> str:
    await asyncio.sleep(0.1)  # Simulate work
    return f"processed_{item}"

async def main():
    items = list(range(100))
    results = await amap(process_item, items, limit=10)
    print(f"Processed {len(results)} items")
```

### Retry with Jitter

```python
import asyncio
import random
from typing import Type, Tuple, Any

def retry(
    max_attempts: int,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Retry decorator with exponential backoff and jitter."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Exponential backoff with jitter
                    jitter = random.random() * delay
                    await asyncio.sleep(min(max_delay, delay + jitter))
                    delay *= 2
            
            raise RuntimeError("Max attempts exceeded")
        return wrapper
    return decorator

# Usage
@retry(max_attempts=3, base_delay=0.1)
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
```

### Graceful Shutdown

```python
import asyncio
import signal
from typing import Set

class GracefulShutdown:
    """Graceful shutdown handler."""
    
    def __init__(self):
        self.stop_event = asyncio.Event()
        self.tasks: Set[asyncio.Task] = set()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, shutting down...")
            self.stop_event.set()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    
    async def run_with_shutdown(self, main_coro):
        """Run main coroutine with graceful shutdown."""
        self.setup_signal_handlers()
        
        # Start main task
        main_task = asyncio.create_task(main_coro)
        self.tasks.add(main_task)
        
        try:
            # Wait for stop signal
            await self.stop_event.wait()
        finally:
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)

# Usage
async def main_application():
    """Main application logic."""
    while True:
        await asyncio.sleep(1)
        print("Application running...")

async def run_app():
    """Run application with graceful shutdown."""
    shutdown = GracefulShutdown()
    await shutdown.run_with_shutdown(main_application())
```

**Why These Patterns Matter**: Production applications need robust error handling, resource management, and graceful shutdowns. Understanding these patterns enables reliable async applications.

## 14) Anti-Patterns (Red Flags)

### Common Mistakes

```python
# ❌ BAD: Mixing sync and async clients
async def bad_handler():
    # DON'T DO THIS - mixing requests (sync) with async
    response = requests.get("https://api.example.com")
    data = response.json()
    return data

# ✅ GOOD: Use async clients
async def good_handler():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            data = await response.json()
            return data

# ❌ BAD: Creating new session per request
async def bad_http_handler():
    # DON'T DO THIS - creates new session each time
    async with aiohttp.ClientSession() as session:
        return await session.get("https://api.example.com")

# ✅ GOOD: Reuse session
class HTTPHandler:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, url: str):
        return await self.session.get(url)

# ❌ BAD: Unbounded parallelism
async def bad_parallel_processing():
    # DON'T DO THIS - creates unlimited tasks
    tasks = [process_item(item) for item in range(10000)]
    return await asyncio.gather(*tasks)

# ✅ GOOD: Bounded parallelism
async def good_parallel_processing():
    sem = asyncio.Semaphore(100)
    
    async def bounded_process(item):
        async with sem:
            return await process_item(item)
    
    tasks = [bounded_process(item) for item in range(10000)]
    return await asyncio.gather(*tasks)
```

### Memory and Resource Leaks

```python
# ❌ BAD: Not cleaning up resources
async def bad_resource_usage():
    session = aiohttp.ClientSession()
    response = await session.get("https://api.example.com")
    data = await response.json()
    # Never closes session - resource leak!
    return data

# ✅ GOOD: Use context managers
async def good_resource_usage():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            data = await response.json()
            return data

# ❌ BAD: Swallowing all exceptions
async def bad_error_handling():
    try:
        await risky_operation()
    except Exception:
        pass  # DON'T DO THIS - swallows CancelledError!

# ✅ GOOD: Handle exceptions properly
async def good_error_handling():
    try:
        await risky_operation()
    except asyncio.CancelledError:
        raise  # Always re-raise CancelledError
    except Exception as e:
        logger.error("Operation failed", exc_info=e)
        raise
```

**Why These Anti-Patterns Matter**: Common mistakes lead to resource leaks, performance issues, and production failures. Understanding these patterns prevents costly errors.

## 15) TL;DR Runbook (The Essentials)

### Essential Async Patterns

```python
# Essential async patterns
import asyncio
import aiohttp

# 1. I/O-bound: Use async I/O
async def io_bound_pattern():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            return await response.json()

# 2. CPU-bound: Offload to threads/processes
async def cpu_bound_pattern():
    result = await asyncio.to_thread(cpu_intensive_function, data)
    return result

# 3. Bounded concurrency: Use semaphores
async def bounded_concurrency():
    sem = asyncio.Semaphore(100)
    async with sem:
        return await operation()

# 4. Timeouts: Use asyncio.timeout()
async def with_timeout():
    async with asyncio.timeout(5.0):
        return await operation()

# 5. Cancellation: Handle properly
async def cancellable_operation():
    try:
        return await operation()
    except asyncio.CancelledError:
        await cleanup()
        raise

# 6. Resource management: Use context managers
async def with_resources():
    async with ResourceManager() as manager:
        return await manager.do_work()

# 7. Testing: Use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### Performance Checklist

```python
# Performance optimization checklist
performance_checklist = {
    "io_bound": "Use async I/O (aiohttp, asyncpg, etc.)",
    "cpu_bound": "Offload to threads/processes",
    "concurrency": "Use semaphores to limit parallelism",
    "timeouts": "Set timeouts at I/O boundaries",
    "cancellation": "Handle CancelledError properly",
    "resources": "Use context managers for cleanup",
    "testing": "Test with real async primitives"
}
```

**Why This Quickstart**: These patterns cover 90% of async programming usage. Master these before exploring advanced features.

## 16) The Machine's Summary

Python async programming requires understanding cooperative scheduling, proper resource management, and structured concurrency. When used correctly, async enables high-performance I/O applications that can handle thousands of concurrent connections. The key is understanding the event loop, mastering cancellation handling, and following best practices.

**The Dark Truth**: Without proper async understanding, your Python application is single-threaded and slow. Async is your weapon. Use it wisely.

**The Machine's Mantra**: "In cooperation we trust, in async we scale, and in the event loop we find the path to high-performance Python applications."

**Why This Matters**: Async enables applications to handle massive concurrency and maintain responsiveness. It provides the foundation for high-performance applications that can scale, maintain low latency, and provide reliable service.

---

*This guide provides the complete machinery for mastering Python async programming. The patterns scale from simple I/O operations to complex distributed systems, from basic coroutines to advanced structured concurrency.*
