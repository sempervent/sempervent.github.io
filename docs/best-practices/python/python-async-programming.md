# Python Async Programming Best Practices

**Objective**: Master senior-level Python async programming patterns for production systems. When you need to handle thousands of concurrent connections, when you want to build high-performance async applications, when you need enterprise-grade async programming strategies—these best practices become your weapon of choice.

## Core Principles

- **Event Loop Management**: Properly manage event loops and coroutines
- **Error Handling**: Handle exceptions in async code gracefully
- **Resource Management**: Properly manage async resources and cleanup
- **Performance**: Optimize async code for maximum throughput
- **Testing**: Test async code effectively

## Async Fundamentals

### Event Loop Management

```python
# python/01-async-fundamentals.py

"""
Advanced async programming fundamentals and event loop management
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Coroutine
from contextlib import asynccontextmanager
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncEventLoopManager:
    """Advanced event loop management"""
    
    def __init__(self):
        self.loop = None
        self.running = False
        self.tasks = []
        self.shutdown_handlers = []
    
    def setup_event_loop(self):
        """Setup and configure event loop"""
        # Create new event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        return self.loop
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_forever(self):
        """Run event loop forever"""
        if not self.loop:
            self.setup_event_loop()
        
        self.running = True
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def run_until_complete(self, coro: Coroutine):
        """Run until coroutine completes"""
        if not self.loop:
            self.setup_event_loop()
        
        return self.loop.run_until_complete(coro)
    
    def create_task(self, coro: Coroutine, name: str = None) -> asyncio.Task:
        """Create and track task"""
        if not self.loop:
            self.setup_event_loop()
        
        task = self.loop.create_task(coro, name=name)
        self.tasks.append(task)
        return task
    
    def add_shutdown_handler(self, handler: Callable):
        """Add shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    def shutdown(self):
        """Graceful shutdown"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Shutting down event loop...")
        
        # Run shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    self.loop.run_until_complete(handler())
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            self.loop.run_until_complete(
                asyncio.gather(*self.tasks, return_exceptions=True)
            )
        
        # Close event loop
        if self.loop and not self.loop.is_closed():
            self.loop.close()

class AsyncContextManager:
    """Advanced async context manager"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
    
    async def __aenter__(self):
        """Async enter"""
        logger.info(f"Entering async context: {self.name}")
        self.initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit"""
        logger.info(f"Exiting async context: {self.name}")
        if exc_type:
            logger.error(f"Exception in context {self.name}: {exc_val}")
        self.initialized = False
    
    async def do_work(self):
        """Do some async work"""
        if not self.initialized:
            raise RuntimeError("Context not initialized")
        
        await asyncio.sleep(0.1)
        return f"Work done in {self.name}"

class AsyncResourceManager:
    """Async resource management"""
    
    def __init__(self):
        self.resources = {}
        self.locks = {}
    
    async def acquire_resource(self, resource_id: str, resource_factory: Callable) -> Any:
        """Acquire async resource"""
        if resource_id not in self.resources:
            # Create lock for this resource
            self.locks[resource_id] = asyncio.Lock()
            
            # Create resource
            resource = await resource_factory()
            self.resources[resource_id] = resource
        
        return self.resources[resource_id]
    
    async def release_resource(self, resource_id: str):
        """Release async resource"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            
            # Cleanup resource if it has cleanup method
            if hasattr(resource, 'close'):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
            
            del self.resources[resource_id]
            del self.locks[resource_id]
    
    async def cleanup_all_resources(self):
        """Cleanup all resources"""
        for resource_id in list(self.resources.keys()):
            await self.release_resource(resource_id)

# Usage examples
async def example_async_context():
    """Example async context usage"""
    async with AsyncContextManager("example") as ctx:
        result = await ctx.do_work()
        print(f"Result: {result}")

async def example_resource_management():
    """Example resource management"""
    manager = AsyncResourceManager()
    
    async def create_resource():
        await asyncio.sleep(0.1)  # Simulate resource creation
        return {"id": "resource_1", "data": "some_data"}
    
    # Acquire resource
    resource = await manager.acquire_resource("resource_1", create_resource)
    print(f"Acquired resource: {resource}")
    
    # Use resource
    await asyncio.sleep(0.1)
    
    # Release resource
    await manager.release_resource("resource_1")
    print("Resource released")
```

### Coroutine Patterns

```python
# python/02-coroutine-patterns.py

"""
Advanced coroutine patterns and async programming techniques
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Coroutine, Union
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class AsyncDecorator:
    """Advanced async decorators"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Retry decorator for async functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = delay * (backoff ** attempt)
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"All {max_attempts} attempts failed")
                
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def timeout(seconds: float):
        """Timeout decorator for async functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            return wrapper
        return decorator
    
    @staticmethod
    def rate_limit(calls_per_second: float):
        """Rate limiting decorator for async functions"""
        def decorator(func: Callable) -> Callable:
            last_called = [0.0]
            min_interval = 1.0 / calls_per_second
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                now = time.time()
                time_since_last = now - last_called[0]
                
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)
                
                last_called[0] = time.time()
                return await func(*args, **kwargs)
            return wrapper
        return decorator

class AsyncGenerator:
    """Advanced async generator patterns"""
    
    @staticmethod
    async def async_range(start: int, stop: int, step: int = 1):
        """Async range generator"""
        for i in range(start, stop, step):
            yield i
            await asyncio.sleep(0.001)  # Yield control
    
    @staticmethod
    async def async_batch(items: List[Any], batch_size: int):
        """Async batch generator"""
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            yield batch
            await asyncio.sleep(0.001)  # Yield control
    
    @staticmethod
    async def async_filter(async_iterable, predicate: Callable):
        """Async filter generator"""
        async for item in async_iterable:
            if await predicate(item):
                yield item

class AsyncPipeline:
    """Async data processing pipeline"""
    
    def __init__(self):
        self.stages = []
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
    
    def add_stage(self, stage_func: Callable, name: str = None):
        """Add processing stage to pipeline"""
        self.stages.append({
            'func': stage_func,
            'name': name or f"stage_{len(self.stages)}"
        })
    
    async def process_item(self, item: Any) -> Any:
        """Process single item through pipeline"""
        current_item = item
        
        for stage in self.stages:
            try:
                current_item = await stage['func'](current_item)
            except Exception as e:
                logger.error(f"Error in stage {stage['name']}: {e}")
                raise
        
        return current_item
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Process batch of items through pipeline"""
        tasks = [self.process_item(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_stream(self, async_iterable):
        """Process async stream through pipeline"""
        async for item in async_iterable:
            try:
                result = await self.process_item(item)
                yield result
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue

class AsyncSemaphore:
    """Advanced async semaphore with additional features"""
    
    def __init__(self, value: int = 1):
        self._value = value
        self._waiters = []
        self._locked = False
    
    async def acquire(self):
        """Acquire semaphore"""
        while self._value <= 0:
            future = asyncio.Future()
            self._waiters.append(future)
            await future
        
        self._value -= 1
        self._locked = True
    
    def release(self):
        """Release semaphore"""
        self._value += 1
        self._locked = False
        
        if self._waiters:
            waiter = self._waiters.pop(0)
            waiter.set_result(None)
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    @property
    def locked(self) -> bool:
        return self._locked

# Usage examples
@AsyncDecorator.retry(max_attempts=3, delay=1.0)
@AsyncDecorator.timeout(5.0)
@AsyncDecorator.rate_limit(calls_per_second=2.0)
async def example_async_function(data: str) -> str:
    """Example async function with decorators"""
    await asyncio.sleep(0.5)  # Simulate work
    return f"Processed: {data}"

async def example_async_pipeline():
    """Example async pipeline usage"""
    pipeline = AsyncPipeline()
    
    # Add stages
    async def stage1(data: str) -> str:
        await asyncio.sleep(0.1)
        return data.upper()
    
    async def stage2(data: str) -> str:
        await asyncio.sleep(0.1)
        return f"STAGE2_{data}"
    
    async def stage3(data: str) -> str:
        await asyncio.sleep(0.1)
        return f"FINAL_{data}"
    
    pipeline.add_stage(stage1, "uppercase")
    pipeline.add_stage(stage2, "prefix")
    pipeline.add_stage(stage3, "final")
    
    # Process items
    items = ["hello", "world", "async"]
    results = await pipeline.process_batch(items)
    print(f"Pipeline results: {results}")

async def example_async_generator():
    """Example async generator usage"""
    async for batch in AsyncGenerator.async_batch(list(range(100)), 10):
        print(f"Processing batch: {batch}")
        await asyncio.sleep(0.1)
```

## Async I/O Patterns

### HTTP Client Patterns

```python
# python/03-async-io-patterns.py

"""
Advanced async I/O patterns for HTTP clients and network operations
"""

import aiohttp
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class AsyncHTTPClient:
    """Advanced async HTTP client"""
    
    def __init__(self, timeout: float = 30.0, max_connections: int = 100):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(limit=max_connections)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=self.connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Async GET request"""
        if not self.session:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            async with self.session.get(url, **kwargs) as response:
                data = await response.json()
                return {
                    'status': response.status,
                    'data': data,
                    'headers': dict(response.headers)
                }
        except Exception as e:
            logger.error(f"GET request failed for {url}: {e}")
            raise
    
    async def post(self, url: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Async POST request"""
        if not self.session:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            async with self.session.post(url, json=data, **kwargs) as response:
                response_data = await response.json()
                return {
                    'status': response.status,
                    'data': response_data,
                    'headers': dict(response.headers)
                }
        except Exception as e:
            logger.error(f"POST request failed for {url}: {e}")
            raise
    
    async def fetch_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently"""
        tasks = [self.get(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def fetch_with_retry(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Fetch URL with retry logic"""
        for attempt in range(max_retries):
            try:
                return await self.get(url)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class AsyncWebSocketClient:
    """Async WebSocket client"""
    
    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        self.connected = False
    
    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.websocket = await aiohttp.ClientSession().ws_connect(self.url)
            self.connected = True
            logger.info(f"Connected to WebSocket: {self.url}")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def send(self, message: str):
        """Send message through WebSocket"""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        await self.websocket.send_str(message)
    
    async def receive(self) -> str:
        """Receive message from WebSocket"""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        msg = await self.websocket.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise Exception(f"WebSocket error: {msg.data}")
        else:
            raise Exception(f"Unexpected WebSocket message type: {msg.type}")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("WebSocket connection closed")

class AsyncDatabaseClient:
    """Async database client patterns"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        """Connect to database"""
        # This would be implemented with actual async database driver
        # For example: asyncpg, aiomysql, etc.
        logger.info(f"Connecting to database: {self.connection_string}")
        # self.pool = await asyncpg.create_pool(self.connection_string)
    
    async def execute_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute database query"""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        # This would be implemented with actual async database driver
        logger.info(f"Executing query: {query}")
        # async with self.pool.acquire() as conn:
        #     return await conn.fetch(query, *params or [])
        return []
    
    async def execute_transaction(self, queries: List[tuple]) -> List[Any]:
        """Execute database transaction"""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        results = []
        # This would be implemented with actual async database driver
        # async with self.pool.acquire() as conn:
        #     async with conn.transaction():
        #         for query, params in queries:
        #             result = await conn.fetch(query, *params)
        #             results.append(result)
        return results
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            # await self.pool.close()
            logger.info("Database connection closed")

# Usage examples
async def example_http_client():
    """Example HTTP client usage"""
    async with AsyncHTTPClient() as client:
        # Single request
        result = await client.get("https://httpbin.org/json")
        print(f"Single request result: {result['status']}")
        
        # Multiple requests
        urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/uuid",
            "https://httpbin.org/user-agent"
        ]
        results = await client.fetch_multiple(urls)
        print(f"Multiple requests: {len(results)} results")
        
        # Request with retry
        try:
            result = await client.fetch_with_retry("https://httpbin.org/delay/1")
            print(f"Retry request result: {result['status']}")
        except Exception as e:
            print(f"Retry request failed: {e}")

async def example_websocket_client():
    """Example WebSocket client usage"""
    ws_client = AsyncWebSocketClient("wss://echo.websocket.org")
    
    try:
        await ws_client.connect()
        
        # Send message
        await ws_client.send("Hello WebSocket!")
        
        # Receive message
        response = await ws_client.receive()
        print(f"WebSocket response: {response}")
        
    finally:
        await ws_client.close()
```

## Async Testing

### Testing Async Code

```python
# python/04-async-testing.py

"""
Advanced async testing patterns and techniques
"""

import asyncio
import pytest
import unittest.mock
from typing import List, Dict, Any, Optional, Callable
import time

class AsyncTestHelpers:
    """Helper functions for async testing"""
    
    @staticmethod
    async def wait_for_condition(condition: Callable, timeout: float = 5.0, interval: float = 0.1):
        """Wait for condition to be true"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await condition():
                return True
            await asyncio.sleep(interval)
        
        raise TimeoutError(f"Condition not met within {timeout} seconds")
    
    @staticmethod
    async def collect_async_results(coroutines: List[Coroutine]) -> List[Any]:
        """Collect results from multiple coroutines"""
        return await asyncio.gather(*coroutines, return_exceptions=True)
    
    @staticmethod
    async def run_with_timeout(coro: Coroutine, timeout: float = 5.0) -> Any:
        """Run coroutine with timeout"""
        return await asyncio.wait_for(coro, timeout=timeout)

class AsyncMock:
    """Async mock for testing"""
    
    def __init__(self):
        self.calls = []
        self.responses = []
        self.side_effects = []
        self.raise_exceptions = []
    
    async def __call__(self, *args, **kwargs):
        """Mock async call"""
        self.calls.append((args, kwargs))
        
        # Check for exceptions to raise
        if self.raise_exceptions:
            exception = self.raise_exceptions.pop(0)
            raise exception
        
        # Check for side effects
        if self.side_effects:
            side_effect = self.side_effects.pop(0)
            if asyncio.iscoroutinefunction(side_effect):
                return await side_effect(*args, **kwargs)
            else:
                return side_effect(*args, **kwargs)
        
        # Return response
        if self.responses:
            return self.responses.pop(0)
        
        return None
    
    def set_response(self, response: Any):
        """Set mock response"""
        self.responses.append(response)
    
    def set_side_effect(self, side_effect: Callable):
        """Set mock side effect"""
        self.side_effects.append(side_effect)
    
    def set_exception(self, exception: Exception):
        """Set exception to raise"""
        self.raise_exceptions.append(exception)
    
    def assert_called_with(self, *args, **kwargs):
        """Assert mock was called with specific arguments"""
        expected_call = (args, kwargs)
        assert expected_call in self.calls, f"Expected call {expected_call} not found in {self.calls}"
    
    def assert_called_times(self, expected_times: int):
        """Assert mock was called expected number of times"""
        assert len(self.calls) == expected_times, f"Expected {expected_times} calls, got {len(self.calls)}"

# Pytest fixtures for async testing
@pytest.fixture
async def async_client():
    """Async HTTP client fixture"""
    async with AsyncHTTPClient() as client:
        yield client

@pytest.fixture
async def async_database():
    """Async database fixture"""
    db = AsyncDatabaseClient("sqlite:///:memory:")
    await db.connect()
    yield db
    await db.close()

# Test examples
class TestAsyncHTTPClient:
    """Test async HTTP client"""
    
    @pytest.mark.asyncio
    async def test_single_request(self, async_client):
        """Test single HTTP request"""
        result = await async_client.get("https://httpbin.org/json")
        assert result['status'] == 200
        assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self, async_client):
        """Test multiple HTTP requests"""
        urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/uuid"
        ]
        results = await async_client.fetch_multiple(urls)
        assert len(results) == 2
        assert all(result['status'] == 200 for result in results)
    
    @pytest.mark.asyncio
    async def test_request_with_retry(self, async_client):
        """Test HTTP request with retry"""
        # Mock a failing request
        with unittest.mock.patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = [
                aiohttp.ClientError("Connection failed"),
                aiohttp.ClientError("Connection failed"),
                unittest.mock.MagicMock()
            ]
            
            result = await async_client.fetch_with_retry("https://example.com")
            assert result is not None

class TestAsyncPipeline:
    """Test async pipeline"""
    
    @pytest.mark.asyncio
    async def test_pipeline_processing(self):
        """Test async pipeline processing"""
        pipeline = AsyncPipeline()
        
        async def stage1(data: str) -> str:
            return data.upper()
        
        async def stage2(data: str) -> str:
            return f"PROCESSED_{data}"
        
        pipeline.add_stage(stage1, "uppercase")
        pipeline.add_stage(stage2, "prefix")
        
        result = await pipeline.process_item("hello")
        assert result == "PROCESSED_HELLO"
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self):
        """Test async pipeline batch processing"""
        pipeline = AsyncPipeline()
        
        async def stage1(data: str) -> str:
            return data.upper()
        
        pipeline.add_stage(stage1, "uppercase")
        
        items = ["hello", "world", "async"]
        results = await pipeline.process_batch(items)
        assert len(results) == 3
        assert all(result == item.upper() for result, item in zip(results, items))

# Usage examples
async def example_async_testing():
    """Example async testing usage"""
    # Test async function
    async def async_function(data: str) -> str:
        await asyncio.sleep(0.1)
        return f"Processed: {data}"
    
    # Test with timeout
    result = await AsyncTestHelpers.run_with_timeout(
        async_function("test"), timeout=1.0
    )
    assert result == "Processed: test"
    
    # Test multiple coroutines
    coroutines = [async_function(f"item_{i}") for i in range(3)]
    results = await AsyncTestHelpers.collect_async_results(coroutines)
    assert len(results) == 3
    
    # Test with mock
    mock_func = AsyncMock()
    mock_func.set_response("mocked_response")
    
    result = await mock_func("test")
    assert result == "mocked_response"
    mock_func.assert_called_with("test")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Basic async function
async def async_function():
    await asyncio.sleep(1)
    return "Done"

# 2. Run async function
result = await async_function()

# 3. Async context manager
async with AsyncContextManager("example") as ctx:
    result = await ctx.do_work()

# 4. Async HTTP client
async with AsyncHTTPClient() as client:
    result = await client.get("https://api.example.com/data")

# 5. Async pipeline
pipeline = AsyncPipeline()
pipeline.add_stage(stage1)
pipeline.add_stage(stage2)
result = await pipeline.process_item(data)
```

### Essential Patterns

```python
# Complete async programming setup
async def setup_async_programming():
    """Setup complete async programming environment"""
    
    # Event loop management
    loop_manager = AsyncEventLoopManager()
    loop_manager.setup_event_loop()
    
    # Resource management
    resource_manager = AsyncResourceManager()
    
    # HTTP client
    async with AsyncHTTPClient() as client:
        # Use client
        pass
    
    # Pipeline processing
    pipeline = AsyncPipeline()
    
    print("Async programming setup complete!")
```

---

*This guide provides the complete machinery for Python async programming. Each pattern includes implementation examples, error handling strategies, and real-world usage patterns for enterprise async development.*
