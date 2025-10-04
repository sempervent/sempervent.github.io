# Python Concurrency Patterns Best Practices

**Objective**: Master senior-level Python concurrency patterns for production systems. When you need to handle multiple tasks efficiently, when you want to leverage parallel processing, when you need enterprise-grade concurrency strategies—these best practices become your weapon of choice.

## Core Principles

- **Choose the Right Tool**: Use threading for I/O-bound, multiprocessing for CPU-bound tasks
- **Avoid Race Conditions**: Use proper synchronization mechanisms
- **Resource Management**: Properly manage shared resources and cleanup
- **Error Handling**: Handle exceptions in concurrent code gracefully
- **Performance**: Balance concurrency overhead with performance gains

## Threading Patterns

### Thread Pool Management

```python
# python/01-threading-patterns.py

"""
Advanced threading patterns and thread pool management
"""

import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreadPoolManager:
    """Advanced thread pool management"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = None
        self.active_tasks = {}
        self.task_counter = 0
    
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the thread pool"""
        if not self.executor:
            raise RuntimeError("ThreadPoolManager not initialized")
        
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        future = self.executor.submit(func, *args, **kwargs)
        self.active_tasks[task_id] = future
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result from a task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.active_tasks[task_id]
        try:
            result = future.result(timeout=timeout)
            del self.active_tasks[task_id]
            return result
        except Exception as e:
            del self.active_tasks[task_id]
            raise e
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all tasks to complete"""
        results = {}
        
        for task_id, future in self.active_tasks.items():
            try:
                result = future.result(timeout=timeout)
                results[task_id] = {"status": "success", "result": result}
            except Exception as e:
                results[task_id] = {"status": "error", "error": str(e)}
        
        self.active_tasks.clear()
        return results

class ThreadSafeCounter:
    """Thread-safe counter implementation"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter thread-safely"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter thread-safely"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get_value(self) -> int:
        """Get current counter value"""
        with self._lock:
            return self._value
    
    def reset(self) -> int:
        """Reset counter to zero"""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value

class ThreadSafeQueue:
    """Thread-safe queue with additional features"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._stats = {"put_count": 0, "get_count": 0, "errors": 0}
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        """Put item in queue with statistics"""
        try:
            self._queue.put(item, block=block, timeout=timeout)
            with self._lock:
                self._stats["put_count"] += 1
        except queue.Full:
            with self._lock:
                self._stats["errors"] += 1
            raise
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Get item from queue with statistics"""
        try:
            item = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                self._stats["get_count"] += 1
            return item
        except queue.Empty:
            with self._lock:
                self._stats["errors"] += 1
            raise
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            return self._stats.copy()
    
    def size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

class ProducerConsumer:
    """Producer-consumer pattern implementation"""
    
    def __init__(self, queue_size: int = 10):
        self.queue = ThreadSafeQueue(maxsize=queue_size)
        self.producers = []
        self.consumers = []
        self.running = False
    
    def add_producer(self, producer_func: Callable, *args, **kwargs):
        """Add a producer function"""
        self.producers.append((producer_func, args, kwargs))
    
    def add_consumer(self, consumer_func: Callable, *args, **kwargs):
        """Add a consumer function"""
        self.consumers.append((consumer_func, args, kwargs))
    
    def start(self):
        """Start producer-consumer system"""
        self.running = True
        
        # Start producer threads
        for producer_func, args, kwargs in self.producers:
            thread = threading.Thread(
                target=self._producer_worker,
                args=(producer_func, args, kwargs)
            )
            thread.daemon = True
            thread.start()
        
        # Start consumer threads
        for consumer_func, args, kwargs in self.consumers:
            thread = threading.Thread(
                target=self._consumer_worker,
                args=(consumer_func, args, kwargs)
            )
            thread.daemon = True
            thread.start()
    
    def stop(self):
        """Stop producer-consumer system"""
        self.running = False
    
    def _producer_worker(self, producer_func: Callable, args: tuple, kwargs: dict):
        """Producer worker thread"""
        while self.running:
            try:
                item = producer_func(*args, **kwargs)
                if item is not None:
                    self.queue.put(item)
            except Exception as e:
                logger.error(f"Producer error: {e}")
                time.sleep(0.1)
    
    def _consumer_worker(self, consumer_func: Callable, args: tuple, kwargs: dict):
        """Consumer worker thread"""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                consumer_func(item, *args, **kwargs)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Consumer error: {e}")

# Usage examples
def example_producer() -> str:
    """Example producer function"""
    time.sleep(0.1)  # Simulate work
    return f"Item_{time.time()}"

def example_consumer(item: str):
    """Example consumer function"""
    print(f"Processing: {item}")
    time.sleep(0.05)  # Simulate work

# Producer-consumer example
pc = ProducerConsumer(queue_size=5)
pc.add_producer(example_producer)
pc.add_consumer(example_consumer)
pc.start()

# Let it run for a bit
time.sleep(2)
pc.stop()
```

### Thread Synchronization

```python
# python/02-thread-synchronization.py

"""
Advanced thread synchronization patterns
"""

import threading
import time
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import random

class ReadWriteLock:
    """Read-write lock implementation"""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    
    def acquire_read(self):
        """Acquire read lock"""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        """Release read lock"""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        """Acquire write lock"""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    
    def release_write(self):
        """Release write lock"""
        self._read_ready.release()
    
    @contextmanager
    def read_lock(self):
        """Context manager for read lock"""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()
    
    @contextmanager
    def write_lock(self):
        """Context manager for write lock"""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

class ThreadSafeCache:
    """Thread-safe cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._max_size = max_size
        self._lock = threading.RLock()
        self._access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    del self._cache[oldest_key]
            
            self._cache[key] = value
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self._lock:
            return len(self._cache)

class Barrier:
    """Thread barrier implementation"""
    
    def __init__(self, count: int):
        self.count = count
        self.current_count = 0
        self.condition = threading.Condition()
        self.generation = 0
    
    def wait(self) -> bool:
        """Wait at barrier"""
        with self.condition:
            current_generation = self.generation
            self.current_count += 1
            
            if self.current_count == self.count:
                self.current_count = 0
                self.generation += 1
                self.condition.notify_all()
                return True
            else:
                while current_generation == self.generation:
                    self.condition.wait()
                return False

class Semaphore:
    """Semaphore implementation"""
    
    def __init__(self, value: int = 1):
        self._value = value
        self._condition = threading.Condition()
    
    def acquire(self, blocking: bool = True) -> bool:
        """Acquire semaphore"""
        with self._condition:
            if self._value > 0:
                self._value -= 1
                return True
            elif not blocking:
                return False
            else:
                while self._value <= 0:
                    self._condition.wait()
                self._value -= 1
                return True
    
    def release(self) -> None:
        """Release semaphore"""
        with self._condition:
            self._value += 1
            self._condition.notify()
    
    @contextmanager
    def acquire_context(self):
        """Context manager for semaphore"""
        self.acquire()
        try:
            yield
        finally:
            self.release()

# Usage examples
def worker_with_barrier(barrier: Barrier, worker_id: int):
    """Worker function using barrier"""
    print(f"Worker {worker_id} starting")
    time.sleep(random.uniform(0.1, 0.5))  # Simulate work
    print(f"Worker {worker_id} waiting at barrier")
    barrier.wait()
    print(f"Worker {worker_id} passed barrier")

def worker_with_semaphore(semaphore: Semaphore, worker_id: int):
    """Worker function using semaphore"""
    with semaphore.acquire_context():
        print(f"Worker {worker_id} acquired semaphore")
        time.sleep(random.uniform(0.1, 0.3))  # Simulate work
        print(f"Worker {worker_id} releasing semaphore")
```

## Multiprocessing Patterns

### Process Pool Management

```python
# python/03-multiprocessing-patterns.py

"""
Advanced multiprocessing patterns and process pool management
"""

import multiprocessing
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict
import logging
from multiprocessing import Queue, Process, Value, Array, Manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessPoolManager:
    """Advanced process pool management"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = None
        self.active_processes = {}
        self.process_counter = 0
    
    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the process pool"""
        if not self.executor:
            raise RuntimeError("ProcessPoolManager not initialized")
        
        task_id = f"process_{self.process_counter}"
        self.process_counter += 1
        
        future = self.executor.submit(func, *args, **kwargs)
        self.active_processes[task_id] = future
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result from a task"""
        if task_id not in self.active_processes:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.active_processes[task_id]
        try:
            result = future.result(timeout=timeout)
            del self.active_processes[task_id]
            return result
        except Exception as e:
            del self.active_processes[task_id]
            raise e
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all tasks to complete"""
        results = {}
        
        for task_id, future in self.active_processes.items():
            try:
                result = future.result(timeout=timeout)
                results[task_id] = {"status": "success", "result": result}
            except Exception as e:
                results[task_id] = {"status": "error", "error": str(e)}
        
        self.active_processes.clear()
        return results

class SharedMemoryManager:
    """Shared memory management for multiprocessing"""
    
    def __init__(self):
        self.shared_arrays = {}
        self.shared_values = {}
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.shared_list = self.manager.list()
    
    def create_shared_array(self, name: str, size: int, data_type: str = 'i') -> Array:
        """Create shared array"""
        if data_type == 'i':
            array = Array('i', size)
        elif data_type == 'd':
            array = Array('d', size)
        elif data_type == 'c':
            array = Array('c', size)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        self.shared_arrays[name] = array
        return array
    
    def create_shared_value(self, name: str, initial_value: int = 0) -> Value:
        """Create shared value"""
        value = Value('i', initial_value)
        self.shared_values[name] = value
        return value
    
    def get_shared_dict(self) -> Dict:
        """Get shared dictionary"""
        return self.shared_dict
    
    def get_shared_list(self) -> List:
        """Get shared list"""
        return self.shared_list

class ProcessCommunication:
    """Inter-process communication patterns"""
    
    def __init__(self):
        self.queues = {}
        self.pipes = {}
        self.events = {}
    
    def create_queue(self, name: str) -> Queue:
        """Create inter-process queue"""
        queue = Queue()
        self.queues[name] = queue
        return queue
    
    def create_pipe(self, name: str) -> tuple:
        """Create inter-process pipe"""
        parent_conn, child_conn = multiprocessing.Pipe()
        self.pipes[name] = (parent_conn, child_conn)
        return parent_conn, child_conn
    
    def create_event(self, name: str) -> multiprocessing.Event:
        """Create inter-process event"""
        event = multiprocessing.Event()
        self.events[name] = event
        return event
    
    def send_message(self, queue_name: str, message: Any) -> None:
        """Send message through queue"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found")
        self.queues[queue_name].put(message)
    
    def receive_message(self, queue_name: str, timeout: Optional[float] = None) -> Any:
        """Receive message from queue"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found")
        return self.queues[queue_name].get(timeout=timeout)

def cpu_intensive_task(data: List[int]) -> int:
    """CPU-intensive task for multiprocessing"""
    result = 0
    for i in data:
        result += i * i
    return result

def worker_process(worker_id: int, shared_array: Array, shared_value: Value):
    """Worker process function"""
    print(f"Worker {worker_id} starting")
    
    # Work with shared memory
    for i in range(10):
        with shared_value.get_lock():
            shared_value.value += 1
        
        shared_array[worker_id] = worker_id * 10 + i
        time.sleep(0.1)
    
    print(f"Worker {worker_id} finished")

# Usage examples
def example_multiprocessing():
    """Example multiprocessing usage"""
    # Create shared memory
    shared_memory = SharedMemoryManager()
    shared_array = shared_memory.create_shared_array("data", 10)
    shared_value = shared_memory.create_shared_value("counter", 0)
    
    # Create processes
    processes = []
    for i in range(3):
        process = Process(
            target=worker_process,
            args=(i, shared_array, shared_value)
        )
        processes.append(process)
        process.start()
    
    # Wait for processes to complete
    for process in processes:
        process.join()
    
    print(f"Final counter value: {shared_value.value}")
    print(f"Shared array: {list(shared_array)}")

# Process pool example
def example_process_pool():
    """Example process pool usage"""
    data = [list(range(i, i + 100)) for i in range(0, 1000, 100)]
    
    with ProcessPoolManager(max_workers=4) as pool:
        task_ids = []
        for chunk in data:
            task_id = pool.submit_task(cpu_intensive_task, chunk)
            task_ids.append(task_id)
        
        # Get results
        results = pool.wait_for_all_tasks()
        print(f"Processed {len(results)} tasks")
```

## Async Patterns

### Async Concurrency

```python
# python/04-async-patterns.py

"""
Advanced async concurrency patterns
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Callable
import logging
from asyncio import Semaphore, Queue, Event, Lock
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncTaskManager:
    """Async task management"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.active_tasks = {}
        self.task_counter = 0
    
    async def submit_task(self, coro: Callable, *args, **kwargs) -> str:
        """Submit async task"""
        task_id = f"async_task_{self.task_counter}"
        self.task_counter += 1
        
        async def wrapped_task():
            async with self.semaphore:
                return await coro(*args, **kwargs)
        
        task = asyncio.create_task(wrapped_task())
        self.active_tasks[task_id] = task
        return task_id
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get task result"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        result = await task
        del self.active_tasks[task_id]
        return result
    
    async def wait_for_all_tasks(self) -> Dict[str, Any]:
        """Wait for all tasks to complete"""
        if not self.active_tasks:
            return {}
        
        results = {}
        for task_id, task in self.active_tasks.items():
            try:
                result = await task
                results[task_id] = {"status": "success", "result": result}
            except Exception as e:
                results[task_id] = {"status": "error", "error": str(e)}
        
        self.active_tasks.clear()
        return results

class AsyncProducerConsumer:
    """Async producer-consumer pattern"""
    
    def __init__(self, queue_size: int = 100):
        self.queue = Queue(maxsize=queue_size)
        self.producers = []
        self.consumers = []
        self.running = False
        self.shutdown_event = Event()
    
    def add_producer(self, producer_coro: Callable, *args, **kwargs):
        """Add producer coroutine"""
        self.producers.append((producer_coro, args, kwargs))
    
    def add_consumer(self, consumer_coro: Callable, *args, **kwargs):
        """Add consumer coroutine"""
        self.consumers.append((consumer_coro, args, kwargs))
    
    async def start(self):
        """Start producer-consumer system"""
        self.running = True
        
        # Start producer tasks
        producer_tasks = []
        for producer_coro, args, kwargs in self.producers:
            task = asyncio.create_task(
                self._producer_worker(producer_coro, args, kwargs)
            )
            producer_tasks.append(task)
        
        # Start consumer tasks
        consumer_tasks = []
        for consumer_coro, args, kwargs in self.consumers:
            task = asyncio.create_task(
                self._consumer_worker(consumer_coro, args, kwargs)
            )
            consumer_tasks.append(task)
        
        # Wait for shutdown
        await self.shutdown_event.wait()
        
        # Cancel all tasks
        for task in producer_tasks + consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*producer_tasks, *consumer_tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop producer-consumer system"""
        self.running = False
        self.shutdown_event.set()
    
    async def _producer_worker(self, producer_coro: Callable, args: tuple, kwargs: dict):
        """Producer worker"""
        while self.running:
            try:
                item = await producer_coro(*args, **kwargs)
                if item is not None:
                    await self.queue.put(item)
            except Exception as e:
                logger.error(f"Producer error: {e}")
                await asyncio.sleep(0.1)
    
    async def _consumer_worker(self, consumer_coro: Callable, args: tuple, kwargs: dict):
        """Consumer worker"""
        while self.running:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await consumer_coro(item, *args, **kwargs)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Consumer error: {e}")

class AsyncRateLimiter:
    """Async rate limiter"""
    
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = Lock()
    
    async def acquire(self) -> bool:
        """Acquire rate limit permission"""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance >= 1:
                self.allowance -= 1
                return True
            else:
                return False
    
    async def wait(self):
        """Wait for rate limit"""
        while not await self.acquire():
            await asyncio.sleep(0.01)

class AsyncBatchProcessor:
    """Async batch processor"""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch = []
        self.lock = Lock()
        self.last_flush = time.time()
    
    async def add_item(self, item: Any):
        """Add item to batch"""
        async with self.lock:
            self.batch.append(item)
            
            if (len(self.batch) >= self.batch_size or 
                time.time() - self.last_flush >= self.flush_interval):
                await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush current batch"""
        if not self.batch:
            return
        
        current_batch = self.batch.copy()
        self.batch.clear()
        self.last_flush = time.time()
        
        # Process batch
        await self._process_batch(current_batch)
    
    async def _process_batch(self, batch: List[Any]):
        """Process batch of items"""
        # Override this method in subclasses
        print(f"Processing batch of {len(batch)} items")

# Usage examples
async def example_async_producer() -> str:
    """Example async producer"""
    await asyncio.sleep(0.1)  # Simulate async work
    return f"Async_Item_{time.time()}"

async def example_async_consumer(item: str):
    """Example async consumer"""
    print(f"Processing async: {item}")
    await asyncio.sleep(0.05)  # Simulate async work

async def example_async_usage():
    """Example async usage"""
    # Async producer-consumer
    pc = AsyncProducerConsumer(queue_size=10)
    pc.add_producer(example_async_producer)
    pc.add_consumer(example_async_consumer)
    
    # Start system
    start_task = asyncio.create_task(pc.start())
    
    # Let it run for a bit
    await asyncio.sleep(2)
    
    # Stop system
    await pc.stop()
    await start_task

# Run example
if __name__ == "__main__":
    asyncio.run(example_async_usage())
```

## TL;DR Runbook

### Quick Start

```python
# 1. Threading for I/O-bound tasks
from python.threading_patterns import ThreadPoolManager
with ThreadPoolManager(max_workers=10) as pool:
    task_id = pool.submit_task(io_bound_function, data)
    result = pool.get_task_result(task_id)

# 2. Multiprocessing for CPU-bound tasks
from python.multiprocessing_patterns import ProcessPoolManager
with ProcessPoolManager(max_workers=4) as pool:
    task_id = pool.submit_task(cpu_intensive_function, data)
    result = pool.get_task_result(task_id)

# 3. Async for concurrent I/O
from python.async_patterns import AsyncTaskManager
async def main():
    manager = AsyncTaskManager(max_concurrent=10)
    task_id = await manager.submit_task(async_function, data)
    result = await manager.get_task_result(task_id)
```

### Essential Patterns

```python
# Complete concurrency setup
def setup_concurrency_patterns():
    """Setup complete concurrency patterns"""
    
    # Threading for I/O
    thread_pool = ThreadPoolManager(max_workers=10)
    
    # Multiprocessing for CPU
    process_pool = ProcessPoolManager(max_workers=4)
    
    # Async for concurrent operations
    async_manager = AsyncTaskManager(max_concurrent=10)
    
    print("Concurrency patterns setup complete!")
```

---

*This guide provides the complete machinery for Python concurrency patterns. Each pattern includes implementation examples, synchronization strategies, and real-world usage patterns for enterprise concurrency management.*
