# Python Performance Tuning Best Practices

**Objective**: Master senior-level Python performance tuning patterns for production systems. When you need to optimize Python applications for speed and efficiency, when you want to identify and resolve performance bottlenecks, when you need enterprise-grade performance optimization strategies—these best practices become your weapon of choice.

## Core Principles

- **Measure First**: Always profile before optimizing
- **Identify Bottlenecks**: Focus on the biggest performance impacts
- **Algorithm Optimization**: Choose the right algorithms and data structures
- **Memory Efficiency**: Optimize memory usage and garbage collection
- **Concurrency**: Leverage parallel processing where appropriate

## Profiling and Benchmarking

### Performance Profiling Tools

```python
# python/01-profiling-benchmarking.py

"""
Comprehensive Python performance profiling and benchmarking
"""

import cProfile
import pstats
import time
import memory_profiler
import psutil
import os
from typing import List, Dict, Any, Callable, Optional
from functools import wraps
import tracemalloc
import line_profiler

class PerformanceProfiler:
    """Comprehensive performance profiling tools"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.memory_traces = []
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            self.profiler.enable()
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Stop profiling
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            self.profiler.disable()
            
            # Record metrics
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"Function: {func.__name__}")
            print(f"Execution time: {execution_time:.4f} seconds")
            print(f"Memory used: {memory_used / 1024 / 1024:.2f} MB")
            
            return result
        return wrapper
    
    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile function with cProfile"""
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        # Get profiling stats
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        # Extract key metrics
        total_calls = stats.total_calls
        total_time = stats.total_tt
        
        return {
            "total_calls": total_calls,
            "total_time": total_time,
            "stats": stats
        }
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a function"""
        # Start memory tracing
        tracemalloc.start()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "current_memory": current,
            "peak_memory": peak,
            "result": result
        }
    
    def line_by_line_profile(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Line-by-line profiling with line_profiler"""
        profiler = line_profiler.LineProfiler()
        profiler.add_function(func)
        
        # Execute function
        result = profiler.runcall(func, *args, **kwargs)
        
        # Get line-by-line stats
        profiler.print_stats()
        
        return {
            "result": result,
            "profiler": profiler
        }

class BenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self):
        self.benchmarks = []
    
    def benchmark_function(self, name: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a single function"""
        times = []
        memory_usage = []
        
        # Warm up
        for _ in range(3):
            func(*args, **kwargs)
        
        # Benchmark
        for _ in range(10):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        benchmark_result = {
            "name": name,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "avg_memory": avg_memory,
            "times": times,
            "memory_usage": memory_usage
        }
        
        self.benchmarks.append(benchmark_result)
        return benchmark_result
    
    def compare_functions(self, functions: List[tuple]) -> Dict[str, Any]:
        """Compare multiple functions"""
        results = {}
        
        for name, func, args, kwargs in functions:
            result = self.benchmark_function(name, func, *args, **kwargs)
            results[name] = result
        
        # Find fastest
        fastest = min(results.items(), key=lambda x: x[1]["avg_time"])
        
        return {
            "results": results,
            "fastest": fastest[0],
            "fastest_time": fastest[1]["avg_time"]
        }
    
    def generate_report(self) -> str:
        """Generate benchmarking report"""
        if not self.benchmarks:
            return "No benchmarks to report"
        
        report = "# Performance Benchmark Report\n\n"
        
        for benchmark in self.benchmarks:
            report += f"## {benchmark['name']}\n"
            report += f"- Average time: {benchmark['avg_time']:.4f}s\n"
            report += f"- Min time: {benchmark['min_time']:.4f}s\n"
            report += f"- Max time: {benchmark['max_time']:.4f}s\n"
            report += f"- Average memory: {benchmark['avg_memory'] / 1024 / 1024:.2f} MB\n\n"
        
        return report

# Usage example
def example_slow_function(n: int) -> List[int]:
    """Example function to profile"""
    result = []
    for i in range(n):
        result.append(i * i)
    return result

def example_fast_function(n: int) -> List[int]:
    """Optimized version of the function"""
    return [i * i for i in range(n)]

# Profile functions
profiler = PerformanceProfiler()

@profiler.profile_function
def profiled_function():
    return example_slow_function(10000)

# Benchmark comparison
benchmark_suite = BenchmarkSuite()
functions_to_compare = [
    ("slow_function", example_slow_function, (10000,), {}),
    ("fast_function", example_fast_function, (10000,), {})
]

comparison_results = benchmark_suite.compare_functions(functions_to_compare)
print(benchmark_suite.generate_report())
```

### Memory Profiling

```python
# python/02-memory-profiling.py

"""
Advanced memory profiling and optimization
"""

import tracemalloc
import gc
import sys
import psutil
from typing import List, Dict, Any, Optional
from memory_profiler import profile
import objgraph

class MemoryProfiler:
    """Advanced memory profiling tools"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.gc_stats = []
    
    def start_memory_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
    
    def stop_memory_tracing(self) -> Dict[str, Any]:
        """Stop memory tracing and get stats"""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "current_memory": current,
            "peak_memory": peak,
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024
        }
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get current memory snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,   # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_memory": psutil.virtual_memory().available
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def analyze_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Analyze memory usage of a function"""
        # Get initial memory
        initial_snapshot = self.get_memory_snapshot()
        
        # Start tracing
        self.start_memory_tracing()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_snapshot = self.get_memory_snapshot()
        trace_stats = self.stop_memory_tracing()
        
        # Calculate memory usage
        memory_used = final_snapshot["rss"] - initial_snapshot["rss"]
        
        return {
            "initial_memory": initial_snapshot,
            "final_memory": final_snapshot,
            "memory_used": memory_used,
            "trace_stats": trace_stats,
            "result": result
        }
    
    def find_memory_leaks(self) -> Dict[str, Any]:
        """Find potential memory leaks"""
        # Force garbage collection
        gc.collect()
        
        # Get object counts
        object_counts = objgraph.most_common_types()
        
        # Get reference counts
        reference_counts = {}
        for obj_type, count in object_counts:
            if count > 100:  # Threshold for potential leaks
                reference_counts[obj_type] = count
        
        return {
            "object_counts": object_counts,
            "potential_leaks": reference_counts,
            "gc_counts": gc.get_count()
        }
    
    def optimize_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Optimize memory usage of a function"""
        # Analyze current memory usage
        analysis = self.analyze_memory_usage(func, *args, **kwargs)
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after GC
        post_gc_snapshot = self.get_memory_snapshot()
        
        return {
            "before_optimization": analysis,
            "after_gc": post_gc_snapshot,
            "gc_effectiveness": analysis["final_memory"]["rss"] - post_gc_snapshot["rss"]
        }

# Memory optimization patterns
class MemoryOptimizer:
    """Memory optimization patterns and techniques"""
    
    @staticmethod
    def use_generators(data: List[Any]) -> List[Any]:
        """Use generators instead of lists for large datasets"""
        return (item for item in data if item is not None)
    
    @staticmethod
    def use_slots(cls):
        """Use __slots__ to reduce memory usage"""
        class OptimizedClass(cls):
            __slots__ = ['attr1', 'attr2', 'attr3']
        return OptimizedClass
    
    @staticmethod
    def lazy_loading(data_source: Callable) -> Any:
        """Implement lazy loading for expensive operations"""
        class LazyLoader:
            def __init__(self, loader):
                self._loader = loader
                self._data = None
            
            def __getattr__(self, name):
                if self._data is None:
                    self._data = self._loader()
                return getattr(self._data, name)
        
        return LazyLoader(data_source)
    
    @staticmethod
    def memory_efficient_processing(data: List[Any], batch_size: int = 1000) -> List[Any]:
        """Process data in batches to reduce memory usage"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # Process batch
            batch_results = [item * 2 for item in batch]
            results.extend(batch_results)
            
            # Force garbage collection after each batch
            gc.collect()
        
        return results

# Usage example
@profile
def memory_intensive_function(n: int) -> List[int]:
    """Function that uses a lot of memory"""
    data = []
    for i in range(n):
        data.append(i * i)
    return data

def optimized_memory_function(n: int) -> List[int]:
    """Memory-optimized version"""
    return [i * i for i in range(n)]

# Profile memory usage
memory_profiler = MemoryProfiler()
memory_analysis = memory_profiler.analyze_memory_usage(memory_intensive_function, 10000)
print(f"Memory used: {memory_analysis['memory_used'] / 1024 / 1024:.2f} MB")
```

## Algorithm Optimization

### Data Structure Optimization

```python
# python/03-algorithm-optimization.py

"""
Algorithm and data structure optimization
"""

import time
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque, Counter
import heapq
import bisect

class AlgorithmOptimizer:
    """Algorithm optimization techniques"""
    
    @staticmethod
    def optimize_list_operations(data: List[int]) -> List[int]:
        """Optimize list operations"""
        # Use list comprehension instead of loops
        return [x * 2 for x in data if x > 0]
    
    @staticmethod
    def optimize_dictionary_operations(data: List[str]) -> Dict[str, int]:
        """Optimize dictionary operations"""
        # Use Counter for counting
        return Counter(data)
    
    @staticmethod
    def optimize_set_operations(data1: Set[int], data2: Set[int]) -> Set[int]:
        """Optimize set operations"""
        # Use set operations for intersection, union, etc.
        return data1.intersection(data2)
    
    @staticmethod
    def optimize_search_operations(data: List[int], target: int) -> Optional[int]:
        """Optimize search operations using binary search"""
        # Use bisect for binary search
        index = bisect.bisect_left(data, target)
        if index < len(data) and data[index] == target:
            return index
        return None
    
    @staticmethod
    def optimize_heap_operations(data: List[int], k: int) -> List[int]:
        """Optimize heap operations for top-k elements"""
        # Use heapq for efficient heap operations
        return heapq.nlargest(k, data)
    
    @staticmethod
    def optimize_string_operations(text: str) -> str:
        """Optimize string operations"""
        # Use join instead of concatenation
        return ''.join(char.upper() for char in text if char.isalpha())

class DataStructureOptimizer:
    """Data structure optimization patterns"""
    
    @staticmethod
    def use_deque_for_fifo(data: List[Any]) -> deque:
        """Use deque for FIFO operations"""
        return deque(data)
    
    @staticmethod
    def use_defaultdict_for_grouping(data: List[tuple]) -> Dict[Any, List[Any]]:
        """Use defaultdict for grouping operations"""
        grouped = defaultdict(list)
        for key, value in data:
            grouped[key].append(value)
        return grouped
    
    @staticmethod
    def use_counter_for_frequency(data: List[Any]) -> Counter:
        """Use Counter for frequency analysis"""
        return Counter(data)
    
    @staticmethod
    def use_heap_for_priority_queue(data: List[tuple]) -> List[tuple]:
        """Use heap for priority queue operations"""
        heapq.heapify(data)
        return data

# Performance comparison examples
def compare_search_algorithms():
    """Compare different search algorithms"""
    data = list(range(1000000))
    target = 500000
    
    # Linear search
    start_time = time.time()
    linear_result = target in data
    linear_time = time.time() - start_time
    
    # Binary search
    start_time = time.time()
    binary_result = AlgorithmOptimizer.optimize_search_operations(data, target)
    binary_time = time.time() - start_time
    
    print(f"Linear search: {linear_time:.6f}s")
    print(f"Binary search: {binary_time:.6f}s")
    print(f"Speedup: {linear_time / binary_time:.2f}x")

def compare_data_structures():
    """Compare different data structures"""
    data = list(range(10000))
    
    # List operations
    start_time = time.time()
    list_result = [x * 2 for x in data]
    list_time = time.time() - start_time
    
    # Set operations
    data_set = set(data)
    start_time = time.time()
    set_result = {x * 2 for x in data_set}
    set_time = time.time() - start_time
    
    print(f"List operations: {list_time:.6f}s")
    print(f"Set operations: {set_time:.6f}s")
```

## Concurrency Optimization

### Parallel Processing

```python
# python/04-concurrency-optimization.py

"""
Concurrency and parallel processing optimization
"""

import asyncio
import concurrent.futures
import multiprocessing
from typing import List, Callable, Any
import time
import threading
from functools import partial

class ConcurrencyOptimizer:
    """Concurrency optimization patterns"""
    
    @staticmethod
    def parallel_processing(data: List[Any], func: Callable, max_workers: int = None) -> List[Any]:
        """Parallel processing with ThreadPoolExecutor"""
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, data))
        
        return results
    
    @staticmethod
    def process_pool_processing(data: List[Any], func: Callable, max_workers: int = None) -> List[Any]:
        """Process pool for CPU-intensive tasks"""
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, data))
        
        return results
    
    @staticmethod
    async def async_processing(data: List[Any], func: Callable) -> List[Any]:
        """Async processing for I/O-intensive tasks"""
        tasks = [func(item) for item in data]
        results = await asyncio.gather(*tasks)
        return results
    
    @staticmethod
    def batch_processing(data: List[Any], func: Callable, batch_size: int = 100) -> List[Any]:
        """Process data in batches"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = [func(item) for item in batch]
            results.extend(batch_results)
        
        return results

class AsyncOptimizer:
    """Async optimization patterns"""
    
    @staticmethod
    async def async_generator(data: List[Any], func: Callable):
        """Async generator for memory efficiency"""
        for item in data:
            result = await func(item)
            yield result
    
    @staticmethod
    async def async_batch_processing(data: List[Any], func: Callable, batch_size: int = 100):
        """Async batch processing"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    async def async_with_semaphore(data: List[Any], func: Callable, max_concurrent: int = 10):
        """Async processing with semaphore for rate limiting"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_func(item):
            async with semaphore:
                return await func(item)
        
        tasks = [limited_func(item) for item in data]
        results = await asyncio.gather(*tasks)
        return results

# Performance comparison
def cpu_intensive_task(n: int) -> int:
    """CPU-intensive task"""
    return sum(i * i for i in range(n))

def io_intensive_task(delay: float) -> str:
    """I/O-intensive task"""
    time.sleep(delay)
    return f"Task completed after {delay}s"

async def async_io_task(delay: float) -> str:
    """Async I/O-intensive task"""
    await asyncio.sleep(delay)
    return f"Async task completed after {delay}s"

def compare_concurrency_methods():
    """Compare different concurrency methods"""
    data = list(range(100))
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [cpu_intensive_task(i) for i in data]
    sequential_time = time.time() - start_time
    
    # Thread pool processing
    start_time = time.time()
    thread_results = ConcurrencyOptimizer.parallel_processing(data, cpu_intensive_task)
    thread_time = time.time() - start_time
    
    # Process pool processing
    start_time = time.time()
    process_results = ConcurrencyOptimizer.process_pool_processing(data, cpu_intensive_task)
    process_time = time.time() - start_time
    
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Thread pool: {thread_time:.4f}s")
    print(f"Process pool: {process_time:.4f}s")
    print(f"Thread speedup: {sequential_time / thread_time:.2f}x")
    print(f"Process speedup: {sequential_time / process_time:.2f}x")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Profile your code
from python.profiling import PerformanceProfiler
profiler = PerformanceProfiler()

@profiler.profile_function
def your_function():
    # Your code here
    pass

# 2. Benchmark different approaches
from python.benchmarking import BenchmarkSuite
benchmark = BenchmarkSuite()
benchmark.benchmark_function("approach1", function1)
benchmark.benchmark_function("approach2", function2)

# 3. Optimize memory usage
from python.memory_optimization import MemoryOptimizer
optimizer = MemoryOptimizer()
optimized_data = optimizer.memory_efficient_processing(data)

# 4. Use concurrency for I/O-bound tasks
from python.concurrency import ConcurrencyOptimizer
results = ConcurrencyOptimizer.parallel_processing(data, your_function)
```

### Essential Patterns

```python
# Complete performance optimization
def optimize_python_application():
    """Complete performance optimization workflow"""
    
    # 1. Profile to identify bottlenecks
    profiler = PerformanceProfiler()
    
    # 2. Benchmark different approaches
    benchmark_suite = BenchmarkSuite()
    
    # 3. Optimize memory usage
    memory_optimizer = MemoryOptimizer()
    
    # 4. Use appropriate concurrency
    concurrency_optimizer = ConcurrencyOptimizer()
    
    # 5. Apply algorithm optimizations
    algorithm_optimizer = AlgorithmOptimizer()
    
    print("Performance optimization complete!")
```

---

*This guide provides the complete machinery for Python performance tuning. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise performance optimization.*
