# Python Memory Management Best Practices

**Objective**: Master senior-level Python memory management patterns for production systems. When you need to optimize memory usage, when you want to prevent memory leaks, when you need enterprise-grade memory management strategies—these best practices become your weapon of choice.

## Core Principles

- **Understand Python's Memory Model**: Reference counting, garbage collection, and memory allocation
- **Monitor Memory Usage**: Track memory consumption and identify bottlenecks
- **Prevent Memory Leaks**: Avoid circular references and properly manage resources
- **Optimize Data Structures**: Choose memory-efficient data structures
- **Use Memory Profiling**: Identify memory hotspots and optimize accordingly

## Memory Model Understanding

### Python's Memory Architecture

```python
# python/01-memory-architecture.py

"""
Understanding Python's memory model and architecture
"""

import sys
import gc
import weakref
from typing import List, Dict, Any, Optional, Callable
import tracemalloc
import psutil
import os

class MemoryAnalyzer:
    """Advanced memory analysis and monitoring"""
    
    def __init__(self):
        self.snapshots = []
        self.memory_traces = []
        self.reference_tracker = {}
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,   # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total,
            "python_objects": len(gc.get_objects()),
            "gc_counts": gc.get_count()
        }
    
    def start_memory_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
        self.memory_traces.append(tracemalloc.get_traced_memory())
    
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
    
    def take_memory_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot"""
        snapshot = self.get_memory_info()
        self.snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(self, snapshot1: int, snapshot2: int) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        if snapshot1 >= len(self.snapshots) or snapshot2 >= len(self.snapshots):
            raise ValueError("Invalid snapshot indices")
        
        snap1 = self.snapshots[snapshot1]
        snap2 = self.snapshots[snapshot2]
        
        return {
            "rss_diff": snap2["rss"] - snap1["rss"],
            "vms_diff": snap2["vms"] - snap1["vms"],
            "percent_diff": snap2["percent"] - snap1["percent"],
            "objects_diff": snap2["python_objects"] - snap1["python_objects"]
        }
    
    def find_memory_leaks(self) -> Dict[str, Any]:
        """Find potential memory leaks"""
        # Force garbage collection
        gc.collect()
        
        # Get object counts by type
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Find objects with high counts (potential leaks)
        potential_leaks = {
            obj_type: count for obj_type, count in object_counts.items()
            if count > 1000  # Threshold for potential leaks
        }
        
        return {
            "object_counts": object_counts,
            "potential_leaks": potential_leaks,
            "gc_counts": gc.get_count(),
            "total_objects": len(gc.get_objects())
        }

class ReferenceTracker:
    """Track object references and prevent leaks"""
    
    def __init__(self):
        self.tracked_objects = {}
        self.weak_refs = {}
        self.circular_refs = []
    
    def track_object(self, obj: Any, name: str) -> str:
        """Track an object and return tracking ID"""
        obj_id = id(obj)
        self.tracked_objects[obj_id] = {
            "object": obj,
            "name": name,
            "type": type(obj).__name__,
            "ref_count": sys.getrefcount(obj)
        }
        return str(obj_id)
    
    def create_weak_reference(self, obj: Any, callback: Optional[Callable] = None) -> weakref.ref:
        """Create weak reference to object"""
        weak_ref = weakref.ref(obj, callback)
        self.weak_refs[id(obj)] = weak_ref
        return weak_ref
    
    def check_circular_references(self) -> List[Dict[str, Any]]:
        """Check for circular references"""
        circular_refs = []
        
        for obj in gc.get_objects():
            if gc.is_tracked(obj):
                referrers = gc.get_referrers(obj)
                for referrer in referrers:
                    if gc.is_tracked(referrer):
                        # Check if they reference each other
                        if obj in gc.get_referrers(referrer):
                            circular_refs.append({
                                "obj1": type(obj).__name__,
                                "obj2": type(referrer).__name__,
                                "obj1_id": id(obj),
                                "obj2_id": id(referrer)
                            })
        
        self.circular_refs = circular_refs
        return circular_refs
    
    def get_reference_info(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Get reference information for tracked object"""
        obj_id_int = int(obj_id)
        if obj_id_int in self.tracked_objects:
            obj_info = self.tracked_objects[obj_id_int]
            return {
                "name": obj_info["name"],
                "type": obj_info["type"],
                "ref_count": sys.getrefcount(obj_info["object"]),
                "is_tracked": gc.is_tracked(obj_info["object"])
            }
        return None
    
    def cleanup_tracked_objects(self):
        """Cleanup tracked objects"""
        self.tracked_objects.clear()
        self.weak_refs.clear()
        self.circular_refs.clear()

# Usage examples
def example_memory_analysis():
    """Example memory analysis usage"""
    analyzer = MemoryAnalyzer()
    
    # Take initial snapshot
    initial_snapshot = analyzer.take_memory_snapshot()
    print(f"Initial memory: {initial_snapshot['rss'] / 1024 / 1024:.2f} MB")
    
    # Create some objects
    data = [i for i in range(10000)]
    
    # Take second snapshot
    second_snapshot = analyzer.take_memory_snapshot()
    print(f"After creating data: {second_snapshot['rss'] / 1024 / 1024:.2f} MB")
    
    # Compare snapshots
    comparison = analyzer.compare_snapshots(0, 1)
    print(f"Memory difference: {comparison['rss_diff'] / 1024 / 1024:.2f} MB")
    
    # Check for leaks
    leak_info = analyzer.find_memory_leaks()
    print(f"Potential leaks: {leak_info['potential_leaks']}")

def example_reference_tracking():
    """Example reference tracking usage"""
    tracker = ReferenceTracker()
    
    # Create and track objects
    obj1 = [1, 2, 3]
    obj2 = {"key": "value"}
    
    id1 = tracker.track_object(obj1, "list_object")
    id2 = tracker.track_object(obj2, "dict_object")
    
    # Get reference info
    info1 = tracker.get_reference_info(id1)
    info2 = tracker.get_reference_info(id2)
    
    print(f"Object 1 ref count: {info1['ref_count']}")
    print(f"Object 2 ref count: {info2['ref_count']}")
    
    # Check for circular references
    circular_refs = tracker.check_circular_references()
    print(f"Circular references found: {len(circular_refs)}")
```

### Garbage Collection Management

```python
# python/02-garbage-collection.py

"""
Advanced garbage collection management and optimization
"""

import gc
import sys
import time
from typing import List, Dict, Any, Optional, Callable
import weakref
from contextlib import contextmanager

class GarbageCollectionManager:
    """Advanced garbage collection management"""
    
    def __init__(self):
        self.gc_stats = []
        self.collection_times = []
        self.thresholds = gc.get_threshold()
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get comprehensive GC statistics"""
        counts = gc.get_count()
        stats = {
            "generation_0": counts[0],
            "generation_1": counts[1],
            "generation_2": counts[2],
            "thresholds": self.thresholds,
            "total_objects": len(gc.get_objects()),
            "tracked_objects": len([obj for obj in gc.get_objects() if gc.is_tracked(obj)])
        }
        
        self.gc_stats.append(stats)
        return stats
    
    def force_collection(self) -> Dict[str, Any]:
        """Force garbage collection and measure performance"""
        start_time = time.time()
        
        # Get initial stats
        initial_stats = self.get_gc_stats()
        
        # Force collection
        collected = gc.collect()
        
        # Get final stats
        final_stats = self.get_gc_stats()
        collection_time = time.time() - start_time
        
        self.collection_times.append(collection_time)
        
        return {
            "collected_objects": collected,
            "collection_time": collection_time,
            "initial_objects": initial_stats["total_objects"],
            "final_objects": final_stats["total_objects"],
            "objects_freed": initial_stats["total_objects"] - final_stats["total_objects"]
        }
    
    def optimize_gc_thresholds(self, target_objects: int = 10000):
        """Optimize GC thresholds based on object count"""
        current_objects = len(gc.get_objects())
        
        if current_objects > target_objects:
            # Increase thresholds to reduce GC frequency
            new_thresholds = (
                self.thresholds[0] * 2,
                self.thresholds[1] * 2,
                self.thresholds[2] * 2
            )
            gc.set_threshold(*new_thresholds)
            self.thresholds = new_thresholds
            return True
        
        return False
    
    def disable_gc(self):
        """Disable garbage collection"""
        gc.disable()
    
    def enable_gc(self):
        """Enable garbage collection"""
        gc.enable()
    
    @contextmanager
    def gc_context(self, enabled: bool = True):
        """Context manager for GC control"""
        original_state = gc.isenabled()
        
        if enabled:
            gc.enable()
        else:
            gc.disable()
        
        try:
            yield
        finally:
            if original_state:
                gc.enable()
            else:
                gc.disable()

class MemoryEfficientDataStructures:
    """Memory-efficient data structure implementations"""
    
    @staticmethod
    def memory_efficient_list(size: int) -> List[int]:
        """Create memory-efficient list"""
        # Use generator for large lists
        return [i for i in range(size)]
    
    @staticmethod
    def memory_efficient_dict(keys: List[str], values: List[Any]) -> Dict[str, Any]:
        """Create memory-efficient dictionary"""
        # Use dict comprehension
        return {k: v for k, v in zip(keys, values)}
    
    @staticmethod
    def memory_efficient_set(items: List[Any]) -> set:
        """Create memory-efficient set"""
        return set(items)
    
    @staticmethod
    def memory_efficient_string_concatenation(strings: List[str]) -> str:
        """Memory-efficient string concatenation"""
        # Use join instead of +=
        return ''.join(strings)
    
    @staticmethod
    def memory_efficient_file_processing(file_path: str, chunk_size: int = 8192):
        """Memory-efficient file processing"""
        with open(file_path, 'r') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

class WeakReferenceManager:
    """Manage weak references to prevent memory leaks"""
    
    def __init__(self):
        self.weak_refs = {}
        self.callbacks = {}
    
    def create_weak_reference(self, obj: Any, name: str, callback: Optional[Callable] = None) -> weakref.ref:
        """Create weak reference with callback"""
        def cleanup_callback(weak_ref):
            if name in self.weak_refs:
                del self.weak_refs[name]
            if name in self.callbacks:
                del self.callbacks[name]
            if callback:
                callback(weak_ref)
        
        weak_ref = weakref.ref(obj, cleanup_callback)
        self.weak_refs[name] = weak_ref
        if callback:
            self.callbacks[name] = callback
        
        return weak_ref
    
    def get_weak_reference(self, name: str) -> Optional[Any]:
        """Get object from weak reference"""
        if name in self.weak_refs:
            return self.weak_refs[name]()
        return None
    
    def cleanup_weak_references(self):
        """Cleanup all weak references"""
        self.weak_refs.clear()
        self.callbacks.clear()

# Usage examples
def example_gc_management():
    """Example garbage collection management"""
    gc_manager = GarbageCollectionManager()
    
    # Get initial GC stats
    initial_stats = gc_manager.get_gc_stats()
    print(f"Initial objects: {initial_stats['total_objects']}")
    
    # Create some objects
    data = [i for i in range(10000)]
    
    # Force collection
    collection_result = gc_manager.force_collection()
    print(f"Collected {collection_result['collected_objects']} objects")
    print(f"Collection time: {collection_result['collection_time']:.4f}s")
    
    # Optimize thresholds
    optimized = gc_manager.optimize_gc_thresholds()
    print(f"GC thresholds optimized: {optimized}")

def example_memory_efficient_structures():
    """Example memory-efficient data structures"""
    # Memory-efficient list
    large_list = MemoryEfficientDataStructures.memory_efficient_list(10000)
    print(f"List size: {sys.getsizeof(large_list)} bytes")
    
    # Memory-efficient string concatenation
    strings = [f"string_{i}" for i in range(1000)]
    result = MemoryEfficientDataStructures.memory_efficient_string_concatenation(strings)
    print(f"Concatenated string length: {len(result)}")
    
    # Memory-efficient file processing
    # This would process a file in chunks
    # for chunk in MemoryEfficientDataStructures.memory_efficient_file_processing("large_file.txt"):
    #     process_chunk(chunk)
```

## Memory Optimization Patterns

### Object Pooling

```python
# python/03-memory-optimization.py

"""
Memory optimization patterns including object pooling and caching
"""

import gc
import sys
import time
from typing import List, Dict, Any, Optional, Type, Callable
from collections import deque
import threading

class ObjectPool:
    """Generic object pool for memory optimization"""
    
    def __init__(self, factory: Callable, max_size: int = 100, reset_func: Optional[Callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self.pool = deque()
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0
    
    def get_object(self) -> Any:
        """Get object from pool"""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
            else:
                obj = self.factory()
                self.created_count += 1
            
            # Reset object if reset function provided
            if self.reset_func:
                self.reset_func(obj)
            
            return obj
    
    def return_object(self, obj: Any):
        """Return object to pool"""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "created_count": self.created_count,
                "reused_count": self.reused_count,
                "total_objects": self.created_count + self.reused_count
            }
    
    def clear_pool(self):
        """Clear object pool"""
        with self.lock:
            self.pool.clear()

class MemoryCache:
    """Memory-efficient cache with size limits"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.access_order = deque()
        self.current_memory = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self.lock:
            # Calculate memory usage
            value_size = sys.getsizeof(value)
            
            # Remove items if cache is full
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + value_size > self.max_memory_bytes):
                if not self.access_order:
                    break
                
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    old_value = self.cache[oldest_key]
                    self.current_memory -= sys.getsizeof(old_value)
                    del self.cache[oldest_key]
            
            # Add new item
            self.cache[key] = value
            self.current_memory += value_size
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "memory_usage": self.current_memory,
                "memory_usage_mb": self.current_memory / 1024 / 1024,
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024
            }

class MemoryMonitor:
    """Monitor memory usage and provide alerts"""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 95.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.alerts = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Memory monitoring loop"""
        while self.monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent >= self.critical_threshold:
                    self.alerts.append({
                        "level": "critical",
                        "memory_percent": memory_percent,
                        "timestamp": time.time()
                    })
                elif memory_percent >= self.warning_threshold:
                    self.alerts.append({
                        "level": "warning",
                        "memory_percent": memory_percent,
                        "timestamp": time.time()
                    })
                
                time.sleep(interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(interval)
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get memory alerts"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear memory alerts"""
        self.alerts.clear()

# Usage examples
def example_object_pooling():
    """Example object pooling usage"""
    # Create object pool
    def create_expensive_object():
        return {"data": [i for i in range(1000)], "timestamp": time.time()}
    
    def reset_object(obj):
        obj["data"].clear()
        obj["timestamp"] = time.time()
    
    pool = ObjectPool(create_expensive_object, max_size=10, reset_func=reset_object)
    
    # Use objects from pool
    obj1 = pool.get_object()
    obj2 = pool.get_object()
    
    # Return objects to pool
    pool.return_object(obj1)
    pool.return_object(obj2)
    
    # Get pool stats
    stats = pool.get_stats()
    print(f"Pool stats: {stats}")

def example_memory_cache():
    """Example memory cache usage"""
    cache = MemoryCache(max_size=100, max_memory_mb=10)
    
    # Add items to cache
    for i in range(50):
        cache.put(f"key_{i}", f"value_{i}" * 100)
    
    # Get cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Get item from cache
    value = cache.get("key_10")
    print(f"Retrieved value: {value}")

def example_memory_monitoring():
    """Example memory monitoring usage"""
    monitor = MemoryMonitor(warning_threshold=70.0, critical_threshold=90.0)
    
    # Start monitoring
    monitor.start_monitoring(interval=0.5)
    
    # Simulate memory usage
    data = []
    for i in range(1000):
        data.append([j for j in range(1000)])
        time.sleep(0.01)
    
    # Check alerts
    alerts = monitor.get_alerts()
    print(f"Memory alerts: {len(alerts)}")
    
    # Stop monitoring
    monitor.stop_monitoring()
```

## Memory Leak Prevention

### Resource Management

```python
# python/04-memory-leak-prevention.py

"""
Memory leak prevention patterns and resource management
"""

import gc
import sys
import weakref
from typing import List, Dict, Any, Optional, Callable, ContextManager
from contextlib import contextmanager
import threading
import time

class ResourceManager:
    """Advanced resource management to prevent memory leaks"""
    
    def __init__(self):
        self.resources = {}
        self.cleanup_handlers = {}
        self.lock = threading.Lock()
    
    def register_resource(self, resource_id: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """Register resource with cleanup function"""
        with self.lock:
            self.resources[resource_id] = resource
            if cleanup_func:
                self.cleanup_handlers[resource_id] = cleanup_func
    
    def unregister_resource(self, resource_id: str):
        """Unregister resource and cleanup"""
        with self.lock:
            if resource_id in self.resources:
                # Call cleanup function if exists
                if resource_id in self.cleanup_handlers:
                    try:
                        self.cleanup_handlers[resource_id]()
                    except Exception as e:
                        print(f"Cleanup error for {resource_id}: {e}")
                
                del self.resources[resource_id]
                if resource_id in self.cleanup_handlers:
                    del self.cleanup_handlers[resource_id]
    
    def cleanup_all_resources(self):
        """Cleanup all registered resources"""
        with self.lock:
            for resource_id in list(self.resources.keys()):
                self.unregister_resource(resource_id)
    
    def get_resource(self, resource_id: str) -> Optional[Any]:
        """Get registered resource"""
        with self.lock:
            return self.resources.get(resource_id)
    
    def list_resources(self) -> List[str]:
        """List all registered resource IDs"""
        with self.lock:
            return list(self.resources.keys())

class CircularReferenceBreaker:
    """Break circular references to prevent memory leaks"""
    
    def __init__(self):
        self.tracked_objects = {}
        self.weak_refs = {}
    
    def track_object(self, obj: Any, name: str) -> str:
        """Track object for circular reference detection"""
        obj_id = id(obj)
        self.tracked_objects[obj_id] = {
            "object": obj,
            "name": name,
            "type": type(obj).__name__
        }
        return str(obj_id)
    
    def create_weak_reference(self, obj: Any, name: str) -> weakref.ref:
        """Create weak reference to break circular references"""
        def cleanup_callback(weak_ref):
            if name in self.weak_refs:
                del self.weak_refs[name]
        
        weak_ref = weakref.ref(obj, cleanup_callback)
        self.weak_refs[name] = weak_ref
        return weak_ref
    
    def break_circular_references(self) -> List[Dict[str, Any]]:
        """Break circular references and return info about broken references"""
        broken_refs = []
        
        # Find circular references
        for obj in gc.get_objects():
            if gc.is_tracked(obj):
                referrers = gc.get_referrers(obj)
                for referrer in referrers:
                    if gc.is_tracked(referrer) and obj in gc.get_referrers(referrer):
                        # Break circular reference by creating weak reference
                        weak_ref = weakref.ref(obj)
                        broken_refs.append({
                            "obj_type": type(obj).__name__,
                            "referrer_type": type(referrer).__name__,
                            "weak_ref": weak_ref
                        })
        
        return broken_refs
    
    def cleanup_tracked_objects(self):
        """Cleanup tracked objects"""
        self.tracked_objects.clear()
        self.weak_refs.clear()

class MemoryLeakDetector:
    """Detect and prevent memory leaks"""
    
    def __init__(self):
        self.baseline_objects = 0
        self.baseline_memory = 0
        self.leak_threshold = 1000  # Objects
        self.memory_threshold = 50 * 1024 * 1024  # 50MB
    
    def set_baseline(self):
        """Set baseline for leak detection"""
        gc.collect()
        self.baseline_objects = len(gc.get_objects())
        self.baseline_memory = psutil.Process().memory_info().rss
    
    def check_for_leaks(self) -> Dict[str, Any]:
        """Check for potential memory leaks"""
        gc.collect()
        current_objects = len(gc.get_objects())
        current_memory = psutil.Process().memory_info().rss
        
        object_diff = current_objects - self.baseline_objects
        memory_diff = current_memory - self.baseline_memory
        
        leak_detected = (
            object_diff > self.leak_threshold or
            memory_diff > self.memory_threshold
        )
        
        return {
            "leak_detected": leak_detected,
            "object_diff": object_diff,
            "memory_diff": memory_diff,
            "memory_diff_mb": memory_diff / 1024 / 1024,
            "current_objects": current_objects,
            "current_memory": current_memory,
            "baseline_objects": self.baseline_objects,
            "baseline_memory": self.baseline_memory
        }
    
    def force_cleanup(self):
        """Force cleanup to prevent leaks"""
        # Force garbage collection
        gc.collect()
        
        # Clear caches if they exist
        for obj in gc.get_objects():
            if hasattr(obj, 'clear') and callable(obj.clear):
                try:
                    obj.clear()
                except:
                    pass

@contextmanager
def memory_context(monitor_leaks: bool = True):
    """Context manager for memory leak monitoring"""
    detector = MemoryLeakDetector()
    detector.set_baseline()
    
    try:
        yield detector
    finally:
        if monitor_leaks:
            leak_info = detector.check_for_leaks()
            if leak_info["leak_detected"]:
                print(f"Memory leak detected: {leak_info}")
                detector.force_cleanup()

# Usage examples
def example_resource_management():
    """Example resource management usage"""
    manager = ResourceManager()
    
    # Register resources
    def cleanup_file(file_obj):
        if hasattr(file_obj, 'close'):
            file_obj.close()
    
    # Simulate resource registration
    resource1 = {"data": "resource1"}
    resource2 = {"data": "resource2"}
    
    manager.register_resource("resource1", resource1, cleanup_file)
    manager.register_resource("resource2", resource2, cleanup_file)
    
    # Use resources
    retrieved = manager.get_resource("resource1")
    print(f"Retrieved resource: {retrieved}")
    
    # Cleanup resources
    manager.cleanup_all_resources()
    print("Resources cleaned up")

def example_memory_leak_detection():
    """Example memory leak detection usage"""
    with memory_context(monitor_leaks=True) as detector:
        # Simulate memory usage
        data = []
        for i in range(1000):
            data.append([j for j in range(1000)])
        
        # Check for leaks
        leak_info = detector.check_for_leaks()
        print(f"Leak detection: {leak_info}")

def example_circular_reference_breaking():
    """Example circular reference breaking usage"""
    breaker = CircularReferenceBreaker()
    
    # Create objects that might have circular references
    obj1 = {"name": "object1"}
    obj2 = {"name": "object2"}
    
    # Create circular reference
    obj1["ref"] = obj2
    obj2["ref"] = obj1
    
    # Track objects
    breaker.track_object(obj1, "obj1")
    breaker.track_object(obj2, "obj2")
    
    # Break circular references
    broken_refs = breaker.break_circular_references()
    print(f"Broken references: {len(broken_refs)}")
    
    # Cleanup
    breaker.cleanup_tracked_objects()
```

## TL;DR Runbook

### Quick Start

```python
# 1. Monitor memory usage
from python.memory_management import MemoryAnalyzer
analyzer = MemoryAnalyzer()
memory_info = analyzer.get_memory_info()
print(f"Memory usage: {memory_info['rss'] / 1024 / 1024:.2f} MB")

# 2. Use object pooling
from python.memory_optimization import ObjectPool
pool = ObjectPool(create_expensive_object, max_size=10)
obj = pool.get_object()
# Use object
pool.return_object(obj)

# 3. Monitor for leaks
from python.memory_leak_prevention import memory_context
with memory_context(monitor_leaks=True) as detector:
    # Your code here
    pass

# 4. Force garbage collection
import gc
gc.collect()
```

### Essential Patterns

```python
# Complete memory management setup
def setup_memory_management():
    """Setup complete memory management environment"""
    
    # Memory analyzer
    analyzer = MemoryAnalyzer()
    
    # Object pool
    pool = ObjectPool(create_expensive_object, max_size=100)
    
    # Memory cache
    cache = MemoryCache(max_size=1000, max_memory_mb=100)
    
    # Resource manager
    resource_manager = ResourceManager()
    
    # Memory monitor
    monitor = MemoryMonitor(warning_threshold=80.0, critical_threshold=95.0)
    monitor.start_monitoring()
    
    print("Memory management setup complete!")
```

---

*This guide provides the complete machinery for Python memory management. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise memory management.*
