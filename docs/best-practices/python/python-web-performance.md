# Python Web Performance Best Practices

**Objective**: Master senior-level Python web performance patterns for production systems. When you need to optimize web applications for speed and scalability, when you want to implement comprehensive performance monitoring, when you need enterprise-grade web performance strategies—these best practices become your weapon of choice.

## Core Principles

- **Response Time**: Minimize response times for better user experience
- **Throughput**: Maximize requests per second
- **Resource Efficiency**: Optimize CPU, memory, and I/O usage
- **Caching**: Implement effective caching strategies
- **Monitoring**: Track performance metrics continuously

## Performance Optimization

### Request Processing Optimization

```python
# python/01-web-performance-optimization.py

"""
Web performance optimization patterns and techniques
"""

from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import json
import gzip
from datetime import datetime, timedelta
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, request_id: str) -> None:
        """Start performance timer"""
        self.start_times[request_id] = time.time()
    
    def end_timer(self, request_id: str, endpoint: str) -> float:
        """End performance timer and record metrics"""
        if request_id not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[request_id]
        
        if endpoint not in self.metrics:
            self.metrics[endpoint] = []
        self.metrics[endpoint].append(duration)
        
        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
        self.request_counts[endpoint] += 1
        
        del self.start_times[request_id]
        return duration
    
    def record_error(self, endpoint: str) -> None:
        """Record error for endpoint"""
        if endpoint not in self.error_counts:
            self.error_counts[endpoint] = 0
        self.error_counts[endpoint] += 1
    
    def get_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get performance stats for endpoint"""
        if endpoint not in self.metrics:
            return {}
        
        durations = self.metrics[endpoint]
        request_count = self.request_counts.get(endpoint, 0)
        error_count = self.error_counts.get(endpoint, 0)
        
        return {
            "endpoint": endpoint,
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": error_count / request_count if request_count > 0 else 0,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "p99_duration": sorted(durations)[int(len(durations) * 0.99)] if durations else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance stats for all endpoints"""
        return {endpoint: self.get_stats(endpoint) for endpoint in self.metrics.keys()}

def performance_monitor(metrics: PerformanceMetrics):
    """Decorator for performance monitoring"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request_id = f"{func.__name__}_{int(time.time() * 1000)}"
            metrics.start_timer(request_id)
            
            try:
                result = await func(*args, **kwargs)
                duration = metrics.end_timer(request_id, func.__name__)
                logger.info(f"Request {func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                metrics.record_error(func.__name__)
                duration = metrics.end_timer(request_id, func.__name__)
                logger.error(f"Request {func.__name__} failed after {duration:.4f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_id = f"{func.__name__}_{int(time.time() * 1000)}"
            metrics.start_timer(request_id)
            
            try:
                result = func(*args, **kwargs)
                duration = metrics.end_timer(request_id, func.__name__)
                logger.info(f"Request {func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                metrics.record_error(func.__name__)
                duration = metrics.end_timer(request_id, func.__name__)
                logger.error(f"Request {func.__name__} failed after {duration:.4f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class ResponseOptimizer:
    """Response optimization utilities"""
    
    @staticmethod
    def compress_response(data: str, compression_level: int = 6) -> bytes:
        """Compress response data"""
        return gzip.compress(data.encode('utf-8'), compresslevel=compression_level)
    
    @staticmethod
    def optimize_json_response(data: Dict[str, Any]) -> str:
        """Optimize JSON response"""
        # Remove None values
        cleaned_data = {k: v for k, v in data.items() if v is not None}
        
        # Use compact JSON
        return json.dumps(cleaned_data, separators=(',', ':'), ensure_ascii=False)
    
    @staticmethod
    def add_cache_headers(headers: Dict[str, str], max_age: int = 3600) -> Dict[str, str]:
        """Add cache headers to response"""
        headers.update({
            "Cache-Control": f"public, max-age={max_age}",
            "ETag": f'"{int(time.time())}"',
            "Last-Modified": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        })
        return headers
    
    @staticmethod
    def add_compression_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Add compression headers"""
        headers.update({
            "Content-Encoding": "gzip",
            "Vary": "Accept-Encoding"
        })
        return headers

class DatabaseOptimizer:
    """Database query optimization"""
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.query_stats: Dict[str, List[float]] = {}
    
    def cache_query(self, query_key: str, result: Any, ttl: int = 300) -> None:
        """Cache query result"""
        self.query_cache[query_key] = {
            "result": result,
            "expires_at": time.time() + ttl
        }
    
    def get_cached_query(self, query_key: str) -> Optional[Any]:
        """Get cached query result"""
        if query_key in self.query_cache:
            cached = self.query_cache[query_key]
            if time.time() < cached["expires_at"]:
                return cached["result"]
            else:
                del self.query_cache[query_key]
        return None
    
    def record_query_time(self, query: str, duration: float) -> None:
        """Record query execution time"""
        if query not in self.query_stats:
            self.query_stats[query] = []
        self.query_stats[query].append(duration)
    
    def get_slow_queries(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get slow queries above threshold"""
        slow_queries = []
        for query, times in self.query_stats.items():
            avg_time = sum(times) / len(times)
            if avg_time > threshold:
                slow_queries.append({
                    "query": query,
                    "avg_time": avg_time,
                    "count": len(times),
                    "max_time": max(times)
                })
        return sorted(slow_queries, key=lambda x: x["avg_time"], reverse=True)
    
    def optimize_query(self, query: str) -> str:
        """Optimize SQL query"""
        # Basic query optimization
        optimized = query.strip()
        
        # Add LIMIT if not present
        if "LIMIT" not in optimized.upper():
            optimized += " LIMIT 1000"
        
        # Remove unnecessary whitespace
        optimized = " ".join(optimized.split())
        
        return optimized

class ConnectionPool:
    """Connection pool for database connections"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: List[Any] = []
        self.available_connections: List[Any] = []
        self.busy_connections: List[Any] = []
        self.lock = asyncio.Lock()
    
    async def get_connection(self) -> Any:
        """Get connection from pool"""
        async with self.lock:
            if self.available_connections:
                connection = self.available_connections.pop()
                self.busy_connections.append(connection)
                return connection
            
            if len(self.connections) < self.max_connections:
                # Create new connection
                connection = await self._create_connection()
                self.connections.append(connection)
                self.busy_connections.append(connection)
                return connection
            
            # Wait for connection to become available
            while not self.available_connections:
                await asyncio.sleep(0.01)
            
            connection = self.available_connections.pop()
            self.busy_connections.append(connection)
            return connection
    
    async def release_connection(self, connection: Any) -> None:
        """Release connection back to pool"""
        async with self.lock:
            if connection in self.busy_connections:
                self.busy_connections.remove(connection)
                self.available_connections.append(connection)
    
    async def _create_connection(self) -> Any:
        """Create new database connection"""
        # Simulate connection creation
        await asyncio.sleep(0.01)
        return f"connection_{len(self.connections)}"
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "total_connections": len(self.connections),
            "available_connections": len(self.available_connections),
            "busy_connections": len(self.busy_connections),
            "max_connections": self.max_connections,
            "utilization": len(self.busy_connections) / self.max_connections if self.max_connections > 0 else 0
        }

# Usage examples
async def example_performance_optimization():
    """Example performance optimization usage"""
    # Create performance metrics
    metrics = PerformanceMetrics()
    
    # Create response optimizer
    optimizer = ResponseOptimizer()
    
    # Create database optimizer
    db_optimizer = DatabaseOptimizer()
    
    # Create connection pool
    pool = ConnectionPool(max_connections=5)
    
    # Example optimized endpoint
    @performance_monitor(metrics)
    async def get_users():
        """Get users with performance monitoring"""
        # Simulate database query
        await asyncio.sleep(0.1)
        return {"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}
    
    # Execute endpoint
    result = await get_users()
    print(f"Result: {result}")
    
    # Get performance stats
    stats = metrics.get_stats("get_users")
    print(f"Performance stats: {stats}")
    
    # Optimize response
    json_data = optimizer.optimize_json_response(result)
    compressed_data = optimizer.compress_response(json_data)
    print(f"Compressed size: {len(compressed_data)} bytes")
    
    # Database optimization
    slow_queries = db_optimizer.get_slow_queries(threshold=0.5)
    print(f"Slow queries: {slow_queries}")
    
    # Connection pool stats
    pool_stats = pool.get_pool_stats()
    print(f"Pool stats: {pool_stats}")
```

### Caching Strategies

```python
# python/02-caching-strategies.py

"""
Advanced caching strategies for web performance
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import json
import hashlib
from datetime import datetime, timedelta
import threading
from collections import OrderedDict

class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

@dataclass
class CacheEntry:
    """Cache entry definition"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)
    
    def update_access(self) -> None:
        """Update access information"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

class LRUCache:
    """LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.update_access()
                return entry.value
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self.lock:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl
            )
            
            self.cache[key] = entry
            
            # Remove oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "max_size": self.max_size,
                "utilization": total_entries / self.max_size if self.max_size > 0 else 0
            }

class DistributedCache:
    """Distributed cache implementation"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.node_count = len(nodes)
        self.local_cache = LRUCache(max_size=1000)
        self.consistency_level = "eventual"
    
    def _get_node(self, key: str) -> str:
        """Get node for key using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        node_index = hash_value % self.node_count
        return self.nodes[node_index]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        # Try local cache first
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try distributed cache
        node = self._get_node(key)
        # In real implementation, this would make HTTP request to node
        distributed_value = await self._get_from_node(node, key)
        
        if distributed_value is not None:
            # Cache locally
            self.local_cache.set(key, distributed_value)
            return distributed_value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in distributed cache"""
        # Set in local cache
        self.local_cache.set(key, value, ttl)
        
        # Set in distributed cache
        node = self._get_node(key)
        await self._set_in_node(node, key, value, ttl)
    
    async def _get_from_node(self, node: str, key: str) -> Optional[Any]:
        """Get value from specific node"""
        # Simulate network request
        await asyncio.sleep(0.01)
        return None  # Simulate cache miss
    
    async def _set_in_node(self, node: str, key: str, value: Any, ttl: Optional[int]) -> None:
        """Set value in specific node"""
        # Simulate network request
        await asyncio.sleep(0.01)

class CacheManager:
    """Cache manager for multiple cache strategies"""
    
    def __init__(self):
        self.caches: Dict[str, Union[LRUCache, DistributedCache]] = {}
        self.strategies: Dict[str, CacheStrategy] = {}
        self.default_ttl = 3600  # 1 hour
    
    def add_cache(self, name: str, cache: Union[LRUCache, DistributedCache], 
                  strategy: CacheStrategy = CacheStrategy.LRU) -> None:
        """Add cache to manager"""
        self.caches[name] = cache
        self.strategies[name] = strategy
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get value from specific cache"""
        if cache_name not in self.caches:
            return None
        
        cache = self.caches[cache_name]
        if isinstance(cache, LRUCache):
            return cache.get(key)
        elif isinstance(cache, DistributedCache):
            # This would be async in real implementation
            return None
    
    def set(self, cache_name: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in specific cache"""
        if cache_name not in self.caches:
            return
        
        cache = self.caches[cache_name]
        if isinstance(cache, LRUCache):
            cache.set(key, value, ttl or self.default_ttl)
        elif isinstance(cache, DistributedCache):
            # This would be async in real implementation
            pass
    
    def invalidate(self, cache_name: str, key: str) -> bool:
        """Invalidate key in specific cache"""
        if cache_name not in self.caches:
            return False
        
        cache = self.caches[cache_name]
        if isinstance(cache, LRUCache):
            return cache.delete(key)
        elif isinstance(cache, DistributedCache):
            # This would be async in real implementation
            return False
        
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches"""
        stats = {}
        for name, cache in self.caches.items():
            if isinstance(cache, LRUCache):
                stats[name] = cache.get_stats()
            elif isinstance(cache, DistributedCache):
                stats[name] = {"type": "distributed", "nodes": len(cache.nodes)}
        return stats

class CacheDecorator:
    """Cache decorator for function results"""
    
    def __init__(self, cache_manager: CacheManager, cache_name: str, ttl: Optional[int] = None):
        self.cache_manager = cache_manager
        self.cache_name = cache_name
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache_manager.get(self.cache_name, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache_manager.set(self.cache_name, cache_key, result, self.ttl)
            
            return result
        
        return wrapper
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments"""
        key_data = {
            "func_name": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

# Usage examples
def example_caching_strategies():
    """Example caching strategies usage"""
    # Create cache manager
    cache_manager = CacheManager()
    
    # Add LRU cache
    lru_cache = LRUCache(max_size=100)
    cache_manager.add_cache("lru", lru_cache, CacheStrategy.LRU)
    
    # Add distributed cache
    distributed_cache = DistributedCache(["node1", "node2", "node3"])
    cache_manager.add_cache("distributed", distributed_cache, CacheStrategy.TTL)
    
    # Cache decorator example
    @CacheDecorator(cache_manager, "lru", ttl=300)
    def expensive_calculation(n: int) -> int:
        """Expensive calculation with caching"""
        time.sleep(0.1)  # Simulate expensive operation
        return n * n
    
    # Test caching
    start_time = time.time()
    result1 = expensive_calculation(10)
    end_time = time.time()
    print(f"First call: {result1} in {end_time - start_time:.4f}s")
    
    start_time = time.time()
    result2 = expensive_calculation(10)
    end_time = time.time()
    print(f"Second call: {result2} in {end_time - start_time:.4f}s")
    
    # Get cache stats
    stats = cache_manager.get_all_stats()
    print(f"Cache stats: {stats}")
```

### Load Balancing

```python
# python/03-load-balancing.py

"""
Load balancing strategies for web performance
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import random
import statistics
from datetime import datetime, timedelta

class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"

@dataclass
class Server:
    """Server definition"""
    server_id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    is_healthy: bool = True
    last_health_check: datetime = None
    
    def get_utilization(self) -> float:
        """Get server utilization"""
        return self.current_connections / self.max_connections if self.max_connections > 0 else 0
    
    def update_response_time(self, response_time: float) -> None:
        """Update server response time"""
        self.response_time = response_time
        self.last_health_check = datetime.utcnow()

class LoadBalancer:
    """Load balancer implementation"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.servers: List[Server] = []
        self.current_index = 0
        self.server_stats: Dict[str, List[float]] = {}
        self.health_check_interval = 30  # seconds
        self.health_check_task = None
    
    def add_server(self, server: Server) -> None:
        """Add server to load balancer"""
        self.servers.append(server)
        self.server_stats[server.server_id] = []
    
    def remove_server(self, server_id: str) -> bool:
        """Remove server from load balancer"""
        for i, server in enumerate(self.servers):
            if server.server_id == server_id:
                del self.servers[i]
                if server_id in self.server_stats:
                    del self.server_stats[server_id]
                return True
        return False
    
    def get_server(self) -> Optional[Server]:
        """Get server based on load balancing strategy"""
        healthy_servers = [s for s in self.servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(healthy_servers)
        else:
            return healthy_servers[0]
    
    def _round_robin_selection(self, servers: List[Server]) -> Server:
        """Round robin server selection"""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server
    
    def _least_connections_selection(self, servers: List[Server]) -> Server:
        """Least connections server selection"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin_selection(self, servers: List[Server]) -> Server:
        """Weighted round robin server selection"""
        total_weight = sum(server.weight for server in servers)
        if total_weight == 0:
            return servers[0]
        
        # Simple weighted selection
        weights = [server.weight for server in servers]
        selected_server = random.choices(servers, weights=weights)[0]
        return selected_server
    
    def _least_response_time_selection(self, servers: List[Server]) -> Server:
        """Least response time server selection"""
        return min(servers, key=lambda s: s.response_time)
    
    def _random_selection(self, servers: List[Server]) -> Server:
        """Random server selection"""
        return random.choice(servers)
    
    def record_request(self, server: Server, response_time: float) -> None:
        """Record request statistics"""
        server.current_connections += 1
        server.update_response_time(response_time)
        
        if server.server_id in self.server_stats:
            self.server_stats[server.server_id].append(response_time)
    
    def record_response(self, server: Server) -> None:
        """Record response completion"""
        server.current_connections = max(0, server.current_connections - 1)
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for servers"""
        while True:
            for server in self.servers:
                await self._health_check_server(server)
            await asyncio.sleep(self.health_check_interval)
    
    async def _health_check_server(self, server: Server) -> None:
        """Perform health check on server"""
        try:
            # Simulate health check
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate network request
            response_time = time.time() - start_time
            
            server.update_response_time(response_time)
            server.is_healthy = response_time < 1.0  # Health threshold
        except Exception as e:
            print(f"Health check failed for {server.server_id}: {e}")
            server.is_healthy = False
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_servers = [s for s in self.servers if s.is_healthy]
        total_connections = sum(s.current_connections for s in self.servers)
        avg_response_time = statistics.mean([s.response_time for s in healthy_servers]) if healthy_servers else 0
        
        return {
            "total_servers": len(self.servers),
            "healthy_servers": len(healthy_servers),
            "total_connections": total_connections,
            "average_response_time": avg_response_time,
            "strategy": self.strategy.value,
            "servers": [
                {
                    "server_id": s.server_id,
                    "host": s.host,
                    "port": s.port,
                    "is_healthy": s.is_healthy,
                    "current_connections": s.current_connections,
                    "response_time": s.response_time,
                    "utilization": s.get_utilization()
                }
                for s in self.servers
            ]
        }

class AutoScaling:
    """Auto-scaling for load balancer"""
    
    def __init__(self, load_balancer: LoadBalancer, min_servers: int = 2, max_servers: int = 10):
        self.load_balancer = load_balancer
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.scale_cooldown = 300  # 5 minutes
        self.last_scale_time = 0
    
    def should_scale_up(self) -> bool:
        """Check if should scale up"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        if len(self.load_balancer.servers) >= self.max_servers:
            return False
        
        # Check if any server is over threshold
        for server in self.load_balancer.servers:
            if server.get_utilization() > self.scale_up_threshold:
                return True
        
        return False
    
    def should_scale_down(self) -> bool:
        """Check if should scale down"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        if len(self.load_balancer.servers) <= self.min_servers:
            return False
        
        # Check if all servers are under threshold
        for server in self.load_balancer.servers:
            if server.get_utilization() > self.scale_down_threshold:
                return False
        
        return True
    
    async def scale_up(self) -> bool:
        """Scale up by adding server"""
        if not self.should_scale_up():
            return False
        
        # Create new server
        new_server = Server(
            server_id=f"server_{len(self.load_balancer.servers) + 1}",
            host="localhost",
            port=8000 + len(self.load_balancer.servers),
            weight=1
        )
        
        self.load_balancer.add_server(new_server)
        self.last_scale_time = time.time()
        
        print(f"Scaled up: Added server {new_server.server_id}")
        return True
    
    async def scale_down(self) -> bool:
        """Scale down by removing server"""
        if not self.should_scale_down():
            return False
        
        # Remove server with least connections
        if self.load_balancer.servers:
            server_to_remove = min(self.load_balancer.servers, key=lambda s: s.current_connections)
            self.load_balancer.remove_server(server_to_remove.server_id)
            self.last_scale_time = time.time()
            
            print(f"Scaled down: Removed server {server_to_remove.server_id}")
            return True
        
        return False

# Usage examples
async def example_load_balancing():
    """Example load balancing usage"""
    # Create load balancer
    load_balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
    
    # Add servers
    for i in range(3):
        server = Server(
            server_id=f"server_{i + 1}",
            host="localhost",
            port=8000 + i,
            weight=1
        )
        load_balancer.add_server(server)
    
    # Create auto-scaling
    auto_scaling = AutoScaling(load_balancer, min_servers=2, max_servers=5)
    
    # Simulate requests
    for i in range(10):
        server = load_balancer.get_server()
        if server:
            # Simulate request processing
            response_time = random.uniform(0.1, 0.5)
            load_balancer.record_request(server, response_time)
            
            # Simulate response completion
            await asyncio.sleep(0.1)
            load_balancer.record_response(server)
            
            print(f"Request {i + 1} handled by {server.server_id}")
    
    # Get load balancer stats
    stats = load_balancer.get_server_stats()
    print(f"Load balancer stats: {stats}")
    
    # Test auto-scaling
    if auto_scaling.should_scale_up():
        await auto_scaling.scale_up()
    
    if auto_scaling.should_scale_down():
        await auto_scaling.scale_down()
```

## TL;DR Runbook

### Quick Start

```python
# 1. Performance monitoring
metrics = PerformanceMetrics()
@performance_monitor(metrics)
async def endpoint():
    return {"data": "result"}

# 2. Response optimization
optimizer = ResponseOptimizer()
compressed_data = optimizer.compress_response(json_data)

# 3. Caching
cache_manager = CacheManager()
cache_manager.add_cache("lru", LRUCache(max_size=1000))

# 4. Load balancing
load_balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
server = load_balancer.get_server()

# 5. Auto-scaling
auto_scaling = AutoScaling(load_balancer, min_servers=2, max_servers=10)
await auto_scaling.scale_up()
```

### Essential Patterns

```python
# Complete web performance setup
def setup_web_performance():
    """Setup complete web performance environment"""
    
    # Performance metrics
    metrics = PerformanceMetrics()
    
    # Response optimizer
    optimizer = ResponseOptimizer()
    
    # Cache manager
    cache_manager = CacheManager()
    
    # Load balancer
    load_balancer = LoadBalancer()
    
    # Auto-scaling
    auto_scaling = AutoScaling(load_balancer)
    
    print("Web performance setup complete!")
```

---

*This guide provides the complete machinery for Python web performance. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise web performance management.*
