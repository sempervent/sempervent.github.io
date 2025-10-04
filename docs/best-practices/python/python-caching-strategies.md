# Python Caching Strategies Best Practices

**Objective**: Master senior-level Python caching patterns for production systems. When you need to implement high-performance caching, when you want to build distributed cache systems, when you need enterprise-grade caching strategies—these best practices become your weapon of choice.

## Core Principles

- **Performance**: Optimize for speed and throughput
- **Consistency**: Balance cache consistency with performance
- **Scalability**: Design for horizontal scaling
- **Reliability**: Implement fault tolerance and fallback mechanisms
- **Security**: Protect cached data and prevent cache poisoning

## Multi-Level Caching

### Cache Architecture

```python
# python/01-multi-level-caching.py

"""
Multi-level caching architecture and strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
import logging
from collections import OrderedDict
import redis
from redis import ConnectionPool as RedisConnectionPool
import memcached
from functools import wraps
import asyncio
import aioredis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache level enumeration"""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Database cache

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
    size: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)
    
    def update_access(self) -> None:
        """Update access information"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

class L1Cache:
    """L1 In-memory cache implementation"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    del self.cache[key]
                    self.metrics["misses"] += 1
                    return None
                
                # Update access information
                entry.update_access()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                
                self.metrics["hits"] += 1
                return entry.value
            
            self.metrics["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in L1 cache"""
        with self.lock:
            # Calculate size
            size = len(pickle.dumps(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl,
                size=size
            )
            
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict if over capacity
            while len(self.cache) > self.max_size:
                self._evict_entry()
    
    def delete(self, key: str) -> bool:
        """Delete key from L1 cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear L1 cache"""
        with self.lock:
            self.cache.clear()
    
    def _evict_entry(self) -> None:
        """Evict entry based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[lfu_key]
        
        self.metrics["evictions"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            "hit_rate": hit_rate,
            "current_size": len(self.cache),
            "max_size": self.max_size
        }

class L2Cache:
    """L2 Redis cache implementation"""
    
    def __init__(self, redis_pool: RedisConnectionPool, key_prefix: str = "cache:"):
        self.redis_pool = redis_pool
        self.key_prefix = key_prefix
        self.client = redis.Redis(connection_pool=redis_pool)
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            value = self.client.get(full_key)
            
            if value is not None:
                self.metrics["hits"] += 1
                return pickle.loads(value)
            else:
                self.metrics["misses"] += 1
                return None
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"L2 cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            serialized_value = pickle.dumps(value)
            
            if ttl:
                self.client.setex(full_key, ttl, serialized_value)
            else:
                self.client.set(full_key, serialized_value)
            
            return True
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"L2 cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            result = self.client.delete(full_key)
            return result > 0
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"L2 cache delete error: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            "hit_rate": hit_rate
        }

class AsyncL2Cache:
    """Async L2 Redis cache implementation"""
    
    def __init__(self, redis_pool, key_prefix: str = "cache:"):
        self.redis_pool = redis_pool
        self.key_prefix = key_prefix
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from async L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            value = await self.redis_pool.get(full_key)
            
            if value is not None:
                self.metrics["hits"] += 1
                return pickle.loads(value)
            else:
                self.metrics["misses"] += 1
                return None
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Async L2 cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in async L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            serialized_value = pickle.dumps(value)
            
            if ttl:
                await self.redis_pool.setex(full_key, ttl, serialized_value)
            else:
                await self.redis_pool.set(full_key, serialized_value)
            
            return True
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Async L2 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from async L2 cache"""
        try:
            full_key = f"{self.key_prefix}{key}"
            result = await self.redis_pool.delete(full_key)
            return result > 0
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Async L2 cache delete error: {e}")
            return False

class MultiLevelCache:
    """Multi-level cache implementation"""
    
    def __init__(self, l1_cache: L1Cache, l2_cache: L2Cache):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "writes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.metrics["l1_hits"] += 1
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            self.metrics["l2_hits"] += 1
            # Populate L1 cache
            self.l1_cache.set(key, value)
            return value
        
        self.metrics["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in multi-level cache"""
        # Set in both levels
        self.l1_cache.set(key, value, ttl)
        self.l2_cache.set(key, value, ttl)
        self.metrics["writes"] += 1
    
    def delete(self, key: str) -> None:
        """Delete key from multi-level cache"""
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get multi-level cache metrics"""
        total_requests = self.metrics["l1_hits"] + self.metrics["l2_hits"] + self.metrics["misses"]
        l1_hit_rate = (self.metrics["l1_hits"] / total_requests * 100) if total_requests > 0 else 0
        l2_hit_rate = (self.metrics["l2_hits"] / total_requests * 100) if total_requests > 0 else 0
        overall_hit_rate = ((self.metrics["l1_hits"] + self.metrics["l2_hits"]) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "l1_metrics": self.l1_cache.get_metrics(),
            "l2_metrics": self.l2_cache.get_metrics()
        }

class CacheDecorator:
    """Cache decorator for function results"""
    
    def __init__(self, cache: MultiLevelCache, ttl: Optional[int] = None, key_prefix: str = ""):
        self.cache = cache
        self.ttl = ttl
        self.key_prefix = key_prefix
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache.set(cache_key, result, self.ttl)
            
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
        return f"{self.key_prefix}{hashlib.md5(key_string.encode()).hexdigest()}"

class CacheInvalidation:
    """Cache invalidation strategies"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.invalidation_patterns = {}
    
    def add_pattern(self, pattern: str, keys: List[str]) -> None:
        """Add invalidation pattern"""
        self.invalidation_patterns[pattern] = keys
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        if pattern not in self.invalidation_patterns:
            return 0
        
        invalidated_count = 0
        for key in self.invalidation_patterns[pattern]:
            if self.cache.l1_cache.delete(key):
                invalidated_count += 1
            if self.cache.l2_cache.delete(key):
                invalidated_count += 1
        
        return invalidated_count
    
    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate keys with prefix"""
        invalidated_count = 0
        
        # Invalidate L1 cache
        with self.cache.l1_cache.lock:
            keys_to_delete = [key for key in self.cache.l1_cache.cache.keys() if key.startswith(prefix)]
            for key in keys_to_delete:
                if self.cache.l1_cache.delete(key):
                    invalidated_count += 1
        
        # Invalidate L2 cache
        try:
            # This would use Redis SCAN in real implementation
            pass
        except Exception as e:
            logger.error(f"Error invalidating L2 cache by prefix: {e}")
        
        return invalidated_count

# Usage examples
def example_multi_level_caching():
    """Example multi-level caching usage"""
    # Create L1 cache
    l1_cache = L1Cache(max_size=100, strategy=CacheStrategy.LRU)
    
    # Create Redis connection pool
    redis_pool = RedisConnectionPool(host='localhost', port=6379, db=0)
    
    # Create L2 cache
    l2_cache = L2Cache(redis_pool, key_prefix="app:")
    
    # Create multi-level cache
    multi_cache = MultiLevelCache(l1_cache, l2_cache)
    
    # Set values
    multi_cache.set("user:1", {"id": 1, "name": "John"}, ttl=3600)
    multi_cache.set("user:2", {"id": 2, "name": "Jane"}, ttl=3600)
    
    # Get values
    user1 = multi_cache.get("user:1")
    print(f"Retrieved user1: {user1}")
    
    # Get cache metrics
    metrics = multi_cache.get_metrics()
    print(f"Cache metrics: {metrics}")
    
    # Cache decorator
    cache_decorator = CacheDecorator(multi_cache, ttl=300)
    
    @cache_decorator
    def expensive_calculation(n: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return n * n
    
    # Test cached function
    result1 = expensive_calculation(10)
    result2 = expensive_calculation(10)  # Should be cached
    print(f"Calculation results: {result1}, {result2}")
    
    # Cache invalidation
    invalidation = CacheInvalidation(multi_cache)
    invalidation.add_pattern("users", ["user:1", "user:2"])
    invalidated = invalidation.invalidate_by_pattern("users")
    print(f"Invalidated {invalidated} cache entries")
```

### Distributed Caching

```python
# python/02-distributed-caching.py

"""
Distributed caching patterns and consistency strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import time
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import redis
from redis import ConnectionPool as RedisConnectionPool
import aioredis
from consistent_hash import ConsistentHash
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ConsistencyLevel(Enum):
    """Consistency level enumeration"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    WEAK = "weak"

class DistributedCache:
    """Distributed cache implementation"""
    
    def __init__(self, nodes: List[str], consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL):
        self.nodes = nodes
        self.consistency_level = consistency_level
        self.consistent_hash = ConsistentHash(nodes)
        self.redis_pools = {}
        self.initialize_connections()
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "replications": 0
        }
    
    def initialize_connections(self) -> None:
        """Initialize connections to all nodes"""
        for node in self.nodes:
            try:
                pool = RedisConnectionPool.from_url(f"redis://{node}")
                self.redis_pools[node] = redis.Redis(connection_pool=pool)
            except Exception as e:
                logger.error(f"Failed to connect to node {node}: {e}")
    
    def get_node(self, key: str) -> str:
        """Get node for key using consistent hashing"""
        return self.consistent_hash.get_node(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        try:
            # Get primary node
            primary_node = self.get_node(key)
            
            # Try primary node first
            value = self._get_from_node(primary_node, key)
            if value is not None:
                self.metrics["hits"] += 1
                return value
            
            # Try other nodes for eventual consistency
            if self.consistency_level == ConsistencyLevel.EVENTUAL:
                for node in self.nodes:
                    if node != primary_node:
                        value = self._get_from_node(node, key)
                        if value is not None:
                            # Update primary node
                            self._set_in_node(primary_node, key, value)
                            self.metrics["hits"] += 1
                            return value
            
            self.metrics["misses"] += 1
            return None
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Distributed cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache"""
        try:
            # Get primary node
            primary_node = self.get_node(key)
            
            # Set in primary node
            success = self._set_in_node(primary_node, key, value, ttl)
            if not success:
                return False
            
            # Replicate to other nodes based on consistency level
            if self.consistency_level == ConsistencyLevel.STRONG:
                # Synchronous replication
                for node in self.nodes:
                    if node != primary_node:
                        self._set_in_node(node, key, value, ttl)
                        self.metrics["replications"] += 1
            elif self.consistency_level == ConsistencyLevel.EVENTUAL:
                # Asynchronous replication
                self._async_replicate(key, value, ttl, primary_node)
            
            return True
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Distributed cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from distributed cache"""
        try:
            # Get primary node
            primary_node = self.get_node(key)
            
            # Delete from primary node
            success = self._delete_from_node(primary_node, key)
            
            # Delete from other nodes
            for node in self.nodes:
                if node != primary_node:
                    self._delete_from_node(node, key)
            
            return success
        
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Distributed cache delete error: {e}")
            return False
    
    def _get_from_node(self, node: str, key: str) -> Optional[Any]:
        """Get value from specific node"""
        try:
            if node in self.redis_pools:
                value = self.redis_pools[node].get(key)
                return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Error getting from node {node}: {e}")
        return None
    
    def _set_in_node(self, node: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in specific node"""
        try:
            if node in self.redis_pools:
                serialized_value = json.dumps(value)
                if ttl:
                    self.redis_pools[node].setex(key, ttl, serialized_value)
                else:
                    self.redis_pools[node].set(key, serialized_value)
                return True
        except Exception as e:
            logger.error(f"Error setting in node {node}: {e}")
        return False
    
    def _delete_from_node(self, node: str, key: str) -> bool:
        """Delete key from specific node"""
        try:
            if node in self.redis_pools:
                result = self.redis_pools[node].delete(key)
                return result > 0
        except Exception as e:
            logger.error(f"Error deleting from node {node}: {e}")
        return False
    
    def _async_replicate(self, key: str, value: Any, ttl: Optional[int], exclude_node: str) -> None:
        """Asynchronously replicate to other nodes"""
        def replicate():
            for node in self.nodes:
                if node != exclude_node:
                    self._set_in_node(node, key, value, ttl)
                    self.metrics["replications"] += 1
        
        # Run in background thread
        thread = threading.Thread(target=replicate)
        thread.daemon = True
        thread.start()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get distributed cache metrics"""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.metrics,
            "hit_rate": hit_rate,
            "nodes": len(self.nodes),
            "consistency_level": self.consistency_level.value
        }

class CacheSharding:
    """Cache sharding implementation"""
    
    def __init__(self, shards: List[str], shard_key_func: Optional[Callable] = None):
        self.shards = shards
        self.shard_key_func = shard_key_func or self._default_shard_key
        self.shard_connections = {}
        self.initialize_shards()
    
    def initialize_shards(self) -> None:
        """Initialize connections to all shards"""
        for shard in self.shards:
            try:
                pool = RedisConnectionPool.from_url(f"redis://{shard}")
                self.shard_connections[shard] = redis.Redis(connection_pool=pool)
            except Exception as e:
                logger.error(f"Failed to connect to shard {shard}: {e}")
    
    def _default_shard_key(self, key: str) -> str:
        """Default shard key function"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.shards[hash_value % len(self.shards)]
    
    def get_shard(self, key: str) -> str:
        """Get shard for key"""
        return self.shard_key_func(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from sharded cache"""
        shard = self.get_shard(key)
        try:
            if shard in self.shard_connections:
                value = self.shard_connections[shard].get(key)
                return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Error getting from shard {shard}: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in sharded cache"""
        shard = self.get_shard(key)
        try:
            if shard in self.shard_connections:
                serialized_value = json.dumps(value)
                if ttl:
                    self.shard_connections[shard].setex(key, ttl, serialized_value)
                else:
                    self.shard_connections[shard].set(key, serialized_value)
                return True
        except Exception as e:
            logger.error(f"Error setting in shard {shard}: {e}")
        return False
    
    def delete(self, key: str) -> bool:
        """Delete key from sharded cache"""
        shard = self.get_shard(key)
        try:
            if shard in self.shard_connections:
                result = self.shard_connections[shard].delete(key)
                return result > 0
        except Exception as e:
            logger.error(f"Error deleting from shard {shard}: {e}")
        return False

class CacheWarming:
    """Cache warming strategies"""
    
    def __init__(self, cache: Union[MultiLevelCache, DistributedCache]):
        self.cache = cache
        self.warming_tasks = []
        self.warming_metrics = {
            "warmed_keys": 0,
            "failed_warms": 0,
            "warming_time": 0.0
        }
    
    def add_warming_task(self, key: str, value_func: Callable, ttl: Optional[int] = None) -> None:
        """Add cache warming task"""
        self.warming_tasks.append({
            "key": key,
            "value_func": value_func,
            "ttl": ttl
        })
    
    def warm_cache(self) -> Dict[str, Any]:
        """Warm cache with all tasks"""
        start_time = time.time()
        warmed_keys = 0
        failed_warms = 0
        
        for task in self.warming_tasks:
            try:
                # Execute value function
                value = task["value_func"]()
                
                # Set in cache
                self.cache.set(task["key"], value, task["ttl"])
                warmed_keys += 1
                
            except Exception as e:
                logger.error(f"Failed to warm cache for key {task['key']}: {e}")
                failed_warms += 1
        
        warming_time = time.time() - start_time
        self.warming_metrics["warmed_keys"] += warmed_keys
        self.warming_metrics["failed_warms"] += failed_warms
        self.warming_metrics["warming_time"] += warming_time
        
        return {
            "warmed_keys": warmed_keys,
            "failed_warms": failed_warms,
            "warming_time": warming_time
        }
    
    def get_warming_metrics(self) -> Dict[str, Any]:
        """Get cache warming metrics"""
        return self.warming_metrics.copy()

class CacheMonitoring:
    """Cache monitoring and alerting"""
    
    def __init__(self, cache: Union[MultiLevelCache, DistributedCache]):
        self.cache = cache
        self.monitoring_metrics = {}
        self.alerts = []
    
    def check_cache_health(self) -> Dict[str, Any]:
        """Check cache health"""
        health_status = {
            "healthy": True,
            "issues": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check hit rate
        metrics = self.cache.get_metrics()
        hit_rate = metrics.get("overall_hit_rate", 0)
        
        if hit_rate < 50:  # Low hit rate threshold
            health_status["healthy"] = False
            health_status["issues"].append(f"Low hit rate: {hit_rate:.2f}%")
        
        # Check error rate
        if "errors" in metrics:
            error_rate = metrics["errors"] / max(metrics.get("hits", 0) + metrics.get("misses", 0), 1) * 100
            if error_rate > 5:  # High error rate threshold
                health_status["healthy"] = False
                health_status["issues"].append(f"High error rate: {error_rate:.2f}%")
        
        return health_status
    
    def add_alert(self, condition: Callable, message: str) -> None:
        """Add monitoring alert"""
        self.alerts.append({
            "condition": condition,
            "message": message
        })
    
    def check_alerts(self) -> List[str]:
        """Check all alerts"""
        triggered_alerts = []
        
        for alert in self.alerts:
            try:
                if alert["condition"]():
                    triggered_alerts.append(alert["message"])
            except Exception as e:
                logger.error(f"Error checking alert: {e}")
        
        return triggered_alerts

# Usage examples
def example_distributed_caching():
    """Example distributed caching usage"""
    # Create distributed cache
    nodes = ["redis-node-1:6379", "redis-node-2:6379", "redis-node-3:6379"]
    distributed_cache = DistributedCache(nodes, ConsistencyLevel.EVENTUAL)
    
    # Set values
    distributed_cache.set("user:1", {"id": 1, "name": "John"}, ttl=3600)
    distributed_cache.set("user:2", {"id": 2, "name": "Jane"}, ttl=3600)
    
    # Get values
    user1 = distributed_cache.get("user:1")
    print(f"Retrieved user1: {user1}")
    
    # Get metrics
    metrics = distributed_cache.get_metrics()
    print(f"Distributed cache metrics: {metrics}")
    
    # Cache sharding
    shards = ["redis-shard-1:6379", "redis-shard-2:6379", "redis-shard-3:6379"]
    sharded_cache = CacheSharding(shards)
    
    # Use sharded cache
    sharded_cache.set("product:1", {"id": 1, "name": "Laptop"}, ttl=1800)
    product = sharded_cache.get("product:1")
    print(f"Retrieved product: {product}")
    
    # Cache warming
    warming = CacheWarming(distributed_cache)
    
    # Add warming tasks
    warming.add_warming_task("popular_users", lambda: [{"id": i, "name": f"User{i}"} for i in range(1, 11)])
    warming.add_warming_task("config", lambda: {"theme": "dark", "language": "en"})
    
    # Warm cache
    warming_result = warming.warm_cache()
    print(f"Cache warming result: {warming_result}")
    
    # Cache monitoring
    monitoring = CacheMonitoring(distributed_cache)
    
    # Add alerts
    monitoring.add_alert(
        lambda: distributed_cache.get_metrics().get("overall_hit_rate", 0) < 50,
        "Low cache hit rate detected"
    )
    
    # Check health
    health = monitoring.check_cache_health()
    print(f"Cache health: {health}")
    
    # Check alerts
    alerts = monitoring.check_alerts()
    if alerts:
        print(f"Triggered alerts: {alerts}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Multi-level caching
l1_cache = L1Cache(max_size=1000, strategy=CacheStrategy.LRU)
l2_cache = L2Cache(redis_pool, key_prefix="app:")
multi_cache = MultiLevelCache(l1_cache, l2_cache)

# 2. Cache decorator
cache_decorator = CacheDecorator(multi_cache, ttl=300)
@cache_decorator
def expensive_function(n: int) -> int:
    return n * n

# 3. Distributed caching
distributed_cache = DistributedCache(nodes, ConsistencyLevel.EVENTUAL)
distributed_cache.set("key", "value", ttl=3600)

# 4. Cache sharding
sharded_cache = CacheSharding(shards)
sharded_cache.set("key", "value")

# 5. Cache warming
warming = CacheWarming(multi_cache)
warming.add_warming_task("data", lambda: fetch_data())
warming.warm_cache()
```

### Essential Patterns

```python
# Complete caching setup
def setup_caching_strategies():
    """Setup complete caching strategies environment"""
    
    # L1 cache
    l1_cache = L1Cache(max_size=1000, strategy=CacheStrategy.LRU)
    
    # L2 cache
    redis_pool = RedisConnectionPool(host='localhost', port=6379)
    l2_cache = L2Cache(redis_pool, key_prefix="app:")
    
    # Multi-level cache
    multi_cache = MultiLevelCache(l1_cache, l2_cache)
    
    # Distributed cache
    distributed_cache = DistributedCache(nodes, ConsistencyLevel.EVENTUAL)
    
    # Cache sharding
    sharded_cache = CacheSharding(shards)
    
    # Cache warming
    warming = CacheWarming(multi_cache)
    
    # Cache monitoring
    monitoring = CacheMonitoring(multi_cache)
    
    print("Caching strategies setup complete!")
```

---

*This guide provides the complete machinery for Python caching strategies. Each pattern includes implementation examples, caching strategies, and real-world usage patterns for enterprise cache management.*
