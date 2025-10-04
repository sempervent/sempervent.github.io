# Caching & Performance Best Practices

**Objective**: Caches are accelerants. Done right, they make pipelines scream. Done wrong, they give you stale lies.

Caches are accelerants. Done right, they make pipelines scream. Done wrong, they give you stale lies.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Correctness before speed**
   - Never cache what you can't invalidate
   - Design invalidation strategy first
   - Test cache behavior under load
   - Monitor for stale data

2. **Layer-aware caching**
   - Choose the right cache at the right layer
   - Database → Tiles → Workflows → APIs → Builds → GPU
   - Don't cache at multiple layers unnecessarily
   - Understand cache boundaries

3. **Expiration & invalidation are critical**
   - Set appropriate TTLs
   - Implement cache invalidation
   - Handle cache misses gracefully
   - Plan for cache warming

4. **Metrics are non-negotiable**
   - Measure hit/miss rates
   - Monitor cache size and memory usage
   - Track latency improvements
   - Alert on cache failures

5. **Cache scope matters**
   - Avoid global caches
   - Scope to specific contexts
   - Use cache keys strategically
   - Plan for cache partitioning

**Why These Principles**: Caching requires understanding performance characteristics, invalidation strategies, and system boundaries. Understanding these patterns prevents performance chaos and enables reliable system acceleration.

## 1) Core Principles

### The Cache Reality

```yaml
# What you thought caching was
cache_fantasy:
  "simplicity": "Just cache everything and it's fast"
  "reliability": "Caches never fail or go stale"
  "performance": "More cache = more speed"
  "maintenance": "Set it and forget it"

# What caching actually is
cache_reality:
  "simplicity": "Complex invalidation and consistency"
  "reliability": "Caches fail and data goes stale"
  "performance": "Wrong cache = worse performance"
  "maintenance": "Constant monitoring and tuning"
```

**Why Reality Checks Matter**: Understanding the true nature of caching enables proper design and maintenance. Understanding these patterns prevents cache chaos and enables reliable performance optimization.

### Cache Layer Architecture

```markdown
## Cache Layer Stack

### Layer 1: Database Caching
- **Postgres**: Connection pooling, query caching, materialized views
- **FDWs**: Materialized tables for external data
- **DuckDB**: Parquet metadata pruning, vectorized execution

### Layer 2: Tile Caching
- **Martin**: Vector/raster tile caching
- **CDN**: Global tile distribution
- **ETag/Last-Modified**: Cache busting headers

### Layer 3: Workflow Caching
- **Prefect**: Result persistence, task caching
- **Dask**: Worker memory, intermediate results
- **Scoped**: Flow/task-specific caching

### Layer 4: API Caching
- **FastAPI**: Request/response caching
- **Redis**: Session and data caching
- **HTTP**: Cache headers and ETags

### Layer 5: Build Caching
- **Docker**: Layer caching, multi-arch builds
- **Dependencies**: Version pinning, reproducible builds
- **CI/CD**: Build artifact caching

### Layer 6: GPU/Compute Caching
- **Models**: Pre-computed tensors, embeddings
- **Geospatial**: Isochrones, OD matrices, raster overlays
- **Storage**: NVMe caching, vector databases
```

**Why Layer Architecture Matters**: Proper cache layer design enables efficient performance optimization and prevents cache conflicts. Understanding these patterns prevents cache chaos and enables reliable performance optimization.

## 2) Database / Query Caching

### Postgres Connection Pooling

```yaml
# pgbouncer.ini
[databases]
mydb = host=postgres port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = 0.0.0.0
auth_type = md5
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
reserve_pool_size = 10
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 50

# Connection pooling metrics
stats_period = 60
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
```

**Why Connection Pooling Matters**: Connection pooling reduces connection overhead and improves database performance. Understanding these patterns prevents connection chaos and enables reliable database access.

### Postgres Query Caching

```sql
-- Enable query plan caching
SET shared_preload_libraries = 'pg_stat_statements';
SET track_activity_query_size = 2048;

-- Create materialized view for expensive queries
CREATE MATERIALIZED VIEW telemetry_daily_summary AS
SELECT 
    sensor_id,
    DATE_TRUNC('day', timestamp) as date,
    AVG(temperature) as avg_temp,
    MIN(temperature) as min_temp,
    MAX(temperature) as max_temp,
    COUNT(*) as record_count
FROM telemetry
WHERE timestamp >= '2024-01-01'
GROUP BY sensor_id, DATE_TRUNC('day', timestamp);

-- Create index on materialized view
CREATE INDEX idx_telemetry_daily_summary_sensor_date 
ON telemetry_daily_summary (sensor_id, date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW telemetry_daily_summary;

-- Query cached results
SELECT sensor_id, date, avg_temp, record_count
FROM telemetry_daily_summary
WHERE sensor_id = 'sensor_001'
  AND date >= '2024-01-01'
ORDER BY date;
```

**Why Query Caching Matters**: Materialized views cache expensive computations and improve query performance. Understanding these patterns prevents computation chaos and enables reliable database performance.

### FDW Materialized Tables

```sql
-- Create materialized table for FDW data
CREATE MATERIALIZED VIEW telemetry_cached AS
SELECT 
    id,
    user_id,
    sensor_id,
    timestamp,
    temperature,
    humidity,
    pressure,
    location
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01';

-- Create indexes on materialized view
CREATE INDEX idx_telemetry_cached_timestamp ON telemetry_cached (timestamp DESC);
CREATE INDEX idx_telemetry_cached_sensor ON telemetry_cached (sensor_id);
CREATE INDEX idx_telemetry_cached_location ON telemetry_cached USING GIST (location);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW telemetry_cached;

-- Query cached data
SELECT sensor_id, AVG(temperature) as avg_temp
FROM telemetry_cached
WHERE timestamp >= '2024-01-01'
  AND sensor_id = 'sensor_001'
GROUP BY sensor_id;
```

**Why FDW Materialized Tables Matter**: Materialized tables cache external data and improve query performance. Understanding these patterns prevents external data chaos and enables reliable federated data access.

### DuckDB Parquet Optimization

```sql
-- Enable parquet metadata caching
SET memory_limit = '8GB';
SET threads = 4;

-- Create parquet table with metadata
CREATE TABLE telemetry_parquet AS
SELECT * FROM read_parquet('s3://my-bucket/telemetry/*.parquet');

-- Use parquet metadata for filtering
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01'
  AND sensor_id = 'sensor_001';

-- Cache parquet metadata
CREATE TABLE telemetry_metadata AS
SELECT 
    file_name,
    row_count,
    min_timestamp,
    max_timestamp,
    sensor_ids
FROM parquet_metadata('s3://my-bucket/telemetry/*.parquet');
```

**Why DuckDB Parquet Optimization Matters**: Parquet metadata caching enables efficient data filtering and improves query performance. Understanding these patterns prevents parquet chaos and enables reliable analytical performance.

## 3) Tile Caching (Vector & Raster)

### Martin Tile Server Caching

```yaml
# martin-config.yaml
listen_addresses: "0.0.0.0:3000"
auto_publish: true

# Tile caching configuration
cache:
  type: "file"
  path: "/var/cache/martin"
  max_size: "1GB"
  ttl: "1h"

# Vector tile caching
sources:
  places:
    schema: "public"
    table: "places"
    geometry_column: "geom"
    srid: 3857
    cache:
      enabled: true
      ttl: "30m"
      max_zoom: 14

# Raster tile caching
sources:
  dem:
    schema: "public"
    table: "dem"
    geometry_column: "rast"
    srid: 3857
    cache:
      enabled: true
      ttl: "1h"
      max_zoom: 12
```

**Why Martin Tile Caching Matters**: Tile caching reduces rendering overhead and improves map performance. Understanding these patterns prevents tile chaos and enables reliable map performance.

### Nginx Tile Caching

```nginx
# nginx.conf
upstream martin {
    server martin:3000;
}

server {
    listen 80;
    server_name tiles.example.com;
    
    # Tile caching
    location /tiles/ {
        proxy_pass http://martin;
        proxy_cache tiles_cache;
        proxy_cache_valid 200 1h;
        proxy_cache_valid 404 1m;
        proxy_cache_use_stale error timeout updating;
        proxy_cache_lock on;
        
        # Cache headers
        add_header X-Cache-Status $upstream_cache_status;
        expires 1h;
    }
    
    # Vector tiles
    location /tiles/vector/ {
        proxy_pass http://martin;
        proxy_cache vector_cache;
        proxy_cache_valid 200 30m;
        proxy_cache_key $uri$is_args$args;
    }
    
    # Raster tiles
    location /tiles/raster/ {
        proxy_pass http://martin;
        proxy_cache raster_cache;
        proxy_cache_valid 200 1h;
        proxy_cache_key $uri$is_args$args;
    }
}

# Cache zones
proxy_cache_path /var/cache/nginx/tiles levels=1:2 keys_zone=tiles_cache:10m max_size=1g inactive=1h;
proxy_cache_path /var/cache/nginx/vector levels=1:2 keys_zone=vector_cache:10m max_size=500m inactive=30m;
proxy_cache_path /var/cache/nginx/raster levels=1:2 keys_zone=raster_cache:10m max_size=2g inactive=1h;
```

**Why Nginx Tile Caching Matters**: Nginx caching provides additional tile caching layer and improves global performance. Understanding these patterns prevents tile chaos and enables reliable map performance.

### CDN Tile Distribution

```yaml
# cloudfront-config.yaml
Distribution:
  Origins:
    - Id: "martin-tiles"
      DomainName: "tiles.example.com"
      CustomOriginConfig:
        HTTPPort: 80
        OriginProtocolPolicy: "http-only"
  
  DefaultCacheBehavior:
    TargetOriginId: "martin-tiles"
    ViewerProtocolPolicy: "redirect-to-https"
    CachePolicyId: "4135ea2d-6df8-44a3-9ef3-4e4c8c8c8c8c"
    TTL:
      DefaultTTL: 3600
      MaxTTL: 86400
  
  CacheBehaviors:
    - PathPattern: "/tiles/vector/*"
      TargetOriginId: "martin-tiles"
      CachePolicyId: "vector-tiles-policy"
      TTL:
        DefaultTTL: 1800
        MaxTTL: 3600
    
    - PathPattern: "/tiles/raster/*"
      TargetOriginId: "martin-tiles"
      CachePolicyId: "raster-tiles-policy"
      TTL:
        DefaultTTL: 3600
        MaxTTL: 86400
```

**Why CDN Tile Distribution Matters**: CDN distribution provides global tile caching and improves worldwide performance. Understanding these patterns prevents global tile chaos and enables reliable map performance.

## 4) Workflow Caching (Prefect & Dask)

### Prefect Result Persistence

```python
# prefect_config.py
from prefect import flow, task
from prefect.blocks.system import Secret
from prefect.filesystems import S3
from prefect.results import PersistedResult

# Configure S3 result storage
s3_block = S3.load("s3-results")

@task(cache_policy=CachePolicy.PERSISTENT)
def process_telemetry_data(data: dict) -> dict:
    """Process telemetry data with result persistence."""
    # Expensive computation
    result = {
        "processed_records": len(data),
        "avg_temperature": sum(data.values()) / len(data),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Store result in S3
    return PersistedResult(value=result, storage=s3_block)

@flow
def telemetry_processing_flow():
    """Main telemetry processing flow."""
    # Load data
    data = load_telemetry_data()
    
    # Process with caching
    result = process_telemetry_data(data)
    
    # Store result
    store_result(result)
    
    return result

# Configure result storage
from prefect import get_client
client = get_client()
client.set_storage(s3_block)
```

**Why Prefect Result Persistence Matters**: Result persistence prevents recomputation and improves workflow performance. Understanding these patterns prevents workflow chaos and enables reliable data processing.

### Dask Worker Memory Caching

```python
# dask_config.py
import dask
from dask.distributed import Client, LocalCluster
from dask.cache import Cache

# Configure Dask caching
dask.config.set({
    'distributed.worker.memory.target': 0.8,
    'distributed.worker.memory.spill': 0.9,
    'distributed.worker.memory.pause': 0.95,
    'distributed.worker.memory.terminate': 0.98
})

# Create cluster with memory management
cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit='8GB',
    processes=True
)

client = Client(cluster)

# Enable worker memory caching
cache = Cache(2e9)  # 2GB cache
cache.register()

# Example: Cache intermediate results
@dask.delayed
def expensive_computation(data):
    """Expensive computation that should be cached."""
    # This will be cached in worker memory
    return process_data(data)

@dask.delayed
def another_computation(cached_result):
    """Uses cached result from previous computation."""
    return transform_data(cached_result)

# Use cached results
data = load_large_dataset()
result1 = expensive_computation(data)
result2 = another_computation(result1)
result3 = another_computation(result1)  # Uses cache

# Compute with caching
final_result = dask.compute(result2, result3)
```

**Why Dask Worker Memory Caching Matters**: Worker memory caching prevents recomputation and improves distributed processing performance. Understanding these patterns prevents distributed chaos and enables reliable data processing.

## 5) API & Service Caching

### FastAPI Redis Caching

```python
# fastapi_cache.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import redis
import json
from typing import Optional

app = FastAPI()

# Configure Redis backend
redis_client = redis.Redis(host='redis', port=6379, db=0)
FastAPICache.init(RedisBackend(redis_client), prefix="api-cache")

@app.get("/telemetry/{sensor_id}")
@cache(expire=300)  # 5 minutes
async def get_telemetry(sensor_id: str, start_date: str, end_date: str):
    """Get telemetry data with caching."""
    # Expensive database query
    data = await query_telemetry_data(sensor_id, start_date, end_date)
    return data

@app.get("/analytics/summary")
@cache(expire=600)  # 10 minutes
async def get_analytics_summary():
    """Get analytics summary with caching."""
    # Expensive analytics computation
    summary = await compute_analytics_summary()
    return summary

@app.post("/telemetry")
async def create_telemetry(data: dict):
    """Create telemetry data and invalidate cache."""
    # Create data
    result = await create_telemetry_record(data)
    
    # Invalidate related caches
    await invalidate_telemetry_cache(data['sensor_id'])
    
    return result

async def invalidate_telemetry_cache(sensor_id: str):
    """Invalidate cache for specific sensor."""
    # Invalidate sensor-specific cache
    cache_key = f"telemetry:{sensor_id}:*"
    await redis_client.delete(cache_key)
```

**Why FastAPI Redis Caching Matters**: Redis caching improves API performance and reduces database load. Understanding these patterns prevents API chaos and enables reliable service performance.

### HTTP Cache Headers

```python
# http_cache.py
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib

app = FastAPI()

@app.get("/tiles/{z}/{x}/{y}.mvt")
async def get_vector_tile(z: int, x: int, y: int, response: Response):
    """Get vector tile with proper cache headers."""
    # Generate ETag based on tile content
    tile_data = await get_vector_tile_data(z, x, y)
    etag = hashlib.md5(tile_data).hexdigest()
    
    # Set cache headers
    response.headers["ETag"] = f'"{etag}"'
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["Last-Modified"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    return Response(content=tile_data, media_type="application/vnd.mapbox-vector-tile")

@app.get("/tiles/raster/{z}/{x}/{y}.png")
async def get_raster_tile(z: int, x: int, y: int, response: Response):
    """Get raster tile with proper cache headers."""
    # Generate ETag based on tile content
    tile_data = await get_raster_tile_data(z, x, y)
    etag = hashlib.md5(tile_data).hexdigest()
    
    # Set cache headers
    response.headers["ETag"] = f'"{etag}"'
    response.headers["Cache-Control"] = "public, max-age=7200"
    response.headers["Last-Modified"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    
    return Response(content=tile_data, media_type="image/png")
```

**Why HTTP Cache Headers Matter**: HTTP cache headers enable browser and CDN caching and improve global performance. Understanding these patterns prevents HTTP chaos and enables reliable web performance.

## 6) Build & Container Caching

### Docker Build Caching

```dockerfile
# Dockerfile with optimal caching
FROM python:3.11-slim as base

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cached layer)
COPY requirements.txt /app/
WORKDIR /app

# Install Python dependencies (cached layer)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (changes frequently)
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why Docker Build Caching Matters**: Docker layer caching reduces build times and improves CI/CD performance. Understanding these patterns prevents build chaos and enables reliable container performance.

### Multi-Arch Build Caching

```yaml
# docker-bake.hcl
group "default" {
  targets = ["app"]
}

target "app" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = [
    "ghcr.io/myorg/myapp:latest",
    "ghcr.io/myorg/myapp:${BAKE_GIT_COMMIT}"
  ]
  cache-from = [
    "type=gha,scope=myapp"
  ]
  cache-to = [
    "type=gha,scope=myapp,mode=max"
  ]
  args = {
    BUILDKIT_INLINE_CACHE = 1
  }
}

target "test" {
  dockerfile = "Dockerfile.test"
  platforms = ["linux/amd64"]
  tags = ["myapp:test"]
  cache-from = [
    "type=gha,scope=myapp-test"
  ]
  cache-to = [
    "type=gha,scope=myapp-test,mode=max"
  ]
}
```

**Why Multi-Arch Build Caching Matters**: Multi-arch build caching enables efficient cross-platform builds and improves CI/CD performance. Understanding these patterns prevents build chaos and enables reliable container performance.

## 7) GPU & Compute Caching

### GPU Model Caching

```python
# gpu_cache.py
import torch
import os
from pathlib import Path
import hashlib
import json

class GPUCache:
    def __init__(self, cache_dir: str = "/tmp/gpu_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_model_path(self, model_name: str, version: str) -> Path:
        """Get cached model path."""
        cache_key = hashlib.md5(f"{model_name}:{version}".encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.pt"
    
    def cache_model(self, model: torch.nn.Module, model_name: str, version: str):
        """Cache model to disk."""
        model_path = self.get_model_path(model_name, version)
        torch.save(model.state_dict(), model_path)
        
        # Cache metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "cached_at": datetime.utcnow().isoformat(),
            "model_size": model_path.stat().st_size
        }
        
        with open(model_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, model_class: torch.nn.Module, model_name: str, version: str):
        """Load cached model."""
        model_path = self.get_model_path(model_name, version)
        
        if not model_path.exists():
            return None
        
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        return model

# Usage
cache = GPUCache()

# Cache model
model = MyModel()
cache.cache_model(model, "telemetry_anomaly", "v1.0")

# Load cached model
cached_model = cache.load_model(MyModel, "telemetry_anomaly", "v1.0")
```

**Why GPU Model Caching Matters**: GPU model caching prevents model reloading and improves inference performance. Understanding these patterns prevents GPU chaos and enables reliable ML performance.

### Geospatial Compute Caching

```python
# geospatial_cache.py
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import pickle
from pathlib import Path

class GeospatialCache:
    def __init__(self, cache_dir: str = "/tmp/geo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def cache_isochrones(self, center_point: Point, travel_time: int, isochrones: gpd.GeoDataFrame):
        """Cache isochrone computation."""
        cache_key = f"isochrones_{center_point.x}_{center_point.y}_{travel_time}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(isochrones, f)
    
    def load_isochrones(self, center_point: Point, travel_time: int) -> gpd.GeoDataFrame:
        """Load cached isochrones."""
        cache_key = f"isochrones_{center_point.x}_{center_point.y}_{travel_time}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def cache_od_matrix(self, origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, 
                       od_matrix: np.ndarray):
        """Cache origin-destination matrix."""
        cache_key = f"od_matrix_{len(origins)}_{len(destinations)}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(od_matrix, f)
    
    def load_od_matrix(self, origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame) -> np.ndarray:
        """Load cached OD matrix."""
        cache_key = f"od_matrix_{len(origins)}_{len(destinations)}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

# Usage
geo_cache = GeospatialCache()

# Cache isochrones
center = Point(-122.4194, 37.7749)
isochrones = compute_isochrones(center, 30)  # 30 minutes
geo_cache.cache_isochrones(center, 30, isochrones)

# Load cached isochrones
cached_isochrones = geo_cache.load_isochrones(center, 30)
```

**Why Geospatial Compute Caching Matters**: Geospatial compute caching prevents expensive spatial computations and improves analysis performance. Understanding these patterns prevents geospatial chaos and enables reliable spatial analysis.

## 8) Anti-Patterns

### Common Caching Mistakes

```yaml
# What NOT to do
cache_anti_patterns:
  "global_cache": "Don't use global cache with no invalidation",
  "blind_db_caching": "Don't blindly cache DB joins instead of redesigning queries",
  "redis_dumping": "Don't use Redis as a dumping ground with no TTLs",
  "docker_cache_bust": "Don't let Docker builds bust cache on every run",
  "gpu_permanent": "Don't treat GPU cache directories as permanent storage",
  "no_metrics": "Don't cache without monitoring hit/miss rates",
  "no_invalidation": "Don't cache without invalidation strategy",
  "wrong_layer": "Don't cache at the wrong layer",
  "no_scope": "Don't use global caches for scoped data",
  "no_expiration": "Don't cache without expiration"
```

**Why Anti-Patterns Matter**: Understanding common mistakes prevents cache failures and performance issues. Understanding these patterns prevents cache chaos and enables reliable performance optimization.

### Cache Horror Stories

```markdown
## Cache Horror Stories

### The Stale Data Incident
**What happened**: Global cache with no invalidation for 6 months
**Result**: Users saw 6-month-old data, business decisions based on stale info
**Lesson**: Always design invalidation strategy first

### The Memory Explosion
**What happened**: Redis cache with no TTL, grew to 100GB
**Result**: System ran out of memory, all services down
**Lesson**: Set TTLs and monitor cache size

### The Docker Build Nightmare
**What happened**: Docker builds busted cache on every run
**Result**: 2-hour builds instead of 5-minute builds
**Lesson**: Order Dockerfile layers for optimal caching

### The GPU Cache Corruption
**What happened**: GPU cache treated as permanent storage
**Result**: Corrupted models, wrong predictions, ML pipeline failures
**Lesson**: GPU cache is temporary, not permanent storage
```

**Why Horror Stories Matter**: Learning from others' mistakes prevents similar failures and improves caching practices. Understanding these patterns prevents cache chaos and enables reliable performance optimization.

## 9) TL;DR Runbook

### Essential Commands

```bash
# Redis cache management
redis-cli FLUSHDB
redis-cli INFO memory
redis-cli CONFIG GET maxmemory

# Docker build with cache
docker build --cache-from myapp:latest -t myapp:new .

# Postgres materialized view refresh
REFRESH MATERIALIZED VIEW telemetry_summary;

# Clear application cache
curl -X POST http://api.example.com/cache/clear
```

### Essential Patterns

```yaml
# Essential caching patterns
cache_patterns:
  "right_layer": "Cache at the right layer: DB, tiles, flows, APIs, builds, GPUs",
  "invalidation_first": "Always design invalidation strategy first",
  "monitor_metrics": "Monitor hit/miss rates, size, and staleness",
  "geospatial_hot_paths": "Prefer materialized views & CDNs for geospatial hot paths",
  "no_correctness_hiding": "Don't let caches hide correctness bugs",
  "scope_appropriately": "Scope caches to specific contexts",
  "set_ttls": "Set appropriate TTLs for all caches",
  "warm_caches": "Plan for cache warming strategies",
  "monitor_failures": "Monitor cache failures and fallbacks",
  "test_invalidation": "Test cache invalidation thoroughly"
```

### Quick Reference

```markdown
## Emergency Cache Response

### If Cache is Stale
1. **Identify stale data**
2. **Invalidate affected caches**
3. **Warm caches with fresh data**
4. **Monitor for consistency**
5. **Update invalidation strategy**

### If Cache is Full
1. **Check cache size limits**
2. **Clear old entries**
3. **Adjust TTLs**
4. **Scale cache storage**
5. **Monitor memory usage**

### If Cache is Slow
1. **Check hit/miss rates**
2. **Optimize cache keys**
3. **Review cache layers**
4. **Monitor network latency**
5. **Consider cache warming**
```

**Why This Runbook**: These patterns cover 90% of caching needs. Master these before exploring advanced caching scenarios.

## 10) The Machine's Summary

Caching requires understanding performance characteristics, invalidation strategies, and system boundaries. When used correctly, effective caching enables system acceleration, reduces resource usage, and improves user experience. The key is understanding cache layers, invalidation strategies, and monitoring.

**The Dark Truth**: Without proper caching understanding, your systems become slow and your users become frustrated. Caching is your weapon. Use it wisely.

**The Machine's Mantra**: "In the invalidation we trust, in the monitoring we find performance, and in the layers we find the path to reliable system acceleration."

**Why This Matters**: Caching enables performance optimization that can handle complex data systems, prevent resource waste, and provide insights into system performance while ensuring technical accuracy and reliability.

---

*This guide provides the complete machinery for caching and performance. The patterns scale from simple database queries to complex geospatial computations, from basic API responses to advanced ML model inference.*
