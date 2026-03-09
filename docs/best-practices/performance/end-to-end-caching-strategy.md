# End-to-End Caching Strategy & Performance Layering: Best Practices for High-Performance Distributed Systems

**Objective**: Master production-grade caching strategies across multi-layer distributed systems. When you need to optimize API performance, accelerate ETL workloads, reduce ML inference latency, improve GIS tile delivery, and eliminate database bottlenecks—this guide provides complete caching patterns and governance.

## Introduction

Caching is the foundation of high-performance distributed systems. Without intentional, multi-layer caching, systems suffer from database overload, slow API responses, expensive recomputation, and poor user experience. This guide provides a complete framework for implementing and governing caching across all system layers.

**What This Guide Covers**:
- Multi-layer caching hierarchy and taxonomy
- Component-specific caching patterns (NGINX, Redis, Postgres, ML, GIS, ETL)
- Cache expiration and invalidation strategies
- Cache warming and precomputation
- Observability and monitoring
- Air-gapped and offline caching
- Security and governance
- Reference architectures and anti-patterns

**Prerequisites**:
- Understanding of distributed systems, databases, and web architectures
- Familiarity with Redis, NGINX, Postgres, and data processing frameworks
- Experience with performance optimization

## Context and Rationale

### Why Multi-Layer Caching Is Required

Modern distributed systems face performance challenges at every layer:

1. **API Performance**: Without caching, every request hits the database, causing latency spikes and connection exhaustion
2. **ETL Workloads**: Expensive transformations recompute unnecessarily, wasting compute and time
3. **ML Inference**: Feature extraction and model inference are expensive; caching reduces latency and cost
4. **GIS Tile Performance**: Tile generation is CPU-intensive; caching enables sub-100ms tile delivery
5. **Database Bottlenecks**: Repeated queries overwhelm databases; caching reduces load by 80-95%
6. **Dashboard Responsiveness**: Aggregations and joins are slow; caching enables instant dashboard loads
7. **Air-Gapped Clusters**: No external CDN access; local caching is essential for performance

### Impact of Caching

**Performance Improvements**:
- **API Latency**: 10-100x reduction (1000ms → 10-100ms)
- **Database Load**: 80-95% reduction in query volume
- **ETL Duration**: 50-90% reduction for repeated transformations
- **ML Inference**: 5-20x faster for cached features/embeddings
- **Tile Delivery**: 100x faster (10s → 100ms)
- **Dashboard Load**: 10-50x faster for cached aggregations

**Cost Reductions**:
- **Database Costs**: 80-95% reduction in compute requirements
- **Compute Costs**: 50-90% reduction for ETL/ML workloads
- **CDN Costs**: 70-90% reduction in origin requests
- **Storage Costs**: Optimized via intelligent caching policies

### System-Wide Caching Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser["Browser Cache<br/>(localStorage, HTTP)"]
        Mobile["Mobile App Cache"]
    end
    
    subgraph Edge["Edge Layer"]
        CDN["CDN Cache<br/>(CloudFlare, CloudFront)"]
        NGINX["NGINX Cache<br/>(microcache, static)"]
    end
    
    subgraph App["Application Layer"]
        AppMem["Application Memory<br/>(L1 cache)"]
        Redis["Redis Cluster<br/>(L2 cache)"]
        LocalDisk["Local Disk Cache<br/>(per node)"]
    end
    
    subgraph Data["Data Layer"]
        PG["Postgres Cache<br/>(buffer, MV)"]
        ObjectStore["Object Store Cache<br/>(S3/MinIO)"]
    end
    
    subgraph Compute["Compute Layer"]
        ETL["ETL Cache<br/>(Prefect/Dask)"]
        ML["ML Cache<br/>(embeddings, features)"]
        GIS["GIS Tile Cache<br/>(mbtiles, XYZ)"]
    end
    
    Browser --> CDN
    Mobile --> CDN
    CDN --> NGINX
    NGINX --> AppMem
    NGINX --> Redis
    AppMem --> Redis
    Redis --> PG
    Redis --> ObjectStore
    PG --> ObjectStore
    ETL --> Redis
    ETL --> ObjectStore
    ML --> Redis
    ML --> ObjectStore
    GIS --> Redis
    GIS --> ObjectStore
```

## Caching Hierarchy & Taxonomy

### Levels of Cache

#### L1: Application Memory Caches

**Use Case**: Ultra-fast, process-local caching for hot data.

**Characteristics**:
- **Latency**: < 1ms
- **Capacity**: Limited by process memory (typically 100MB-10GB)
- **Scope**: Single process/container
- **Persistence**: None (lost on restart)

**Examples**:
- Python `functools.lru_cache`
- Go `sync.Map` with TTL
- In-memory lookup tables

#### L2: Redis Cluster Caches

**Use Case**: Shared, distributed caching across services.

**Characteristics**:
- **Latency**: 1-5ms (local), 10-50ms (remote)
- **Capacity**: 100GB-10TB per cluster
- **Scope**: All services in cluster
- **Persistence**: Optional (RDB, AOF)

**Examples**:
- API response caching
- Session storage
- Feature stores
- Geospatial query results

#### L3: Local Disk Caches

**Use Case**: Per-node caching for large datasets.

**Characteristics**:
- **Latency**: 1-10ms (SSD), 10-100ms (HDD)
- **Capacity**: 100GB-10TB per node
- **Scope**: Single node
- **Persistence**: Survives process restart

**Examples**:
- Parquet file caching
- Model artifact caching
- Tile file caching

#### L4: NGINX Reverse-Proxy Caches

**Use Case**: HTTP response caching at the edge.

**Characteristics**:
- **Latency**: 1-5ms
- **Capacity**: 10GB-1TB per node
- **Scope**: All requests through NGINX
- **Persistence**: Optional (disk cache)

**Examples**:
- API response caching
- Static asset caching
- Microcache for dynamic content

#### L5: Database-Level Caching

**Use Case**: Postgres buffer cache, prepared statements, materialized views.

**Characteristics**:
- **Latency**: 0.1-1ms (buffer hit)
- **Capacity**: 10GB-1TB (shared_buffers)
- **Scope**: Database instance
- **Persistence**: Buffer cache lost on restart, MV persisted

**Examples**:
- Postgres shared_buffers
- Prepared statement cache
- Materialized views
- Query result cache (pg_stat_statements)

#### L6: Object Store Caches

**Use Case**: S3/MinIO caching layers, CDN edge caches.

**Characteristics**:
- **Latency**: 10-100ms (local), 100-500ms (remote)
- **Capacity**: Unlimited
- **Scope**: All services accessing object store
- **Persistence**: Permanent

**Examples**:
- S3 CloudFront edge cache
- MinIO local cache
- Parquet metadata cache

#### L7: ETL Pipeline Caches

**Use Case**: Prefect/Dask/Spark intermediate result caching.

**Characteristics**:
- **Latency**: 10ms-10s (depends on storage)
- **Capacity**: Limited by storage backend
- **Scope**: Pipeline execution
- **Persistence**: Configurable (memory, disk, object store)

**Examples**:
- Prefect result caching
- Dask distributed cache
- Spark RDD caching

#### L8: ML Model Caches

**Use Case**: Embedding stores, feature stores, ONNX warm-start caches.

**Characteristics**:
- **Latency**: 1-10ms (memory), 10-100ms (disk)
- **Capacity**: 10GB-1TB
- **Scope**: ML inference services
- **Persistence**: Model artifacts persisted

**Examples**:
- Embedding vector stores (FAISS, Pinecone)
- Feature stores (Feast, Tecton)
- ONNX model cache
- MLflow artifact cache

#### L9: Browser-Side Caches

**Use Case**: Client-side caching for web applications.

**Characteristics**:
- **Latency**: < 1ms
- **Capacity**: 5-50MB (localStorage), 50MB-1GB (IndexedDB)
- **Scope**: Single browser session
- **Persistence**: localStorage (persistent), sessionStorage (session)

**Examples**:
- HTTP cache (browser cache)
- localStorage for user preferences
- IndexedDB for large datasets
- Service Worker cache

#### L10: GIS Tile Caches

**Use Case**: XYZ vector tiles, mbtiles, precomputed raster tiles.

**Characteristics**:
- **Latency**: 1-10ms (local), 10-100ms (remote)
- **Capacity**: 10GB-1TB
- **Scope**: Tile server and clients
- **Persistence**: mbtiles files, tile directories

**Examples**:
- Vector tile cache (mbtiles)
- Raster tile cache
- XYZ tile server cache
- MapLibre GL tile cache

### Caching Modes

#### Read-Through Cache

**Definition**: Cache automatically loads data from source on miss.

**Use Case**: When you want transparent caching without application logic.

**Example**:
```python
# Redis read-through pattern
def get_user(user_id: int) -> dict:
    cache_key = f"user:{user_id}"
    
    # Try cache first
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss: load from DB
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    
    # Write to cache
    redis.setex(cache_key, 3600, json.dumps(user))
    
    return user
```

#### Write-Through Cache

**Definition**: Writes update both cache and source synchronously.

**Use Case**: When cache and source must always be consistent.

**Example**:
```python
def update_user(user_id: int, data: dict):
    # Update database
    db.execute("UPDATE users SET ... WHERE id = %s", user_id, data)
    
    # Update cache
    cache_key = f"user:{user_id}"
    redis.setex(cache_key, 3600, json.dumps(data))
```

#### Write-Behind Cache

**Definition**: Writes update cache immediately, source asynchronously.

**Use Case**: When write performance is critical and eventual consistency is acceptable.

**Example**:
```python
async def update_user_async(user_id: int, data: dict):
    # Update cache immediately
    cache_key = f"user:{user_id}"
    await redis.setex(cache_key, 3600, json.dumps(data))
    
    # Queue database update
    await write_queue.put({
        "type": "update_user",
        "user_id": user_id,
        "data": data
    })
```

#### Cache-Aside (Lazy Loading)

**Definition**: Application manages cache explicitly; cache doesn't interact with source.

**Use Case**: Most common pattern; gives application full control.

**Example**:
```python
def get_user(user_id: int) -> dict:
    cache_key = f"user:{user_id}"
    
    # Check cache
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Load from database
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    
    # Populate cache
    if user:
        redis.setex(cache_key, 3600, json.dumps(user))
    
    return user
```

#### Streaming Cache

**Definition**: Cache streams data as it's generated (e.g., large file downloads).

**Use Case**: Large files, streaming responses, progressive loading.

**Example**:
```python
def stream_large_file(file_id: str):
    cache_key = f"file:{file_id}"
    
    # Check if cached
    if redis.exists(cache_key):
        # Stream from cache
        for chunk in redis.get_stream(cache_key):
            yield chunk
    else:
        # Stream from source and cache
        for chunk in source.get_stream(file_id):
            redis.append(cache_key, chunk)
            yield chunk
```

#### Model-Level Feature Caching

**Definition**: Cache ML features and embeddings to avoid recomputation.

**Use Case**: ML inference pipelines where feature extraction is expensive.

**Example**:
```python
def get_embeddings(text: str) -> np.ndarray:
    cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
    
    # Check cache
    cached = redis.get(cache_key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    
    # Compute embedding
    embedding = model.encode(text)
    
    # Cache result
    redis.setex(cache_key, 86400, embedding.tobytes())
    
    return embedding
```

## Caching for Common System Components

### NGINX Caching

#### Microcache Patterns

**Use Case**: Cache dynamic API responses for short durations (1-60s).

**Configuration**:

```nginx
# nginx.conf
http {
    # Cache zone for microcache
    proxy_cache_path /var/cache/nginx/microcache
        levels=1:2
        keys_zone=microcache:10m
        max_size=1g
        inactive=60s
        use_temp_path=off;
    
    # Cache zone for static assets
    proxy_cache_path /var/cache/nginx/static
        levels=1:2
        keys_zone=static:100m
        max_size=10g
        inactive=7d
        use_temp_path=off;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # Microcache for API responses
        location /api/ {
            proxy_pass http://backend;
            proxy_cache microcache;
            proxy_cache_valid 200 10s;  # Cache 200 responses for 10s
            proxy_cache_valid 404 1s;   # Cache 404s for 1s
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_background_update on;
            proxy_cache_lock on;
            
            # Headers
            add_header X-Cache-Status $upstream_cache_status;
            add_header Cache-Control "public, max-age=10";
        }
        
        # Static assets
        location /static/ {
            proxy_pass http://backend;
            proxy_cache static;
            proxy_cache_valid 200 7d;
            proxy_cache_valid 404 1h;
            
            # Long cache for static assets
            add_header Cache-Control "public, max-age=604800, immutable";
        }
    }
}
```

#### Caching API Responses

**Pattern**: Cache GET requests, invalidate on POST/PUT/DELETE.

```nginx
# API caching with invalidation
location /api/v1/ {
    proxy_pass http://backend;
    proxy_cache api_cache;
    proxy_cache_methods GET HEAD;
    proxy_cache_valid 200 60s;
    proxy_cache_key "$scheme$request_method$host$request_uri";
    
    # Invalidate on write
    proxy_cache_bypass $http_x_cache_invalidate;
    
    # Stale-while-revalidate
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    proxy_cache_background_update on;
}
```

#### ETag & Last-Modified Support

```nginx
# ETag and conditional requests
location /api/ {
    proxy_pass http://backend;
    proxy_cache api_cache;
    
    # Pass ETag and Last-Modified from backend
    proxy_pass_header ETag;
    proxy_pass_header Last-Modified;
    
    # Support conditional requests
    proxy_cache_revalidate on;
    proxy_cache_use_stale error timeout updating;
}
```

#### TTL Policies

**TTL Matrix**:

| Content Type | TTL | Reason |
|-------------|-----|--------|
| Static assets (JS/CSS) | 7 days | Immutable, versioned |
| API responses (read-only) | 10s-60s | Balance freshness and performance |
| API responses (user-specific) | 1s-10s | Short TTL for personalization |
| Geospatial tiles | 24h-7d | Tiles change infrequently |
| ML inference results | 1h-24h | Results stable for input |
| Dashboard aggregations | 5m-1h | Balance freshness and load |

### Redis Caching

#### Key Naming Conventions

**Standard Pattern**: `{namespace}:{entity}:{identifier}:{field?}`

**Examples**:
```
user:123:profile
user:123:permissions
dataset:roads-v2:metadata
dataset:roads-v2:schema
ml:embedding:sha256:abc123...
gis:tile:z12:x1234:y5678
api:response:GET:/api/v1/users/123
```

**Python Helper**:

```python
# redis_key_builder.py
class RedisKeyBuilder:
    @staticmethod
    def user_profile(user_id: int) -> str:
        return f"user:{user_id}:profile"
    
    @staticmethod
    def dataset_metadata(dataset_id: str, version: str) -> str:
        return f"dataset:{dataset_id}:{version}:metadata"
    
    @staticmethod
    def ml_embedding(text_hash: str) -> str:
        return f"ml:embedding:{text_hash}"
    
    @staticmethod
    def gis_tile(z: int, x: int, y: int) -> str:
        return f"gis:tile:{z}:{x}:{y}"
```

#### TTL Conventions

**TTL Standards**:

| Data Type | TTL | Pattern |
|-----------|-----|---------|
| User sessions | 1h-24h | `EXPIRE key 3600` |
| API responses | 10s-60s | `EXPIRE key 60` |
| Geospatial queries | 1h-24h | `EXPIRE key 3600` |
| ML embeddings | 24h-7d | `EXPIRE key 86400` |
| Dataset metadata | 1h-24h | `EXPIRE key 3600` |
| Feature stores | 1h-24h | `EXPIRE key 3600` |

#### Structured Caching vs Raw Blob Caching

**Structured Caching (Hash)**:

```python
# Use Redis Hash for structured data
def cache_user_profile(user_id: int, profile: dict):
    key = f"user:{user_id}:profile"
    redis.hset(key, mapping=profile)
    redis.expire(key, 3600)

def get_user_profile(user_id: int) -> dict:
    key = f"user:{user_id}:profile"
    return redis.hgetall(key)
```

**Raw Blob Caching (String)**:

```python
# Use String for serialized data
def cache_ml_embedding(text_hash: str, embedding: np.ndarray):
    key = f"ml:embedding:{text_hash}"
    redis.setex(key, 86400, embedding.tobytes())

def get_ml_embedding(text_hash: str) -> np.ndarray:
    key = f"ml:embedding:{text_hash}"
    data = redis.get(key)
    if data:
        return np.frombuffer(data, dtype=np.float32)
    return None
```

#### Caching Geospatial Queries

**Pattern**: Cache PostGIS query results.

```python
# cache_geospatial_query.py
import hashlib
import json

def cache_geospatial_query(query: str, params: dict, result: list) -> str:
    """Cache PostGIS query result"""
    # Create cache key from query and params
    cache_data = json.dumps({"query": query, "params": params}, sort_keys=True)
    cache_key = f"gis:query:{hashlib.sha256(cache_data.encode()).hexdigest()}"
    
    # Cache result
    redis.setex(cache_key, 3600, json.dumps(result))
    
    return cache_key

def get_cached_geospatial_query(query: str, params: dict) -> list:
    """Get cached PostGIS query result"""
    cache_data = json.dumps({"query": query, "params": params}, sort_keys=True)
    cache_key = f"gis:query:{hashlib.sha256(cache_data.encode()).hexdigest()}"
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

**PostGIS Query Caching**:

```sql
-- Cache expensive spatial joins
CREATE MATERIALIZED VIEW cached_roads_with_risk AS
SELECT 
    r.road_id,
    r.geometry,
    r.risk_level,
    z.zone_id,
    z.zone_name
FROM roads_v2 r
JOIN risk_zones z ON ST_Intersects(r.geometry, z.geometry);

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY cached_roads_with_risk;
```

#### Caching ML Inferences

**Pattern**: Cache inference results by input hash.

```python
# cache_ml_inference.py
import hashlib
import json
import numpy as np

def cache_inference(model_id: str, input_data: dict, output: dict) -> str:
    """Cache ML model inference result"""
    # Create cache key from model and input
    input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"ml:inference:{model_id}:{input_hash}"
    
    # Cache result
    redis.setex(cache_key, 3600, json.dumps(output))
    
    return cache_key

def get_cached_inference(model_id: str, input_data: dict) -> dict:
    """Get cached ML model inference result"""
    input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"ml:inference:{model_id}:{input_hash}"
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

#### Redis Streams vs Redis Keys

**Use Redis Keys For**:
- Simple key-value lookups
- Cached query results
- Session storage
- Feature stores

**Use Redis Streams For**:
- Event logs
- Message queues
- Time-series data
- Real-time updates

**Example**:

```python
# Redis Streams for event logging
def log_cache_event(event_type: str, key: str, metadata: dict):
    redis.xadd(
        "cache:events",
        {
            "type": event_type,
            "key": key,
            "metadata": json.dumps(metadata),
            "timestamp": time.time()
        }
    )

# Redis Keys for caching
def cache_api_response(endpoint: str, params: dict, response: dict):
    cache_key = f"api:response:{endpoint}:{hashlib.sha256(json.dumps(params).encode()).hexdigest()}"
    redis.setex(cache_key, 60, json.dumps(response))
```

### Postgres/PostGIS Caching

#### Materialized Views and Refresh Patterns

**Create Materialized View**:

```sql
-- Materialized view for expensive aggregations
CREATE MATERIALIZED VIEW cached_road_statistics AS
SELECT 
    risk_level,
    COUNT(*) AS road_count,
    SUM(ST_Length(geometry::geography)) AS total_length_km,
    AVG(ST_Length(geometry::geography)) AS avg_length_km
FROM roads_v2
GROUP BY risk_level;

-- Create index on materialized view
CREATE INDEX idx_cached_road_statistics_risk_level 
ON cached_road_statistics(risk_level);

-- Refresh pattern: daily at 2 AM
-- Via pg_cron
SELECT cron.schedule(
    'refresh-road-statistics',
    '0 2 * * *',
    $$REFRESH MATERIALIZED VIEW CONCURRENTLY cached_road_statistics;$$
);
```

**Refresh Strategies**:

| Strategy | Use Case | Example |
|----------|----------|---------|
| **CONCURRENTLY** | Large views, zero downtime | `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_name;` |
| **Full refresh** | Small views, can tolerate downtime | `REFRESH MATERIALIZED VIEW mv_name;` |
| **Incremental** | Append-only data | Custom refresh logic |

#### Prepared Statement Caching

**Postgres Prepared Statements**:

```python
# Prepared statement caching
import psycopg2
from psycopg2 import sql

# Create prepared statement
conn = psycopg2.connect("postgresql://user:pass@localhost/db")
cur = conn.cursor()

# Prepare statement (cached by Postgres)
cur.execute("""
    PREPARE get_road AS
    SELECT * FROM roads_v2 WHERE road_id = $1
""")

# Execute prepared statement (fast)
cur.execute("EXECUTE get_road (%s)", (123,))
result = cur.fetchone()
```

**Connection Pooling with Prepared Statements**:

```python
# asyncpg with prepared statements
import asyncpg

async def get_road_cached(pool: asyncpg.Pool, road_id: int):
    """Get road using prepared statement (cached by asyncpg)"""
    async with pool.acquire() as conn:
        # Prepared statement is cached per connection
        return await conn.fetchrow(
            "SELECT * FROM roads_v2 WHERE road_id = $1",
            road_id
        )
```

#### Reducing GIST Index Scans via Memoization

**Pattern**: Cache GIST index scan results.

```python
# cache_gist_scan.py
def cache_gist_intersection(geom1_hash: str, geom2_hash: str, result: bool):
    """Cache ST_Intersects result"""
    cache_key = f"gis:intersects:{geom1_hash}:{geom2_hash}"
    redis.setex(cache_key, 3600, "1" if result else "0")

def get_cached_gist_intersection(geom1_hash: str, geom2_hash: str) -> bool:
    """Get cached ST_Intersects result"""
    cache_key = f"gis:intersects:{geom1_hash}:{geom2_hash}"
    cached = redis.get(cache_key)
    if cached:
        return cached == "1"
    return None
```

#### Caching H3 → Geometry Lookups

**Pattern**: Cache H3 cell to geometry conversions.

```python
# cache_h3_geometry.py
import h3

def cache_h3_geometry(h3_index: str, geometry: dict):
    """Cache H3 cell geometry"""
    cache_key = f"h3:geometry:{h3_index}"
    redis.setex(cache_key, 86400, json.dumps(geometry))

def get_cached_h3_geometry(h3_index: str) -> dict:
    """Get cached H3 cell geometry"""
    cache_key = f"h3:geometry:{h3_index}"
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

#### Caching Heavy Geospatial Joins

**Pattern**: Materialize expensive spatial joins.

```sql
-- Cache expensive ST_Intersects join
CREATE MATERIALIZED VIEW cached_roads_zones AS
SELECT 
    r.road_id,
    r.geometry AS road_geometry,
    z.zone_id,
    z.geometry AS zone_geometry,
    ST_Intersects(r.geometry, z.geometry) AS intersects
FROM roads_v2 r
CROSS JOIN risk_zones z
WHERE ST_DWithin(r.geometry, z.geometry, 1000);  -- 1km buffer

-- Index for fast lookups
CREATE INDEX idx_cached_roads_zones_road ON cached_roads_zones(road_id);
CREATE INDEX idx_cached_roads_zones_zone ON cached_roads_zones(zone_id);
CREATE INDEX idx_cached_roads_zones_geom ON cached_roads_zones USING GIST(road_geometry);
```

#### Partition-Level Caching

**Pattern**: Cache partition metadata and statistics.

```sql
-- Cache partition statistics
CREATE MATERIALIZED VIEW cached_partition_stats AS
SELECT 
    schemaname,
    tablename,
    partition_name,
    pg_size_pretty(pg_total_relation_size(partition_name::regclass)) AS size,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE tablename LIKE 'roads_v2_%';

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY cached_partition_stats;
```

### ONNX / MLflow / ML Inference Caching

#### Embedding Store Caching

**Pattern**: Cache embeddings in Redis or vector store.

```python
# cache_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingCache:
    def __init__(self, redis_client, model: SentenceTransformer):
        self.redis = redis_client
        self.model = model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from cache or compute"""
        cache_key = f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"
        
        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        
        # Compute embedding
        embedding = self.model.encode(text)
        
        # Cache result
        self.redis.setex(cache_key, 86400, embedding.tobytes())
        
        return embedding
```

#### Cached Inference Results

**Pattern**: Cache model inference by input hash.

```python
# cache_inference.py
def cache_model_inference(
    model_id: str,
    input_features: dict,
    output: dict
):
    """Cache model inference result"""
    input_hash = hashlib.sha256(
        json.dumps(input_features, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"ml:inference:{model_id}:{input_hash}"
    
    redis.setex(cache_key, 3600, json.dumps(output))

def get_cached_inference(model_id: str, input_features: dict) -> dict:
    """Get cached model inference result"""
    input_hash = hashlib.sha256(
        json.dumps(input_features, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"ml:inference:{model_id}:{input_hash}"
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

#### Feature Transformations Caching

**Pattern**: Cache expensive feature transformations.

```python
# cache_feature_transforms.py
def cache_feature_transform(
    transform_name: str,
    input_data: dict,
    output_features: dict
):
    """Cache feature transformation result"""
    input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"ml:features:{transform_name}:{input_hash}"
    
    redis.setex(cache_key, 3600, json.dumps(output_features))
```

#### Cache Invalidation When Model Version Changes

**Pattern**: Invalidate cache on model version update.

```python
# invalidate_model_cache.py
def invalidate_model_cache(model_id: str, new_version: str):
    """Invalidate all cache entries for a model when version changes"""
    pattern = f"ml:inference:{model_id}:*"
    
    # Find all keys matching pattern
    keys = []
    for key in redis.scan_iter(match=pattern):
        keys.append(key)
    
    # Delete all keys
    if keys:
        redis.delete(*keys)
    
    # Update model version in cache
    redis.set(f"ml:model:{model_id}:version", new_version)
```

### NiceGUI / Frontend Caching

#### In-Browser Caching

**Pattern**: Use browser cache for static assets and API responses.

```python
# nicegui/caching.py
from nicegui import ui

@ui.page("/")
async def dashboard():
    """Dashboard with caching headers"""
    ui.add_head_html("""
        <meta http-equiv="Cache-Control" content="public, max-age=3600">
    """)
    
    # Load data with cache headers
    data = await load_cached_data()
    
    ui.label("Dashboard").classes("text-2xl")
    # ... rest of UI
```

#### Component-Level Cache

**Pattern**: Cache expensive component renders.

```python
# nicegui/component_cache.py
from functools import lru_cache
from nicegui import ui

@lru_cache(maxsize=100)
def render_expensive_component(data_id: str) -> str:
    """Cache expensive component rendering"""
    # Expensive computation
    result = compute_component(data_id)
    return result

@ui.page("/")
async def page():
    data_id = "123"
    cached_result = render_expensive_component(data_id)
    ui.label(cached_result)
```

#### WebSocket Update Throttling

**Pattern**: Throttle WebSocket updates to reduce load.

```python
# nicegui/websocket_throttle.py
from nicegui import ui
import asyncio

class ThrottledUpdates:
    def __init__(self, update_func, throttle_seconds: float = 1.0):
        self.update_func = update_func
        self.throttle_seconds = throttle_seconds
        self.last_update = 0
        self.pending_update = False
    
    async def update(self, *args, **kwargs):
        """Throttled update function"""
        now = time.time()
        if now - self.last_update < self.throttle_seconds:
            self.pending_update = True
            return
        
        self.last_update = now
        self.pending_update = False
        await self.update_func(*args, **kwargs)
```

#### localStorage/sessionStorage Patterns

**Pattern**: Cache user preferences and session data.

```javascript
// frontend/cache.js
// localStorage for persistent cache
function cacheUserPreferences(prefs) {
    localStorage.setItem('user_preferences', JSON.stringify(prefs));
}

function getCachedUserPreferences() {
    const cached = localStorage.getItem('user_preferences');
    return cached ? JSON.parse(cached) : null;
}

// sessionStorage for session cache
function cacheSessionData(key, data) {
    sessionStorage.setItem(key, JSON.stringify(data));
}

function getCachedSessionData(key) {
    const cached = sessionStorage.getItem(key);
    return cached ? JSON.parse(cached) : null;
}
```

#### Cache Invalidation on User Logout

**Pattern**: Clear cache on logout or permission changes.

```python
# nicegui/cache_invalidation.py
from nicegui import ui

@ui.page("/logout")
async def logout():
    """Logout and clear cache"""
    # Clear server-side cache
    await clear_user_cache(user_id)
    
    # Redirect to login
    ui.open("/login")
    
    # Client-side: clear localStorage and sessionStorage
    ui.run_javascript("""
        localStorage.clear();
        sessionStorage.clear();
    """)
```

### Prefect / Dask Caching

#### Caching Expensive Tasks

**Prefect Task Caching**:

```python
# prefect/cached_tasks.py
from prefect import task, flow
from prefect.tasks import task_input_hash

@task(cache_key_fn=task_input_hash, cache_expiration=3600)
def expensive_transformation(data: dict) -> dict:
    """Expensive transformation with caching"""
    # Expensive computation
    result = compute_transformation(data)
    return result

@flow
def etl_flow():
    data = load_data()
    transformed = expensive_transformation(data)  # Cached if inputs unchanged
    save_data(transformed)
```

#### Fingerprint-Based Caching

**Pattern**: Cache based on input fingerprint.

```python
# prefect/fingerprint_cache.py
import hashlib
import json

def compute_fingerprint(data: dict) -> str:
    """Compute fingerprint for caching"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

@task
def cached_task(data: dict):
    """Task with fingerprint-based caching"""
    fingerprint = compute_fingerprint(data)
    cache_key = f"prefect:task:{fingerprint}"
    
    # Check cache
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute result
    result = expensive_computation(data)
    
    # Cache result
    redis.setex(cache_key, 3600, json.dumps(result))
    
    return result
```

#### Distributed Result Caching

**Dask Distributed Caching**:

```python
# dask/distributed_cache.py
from dask.distributed import Client
import dask

# Configure result caching
dask.config.set({
    'distributed.worker.memory.target': 0.8,
    'distributed.worker.memory.spill': 0.9,
    'distributed.worker.memory.pause': 0.95
})

@dask.delayed
def expensive_computation(data):
    """Expensive computation cached by Dask"""
    return compute(data)

# Computation results are cached in worker memory
client = Client()
result = expensive_computation(large_data)
result.compute()  # Cached on subsequent calls with same inputs
```

#### Persistence of Intermediate DAG Outputs

**Pattern**: Persist intermediate results to object store.

```python
# prefect/persist_intermediates.py
from prefect import task, flow
import polars as pl

@task
def load_data() -> pl.DataFrame:
    """Load data from source"""
    return pl.read_parquet("s3://data-lake/raw/data.parquet")

@task
def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    """Transform data and persist intermediate result"""
    transformed = df.with_columns([
        pl.col("value") * 2
    ])
    
    # Persist to cache
    transformed.write_parquet("s3://data-lake/cache/transformed.parquet")
    
    return transformed

@task
def aggregate_data(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate data"""
    return df.group_by("category").agg([
        pl.col("value").sum()
    ])

@flow
def etl_pipeline():
    """ETL pipeline with intermediate caching"""
    # Check if intermediate result exists
    if s3_object_exists("s3://data-lake/cache/transformed.parquet"):
        df = pl.read_parquet("s3://data-lake/cache/transformed.parquet")
    else:
        raw_data = load_data()
        df = transform_data(raw_data)
    
    result = aggregate_data(df)
    return result
```

### Object Store / Parquet / GeoParquet Caching

#### Caching Metadata

**Pattern**: Cache Parquet file metadata (schema, row counts, statistics).

```python
# cache_parquet_metadata.py
import pyarrow.parquet as pq

def cache_parquet_metadata(parquet_path: str):
    """Cache Parquet file metadata"""
    # Read metadata
    parquet_file = pq.ParquetFile(parquet_path)
    metadata = {
        "schema": parquet_file.schema.to_arrow_schema().to_pyarrow().to_json(),
        "num_rows": parquet_file.metadata.num_rows,
        "num_row_groups": parquet_file.num_row_groups,
        "file_size": parquet_file.metadata.serialized_size
    }
    
    # Cache in Redis
    cache_key = f"parquet:metadata:{hashlib.sha256(parquet_path.encode()).hexdigest()}"
    redis.setex(cache_key, 86400, json.dumps(metadata))
    
    return metadata
```

#### Caching Filtered Subsets

**Pattern**: Cache filtered Parquet subsets.

```python
# cache_filtered_parquet.py
def cache_filtered_subset(
    parquet_path: str,
    filters: list,
    result: pl.DataFrame
):
    """Cache filtered Parquet subset"""
    filter_hash = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()
    cache_key = f"parquet:filtered:{hashlib.sha256(parquet_path.encode()).hexdigest()}:{filter_hash}"
    
    # Cache result
    redis.setex(cache_key, 3600, result.write_parquet().to_bytes())
```

#### Caching Commonly Accessed Parquet Row Groups

**Pattern**: Cache hot row groups in memory.

```python
# cache_row_groups.py
def cache_hot_row_groups(parquet_path: str, row_group_indices: list):
    """Cache frequently accessed row groups"""
    parquet_file = pq.ParquetFile(parquet_path)
    
    for idx in row_group_indices:
        row_group = parquet_file.read_row_group(idx)
        cache_key = f"parquet:rowgroup:{hashlib.sha256(parquet_path.encode()).hexdigest()}:{idx}"
        redis.setex(cache_key, 3600, row_group.to_pyarrow().serialize().to_pybytes())
```

#### Cold-Storage Caching Policies

**Pattern**: Tiered caching for cold storage.

```yaml
# cold_storage_cache_policy.yaml
cache_policies:
  hot:
    storage: redis
    ttl: 3600
    access_frequency: "> 10/day"
  
  warm:
    storage: local_disk
    ttl: 86400
    access_frequency: "1-10/day"
  
  cold:
    storage: object_store
    ttl: 604800
    access_frequency: "< 1/day"
```

#### Precomputed Tiles for GIS Workloads

**Pattern**: Precompute and cache GIS tiles.

```python
# cache_gis_tiles.py
def precompute_tiles(bbox: dict, zoom_levels: list):
    """Precompute GIS tiles and cache"""
    for z in zoom_levels:
        for x, y in generate_tile_coords(bbox, z):
            tile = generate_tile(x, y, z)
            cache_key = f"gis:tile:{z}:{x}:{y}"
            
            # Cache tile
            redis.setex(cache_key, 604800, tile.to_bytes())
            
            # Also cache to mbtiles
            save_to_mbtiles(z, x, y, tile)
```

## Cache Expiration & Invalidation Strategy

### TTL Standards

**TTL Matrix**:

| Data Type | TTL | Use Case |
|-----------|-----|----------|
| **User sessions** | 1h-24h | Session data, authentication tokens |
| **API responses (dynamic)** | 10s-60s | Frequently changing data |
| **API responses (static)** | 1h-24h | Rarely changing data |
| **Geospatial queries** | 1h-24h | Spatial data changes infrequently |
| **ML embeddings** | 24h-7d | Embeddings are deterministic |
| **ML inference** | 1h-24h | Inference results for same input |
| **Dataset metadata** | 1h-24h | Metadata changes infrequently |
| **Feature stores** | 1h-24h | Features change with data updates |
| **GIS tiles** | 24h-7d | Tiles change infrequently |
| **Dashboard aggregations** | 5m-1h | Balance freshness and performance |
| **Static assets** | 7d-30d | Immutable, versioned assets |

### Type-Based Caching Rules

**Configuration**:

```yaml
# cache_ttl_rules.yaml
ttl_rules:
  user_sessions:
    default: 3600
    extended: 86400
  
  api_responses:
    dynamic: 60
    static: 3600
    user_specific: 10
  
  geospatial:
    queries: 3600
    tiles: 86400
    metadata: 3600
  
  ml:
    embeddings: 86400
    inference: 3600
    features: 3600
  
  datasets:
    metadata: 3600
    schema: 86400
    statistics: 3600
```

### Dataset Reload/Invalidation Workflow

**Pattern**: Invalidate cache on dataset update.

```python
# cache_invalidation.py
def invalidate_dataset_cache(dataset_id: str, version: str):
    """Invalidate all cache entries for a dataset"""
    patterns = [
        f"dataset:{dataset_id}:*",
        f"api:response:*dataset*{dataset_id}*",
        f"gis:query:*dataset*{dataset_id}*"
    ]
    
    for pattern in patterns:
        keys = []
        for key in redis.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            redis.delete(*keys)
    
    # Log invalidation
    log_cache_event("invalidation", dataset_id, {"version": version})
```

### Model Version Invalidation

**Pattern**: Invalidate cache on model version change.

```python
def invalidate_model_cache(model_id: str, new_version: str):
    """Invalidate cache when model version changes"""
    pattern = f"ml:inference:{model_id}:*"
    
    keys = []
    for key in redis.scan_iter(match=pattern):
        keys.append(key)
    
    if keys:
        redis.delete(*keys)
    
    # Update model version
    redis.set(f"ml:model:{model_id}:version", new_version)
```

### Schema Change Invalidation

**Pattern**: Invalidate cache on schema change.

```python
def invalidate_schema_cache(dataset_id: str, schema_version: str):
    """Invalidate cache on schema version change"""
    patterns = [
        f"dataset:{dataset_id}:*schema*",
        f"api:response:*{dataset_id}*",
        f"parquet:metadata:*{dataset_id}*"
    ]
    
    for pattern in patterns:
        keys = []
        for key in redis.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            redis.delete(*keys)
```

### GIS Tile Regeneration Strategy

**Pattern**: Regenerate tiles on data update.

```python
def regenerate_tiles(bbox: dict, zoom_levels: list):
    """Regenerate and cache GIS tiles"""
    # Invalidate existing tiles
    pattern = f"gis:tile:*"
    keys = []
    for key in redis.scan_iter(match=pattern):
        if is_tile_in_bbox(key, bbox):
            keys.append(key)
    
    if keys:
        redis.delete(*keys)
    
    # Regenerate tiles
    precompute_tiles(bbox, zoom_levels)
```

### Cache Poisoning Prevention

**Pattern**: Validate cached data before use.

```python
def get_cached_data(key: str, validator: callable) -> dict:
    """Get cached data with validation"""
    cached = redis.get(key)
    if cached:
        data = json.loads(cached)
        
        # Validate data
        if validator(data):
            return data
        else:
            # Invalid data: delete and return None
            redis.delete(key)
            return None
    
    return None
```

### Cron-Based Invalidation and Refresh Jobs

**Cron Jobs**:

```bash
# cron/cache_refresh.sh
#!/bin/bash

# Refresh materialized views
psql -U postgres -d mydb -c "REFRESH MATERIALIZED VIEW CONCURRENTLY cached_road_statistics;"

# Invalidate stale API responses
redis-cli --scan --pattern "api:response:*" | while read key; do
    ttl=$(redis-cli TTL "$key")
    if [ "$ttl" -lt 0 ]; then
        redis-cli DEL "$key"
    fi
done

# Warm cache for hot queries
python scripts/warm_cache.py
```

### Stale-While-Revalidate Model

**Pattern**: Serve stale data while refreshing in background.

```nginx
# nginx stale-while-revalidate
location /api/ {
    proxy_pass http://backend;
    proxy_cache api_cache;
    proxy_cache_valid 200 60s;
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    proxy_cache_background_update on;
    proxy_cache_lock on;
}
```

**Application-Level**:

```python
# stale_while_revalidate.py
async def get_data_with_stale(key: str, fetch_func: callable):
    """Get data with stale-while-revalidate"""
    # Try cache
    cached = redis.get(key)
    if cached:
        data = json.loads(cached)
        
        # Check if stale
        ttl = redis.ttl(key)
        if ttl > 10:  # Not stale
            return data
        else:
            # Stale: return cached, refresh in background
            asyncio.create_task(refresh_cache(key, fetch_func))
            return data
    
    # Cache miss: fetch and cache
    data = await fetch_func()
    redis.setex(key, 3600, json.dumps(data))
    return data
```

### Zero-Downtime Cache Invalidation

**Pattern**: Blue-green cache deployment.

```python
# blue_green_cache.py
def switch_cache_version(new_version: str):
    """Switch to new cache version (blue-green)"""
    # Set new version as active
    redis.set("cache:version:active", new_version)
    
    # Old version keys will expire naturally
    # New requests use new version
```

### Blue/Green Cache Deployment

**Pattern**: Deploy new cache alongside old, switch traffic.

```python
# cache_deployment.py
def deploy_cache_blue_green(new_cache_config: dict):
    """Deploy cache using blue-green pattern"""
    # Deploy new cache (green)
    deploy_cache_cluster("green", new_cache_config)
    
    # Warm green cache
    warm_cache("green")
    
    # Switch traffic to green
    switch_traffic("green")
    
    # Keep blue for rollback
    # After validation period, decommission blue
```

## Cache Warming

### ML Inference Pipeline Warming

**Pattern**: Pre-warm ML inference cache.

```python
# warm_ml_cache.py
def warm_ml_inference_cache(model_id: str, common_inputs: list):
    """Warm ML inference cache with common inputs"""
    for input_data in common_inputs:
        # Run inference
        output = model.predict(input_data)
        
        # Cache result
        cache_inference(model_id, input_data, output)
```

### Web Dashboard Warming

**Pattern**: Pre-warm dashboard aggregations.

```python
# warm_dashboard_cache.py
def warm_dashboard_cache():
    """Warm dashboard cache with common queries"""
    common_queries = [
        {"type": "road_statistics", "filters": {}},
        {"type": "risk_distribution", "filters": {}},
        {"type": "spatial_aggregation", "filters": {"bbox": default_bbox}}
    ]
    
    for query in common_queries:
        result = execute_dashboard_query(query)
        cache_key = f"dashboard:{hashlib.sha256(json.dumps(query).encode()).hexdigest()}"
        redis.setex(cache_key, 3600, json.dumps(result))
```

### Tile Server Warming

**Pattern**: Pre-generate and cache tiles.

```python
# warm_tile_cache.py
def warm_tile_cache(bbox: dict, zoom_levels: list):
    """Warm tile cache for common areas"""
    for z in zoom_levels:
        for x, y in generate_tile_coords(bbox, z):
            tile = generate_tile(x, y, z)
            cache_key = f"gis:tile:{z}:{x}:{y}"
            redis.setex(cache_key, 604800, tile.to_bytes())
```

### Prefect Pipeline Warming

**Pattern**: Pre-run expensive Prefect tasks.

```python
# warm_prefect_cache.py
from prefect import flow, task

@task
def expensive_task(data: dict):
    """Expensive task to warm cache"""
    return compute_expensive_result(data)

@flow
def warm_cache_flow():
    """Flow to warm Prefect cache"""
    common_inputs = load_common_inputs()
    
    for input_data in common_inputs:
        expensive_task(input_data)  # Results cached by Prefect
```

### Postgres Materialized View Warming

**Pattern**: Refresh materialized views proactively.

```sql
-- Refresh materialized views via pg_cron
SELECT cron.schedule(
    'refresh-materialized-views',
    '0 2 * * *',  -- Daily at 2 AM
    $$
    REFRESH MATERIALIZED VIEW CONCURRENTLY cached_road_statistics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY cached_roads_zones;
    $$
);
```

### Redis Keyspace Warming

**Pattern**: Pre-populate Redis with hot keys.

```python
# warm_redis_cache.py
def warm_redis_keyspace(hot_keys: list, fetch_func: callable):
    """Warm Redis keyspace with hot keys"""
    for key_pattern, fetch_params in hot_keys:
        # Fetch data
        data = fetch_func(**fetch_params)
        
        # Cache data
        redis.setex(key_pattern, 3600, json.dumps(data))
```

## Observability for Caching

### Cache Hit Rates

**Prometheus Metrics**:

```python
# cache_metrics.py
from prometheus_client import Counter, Histogram, Gauge

cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_type', 'key_pattern'])
cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_type', 'key_pattern'])
cache_operations = Histogram('cache_operation_duration_seconds', 'Cache operation duration', ['operation'])
cache_size = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'])

def record_cache_hit(cache_type: str, key_pattern: str):
    cache_hits.labels(cache_type=cache_type, key_pattern=key_pattern).inc()

def record_cache_miss(cache_type: str, key_pattern: str):
    cache_misses.labels(cache_type=cache_type, key_pattern=key_pattern).inc()
```

**PromQL Queries**:

```promql
# Cache hit rate
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Cache hit rate by type
sum(rate(cache_hits_total[5m])) by (cache_type) / 
(sum(rate(cache_hits_total[5m])) by (cache_type) + sum(rate(cache_misses_total[5m])) by (cache_type))

# Cache operation latency
histogram_quantile(0.95, rate(cache_operation_duration_seconds_bucket[5m]))
```

### TTL Expirations

**Monitoring TTL Distribution**:

```python
# monitor_ttl.py
def monitor_ttl_distribution():
    """Monitor TTL distribution for cache keys"""
    ttl_buckets = {
        "0-60s": 0,
        "60s-1h": 0,
        "1h-24h": 0,
        "24h+": 0
    }
    
    for key in redis.scan_iter(match="*"):
        ttl = redis.ttl(key)
        if ttl < 60:
            ttl_buckets["0-60s"] += 1
        elif ttl < 3600:
            ttl_buckets["60s-1h"] += 1
        elif ttl < 86400:
            ttl_buckets["1h-24h"] += 1
        else:
            ttl_buckets["24h+"] += 1
    
    return ttl_buckets
```

### Eviction Frequencies

**Redis Eviction Monitoring**:

```python
# monitor_evictions.py
def monitor_redis_evictions():
    """Monitor Redis eviction frequency"""
    info = redis.info("stats")
    evictions = info.get("evicted_keys", 0)
    
    # Export to Prometheus
    cache_evictions.set(evictions)
    
    return evictions
```

### Memory Fragmentation

**Redis Memory Monitoring**:

```python
# monitor_memory.py
def monitor_redis_memory():
    """Monitor Redis memory usage and fragmentation"""
    info = redis.info("memory")
    
    used_memory = info.get("used_memory", 0)
    used_memory_peak = info.get("used_memory_peak", 0)
    mem_fragmentation_ratio = info.get("mem_fragmentation_ratio", 1.0)
    
    # Export to Prometheus
    cache_memory_used.set(used_memory)
    cache_memory_peak.set(used_memory_peak)
    cache_memory_fragmentation.set(mem_fragmentation_ratio)
    
    return {
        "used_memory": used_memory,
        "peak_memory": used_memory_peak,
        "fragmentation": mem_fragmentation_ratio
    }
```

### Redis Cluster Health

**Cluster Health Monitoring**:

```python
# monitor_redis_cluster.py
def monitor_redis_cluster_health():
    """Monitor Redis cluster health"""
    cluster_info = redis.cluster_info()
    
    nodes_ok = sum(1 for node in cluster_info["nodes"] if node["status"] == "ok")
    nodes_total = len(cluster_info["nodes"])
    
    health_ratio = nodes_ok / nodes_total if nodes_total > 0 else 0
    
    # Export to Prometheus
    redis_cluster_health.set(health_ratio)
    
    return {
        "nodes_ok": nodes_ok,
        "nodes_total": nodes_total,
        "health_ratio": health_ratio
    }
```

### NGINX Microcache Logs

**NGINX Cache Logging**:

```nginx
# nginx cache logging
log_format cache_log '$remote_addr - $remote_user [$time_local] '
                     '"$request" $status $body_bytes_sent '
                     '"$http_referer" "$http_user_agent" '
                     'cache_status=$upstream_cache_status '
                     'cache_key=$request_uri';

access_log /var/log/nginx/cache.log cache_log;
```

**Parse Cache Logs**:

```python
# parse_cache_logs.py
def parse_nginx_cache_logs(log_file: str):
    """Parse NGINX cache logs for metrics"""
    cache_stats = {
        "HIT": 0,
        "MISS": 0,
        "BYPASS": 0,
        "EXPIRED": 0,
        "STALE": 0
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            if "cache_status=" in line:
                status = line.split("cache_status=")[1].split()[0]
                cache_stats[status] = cache_stats.get(status, 0) + 1
    
    return cache_stats
```

### Postgres Buffer Cache Hit Ratio

**Postgres Buffer Cache Monitoring**:

```sql
-- Buffer cache hit ratio
SELECT 
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as hit_ratio
FROM pg_statio_user_tables;

-- Index cache hit ratio
SELECT 
    sum(idx_blks_read) as idx_read,
    sum(idx_blks_hit) as idx_hit,
    sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read)) as hit_ratio
FROM pg_statio_user_indexes;
```

**Prometheus Exporter**:

```python
# postgres_cache_exporter.py
from prometheus_client import Gauge
import psycopg2

buffer_cache_hit_ratio = Gauge('postgres_buffer_cache_hit_ratio', 'Postgres buffer cache hit ratio')

def export_postgres_cache_metrics():
    """Export Postgres cache metrics to Prometheus"""
    conn = psycopg2.connect("postgresql://user:pass@localhost/db")
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            sum(heap_blks_hit)::float / 
            (sum(heap_blks_hit) + sum(heap_blks_read))::float as hit_ratio
        FROM pg_statio_user_tables
    """)
    
    hit_ratio = cur.fetchone()[0]
    buffer_cache_hit_ratio.set(hit_ratio)
```

## Air-Gapped & Offline Caching Strategy

### Offline Tile Caching

**Pattern**: Cache tiles locally for offline use.

```python
# offline_tile_cache.py
def cache_tiles_offline(bbox: dict, zoom_levels: list, cache_dir: str):
    """Cache tiles locally for offline use"""
    os.makedirs(cache_dir, exist_ok=True)
    
    for z in zoom_levels:
        for x, y in generate_tile_coords(bbox, z):
            tile_path = f"{cache_dir}/{z}/{x}/{y}.pbf"
            os.makedirs(os.path.dirname(tile_path), exist_ok=True)
            
            # Generate and save tile
            tile = generate_tile(x, y, z)
            with open(tile_path, 'wb') as f:
                f.write(tile.to_bytes())
```

### Local Embedding Stores

**Pattern**: Cache embeddings locally for offline ML.

```python
# offline_embedding_cache.py
def cache_embeddings_offline(embeddings: dict, cache_file: str):
    """Cache embeddings locally for offline use"""
    import pickle
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

def load_offline_embeddings(cache_file: str) -> dict:
    """Load cached embeddings for offline use"""
    import pickle
    
    with open(cache_file, 'rb') as f:
        return pickle.load(f)
```

### Offline Certificate Lookup Caches

**Pattern**: Cache certificate lookups for offline systems.

```python
# offline_cert_cache.py
def cache_certificates_offline(certs: list, cache_dir: str):
    """Cache certificates locally for offline lookup"""
    os.makedirs(cache_dir, exist_ok=True)
    
    for cert in certs:
        cert_path = f"{cache_dir}/{cert['id']}.pem"
        with open(cert_path, 'wb') as f:
            f.write(cert['data'])
```

### Offline ML Inference Warming

**Pattern**: Pre-warm ML models for offline inference.

```python
# offline_ml_warming.py
def warm_offline_ml_cache(model_path: str, common_inputs: list, cache_dir: str):
    """Warm ML inference cache for offline use"""
    model = load_model(model_path)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for input_data in common_inputs:
        output = model.predict(input_data)
        
        # Cache result
        input_hash = hashlib.sha256(json.dumps(input_data).encode()).hexdigest()
        cache_path = f"{cache_dir}/{input_hash}.json"
        
        with open(cache_path, 'w') as f:
            json.dump(output, f)
```

### Cache Synchronization Strategies

**Pattern**: Sync cache between nodes in air-gapped cluster.

```bash
# sync_cache.sh
#!/bin/bash

# Sync Redis cache between nodes
SOURCE_NODE="node-01"
TARGET_NODES=("node-02" "node-03" "node-04")

for target in "${TARGET_NODES[@]}"; do
    # Dump cache from source
    redis-cli --rdb /tmp/cache.rdb
    
    # Copy to target
    scp /tmp/cache.rdb "$target:/tmp/cache.rdb"
    
    # Load on target
    ssh "$target" "redis-cli --rdb /tmp/cache.rdb"
done
```

**Docker Volume Persistence**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  tile_cache:
    image: tileserver-gl
    volumes:
      - tile_cache:/data/tiles

volumes:
  redis_data:
    driver: local
  tile_cache:
    driver: local
```

## Security & Governance of Caches

### Encrypting Sensitive Caches

**Redis Encryption at Rest**:

```bash
# redis.conf
# Enable encryption
requirepass "strong-password"

# Enable TLS
tls-port 6380
tls-cert-file /etc/redis/tls/redis.crt
tls-key-file /etc/redis/tls/redis.key
tls-ca-cert-file /etc/redis/tls/ca.crt
```

**Redis Encryption in Transit**:

```python
# redis_tls.py
import redis
import ssl

redis_client = redis.Redis(
    host='redis.example.com',
    port=6380,
    ssl=True,
    ssl_cert_reqs=ssl.CERT_REQUIRED,
    ssl_ca_certs='/etc/redis/tls/ca.crt'
)
```

### Preventing Cache Leaks via Logs

**Pattern**: Sanitize cache keys in logs.

```python
# sanitize_cache_logs.py
import re

def sanitize_cache_key(key: str) -> str:
    """Sanitize cache key for logging"""
    # Remove sensitive parts
    if ":" in key:
        parts = key.split(":")
        if len(parts) > 2:
            # Keep namespace and type, hash identifier
            parts[2] = hashlib.sha256(parts[2].encode()).hexdigest()[:8]
        return ":".join(parts)
    return key

def log_cache_operation(operation: str, key: str):
    """Log cache operation with sanitized key"""
    sanitized_key = sanitize_cache_key(key)
    logger.info(f"Cache {operation}: {sanitized_key}")
```

### Sanitizing Cached API Responses

**Pattern**: Remove sensitive data before caching.

```python
# sanitize_api_response.py
def sanitize_response_for_cache(response: dict) -> dict:
    """Remove sensitive data before caching"""
    sensitive_fields = ["password", "token", "secret", "api_key", "ssn"]
    
    sanitized = response.copy()
    
    for field in sensitive_fields:
        if field in sanitized:
            del sanitized[field]
    
    return sanitized
```

### Isolation Between Tenants

**Pattern**: Namespace cache keys by tenant.

```python
# tenant_isolation.py
def get_tenant_cache_key(tenant_id: str, key: str) -> str:
    """Get cache key with tenant isolation"""
    return f"tenant:{tenant_id}:{key}"

def get_cached_data(tenant_id: str, key: str):
    """Get cached data with tenant isolation"""
    cache_key = get_tenant_cache_key(tenant_id, key)
    return redis.get(cache_key)
```

### Secure Caching for Authentication Data

**Pattern**: Short TTL, encrypted storage for auth data.

```python
# secure_auth_cache.py
from cryptography.fernet import Fernet

def cache_auth_token(user_id: int, token: str, encryption_key: bytes):
    """Cache auth token with encryption"""
    fernet = Fernet(encryption_key)
    encrypted_token = fernet.encrypt(token.encode())
    
    cache_key = f"auth:token:{user_id}"
    redis.setex(cache_key, 3600, encrypted_token)  # Short TTL

def get_cached_auth_token(user_id: int, encryption_key: bytes) -> str:
    """Get cached auth token with decryption"""
    cache_key = f"auth:token:{user_id}"
    encrypted = redis.get(cache_key)
    
    if encrypted:
        fernet = Fernet(encryption_key)
        return fernet.decrypt(encrypted).decode()
    
    return None
```

### Avoiding Caching PII or Secrets

**Pattern**: Never cache PII or secrets.

```python
# no_pii_caching.py
PII_FIELDS = ["ssn", "credit_card", "email", "phone", "address"]

def should_cache(key: str, data: dict) -> bool:
    """Check if data should be cached (no PII)"""
    # Check key
    if any(field in key.lower() for field in PII_FIELDS):
        return False
    
    # Check data
    if any(field in data for field in PII_FIELDS):
        return False
    
    return True
```

## Example End-to-End Reference Architecture

### Multi-Layer Caching Flow

```mermaid
sequenceDiagram
    participant User as User/Browser
    participant CDN as CDN Cache
    participant NGINX as NGINX Cache
    participant App as Application
    participant Redis as Redis Cache
    participant PG as Postgres
    participant ETL as ETL Pipeline
    participant ML as ML Service
    participant S3 as Object Store
    
    User->>CDN: Request /api/data
    CDN->>CDN: Check cache (HIT)
    CDN-->>User: Cached response (10ms)
    
    User->>CDN: Request /api/users/123
    CDN->>CDN: Check cache (MISS)
    CDN->>NGINX: Forward request
    NGINX->>NGINX: Check microcache (MISS)
    NGINX->>App: Forward request
    App->>Redis: Check cache (MISS)
    Redis-->>App: Cache miss
    App->>PG: Query database
    PG-->>App: User data
    App->>Redis: Cache result (TTL: 60s)
    App-->>NGINX: Response
    NGINX->>NGINX: Cache response (TTL: 10s)
    NGINX-->>CDN: Response
    CDN->>CDN: Cache response (TTL: 60s)
    CDN-->>User: Response (150ms)
    
    User->>App: Request ML inference
    App->>Redis: Check embedding cache (HIT)
    Redis-->>App: Cached embedding
    App->>ML: Inference request
    ML->>Redis: Check inference cache (MISS)
    Redis-->>ML: Cache miss
    ML->>ML: Run inference
    ML->>Redis: Cache result (TTL: 3600s)
    ML-->>App: Inference result
    App-->>User: Response (50ms)
    
    ETL->>S3: Load Parquet file
    S3->>ETL: File data
    ETL->>ETL: Transform data
    ETL->>Redis: Cache intermediate result
    ETL->>S3: Write result
```

### Architecture Diagram

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser["Browser<br/>(HTTP Cache, localStorage)"]
        Mobile["Mobile App<br/>(Local Cache)"]
    end
    
    subgraph Edge["Edge Layer"]
        CDN["CDN<br/>(CloudFlare, CloudFront)"]
        NGINX["NGINX<br/>(Microcache, Static)"]
    end
    
    subgraph App["Application Layer"]
        API["API Service<br/>(L1 Memory Cache)"]
        Redis["Redis Cluster<br/>(L2 Shared Cache)"]
        LocalDisk["Local Disk<br/>(L3 Per-Node Cache)"]
    end
    
    subgraph Data["Data Layer"]
        PG["Postgres<br/>(Buffer Cache, MV)"]
        S3["Object Store<br/>(S3/MinIO Cache)"]
    end
    
    subgraph Compute["Compute Layer"]
        ETL["ETL Pipeline<br/>(Prefect/Dask Cache)"]
        ML["ML Service<br/>(Embedding, Inference Cache)"]
        GIS["GIS Service<br/>(Tile Cache)"]
    end
    
    Browser --> CDN
    Mobile --> CDN
    CDN --> NGINX
    NGINX --> API
    API --> Redis
    API --> LocalDisk
    Redis --> PG
    Redis --> S3
    PG --> S3
    ETL --> Redis
    ETL --> S3
    ML --> Redis
    ML --> S3
    GIS --> Redis
    GIS --> S3
```

## Anti-Patterns

### Caching Unbounded Data Structures

**Symptom**: Cache grows unbounded, causing OOM errors.

**Why It's Dangerous**: Memory exhaustion, service crashes, data loss.

**Fix**: Set max size, use LRU eviction, implement TTL.

```python
# Fix: Bounded cache with LRU
from functools import lru_cache

@lru_cache(maxsize=1000)  # Limit to 1000 entries
def expensive_function(arg):
    return compute(arg)
```

### Mixing Unrelated Cache Keys

**Symptom**: Cache keys collide, causing data corruption.

**Why It's Dangerous**: Wrong data returned, security issues.

**Fix**: Use namespaced keys, consistent naming convention.

```python
# Fix: Namespaced keys
def get_cache_key(namespace: str, entity: str, id: str) -> str:
    return f"{namespace}:{entity}:{id}"
```

### No TTL on Hot Keys

**Symptom**: Cache never expires, stale data served indefinitely.

**Why It's Dangerous**: Users see outdated data, incorrect decisions.

**Fix**: Always set TTL, use appropriate expiration times.

```python
# Fix: Always set TTL
redis.setex(key, 3600, value)  # TTL: 1 hour
```

### Caching Secrets

**Symptom**: Secrets stored in cache, exposed if cache compromised.

**Why It's Dangerous**: Security breach, credential theft.

**Fix**: Never cache secrets, use secure storage (Vault).

```python
# Fix: Don't cache secrets
def get_secret(key: str):
    # Never cache secrets
    return vault.get_secret(key)
```

### Cache Stampede

**Symptom**: Many requests miss cache simultaneously, all hit database.

**Why It's Dangerous**: Database overload, service degradation.

**Fix**: Use cache locking, exponential backoff, pre-warming.

```python
# Fix: Cache locking
import redis.lock

def get_data_with_lock(key: str):
    lock = redis.lock.Lock(redis, f"lock:{key}", timeout=5)
    
    if lock.acquire(blocking=False):
        try:
            # Check cache again
            cached = redis.get(key)
            if cached:
                return json.loads(cached)
            
            # Fetch and cache
            data = fetch_from_db()
            redis.setex(key, 3600, json.dumps(data))
            return data
        finally:
            lock.release()
    else:
        # Wait for other request to finish
        time.sleep(0.1)
        return get_data_with_lock(key)
```

### Cache Avalanche

**Symptom**: Many cache keys expire simultaneously, causing thundering herd.

**Why It's Dangerous**: Sudden load spike, service degradation.

**Fix**: Stagger TTLs, use jitter, implement circuit breakers.

```python
# Fix: TTL jitter
import random

def set_cache_with_jitter(key: str, value: str, base_ttl: int):
    jitter = random.randint(-300, 300)  # ±5 minutes
    ttl = base_ttl + jitter
    redis.setex(key, ttl, value)
```

### Cache Over-Invalidation

**Symptom**: Cache invalidated too frequently, low hit rate.

**Why It's Dangerous**: Poor performance, wasted resources.

**Fix**: Invalidate only when necessary, use versioned keys.

```python
# Fix: Versioned keys
def get_versioned_key(key: str, version: str) -> str:
    return f"{key}:v{version}"

# Invalidate by updating version, not deleting keys
def invalidate_cache(namespace: str):
    new_version = str(int(time.time()))
    redis.set(f"{namespace}:version", new_version)
```

### Silent Cache Poisoning

**Symptom**: Invalid data cached, served to users without detection.

**Why It's Dangerous**: Wrong results, security issues.

**Fix**: Validate cached data, use checksums, monitor cache health.

```python
# Fix: Validate cached data
def get_validated_cache(key: str, validator: callable):
    cached = redis.get(key)
    if cached:
        data = json.loads(cached)
        if validator(data):
            return data
        else:
            # Invalid: delete and return None
            redis.delete(key)
    return None
```

### Double-Writing to Cache and DB Without Transactionality

**Symptom**: Cache and DB out of sync, inconsistent data.

**Why It's Dangerous**: Data inconsistency, wrong results.

**Fix**: Use transactions, write-through pattern, eventual consistency.

```python
# Fix: Transactional write
def update_data_transactional(key: str, data: dict):
    with db.transaction():
        # Update database
        db.execute("UPDATE table SET ... WHERE id = %s", key, data)
        
        # Update cache
        redis.setex(f"data:{key}", 3600, json.dumps(data))
```

## Implementation Checklist

### Naming Conventions

- [ ] Use consistent namespace pattern: `{namespace}:{entity}:{id}:{field?}`
- [ ] Document namespace meanings
- [ ] Use lowercase, hyphens for multi-word names
- [ ] Include version in key when applicable
- [ ] Avoid special characters in keys

### TTL Rules

- [ ] Define TTL standards for each data type
- [ ] Use jitter for TTL to prevent stampedes
- [ ] Document TTL rationale
- [ ] Monitor TTL distribution
- [ ] Review TTLs quarterly

### Monitoring Rules

- [ ] Track cache hit rates by type
- [ ] Monitor cache size and memory usage
- [ ] Alert on low hit rates (< 80%)
- [ ] Alert on high eviction rates
- [ ] Track cache operation latency
- [ ] Monitor cache error rates

### Patterns per System

- [ ] **NGINX**: Microcache for dynamic content, long cache for static
- [ ] **Redis**: Structured keys, appropriate TTLs, eviction policies
- [ ] **Postgres**: Materialized views, prepared statements, buffer cache tuning
- [ ] **ML**: Embedding cache, inference cache, feature store
- [ ] **ETL**: Task result caching, intermediate result persistence
- [ ] **GIS**: Tile cache, query result cache, metadata cache

### Steps for Adding a New Cache

1. **Identify caching need**: Slow query, repeated computation, high load
2. **Choose cache layer**: L1 (memory), L2 (Redis), L3 (disk), etc.
3. **Define cache key pattern**: Follow naming convention
4. **Set TTL**: Based on data volatility
5. **Implement cache logic**: Read-through, write-through, etc.
6. **Add monitoring**: Hit rate, latency, size
7. **Test**: Verify correctness, performance improvement
8. **Document**: Cache purpose, TTL, invalidation strategy

### Governance & Review Process

- [ ] **Quarterly cache review**: Review hit rates, TTLs, patterns
- [ ] **Cache audit**: Identify unused caches, optimize TTLs
- [ ] **Performance review**: Measure cache impact on system performance
- [ ] **Security review**: Ensure no PII/secrets cached
- [ ] **Documentation**: Keep cache patterns documented

## See Also

- **[Caching & Performance Layers](../architecture-design/caching-performance.md)** - High-level caching patterns
- **[NGINX Best Practices](../docker-infrastructure/nginx-best-practices.md)** - NGINX configuration
- **[PostgreSQL Performance Tuning](../postgres/postgres-performance-tuning.md)** - Database optimization
- **[Prefect FIFO with Redis](../../tutorials/system-administration/prefect-fifo-redis.md)** - Redis-backed scheduling; **[Performance Best Practices](index.md)** - Caching index

---

*This guide provides a complete framework for end-to-end caching strategies. Start with monitoring to identify bottlenecks, then implement caching layers systematically. The goal is high performance without operational complexity.*

