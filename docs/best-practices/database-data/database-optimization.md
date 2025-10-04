# Database Optimization Best Practices

This document establishes production-ready database optimization patterns for geospatial systems, covering PostgreSQL/PostGIS tuning, spatial indexing, and query optimization.

## PostgreSQL/PostGIS Tuning

### Spatial Database Configuration

```sql
-- PostGIS spatial database optimization
-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Optimize PostgreSQL for spatial workloads
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Spatial indexing best practices
CREATE INDEX CONCURRENTLY idx_spatial_data_geom 
ON spatial_data USING GIST (geometry);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_spatial_data_active 
ON spatial_data USING GIST (geometry) 
WHERE status = 'active';

-- Covering indexes for performance
CREATE INDEX CONCURRENTLY idx_spatial_data_covering 
ON spatial_data USING GIST (geometry) 
INCLUDE (id, name, created_at);

-- Analyze tables for query optimization
ANALYZE spatial_data;
```

**Why:** PostGIS extensions provide spatial data types and functions. Optimized PostgreSQL settings improve spatial query performance. Spatial indexes (GIST) enable fast geometric operations.

### Query Optimization Patterns

```sql
-- Optimized spatial queries
-- Use spatial indexes effectively
EXPLAIN (ANALYZE, BUFFERS) 
SELECT s1.id, s1.name, ST_Distance(s1.geometry, s2.geometry) as distance
FROM spatial_data s1
CROSS JOIN LATERAL (
    SELECT s2.geometry
    FROM spatial_data s2
    WHERE s2.id != s1.id
    ORDER BY s1.geometry <-> s2.geometry
    LIMIT 1
) s2;

-- Spatial joins with proper indexing
SELECT a.id, b.id, ST_Area(ST_Intersection(a.geometry, b.geometry)) as intersection_area
FROM polygons_a a
JOIN polygons_b b ON ST_Intersects(a.geometry, b.geometry)
WHERE ST_Area(ST_Intersection(a.geometry, b.geometry)) > 1000;

-- Clustering for spatial locality
CLUSTER spatial_data USING idx_spatial_data_geom;
```

**Why:** LATERAL joins enable efficient nearest neighbor queries. Spatial clustering improves cache locality and query performance for spatially-related data.

## Spatial Indexing Strategies

### Advanced Spatial Indexing

```sql
-- Multi-dimensional spatial index
CREATE INDEX CONCURRENTLY idx_spatial_multi 
ON spatial_data USING GIST (geometry, created_at, status);

-- Functional indexes for computed values
CREATE INDEX CONCURRENTLY idx_spatial_area 
ON spatial_data USING BTREE (ST_Area(geometry));

-- Partial spatial indexes for filtered data
CREATE INDEX CONCURRENTLY idx_spatial_high_priority 
ON spatial_data USING GIST (geometry) 
WHERE priority = 'high';

-- Covering indexes for common queries
CREATE INDEX CONCURRENTLY idx_spatial_covering 
ON spatial_data USING GIST (geometry) 
INCLUDE (id, name, area, centroid);
```

**Why:** Multi-dimensional indexes support complex spatial queries. Functional indexes enable efficient queries on computed spatial properties. Partial indexes reduce index size and maintenance overhead.

### Spatial Index Maintenance

```sql
-- Monitor spatial index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%spatial%';

-- Reindex spatial indexes
REINDEX INDEX CONCURRENTLY idx_spatial_data_geom;

-- Update table statistics
ANALYZE spatial_data;

-- Vacuum spatial tables
VACUUM ANALYZE spatial_data;
```

**Why:** Regular index maintenance ensures optimal query performance. Monitoring index usage helps identify unused or inefficient indexes.

## Query Performance Optimization

### Spatial Query Patterns

```sql
-- Efficient spatial filtering
SELECT id, name, geometry
FROM spatial_data
WHERE geometry && ST_MakeEnvelope(-180, -90, 180, 90, 4326)
  AND ST_Intersects(geometry, ST_MakeEnvelope(-180, -90, 180, 90, 4326));

-- Optimized spatial joins
SELECT a.id, b.id, ST_Distance(a.geometry, b.geometry) as distance
FROM points_a a
JOIN points_b b ON ST_DWithin(a.geometry, b.geometry, 1000)
WHERE ST_Distance(a.geometry, b.geometry) < 1000;

-- Spatial aggregation with proper indexing
SELECT 
    ST_ClusterKMeans(geometry, 10) as cluster_id,
    COUNT(*) as point_count,
    ST_Centroid(ST_Collect(geometry)) as centroid
FROM spatial_points
GROUP BY ST_ClusterKMeans(geometry, 10);
```

**Why:** Bounding box operations (&&) are faster than precise geometric operations. ST_DWithin enables efficient distance-based queries. Spatial clustering provides data aggregation capabilities.

### Query Plan Analysis

```sql
-- Analyze query execution plans
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT s1.id, s1.name, s2.id, s2.name
FROM spatial_data s1
JOIN spatial_data s2 ON ST_Intersects(s1.geometry, s2.geometry)
WHERE s1.id != s2.id;

-- Monitor slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE query LIKE '%spatial%'
ORDER BY mean_time DESC
LIMIT 10;
```

**Why:** Query plan analysis identifies performance bottlenecks. Monitoring slow queries enables targeted optimization efforts.

## Database Partitioning

### Spatial Data Partitioning

```sql
-- Create partitioned table for spatial data
CREATE TABLE spatial_data_partitioned (
    id SERIAL,
    name VARCHAR(255),
    geometry GEOMETRY,
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE spatial_data_2024_01 PARTITION OF spatial_data_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE spatial_data_2024_02 PARTITION OF spatial_data_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Create spatial indexes on partitions
CREATE INDEX CONCURRENTLY idx_spatial_part_2024_01_geom 
ON spatial_data_2024_01 USING GIST (geometry);

CREATE INDEX CONCURRENTLY idx_spatial_part_2024_02_geom 
ON spatial_data_2024_02 USING GIST (geometry);

-- Automatic partition creation
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
    
    EXECUTE format('CREATE INDEX CONCURRENTLY %I ON %I USING GIST (geometry)',
                   'idx_' || partition_name || '_geom', partition_name);
END;
$$ LANGUAGE plpgsql;
```

**Why:** Partitioning improves query performance for large spatial datasets. Automatic partition creation reduces maintenance overhead.

### Partition Pruning

```sql
-- Enable partition pruning
SET enable_partition_pruning = on;

-- Query with partition pruning
SELECT COUNT(*)
FROM spatial_data_partitioned
WHERE created_at >= '2024-01-01' 
  AND created_at < '2024-02-01'
  AND ST_Intersects(geometry, ST_MakeEnvelope(-180, -90, 180, 90, 4326));

-- Monitor partition usage
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
WHERE tablename LIKE 'spatial_data_%'
ORDER BY n_tup_ins DESC;
```

**Why:** Partition pruning eliminates unnecessary partition scans. Monitoring partition usage helps optimize partition strategy.

## Connection Pooling and Resource Management

### Connection Pool Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2 import pool

# SQLAlchemy connection pooling
engine = create_engine(
    "postgresql://user:password@localhost/geospatial",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30
)

# Direct psycopg2 connection pooling
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    database="geospatial",
    user="user",
    password="password"
)

def get_connection():
    """Get connection from pool"""
    return connection_pool.getconn()

def return_connection(conn):
    """Return connection to pool"""
    connection_pool.putconn(conn)
```

**Why:** Connection pooling reduces connection overhead and improves resource utilization. Pre-ping ensures connection health.

### Database Resource Monitoring

```sql
-- Monitor database connections
SELECT 
    state,
    COUNT(*) as connection_count
FROM pg_stat_activity
GROUP BY state;

-- Monitor database size
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Monitor spatial index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE indexname LIKE '%spatial%'
ORDER BY idx_scan DESC;
```

**Why:** Resource monitoring enables capacity planning and performance optimization. Connection monitoring helps identify connection leaks.

## Backup and Recovery

### Spatial Data Backup

```bash
#!/bin/bash
# Spatial database backup script

# Create backup directory
BACKUP_DIR="/backups/geospatial"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/geospatial_backup_$DATE.sql"

# Create backup
pg_dump \
    --host=localhost \
    --port=5432 \
    --username=postgres \
    --dbname=geospatial \
    --verbose \
    --clean \
    --if-exists \
    --create \
    --format=plain \
    --file="$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"

# Remove old backups (keep last 7 days)
find "$BACKUP_DIR" -name "geospatial_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### Point-in-Time Recovery

```sql
-- Enable WAL archiving
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET archive_mode = on;
ALTER SYSTEM SET archive_command = 'cp %p /archive/%f';
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET hot_standby = on;

-- Create recovery configuration
-- recovery.conf
standby_mode = 'on'
primary_conninfo = 'host=primary_server port=5432 user=replication'
trigger_file = '/tmp/postgresql.trigger'
```

**Why:** Regular backups ensure data protection. Point-in-time recovery enables recovery to specific timestamps.

## Performance Tuning

### Database Configuration Tuning

```sql
-- Memory configuration
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Checkpoint configuration
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET max_wal_size = '4GB';
ALTER SYSTEM SET min_wal_size = '1GB';

-- Query planning
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Apply configuration
SELECT pg_reload_conf();
```

**Why:** Optimized configuration improves spatial query performance. Memory settings balance between different workload types.

### Spatial Query Optimization

```sql
-- Use appropriate spatial functions
-- Fast bounding box check
SELECT * FROM spatial_data 
WHERE geometry && ST_MakeEnvelope(-180, -90, 180, 90, 4326);

-- Efficient distance queries
SELECT * FROM spatial_data 
WHERE ST_DWithin(geometry, ST_Point(0, 0), 1000);

-- Spatial joins with proper indexing
SELECT a.id, b.id
FROM spatial_data a
JOIN spatial_data b ON ST_Intersects(a.geometry, b.geometry)
WHERE a.id < b.id;  -- Avoid duplicate pairs

-- Use spatial clustering
SELECT ST_ClusterKMeans(geometry, 10) as cluster_id, COUNT(*)
FROM spatial_points
GROUP BY ST_ClusterKMeans(geometry, 10);
```

**Why:** Appropriate spatial functions leverage spatial indexes effectively. Spatial clustering enables efficient data aggregation and analysis.
