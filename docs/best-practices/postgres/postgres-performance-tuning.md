# PostgreSQL Performance Tuning Best Practices

**Objective**: Master senior-level PostgreSQL performance optimization for production systems. When you need to optimize query performance, when you want to tune database configuration, when you need enterprise-grade performance strategies—these best practices become your weapon of choice.

## Core Principles

- **Measure First**: Profile before optimizing
- **Index Strategy**: Right indexes for right queries
- **Configuration Tuning**: Optimize for your workload
- **Query Optimization**: Write efficient SQL
- **Monitoring**: Continuous performance monitoring

## Performance Analysis

### Query Performance Monitoring

```sql
-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View slowest queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 20;

-- Reset statistics
SELECT pg_stat_statements_reset();

-- View query execution plans
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM users WHERE email = 'user@example.com';

-- View table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### Performance Monitoring Scripts

```python
# monitoring/performance_monitor.py
import psycopg2
import json
import time
from datetime import datetime, timedelta

class PostgreSQLPerformanceMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(**self.conn_params)
    
    def get_slow_queries(self, limit=10):
        """Get slowest queries."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                ORDER BY total_time DESC 
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()
    
    def get_table_bloat(self):
        """Get table bloat information."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    n_live_tup,
                    n_dead_tup,
                    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS bloat_percent
                FROM pg_stat_user_tables
                WHERE n_live_tup > 0
                ORDER BY bloat_percent DESC;
            """)
            return cur.fetchall()
    
    def get_index_usage(self):
        """Get index usage statistics."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch,
                    idx_scan,
                    CASE 
                        WHEN idx_scan = 0 THEN 0
                        ELSE ROUND(100.0 * idx_tup_fetch / NULLIF(idx_tup_read, 0), 2)
                    END AS efficiency_percent
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC;
            """)
            return cur.fetchall()
    
    def get_connection_stats(self):
        """Get connection statistics."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections,
                    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                FROM pg_stat_activity;
            """)
            return cur.fetchone()
    
    def get_database_size(self):
        """Get database size information."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    pg_database.datname,
                    pg_size_pretty(pg_database_size(pg_database.datname)) as size
                FROM pg_database
                WHERE datname NOT IN ('template0', 'template1', 'postgres')
                ORDER BY pg_database_size(pg_database.datname) DESC;
            """)
            return cur.fetchall()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'slow_queries': self.get_slow_queries(),
            'table_bloat': self.get_table_bloat(),
            'index_usage': self.get_index_usage(),
            'connection_stats': self.get_connection_stats(),
            'database_sizes': self.get_database_size()
        }
        return report

# Usage
if __name__ == "__main__":
    monitor = PostgreSQLPerformanceMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    monitor.connect()
    report = monitor.generate_performance_report()
    print(json.dumps(report, indent=2))
```

## Configuration Optimization

### Memory Configuration

```sql
-- postgresql.conf memory settings
-- Adjust based on your system's available RAM

-- Shared memory (25% of total RAM)
shared_buffers = 2GB

-- Effective cache size (75% of total RAM)
effective_cache_size = 6GB

-- Work memory for sorting and hashing
work_mem = 64MB

-- Maintenance work memory
maintenance_work_mem = 256MB

-- Hash table size for hash joins
hash_mem_multiplier = 2.0

-- Maximum memory for parallel workers
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4
```

### I/O Configuration

```sql
-- I/O configuration for different storage types

-- For SSD storage
random_page_cost = 1.1
effective_io_concurrency = 200

-- For traditional HDD
-- random_page_cost = 4.0
-- effective_io_concurrency = 1

-- WAL configuration
wal_level = replica
max_wal_size = 2GB
min_wal_size = 80MB
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min

-- Background writer
bgwriter_delay = 200ms
bgwriter_lru_maxpages = 100
bgwriter_lru_multiplier = 2.0
```

### Connection Configuration

```sql
-- Connection settings
max_connections = 200
superuser_reserved_connections = 3

-- Connection pooling settings
tcp_keepalives_idle = 600
tcp_keepalives_interval = 30
tcp_keepalives_count = 3

-- Statement timeout
statement_timeout = 0
lock_timeout = 0
idle_in_transaction_session_timeout = 0
```

## Indexing Strategies

### Index Types and Usage

```sql
-- B-tree indexes (default)
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_created_at ON posts(created_at);

-- Partial indexes for filtered queries
CREATE INDEX idx_posts_published ON posts(created_at) 
WHERE published = true;

-- Composite indexes for multi-column queries
CREATE INDEX idx_posts_user_published ON posts(user_id, published, created_at);

-- Covering indexes (include columns)
CREATE INDEX idx_posts_covering ON posts(user_id, created_at) 
INCLUDE (title, published);

-- GIN indexes for array and JSONB columns
CREATE INDEX idx_posts_tags_gin ON posts USING GIN(tags);
CREATE INDEX idx_posts_metadata_gin ON posts USING GIN(metadata);

-- GiST indexes for geometric data
CREATE INDEX idx_locations_geom_gist ON locations USING GIST(geom);

-- Hash indexes for equality comparisons
CREATE INDEX idx_users_username_hash ON users USING HASH(username);

-- BRIN indexes for large tables with natural ordering
CREATE INDEX idx_logs_timestamp_brin ON logs USING BRIN(created_at);
```

### Index Maintenance

```sql
-- Analyze index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY schemaname, tablename;

-- Get index size information
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Query Optimization

### Query Analysis Tools

```sql
-- Enable query timing
\timing on

-- Explain plans with different options
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT u.username, p.title, p.created_at
FROM users u
JOIN posts p ON u.id = p.user_id
WHERE u.email = 'user@example.com'
ORDER BY p.created_at DESC
LIMIT 10;

-- Explain with verbose output
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)
SELECT * FROM posts 
WHERE tags @> '["postgresql"]'::jsonb
AND created_at > '2024-01-01';

-- Cost analysis
EXPLAIN (ANALYZE, BUFFERS, COSTS, FORMAT JSON)
SELECT COUNT(*) FROM posts 
WHERE published = true 
AND created_at BETWEEN '2024-01-01' AND '2024-12-31';
```

### Common Optimization Patterns

```sql
-- 1. Use appropriate WHERE clauses
-- Bad: Full table scan
SELECT * FROM posts WHERE LOWER(title) LIKE '%postgresql%';

-- Good: Use indexes
SELECT * FROM posts WHERE title ILIKE '%postgresql%';

-- 2. Avoid SELECT * in production
-- Bad
SELECT * FROM users WHERE email = 'user@example.com';

-- Good
SELECT id, username, email FROM users WHERE email = 'user@example.com';

-- 3. Use LIMIT for large result sets
SELECT * FROM posts 
WHERE published = true 
ORDER BY created_at DESC 
LIMIT 100;

-- 4. Optimize JOINs
-- Use appropriate JOIN types
SELECT u.username, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id, u.username;

-- 5. Use EXISTS instead of IN for large datasets
-- Bad
SELECT * FROM users 
WHERE id IN (SELECT user_id FROM posts WHERE published = true);

-- Good
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM posts p 
    WHERE p.user_id = u.id AND p.published = true
);
```

## Partitioning for Performance

### Table Partitioning

```sql
-- Create partitioned table
CREATE TABLE posts_partitioned (
    id SERIAL,
    user_id INTEGER,
    title VARCHAR(200),
    content TEXT,
    published BOOLEAN,
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE posts_2024_q1 PARTITION OF posts_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE posts_2024_q2 PARTITION OF posts_partitioned
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE posts_2024_q3 PARTITION OF posts_partitioned
FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE posts_2024_q4 PARTITION OF posts_partitioned
FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

-- Create indexes on partitions
CREATE INDEX idx_posts_2024_q1_created_at ON posts_2024_q1(created_at);
CREATE INDEX idx_posts_2024_q1_user_id ON posts_2024_q1(user_id);

-- Automatic partition creation
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
    
    EXECUTE format('CREATE INDEX %I ON %I(created_at)', 
                   'idx_' || partition_name || '_created_at', partition_name);
END;
$$ LANGUAGE plpgsql;
```

## Connection Pooling

### PgBouncer Configuration

```ini
# pgbouncer.ini
[databases]
production = host=localhost port=5432 dbname=production
staging = host=localhost port=5432 dbname=staging

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 50
server_round_robin = 1
ignore_startup_parameters = extra_float_digits
application_name_add_host = 1

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
```

### Application-Level Pooling

```python
# connection_pool.py
import psycopg2
from psycopg2 import pool
import threading
import time

class PostgreSQLConnectionPool:
    def __init__(self, min_conn=5, max_conn=20, **kwargs):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, **kwargs
        )
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Get connection from pool."""
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def execute_query(self, query, params=None):
        """Execute query using connection pool."""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return cur.rowcount
        finally:
            if conn:
                self.return_connection(conn)
    
    def close_all(self):
        """Close all connections in pool."""
        self.pool.closeall()

# Usage
pool = PostgreSQLConnectionPool(
    min_conn=5,
    max_conn=20,
    host='localhost',
    database='production',
    user='app_user',
    password='app_password'
)

# Execute queries
results = pool.execute_query("SELECT * FROM users WHERE active = %s", (True,))
```

## Monitoring and Alerting

### Performance Metrics Collection

```python
# monitoring/metrics_collector.py
import psycopg2
import time
import json
from prometheus_client import Gauge, Counter, Histogram, start_http_server

class PostgreSQLMetrics:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        
        # Prometheus metrics
        self.connections_total = Gauge('postgresql_connections_total', 'Total connections')
        self.active_connections = Gauge('postgresql_active_connections', 'Active connections')
        self.database_size = Gauge('postgresql_database_size_bytes', 'Database size in bytes')
        self.slow_queries = Counter('postgresql_slow_queries_total', 'Total slow queries')
        self.query_duration = Histogram('postgresql_query_duration_seconds', 'Query duration')
        
    def collect_metrics(self):
        """Collect PostgreSQL metrics."""
        try:
            conn = psycopg2.connect(**self.conn_params)
            
            with conn.cursor() as cur:
                # Connection metrics
                cur.execute("SELECT count(*) FROM pg_stat_activity;")
                total_connections = cur.fetchone()[0]
                self.connections_total.set(total_connections)
                
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
                active_connections = cur.fetchone()[0]
                self.active_connections.set(active_connections)
                
                # Database size
                cur.execute("SELECT pg_database_size(current_database());")
                db_size = cur.fetchone()[0]
                self.database_size.set(db_size)
                
                # Slow queries
                cur.execute("""
                    SELECT count(*) FROM pg_stat_statements 
                    WHERE mean_time > 1000;
                """)
                slow_queries = cur.fetchone()[0]
                self.slow_queries.inc(slow_queries)
            
            conn.close()
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
    
    def start_monitoring(self, port=8000):
        """Start metrics server."""
        start_http_server(port)
        print(f"Metrics server started on port {port}")
        
        while True:
            self.collect_metrics()
            time.sleep(30)

# Usage
if __name__ == "__main__":
    metrics = PostgreSQLMetrics({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    metrics.start_monitoring()
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Enable performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 2. Analyze slow queries
SELECT query, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;

-- 3. Check table bloat
SELECT schemaname, tablename, n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000;
```

### Essential Patterns

```python
# Complete PostgreSQL performance tuning setup
def setup_postgresql_performance_tuning():
    # 1. Performance monitoring
    # 2. Configuration optimization
    # 3. Indexing strategies
    # 4. Query optimization
    # 5. Partitioning
    # 6. Connection pooling
    # 7. Metrics collection
    # 8. Alerting
    
    print("PostgreSQL performance tuning setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL performance excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL performance systems.*
