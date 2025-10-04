# PostgreSQL Indexing Strategies Best Practices

**Objective**: Master senior-level PostgreSQL indexing patterns for production systems. When you need to optimize query performance, when you want to design efficient indexes, when you need enterprise-grade indexing strategies—these best practices become your weapon of choice.

## Core Principles

- **Query-Driven**: Design indexes based on actual query patterns
- **Selectivity**: Prioritize high-selectivity columns
- **Composite Indexes**: Order columns by selectivity and usage
- **Maintenance**: Monitor and maintain index health
- **Storage**: Balance performance with storage overhead

## Index Types and Usage

### B-tree Indexes

```sql
-- Create B-tree indexes for equality and range queries
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    age INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Single column B-tree indexes
CREATE INDEX idx_users_username ON users (username);
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_age ON users (age);
CREATE INDEX idx_users_created_at ON users (created_at);
CREATE INDEX idx_users_is_active ON users (is_active);

-- Composite B-tree indexes (order by selectivity)
CREATE INDEX idx_users_name_composite ON users (last_name, first_name);
CREATE INDEX idx_users_active_created ON users (is_active, created_at);
CREATE INDEX idx_users_age_active ON users (age, is_active);

-- Partial B-tree indexes
CREATE INDEX idx_users_active_users ON users (username) WHERE is_active = TRUE;
CREATE INDEX idx_users_recent_users ON users (created_at) WHERE created_at > '2024-01-01';
```

### GIN Indexes

```sql
-- Create GIN indexes for array and JSONB columns
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    tags TEXT[],
    metadata JSONB,
    search_vector TSVECTOR,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- GIN indexes for arrays
CREATE INDEX idx_articles_tags_gin ON articles USING GIN (tags);

-- GIN indexes for JSONB
CREATE INDEX idx_articles_metadata_gin ON articles USING GIN (metadata);

-- GIN indexes for full-text search
CREATE INDEX idx_articles_search_vector_gin ON articles USING GIN (search_vector);

-- GIN indexes for specific JSONB paths
CREATE INDEX idx_articles_metadata_author_gin ON articles USING GIN ((metadata -> 'author'));
CREATE INDEX idx_articles_metadata_category_gin ON articles USING GIN ((metadata -> 'category'));
```

### GiST Indexes

```sql
-- Create GiST indexes for geometric and range data
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    geom GEOMETRY(POINT, 4326),
    bbox BOX,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- GiST index for geometric data
CREATE INDEX idx_locations_geom_gist ON locations USING GIST (geom);

-- GiST index for range data
CREATE INDEX idx_locations_bbox_gist ON locations USING GIST (bbox);

-- GiST index for text search (alternative to GIN)
CREATE INDEX idx_locations_name_gist ON locations USING GIST (name gist_trgm_ops);
```

### Hash Indexes

```sql
-- Create hash indexes for equality-only queries
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Hash index for session token lookups
CREATE INDEX idx_sessions_token_hash ON sessions USING HASH (session_token);

-- Note: Hash indexes only support equality operations
-- Use B-tree for range queries or sorting
```

### BRIN Indexes

```sql
-- Create BRIN indexes for large tables with natural ordering
CREATE TABLE sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    sensor_id INTEGER NOT NULL,
    reading_value DOUBLE PRECISION NOT NULL,
    reading_time TIMESTAMPTZ NOT NULL,
    location_id INTEGER
);

-- BRIN index for time-series data
CREATE INDEX idx_sensor_readings_time_brin ON sensor_readings USING BRIN (reading_time);

-- BRIN index for sensor ID (if data is ordered by sensor)
CREATE INDEX idx_sensor_readings_sensor_brin ON sensor_readings USING BRIN (sensor_id);

-- BRIN index for location (if data is ordered by location)
CREATE INDEX idx_sensor_readings_location_brin ON sensor_readings USING BRIN (location_id);
```

## Advanced Indexing Patterns

### Covering Indexes

```sql
-- Create covering indexes to avoid table lookups
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL,
    shipping_address TEXT,
    billing_address TEXT
);

-- Covering index for order lookups
CREATE INDEX idx_orders_customer_date_covering ON orders (customer_id, order_date) 
INCLUDE (total_amount, status);

-- Covering index for status queries
CREATE INDEX idx_orders_status_date_covering ON orders (status, order_date) 
INCLUDE (customer_id, total_amount);
```

### Expression Indexes

```sql
-- Create indexes on expressions
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Expression index for case-insensitive searches
CREATE INDEX idx_products_name_lower ON products (LOWER(name));

-- Expression index for date functions
CREATE INDEX idx_products_created_year ON products (EXTRACT(YEAR FROM created_at));

-- Expression index for concatenated fields
CREATE INDEX idx_products_name_category ON products (name || ' ' || category);

-- Expression index for JSONB operations
CREATE INDEX idx_products_metadata_price ON products ((metadata ->> 'price')::numeric);
```

### Partial Indexes

```sql
-- Create partial indexes for filtered data
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL,
    event_data JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

-- Partial index for unprocessed events
CREATE INDEX idx_events_unprocessed ON events (created_at) WHERE processed = FALSE;

-- Partial index for specific event types
CREATE INDEX idx_events_login_events ON events (user_id, created_at) WHERE event_type = 'login';

-- Partial index for recent events
CREATE INDEX idx_events_recent ON events (event_type, created_at) 
WHERE created_at > CURRENT_DATE - INTERVAL '30 days';
```

## Index Optimization Strategies

### Index Usage Analysis

```sql
-- Analyze index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_scan = 0 THEN 0
        ELSE ROUND(100.0 * idx_tup_fetch / NULLIF(idx_tup_read, 0), 2)
    END AS efficiency_percent
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- Find duplicate indexes
SELECT 
    t1.schemaname,
    t1.tablename,
    t1.indexname as index1,
    t2.indexname as index2,
    pg_get_indexdef(t1.indexrelid) as index1_def,
    pg_get_indexdef(t2.indexrelid) as index2_def
FROM pg_stat_user_indexes t1
JOIN pg_stat_user_indexes t2 ON t1.schemaname = t2.schemaname 
    AND t1.tablename = t2.tablename
    AND t1.indexname < t2.indexname
WHERE pg_get_indexdef(t1.indexrelid) = pg_get_indexdef(t2.indexrelid);
```

### Index Maintenance

```sql
-- Analyze table statistics for better index usage
ANALYZE users;
ANALYZE articles;
ANALYZE orders;

-- Reindex specific indexes
REINDEX INDEX idx_users_username;
REINDEX INDEX idx_articles_search_vector_gin;

-- Reindex all indexes on a table
REINDEX TABLE users;

-- Reindex all indexes in a database
REINDEX DATABASE production;

-- Update table statistics
UPDATE pg_stat_user_tables SET last_autoanalyze = NULL WHERE relname = 'users';
```

### Index Monitoring

```python
# monitoring/index_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class IndexMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_index_usage_stats(self):
        """Get index usage statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                        pg_relation_size(indexrelid) as index_size_bytes
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                """)
                
                stats = cur.fetchall()
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting index usage stats: {e}")
            return []
        finally:
            conn.close()
    
    def find_unused_indexes(self):
        """Find unused indexes."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                        pg_relation_size(indexrelid) as index_size_bytes
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                    ORDER BY pg_relation_size(indexrelid) DESC
                """)
                
                unused_indexes = cur.fetchall()
                return unused_indexes
                
        except Exception as e:
            self.logger.error(f"Error finding unused indexes: {e}")
            return []
        finally:
            conn.close()
    
    def find_duplicate_indexes(self):
        """Find duplicate indexes."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        t1.schemaname,
                        t1.tablename,
                        t1.indexname as index1,
                        t2.indexname as index2,
                        pg_get_indexdef(t1.indexrelid) as index_def
                    FROM pg_stat_user_indexes t1
                    JOIN pg_stat_user_indexes t2 ON t1.schemaname = t2.schemaname 
                        AND t1.tablename = t2.tablename
                        AND t1.indexname < t2.indexname
                    WHERE pg_get_indexdef(t1.indexrelid) = pg_get_indexdef(t2.indexrelid)
                """)
                
                duplicate_indexes = cur.fetchall()
                return duplicate_indexes
                
        except Exception as e:
            self.logger.error(f"Error finding duplicate indexes: {e}")
            return []
        finally:
            conn.close()
    
    def get_index_bloat_info(self):
        """Get index bloat information."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                        pg_relation_size(indexrelid) as index_size_bytes,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes
                    FROM pg_stat_user_indexes
                    ORDER BY pg_relation_size(indexrelid) DESC
                """)
                
                bloat_info = cur.fetchall()
                return bloat_info
                
        except Exception as e:
            self.logger.error(f"Error getting index bloat info: {e}")
            return []
        finally:
            conn.close()
    
    def generate_index_report(self):
        """Generate comprehensive index report."""
        usage_stats = self.get_index_usage_stats()
        unused_indexes = self.find_unused_indexes()
        duplicate_indexes = self.find_duplicate_indexes()
        bloat_info = self.get_index_bloat_info()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_indexes': len(usage_stats),
            'unused_indexes_count': len(unused_indexes),
            'duplicate_indexes_count': len(duplicate_indexes),
            'usage_stats': usage_stats,
            'unused_indexes': unused_indexes,
            'duplicate_indexes': duplicate_indexes,
            'bloat_info': bloat_info
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = IndexMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_index_report()
    print(json.dumps(report, indent=2))
```

## Index Design Patterns

### Composite Index Design

```sql
-- Design composite indexes based on query patterns
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    region VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Composite index for customer order queries
CREATE INDEX idx_orders_customer_date_status ON orders (customer_id, order_date, status);

-- Composite index for regional sales analysis
CREATE INDEX idx_orders_region_date_amount ON orders (region, order_date, total_amount);

-- Composite index for status-based queries
CREATE INDEX idx_orders_status_date_customer ON orders (status, order_date, customer_id);

-- Covering index for order summaries
CREATE INDEX idx_orders_customer_summary ON orders (customer_id, order_date) 
INCLUDE (status, total_amount, region);
```

### Index for JSONB Queries

```sql
-- Create indexes for JSONB query patterns
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- GIN index for full JSONB content
CREATE INDEX idx_user_profiles_data_gin ON user_profiles USING GIN (profile_data);

-- B-tree index for specific JSONB values
CREATE INDEX idx_user_profiles_email_btree ON user_profiles 
((profile_data ->> 'email'));

-- GIN index for specific JSONB paths
CREATE INDEX idx_user_profiles_preferences_gin ON user_profiles 
USING GIN ((profile_data -> 'preferences'));

-- Expression index for JSONB operations
CREATE INDEX idx_user_profiles_age_btree ON user_profiles 
(((profile_data ->> 'age')::integer));
```

## Index Maintenance Automation

### Automated Index Maintenance

```python
# maintenance/index_maintenance.py
import psycopg2
import logging
from datetime import datetime, timedelta

class IndexMaintenance:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_tables(self, table_names=None):
        """Analyze table statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                if table_names:
                    for table_name in table_names:
                        cur.execute(f"ANALYZE {table_name}")
                        self.logger.info(f"Analyzed table: {table_name}")
                else:
                    cur.execute("ANALYZE")
                    self.logger.info("Analyzed all tables")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error analyzing tables: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def reindex_table(self, table_name):
        """Reindex all indexes on a table."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX TABLE {table_name}")
                self.logger.info(f"Reindexed table: {table_name}")
                
        except Exception as e:
            self.logger.error(f"Error reindexing table {table_name}: {e}")
        finally:
            conn.close()
    
    def reindex_index(self, index_name):
        """Reindex a specific index."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX INDEX {index_name}")
                self.logger.info(f"Reindexed index: {index_name}")
                
        except Exception as e:
            self.logger.error(f"Error reindexing index {index_name}: {e}")
        finally:
            conn.close()
    
    def drop_unused_indexes(self, dry_run=True):
        """Drop unused indexes."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                    AND indexname NOT LIKE '%_pkey'
                    AND indexname NOT LIKE '%_unique_%'
                """)
                
                unused_indexes = cur.fetchall()
                
                for schema, table, index in unused_indexes:
                    if dry_run:
                        self.logger.info(f"Would drop unused index: {schema}.{index}")
                    else:
                        cur.execute(f"DROP INDEX {schema}.{index}")
                        self.logger.info(f"Dropped unused index: {schema}.{index}")
                
                if not dry_run:
                    conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error dropping unused indexes: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def optimize_indexes(self):
        """Run comprehensive index optimization."""
        self.logger.info("Starting index optimization")
        
        # Analyze all tables
        self.analyze_tables()
        
        # Reindex critical tables
        critical_tables = ['users', 'orders', 'products', 'articles']
        for table in critical_tables:
            try:
                self.reindex_table(table)
            except Exception as e:
                self.logger.error(f"Error reindexing {table}: {e}")
        
        # Drop unused indexes (dry run first)
        self.drop_unused_indexes(dry_run=True)
        
        self.logger.info("Index optimization completed")

# Usage
if __name__ == "__main__":
    maintenance = IndexMaintenance({
        'host': 'localhost',
        'database': 'production',
        'user': 'maintenance_user',
        'password': 'maintenance_password'
    })
    
    maintenance.optimize_indexes()
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create B-tree indexes for equality and range queries
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_created_at ON users (created_at);

-- 2. Create GIN indexes for arrays and JSONB
CREATE INDEX idx_articles_tags_gin ON articles USING GIN (tags);
CREATE INDEX idx_articles_metadata_gin ON articles USING GIN (metadata);

-- 3. Create composite indexes for complex queries
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);

-- 4. Create partial indexes for filtered data
CREATE INDEX idx_events_unprocessed ON events (created_at) WHERE processed = FALSE;
```

### Essential Patterns

```python
# Complete PostgreSQL indexing setup
def setup_postgresql_indexing():
    # 1. Index types and usage
    # 2. Advanced indexing patterns
    # 3. Index optimization strategies
    # 4. Index design patterns
    # 5. Index maintenance automation
    # 6. Performance monitoring
    # 7. Index health checks
    # 8. Automated maintenance
    
    print("PostgreSQL indexing setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL indexing excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL indexing systems.*
