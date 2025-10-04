# Postgres FDW Best Practices

**Objective**: FDWs make Postgres the query engine for your data lake — but misuse them, and you'll summon performance demons. Here's how to tame them.

FDWs make Postgres the query engine for your data lake — but misuse them, and you'll summon performance demons. Here's how to tame them.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **FDWs ≠ free lunch**
   - They delegate work to external engines/filesystems
   - Network latency and I/O costs are real
   - Plan for performance implications
   - Monitor resource usage

2. **Schema drift is real**
   - Always validate column types before FDW creation
   - Handle nullability mismatches gracefully
   - Plan for schema evolution
   - Document schema changes

3. **Push filtering/projection down whenever possible**
   - Reduce data transfer over network
   - Leverage external engine optimizations
   - Minimize Postgres processing overhead
   - Use WHERE clauses aggressively

4. **Keep FDW schemas small, compose with materialized views**
   - Avoid massive monolithic FDW tables
   - Use materialized views for heavy joins
   - Cache frequently accessed data
   - Plan for query optimization

5. **Security and governance are non-negotiable**
   - Store credentials securely
   - Audit access patterns
   - Limit FDW privileges
   - Monitor for data leakage

**Why These Principles**: FDWs require understanding external data sources, performance implications, and security considerations. Understanding these patterns prevents performance chaos and enables reliable federated data access.

## 1) Core Principles

### The FDW Reality

```yaml
# What you thought FDWs were
fdw_fantasy:
  "performance": "FDWs are as fast as local tables"
  "simplicity": "Just point at files and query"
  "reliability": "FDWs never fail or change"
  "security": "FDWs are secure by default"

# What FDWs actually are
fdw_reality:
  "performance": "Network and I/O costs are real"
  "simplicity": "Complex configuration and maintenance"
  "reliability": "External dependencies can break"
  "security": "Credentials and access must be managed"
```

**Why Reality Checks Matter**: Understanding the true nature of FDWs enables proper planning and risk management. Understanding these patterns prevents performance chaos and enables reliable federated data access.

### FDW Performance Characteristics

```markdown
## FDW Performance Profile

### Network I/O
- **Latency**: 10-100ms per query (depending on distance)
- **Bandwidth**: Limited by network capacity
- **Caching**: Minimal, mostly at filesystem level
- **Concurrency**: Limited by external system capacity

### Memory Usage
- **Buffer**: Limited by external system
- **Sorting**: Often done in external system
- **Joins**: May require data transfer to Postgres
- **Aggregations**: Pushdown when possible

### CPU Usage
- **Pushdown**: Filters and projections
- **Local Processing**: Complex joins and aggregations
- **Type Conversion**: Overhead for data type mismatches
- **Query Planning**: Additional complexity
```

**Why Performance Understanding Matters**: Proper FDW performance planning enables efficient federated queries and prevents system overload. Understanding these patterns prevents performance chaos and enables reliable federated data access.

## 2) Common FDWs in Practice

### parquet_fdw (Local Parquet Files)

```sql
-- Install parquet_fdw extension
CREATE EXTENSION IF NOT EXISTS parquet_fdw;

-- Create foreign server
CREATE SERVER parquet_server
FOREIGN DATA WRAPPER parquet_fdw
OPTIONS (
    filename '/path/to/data/'
);

-- Create foreign table
CREATE FOREIGN TABLE telemetry_parquet (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER parquet_server
OPTIONS (
    filename '/path/to/data/telemetry_2024_01.parquet',
    format 'parquet'
);

-- Query with pushdown filters
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
  AND temperature > 25.0
  AND sensor_id = 'sensor_001';
```

**Why parquet_fdw Matters**: Local parquet_fdw enables efficient querying of parquet files with pushdown optimization. Understanding these patterns prevents I/O chaos and enables reliable local data access.

### parquet_s3_fdw (S3/MinIO/VAST)

```sql
-- Install parquet_s3_fdw extension
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;

-- Create foreign server with credentials
CREATE SERVER parquet_s3_server
FOREIGN DATA WRAPPER parquet_s3_fdw
OPTIONS (
    aws_region 'us-west-2',
    aws_access_key_id 'your_access_key',
    aws_secret_access_key 'your_secret_key',
    aws_session_token 'your_session_token'  -- Optional
);

-- Create foreign table
CREATE FOREIGN TABLE telemetry_s3 (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/2024/01/',
    format 'parquet'
);

-- Query with S3 pushdown
SELECT sensor_id, AVG(temperature) as avg_temp
FROM telemetry_s3
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01'
  AND sensor_id IN ('sensor_001', 'sensor_002')
GROUP BY sensor_id;
```

**Why parquet_s3_fdw Matters**: S3-based parquet_fdw enables cloud-scale data lake querying with pushdown optimization. Understanding these patterns prevents cloud chaos and enables reliable cloud data access.

### duckdb_fdw (DuckDB Vectorized Execution)

```sql
-- Install duckdb_fdw extension
CREATE EXTENSION IF NOT EXISTS duckdb_fdw;

-- Create foreign server
CREATE SERVER duckdb_server
FOREIGN DATA WRAPPER duckdb_fdw
OPTIONS (
    database '/path/to/data.duckdb'
);

-- Create foreign table
CREATE FOREIGN TABLE telemetry_duckdb (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER duckdb_server
OPTIONS (
    table 'telemetry'
);

-- Leverage DuckDB's vectorized execution
SELECT 
    sensor_id,
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(temperature) as avg_temp,
    STDDEV(temperature) as temp_stddev
FROM telemetry_duckdb
WHERE timestamp >= '2024-01-01'
  AND temperature IS NOT NULL
GROUP BY sensor_id, DATE_TRUNC('hour', timestamp)
ORDER BY sensor_id, hour;
```

**Why duckdb_fdw Matters**: DuckDB FDW leverages vectorized execution for analytical workloads with excellent performance. Understanding these patterns prevents analytical chaos and enables reliable analytical data access.

### postgres_fdw (Postgres-to-Postgres)

```sql
-- Install postgres_fdw extension
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

-- Create foreign server
CREATE SERVER remote_postgres
FOREIGN DATA WRAPPER postgres_fdw
OPTIONS (
    host 'remote-db.example.com',
    port '5432',
    dbname 'analytics_db'
);

-- Create user mapping
CREATE USER MAPPING FOR current_user
SERVER remote_postgres
OPTIONS (
    user 'analytics_user',
    password 'secure_password'
);

-- Create foreign table
CREATE FOREIGN TABLE remote_analytics (
    id INTEGER,
    metric_name VARCHAR(100),
    metric_value DOUBLE PRECISION,
    timestamp TIMESTAMP WITH TIME ZONE,
    tags JSONB
) SERVER remote_postgres
OPTIONS (
    schema_name 'public',
    table_name 'analytics_metrics'
);

-- Query with pushdown
SELECT metric_name, AVG(metric_value) as avg_value
FROM remote_analytics
WHERE timestamp >= '2024-01-01'
  AND metric_name IN ('cpu_usage', 'memory_usage')
GROUP BY metric_name;
```

**Why postgres_fdw Matters**: Postgres-to-Postgres FDW enables federated queries across Postgres clusters with pushdown optimization. Understanding these patterns prevents federation chaos and enables reliable distributed data access.

## 3) Schema Management & Type Safety

### Schema Discovery

```sql
-- Discover parquet schema using DuckDB
SELECT 
    column_name,
    data_type,
    is_nullable,
    comment
FROM parquet_metadata('/path/to/data.parquet')
ORDER BY column_name;

-- Alternative: Use parquet_schema function
SELECT parquet_schema('/path/to/data.parquet');

-- Check for schema drift
SELECT 
    column_name,
    data_type,
    is_nullable
FROM parquet_metadata('/path/to/data.parquet')
WHERE column_name IN ('temperature', 'humidity', 'pressure')
ORDER BY column_name;
```

**Why Schema Discovery Matters**: Proper schema discovery prevents type mismatches and query failures. Understanding these patterns prevents schema chaos and enables reliable data access.

### Type Safety and Validation

```sql
-- Create wrapper function for type-safe FDW creation
CREATE OR REPLACE FUNCTION create_telemetry_fdw(
    table_name TEXT,
    file_path TEXT,
    server_name TEXT DEFAULT 'parquet_server'
) RETURNS VOID AS $$
DECLARE
    schema_info RECORD;
    column_def TEXT;
    column_defs TEXT[] := ARRAY[]::TEXT[];
BEGIN
    -- Validate file exists and is readable
    IF NOT EXISTS (SELECT 1 FROM pg_stat_file(file_path)) THEN
        RAISE EXCEPTION 'File % does not exist or is not readable', file_path;
    END IF;
    
    -- Get schema information
    FOR schema_info IN 
        SELECT column_name, data_type, is_nullable
        FROM parquet_metadata(file_path)
        ORDER BY column_name
    LOOP
        -- Map parquet types to Postgres types
        CASE schema_info.data_type
            WHEN 'INTEGER' THEN column_def := schema_info.column_name || ' INTEGER'
            WHEN 'DOUBLE' THEN column_def := schema_info.column_name || ' DOUBLE PRECISION'
            WHEN 'VARCHAR' THEN column_def := schema_info.column_name || ' VARCHAR(255)'
            WHEN 'TIMESTAMP' THEN column_def := schema_info.column_name || ' TIMESTAMP WITH TIME ZONE'
            ELSE column_def := schema_info.column_name || ' TEXT'
        END CASE;
        
        -- Handle nullability
        IF NOT schema_info.is_nullable THEN
            column_def := column_def || ' NOT NULL';
        END IF;
        
        column_defs := array_append(column_defs, column_def);
    END LOOP;
    
    -- Create foreign table
    EXECUTE format('
        CREATE FOREIGN TABLE %I (%s)
        SERVER %I
        OPTIONS (filename %L, format ''parquet'')
    ', table_name, array_to_string(column_defs, ', '), server_name, file_path);
    
    RAISE NOTICE 'Created foreign table % with % columns', table_name, array_length(column_defs, 1);
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT create_telemetry_fdw('telemetry_2024_01', '/path/to/telemetry_2024_01.parquet');
```

**Why Type Safety Matters**: Type-safe FDW creation prevents runtime errors and ensures data integrity. Understanding these patterns prevents type chaos and enables reliable data access.

### Schema Evolution Handling

```sql
-- Create metadata table for FDW schema tracking
CREATE TABLE fdw_schema_metadata (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    column_definitions JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Function to track schema changes
CREATE OR REPLACE FUNCTION track_fdw_schema(
    table_name TEXT,
    file_path TEXT,
    schema_version TEXT DEFAULT '1.0'
) RETURNS VOID AS $$
DECLARE
    column_defs JSONB;
BEGIN
    -- Get current schema
    SELECT jsonb_agg(
        jsonb_build_object(
            'column_name', column_name,
            'data_type', data_type,
            'is_nullable', is_nullable
        )
    ) INTO column_defs
    FROM parquet_metadata(file_path);
    
    -- Insert or update schema metadata
    INSERT INTO fdw_schema_metadata (table_name, file_path, schema_version, column_definitions)
    VALUES (table_name, file_path, schema_version, column_defs)
    ON CONFLICT (table_name, file_path) 
    DO UPDATE SET 
        schema_version = EXCLUDED.schema_version,
        column_definitions = EXCLUDED.column_definitions,
        updated_at = NOW();
    
    RAISE NOTICE 'Tracked schema for table % version %', table_name, schema_version;
END;
$$ LANGUAGE plpgsql;

-- Check for schema drift
CREATE OR REPLACE FUNCTION check_schema_drift(
    table_name TEXT,
    file_path TEXT
) RETURNS TABLE(
    column_name TEXT,
    current_type TEXT,
    new_type TEXT,
    is_drift BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH current_schema AS (
        SELECT column_name, data_type
        FROM parquet_metadata(file_path)
    ),
    stored_schema AS (
        SELECT 
            (col->>'column_name')::TEXT as column_name,
            (col->>'data_type')::TEXT as data_type
        FROM fdw_schema_metadata fsm,
             jsonb_array_elements(fsm.column_definitions) as col
        WHERE fsm.table_name = check_schema_drift.table_name
    )
    SELECT 
        cs.column_name,
        cs.data_type as current_type,
        ss.data_type as new_type,
        (cs.data_type != ss.data_type) as is_drift
    FROM current_schema cs
    FULL OUTER JOIN stored_schema ss ON cs.column_name = ss.column_name
    WHERE cs.data_type != ss.data_type OR cs.column_name IS NULL OR ss.column_name IS NULL;
END;
$$ LANGUAGE plpgsql;
```

**Why Schema Evolution Matters**: Proper schema evolution handling prevents query failures and enables reliable data access. Understanding these patterns prevents evolution chaos and enables reliable data access.

## 4) Performance Best Practices

### Pushdown Optimization

```sql
-- Good: Push filters down to FDW
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01'
  AND temperature > 25.0
  AND sensor_id = 'sensor_001';

-- Bad: Filter after data transfer
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE temperature > 25.0;

-- Good: Use column projection
SELECT sensor_id, temperature
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01';

-- Bad: Select all columns
SELECT *
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01';
```

**Why Pushdown Optimization Matters**: Pushdown optimization reduces network I/O and improves query performance. Understanding these patterns prevents performance chaos and enables reliable data access.

### Partitioning Strategies

```sql
-- Create partitioned foreign tables
CREATE FOREIGN TABLE telemetry_2024_01 (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/2024/01/',
    format 'parquet'
);

CREATE FOREIGN TABLE telemetry_2024_02 (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/2024/02/',
    format 'parquet'
);

-- Create partitioned view
CREATE VIEW telemetry_partitioned AS
SELECT * FROM telemetry_2024_01
UNION ALL
SELECT * FROM telemetry_2024_02;

-- Query with partition pruning
SELECT sensor_id, AVG(temperature) as avg_temp
FROM telemetry_partitioned
WHERE timestamp >= '2024-01-15'
  AND timestamp < '2024-02-15'
  AND sensor_id = 'sensor_001'
GROUP BY sensor_id;
```

**Why Partitioning Matters**: Partitioning enables efficient data access and reduces scan overhead. Understanding these patterns prevents scan chaos and enables reliable data access.

### Materialized View Caching

```sql
-- Create materialized view for heavy joins
CREATE MATERIALIZED VIEW telemetry_summary AS
SELECT 
    sensor_id,
    DATE_TRUNC('day', timestamp) as date,
    AVG(temperature) as avg_temp,
    MIN(temperature) as min_temp,
    MAX(temperature) as max_temp,
    AVG(humidity) as avg_humidity,
    COUNT(*) as record_count
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
GROUP BY sensor_id, DATE_TRUNC('day', timestamp);

-- Create index on materialized view
CREATE INDEX idx_telemetry_summary_sensor_date 
ON telemetry_summary (sensor_id, date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW telemetry_summary;

-- Query materialized view
SELECT sensor_id, date, avg_temp, record_count
FROM telemetry_summary
WHERE sensor_id = 'sensor_001'
  AND date >= '2024-01-01'
ORDER BY date;
```

**Why Materialized View Caching Matters**: Materialized views cache expensive computations and improve query performance. Understanding these patterns prevents computation chaos and enables reliable data access.

### Hybrid Patterns

```sql
-- Stage frequently accessed data locally
CREATE TABLE telemetry_staged (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions
CREATE TABLE telemetry_staged_2024_01 PARTITION OF telemetry_staged
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ETL process to stage data
INSERT INTO telemetry_staged_2024_01
SELECT * FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01';

-- Query staged data
SELECT sensor_id, AVG(temperature) as avg_temp
FROM telemetry_staged
WHERE timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01'
GROUP BY sensor_id;
```

**Why Hybrid Patterns Matter**: Hybrid patterns combine FDW benefits with local performance for frequently accessed data. Understanding these patterns prevents access chaos and enables reliable data access.

## 5) Security & Governance

### Credential Management

```sql
-- Create secure server with credentials
CREATE SERVER parquet_s3_secure
FOREIGN DATA WRAPPER parquet_s3_fdw
OPTIONS (
    aws_region 'us-west-2',
    aws_access_key_id 'AKIAIOSFODNN7EXAMPLE',
    aws_secret_access_key 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
);

-- Create user mapping with limited privileges
CREATE USER MAPPING FOR analytics_user
SERVER parquet_s3_secure
OPTIONS (
    user 'analytics_user',
    password 'secure_password'
);

-- Grant limited access
GRANT USAGE ON FOREIGN SERVER parquet_s3_secure TO analytics_user;
GRANT SELECT ON telemetry_parquet TO analytics_user;
```

**Why Credential Management Matters**: Secure credential management prevents unauthorized access and data leakage. Understanding these patterns prevents security chaos and enables reliable data access.

### Access Auditing

```sql
-- Create audit table
CREATE TABLE fdw_access_audit (
    id SERIAL PRIMARY KEY,
    user_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    query_text TEXT,
    access_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    success BOOLEAN DEFAULT TRUE
);

-- Create audit function
CREATE OR REPLACE FUNCTION audit_fdw_access()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO fdw_access_audit (user_name, table_name, query_text, ip_address)
    VALUES (
        current_user,
        TG_TABLE_NAME,
        current_query(),
        inet_client_addr()
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit trigger
CREATE TRIGGER audit_telemetry_parquet_access
    AFTER SELECT ON telemetry_parquet
    FOR EACH STATEMENT
    EXECUTE FUNCTION audit_fdw_access();

-- Query audit logs
SELECT 
    user_name,
    table_name,
    access_time,
    ip_address,
    success
FROM fdw_access_audit
WHERE access_time >= '2024-01-01'
ORDER BY access_time DESC;
```

**Why Access Auditing Matters**: Access auditing enables security monitoring and compliance. Understanding these patterns prevents security chaos and enables reliable data access.

### Role-Based Access Control

```sql
-- Create roles for different access levels
CREATE ROLE fdw_readonly;
CREATE ROLE fdw_analytics;
CREATE ROLE fdw_admin;

-- Grant server usage
GRANT USAGE ON FOREIGN SERVER parquet_s3_secure TO fdw_readonly;
GRANT USAGE ON FOREIGN SERVER parquet_s3_secure TO fdw_analytics;
GRANT USAGE ON FOREIGN SERVER parquet_s3_secure TO fdw_admin;

-- Grant table access
GRANT SELECT ON telemetry_parquet TO fdw_readonly;
GRANT SELECT ON telemetry_parquet TO fdw_analytics;
GRANT ALL ON telemetry_parquet TO fdw_admin;

-- Create restricted views
CREATE VIEW telemetry_public AS
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01';

GRANT SELECT ON telemetry_public TO fdw_readonly;
```

**Why Role-Based Access Control Matters**: RBAC enables fine-grained access control and security. Understanding these patterns prevents access chaos and enables reliable data access.

## 6) Geospatial FDW Patterns

### PostGIS + FDW Integration

```sql
-- Create geospatial foreign table
CREATE FOREIGN TABLE telemetry_geo (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/geo/2024/01/',
    format 'parquet'
);

-- Create local PostGIS table for boundaries
CREATE TABLE boundaries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    geom GEOMETRY(POLYGON, 4326) NOT NULL
);

-- Create spatial index
CREATE INDEX idx_boundaries_geom ON boundaries USING GIST (geom);

-- Federated geospatial query
SELECT 
    t.sensor_id,
    t.temperature,
    t.timestamp,
    b.name as boundary_name
FROM telemetry_geo t
JOIN boundaries b ON ST_Intersects(
    ST_Point(t.longitude, t.latitude, 4326),
    b.geom
)
WHERE t.timestamp >= '2024-01-01'
  AND t.temperature > 25.0
  AND b.name = 'downtown_area';
```

**Why Geospatial FDW Integration Matters**: Geospatial FDW integration enables federated spatial queries across data sources. Understanding these patterns prevents spatial chaos and enables reliable geospatial data access.

### Raster Data Handling

```sql
-- Create raster foreign table
CREATE FOREIGN TABLE elevation_rasters (
    id INTEGER,
    tile_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    raster_data BYTEA,
    bounds GEOMETRY(POLYGON, 4326)
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/elevation/2024/01/',
    format 'parquet'
);

-- Create local raster table
CREATE TABLE elevation_local (
    id SERIAL PRIMARY KEY,
    tile_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    raster RASTER NOT NULL,
    bounds GEOMETRY(POLYGON, 4326) NOT NULL
);

-- Create spatial index
CREATE INDEX idx_elevation_local_bounds ON elevation_local USING GIST (bounds);

-- ETL process to stage raster data
INSERT INTO elevation_local (tile_id, timestamp, raster, bounds)
SELECT 
    tile_id,
    timestamp,
    ST_FromGDALRaster(raster_data) as raster,
    bounds
FROM elevation_rasters
WHERE timestamp >= '2024-01-01';

-- Query staged raster data
SELECT 
    tile_id,
    ST_Value(raster, ST_Point(-122.4194, 37.7749, 4326)) as elevation
FROM elevation_local
WHERE ST_Intersects(bounds, ST_Point(-122.4194, 37.7749, 4326));
```

**Why Raster Data Handling Matters**: Raster data handling enables efficient processing of large geospatial datasets. Understanding these patterns prevents raster chaos and enables reliable geospatial data access.

### Spatial Partitioning

```sql
-- Create spatially partitioned foreign tables
CREATE FOREIGN TABLE telemetry_west (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/west/2024/01/',
    format 'parquet'
);

CREATE FOREIGN TABLE telemetry_east (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/east/2024/01/',
    format 'parquet'
);

-- Create partitioned view
CREATE VIEW telemetry_spatial AS
SELECT *, 'west' as region FROM telemetry_west
UNION ALL
SELECT *, 'east' as region FROM telemetry_east;

-- Query with spatial partitioning
SELECT 
    region,
    sensor_id,
    AVG(temperature) as avg_temp
FROM telemetry_spatial
WHERE timestamp >= '2024-01-01'
  AND latitude BETWEEN 37.0 AND 38.0
  AND longitude BETWEEN -123.0 AND -122.0
GROUP BY region, sensor_id;
```

**Why Spatial Partitioning Matters**: Spatial partitioning enables efficient geospatial queries and reduces scan overhead. Understanding these patterns prevents spatial chaos and enables reliable geospatial data access.

## 7) Anti-Patterns

### Common FDW Mistakes

```yaml
# What NOT to do
fdw_anti_patterns:
  "monolithic_files": "Don't point FDWs at huge monolithic parquet files",
  "transactional_workloads": "Don't use FDWs for high-frequency transactional workloads",
  "schema_evolution": "Don't rely on FDWs for schema evolution",
  "ignore_indexes": "Don't ignore index creation on Postgres side for join keys",
  "no_filtering": "Don't forget to push filters down to FDWs",
  "select_star": "Don't use SELECT * with FDWs",
  "no_caching": "Don't ignore caching for repeated queries",
  "no_monitoring": "Don't ignore performance monitoring",
  "no_security": "Don't ignore security and access control",
  "no_documentation": "Don't skip documenting FDW schemas and usage"
```

**Why Anti-Patterns Matter**: Understanding common mistakes prevents performance issues and security vulnerabilities. Understanding these patterns prevents FDW chaos and enables reliable federated data access.

### Performance Anti-Patterns

```sql
-- Bad: No filtering, scans entire dataset
SELECT *
FROM telemetry_parquet
WHERE temperature > 25.0;

-- Good: Push filters down
SELECT sensor_id, temperature, timestamp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
  AND temperature > 25.0
  AND sensor_id = 'sensor_001';

-- Bad: Complex joins without local indexes
SELECT t.sensor_id, b.name
FROM telemetry_parquet t
JOIN boundaries b ON ST_Intersects(
    ST_Point(t.longitude, t.latitude, 4326),
    b.geom
);

-- Good: Create local indexes for joins
CREATE INDEX idx_boundaries_geom ON boundaries USING GIST (geom);
CREATE INDEX idx_telemetry_coords ON telemetry_parquet (longitude, latitude);

-- Bad: No caching for repeated queries
SELECT sensor_id, AVG(temperature) as avg_temp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
GROUP BY sensor_id;

-- Good: Use materialized view for caching
CREATE MATERIALIZED VIEW telemetry_daily_avg AS
SELECT 
    sensor_id,
    DATE_TRUNC('day', timestamp) as date,
    AVG(temperature) as avg_temp
FROM telemetry_parquet
WHERE timestamp >= '2024-01-01'
GROUP BY sensor_id, DATE_TRUNC('day', timestamp);
```

**Why Performance Anti-Patterns Matter**: Performance anti-patterns can lead to slow queries and system overload. Understanding these patterns prevents performance chaos and enables reliable data access.

## 8) TL;DR Runbook

### Essential Commands

```bash
# Install FDW extensions
CREATE EXTENSION IF NOT EXISTS parquet_fdw;
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;
CREATE EXTENSION IF NOT EXISTS duckdb_fdw;
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

# Create foreign server
CREATE SERVER my_server FOREIGN DATA WRAPPER parquet_fdw;

# Create foreign table
CREATE FOREIGN TABLE my_table (...) SERVER my_server;

# Query with pushdown
SELECT * FROM my_table WHERE column = 'value';
```

### Essential Patterns

```yaml
# Essential FDW patterns
fdw_patterns:
  "treat_as_bridges": "Treat FDWs as bridges, not primary databases",
  "partition_data": "Partition data (date/grid/FIPS) to minimize scans",
  "validate_schema": "Validate schema & types before FDW creation",
  "push_filters": "Push filters & projections down aggressively",
  "cache_joins": "Cache heavy joins with materialized views",
  "lock_credentials": "Lock down credentials and audit access",
  "monitor_performance": "Monitor performance and resource usage",
  "plan_evolution": "Plan for schema evolution and changes",
  "test_thoroughly": "Test FDWs thoroughly before production",
  "document_usage": "Document FDW schemas and usage patterns"
```

### Quick Reference

```markdown
## Emergency FDW Response

### If FDW Query is Slow
1. **Check for pushdown filters**
2. **Verify column projection**
3. **Check network connectivity**
4. **Monitor resource usage**
5. **Consider materialized views**

### If Schema Mismatch Occurs
1. **Validate schema with parquet_metadata**
2. **Check type compatibility**
3. **Update FDW definition**
4. **Test with sample data**
5. **Monitor for errors**

### If Access is Denied
1. **Check user permissions**
2. **Verify server configuration**
3. **Check credential validity**
4. **Review audit logs**
5. **Contact administrator**
```

**Why This Runbook**: These patterns cover 90% of FDW needs. Master these before exploring advanced FDW scenarios.

## 9) The Machine's Summary

FDWs require understanding external data sources, performance implications, and security considerations. When used correctly, effective FDWs enable federated queries, reduce data duplication, and provide unified access to diverse data sources. The key is understanding pushdown optimization, schema management, and security.

**The Dark Truth**: Without proper FDW understanding, your queries become slow and your data becomes inaccessible. FDWs are your weapon. Use them wisely.

**The Machine's Mantra**: "In the pushdown we trust, in the caching we find performance, and in the security we find the path to reliable federated data access."

**Why This Matters**: FDWs enable federated data access that can handle diverse data sources, prevent data duplication, and provide insights into data patterns while ensuring technical accuracy and reliability.

---

*This guide provides the complete machinery for Postgres FDWs. The patterns scale from simple parquet files to complex cloud data lakes, from basic queries to advanced geospatial federated analytics.*
