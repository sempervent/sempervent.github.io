# PostgreSQL Time Series Data Best Practices

**Objective**: Master senior-level PostgreSQL time series data patterns for production systems. When you need to handle time series data, when you want to optimize time-based queries, when you need enterprise-grade time series strategies—these best practices become your weapon of choice.

## Core Principles

- **Partitioning**: Use time-based partitioning for large datasets
- **Indexing**: Create appropriate indexes for time series queries
- **Compression**: Implement data compression for storage efficiency
- **Aggregation**: Use materialized views for pre-computed aggregations
- **Retention**: Implement data retention policies

## Time Series Data Design

### Time Series Table Design

```sql
-- Create time series tables with appropriate design
CREATE TABLE sensor_readings (
    id BIGSERIAL,
    sensor_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20) NOT NULL,
    quality_score NUMERIC(3,2) DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE sensor_readings_2024_01 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE sensor_readings_2024_02 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE sensor_readings_2024_03 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Create indexes on partitions
CREATE INDEX idx_sensor_readings_2024_01_sensor_timestamp 
    ON sensor_readings_2024_01 (sensor_id, timestamp);
CREATE INDEX idx_sensor_readings_2024_01_timestamp 
    ON sensor_readings_2024_01 (timestamp);

-- Create BRIN indexes for time series data
CREATE INDEX idx_sensor_readings_2024_01_timestamp_brin 
    ON sensor_readings_2024_01 USING BRIN (timestamp);
CREATE INDEX idx_sensor_readings_2024_01_sensor_brin 
    ON sensor_readings_2024_01 USING BRIN (sensor_id);
```

### Time Series Data Types

```sql
-- Create tables for different time series use cases
CREATE TABLE financial_data (
    id BIGSERIAL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price NUMERIC(10,4) NOT NULL,
    high_price NUMERIC(10,4) NOT NULL,
    low_price NUMERIC(10,4) NOT NULL,
    close_price NUMERIC(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close NUMERIC(10,4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

CREATE TABLE user_events (
    id BIGSERIAL,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    session_id VARCHAR(100),
    properties JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

CREATE TABLE system_metrics (
    id BIGSERIAL,
    hostname VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);
```

## Time Series Partitioning

### Automatic Partitioning

```sql
-- Create function for automatic partition creation
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name TEXT,
    start_date DATE
)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
    
    -- Create indexes on new partition
    EXECUTE format('CREATE INDEX %I ON %I (timestamp)', 
                   'idx_' || partition_name || '_timestamp', partition_name);
    EXECUTE format('CREATE INDEX %I ON %I USING BRIN (timestamp)', 
                   'idx_' || partition_name || '_timestamp_brin', partition_name);
END;
$$ LANGUAGE plpgsql;

-- Create function to create partitions for the next 12 months
CREATE OR REPLACE FUNCTION create_future_partitions(table_name TEXT)
RETURNS VOID AS $$
DECLARE
    current_date DATE := CURRENT_DATE;
    i INTEGER;
BEGIN
    FOR i IN 0..11 LOOP
        PERFORM create_monthly_partition(table_name, current_date + (i || ' months')::interval);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create future partitions
SELECT create_future_partitions('sensor_readings');
SELECT create_future_partitions('financial_data');
SELECT create_future_partitions('user_events');
SELECT create_future_partitions('system_metrics');
```

### Partition Management

```sql
-- Create function to drop old partitions
CREATE OR REPLACE FUNCTION drop_old_partitions(
    table_name TEXT,
    retention_months INTEGER DEFAULT 12
)
RETURNS INTEGER AS $$
DECLARE
    partition_name TEXT;
    partition_date DATE;
    dropped_count INTEGER := 0;
BEGIN
    FOR partition_name IN 
        SELECT schemaname||'.'||tablename 
        FROM pg_tables 
        WHERE tablename LIKE table_name || '_%'
        AND schemaname = 'public'
    LOOP
        -- Extract date from partition name
        partition_date := to_date(
            regexp_replace(partition_name, '.*_(\d{4}_\d{2})', '\1'), 
            'YYYY_MM'
        );
        
        -- Drop partition if older than retention period
        IF partition_date < CURRENT_DATE - (retention_months || ' months')::interval THEN
            EXECUTE 'DROP TABLE ' || partition_name;
            dropped_count := dropped_count + 1;
        END IF;
    END LOOP;
    
    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to archive old partitions
CREATE OR REPLACE FUNCTION archive_old_partitions(
    table_name TEXT,
    archive_table_name TEXT,
    retention_months INTEGER DEFAULT 6
)
RETURNS INTEGER AS $$
DECLARE
    partition_name TEXT;
    partition_date DATE;
    archived_count INTEGER := 0;
BEGIN
    FOR partition_name IN 
        SELECT schemaname||'.'||tablename 
        FROM pg_tables 
        WHERE tablename LIKE table_name || '_%'
        AND schemaname = 'public'
    LOOP
        -- Extract date from partition name
        partition_date := to_date(
            regexp_replace(partition_name, '.*_(\d{4}_\d{2})', '\1'), 
            'YYYY_MM'
        );
        
        -- Archive partition if older than retention period
        IF partition_date < CURRENT_DATE - (retention_months || ' months')::interval THEN
            EXECUTE format('INSERT INTO %I SELECT * FROM %I', archive_table_name, partition_name);
            EXECUTE 'DROP TABLE ' || partition_name;
            archived_count := archived_count + 1;
        END IF;
    END LOOP;
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;
```

## Time Series Indexing

### Time Series Index Strategies

```sql
-- Create comprehensive indexes for time series data
CREATE INDEX idx_sensor_readings_sensor_timestamp 
    ON sensor_readings (sensor_id, timestamp DESC);

CREATE INDEX idx_sensor_readings_timestamp_value 
    ON sensor_readings (timestamp DESC, value);

CREATE INDEX idx_sensor_readings_sensor_timestamp_value 
    ON sensor_readings (sensor_id, timestamp DESC, value);

-- Create partial indexes for active data
CREATE INDEX idx_sensor_readings_active 
    ON sensor_readings (sensor_id, timestamp DESC) 
    WHERE timestamp > CURRENT_DATE - INTERVAL '30 days';

-- Create indexes for specific time ranges
CREATE INDEX idx_sensor_readings_recent 
    ON sensor_readings (timestamp DESC) 
    WHERE timestamp > CURRENT_DATE - INTERVAL '7 days';

-- Create indexes for aggregation queries
CREATE INDEX idx_sensor_readings_hourly_agg 
    ON sensor_readings (sensor_id, date_trunc('hour', timestamp));
```

### BRIN Indexes for Time Series

```sql
-- Create BRIN indexes for time series data
CREATE INDEX idx_sensor_readings_timestamp_brin 
    ON sensor_readings USING BRIN (timestamp);

CREATE INDEX idx_sensor_readings_sensor_brin 
    ON sensor_readings USING BRIN (sensor_id);

CREATE INDEX idx_sensor_readings_value_brin 
    ON sensor_readings USING BRIN (value);

-- Create BRIN indexes on partitions
CREATE INDEX idx_sensor_readings_2024_01_timestamp_brin 
    ON sensor_readings_2024_01 USING BRIN (timestamp);
CREATE INDEX idx_sensor_readings_2024_01_sensor_brin 
    ON sensor_readings_2024_01 USING BRIN (sensor_id);
```

## Time Series Aggregation

### Materialized Views for Aggregations

```sql
-- Create materialized views for time series aggregations
CREATE MATERIALIZED VIEW sensor_readings_hourly AS
SELECT 
    sensor_id,
    date_trunc('hour', timestamp) as hour,
    COUNT(*) as reading_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    STDDEV(value) as stddev_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95_value
FROM sensor_readings
WHERE timestamp > CURRENT_DATE - INTERVAL '30 days'
GROUP BY sensor_id, date_trunc('hour', timestamp);

-- Create index on materialized view
CREATE INDEX idx_sensor_readings_hourly_sensor_hour 
    ON sensor_readings_hourly (sensor_id, hour);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_sensor_aggregations()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY sensor_readings_hourly;
END;
$$ LANGUAGE plpgsql;
```

### Time Series Functions

```sql
-- Create function for time series aggregation
CREATE OR REPLACE FUNCTION get_sensor_statistics(
    p_sensor_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_interval INTERVAL DEFAULT '1 hour'
)
RETURNS TABLE (
    time_bucket TIMESTAMPTZ,
    reading_count BIGINT,
    avg_value DOUBLE PRECISION,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        date_trunc('hour', timestamp) as time_bucket,
        COUNT(*) as reading_count,
        AVG(value) as avg_value,
        MIN(value) as min_value,
        MAX(value) as max_value,
        STDDEV(value) as stddev_value
    FROM sensor_readings
    WHERE sensor_id = p_sensor_id
    AND timestamp BETWEEN p_start_time AND p_end_time
    GROUP BY date_trunc('hour', timestamp)
    ORDER BY time_bucket;
END;
$$ LANGUAGE plpgsql;

-- Create function for time series interpolation
CREATE OR REPLACE FUNCTION interpolate_sensor_values(
    p_sensor_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_interval INTERVAL DEFAULT '1 minute'
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    interpolated_value DOUBLE PRECISION
) AS $$
DECLARE
    current_time TIMESTAMPTZ;
    prev_value DOUBLE PRECISION;
    next_value DOUBLE PRECISION;
    prev_time TIMESTAMPTZ;
    next_time TIMESTAMPTZ;
    interpolation_factor DOUBLE PRECISION;
BEGIN
    current_time := p_start_time;
    
    WHILE current_time <= p_end_time LOOP
        -- Find previous and next values
        SELECT value, timestamp INTO prev_value, prev_time
        FROM sensor_readings
        WHERE sensor_id = p_sensor_id
        AND timestamp <= current_time
        ORDER BY timestamp DESC
        LIMIT 1;
        
        SELECT value, timestamp INTO next_value, next_time
        FROM sensor_readings
        WHERE sensor_id = p_sensor_id
        AND timestamp >= current_time
        ORDER BY timestamp ASC
        LIMIT 1;
        
        -- Interpolate value
        IF prev_value IS NOT NULL AND next_value IS NOT NULL THEN
            interpolation_factor := EXTRACT(EPOCH FROM (current_time - prev_time)) / 
                                  EXTRACT(EPOCH FROM (next_time - prev_time));
            interpolated_value := prev_value + (next_value - prev_value) * interpolation_factor;
        ELSIF prev_value IS NOT NULL THEN
            interpolated_value := prev_value;
        ELSIF next_value IS NOT NULL THEN
            interpolated_value := next_value;
        ELSE
            interpolated_value := NULL;
        END IF;
        
        RETURN NEXT;
        current_time := current_time + p_interval;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

## Time Series Compression

### Data Compression Strategies

```sql
-- Create function for time series compression
CREATE OR REPLACE FUNCTION compress_sensor_data(
    p_sensor_id INTEGER,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_compression_ratio NUMERIC DEFAULT 0.1
)
RETURNS TABLE (
    compressed_timestamp TIMESTAMPTZ,
    compressed_value DOUBLE PRECISION,
    compression_method TEXT
) AS $$
DECLARE
    total_points INTEGER;
    target_points INTEGER;
    step_size INTEGER;
    current_time TIMESTAMPTZ;
    current_value DOUBLE PRECISION;
BEGIN
    -- Calculate compression parameters
    SELECT COUNT(*) INTO total_points
    FROM sensor_readings
    WHERE sensor_id = p_sensor_id
    AND timestamp BETWEEN p_start_time AND p_end_time;
    
    target_points := GREATEST(1, FLOOR(total_points * p_compression_ratio));
    step_size := GREATEST(1, FLOOR(total_points / target_points));
    
    -- Return compressed data
    RETURN QUERY
    SELECT 
        timestamp as compressed_timestamp,
        value as compressed_value,
        'uniform_sampling'::TEXT as compression_method
    FROM (
        SELECT timestamp, value,
               ROW_NUMBER() OVER (ORDER BY timestamp) as rn
        FROM sensor_readings
        WHERE sensor_id = p_sensor_id
        AND timestamp BETWEEN p_start_time AND p_end_time
    ) t
    WHERE rn % step_size = 1
    ORDER BY timestamp;
END;
$$ LANGUAGE plpgsql;
```

### Data Retention Policies

```sql
-- Create function for data retention
CREATE OR REPLACE FUNCTION apply_retention_policy(
    table_name TEXT,
    retention_days INTEGER DEFAULT 365
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (retention_days || ' days')::interval;
    
    EXECUTE format('DELETE FROM %I WHERE timestamp < %L', table_name, cutoff_date);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for data archiving
CREATE OR REPLACE FUNCTION archive_old_data(
    source_table TEXT,
    archive_table TEXT,
    retention_days INTEGER DEFAULT 365
)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
    cutoff_date TIMESTAMPTZ;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (retention_days || ' days')::interval;
    
    EXECUTE format('INSERT INTO %I SELECT * FROM %I WHERE timestamp < %L', 
                   archive_table, source_table, cutoff_date);
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    EXECUTE format('DELETE FROM %I WHERE timestamp < %L', source_table, cutoff_date);
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;
```

## Time Series Monitoring

### Time Series Performance Monitoring

```python
# monitoring/timeseries_monitor.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging

class TimeSeriesMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_partition_statistics(self):
        """Get partition statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples
                    FROM pg_stat_user_tables
                    WHERE tablename LIKE '%_2024_%'
                    ORDER BY size_bytes DESC
                """)
                
                partition_stats = cur.fetchall()
                return partition_stats
                
        except Exception as e:
            self.logger.error(f"Error getting partition statistics: {e}")
            return []
        finally:
            conn.close()
    
    def get_time_series_performance(self):
        """Get time series query performance."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Test time series query performance
                start_time = datetime.now()
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM sensor_readings 
                    WHERE timestamp > CURRENT_DATE - INTERVAL '1 day'
                """)
                daily_count_time = (datetime.now() - start_time).total_seconds()
                
                start_time = datetime.now()
                cur.execute("""
                    SELECT sensor_id, AVG(value) 
                    FROM sensor_readings 
                    WHERE timestamp > CURRENT_DATE - INTERVAL '1 hour'
                    GROUP BY sensor_id
                """)
                hourly_agg_time = (datetime.now() - start_time).total_seconds()
                
                start_time = datetime.now()
                cur.execute("""
                    SELECT * 
                    FROM sensor_readings 
                    WHERE sensor_id = 1 
                    AND timestamp > CURRENT_DATE - INTERVAL '1 day'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)
                recent_data_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'daily_count_time': daily_count_time,
                    'hourly_agg_time': hourly_agg_time,
                    'recent_data_time': recent_data_time
                }
                
        except Exception as e:
            self.logger.error(f"Error getting time series performance: {e}")
            return {}
        finally:
            conn.close()
    
    def get_data_retention_status(self):
        """Get data retention status."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        'sensor_readings' as table_name,
                        MIN(timestamp) as oldest_data,
                        MAX(timestamp) as newest_data,
                        COUNT(*) as total_records,
                        COUNT(*) FILTER (WHERE timestamp > CURRENT_DATE - INTERVAL '30 days') as recent_records
                    FROM sensor_readings
                    UNION ALL
                    SELECT 
                        'financial_data' as table_name,
                        MIN(timestamp) as oldest_data,
                        MAX(timestamp) as newest_data,
                        COUNT(*) as total_records,
                        COUNT(*) FILTER (WHERE timestamp > CURRENT_DATE - INTERVAL '30 days') as recent_records
                    FROM financial_data
                """)
                
                retention_status = cur.fetchall()
                return retention_status
                
        except Exception as e:
            self.logger.error(f"Error getting data retention status: {e}")
            return []
        finally:
            conn.close()
    
    def generate_timeseries_report(self):
        """Generate comprehensive time series report."""
        partition_stats = self.get_partition_statistics()
        performance_metrics = self.get_time_series_performance()
        retention_status = self.get_data_retention_status()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'partition_statistics': partition_stats,
            'performance_metrics': performance_metrics,
            'retention_status': retention_status
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = TimeSeriesMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_timeseries_report()
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create time series table with partitioning
CREATE TABLE sensor_readings (
    id BIGSERIAL,
    sensor_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- 2. Create partitions
CREATE TABLE sensor_readings_2024_01 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 3. Create indexes
CREATE INDEX idx_sensor_readings_sensor_timestamp 
    ON sensor_readings (sensor_id, timestamp DESC);
CREATE INDEX idx_sensor_readings_timestamp_brin 
    ON sensor_readings USING BRIN (timestamp);

-- 4. Insert data
INSERT INTO sensor_readings (sensor_id, timestamp, value) VALUES
    (1, CURRENT_TIMESTAMP, 25.5);

-- 5. Query time series data
SELECT sensor_id, AVG(value) as avg_value
FROM sensor_readings
WHERE timestamp > CURRENT_DATE - INTERVAL '1 day'
GROUP BY sensor_id;
```

### Essential Patterns

```python
# Complete PostgreSQL time series setup
def setup_postgresql_timeseries():
    # 1. Time series table design
    # 2. Partitioning strategies
    # 3. Indexing strategies
    # 4. Aggregation patterns
    # 5. Compression strategies
    # 6. Retention policies
    # 7. Performance monitoring
    # 8. Data archiving
    
    print("PostgreSQL time series setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL time series data excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL time series systems.*
