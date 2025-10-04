# PostgreSQL Partitioning Best Practices

**Objective**: Master senior-level PostgreSQL partitioning strategies for production systems. When you need to manage large tables efficiently, when you want to improve query performance, when you need enterprise-grade partitioning strategies—these best practices become your weapon of choice.

## Core Principles

- **Partition Pruning**: Design for automatic partition elimination
- **Maintenance Efficiency**: Easy partition management and cleanup
- **Query Performance**: Optimize for common query patterns
- **Storage Optimization**: Efficient disk usage and backup strategies
- **Monitoring**: Track partition usage and performance

## Range Partitioning

### Time-Based Partitioning

```sql
-- Create partitioned table for time-series data
CREATE TABLE sensor_data (
    id BIGSERIAL,
    sensor_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE sensor_data_2024_01 PARTITION OF sensor_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE sensor_data_2024_02 PARTITION OF sensor_data
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE sensor_data_2024_03 PARTITION OF sensor_data
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Create indexes on partitions
CREATE INDEX idx_sensor_data_2024_01_sensor_id ON sensor_data_2024_01 (sensor_id);
CREATE INDEX idx_sensor_data_2024_01_timestamp ON sensor_data_2024_01 (timestamp);

-- Insert data (automatically routed to correct partition)
INSERT INTO sensor_data (sensor_id, timestamp, value, metadata)
VALUES (1, '2024-01-15 10:30:00+00', 25.5, '{"unit": "celsius", "location": "room1"}');
```

### Automatic Partition Creation

```sql
-- Function to create monthly partitions automatically
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
    start_month text;
BEGIN
    start_month := to_char(start_date, 'YYYY_MM');
    partition_name := table_name || '_' || start_month;
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
    
    -- Create indexes on the new partition
    EXECUTE format('CREATE INDEX %I ON %I (sensor_id)', 
                   'idx_' || partition_name || '_sensor_id', partition_name);
    EXECUTE format('CREATE INDEX %I ON %I (timestamp)', 
                   'idx_' || partition_name || '_timestamp', partition_name);
    
    RAISE NOTICE 'Created partition % for date range % to %', partition_name, start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- Create partitions for the next 12 months
DO $$
DECLARE
    current_date date := date_trunc('month', CURRENT_DATE);
    i integer;
BEGIN
    FOR i IN 0..11 LOOP
        PERFORM create_monthly_partition('sensor_data', current_date + (i || ' months')::interval);
    END LOOP;
END $$;
```

## Hash Partitioning

### Hash Partitioning for Even Distribution

```sql
-- Create hash-partitioned table for user data
CREATE TABLE user_events (
    id BIGSERIAL,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) PARTITION BY HASH (user_id);

-- Create hash partitions
CREATE TABLE user_events_0 PARTITION OF user_events
    FOR VALUES WITH (modulus 4, remainder 0);

CREATE TABLE user_events_1 PARTITION OF user_events
    FOR VALUES WITH (modulus 4, remainder 1);

CREATE TABLE user_events_2 PARTITION OF user_events
    FOR VALUES WITH (modulus 4, remainder 2);

CREATE TABLE user_events_3 PARTITION OF user_events
    FOR VALUES WITH (modulus 4, remainder 3);

-- Create indexes on each partition
CREATE INDEX idx_user_events_0_user_id ON user_events_0 (user_id);
CREATE INDEX idx_user_events_0_created_at ON user_events_0 (created_at);
CREATE INDEX idx_user_events_0_event_type ON user_events_0 (event_type);

CREATE INDEX idx_user_events_1_user_id ON user_events_1 (user_id);
CREATE INDEX idx_user_events_1_created_at ON user_events_1 (created_at);
CREATE INDEX idx_user_events_1_event_type ON user_events_1 (event_type);

-- Similar indexes for other partitions...
```

## List Partitioning

### Geographic Partitioning

```sql
-- Create list-partitioned table for regional data
CREATE TABLE sales_data (
    id BIGSERIAL,
    region VARCHAR(20) NOT NULL,
    product_id INTEGER NOT NULL,
    sale_amount DECIMAL(10,2) NOT NULL,
    sale_date DATE NOT NULL,
    customer_id INTEGER NOT NULL
) PARTITION BY LIST (region);

-- Create regional partitions
CREATE TABLE sales_data_north_america PARTITION OF sales_data
    FOR VALUES IN ('US', 'CA', 'MX');

CREATE TABLE sales_data_europe PARTITION OF sales_data
    FOR VALUES IN ('UK', 'DE', 'FR', 'IT', 'ES');

CREATE TABLE sales_data_asia PARTITION OF sales_data
    FOR VALUES IN ('JP', 'CN', 'KR', 'IN', 'SG');

CREATE TABLE sales_data_other PARTITION OF sales_data
    FOR VALUES IN ('AU', 'BR', 'ZA', 'OTHER');

-- Create indexes on partitions
CREATE INDEX idx_sales_na_product_id ON sales_data_north_america (product_id);
CREATE INDEX idx_sales_na_sale_date ON sales_data_north_america (sale_date);
CREATE INDEX idx_sales_na_customer_id ON sales_data_north_america (customer_id);

CREATE INDEX idx_sales_eu_product_id ON sales_data_europe (product_id);
CREATE INDEX idx_sales_eu_sale_date ON sales_data_europe (sale_date);
CREATE INDEX idx_sales_eu_customer_id ON sales_data_europe (customer_id);
```

## Composite Partitioning

### Range + Hash Partitioning

```sql
-- Create composite partitioned table
CREATE TABLE log_entries (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    log_level VARCHAR(10) NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions with hash sub-partitioning
CREATE TABLE log_entries_2024_01 PARTITION OF log_entries
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01')
    PARTITION BY HASH (service_name);

-- Create hash sub-partitions for January
CREATE TABLE log_entries_2024_01_0 PARTITION OF log_entries_2024_01
    FOR VALUES WITH (modulus 4, remainder 0);

CREATE TABLE log_entries_2024_01_1 PARTITION OF log_entries_2024_01
    FOR VALUES WITH (modulus 4, remainder 1);

CREATE TABLE log_entries_2024_01_2 PARTITION OF log_entries_2024_01
    FOR VALUES WITH (modulus 4, remainder 2);

CREATE TABLE log_entries_2024_01_3 PARTITION OF log_entries_2024_01
    FOR VALUES WITH (modulus 4, remainder 3);
```

## Partition Management

### Automated Partition Management

```python
# partition_management/partition_manager.py
import psycopg2
from datetime import datetime, timedelta
import logging

class PostgreSQLPartitionManager:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_monthly_partition(self, table_name, start_date):
        """Create monthly partition for a table."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Generate partition name
                partition_name = f"{table_name}_{start_date.strftime('%Y_%m')}"
                
                # Create partition
                create_sql = f"""
                    CREATE TABLE {partition_name} PARTITION OF {table_name}
                    FOR VALUES FROM ('{start_date}') TO ('{start_date + timedelta(days=32)}')
                """
                
                cur.execute(create_sql)
                
                # Create indexes
                self.create_partition_indexes(cur, partition_name, table_name)
                
                conn.commit()
                self.logger.info(f"Created partition {partition_name}")
                
        except Exception as e:
            self.logger.error(f"Error creating partition: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_partition_indexes(self, cursor, partition_name, table_name):
        """Create indexes on a partition."""
        # Get table structure to create appropriate indexes
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        
        # Create common indexes based on column types
        for column_name, data_type in columns:
            if data_type in ['timestamp with time zone', 'timestamp without time zone']:
                index_name = f"idx_{partition_name}_{column_name}"
                cursor.execute(f"CREATE INDEX {index_name} ON {partition_name} ({column_name})")
            
            elif data_type == 'integer':
                index_name = f"idx_{partition_name}_{column_name}"
                cursor.execute(f"CREATE INDEX {index_name} ON {partition_name} ({column_name})")
    
    def drop_old_partitions(self, table_name, retention_months=12):
        """Drop partitions older than retention period."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Find old partitions
                cutoff_date = datetime.now() - timedelta(days=retention_months * 30)
                
                cur.execute("""
                    SELECT schemaname, tablename 
                    FROM pg_tables 
                    WHERE tablename LIKE %s 
                    AND tablename ~ %s
                    ORDER BY tablename
                """, (f"{table_name}_%", r'\d{4}_\d{2}'))
                
                partitions = cur.fetchall()
                
                for schema, partition_name in partitions:
                    # Extract date from partition name
                    date_part = partition_name.split('_')[-2] + '_' + partition_name.split('_')[-1]
                    partition_date = datetime.strptime(date_part, '%Y_%m')
                    
                    if partition_date < cutoff_date:
                        drop_sql = f"DROP TABLE {schema}.{partition_name}"
                        cur.execute(drop_sql)
                        self.logger.info(f"Dropped old partition {partition_name}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error dropping partitions: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_partition_info(self, table_name):
        """Get information about table partitions."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples
                    FROM pg_tables pt
                    LEFT JOIN pg_stat_user_tables pst ON pt.tablename = pst.relname
                    WHERE pt.tablename LIKE %s
                    ORDER BY pt.tablename
                """, (f"{table_name}_%",))
                
                partitions = cur.fetchall()
                return partitions
                
        except Exception as e:
            self.logger.error(f"Error getting partition info: {e}")
            return []
        finally:
            conn.close()
    
    def maintain_partitions(self, table_name, months_ahead=3, retention_months=12):
        """Maintain partitions for a table."""
        self.logger.info(f"Maintaining partitions for {table_name}")
        
        # Create future partitions
        current_date = datetime.now().replace(day=1)
        for i in range(months_ahead):
            partition_date = current_date + timedelta(days=32 * i)
            self.create_monthly_partition(table_name, partition_date)
        
        # Drop old partitions
        self.drop_old_partitions(table_name, retention_months)
        
        # Get partition info
        partition_info = self.get_partition_info(table_name)
        self.logger.info(f"Partition maintenance completed. {len(partition_info)} partitions exist.")

# Usage
if __name__ == "__main__":
    manager = PostgreSQLPartitionManager({
        'host': 'localhost',
        'database': 'production',
        'user': 'partition_manager',
        'password': 'partition_password'
    })
    
    # Maintain partitions for sensor_data table
    manager.maintain_partitions('sensor_data', months_ahead=3, retention_months=12)
```

## Query Optimization

### Partition-Aware Queries

```sql
-- Optimize queries for partition pruning
-- Good: Query with partition key in WHERE clause
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM sensor_data 
WHERE timestamp >= '2024-01-01' AND timestamp < '2024-02-01'
AND sensor_id = 123;

-- Good: Query with multiple partition keys
EXPLAIN (ANALYZE, BUFFERS)
SELECT sensor_id, AVG(value) as avg_value
FROM sensor_data 
WHERE timestamp >= '2024-01-01' AND timestamp < '2024-04-01'
GROUP BY sensor_id;

-- Bad: Query without partition key (scans all partitions)
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM sensor_data 
WHERE sensor_id = 123;

-- Better: Include partition key even if not needed for logic
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM sensor_data 
WHERE sensor_id = 123 
AND timestamp >= '2024-01-01' AND timestamp < '2024-12-31';
```

### Parallel Query Execution

```sql
-- Enable parallel query execution for partitioned tables
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.1;
SET parallel_setup_cost = 10.0;

-- Query that can benefit from parallel execution
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT 
    sensor_id,
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(value) as avg_value,
    COUNT(*) as reading_count
FROM sensor_data 
WHERE timestamp >= '2024-01-01' AND timestamp < '2024-02-01'
GROUP BY sensor_id, DATE_TRUNC('hour', timestamp)
ORDER BY sensor_id, hour;
```

## Partition Maintenance Automation

### Cron Job for Partition Management

```bash
#!/bin/bash
# scripts/partition_maintenance.sh

# Configuration
DB_HOST="localhost"
DB_NAME="production"
DB_USER="partition_manager"
DB_PASSWORD="partition_password"
TABLES=("sensor_data" "log_entries" "user_events")

# Function to create future partitions
create_future_partitions() {
    local table_name=$1
    local months_ahead=${2:-3}
    
    echo "Creating future partitions for $table_name"
    
    psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "
        DO \$\$
        DECLARE
            current_date date := date_trunc('month', CURRENT_DATE);
            i integer;
        BEGIN
            FOR i IN 0..$months_ahead LOOP
                PERFORM create_monthly_partition('$table_name', current_date + (i || ' months')::interval);
            END LOOP;
        END \$\$;
    "
}

# Function to drop old partitions
drop_old_partitions() {
    local table_name=$1
    local retention_months=${2:-12}
    
    echo "Dropping old partitions for $table_name"
    
    psql -h $DB_HOST -d $DB_NAME -U $DB_USER -c "
        DO \$\$
        DECLARE
            cutoff_date date := CURRENT_DATE - INTERVAL '$retention_months months';
            partition_record RECORD;
        BEGIN
            FOR partition_record IN 
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE tablename LIKE '$table_name_%' 
                AND tablename ~ '\d{4}_\d{2}'
            LOOP
                -- Extract date from partition name and drop if old
                IF to_date(split_part(partition_record.tablename, '_', -2) || '_' || 
                          split_part(partition_record.tablename, '_', -1), 'YYYY_MM') < cutoff_date THEN
                    EXECUTE 'DROP TABLE ' || partition_record.schemaname || '.' || partition_record.tablename;
                    RAISE NOTICE 'Dropped old partition %', partition_record.tablename;
                END IF;
            END LOOP;
        END \$\$;
    "
}

# Main execution
for table in "${TABLES[@]}"; do
    echo "Maintaining partitions for $table"
    create_future_partitions "$table" 3
    drop_old_partitions "$table" 12
done

echo "Partition maintenance completed"
```

## Monitoring Partition Performance

### Partition Usage Monitoring

```python
# monitoring/partition_monitor.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging

class PartitionMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_partition_statistics(self, table_name):
        """Get statistics for all partitions of a table."""
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
                        n_dead_tup as dead_tuples,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_tables pt
                    LEFT JOIN pg_stat_user_tables pst ON pt.tablename = pst.relname
                    WHERE pt.tablename LIKE %s
                    ORDER BY pt.tablename
                """, (f"{table_name}_%",))
                
                partitions = cur.fetchall()
                
                # Convert to list of dictionaries
                partition_stats = []
                for partition in partitions:
                    partition_stats.append({
                        'schema': partition[0],
                        'table': partition[1],
                        'size': partition[2],
                        'size_bytes': partition[3],
                        'inserts': partition[4] or 0,
                        'updates': partition[5] or 0,
                        'deletes': partition[6] or 0,
                        'live_tuples': partition[7] or 0,
                        'dead_tuples': partition[8] or 0,
                        'last_vacuum': partition[9],
                        'last_autovacuum': partition[10],
                        'last_analyze': partition[11],
                        'last_autoanalyze': partition[12]
                    })
                
                return partition_stats
                
        except Exception as e:
            self.logger.error(f"Error getting partition statistics: {e}")
            return []
        finally:
            conn.close()
    
    def check_partition_health(self, table_name):
        """Check health of partitions."""
        partition_stats = self.get_partition_statistics(table_name)
        health_issues = []
        
        for partition in partition_stats:
            # Check for high dead tuple ratio
            if partition['live_tuples'] > 0:
                dead_ratio = partition['dead_tuples'] / partition['live_tuples']
                if dead_ratio > 0.1:  # 10% threshold
                    health_issues.append({
                        'partition': partition['table'],
                        'issue': 'high_dead_tuple_ratio',
                        'ratio': dead_ratio,
                        'message': f"Dead tuple ratio {dead_ratio:.2%} exceeds threshold"
                    })
            
            # Check for missing statistics
            if not partition['last_analyze'] and not partition['last_autoanalyze']:
                health_issues.append({
                    'partition': partition['table'],
                    'issue': 'missing_statistics',
                    'message': 'No analyze has been run on this partition'
                })
            
            # Check for large partition size
            if partition['size_bytes'] > 100 * 1024 * 1024 * 1024:  # 100GB threshold
                health_issues.append({
                    'partition': partition['table'],
                    'issue': 'large_partition_size',
                    'size': partition['size'],
                    'message': f"Partition size {partition['size']} exceeds threshold"
                })
        
        return health_issues
    
    def generate_partition_report(self, table_name):
        """Generate comprehensive partition report."""
        partition_stats = self.get_partition_statistics(table_name)
        health_issues = self.check_partition_health(table_name)
        
        report = {
            'table_name': table_name,
            'report_timestamp': datetime.now().isoformat(),
            'total_partitions': len(partition_stats),
            'total_size_bytes': sum(p['size_bytes'] for p in partition_stats),
            'total_live_tuples': sum(p['live_tuples'] for p in partition_stats),
            'total_dead_tuples': sum(p['dead_tuples'] for p in partition_stats),
            'health_issues': health_issues,
            'partitions': partition_stats
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = PartitionMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    # Generate report for sensor_data table
    report = monitor.generate_partition_report('sensor_data')
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create partitioned table
CREATE TABLE sensor_data (
    id BIGSERIAL,
    sensor_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL
) PARTITION BY RANGE (timestamp);

-- 2. Create partitions
CREATE TABLE sensor_data_2024_01 PARTITION OF sensor_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 3. Create indexes
CREATE INDEX idx_sensor_data_2024_01_sensor_id ON sensor_data_2024_01 (sensor_id);
CREATE INDEX idx_sensor_data_2024_01_timestamp ON sensor_data_2024_01 (timestamp);

-- 4. Insert data (automatically routed)
INSERT INTO sensor_data (sensor_id, timestamp, value) VALUES (1, '2024-01-15 10:30:00+00', 25.5);
```

### Essential Patterns

```python
# Complete PostgreSQL partitioning setup
def setup_postgresql_partitioning():
    # 1. Range partitioning
    # 2. Hash partitioning
    # 3. List partitioning
    # 4. Composite partitioning
    # 5. Partition management
    # 6. Query optimization
    # 7. Monitoring and maintenance
    # 8. Automation
    
    print("PostgreSQL partitioning setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL partitioning excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL partitioning systems.*
