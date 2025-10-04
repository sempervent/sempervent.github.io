# PostgreSQL Maintenance & Vacuum Best Practices

**Objective**: Master senior-level PostgreSQL maintenance and vacuum patterns for production systems. When you need to maintain database health, when you want to optimize vacuum operations, when you need enterprise-grade maintenance strategies—these best practices become your weapon of choice.

## Core Principles

- **Automated Maintenance**: Implement automated maintenance procedures
- **Vacuum Optimization**: Tune vacuum operations for performance
- **Statistics Updates**: Keep table statistics current
- **Index Maintenance**: Maintain index health and performance
- **Monitoring**: Monitor maintenance operations and their impact

## Vacuum Operations

### Vacuum Configuration

```sql
-- Create vacuum configuration table
CREATE TABLE vacuum_config (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    vacuum_enabled BOOLEAN DEFAULT TRUE,
    autovacuum_enabled BOOLEAN DEFAULT TRUE,
    vacuum_scale_factor NUMERIC DEFAULT 0.2,
    vacuum_threshold INTEGER DEFAULT 50,
    analyze_scale_factor NUMERIC DEFAULT 0.1,
    analyze_threshold INTEGER DEFAULT 50,
    freeze_min_age INTEGER DEFAULT 200000000,
    freeze_max_age INTEGER DEFAULT 2000000000,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to configure table vacuum settings
CREATE OR REPLACE FUNCTION configure_table_vacuum(
    p_table_name VARCHAR(100),
    p_vacuum_enabled BOOLEAN DEFAULT TRUE,
    p_autovacuum_enabled BOOLEAN DEFAULT TRUE,
    p_vacuum_scale_factor NUMERIC DEFAULT 0.2,
    p_vacuum_threshold INTEGER DEFAULT 50
)
RETURNS VOID AS $$
BEGIN
    -- Insert or update vacuum configuration
    INSERT INTO vacuum_config (
        table_name, vacuum_enabled, autovacuum_enabled, 
        vacuum_scale_factor, vacuum_threshold
    ) VALUES (
        p_table_name, p_vacuum_enabled, p_autovacuum_enabled,
        p_vacuum_scale_factor, p_vacuum_threshold
    ) ON CONFLICT (table_name) 
    DO UPDATE SET 
        vacuum_enabled = EXCLUDED.vacuum_enabled,
        autovacuum_enabled = EXCLUDED.autovacuum_enabled,
        vacuum_scale_factor = EXCLUDED.vacuum_scale_factor,
        vacuum_threshold = EXCLUDED.vacuum_threshold,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Apply configuration to table
    EXECUTE format('ALTER TABLE %I SET (autovacuum_enabled = %s)', 
                   p_table_name, p_autovacuum_enabled);
    EXECUTE format('ALTER TABLE %I SET (autovacuum_vacuum_scale_factor = %s)', 
                   p_table_name, p_vacuum_scale_factor);
    EXECUTE format('ALTER TABLE %I SET (autovacuum_vacuum_threshold = %s)', 
                   p_table_name, p_vacuum_threshold);
END;
$$ LANGUAGE plpgsql;
```

### Vacuum Monitoring

```sql
-- Create vacuum monitoring table
CREATE TABLE vacuum_history (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    vacuum_type VARCHAR(20) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    pages_removed INTEGER,
    pages_remain INTEGER,
    pages_new INTEGER,
    pages_deleted INTEGER,
    pages_vacuumed INTEGER,
    pages_frozen INTEGER,
    tuples_deleted INTEGER,
    tuples_updated INTEGER,
    tuples_moved INTEGER,
    status VARCHAR(20) DEFAULT 'running',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to start vacuum monitoring
CREATE OR REPLACE FUNCTION start_vacuum_monitoring(
    p_table_name VARCHAR(100),
    p_vacuum_type VARCHAR(20) DEFAULT 'manual'
)
RETURNS INTEGER AS $$
DECLARE
    vacuum_id INTEGER;
BEGIN
    INSERT INTO vacuum_history (
        table_name, vacuum_type, started_at
    ) VALUES (
        p_table_name, p_vacuum_type, CURRENT_TIMESTAMP
    ) RETURNING id INTO vacuum_id;
    
    RETURN vacuum_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to complete vacuum monitoring
CREATE OR REPLACE FUNCTION complete_vacuum_monitoring(
    p_vacuum_id INTEGER,
    p_pages_removed INTEGER DEFAULT 0,
    p_pages_remain INTEGER DEFAULT 0,
    p_pages_new INTEGER DEFAULT 0,
    p_pages_deleted INTEGER DEFAULT 0,
    p_pages_vacuumed INTEGER DEFAULT 0,
    p_pages_frozen INTEGER DEFAULT 0,
    p_tuples_deleted INTEGER DEFAULT 0,
    p_tuples_updated INTEGER DEFAULT 0,
    p_tuples_moved INTEGER DEFAULT 0
)
RETURNS VOID AS $$
DECLARE
    start_time TIMESTAMPTZ;
    duration_seconds INTEGER;
BEGIN
    -- Get start time
    SELECT started_at INTO start_time
    FROM vacuum_history
    WHERE id = p_vacuum_id;
    
    -- Calculate duration
    duration_seconds := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time));
    
    -- Update vacuum record
    UPDATE vacuum_history
    SET completed_at = CURRENT_TIMESTAMP,
        duration_seconds = duration_seconds,
        pages_removed = p_pages_removed,
        pages_remain = p_pages_remain,
        pages_new = p_pages_new,
        pages_deleted = p_pages_deleted,
        pages_vacuumed = p_pages_vacuumed,
        pages_frozen = p_pages_frozen,
        tuples_deleted = p_tuples_deleted,
        tuples_updated = p_tuples_updated,
        tuples_moved = p_tuples_moved,
        status = 'completed'
    WHERE id = p_vacuum_id;
END;
$$ LANGUAGE plpgsql;
```

## Automated Maintenance

### Maintenance Scheduling

```sql
-- Create maintenance schedule table
CREATE TABLE maintenance_schedule (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(100) UNIQUE NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    schedule_cron VARCHAR(100) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to schedule maintenance task
CREATE OR REPLACE FUNCTION schedule_maintenance_task(
    p_task_name VARCHAR(100),
    p_task_type VARCHAR(50),
    p_schedule_cron VARCHAR(100),
    p_is_enabled BOOLEAN DEFAULT TRUE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO maintenance_schedule (
        task_name, task_type, schedule_cron, is_enabled
    ) VALUES (
        p_task_name, p_task_type, p_schedule_cron, p_is_enabled
    ) ON CONFLICT (task_name) 
    DO UPDATE SET 
        task_type = EXCLUDED.task_type,
        schedule_cron = EXCLUDED.schedule_cron,
        is_enabled = EXCLUDED.is_enabled;
END;
$$ LANGUAGE plpgsql;

-- Create function to run maintenance task
CREATE OR REPLACE FUNCTION run_maintenance_task(p_task_name VARCHAR(100))
RETURNS BOOLEAN AS $$
DECLARE
    task_type VARCHAR(50);
    task_sql TEXT;
BEGIN
    -- Get task type
    SELECT task_type INTO task_type
    FROM maintenance_schedule
    WHERE task_name = p_task_name AND is_enabled = TRUE;
    
    IF task_type IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Execute task based on type
    CASE task_type
        WHEN 'vacuum' THEN
            EXECUTE format('VACUUM ANALYZE %I', p_task_name);
        WHEN 'reindex' THEN
            EXECUTE format('REINDEX TABLE %I', p_task_name);
        WHEN 'analyze' THEN
            EXECUTE format('ANALYZE %I', p_task_name);
        WHEN 'vacuum_full' THEN
            EXECUTE format('VACUUM FULL %I', p_task_name);
        ELSE
            RAISE EXCEPTION 'Unknown task type: %', task_type;
    END CASE;
    
    -- Update last run time
    UPDATE maintenance_schedule
    SET last_run = CURRENT_TIMESTAMP
    WHERE task_name = p_task_name;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Maintenance Automation

```python
# maintenance/postgres_maintenance.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging
import schedule
import time

class PostgreSQLMaintenance:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_vacuum_analyze(self, table_name):
        """Run VACUUM ANALYZE on table."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Start monitoring
                cur.execute("SELECT start_vacuum_monitoring(%s, %s)", (table_name, 'automated'))
                vacuum_id = cur.fetchone()[0]
                
                # Run vacuum analyze
                cur.execute(f"VACUUM ANALYZE {table_name}")
                
                # Complete monitoring
                cur.execute("SELECT complete_vacuum_monitoring(%s)", (vacuum_id,))
                
                conn.commit()
                self.logger.info(f"VACUUM ANALYZE completed for {table_name}")
                
        except Exception as e:
            self.logger.error(f"VACUUM ANALYZE failed for {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def run_reindex(self, table_name):
        """Run REINDEX on table."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX TABLE {table_name}")
                conn.commit()
                self.logger.info(f"REINDEX completed for {table_name}")
                
        except Exception as e:
            self.logger.error(f"REINDEX failed for {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def run_analyze(self, table_name):
        """Run ANALYZE on table."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"ANALYZE {table_name}")
                conn.commit()
                self.logger.info(f"ANALYZE completed for {table_name}")
                
        except Exception as e:
            self.logger.error(f"ANALYZE failed for {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_maintenance_tasks(self):
        """Get scheduled maintenance tasks."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT task_name, task_type, schedule_cron, is_enabled, last_run, next_run
                    FROM maintenance_schedule
                    WHERE is_enabled = TRUE
                    ORDER BY next_run
                """)
                
                tasks = cur.fetchall()
                return tasks
                
        except Exception as e:
            self.logger.error(f"Error getting maintenance tasks: {e}")
            return []
        finally:
            conn.close()
    
    def run_scheduled_tasks(self):
        """Run all scheduled maintenance tasks."""
        tasks = self.get_maintenance_tasks()
        current_time = datetime.now()
        
        for task_name, task_type, schedule_cron, is_enabled, last_run, next_run in tasks:
            if next_run and next_run <= current_time:
                try:
                    if task_type == 'vacuum':
                        self.run_vacuum_analyze(task_name)
                    elif task_type == 'reindex':
                        self.run_reindex(task_name)
                    elif task_type == 'analyze':
                        self.run_analyze(task_name)
                    
                    self.logger.info(f"Scheduled task {task_name} completed")
                    
                except Exception as e:
                    self.logger.error(f"Scheduled task {task_name} failed: {e}")

# Usage
if __name__ == "__main__":
    maintenance = PostgreSQLMaintenance({
        'host': 'localhost',
        'database': 'production',
        'user': 'maintenance_user',
        'password': 'maintenance_password'
    })
    
    # Run maintenance tasks
    maintenance.run_scheduled_tasks()
```

## Statistics Management

### Statistics Monitoring

```sql
-- Create function to get table statistics
CREATE OR REPLACE FUNCTION get_table_statistics(p_table_name VARCHAR(100))
RETURNS TABLE (
    column_name VARCHAR(100),
    n_distinct NUMERIC,
    correlation NUMERIC,
    most_common_vals TEXT,
    most_common_freqs TEXT,
    histogram_bounds TEXT,
    null_frac NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.attname::VARCHAR(100) as column_name,
        s.n_distinct,
        s.correlation,
        s.most_common_vals::TEXT,
        s.most_common_freqs::TEXT,
        s.histogram_bounds::TEXT,
        s.null_frac
    FROM pg_stats s
    WHERE s.tablename = p_table_name
    ORDER BY s.attname;
END;
$$ LANGUAGE plpgsql;

-- Create function to check statistics age
CREATE OR REPLACE FUNCTION check_statistics_age(p_table_name VARCHAR(100))
RETURNS TABLE (
    table_name VARCHAR(100),
    last_analyze TIMESTAMPTZ,
    last_autoanalyze TIMESTAMPTZ,
    analyze_age_hours NUMERIC,
    autoanalyze_age_hours NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.tablename::VARCHAR(100) as table_name,
        t.last_analyze,
        t.last_autoanalyze,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - t.last_analyze)) / 3600 as analyze_age_hours,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - t.last_autoanalyze)) / 3600 as autoanalyze_age_hours
    FROM pg_stat_user_tables t
    WHERE t.tablename = p_table_name;
END;
$$ LANGUAGE plpgsql;
```

### Statistics Maintenance

```sql
-- Create function to update statistics for all tables
CREATE OR REPLACE FUNCTION update_all_statistics()
RETURNS INTEGER AS $$
DECLARE
    table_record RECORD;
    updated_count INTEGER := 0;
BEGIN
    FOR table_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
    LOOP
        EXECUTE format('ANALYZE %I.%I', table_record.schemaname, table_record.tablename);
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to update statistics for stale tables
CREATE OR REPLACE FUNCTION update_stale_statistics(p_hours_threshold INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    table_record RECORD;
    updated_count INTEGER := 0;
BEGIN
    FOR table_record IN 
        SELECT schemaname, tablename 
        FROM pg_stat_user_tables 
        WHERE last_autoanalyze < CURRENT_TIMESTAMP - (p_hours_threshold || ' hours')::interval
        OR last_autoanalyze IS NULL
    LOOP
        EXECUTE format('ANALYZE %I.%I', table_record.schemaname, table_record.tablename);
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;
```

## Index Maintenance

### Index Health Monitoring

```sql
-- Create function to check index health
CREATE OR REPLACE FUNCTION check_index_health()
RETURNS TABLE (
    schemaname VARCHAR(100),
    tablename VARCHAR(100),
    indexname VARCHAR(100),
    index_size TEXT,
    idx_scan BIGINT,
    idx_tup_read BIGINT,
    idx_tup_fetch BIGINT,
    bloat_ratio NUMERIC,
    is_healthy BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::VARCHAR(100),
        s.tablename::VARCHAR(100),
        s.indexname::VARCHAR(100),
        pg_size_pretty(pg_relation_size(s.indexrelid)) as index_size,
        s.idx_scan,
        s.idx_tup_read,
        s.idx_tup_fetch,
        CASE 
            WHEN s.idx_tup_read > 0 THEN (s.idx_tup_read - s.idx_tup_fetch)::NUMERIC / s.idx_tup_read
            ELSE 0
        END as bloat_ratio,
        (s.idx_scan > 0 AND s.idx_tup_fetch > 0) as is_healthy
    FROM pg_stat_user_indexes s
    ORDER BY s.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to find unused indexes
CREATE OR REPLACE FUNCTION find_unused_indexes()
RETURNS TABLE (
    schemaname VARCHAR(100),
    tablename VARCHAR(100),
    indexname VARCHAR(100),
    index_size TEXT,
    index_size_bytes BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::VARCHAR(100),
        s.tablename::VARCHAR(100),
        s.indexname::VARCHAR(100),
        pg_size_pretty(pg_relation_size(s.indexrelid)) as index_size,
        pg_relation_size(s.indexrelid) as index_size_bytes
    FROM pg_stat_user_indexes s
    WHERE s.idx_scan = 0
    AND s.indexname NOT LIKE '%_pkey'
    AND s.indexname NOT LIKE '%_unique_%'
    ORDER BY pg_relation_size(s.indexrelid) DESC;
END;
$$ LANGUAGE plpgsql;
```

### Index Maintenance Automation

```sql
-- Create function to reindex unused indexes
CREATE OR REPLACE FUNCTION reindex_unused_indexes()
RETURNS INTEGER AS $$
DECLARE
    index_record RECORD;
    reindexed_count INTEGER := 0;
BEGIN
    FOR index_record IN 
        SELECT schemaname, indexname 
        FROM find_unused_indexes()
    LOOP
        EXECUTE format('REINDEX INDEX %I.%I', index_record.schemaname, index_record.indexname);
        reindexed_count := reindexed_count + 1;
    END LOOP;
    
    RETURN reindexed_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to drop unused indexes
CREATE OR REPLACE FUNCTION drop_unused_indexes()
RETURNS INTEGER AS $$
DECLARE
    index_record RECORD;
    dropped_count INTEGER := 0;
BEGIN
    FOR index_record IN 
        SELECT schemaname, indexname 
        FROM find_unused_indexes()
    LOOP
        EXECUTE format('DROP INDEX %I.%I', index_record.schemaname, index_record.indexname);
        dropped_count := dropped_count + 1;
    END LOOP;
    
    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;
```

## Maintenance Monitoring

### Maintenance Performance Monitoring

```python
# monitoring/maintenance_monitor.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging

class MaintenanceMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_vacuum_statistics(self):
        """Get vacuum statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        table_name,
                        COUNT(*) as vacuum_count,
                        AVG(duration_seconds) as avg_duration,
                        MAX(duration_seconds) as max_duration,
                        SUM(pages_removed) as total_pages_removed,
                        SUM(tuples_deleted) as total_tuples_deleted
                    FROM vacuum_history
                    WHERE started_at > CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY table_name
                    ORDER BY vacuum_count DESC
                """)
                
                vacuum_stats = cur.fetchall()
                return vacuum_stats
                
        except Exception as e:
            self.logger.error(f"Error getting vacuum statistics: {e}")
            return []
        finally:
            conn.close()
    
    def get_index_health(self):
        """Get index health metrics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM check_index_health()")
                
                index_health = cur.fetchall()
                return index_health
                
        except Exception as e:
            self.logger.error(f"Error getting index health: {e}")
            return []
        finally:
            conn.close()
    
    def get_maintenance_summary(self):
        """Get maintenance summary."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        'vacuum' as task_type,
                        COUNT(*) as task_count,
                        AVG(duration_seconds) as avg_duration
                    FROM vacuum_history
                    WHERE started_at > CURRENT_DATE - INTERVAL '7 days'
                    UNION ALL
                    SELECT 
                        'analyze' as task_type,
                        COUNT(*) as task_count,
                        AVG(duration_seconds) as avg_duration
                    FROM vacuum_history
                    WHERE vacuum_type = 'analyze'
                    AND started_at > CURRENT_DATE - INTERVAL '7 days'
                """)
                
                summary = cur.fetchall()
                return summary
                
        except Exception as e:
            self.logger.error(f"Error getting maintenance summary: {e}")
            return []
        finally:
            conn.close()
    
    def generate_maintenance_report(self):
        """Generate maintenance report."""
        vacuum_stats = self.get_vacuum_statistics()
        index_health = self.get_index_health()
        maintenance_summary = self.get_maintenance_summary()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'vacuum_statistics': vacuum_stats,
            'index_health': index_health,
            'maintenance_summary': maintenance_summary
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = MaintenanceMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_maintenance_report()
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Configure table vacuum settings
SELECT configure_table_vacuum('users', TRUE, TRUE, 0.1, 100);

-- 2. Run manual vacuum
VACUUM ANALYZE users;

-- 3. Check vacuum statistics
SELECT * FROM get_table_statistics('users');

-- 4. Monitor index health
SELECT * FROM check_index_health();

-- 5. Update statistics
SELECT update_stale_statistics(24);
```

### Essential Patterns

```python
# Complete PostgreSQL maintenance and vacuum setup
def setup_postgresql_maintenance_vacuum():
    # 1. Vacuum operations
    # 2. Automated maintenance
    # 3. Statistics management
    # 4. Index maintenance
    # 5. Maintenance monitoring
    # 6. Performance optimization
    # 7. Health checks
    # 8. Automation scheduling
    
    print("PostgreSQL maintenance and vacuum setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL maintenance and vacuum excellence. Each pattern includes implementation examples, maintenance strategies, and real-world usage patterns for enterprise PostgreSQL maintenance systems.*
