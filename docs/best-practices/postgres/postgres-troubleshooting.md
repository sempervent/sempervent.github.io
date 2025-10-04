# PostgreSQL Troubleshooting Best Practices

**Objective**: Master senior-level PostgreSQL troubleshooting patterns for production systems. When you need to diagnose database issues, when you want to implement systematic debugging, when you need enterprise-grade troubleshooting strategies—these best practices become your weapon of choice.

## Core Principles

- **Systematic Approach**: Follow structured troubleshooting methodology
- **Log Analysis**: Leverage PostgreSQL logs for diagnosis
- **Performance Monitoring**: Use metrics to identify bottlenecks
- **Root Cause Analysis**: Identify underlying issues, not just symptoms
- **Prevention**: Implement proactive monitoring and alerting

## Diagnostic Tools and Queries

### System Health Checks

```sql
-- Create diagnostic queries table
CREATE TABLE diagnostic_queries (
    id SERIAL PRIMARY KEY,
    query_name VARCHAR(100) UNIQUE NOT NULL,
    query_sql TEXT NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert common diagnostic queries
INSERT INTO diagnostic_queries (query_name, query_sql, description, category, severity) VALUES
('active_connections', 
 'SELECT COUNT(*) as active_connections FROM pg_stat_activity WHERE state = ''active''',
 'Check number of active connections', 'connections', 'warning'),
('long_running_queries',
 'SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval ''5 minutes''',
 'Find long-running queries', 'performance', 'critical'),
('database_size',
 'SELECT pg_size_pretty(pg_database_size(current_database())) as database_size',
 'Check database size', 'storage', 'info'),
('table_bloat',
 'SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||''.''||tablename)) as size FROM pg_tables ORDER BY pg_total_relation_size(schemaname||''.''||tablename) DESC LIMIT 10',
 'Find largest tables', 'storage', 'info');

-- Create function to run diagnostic query
CREATE OR REPLACE FUNCTION run_diagnostic_query(p_query_name VARCHAR(100))
RETURNS TABLE (
    result_data JSONB
) AS $$
DECLARE
    query_sql TEXT;
BEGIN
    SELECT dq.query_sql INTO query_sql
    FROM diagnostic_queries dq
    WHERE dq.query_name = p_query_name;
    
    IF query_sql IS NULL THEN
        RAISE EXCEPTION 'Diagnostic query % not found', p_query_name;
    END IF;
    
    RETURN QUERY EXECUTE query_sql;
END;
$$ LANGUAGE plpgsql;
```

### Performance Diagnostics

```sql
-- Create function to analyze slow queries
CREATE OR REPLACE FUNCTION analyze_slow_queries(p_duration_threshold INTERVAL DEFAULT '1 minute')
RETURNS TABLE (
    pid INTEGER,
    duration INTERVAL,
    query TEXT,
    state VARCHAR(20),
    wait_event_type VARCHAR(50),
    wait_event VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.pid,
        now() - a.query_start as duration,
        a.query,
        a.state,
        a.wait_event_type,
        a.wait_event
    FROM pg_stat_activity a
    WHERE a.query_start IS NOT NULL
    AND now() - a.query_start > p_duration_threshold
    AND a.state != 'idle'
    ORDER BY duration DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to check lock contention
CREATE OR REPLACE FUNCTION check_lock_contention()
RETURNS TABLE (
    blocked_pid INTEGER,
    blocked_query TEXT,
    blocking_pid INTEGER,
    blocking_query TEXT,
    lock_type VARCHAR(50),
    mode VARCHAR(20),
    granted BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        blocked_locks.pid AS blocked_pid,
        blocked_activity.query AS blocked_query,
        blocking_locks.pid AS blocking_pid,
        blocking_activity.query AS blocking_query,
        blocked_locks.locktype AS lock_type,
        blocked_locks.mode AS mode,
        blocked_locks.granted AS granted
    FROM pg_catalog.pg_locks blocked_locks
    JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
    JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
        AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
        AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
        AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
        AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
        AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
        AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
        AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
        AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
        AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
        AND blocking_locks.pid != blocked_locks.pid
    JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
    WHERE NOT blocked_locks.granted;
END;
$$ LANGUAGE plpgsql;
```

## Common Issues and Solutions

### Connection Issues

```sql
-- Create function to diagnose connection issues
CREATE OR REPLACE FUNCTION diagnose_connection_issues()
RETURNS TABLE (
    issue_type VARCHAR(50),
    description TEXT,
    current_value TEXT,
    recommended_value TEXT,
    severity VARCHAR(20)
) AS $$
BEGIN
    -- Check max_connections
    RETURN QUERY
    SELECT 
        'max_connections'::VARCHAR(50),
        'Maximum number of concurrent connections'::TEXT,
        current_setting('max_connections')::TEXT,
        '200'::TEXT,
        CASE 
            WHEN current_setting('max_connections')::INTEGER < 100 THEN 'critical'
            WHEN current_setting('max_connections')::INTEGER < 200 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
    
    -- Check active connections
    RETURN QUERY
    SELECT 
        'active_connections'::VARCHAR(50),
        'Current number of active connections'::TEXT,
        (SELECT COUNT(*)::TEXT FROM pg_stat_activity)::TEXT,
        'Less than 80% of max_connections'::TEXT,
        CASE 
            WHEN (SELECT COUNT(*) FROM pg_stat_activity)::FLOAT / current_setting('max_connections')::FLOAT > 0.8 THEN 'critical'
            WHEN (SELECT COUNT(*) FROM pg_stat_activity)::FLOAT / current_setting('max_connections')::FLOAT > 0.6 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
    
    -- Check connection timeout
    RETURN QUERY
    SELECT 
        'idle_in_transaction_session_timeout'::VARCHAR(50),
        'Timeout for idle transactions'::TEXT,
        current_setting('idle_in_transaction_session_timeout')::TEXT,
        '300000'::TEXT,
        CASE 
            WHEN current_setting('idle_in_transaction_session_timeout')::INTEGER = 0 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
END;
$$ LANGUAGE plpgsql;
```

### Performance Issues

```sql
-- Create function to diagnose performance issues
CREATE OR REPLACE FUNCTION diagnose_performance_issues()
RETURNS TABLE (
    issue_type VARCHAR(50),
    description TEXT,
    current_value TEXT,
    recommended_value TEXT,
    severity VARCHAR(20)
) AS $$
BEGIN
    -- Check shared_buffers
    RETURN QUERY
    SELECT 
        'shared_buffers'::VARCHAR(50),
        'Shared buffer pool size'::TEXT,
        current_setting('shared_buffers')::TEXT,
        '25% of total RAM'::TEXT,
        CASE 
            WHEN current_setting('shared_buffers')::INTEGER < 128 THEN 'critical'
            WHEN current_setting('shared_buffers')::INTEGER < 256 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
    
    -- Check work_mem
    RETURN QUERY
    SELECT 
        'work_mem'::VARCHAR(50),
        'Memory for sorting and hashing'::TEXT,
        current_setting('work_mem')::TEXT,
        '4MB'::TEXT,
        CASE 
            WHEN current_setting('work_mem')::INTEGER < 1024 THEN 'critical'
            WHEN current_setting('work_mem')::INTEGER < 4096 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
    
    -- Check effective_cache_size
    RETURN QUERY
    SELECT 
        'effective_cache_size'::VARCHAR(50),
        'Estimated cache size for query planning'::TEXT,
        current_setting('effective_cache_size')::TEXT,
        '75% of total RAM'::TEXT,
        CASE 
            WHEN current_setting('effective_cache_size')::INTEGER < 1024 THEN 'critical'
            WHEN current_setting('effective_cache_size')::INTEGER < 2048 THEN 'warning'
            ELSE 'info'
        END::VARCHAR(20);
END;
$$ LANGUAGE plpgsql;
```

## Log Analysis

### Log Parsing and Analysis

```sql
-- Create log analysis table
CREATE TABLE log_analysis (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    log_message TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    process_id INTEGER,
    session_id VARCHAR(50),
    user_name VARCHAR(100),
    database_name VARCHAR(100),
    application_name VARCHAR(100),
    client_addr INET,
    parsed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to parse log entries
CREATE OR REPLACE FUNCTION parse_log_entry(p_log_line TEXT)
RETURNS TABLE (
    log_level VARCHAR(20),
    timestamp TIMESTAMPTZ,
    process_id INTEGER,
    session_id VARCHAR(50),
    user_name VARCHAR(100),
    database_name VARCHAR(100),
    application_name VARCHAR(100),
    client_addr INET,
    message TEXT
) AS $$
DECLARE
    log_parts TEXT[];
BEGIN
    -- Parse PostgreSQL log format
    -- Format: timestamp [pid] level: message
    log_parts := string_to_array(p_log_line, ' ');
    
    IF array_length(log_parts, 1) >= 4 THEN
        RETURN QUERY
        SELECT 
            log_parts[3]::VARCHAR(20) as log_level,
            log_parts[1]::TIMESTAMPTZ as timestamp,
            (regexp_match(log_parts[2], '\[(\d+)\]'))[1]::INTEGER as process_id,
            NULL::VARCHAR(50) as session_id,
            NULL::VARCHAR(100) as user_name,
            NULL::VARCHAR(100) as database_name,
            NULL::VARCHAR(100) as application_name,
            NULL::INET as client_addr,
            array_to_string(log_parts[4:], ' ')::TEXT as message;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Error Pattern Detection

```sql
-- Create function to detect error patterns
CREATE OR REPLACE FUNCTION detect_error_patterns(p_hours INTEGER DEFAULT 24)
RETURNS TABLE (
    error_pattern VARCHAR(100),
    error_count BIGINT,
    first_occurrence TIMESTAMPTZ,
    last_occurrence TIMESTAMPTZ,
    severity VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN la.log_message ILIKE '%connection%' THEN 'connection_errors'
            WHEN la.log_message ILIKE '%timeout%' THEN 'timeout_errors'
            WHEN la.log_message ILIKE '%deadlock%' THEN 'deadlock_errors'
            WHEN la.log_message ILIKE '%constraint%' THEN 'constraint_errors'
            WHEN la.log_message ILIKE '%permission%' THEN 'permission_errors'
            ELSE 'other_errors'
        END as error_pattern,
        COUNT(*) as error_count,
        MIN(la.timestamp) as first_occurrence,
        MAX(la.timestamp) as last_occurrence,
        CASE 
            WHEN COUNT(*) > 100 THEN 'critical'
            WHEN COUNT(*) > 50 THEN 'warning'
            ELSE 'info'
        END as severity
    FROM log_analysis la
    WHERE la.log_level IN ('ERROR', 'FATAL', 'PANIC')
    AND la.timestamp > CURRENT_TIMESTAMP - (p_hours || ' hours')::interval
    GROUP BY error_pattern
    ORDER BY error_count DESC;
END;
$$ LANGUAGE plpgsql;
```

## Automated Troubleshooting

### Issue Detection

```python
# troubleshooting/postgres_troubleshooter.py
import psycopg2
import json
import re
from datetime import datetime, timedelta
import logging

class PostgreSQLTroubleshooter:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_health_checks(self):
        """Run comprehensive health checks."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Check active connections
                cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
                active_connections = cur.fetchone()[0]
                
                # Check long-running queries
                cur.execute("""
                    SELECT COUNT(*) FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND now() - query_start > interval '5 minutes'
                """)
                long_queries = cur.fetchone()[0]
                
                # Check lock contention
                cur.execute("SELECT COUNT(*) FROM pg_locks WHERE NOT granted")
                lock_contention = cur.fetchone()[0]
                
                # Check database size
                cur.execute("SELECT pg_database_size(current_database())")
                db_size = cur.fetchone()[0]
                
                return {
                    'active_connections': active_connections,
                    'long_queries': long_queries,
                    'lock_contention': lock_contention,
                    'database_size': db_size
                }
                
        except Exception as e:
            self.logger.error(f"Error running health checks: {e}")
            return {}
        finally:
            conn.close()
    
    def diagnose_performance_issues(self):
        """Diagnose performance issues."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Get slow queries
                cur.execute("""
                    SELECT pid, now() - query_start as duration, query
                    FROM pg_stat_activity 
                    WHERE state = 'active' 
                    AND now() - query_start > interval '1 minute'
                    ORDER BY duration DESC
                """)
                slow_queries = cur.fetchall()
                
                # Get lock information
                cur.execute("""
                    SELECT blocked_locks.pid as blocked_pid,
                           blocked_activity.query as blocked_query,
                           blocking_locks.pid as blocking_pid,
                           blocking_activity.query as blocking_query
                    FROM pg_catalog.pg_locks blocked_locks
                    JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
                    JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
                    WHERE NOT blocked_locks.granted
                """)
                lock_info = cur.fetchall()
                
                return {
                    'slow_queries': slow_queries,
                    'lock_info': lock_info
                }
                
        except Exception as e:
            self.logger.error(f"Error diagnosing performance issues: {e}")
            return {}
        finally:
            conn.close()
    
    def analyze_logs(self, log_file_path):
        """Analyze PostgreSQL logs."""
        error_patterns = {
            'connection_errors': r'connection.*refused|connection.*reset',
            'timeout_errors': r'timeout|timed out',
            'deadlock_errors': r'deadlock|deadlock detected',
            'constraint_errors': r'constraint.*violation',
            'permission_errors': r'permission.*denied|access.*denied'
        }
        
        error_counts = {}
        
        try:
            with open(log_file_path, 'r') as f:
                for line in f:
                    for pattern_name, pattern in error_patterns.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            error_counts[pattern_name] = error_counts.get(pattern_name, 0) + 1
            
            return error_counts
            
        except Exception as e:
            self.logger.error(f"Error analyzing logs: {e}")
            return {}
    
    def generate_troubleshooting_report(self):
        """Generate comprehensive troubleshooting report."""
        health_checks = self.run_health_checks()
        performance_issues = self.diagnose_performance_issues()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'health_checks': health_checks,
            'performance_issues': performance_issues,
            'recommendations': self.generate_recommendations(health_checks, performance_issues)
        }
        
        return report
    
    def generate_recommendations(self, health_checks, performance_issues):
        """Generate troubleshooting recommendations."""
        recommendations = []
        
        # Connection recommendations
        if health_checks.get('active_connections', 0) > 100:
            recommendations.append({
                'issue': 'High connection count',
                'recommendation': 'Consider connection pooling or increasing max_connections',
                'severity': 'warning'
            })
        
        # Performance recommendations
        if performance_issues.get('slow_queries'):
            recommendations.append({
                'issue': 'Slow queries detected',
                'recommendation': 'Review and optimize slow queries, consider adding indexes',
                'severity': 'critical'
            })
        
        if performance_issues.get('lock_info'):
            recommendations.append({
                'issue': 'Lock contention detected',
                'recommendation': 'Review transaction isolation levels and query patterns',
                'severity': 'warning'
            })
        
        return recommendations

# Usage
if __name__ == "__main__":
    troubleshooter = PostgreSQLTroubleshooter({
        'host': 'localhost',
        'database': 'production',
        'user': 'troubleshooter_user',
        'password': 'troubleshooter_password'
    })
    
    report = troubleshooter.generate_troubleshooting_report()
    print(json.dumps(report, indent=2))
```

## Emergency Procedures

### Emergency Response

```sql
-- Create emergency procedures table
CREATE TABLE emergency_procedures (
    id SERIAL PRIMARY KEY,
    procedure_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    sql_command TEXT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert emergency procedures
INSERT INTO emergency_procedures (procedure_name, description, sql_command, risk_level) VALUES
('kill_long_queries', 
 'Kill long-running queries', 
 'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = ''active'' AND now() - query_start > interval ''10 minutes''',
 'high'),
('kill_idle_connections',
 'Kill idle connections',
 'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = ''idle'' AND now() - state_change > interval ''1 hour''',
 'medium'),
('force_vacuum',
 'Force VACUUM on all tables',
 'VACUUM ANALYZE',
 'low'),
('check_disk_space',
 'Check disk space usage',
 'SELECT pg_size_pretty(pg_database_size(current_database()))',
 'low');

-- Create function to execute emergency procedure
CREATE OR REPLACE FUNCTION execute_emergency_procedure(p_procedure_name VARCHAR(100))
RETURNS TEXT AS $$
DECLARE
    procedure_sql TEXT;
    result_text TEXT;
BEGIN
    SELECT sql_command INTO procedure_sql
    FROM emergency_procedures
    WHERE procedure_name = p_procedure_name;
    
    IF procedure_sql IS NULL THEN
        RAISE EXCEPTION 'Emergency procedure % not found', p_procedure_name;
    END IF;
    
    EXECUTE procedure_sql;
    
    RETURN 'Emergency procedure ' || p_procedure_name || ' executed successfully';
END;
$$ LANGUAGE plpgsql;
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Run health checks
SELECT * FROM diagnose_connection_issues();
SELECT * FROM diagnose_performance_issues();

-- 2. Check slow queries
SELECT * FROM analyze_slow_queries('1 minute');

-- 3. Check lock contention
SELECT * FROM check_lock_contention();

-- 4. Run emergency procedures if needed
SELECT execute_emergency_procedure('kill_long_queries');
```

### Essential Patterns

```python
# Complete PostgreSQL troubleshooting setup
def setup_postgresql_troubleshooting():
    # 1. Diagnostic tools and queries
    # 2. Common issues and solutions
    # 3. Log analysis
    # 4. Automated troubleshooting
    # 5. Emergency procedures
    # 6. Performance diagnostics
    # 7. Issue detection
    # 8. Root cause analysis
    
    print("PostgreSQL troubleshooting setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL troubleshooting excellence. Each pattern includes implementation examples, diagnostic strategies, and real-world usage patterns for enterprise PostgreSQL troubleshooting systems.*
