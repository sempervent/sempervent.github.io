# PostgreSQL Monitoring & Observability Best Practices

**Objective**: Master senior-level PostgreSQL monitoring and observability patterns for production systems. When you need to implement comprehensive monitoring, when you want to track performance metrics, when you need enterprise-grade observability strategies—these best practices become your weapon of choice.

## Core Principles

- **Three Pillars**: Metrics, Logs, and Traces
- **Proactive Monitoring**: Detect issues before they impact users
- **Comprehensive Coverage**: Monitor all aspects of PostgreSQL
- **Actionable Alerts**: Clear, actionable alerting strategies
- **Historical Analysis**: Long-term trend analysis and capacity planning

## Metrics Collection

### PostgreSQL Statistics

```sql
-- Enable essential extensions for monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements_reset;

-- View database statistics
SELECT 
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted,
    conflicts,
    temp_files,
    temp_bytes,
    deadlocks,
    blk_read_time,
    blk_write_time
FROM pg_stat_database 
WHERE datname = current_database();

-- View table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    n_mod_since_analyze,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    vacuum_count,
    autovacuum_count,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- View index statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Performance Metrics

```sql
-- Query performance statistics
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

-- Connection statistics
SELECT 
    count(*) as total_connections,
    count(*) FILTER (WHERE state = 'active') as active_connections,
    count(*) FILTER (WHERE state = 'idle') as idle_connections,
    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
    count(*) FILTER (WHERE state = 'idle in transaction (aborted)') as idle_in_transaction_aborted
FROM pg_stat_activity;

-- Lock statistics
SELECT 
    mode,
    count(*) as lock_count
FROM pg_locks 
GROUP BY mode 
ORDER BY lock_count DESC;
```

## Prometheus Integration

### PostgreSQL Exporter

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: production
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  postgres_exporter:
    image: prometheuscommunity/postgres-exporter
    environment:
      DATA_SOURCE_NAME: "postgresql://postgres:postgres@postgres:5432/production?sslmode=disable"
    ports:
      - "9187:9187"
    depends_on:
      - postgres

  prometheus:
    image: prometheus/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 5s

rule_files:
  - "postgresql_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# postgresql_rules.yml
groups:
  - name: postgresql
    rules:
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL instance is down"
          description: "PostgreSQL instance {{ $labels.instance }} has been down for more than 1 minute."

      - alert: PostgreSQLHighConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL high connection usage"
          description: "PostgreSQL instance {{ $labels.instance }} has {{ $value | humanizePercentage }} connection usage."

      - alert: PostgreSQLSlowQueries
        expr: rate(pg_stat_statements_total_time[5m]) > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL slow queries detected"
          description: "PostgreSQL instance {{ $labels.instance }} has slow queries with {{ $value }}ms average execution time."

      - alert: PostgreSQLHighDeadTuples
        expr: pg_stat_user_tables_n_dead_tup / pg_stat_user_tables_n_live_tup > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL high dead tuple ratio"
          description: "Table {{ $labels.schemaname }}.{{ $labels.relname }} has {{ $value | humanizePercentage }} dead tuple ratio."

      - alert: PostgreSQLReplicationLag
        expr: pg_replication_lag > 300
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL replication lag high"
          description: "PostgreSQL replication lag is {{ $value }}s on {{ $labels.instance }}."
```

## Custom Metrics Collection

### Python Metrics Collector

```python
# monitoring/metrics_collector.py
import psycopg2
import time
import json
from prometheus_client import Gauge, Counter, Histogram, start_http_server
from datetime import datetime, timedelta
import logging

class PostgreSQLMetricsCollector:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        
        # Prometheus metrics
        self.connections_total = Gauge('postgresql_connections_total', 'Total connections')
        self.active_connections = Gauge('postgresql_active_connections', 'Active connections')
        self.database_size = Gauge('postgresql_database_size_bytes', 'Database size in bytes')
        self.table_size = Gauge('postgresql_table_size_bytes', 'Table size in bytes', ['schema', 'table'])
        self.index_size = Gauge('postgresql_index_size_bytes', 'Index size in bytes', ['schema', 'table', 'index'])
        self.slow_queries = Counter('postgresql_slow_queries_total', 'Total slow queries')
        self.query_duration = Histogram('postgresql_query_duration_seconds', 'Query duration')
        self.dead_tuples = Gauge('postgresql_dead_tuples', 'Dead tuples', ['schema', 'table'])
        self.lock_waits = Counter('postgresql_lock_waits_total', 'Total lock waits')
        self.checkpoints = Counter('postgresql_checkpoints_total', 'Total checkpoints')
        self.wal_generated = Counter('postgresql_wal_generated_bytes', 'WAL generated in bytes')
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to PostgreSQL."""
        return psycopg2.connect(**self.conn_params)
    
    def collect_connection_metrics(self, conn):
        """Collect connection metrics."""
        with conn.cursor() as cur:
            # Total connections
            cur.execute("SELECT count(*) FROM pg_stat_activity;")
            total_connections = cur.fetchone()[0]
            self.connections_total.set(total_connections)
            
            # Active connections
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
            active_connections = cur.fetchone()[0]
            self.active_connections.set(active_connections)
    
    def collect_database_metrics(self, conn):
        """Collect database metrics."""
        with conn.cursor() as cur:
            # Database size
            cur.execute("SELECT pg_database_size(current_database());")
            db_size = cur.fetchone()[0]
            self.database_size.set(db_size)
            
            # Table sizes
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_total_relation_size(schemaname||'.'||tablename) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY size DESC;
            """)
            
            for schema, table, size in cur.fetchall():
                self.table_size.labels(schema=schema, table=table).set(size)
            
            # Index sizes
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_relation_size(schemaname||'.'||indexname) as size
                FROM pg_stat_user_indexes
                ORDER BY size DESC;
            """)
            
            for schema, table, index, size in cur.fetchall():
                self.index_size.labels(schema=schema, table=table, index=index).set(size)
    
    def collect_performance_metrics(self, conn):
        """Collect performance metrics."""
        with conn.cursor() as cur:
            # Slow queries
            cur.execute("""
                SELECT count(*) FROM pg_stat_statements 
                WHERE mean_time > 1000;
            """)
            slow_queries = cur.fetchone()[0]
            self.slow_queries.inc(slow_queries)
            
            # Dead tuples
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    n_dead_tup
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 0;
            """)
            
            for schema, table, dead_tuples in cur.fetchall():
                self.dead_tuples.labels(schema=schema, table=table).set(dead_tuples)
            
            # Lock waits
            cur.execute("""
                SELECT count(*) FROM pg_stat_activity 
                WHERE wait_event_type = 'Lock';
            """)
            lock_waits = cur.fetchone()[0]
            self.lock_waits.inc(lock_waits)
    
    def collect_wal_metrics(self, conn):
        """Collect WAL metrics."""
        with conn.cursor() as cur:
            # WAL generated
            cur.execute("""
                SELECT 
                    pg_current_wal_lsn(),
                    pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0') as wal_bytes;
            """)
            wal_bytes = cur.fetchone()[1]
            self.wal_generated.inc(wal_bytes)
    
    def collect_all_metrics(self):
        """Collect all metrics."""
        try:
            conn = self.connect()
            
            self.collect_connection_metrics(conn)
            self.collect_database_metrics(conn)
            self.collect_performance_metrics(conn)
            self.collect_wal_metrics(conn)
            
            conn.close()
            self.logger.info("Metrics collected successfully")
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def start_metrics_server(self, port=8000):
        """Start Prometheus metrics server."""
        start_http_server(port)
        self.logger.info(f"Metrics server started on port {port}")
        
        while True:
            self.collect_all_metrics()
            time.sleep(30)

# Usage
if __name__ == "__main__":
    collector = PostgreSQLMetricsCollector({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    collector.start_metrics_server()
```

## Logging and Analysis

### Structured Logging

```python
# logging/structured_logger.py
import psycopg2
import json
import logging
from datetime import datetime
from typing import Dict, Any

class PostgreSQLStructuredLogger:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/postgresql/structured.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_query_execution(self, query: str, duration: float, rows: int, user: str):
        """Log query execution details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'query_execution',
            'query': query,
            'duration_ms': duration * 1000,
            'rows_affected': rows,
            'user': user,
            'database': self.conn_params['database']
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_connection_event(self, event_type: str, user: str, client_ip: str):
        """Log connection events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'client_ip': client_ip,
            'database': self.conn_params['database']
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type: str, error_message: str, query: str = None):
        """Log error events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'query': query,
            'database': self.conn_params['database']
        }
        
        self.logger.error(json.dumps(log_entry))
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'performance_metrics',
            'metrics': metrics,
            'database': self.conn_params['database']
        }
        
        self.logger.info(json.dumps(log_entry))

# Usage
if __name__ == "__main__":
    logger = PostgreSQLStructuredLogger({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    # Log query execution
    logger.log_query_execution(
        query="SELECT * FROM users WHERE active = true",
        duration=0.150,
        rows=1000,
        user="app_user"
    )
    
    # Log connection event
    logger.log_connection_event(
        event_type="connection_established",
        user="app_user",
        client_ip="192.168.1.100"
    )
```

### Log Analysis

```python
# analysis/log_analyzer.py
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd

class PostgreSQLLogAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logs = []
    
    def parse_logs(self):
        """Parse structured logs."""
        with open(self.log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    self.logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    def analyze_slow_queries(self, threshold_ms=1000):
        """Analyze slow queries."""
        slow_queries = []
        
        for log in self.logs:
            if log.get('event_type') == 'query_execution':
                if log.get('duration_ms', 0) > threshold_ms:
                    slow_queries.append({
                        'query': log.get('query'),
                        'duration_ms': log.get('duration_ms'),
                        'rows': log.get('rows_affected'),
                        'user': log.get('user'),
                        'timestamp': log.get('timestamp')
                    })
        
        return slow_queries
    
    def analyze_connection_patterns(self):
        """Analyze connection patterns."""
        connections = []
        disconnections = []
        
        for log in self.logs:
            if log.get('event_type') == 'connection_established':
                connections.append(log)
            elif log.get('event_type') == 'connection_closed':
                disconnections.append(log)
        
        # Analyze connection patterns
        connection_stats = {
            'total_connections': len(connections),
            'total_disconnections': len(disconnections),
            'unique_users': len(set(log.get('user') for log in connections)),
            'unique_ips': len(set(log.get('client_ip') for log in connections))
        }
        
        return connection_stats
    
    def analyze_error_patterns(self):
        """Analyze error patterns."""
        errors = []
        
        for log in self.logs:
            if log.get('event_type') == 'error':
                errors.append({
                    'error_type': log.get('error_type'),
                    'error_message': log.get('error_message'),
                    'query': log.get('query'),
                    'timestamp': log.get('timestamp')
                })
        
        # Group errors by type
        error_counts = Counter(error['error_type'] for error in errors)
        
        return {
            'total_errors': len(errors),
            'error_types': dict(error_counts),
            'recent_errors': errors[-10:]  # Last 10 errors
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        self.parse_logs()
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_log_entries': len(self.logs),
            'slow_queries': self.analyze_slow_queries(),
            'connection_patterns': self.analyze_connection_patterns(),
            'error_patterns': self.analyze_error_patterns()
        }
        
        return report

# Usage
if __name__ == "__main__":
    analyzer = PostgreSQLLogAnalyzer('/var/log/postgresql/structured.log')
    report = analyzer.generate_report()
    
    print(json.dumps(report, indent=2))
```

## Distributed Tracing

### OpenTelemetry Integration

```python
# tracing/postgresql_tracer.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
import psycopg2

class PostgreSQLTracer:
    def __init__(self, service_name="postgresql-service"):
        self.service_name = service_name
        self.setup_tracing()
    
    def setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        # Create tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        
        # Create span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument psycopg2
        Psycopg2Instrumentor().instrument()
    
    def trace_query(self, query: str, params: tuple = None):
        """Trace a database query."""
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span("postgresql_query") as span:
            span.set_attribute("db.statement", query)
            span.set_attribute("db.system", "postgresql")
            
            if params:
                span.set_attribute("db.parameters", str(params))
            
            # Execute query
            conn = psycopg2.connect(
                host="localhost",
                database="production",
                user="app_user",
                password="app_password"
            )
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchall()
                
                span.set_attribute("db.rows_affected", len(result))
                return result
            
            conn.close()

# Usage
if __name__ == "__main__":
    tracer = PostgreSQLTracer()
    
    # Trace a query
    result = tracer.trace_query(
        "SELECT * FROM users WHERE active = %s",
        (True,)
    )
    
    print(f"Query returned {len(result)} rows")
```

## Dashboard and Visualization

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "PostgreSQL Monitoring",
    "panels": [
      {
        "title": "Database Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "postgresql_connections_total",
            "legendFormat": "Total Connections"
          },
          {
            "expr": "postgresql_active_connections",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Database Size",
        "type": "stat",
        "targets": [
          {
            "expr": "postgresql_database_size_bytes",
            "legendFormat": "Database Size"
          }
        ]
      },
      {
        "title": "Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(postgresql_slow_queries_total[5m])",
            "legendFormat": "Slow Queries/sec"
          }
        ]
      },
      {
        "title": "Table Sizes",
        "type": "table",
        "targets": [
          {
            "expr": "postgresql_table_size_bytes",
            "legendFormat": "{{schema}}.{{table}}"
          }
        ]
      }
    ]
  }
}
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Start PostgreSQL exporter
docker run -d --name postgres_exporter \
  -e DATA_SOURCE_NAME="postgresql://user:pass@localhost:5432/db?sslmode=disable" \
  -p 9187:9187 \
  prometheuscommunity/postgres-exporter

# 2. Start Prometheus
docker run -d --name prometheus \
  -p 9090:9090 \
  -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# 3. Start Grafana
docker run -d --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana
```

### Essential Patterns

```python
# Complete PostgreSQL monitoring setup
def setup_postgresql_monitoring():
    # 1. Metrics collection
    # 2. Prometheus integration
    # 3. Alerting rules
    # 4. Structured logging
    # 5. Log analysis
    # 6. Distributed tracing
    # 7. Dashboard visualization
    # 8. Performance monitoring
    
    print("PostgreSQL monitoring setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL monitoring excellence. Each pattern includes implementation examples, monitoring strategies, and real-world usage patterns for enterprise PostgreSQL monitoring systems.*
