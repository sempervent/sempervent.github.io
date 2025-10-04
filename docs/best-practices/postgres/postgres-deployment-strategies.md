# PostgreSQL Deployment Strategies Best Practices

**Objective**: Master senior-level PostgreSQL deployment patterns for production systems. When you need to deploy PostgreSQL at scale, when you want to implement zero-downtime deployments, when you need enterprise-grade deployment strategies—these best practices become your weapon of choice.

## Core Principles

- **Zero Downtime**: Minimize service interruption during deployments
- **Rollback Strategy**: Implement safe rollback mechanisms
- **Environment Parity**: Maintain consistency across environments
- **Automation**: Automate deployment processes
- **Monitoring**: Monitor deployment health and performance

## Deployment Architecture Patterns

### Blue-Green Deployment

```sql
-- Create deployment tracking table
CREATE TABLE deployment_history (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(50) UNIQUE NOT NULL,
    environment VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    rollback_version VARCHAR(20),
    created_by VARCHAR(100) NOT NULL
);

-- Create function to switch traffic
CREATE OR REPLACE FUNCTION switch_traffic_to_green()
RETURNS BOOLEAN AS $$
DECLARE
    current_environment VARCHAR(20);
BEGIN
    -- Get current environment
    SELECT environment INTO current_environment
    FROM deployment_history
    WHERE status = 'active'
    ORDER BY started_at DESC
    LIMIT 1;
    
    -- Switch to green environment
    IF current_environment = 'blue' THEN
        UPDATE deployment_history 
        SET status = 'inactive' 
        WHERE environment = 'blue' AND status = 'active';
        
        UPDATE deployment_history 
        SET status = 'active' 
        WHERE environment = 'green' AND status = 'ready';
        
        RETURN TRUE;
    ELSE
        RAISE EXCEPTION 'Cannot switch from % environment', current_environment;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function to rollback to blue
CREATE OR REPLACE FUNCTION rollback_to_blue()
RETURNS BOOLEAN AS $$
BEGIN
    -- Switch back to blue environment
    UPDATE deployment_history 
    SET status = 'inactive' 
    WHERE environment = 'green' AND status = 'active';
    
    UPDATE deployment_history 
    SET status = 'active' 
    WHERE environment = 'blue' AND status = 'ready';
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Canary Deployment

```sql
-- Create canary deployment tracking
CREATE TABLE canary_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(50) UNIQUE NOT NULL,
    version VARCHAR(20) NOT NULL,
    canary_percentage INTEGER DEFAULT 10,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    metrics JSONB,
    created_by VARCHAR(100) NOT NULL
);

-- Create function to adjust canary traffic
CREATE OR REPLACE FUNCTION adjust_canary_traffic(
    p_deployment_id VARCHAR(50),
    p_percentage INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Validate percentage
    IF p_percentage < 0 OR p_percentage > 100 THEN
        RAISE EXCEPTION 'Canary percentage must be between 0 and 100';
    END IF;
    
    -- Update canary percentage
    UPDATE canary_deployments
    SET canary_percentage = p_percentage
    WHERE deployment_id = p_deployment_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create function to promote canary
CREATE OR REPLACE FUNCTION promote_canary(p_deployment_id VARCHAR(50))
RETURNS BOOLEAN AS $$
BEGIN
    -- Promote canary to full deployment
    UPDATE canary_deployments
    SET status = 'promoted',
        canary_percentage = 100,
        completed_at = CURRENT_TIMESTAMP
    WHERE deployment_id = p_deployment_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## Database Migration Strategies

### Schema Migration Management

```sql
-- Create migration tracking table
CREATE TABLE schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100) NOT NULL,
    rollback_sql TEXT,
    checksum VARCHAR(64)
);

-- Create function to apply migration
CREATE OR REPLACE FUNCTION apply_migration(
    p_version VARCHAR(50),
    p_description TEXT,
    p_sql TEXT,
    p_rollback_sql TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    migration_checksum VARCHAR(64);
BEGIN
    -- Check if migration already applied
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = p_version) THEN
        RAISE EXCEPTION 'Migration % already applied', p_version;
    END IF;
    
    -- Calculate checksum
    migration_checksum := encode(digest(p_sql, 'sha256'), 'hex');
    
    -- Apply migration
    EXECUTE p_sql;
    
    -- Record migration
    INSERT INTO schema_migrations (version, description, rollback_sql, checksum, applied_by)
    VALUES (p_version, p_description, p_rollback_sql, migration_checksum, current_user);
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create function to rollback migration
CREATE OR REPLACE FUNCTION rollback_migration(p_version VARCHAR(50))
RETURNS BOOLEAN AS $$
DECLARE
    rollback_sql TEXT;
BEGIN
    -- Get rollback SQL
    SELECT rollback_sql INTO rollback_sql
    FROM schema_migrations
    WHERE version = p_version;
    
    IF rollback_sql IS NULL THEN
        RAISE EXCEPTION 'No rollback SQL for migration %', p_version;
    END IF;
    
    -- Execute rollback
    EXECUTE rollback_sql;
    
    -- Remove migration record
    DELETE FROM schema_migrations WHERE version = p_version;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Zero-Downtime Migrations

```sql
-- Create function for zero-downtime column addition
CREATE OR REPLACE FUNCTION add_column_zero_downtime(
    p_table_name TEXT,
    p_column_name TEXT,
    p_column_type TEXT,
    p_default_value TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    -- Add column with default value
    EXECUTE format('ALTER TABLE %I ADD COLUMN %I %s', 
                   p_table_name, p_column_name, p_column_type);
    
    IF p_default_value IS NOT NULL THEN
        EXECUTE format('UPDATE %I SET %I = %L', 
                       p_table_name, p_column_name, p_default_value);
    END IF;
    
    -- Add NOT NULL constraint if default value provided
    IF p_default_value IS NOT NULL THEN
        EXECUTE format('ALTER TABLE %I ALTER COLUMN %I SET NOT NULL', 
                       p_table_name, p_column_name);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create function for zero-downtime column removal
CREATE OR REPLACE FUNCTION remove_column_zero_downtime(
    p_table_name TEXT,
    p_column_name TEXT
)
RETURNS VOID AS $$
BEGIN
    -- Remove column
    EXECUTE format('ALTER TABLE %I DROP COLUMN %I', 
                   p_table_name, p_column_name);
END;
$$ LANGUAGE plpgsql;
```

## Environment Management

### Environment Configuration

```sql
-- Create environment configuration table
CREATE TABLE environment_config (
    id SERIAL PRIMARY KEY,
    environment VARCHAR(20) NOT NULL,
    config_key VARCHAR(100) NOT NULL,
    config_value TEXT NOT NULL,
    is_secret BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(environment, config_key)
);

-- Create function to get environment config
CREATE OR REPLACE FUNCTION get_environment_config(
    p_environment VARCHAR(20),
    p_config_key VARCHAR(100)
)
RETURNS TEXT AS $$
DECLARE
    config_value TEXT;
BEGIN
    SELECT config_value INTO config_value
    FROM environment_config
    WHERE environment = p_environment AND config_key = p_config_key;
    
    RETURN config_value;
END;
$$ LANGUAGE plpgsql;

-- Create function to set environment config
CREATE OR REPLACE FUNCTION set_environment_config(
    p_environment VARCHAR(20),
    p_config_key VARCHAR(100),
    p_config_value TEXT,
    p_is_secret BOOLEAN DEFAULT FALSE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO environment_config (environment, config_key, config_value, is_secret)
    VALUES (p_environment, p_config_key, p_config_value, p_is_secret)
    ON CONFLICT (environment, config_key) 
    DO UPDATE SET 
        config_value = EXCLUDED.config_value,
        is_secret = EXCLUDED.is_secret,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
```

### Environment Validation

```sql
-- Create function to validate environment
CREATE OR REPLACE FUNCTION validate_environment(p_environment VARCHAR(20))
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    message TEXT
) AS $$
BEGIN
    -- Check database connectivity
    RETURN QUERY SELECT 'database_connectivity'::TEXT, 'PASS'::TEXT, 'Database connection successful'::TEXT;
    
    -- Check required extensions
    RETURN QUERY
    SELECT 
        'required_extensions'::TEXT,
        CASE WHEN COUNT(*) = 3 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        'Required extensions: ' || string_agg(extname, ', ')::TEXT
    FROM pg_extension
    WHERE extname IN ('postgis', 'uuid-ossp', 'pgcrypto');
    
    -- Check required tables
    RETURN QUERY
    SELECT 
        'required_tables'::TEXT,
        CASE WHEN COUNT(*) = 3 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        'Required tables: ' || string_agg(tablename, ', ')::TEXT
    FROM pg_tables
    WHERE tablename IN ('users', 'orders', 'products');
    
    -- Check database size
    RETURN QUERY
    SELECT 
        'database_size'::TEXT,
        CASE WHEN pg_database_size(current_database()) < 1073741824 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Database size: ' || pg_size_pretty(pg_database_size(current_database()))::TEXT;
END;
$$ LANGUAGE plpgsql;
```

## Deployment Automation

### Deployment Scripts

```python
# deployment/postgres_deployer.py
import psycopg2
import json
import subprocess
import time
from datetime import datetime
import logging

class PostgreSQLDeployer:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def deploy_schema_migration(self, migration_file):
        """Deploy schema migration."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Read migration file
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                # Apply migration
                cur.execute(migration_sql)
                conn.commit()
                
                self.logger.info(f"Migration {migration_file} applied successfully")
                
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def deploy_blue_green(self, new_version):
        """Deploy using blue-green strategy."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Start green deployment
                cur.execute("""
                    INSERT INTO deployment_history (deployment_id, environment, version, created_by)
                    VALUES (%s, 'green', %s, %s)
                """, (f"deploy_{int(time.time())}", new_version, 'deployer'))
                
                # Wait for green environment to be ready
                time.sleep(30)
                
                # Switch traffic to green
                cur.execute("SELECT switch_traffic_to_green()")
                
                conn.commit()
                self.logger.info(f"Blue-green deployment completed for version {new_version}")
                
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def deploy_canary(self, new_version, canary_percentage=10):
        """Deploy using canary strategy."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Start canary deployment
                cur.execute("""
                    INSERT INTO canary_deployments (deployment_id, version, canary_percentage, created_by)
                    VALUES (%s, %s, %s, %s)
                """, (f"canary_{int(time.time())}", new_version, canary_percentage, 'deployer'))
                
                conn.commit()
                self.logger.info(f"Canary deployment started for version {new_version}")
                
        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def rollback_deployment(self, deployment_id):
        """Rollback deployment."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Rollback blue-green deployment
                cur.execute("SELECT rollback_to_blue()")
                
                conn.commit()
                self.logger.info(f"Rollback completed for deployment {deployment_id}")
                
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    deployer = PostgreSQLDeployer({
        'host': 'localhost',
        'database': 'production',
        'user': 'deployer',
        'password': 'deployer_password'
    })
    
    # Deploy schema migration
    deployer.deploy_schema_migration('migrations/001_add_users_table.sql')
    
    # Deploy using blue-green
    deployer.deploy_blue_green('v1.2.0')
```

## Health Checks and Monitoring

### Deployment Health Checks

```sql
-- Create function for deployment health check
CREATE OR REPLACE FUNCTION check_deployment_health()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    message TEXT,
    timestamp TIMESTAMPTZ
) AS $$
BEGIN
    -- Check database connectivity
    RETURN QUERY SELECT 'database_connectivity'::TEXT, 'PASS'::TEXT, 'Database connection successful'::TEXT, CURRENT_TIMESTAMP;
    
    -- Check active connections
    RETURN QUERY
    SELECT 
        'active_connections'::TEXT,
        CASE WHEN COUNT(*) < 100 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Active connections: ' || COUNT(*)::TEXT,
        CURRENT_TIMESTAMP
    FROM pg_stat_activity;
    
    -- Check replication lag
    RETURN QUERY
    SELECT 
        'replication_lag'::TEXT,
        CASE WHEN COALESCE(EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())), 0) < 60 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Replication lag: ' || COALESCE(EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())), 0)::TEXT || ' seconds',
        CURRENT_TIMESTAMP;
    
    -- Check disk space
    RETURN QUERY
    SELECT 
        'disk_space'::TEXT,
        CASE WHEN pg_database_size(current_database()) < 1073741824 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Database size: ' || pg_size_pretty(pg_database_size(current_database())),
        CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
```

### Deployment Monitoring

```python
# monitoring/deployment_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class DeploymentMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_deployment_status(self):
        """Get current deployment status."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        deployment_id,
                        environment,
                        version,
                        status,
                        started_at,
                        completed_at
                    FROM deployment_history
                    ORDER BY started_at DESC
                    LIMIT 10
                """)
                
                deployments = cur.fetchall()
                return deployments
                
        except Exception as e:
            self.logger.error(f"Error getting deployment status: {e}")
            return []
        finally:
            conn.close()
    
    def get_health_metrics(self):
        """Get deployment health metrics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM check_deployment_health()")
                
                health_metrics = cur.fetchall()
                return health_metrics
                
        except Exception as e:
            self.logger.error(f"Error getting health metrics: {e}")
            return []
        finally:
            conn.close()
    
    def generate_deployment_report(self):
        """Generate deployment report."""
        deployment_status = self.get_deployment_status()
        health_metrics = self.get_health_metrics()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'deployment_status': deployment_status,
            'health_metrics': health_metrics
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = DeploymentMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_deployment_report()
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create deployment tracking tables
CREATE TABLE deployment_history (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(50) UNIQUE NOT NULL,
    environment VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
);

-- 2. Create deployment functions
CREATE OR REPLACE FUNCTION switch_traffic_to_green()
RETURNS BOOLEAN AS $$
-- Function implementation
$$ LANGUAGE plpgsql;

-- 3. Deploy using blue-green
INSERT INTO deployment_history (deployment_id, environment, version)
VALUES ('deploy_001', 'green', 'v1.2.0');

-- 4. Switch traffic
SELECT switch_traffic_to_green();
```

### Essential Patterns

```python
# Complete PostgreSQL deployment setup
def setup_postgresql_deployment():
    # 1. Deployment architecture patterns
    # 2. Database migration strategies
    # 3. Environment management
    # 4. Deployment automation
    # 5. Health checks and monitoring
    # 6. Rollback strategies
    # 7. Zero-downtime deployments
    # 8. Deployment monitoring
    
    print("PostgreSQL deployment setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL deployment excellence. Each pattern includes implementation examples, deployment strategies, and real-world usage patterns for enterprise PostgreSQL deployment systems.*
