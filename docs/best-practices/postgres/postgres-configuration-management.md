# PostgreSQL Configuration Management Best Practices

**Objective**: Master senior-level PostgreSQL configuration management patterns for production systems. When you need to manage database configurations, when you want to implement configuration as code, when you need enterprise-grade configuration strategies—these best practices become your weapon of choice.

## Core Principles

- **Configuration as Code**: Version control all configuration changes
- **Environment Parity**: Maintain consistency across environments
- **Automation**: Automate configuration deployment
- **Validation**: Validate configuration before deployment
- **Rollback**: Implement safe configuration rollback

## Configuration Management Patterns

### Configuration Schema Design

```sql
-- Create configuration management tables
CREATE TABLE configuration_settings (
    id SERIAL PRIMARY KEY,
    setting_name VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    setting_type VARCHAR(20) NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    environment VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE configuration_history (
    id SERIAL PRIMARY KEY,
    setting_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT NOT NULL,
    changed_by VARCHAR(100) NOT NULL,
    change_reason TEXT,
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE configuration_templates (
    id SERIAL PRIMARY KEY,
    template_name VARCHAR(100) UNIQUE NOT NULL,
    environment VARCHAR(20) NOT NULL,
    settings JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Configuration Functions

```sql
-- Create function to get configuration value
CREATE OR REPLACE FUNCTION get_config_value(
    p_setting_name VARCHAR(100),
    p_environment VARCHAR(20) DEFAULT 'production'
)
RETURNS TEXT AS $$
DECLARE
    config_value TEXT;
BEGIN
    SELECT setting_value INTO config_value
    FROM configuration_settings
    WHERE setting_name = p_setting_name 
    AND environment = p_environment;
    
    RETURN config_value;
END;
$$ LANGUAGE plpgsql;

-- Create function to set configuration value
CREATE OR REPLACE FUNCTION set_config_value(
    p_setting_name VARCHAR(100),
    p_setting_value TEXT,
    p_setting_type VARCHAR(20),
    p_environment VARCHAR(20),
    p_description TEXT DEFAULT NULL,
    p_is_sensitive BOOLEAN DEFAULT FALSE,
    p_changed_by VARCHAR(100) DEFAULT current_user
)
RETURNS VOID AS $$
DECLARE
    old_value TEXT;
BEGIN
    -- Get old value for history
    SELECT setting_value INTO old_value
    FROM configuration_settings
    WHERE setting_name = p_setting_name AND environment = p_environment;
    
    -- Update or insert configuration
    INSERT INTO configuration_settings (
        setting_name, setting_value, setting_type, description, 
        is_sensitive, environment
    ) VALUES (
        p_setting_name, p_setting_value, p_setting_type, p_description,
        p_is_sensitive, p_environment
    ) ON CONFLICT (setting_name, environment) 
    DO UPDATE SET 
        setting_value = EXCLUDED.setting_value,
        setting_type = EXCLUDED.setting_type,
        description = EXCLUDED.description,
        is_sensitive = EXCLUDED.is_sensitive,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Record change in history
    INSERT INTO configuration_history (
        setting_name, old_value, new_value, changed_by, change_reason
    ) VALUES (
        p_setting_name, old_value, p_setting_value, p_changed_by, 
        'Configuration updated'
    );
END;
$$ LANGUAGE plpgsql;
```

## Environment-Specific Configuration

### Environment Configuration Management

```sql
-- Create function to apply configuration template
CREATE OR REPLACE FUNCTION apply_configuration_template(
    p_template_name VARCHAR(100),
    p_environment VARCHAR(20),
    p_applied_by VARCHAR(100) DEFAULT current_user
)
RETURNS INTEGER AS $$
DECLARE
    template_settings JSONB;
    setting_key TEXT;
    setting_value TEXT;
    applied_count INTEGER := 0;
BEGIN
    -- Get template settings
    SELECT settings INTO template_settings
    FROM configuration_templates
    WHERE template_name = p_template_name AND environment = p_environment;
    
    IF template_settings IS NULL THEN
        RAISE EXCEPTION 'Template % not found for environment %', p_template_name, p_environment;
    END IF;
    
    -- Apply each setting from template
    FOR setting_key, setting_value IN 
        SELECT key, value FROM jsonb_each_text(template_settings)
    LOOP
        PERFORM set_config_value(
            setting_key, 
            setting_value, 
            'string', 
            p_environment, 
            'Applied from template ' || p_template_name,
            FALSE,
            p_applied_by
        );
        applied_count := applied_count + 1;
    END LOOP;
    
    RETURN applied_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate configuration
CREATE OR REPLACE FUNCTION validate_configuration(p_environment VARCHAR(20))
RETURNS TABLE (
    setting_name VARCHAR(100),
    status TEXT,
    message TEXT
) AS $$
BEGIN
    -- Check required settings
    RETURN QUERY
    SELECT 
        'max_connections'::VARCHAR(100),
        CASE WHEN get_config_value('max_connections', p_environment) IS NOT NULL 
             THEN 'PASS' ELSE 'FAIL' END::TEXT,
        CASE WHEN get_config_value('max_connections', p_environment) IS NOT NULL 
             THEN 'max_connections is configured' 
             ELSE 'max_connections is missing' END::TEXT;
    
    RETURN QUERY
    SELECT 
        'shared_buffers'::VARCHAR(100),
        CASE WHEN get_config_value('shared_buffers', p_environment) IS NOT NULL 
             THEN 'PASS' ELSE 'FAIL' END::TEXT,
        CASE WHEN get_config_value('shared_buffers', p_environment) IS NOT NULL 
             THEN 'shared_buffers is configured' 
             ELSE 'shared_buffers is missing' END::TEXT;
    
    RETURN QUERY
    SELECT 
        'wal_level'::VARCHAR(100),
        CASE WHEN get_config_value('wal_level', p_environment) IS NOT NULL 
             THEN 'PASS' ELSE 'FAIL' END::TEXT,
        CASE WHEN get_config_value('wal_level', p_environment) IS NOT NULL 
             THEN 'wal_level is configured' 
             ELSE 'wal_level is missing' END::TEXT;
END;
$$ LANGUAGE plpgsql;
```

## Configuration Automation

### Configuration Deployment

```python
# configuration/config_manager.py
import psycopg2
import json
import yaml
from datetime import datetime
import logging

class PostgreSQLConfigManager:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_config_from_file(self, config_file):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def apply_configuration(self, config_data, environment):
        """Apply configuration to database."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                for setting_name, setting_value in config_data.items():
                    cur.execute("""
                        SELECT set_config_value(%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        setting_name,
                        str(setting_value),
                        'string',
                        environment,
                        f'Applied from config file',
                        False,
                        'config_manager'
                    ))
                
                conn.commit()
                self.logger.info(f"Configuration applied for environment {environment}")
                
        except Exception as e:
            self.logger.error(f"Configuration application failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_configuration(self, environment):
        """Get current configuration for environment."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT setting_name, setting_value, setting_type, description
                    FROM configuration_settings
                    WHERE environment = %s
                    ORDER BY setting_name
                """, (environment,))
                
                config = cur.fetchall()
                return config
                
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return []
        finally:
            conn.close()
    
    def validate_configuration(self, environment):
        """Validate configuration for environment."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM validate_configuration(%s)", (environment,))
                
                validation_results = cur.fetchall()
                return validation_results
                
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return []
        finally:
            conn.close()
    
    def rollback_configuration(self, environment, rollback_to_datetime):
        """Rollback configuration to specific datetime."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Get configuration state at rollback time
                cur.execute("""
                    SELECT DISTINCT ON (setting_name) setting_name, old_value
                    FROM configuration_history
                    WHERE changed_at <= %s
                    ORDER BY setting_name, changed_at DESC
                """, (rollback_to_datetime,))
                
                rollback_settings = cur.fetchall()
                
                # Apply rollback settings
                for setting_name, old_value in rollback_settings:
                    if old_value is not None:
                        cur.execute("""
                            SELECT set_config_value(%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            setting_name,
                            old_value,
                            'string',
                            environment,
                            f'Rollback to {rollback_to_datetime}',
                            False,
                            'config_manager'
                        ))
                
                conn.commit()
                self.logger.info(f"Configuration rolled back for environment {environment}")
                
        except Exception as e:
            self.logger.error(f"Configuration rollback failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    config_manager = PostgreSQLConfigManager({
        'host': 'localhost',
        'database': 'production',
        'user': 'config_user',
        'password': 'config_password'
    })
    
    # Load and apply configuration
    config_data = config_manager.load_config_from_file('config/production.yaml')
    config_manager.apply_configuration(config_data, 'production')
    
    # Validate configuration
    validation_results = config_manager.validate_configuration('production')
    print(json.dumps(validation_results, indent=2))
```

## Configuration Templates

### Template Management

```sql
-- Create function to create configuration template
CREATE OR REPLACE FUNCTION create_configuration_template(
    p_template_name VARCHAR(100),
    p_environment VARCHAR(20),
    p_settings JSONB,
    p_description TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO configuration_templates (
        template_name, environment, settings, description
    ) VALUES (
        p_template_name, p_environment, p_settings, p_description
    ) ON CONFLICT (template_name, environment) 
    DO UPDATE SET 
        settings = EXCLUDED.settings,
        description = EXCLUDED.description;
END;
$$ LANGUAGE plpgsql;

-- Create function to get configuration template
CREATE OR REPLACE FUNCTION get_configuration_template(
    p_template_name VARCHAR(100),
    p_environment VARCHAR(20)
)
RETURNS JSONB AS $$
DECLARE
    template_settings JSONB;
BEGIN
    SELECT settings INTO template_settings
    FROM configuration_templates
    WHERE template_name = p_template_name AND environment = p_environment;
    
    RETURN template_settings;
END;
$$ LANGUAGE plpgsql;
```

### Configuration Validation

```sql
-- Create function to validate configuration values
CREATE OR REPLACE FUNCTION validate_configuration_value(
    p_setting_name VARCHAR(100),
    p_setting_value TEXT,
    p_setting_type VARCHAR(20)
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Validate based on setting type
    CASE p_setting_type
        WHEN 'integer' THEN
            RETURN p_setting_value ~ '^\d+$';
        WHEN 'boolean' THEN
            RETURN p_setting_value IN ('true', 'false', 'on', 'off', 'yes', 'no');
        WHEN 'size' THEN
            RETURN p_setting_value ~ '^\d+[KMGT]?B?$';
        WHEN 'time' THEN
            RETURN p_setting_value ~ '^\d+[smhd]?$';
        ELSE
            RETURN TRUE; -- String type, always valid
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate all configuration
CREATE OR REPLACE FUNCTION validate_all_configuration(p_environment VARCHAR(20))
RETURNS TABLE (
    setting_name VARCHAR(100),
    setting_value TEXT,
    setting_type VARCHAR(20),
    is_valid BOOLEAN,
    error_message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.setting_name,
        cs.setting_value,
        cs.setting_type,
        validate_configuration_value(cs.setting_name, cs.setting_value, cs.setting_type) as is_valid,
        CASE 
            WHEN validate_configuration_value(cs.setting_name, cs.setting_value, cs.setting_type) THEN NULL
            ELSE 'Invalid value for ' || cs.setting_type || ' type'
        END as error_message
    FROM configuration_settings cs
    WHERE cs.environment = p_environment;
END;
$$ LANGUAGE plpgsql;
```

## Configuration Monitoring

### Configuration Change Tracking

```sql
-- Create function to get configuration changes
CREATE OR REPLACE FUNCTION get_configuration_changes(
    p_environment VARCHAR(20),
    p_start_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP - INTERVAL '7 days',
    p_end_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    setting_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(100),
    change_reason TEXT,
    changed_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ch.setting_name,
        ch.old_value,
        ch.new_value,
        ch.changed_by,
        ch.change_reason,
        ch.changed_at
    FROM configuration_history ch
    WHERE ch.changed_at BETWEEN p_start_date AND p_end_date
    ORDER BY ch.changed_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to get configuration drift
CREATE OR REPLACE FUNCTION detect_configuration_drift(
    p_environment VARCHAR(20),
    p_template_name VARCHAR(100)
)
RETURNS TABLE (
    setting_name VARCHAR(100),
    current_value TEXT,
    template_value TEXT,
    is_drifted BOOLEAN
) AS $$
DECLARE
    template_settings JSONB;
BEGIN
    -- Get template settings
    SELECT settings INTO template_settings
    FROM configuration_templates
    WHERE template_name = p_template_name AND environment = p_environment;
    
    -- Compare current settings with template
    RETURN QUERY
    SELECT 
        cs.setting_name,
        cs.setting_value as current_value,
        template_settings ->> cs.setting_name as template_value,
        (cs.setting_value != (template_settings ->> cs.setting_name)) as is_drifted
    FROM configuration_settings cs
    WHERE cs.environment = p_environment
    AND template_settings ? cs.setting_name;
END;
$$ LANGUAGE plpgsql;
```

### Configuration Monitoring

```python
# monitoring/config_monitor.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging

class ConfigurationMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_configuration_changes(self, environment, days=7):
        """Get configuration changes for the last N days."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM get_configuration_changes(%s, %s, %s)
                """, (environment, datetime.now() - timedelta(days=days), datetime.now()))
                
                changes = cur.fetchall()
                return changes
                
        except Exception as e:
            self.logger.error(f"Error getting configuration changes: {e}")
            return []
        finally:
            conn.close()
    
    def detect_configuration_drift(self, environment, template_name):
        """Detect configuration drift from template."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM detect_configuration_drift(%s, %s)
                """, (environment, template_name))
                
                drift = cur.fetchall()
                return drift
                
        except Exception as e:
            self.logger.error(f"Error detecting configuration drift: {e}")
            return []
        finally:
            conn.close()
    
    def get_configuration_summary(self, environment):
        """Get configuration summary for environment."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_settings,
                        COUNT(*) FILTER (WHERE is_sensitive = true) as sensitive_settings,
                        COUNT(*) FILTER (WHERE updated_at > CURRENT_DATE - INTERVAL '7 days') as recent_changes
                    FROM configuration_settings
                    WHERE environment = %s
                """, (environment,))
                
                summary = cur.fetchone()
                return summary
                
        except Exception as e:
            self.logger.error(f"Error getting configuration summary: {e}")
            return None
        finally:
            conn.close()
    
    def generate_configuration_report(self, environment):
        """Generate comprehensive configuration report."""
        changes = self.get_configuration_changes(environment, 7)
        drift = self.detect_configuration_drift(environment, 'production_template')
        summary = self.get_configuration_summary(environment)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'environment': environment,
            'summary': summary,
            'recent_changes': changes,
            'configuration_drift': drift
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = ConfigurationMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_configuration_report('production')
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create configuration tables
CREATE TABLE configuration_settings (
    id SERIAL PRIMARY KEY,
    setting_name VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    environment VARCHAR(20) NOT NULL
);

-- 2. Create configuration functions
CREATE OR REPLACE FUNCTION get_config_value(
    p_setting_name VARCHAR(100),
    p_environment VARCHAR(20)
) RETURNS TEXT AS $$
-- Function implementation
$$ LANGUAGE plpgsql;

-- 3. Set configuration values
SELECT set_config_value('max_connections', '200', 'integer', 'production');

-- 4. Get configuration values
SELECT get_config_value('max_connections', 'production');
```

### Essential Patterns

```python
# Complete PostgreSQL configuration management setup
def setup_postgresql_configuration_management():
    # 1. Configuration schema design
    # 2. Environment-specific configuration
    # 3. Configuration automation
    # 4. Configuration templates
    # 5. Configuration validation
    # 6. Configuration monitoring
    # 7. Change tracking
    # 8. Drift detection
    
    print("PostgreSQL configuration management setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL configuration management excellence. Each pattern includes implementation examples, configuration strategies, and real-world usage patterns for enterprise PostgreSQL configuration systems.*
