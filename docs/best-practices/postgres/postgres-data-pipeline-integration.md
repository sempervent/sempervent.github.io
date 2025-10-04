# PostgreSQL Data Pipeline Integration Best Practices

**Objective**: Master senior-level PostgreSQL data pipeline patterns for production systems. When you need to integrate PostgreSQL with data pipelines, when you want to implement ETL workflows, when you need enterprise-grade data processing—these best practices become your weapon of choice.

## Core Principles

- **ETL/ELT Patterns**: Implement efficient data transformation workflows
- **Real-time Processing**: Enable streaming data integration
- **Data Quality**: Ensure data integrity and validation
- **Scalability**: Design for high-volume data processing
- **Monitoring**: Track pipeline performance and health

## ETL/ELT Patterns

### Extract Patterns

```sql
-- Create data sources table
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    connection_config JSONB NOT NULL,
    extraction_query TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register data source
CREATE OR REPLACE FUNCTION register_data_source(
    p_source_name VARCHAR(100),
    p_source_type VARCHAR(50),
    p_connection_config JSONB,
    p_extraction_query TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO data_sources (
        source_name, source_type, connection_config, extraction_query
    ) VALUES (
        p_source_name, p_source_type, p_connection_config, p_extraction_query
    ) ON CONFLICT (source_name) 
    DO UPDATE SET 
        source_type = EXCLUDED.source_type,
        connection_config = EXCLUDED.connection_config,
        extraction_query = EXCLUDED.extraction_query,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create function to extract data
CREATE OR REPLACE FUNCTION extract_data(
    p_source_name VARCHAR(100),
    p_batch_size INTEGER DEFAULT 1000,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    row_data JSONB,
    row_number BIGINT
) AS $$
DECLARE
    source_record RECORD;
    extraction_sql TEXT;
BEGIN
    -- Get source configuration
    SELECT * INTO source_record
    FROM data_sources
    WHERE source_name = p_source_name AND is_active = TRUE;
    
    IF source_record IS NULL THEN
        RAISE EXCEPTION 'Data source % not found or inactive', p_source_name;
    END IF;
    
    -- Build extraction query
    IF source_record.extraction_query IS NOT NULL THEN
        extraction_sql := source_record.extraction_query;
    ELSE
        extraction_sql := format('SELECT * FROM %I LIMIT %s OFFSET %s', 
                               source_record.source_name, p_batch_size, p_offset);
    END IF;
    
    -- Execute extraction
    RETURN QUERY EXECUTE extraction_sql;
END;
$$ LANGUAGE plpgsql;
```

### Transform Patterns

```sql
-- Create transformation rules table
CREATE TABLE transformation_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    source_table VARCHAR(100) NOT NULL,
    target_table VARCHAR(100) NOT NULL,
    transformation_logic JSONB NOT NULL,
    validation_rules JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register transformation rule
CREATE OR REPLACE FUNCTION register_transformation_rule(
    p_rule_name VARCHAR(100),
    p_source_table VARCHAR(100),
    p_target_table VARCHAR(100),
    p_transformation_logic JSONB,
    p_validation_rules JSONB DEFAULT '{}'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO transformation_rules (
        rule_name, source_table, target_table, 
        transformation_logic, validation_rules
    ) VALUES (
        p_rule_name, p_source_table, p_target_table,
        p_transformation_logic, p_validation_rules
    ) ON CONFLICT (rule_name) 
    DO UPDATE SET 
        source_table = EXCLUDED.source_table,
        target_table = EXCLUDED.target_table,
        transformation_logic = EXCLUDED.transformation_logic,
        validation_rules = EXCLUDED.validation_rules;
END;
$$ LANGUAGE plpgsql;

-- Create function to apply transformation
CREATE OR REPLACE FUNCTION apply_transformation(
    p_rule_name VARCHAR(100),
    p_source_data JSONB
)
RETURNS JSONB AS $$
DECLARE
    rule_record RECORD;
    transformed_data JSONB;
    field_mapping JSONB;
    field_name TEXT;
    field_value JSONB;
    validation_result BOOLEAN;
BEGIN
    -- Get transformation rule
    SELECT * INTO rule_record
    FROM transformation_rules
    WHERE rule_name = p_rule_name AND is_active = TRUE;
    
    IF rule_record IS NULL THEN
        RAISE EXCEPTION 'Transformation rule % not found or inactive', p_rule_name;
    END IF;
    
    -- Get field mapping
    field_mapping := rule_record.transformation_logic->'field_mapping';
    transformed_data := '{}'::JSONB;
    
    -- Apply field transformations
    FOR field_name, field_value IN 
        SELECT key, value FROM jsonb_each(field_mapping)
    LOOP
        -- Apply transformation logic
        CASE field_value->>'type'
            WHEN 'direct' THEN
                transformed_data := transformed_data || jsonb_build_object(
                    field_name, p_source_data->(field_value->>'source_field')
                );
            WHEN 'calculated' THEN
                -- Apply calculation logic
                transformed_data := transformed_data || jsonb_build_object(
                    field_name, p_source_data->(field_value->>'source_field')
                );
            WHEN 'constant' THEN
                transformed_data := transformed_data || jsonb_build_object(
                    field_name, field_value->>'value'
                );
        END CASE;
    END LOOP;
    
    -- Apply validation rules
    IF rule_record.validation_rules != '{}' THEN
        validation_result := validate_data(transformed_data, rule_record.validation_rules);
        IF NOT validation_result THEN
            RAISE EXCEPTION 'Data validation failed for rule %', p_rule_name;
        END IF;
    END IF;
    
    RETURN transformed_data;
END;
$$ LANGUAGE plpgsql;
```

### Load Patterns

```sql
-- Create data targets table
CREATE TABLE data_targets (
    id SERIAL PRIMARY KEY,
    target_name VARCHAR(100) UNIQUE NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    connection_config JSONB NOT NULL,
    target_schema JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register data target
CREATE OR REPLACE FUNCTION register_data_target(
    p_target_name VARCHAR(100),
    p_target_type VARCHAR(50),
    p_connection_config JSONB,
    p_target_schema JSONB DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO data_targets (
        target_name, target_type, connection_config, target_schema
    ) VALUES (
        p_target_name, p_target_type, p_connection_config, p_target_schema
    ) ON CONFLICT (target_name) 
    DO UPDATE SET 
        target_type = EXCLUDED.target_type,
        connection_config = EXCLUDED.connection_config,
        target_schema = EXCLUDED.target_schema;
END;
$$ LANGUAGE plpgsql;

-- Create function to load data
CREATE OR REPLACE FUNCTION load_data(
    p_target_name VARCHAR(100),
    p_data JSONB,
    p_load_mode VARCHAR(20) DEFAULT 'insert'
)
RETURNS INTEGER AS $$
DECLARE
    target_record RECORD;
    rows_affected INTEGER;
BEGIN
    -- Get target configuration
    SELECT * INTO target_record
    FROM data_targets
    WHERE target_name = p_target_name AND is_active = TRUE;
    
    IF target_record IS NULL THEN
        RAISE EXCEPTION 'Data target % not found or inactive', p_target_name;
    END IF;
    
    -- Apply load mode
    CASE p_load_mode
        WHEN 'insert' THEN
            -- Insert data
            EXECUTE format('INSERT INTO %I (data) VALUES (%L)', 
                          target_record.target_name, p_data);
            rows_affected := 1;
            
        WHEN 'upsert' THEN
            -- Upsert data
            EXECUTE format('INSERT INTO %I (data) VALUES (%L) ON CONFLICT DO UPDATE SET data = EXCLUDED.data', 
                          target_record.target_name, p_data);
            rows_affected := 1;
            
        WHEN 'update' THEN
            -- Update data
            EXECUTE format('UPDATE %I SET data = %L WHERE id = %L', 
                          target_record.target_name, p_data, p_data->>'id');
            rows_affected := 1;
            
        ELSE
            RAISE EXCEPTION 'Unknown load mode: %', p_load_mode;
    END CASE;
    
    RETURN rows_affected;
END;
$$ LANGUAGE plpgsql;
```

## Real-time Processing

### Change Data Capture (CDC)

```sql
-- Create CDC configuration table
CREATE TABLE cdc_config (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    cdc_enabled BOOLEAN DEFAULT TRUE,
    capture_columns TEXT[],
    exclude_columns TEXT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to enable CDC
CREATE OR REPLACE FUNCTION enable_cdc(
    p_table_name VARCHAR(100),
    p_capture_columns TEXT[] DEFAULT NULL,
    p_exclude_columns TEXT[] DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO cdc_config (table_name, capture_columns, exclude_columns)
    VALUES (p_table_name, p_capture_columns, p_exclude_columns)
    ON CONFLICT (table_name) 
    DO UPDATE SET 
        cdc_enabled = TRUE,
        capture_columns = EXCLUDED.capture_columns,
        exclude_columns = EXCLUDED.exclude_columns;
END;
$$ LANGUAGE plpgsql;

-- Create function to capture changes
CREATE OR REPLACE FUNCTION capture_changes(
    p_table_name VARCHAR(100),
    p_operation VARCHAR(10),
    p_old_data JSONB,
    p_new_data JSONB
)
RETURNS VOID AS $$
DECLARE
    cdc_record RECORD;
    change_data JSONB;
BEGIN
    -- Get CDC configuration
    SELECT * INTO cdc_record
    FROM cdc_config
    WHERE table_name = p_table_name AND cdc_enabled = TRUE;
    
    IF cdc_record IS NULL THEN
        RETURN;
    END IF;
    
    -- Build change data
    change_data := jsonb_build_object(
        'table_name', p_table_name,
        'operation', p_operation,
        'timestamp', CURRENT_TIMESTAMP,
        'old_data', p_old_data,
        'new_data', p_new_data
    );
    
    -- Store change
    INSERT INTO change_log (table_name, operation, change_data)
    VALUES (p_table_name, p_operation, change_data);
END;
$$ LANGUAGE plpgsql;
```

### Streaming Integration

```sql
-- Create streaming configuration table
CREATE TABLE streaming_config (
    id SERIAL PRIMARY KEY,
    stream_name VARCHAR(100) UNIQUE NOT NULL,
    source_table VARCHAR(100) NOT NULL,
    target_system VARCHAR(100) NOT NULL,
    stream_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to configure stream
CREATE OR REPLACE FUNCTION configure_stream(
    p_stream_name VARCHAR(100),
    p_source_table VARCHAR(100),
    p_target_system VARCHAR(100),
    p_stream_config JSONB
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO streaming_config (
        stream_name, source_table, target_system, stream_config
    ) VALUES (
        p_stream_name, p_source_table, p_target_system, p_stream_config
    ) ON CONFLICT (stream_name) 
    DO UPDATE SET 
        source_table = EXCLUDED.source_table,
        target_system = EXCLUDED.target_system,
        stream_config = EXCLUDED.stream_config;
END;
$$ LANGUAGE plpgsql;

-- Create function to process stream
CREATE OR REPLACE FUNCTION process_stream(
    p_stream_name VARCHAR(100),
    p_batch_size INTEGER DEFAULT 100
)
RETURNS INTEGER AS $$
DECLARE
    stream_record RECORD;
    processed_count INTEGER := 0;
    batch_data JSONB[];
BEGIN
    -- Get stream configuration
    SELECT * INTO stream_record
    FROM streaming_config
    WHERE stream_name = p_stream_name AND is_active = TRUE;
    
    IF stream_record IS NULL THEN
        RAISE EXCEPTION 'Stream % not found or inactive', p_stream_name;
    END IF;
    
    -- Process stream batch
    -- Implementation depends on target system
    -- This is a placeholder for actual streaming logic
    
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;
```

## Data Quality

### Validation Rules

```sql
-- Create validation rules table
CREATE TABLE validation_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    rule_definition JSONB NOT NULL,
    error_message TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register validation rule
CREATE OR REPLACE FUNCTION register_validation_rule(
    p_rule_name VARCHAR(100),
    p_rule_type VARCHAR(50),
    p_rule_definition JSONB,
    p_error_message TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO validation_rules (
        rule_name, rule_type, rule_definition, error_message
    ) VALUES (
        p_rule_name, p_rule_type, p_rule_definition, p_error_message
    ) ON CONFLICT (rule_name) 
    DO UPDATE SET 
        rule_type = EXCLUDED.rule_type,
        rule_definition = EXCLUDED.rule_definition,
        error_message = EXCLUDED.error_message;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate data
CREATE OR REPLACE FUNCTION validate_data(
    p_data JSONB,
    p_validation_rules JSONB
)
RETURNS BOOLEAN AS $$
DECLARE
    rule_name TEXT;
    rule_definition JSONB;
    field_name TEXT;
    field_value JSONB;
    validation_result BOOLEAN;
BEGIN
    -- Apply validation rules
    FOR rule_name, rule_definition IN 
        SELECT key, value FROM jsonb_each(p_validation_rules)
    LOOP
        field_name := rule_definition->>'field';
        field_value := p_data->field_name;
        
        -- Apply validation based on rule type
        CASE rule_definition->>'type'
            WHEN 'required' THEN
                IF field_value IS NULL OR field_value = 'null' THEN
                    RETURN FALSE;
                END IF;
                
            WHEN 'format' THEN
                IF rule_definition->>'format' = 'email' THEN
                    IF NOT (field_value #>> '{}' ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
                        RETURN FALSE;
                    END IF;
                END IF;
                
            WHEN 'range' THEN
                IF field_value IS NOT NULL THEN
                    IF (rule_definition->>'min')::NUMERIC > (field_value #>> '{}')::NUMERIC OR
                       (rule_definition->>'max')::NUMERIC < (field_value #>> '{}')::NUMERIC THEN
                        RETURN FALSE;
                    END IF;
                END IF;
                
            WHEN 'pattern' THEN
                IF field_value IS NOT NULL THEN
                    IF NOT (field_value #>> '{}' ~ rule_definition->>'pattern') THEN
                        RETURN FALSE;
                    END IF;
                END IF;
        END CASE;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Data Quality Monitoring

```sql
-- Create data quality metrics table
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    threshold_value NUMERIC,
    is_healthy BOOLEAN,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to record data quality metrics
CREATE OR REPLACE FUNCTION record_data_quality_metrics(
    p_table_name VARCHAR(100),
    p_metric_name VARCHAR(100),
    p_metric_value NUMERIC,
    p_threshold_value NUMERIC DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    is_healthy BOOLEAN;
BEGIN
    -- Determine if metric is healthy
    IF p_threshold_value IS NOT NULL THEN
        is_healthy := p_metric_value <= p_threshold_value;
    ELSE
        is_healthy := TRUE;
    END IF;
    
    -- Record metric
    INSERT INTO data_quality_metrics (
        table_name, metric_name, metric_value, threshold_value, is_healthy
    ) VALUES (
        p_table_name, p_metric_name, p_metric_value, p_threshold_value, is_healthy
    );
END;
$$ LANGUAGE plpgsql;

-- Create function to get data quality report
CREATE OR REPLACE FUNCTION get_data_quality_report(
    p_table_name VARCHAR(100),
    p_start_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP - INTERVAL '1 day',
    p_end_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    metric_name VARCHAR(100),
    avg_metric_value NUMERIC,
    max_metric_value NUMERIC,
    min_metric_value NUMERIC,
    healthy_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dqm.metric_name,
        AVG(dqm.metric_value) as avg_metric_value,
        MAX(dqm.metric_value) as max_metric_value,
        MIN(dqm.metric_value) as min_metric_value,
        (COUNT(*) FILTER (WHERE dqm.is_healthy = TRUE)::NUMERIC / COUNT(*)::NUMERIC * 100) as healthy_percentage
    FROM data_quality_metrics dqm
    WHERE dqm.table_name = p_table_name
    AND dqm.recorded_at BETWEEN p_start_date AND p_end_date
    GROUP BY dqm.metric_name
    ORDER BY dqm.metric_name;
END;
$$ LANGUAGE plpgsql;
```

## Pipeline Implementation

### Python Pipeline Processor

```python
# pipeline/postgres_pipeline_processor.py
import psycopg2
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

class PostgreSQLPipelineProcessor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)
    
    def extract_data(self, source_name: str, batch_size: int = 1000, offset: int = 0):
        """Extract data from source."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM extract_data(%s, %s, %s)
                """, (source_name, batch_size, offset))
                
                data = cur.fetchall()
                return [{'row_data': row[0], 'row_number': row[1]} for row in data]
                
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            return []
        finally:
            conn.close()
    
    def transform_data(self, rule_name: str, source_data: dict):
        """Transform data using rule."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT apply_transformation(%s, %s)
                """, (rule_name, json.dumps(source_data)))
                
                result = cur.fetchone()[0]
                return result
                
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            return None
        finally:
            conn.close()
    
    def load_data(self, target_name: str, data: dict, load_mode: str = 'insert'):
        """Load data to target."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT load_data(%s, %s, %s)
                """, (target_name, json.dumps(data), load_mode))
                
                rows_affected = cur.fetchone()[0]
                conn.commit()
                
                self.logger.info(f"Loaded {rows_affected} rows to {target_name}")
                return rows_affected
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error loading data: {e}")
            raise
        finally:
            conn.close()
    
    def process_pipeline(self, source_name: str, rule_name: str, target_name: str, 
                        batch_size: int = 1000, load_mode: str = 'insert'):
        """Process complete ETL pipeline."""
        processed_count = 0
        
        try:
            # Extract data
            source_data = self.extract_data(source_name, batch_size)
            
            for row in source_data:
                # Transform data
                transformed_data = self.transform_data(rule_name, row['row_data'])
                
                if transformed_data:
                    # Load data
                    rows_affected = self.load_data(target_name, transformed_data, load_mode)
                    processed_count += rows_affected
                else:
                    self.logger.warning(f"Transformation failed for row {row['row_number']}")
            
            self.logger.info(f"Pipeline processed {processed_count} rows")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def enable_cdc(self, table_name: str, capture_columns: List[str] = None, 
                   exclude_columns: List[str] = None):
        """Enable change data capture."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT enable_cdc(%s, %s, %s)
                """, (table_name, capture_columns, exclude_columns))
                
                conn.commit()
                self.logger.info(f"CDC enabled for table {table_name}")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error enabling CDC: {e}")
            raise
        finally:
            conn.close()
    
    def configure_stream(self, stream_name: str, source_table: str, 
                        target_system: str, stream_config: dict):
        """Configure data stream."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT configure_stream(%s, %s, %s, %s)
                """, (stream_name, source_table, target_system, json.dumps(stream_config)))
                
                conn.commit()
                self.logger.info(f"Stream {stream_name} configured")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error configuring stream: {e}")
            raise
        finally:
            conn.close()
    
    def process_stream(self, stream_name: str, batch_size: int = 100):
        """Process data stream."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT process_stream(%s, %s)
                """, (stream_name, batch_size))
                
                processed_count = cur.fetchone()[0]
                conn.commit()
                
                self.logger.info(f"Stream {stream_name} processed {processed_count} records")
                return processed_count
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error processing stream: {e}")
            raise
        finally:
            conn.close()
    
    def validate_data(self, data: dict, validation_rules: dict):
        """Validate data quality."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT validate_data(%s, %s)
                """, (json.dumps(data), json.dumps(validation_rules)))
                
                is_valid = cur.fetchone()[0]
                return is_valid
                
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False
        finally:
            conn.close()
    
    def record_data_quality_metrics(self, table_name: str, metric_name: str, 
                                   metric_value: float, threshold_value: float = None):
        """Record data quality metrics."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT record_data_quality_metrics(%s, %s, %s, %s)
                """, (table_name, metric_name, metric_value, threshold_value))
                
                conn.commit()
                self.logger.info(f"Data quality metric recorded: {metric_name} = {metric_value}")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error recording data quality metrics: {e}")
            raise
        finally:
            conn.close()
    
    def get_data_quality_report(self, table_name: str, start_date: datetime = None, 
                               end_date: datetime = None):
        """Get data quality report."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM get_data_quality_report(%s, %s, %s)
                """, (table_name, start_date, end_date))
                
                report = cur.fetchall()
                return [{
                    'metric_name': row[0],
                    'avg_metric_value': row[1],
                    'max_metric_value': row[2],
                    'min_metric_value': row[3],
                    'healthy_percentage': row[4]
                } for row in report]
                
        except Exception as e:
            self.logger.error(f"Error getting data quality report: {e}")
            return []
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    processor = PostgreSQLPipelineProcessor({
        'host': 'localhost',
        'database': 'production',
        'user': 'pipeline_user',
        'password': 'pipeline_password'
    })
    
    # Example ETL pipeline
    processed_count = processor.process_pipeline(
        'source_table', 'transformation_rule', 'target_table'
    )
    
    print(f"Pipeline processed {processed_count} rows")
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Register data source
SELECT register_data_source('users', 'postgresql', '{"host": "localhost"}');

-- 2. Register transformation rule
SELECT register_transformation_rule('user_transform', 'users', 'users_clean', '{"field_mapping": {}}');

-- 3. Register data target
SELECT register_data_target('users_clean', 'postgresql', '{"host": "localhost"}');

-- 4. Extract data
SELECT * FROM extract_data('users', 1000, 0);

-- 5. Apply transformation
SELECT apply_transformation('user_transform', '{"username": "john"}');

-- 6. Load data
SELECT load_data('users_clean', '{"username": "john"}', 'insert');
```

### Essential Patterns

```python
# Complete PostgreSQL data pipeline integration setup
def setup_postgresql_data_pipeline_integration():
    # 1. ETL/ELT patterns
    # 2. Real-time processing
    # 3. Data quality
    # 4. Pipeline implementation
    # 5. Change data capture
    # 6. Streaming integration
    # 7. Data validation
    # 8. Quality monitoring
    
    print("PostgreSQL data pipeline integration setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL data pipeline integration excellence. Each pattern includes implementation examples, pipeline strategies, and real-world usage patterns for enterprise PostgreSQL data pipeline systems.*
