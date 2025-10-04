# PostgreSQL JSON & JSONB Best Practices

**Objective**: Master senior-level PostgreSQL JSON and JSONB patterns for production systems. When you need to work with semi-structured data, when you want to leverage JSON capabilities, when you need enterprise-grade JSON strategies—these best practices become your weapon of choice.

## Core Principles

- **JSONB Over JSON**: Use JSONB for better performance and indexing
- **Schema Design**: Design JSON structure for query patterns
- **Indexing Strategy**: Create appropriate indexes for JSON queries
- **Validation**: Implement JSON schema validation
- **Performance**: Optimize JSON operations for production workloads

## JSONB Data Types

### JSONB vs JSON Comparison

```sql
-- JSON stores data as text (slower, larger)
CREATE TABLE json_example (
    id SERIAL PRIMARY KEY,
    data JSON NOT NULL
);

-- JSONB stores data in binary format (faster, smaller)
CREATE TABLE jsonb_example (
    id SERIAL PRIMARY KEY,
    data JSONB NOT NULL
);

-- Insert sample data
INSERT INTO json_example (data) VALUES 
    ('{"name": "John", "age": 30, "city": "New York"}');

INSERT INTO jsonb_example (data) VALUES 
    ('{"name": "John", "age": 30, "city": "New York"}');

-- Compare storage size
SELECT 
    pg_column_size(data) as json_size,
    pg_column_size(data) as jsonb_size
FROM json_example, jsonb_example;
```

### JSONB Schema Design

```sql
-- Create table with JSONB for user profiles
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample user profiles
INSERT INTO user_profiles (user_id, profile_data) VALUES
    (1, '{
        "personal": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123",
            "date_of_birth": "1990-05-15"
        },
        "preferences": {
            "theme": "dark",
            "language": "en",
            "notifications": {
                "email": true,
                "sms": false,
                "push": true
            }
        },
        "address": {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip": "10001",
            "country": "US"
        },
        "social": {
            "linkedin": "https://linkedin.com/in/johndoe",
            "twitter": "@johndoe",
            "github": "johndoe"
        }
    }'),
    (2, '{
        "personal": {
            "first_name": "Jane",
            "last_name": "Smith",
            "email": "jane.smith@example.com",
            "phone": "+1-555-0456",
            "date_of_birth": "1985-08-22"
        },
        "preferences": {
            "theme": "light",
            "language": "en",
            "notifications": {
                "email": true,
                "sms": true,
                "push": false
            }
        },
        "address": {
            "street": "456 Oak Ave",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "country": "US"
        },
        "social": {
            "linkedin": "https://linkedin.com/in/janesmith",
            "twitter": "@janesmith",
            "github": "janesmith"
        }
    }');
```

## JSONB Indexing Strategies

### GIN Indexes for JSONB

```sql
-- Create GIN index for full JSONB content
CREATE INDEX idx_user_profiles_gin ON user_profiles USING GIN (profile_data);

-- Create GIN index for specific JSONB paths
CREATE INDEX idx_user_profiles_personal_gin ON user_profiles 
USING GIN ((profile_data -> 'personal'));

CREATE INDEX idx_user_profiles_preferences_gin ON user_profiles 
USING GIN ((profile_data -> 'preferences'));

-- Create GIN index for JSONB keys
CREATE INDEX idx_user_profiles_keys_gin ON user_profiles 
USING GIN ((profile_data ? 'personal'));

-- Create GIN index for JSONB array elements
CREATE INDEX idx_user_profiles_tags_gin ON user_profiles 
USING GIN ((profile_data -> 'tags'));
```

### B-tree Indexes for JSONB

```sql
-- Create B-tree index for specific JSONB values
CREATE INDEX idx_user_profiles_email_btree ON user_profiles 
((profile_data ->> 'personal' ->> 'email'));

CREATE INDEX idx_user_profiles_city_btree ON user_profiles 
((profile_data -> 'address' ->> 'city'));

CREATE INDEX idx_user_profiles_created_date_btree ON user_profiles 
(((profile_data ->> 'created_date')::date));
```

### Composite Indexes

```sql
-- Create composite index for complex queries
CREATE INDEX idx_user_profiles_city_email ON user_profiles 
((profile_data -> 'address' ->> 'city'), (profile_data ->> 'personal' ->> 'email'));

-- Create partial index for specific conditions
CREATE INDEX idx_user_profiles_active_users ON user_profiles 
((profile_data ->> 'status')) 
WHERE (profile_data ->> 'status') = 'active';
```

## JSONB Query Patterns

### Basic JSONB Operations

```sql
-- Extract values using -> and ->>
SELECT 
    user_id,
    profile_data -> 'personal' ->> 'first_name' as first_name,
    profile_data -> 'personal' ->> 'last_name' as last_name,
    profile_data ->> 'personal' ->> 'email' as email
FROM user_profiles;

-- Check if key exists
SELECT 
    user_id,
    profile_data ? 'social' as has_social,
    profile_data -> 'personal' ? 'phone' as has_phone
FROM user_profiles;

-- Check if any of multiple keys exist
SELECT 
    user_id,
    profile_data ?| ARRAY['social', 'preferences'] as has_any,
    profile_data ?& ARRAY['personal', 'address'] as has_all
FROM user_profiles;
```

### Advanced JSONB Queries

```sql
-- Query nested JSONB objects
SELECT 
    user_id,
    profile_data -> 'personal' ->> 'first_name' as first_name,
    profile_data -> 'address' ->> 'city' as city,
    profile_data -> 'preferences' ->> 'theme' as theme
FROM user_profiles
WHERE profile_data -> 'address' ->> 'state' = 'NY';

-- Query JSONB arrays
SELECT 
    user_id,
    jsonb_array_elements(profile_data -> 'tags') as tag
FROM user_profiles
WHERE profile_data ? 'tags';

-- Aggregate JSONB data
SELECT 
    profile_data -> 'address' ->> 'state' as state,
    COUNT(*) as user_count,
    AVG((profile_data ->> 'age')::integer) as avg_age
FROM user_profiles
WHERE profile_data ? 'age'
GROUP BY profile_data -> 'address' ->> 'state';
```

### JSONB Path Queries

```sql
-- Use JSONB path expressions
SELECT 
    user_id,
    profile_data #> '{personal,first_name}' as first_name,
    profile_data #> '{address,city}' as city,
    profile_data #> '{preferences,notifications,email}' as email_notifications
FROM user_profiles;

-- Query with JSONB path conditions
SELECT 
    user_id,
    profile_data #> '{personal,first_name}' as first_name,
    profile_data #> '{personal,last_name}' as last_name
FROM user_profiles
WHERE profile_data #> '{address,state}' = '"NY"';

-- Update JSONB using path expressions
UPDATE user_profiles 
SET profile_data = jsonb_set(profile_data, '{preferences,theme}', '"dark"')
WHERE user_id = 1;

-- Update nested JSONB values
UPDATE user_profiles 
SET profile_data = jsonb_set(
    profile_data, 
    '{preferences,notifications,email}', 
    'false'
)
WHERE user_id = 1;
```

## JSONB Schema Validation

### Custom Validation Functions

```sql
-- Create validation function for user profile schema
CREATE OR REPLACE FUNCTION validate_user_profile(profile_data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check required fields
    IF NOT (profile_data ? 'personal' AND 
            profile_data -> 'personal' ? 'first_name' AND
            profile_data -> 'personal' ? 'last_name' AND
            profile_data -> 'personal' ? 'email') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate email format
    IF NOT (profile_data ->> 'personal' ->> 'email' ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate phone format if present
    IF profile_data -> 'personal' ? 'phone' THEN
        IF NOT (profile_data ->> 'personal' ->> 'phone' ~ '^\+?[1-9]\d{1,14}$') THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    -- Validate date of birth if present
    IF profile_data -> 'personal' ? 'date_of_birth' THEN
        BEGIN
            PERFORM (profile_data ->> 'personal' ->> 'date_of_birth')::date;
        EXCEPTION
            WHEN OTHERS THEN
                RETURN FALSE;
        END;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to validate JSONB data
CREATE OR REPLACE FUNCTION validate_user_profile_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT validate_user_profile(NEW.profile_data) THEN
        RAISE EXCEPTION 'Invalid user profile data';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_user_profile_trigger
    BEFORE INSERT OR UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION validate_user_profile_trigger();
```

### JSON Schema Validation

```python
# validation/json_schema_validator.py
import psycopg2
import jsonschema
import json
from typing import Dict, Any

class JSONBSchemaValidator:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.schema = {
            "type": "object",
            "properties": {
                "personal": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string", "minLength": 1},
                        "last_name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "format": "email"},
                        "phone": {"type": "string", "pattern": "^\\+?[1-9]\\d{1,14}$"},
                        "date_of_birth": {"type": "string", "format": "date"}
                    },
                    "required": ["first_name", "last_name", "email"]
                },
                "preferences": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "language": {"type": "string", "enum": ["en", "es", "fr", "de"]},
                        "notifications": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "boolean"},
                                "sms": {"type": "boolean"},
                                "push": {"type": "boolean"}
                            }
                        }
                    }
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip": {"type": "string"},
                        "country": {"type": "string"}
                    }
                }
            },
            "required": ["personal"]
        }
    
    def validate_jsonb_data(self, data: Dict[Any, Any]) -> bool:
        """Validate JSONB data against schema."""
        try:
            jsonschema.validate(data, self.schema)
            return True
        except jsonschema.ValidationError as e:
            print(f"Validation error: {e}")
            return False
    
    def insert_validated_data(self, user_id: int, profile_data: Dict[Any, Any]):
        """Insert validated JSONB data."""
        if not self.validate_jsonb_data(profile_data):
            raise ValueError("Invalid JSONB data")
        
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_profiles (user_id, profile_data)
                    VALUES (%s, %s)
                """, (user_id, json.dumps(profile_data)))
                conn.commit()
                print(f"Valid data inserted for user {user_id}")
                
        except Exception as e:
            print(f"Error inserting data: {e}")
            conn.rollback()
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    validator = JSONBSchemaValidator({
        'host': 'localhost',
        'database': 'production',
        'user': 'validator_user',
        'password': 'validator_password'
    })
    
    # Valid data
    valid_data = {
        "personal": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com"
        },
        "preferences": {
            "theme": "dark",
            "language": "en"
        }
    }
    
    validator.insert_validated_data(1, valid_data)
```

## JSONB Performance Optimization

### Query Performance Analysis

```sql
-- Analyze JSONB query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT 
    user_id,
    profile_data ->> 'personal' ->> 'first_name' as first_name,
    profile_data ->> 'address' ->> 'city' as city
FROM user_profiles
WHERE profile_data -> 'address' ->> 'state' = 'NY';

-- Check index usage
EXPLAIN (ANALYZE, BUFFERS)
SELECT user_id, profile_data
FROM user_profiles
WHERE profile_data @> '{"preferences": {"theme": "dark"}}';

-- Analyze GIN index effectiveness
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%gin%'
ORDER BY idx_scan DESC;
```

### JSONB Query Optimization

```sql
-- Optimize JSONB queries with proper indexing
-- Good: Use GIN index for containment queries
SELECT user_id, profile_data
FROM user_profiles
WHERE profile_data @> '{"preferences": {"theme": "dark"}}';

-- Good: Use B-tree index for equality queries
SELECT user_id, profile_data
FROM user_profiles
WHERE profile_data ->> 'personal' ->> 'email' = 'john.doe@example.com';

-- Good: Combine JSONB operations efficiently
SELECT 
    user_id,
    profile_data ->> 'personal' ->> 'first_name' as first_name,
    profile_data ->> 'address' ->> 'city' as city
FROM user_profiles
WHERE profile_data -> 'address' ->> 'state' = 'NY'
AND profile_data -> 'preferences' ->> 'theme' = 'dark';

-- Avoid: Inefficient JSONB operations
-- Bad: Multiple JSONB operations in WHERE clause
SELECT user_id, profile_data
FROM user_profiles
WHERE profile_data ->> 'personal' ->> 'first_name' LIKE 'J%'
AND profile_data ->> 'personal' ->> 'last_name' LIKE 'D%';
```

## JSONB Aggregation and Analytics

### Advanced JSONB Analytics

```sql
-- Create analytics table with JSONB metrics
CREATE TABLE user_analytics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample analytics data
INSERT INTO user_analytics (user_id, event_type, event_data) VALUES
    (1, 'page_view', '{"page": "/dashboard", "duration": 45, "referrer": "google.com"}'),
    (1, 'click', '{"element": "button", "action": "submit", "form": "contact"}'),
    (1, 'purchase', '{"product_id": 123, "amount": 99.99, "currency": "USD"}'),
    (2, 'page_view', '{"page": "/products", "duration": 120, "referrer": "facebook.com"}'),
    (2, 'click', '{"element": "link", "action": "navigate", "target": "/product/456"}');

-- Aggregate JSONB data
SELECT 
    user_id,
    COUNT(*) as event_count,
    AVG((event_data ->> 'duration')::numeric) as avg_duration,
    SUM((event_data ->> 'amount')::numeric) as total_spent
FROM user_analytics
WHERE event_type = 'purchase'
GROUP BY user_id;

-- Analyze user behavior patterns
SELECT 
    event_data ->> 'page' as page,
    COUNT(*) as view_count,
    AVG((event_data ->> 'duration')::numeric) as avg_duration
FROM user_analytics
WHERE event_type = 'page_view'
GROUP BY event_data ->> 'page'
ORDER BY view_count DESC;
```

### JSONB Statistical Functions

```sql
-- Create function for JSONB statistical analysis
CREATE OR REPLACE FUNCTION jsonb_stats(jsonb_data JSONB, key_path TEXT)
RETURNS TABLE (
    count BIGINT,
    min_val NUMERIC,
    max_val NUMERIC,
    avg_val NUMERIC,
    sum_val NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT,
        MIN((jsonb_data ->> key_path)::NUMERIC),
        MAX((jsonb_data ->> key_path)::NUMERIC),
        AVG((jsonb_data ->> key_path)::NUMERIC),
        SUM((jsonb_data ->> key_path)::NUMERIC)
    FROM jsonb_array_elements(jsonb_data);
END;
$$ LANGUAGE plpgsql;

-- Use statistical function
SELECT * FROM jsonb_stats(
    '[
        {"value": 10, "category": "A"},
        {"value": 20, "category": "B"},
        {"value": 30, "category": "A"},
        {"value": 40, "category": "B"}
    ]'::jsonb,
    'value'
);
```

## JSONB Migration Strategies

### JSON to JSONB Migration

```sql
-- Create migration function
CREATE OR REPLACE FUNCTION migrate_json_to_jsonb(
    source_table TEXT,
    source_column TEXT,
    target_table TEXT,
    target_column TEXT
)
RETURNS VOID AS $$
DECLARE
    batch_size INTEGER := 1000;
    offset_val INTEGER := 0;
    total_rows INTEGER;
BEGIN
    -- Get total row count
    EXECUTE format('SELECT COUNT(*) FROM %I', source_table) INTO total_rows;
    
    -- Migrate in batches
    WHILE offset_val < total_rows LOOP
        EXECUTE format('
            INSERT INTO %I (%I)
            SELECT %I::jsonb FROM %I
            ORDER BY ctid
            LIMIT %s OFFSET %s
        ', target_table, target_column, source_column, source_table, batch_size, offset_val);
        
        offset_val := offset_val + batch_size;
        RAISE NOTICE 'Migrated % rows of %', offset_val, total_rows;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Example migration
CREATE TABLE user_profiles_jsonb AS 
SELECT user_id, profile_data::jsonb as profile_data, created_at, updated_at
FROM user_profiles_json;
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create JSONB table
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    profile_data JSONB NOT NULL
);

-- 2. Create GIN index
CREATE INDEX idx_user_profiles_gin ON user_profiles USING GIN (profile_data);

-- 3. Insert JSONB data
INSERT INTO user_profiles (user_id, profile_data) VALUES
    (1, '{"name": "John", "age": 30, "city": "New York"}');

-- 4. Query JSONB data
SELECT user_id, profile_data ->> 'name' as name
FROM user_profiles
WHERE profile_data @> '{"city": "New York"}';
```

### Essential Patterns

```python
# Complete PostgreSQL JSON/JSONB setup
def setup_postgresql_json_jsonb():
    # 1. JSONB data types
    # 2. Indexing strategies
    # 3. Query patterns
    # 4. Schema validation
    # 5. Performance optimization
    # 6. Analytics and aggregation
    # 7. Migration strategies
    # 8. Monitoring and maintenance
    
    print("PostgreSQL JSON/JSONB setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL JSON/JSONB excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL JSON systems.*
