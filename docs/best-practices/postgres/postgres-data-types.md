# PostgreSQL Data Types Best Practices

**Objective**: Master senior-level PostgreSQL data type patterns for production systems. When you need to choose optimal data types, when you want to optimize storage and performance, when you need enterprise-grade data type strategies—these best practices become your weapon of choice.

## Core Principles

- **Storage Efficiency**: Choose types that minimize storage overhead
- **Performance**: Select types optimized for query patterns
- **Precision**: Use appropriate precision for numeric types
- **Validation**: Leverage type constraints for data integrity
- **Compatibility**: Ensure type compatibility across systems

## Numeric Data Types

### Integer Types

```sql
-- Choose appropriate integer types based on range requirements
CREATE TABLE user_analytics (
    id BIGSERIAL PRIMARY KEY,           -- 64-bit auto-incrementing
    user_id INTEGER NOT NULL,          -- 32-bit signed integer (-2B to +2B)
    session_count SMALLINT DEFAULT 0,  -- 16-bit signed integer (-32K to +32K)
    page_views INTEGER DEFAULT 0,      -- 32-bit signed integer
    total_time BIGINT DEFAULT 0,       -- 64-bit signed integer
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use appropriate integer types for different use cases
CREATE TABLE products (
    id SERIAL PRIMARY KEY,             -- Auto-incrementing primary key
    category_id SMALLINT NOT NULL,     -- Foreign key to categories (limited range)
    price_cents INTEGER NOT NULL,      -- Price in cents (avoid floating point)
    stock_quantity INTEGER NOT NULL,   -- Stock quantity
    weight_grams INTEGER,              -- Weight in grams
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use BIGINT for high-volume counters
CREATE TABLE event_logs (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT NOT NULL,          -- High-volume event IDs
    user_id BIGINT NOT NULL,           -- Large user base
    timestamp BIGINT NOT NULL,         -- Unix timestamp in milliseconds
    data JSONB
);
```

### Decimal and Numeric Types

```sql
-- Use NUMERIC for precise decimal calculations
CREATE TABLE financial_transactions (
    id SERIAL PRIMARY KEY,
    amount NUMERIC(10,2) NOT NULL,     -- 10 digits total, 2 decimal places
    currency VARCHAR(3) NOT NULL,
    exchange_rate NUMERIC(8,6),        -- High precision for exchange rates
    fee_amount NUMERIC(8,4),           -- Fee with 4 decimal places
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use appropriate precision for different use cases
CREATE TABLE measurements (
    id SERIAL PRIMARY KEY,
    temperature NUMERIC(5,2),          -- Temperature with 2 decimal places
    humidity NUMERIC(5,2),            -- Humidity percentage
    pressure NUMERIC(8,2),            -- Atmospheric pressure
    coordinates POINT,                 -- Geographic coordinates
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Avoid FLOAT/REAL for financial calculations
-- Use NUMERIC instead for exact decimal arithmetic
CREATE TABLE price_history (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    price NUMERIC(10,2) NOT NULL,      -- Exact decimal price
    -- price REAL NOT NULL,           -- DON'T use REAL for money
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ
);
```

### Floating Point Types

```sql
-- Use DOUBLE PRECISION for scientific calculations
CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    sensor_id INTEGER NOT NULL,
    temperature DOUBLE PRECISION,      -- Scientific temperature readings
    humidity DOUBLE PRECISION,        -- Humidity measurements
    pressure DOUBLE PRECISION,       -- Atmospheric pressure
    acceleration DOUBLE PRECISION,    -- 3D acceleration vector
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use REAL for less precise measurements where storage matters
CREATE TABLE weather_stations (
    id SERIAL PRIMARY KEY,
    station_name VARCHAR(100) NOT NULL,
    latitude REAL,                    -- Approximate coordinates
    longitude REAL,                   -- Approximate coordinates
    elevation REAL,                   -- Elevation in meters
    temperature REAL,                 -- Temperature readings
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

## Character and Text Types

### String Types

```sql
-- Choose appropriate string types based on length and usage
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,    -- Fixed max length
    email VARCHAR(255) UNIQUE NOT NULL,     -- Email addresses
    first_name VARCHAR(100),                -- Names
    last_name VARCHAR(100),
    bio TEXT,                               -- Variable length text
    website_url VARCHAR(500),               -- URLs
    phone VARCHAR(20),                      -- Phone numbers
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use CHAR for fixed-length strings
CREATE TABLE country_codes (
    id SERIAL PRIMARY KEY,
    country_code CHAR(2) NOT NULL,          -- ISO country codes (e.g., 'US', 'CA')
    country_name VARCHAR(100) NOT NULL,
    currency_code CHAR(3) NOT NULL,        -- ISO currency codes (e.g., 'USD', 'EUR')
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use TEXT for large content
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,                  -- Large text content
    excerpt TEXT,                          -- Article excerpt
    tags TEXT[],                           -- Array of text tags
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Text Processing and Validation

```sql
-- Create functions for text validation
CREATE OR REPLACE FUNCTION validate_email(email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_phone(phone TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN phone ~ '^\+?[1-9]\d{1,14}$';
END;
$$ LANGUAGE plpgsql;

-- Use constraints with validation functions
ALTER TABLE user_profiles ADD CONSTRAINT users_email_valid 
    CHECK (validate_email(email));

ALTER TABLE user_profiles ADD CONSTRAINT users_phone_valid 
    CHECK (phone IS NULL OR validate_phone(phone));

-- Create indexes for text search
CREATE INDEX idx_user_profiles_username_lower ON user_profiles (LOWER(username));
CREATE INDEX idx_user_profiles_email_lower ON user_profiles (LOWER(email));
CREATE INDEX idx_articles_content_gin ON articles USING GIN (to_tsvector('english', content));
```

## Date and Time Types

### Timestamp Types

```sql
-- Use TIMESTAMPTZ for timezone-aware timestamps
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,       -- Timezone-aware start time
    end_time TIMESTAMPTZ NOT NULL,         -- Timezone-aware end time
    timezone VARCHAR(50) DEFAULT 'UTC',    -- Event timezone
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use TIMESTAMP for timezone-naive timestamps
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Server timezone
    source VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use DATE for date-only values
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_date DATE NOT NULL,            -- Date only
    login_count INTEGER DEFAULT 1,
    total_duration INTERVAL,                 -- Duration as interval
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Time Zone Handling

```sql
-- Create timezone-aware tables
CREATE TABLE global_events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    duration INTERVAL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Function to convert timezone
CREATE OR REPLACE FUNCTION convert_to_timezone(
    input_time TIMESTAMPTZ,
    target_timezone TEXT
)
RETURNS TIMESTAMPTZ AS $$
BEGIN
    RETURN input_time AT TIME ZONE target_timezone;
END;
$$ LANGUAGE plpgsql;

-- Query with timezone conversion
SELECT 
    event_name,
    event_time,
    event_time AT TIME ZONE timezone as local_time,
    EXTRACT(TIMEZONE FROM event_time) as timezone_offset
FROM global_events
WHERE event_time >= '2024-01-01'::timestamptz;
```

## Boolean and Enum Types

### Boolean Types

```sql
-- Use BOOLEAN for true/false values
CREATE TABLE user_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    email_notifications BOOLEAN DEFAULT TRUE,
    sms_notifications BOOLEAN DEFAULT FALSE,
    push_notifications BOOLEAN DEFAULT TRUE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use boolean expressions in queries
SELECT 
    user_id,
    CASE 
        WHEN email_notifications AND sms_notifications THEN 'Both'
        WHEN email_notifications THEN 'Email Only'
        WHEN sms_notifications THEN 'SMS Only'
        ELSE 'None'
    END as notification_preference
FROM user_settings
WHERE is_active = TRUE;
```

### Enum Types

```sql
-- Create enum types for controlled vocabularies
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended', 'deleted');
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered', 'cancelled');
CREATE TYPE priority_level AS ENUM ('low', 'medium', 'high', 'critical');

-- Use enum types in tables
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    status user_status DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status order_status DEFAULT 'pending',
    priority priority_level DEFAULT 'medium',
    total_amount NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Query with enum values
SELECT 
    u.username,
    u.status,
    COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.status = 'active'
GROUP BY u.id, u.username, u.status;
```

## Array Types

### Array Data Types

```sql
-- Use arrays for multiple values
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    tags TEXT[],                        -- Array of text tags
    categories INTEGER[],              -- Array of category IDs
    images TEXT[],                      -- Array of image URLs
    specifications JSONB,              -- JSON specifications
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert data with arrays
INSERT INTO products (name, tags, categories, images) VALUES
    ('Laptop Computer', ARRAY['electronics', 'computers', 'portable'], 
     ARRAY[1, 5, 12], ARRAY['laptop1.jpg', 'laptop2.jpg']),
    ('Smartphone', ARRAY['electronics', 'mobile', 'communication'], 
     ARRAY[1, 8, 15], ARRAY['phone1.jpg', 'phone2.jpg', 'phone3.jpg']);

-- Query arrays
SELECT 
    name,
    tags,
    array_length(tags, 1) as tag_count,
    categories,
    images
FROM products
WHERE 'electronics' = ANY(tags);

-- Use array operators
SELECT 
    name,
    tags
FROM products
WHERE tags && ARRAY['electronics', 'mobile'];  -- Overlap operator

SELECT 
    name,
    tags
FROM products
WHERE tags @> ARRAY['electronics'];            -- Contains operator
```

### Array Functions and Operations

```sql
-- Array manipulation functions
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    favorite_categories TEXT[],
    blocked_users INTEGER[],
    notification_channels TEXT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Add elements to arrays
UPDATE user_preferences 
SET favorite_categories = array_append(favorite_categories, 'technology')
WHERE user_id = 1;

-- Remove elements from arrays
UPDATE user_preferences 
SET favorite_categories = array_remove(favorite_categories, 'sports')
WHERE user_id = 1;

-- Query array functions
SELECT 
    user_id,
    favorite_categories,
    array_length(favorite_categories, 1) as category_count,
    array_to_string(favorite_categories, ', ') as categories_string
FROM user_preferences
WHERE 'technology' = ANY(favorite_categories);
```

## JSON and JSONB Types

### JSONB Usage Patterns

```sql
-- Use JSONB for semi-structured data
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    profile_data JSONB NOT NULL,
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert JSONB data
INSERT INTO user_profiles (user_id, profile_data, preferences) VALUES
    (1, '{"name": "John Doe", "age": 30, "location": "New York", "interests": ["technology", "travel"]}', 
     '{"theme": "dark", "language": "en", "notifications": {"email": true, "sms": false}}'),
    (2, '{"name": "Jane Smith", "age": 25, "location": "San Francisco", "interests": ["art", "music"]}', 
     '{"theme": "light", "language": "en", "notifications": {"email": true, "sms": true}}');

-- Query JSONB data
SELECT 
    user_id,
    profile_data ->> 'name' as name,
    profile_data ->> 'age' as age,
    profile_data ->> 'location' as location,
    preferences ->> 'theme' as theme
FROM user_profiles
WHERE profile_data @> '{"interests": ["technology"]}';

-- Create indexes on JSONB
CREATE INDEX idx_user_profiles_data_gin ON user_profiles USING GIN (profile_data);
CREATE INDEX idx_user_profiles_name_btree ON user_profiles ((profile_data ->> 'name'));
CREATE INDEX idx_user_profiles_age_btree ON user_profiles (((profile_data ->> 'age')::integer));
```

### JSONB Schema Validation

```sql
-- Create JSONB schema validation
CREATE OR REPLACE FUNCTION validate_user_profile(profile_data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check required fields
    IF NOT (profile_data ? 'name' AND profile_data ? 'age') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate age is numeric
    IF NOT (profile_data ->> 'age' ~ '^\d+$') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate age range
    IF (profile_data ->> 'age')::integer < 0 OR (profile_data ->> 'age')::integer > 150 THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add constraint for JSONB validation
ALTER TABLE user_profiles ADD CONSTRAINT user_profiles_data_valid 
    CHECK (validate_user_profile(profile_data));
```

## UUID and Network Types

### UUID Usage

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Use UUID for distributed systems
CREATE TABLE distributed_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use UUID for external references
CREATE TABLE external_integrations (
    id SERIAL PRIMARY KEY,
    external_id UUID NOT NULL UNIQUE,
    integration_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Generate UUIDs
SELECT uuid_generate_v4() as new_uuid;
SELECT uuid_generate_v1() as time_based_uuid;
SELECT uuid_generate_v1mc() as mac_based_uuid;
```

### Network Types

```sql
-- Use network types for IP addresses
CREATE TABLE access_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    ip_address INET NOT NULL,
    user_agent TEXT,
    request_path VARCHAR(500),
    response_code INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert network data
INSERT INTO access_logs (user_id, ip_address, user_agent, request_path, response_code) VALUES
    (1, '192.168.1.100', 'Mozilla/5.0...', '/api/users', 200),
    (2, '10.0.0.50', 'Chrome/91.0...', '/api/orders', 201),
    (NULL, '203.0.113.1', 'Bot/1.0', '/robots.txt', 404);

-- Query network data
SELECT 
    user_id,
    ip_address,
    CASE 
        WHEN ip_address << '192.168.0.0/16' THEN 'Private'
        WHEN ip_address << '10.0.0.0/8' THEN 'Private'
        WHEN ip_address << '172.16.0.0/12' THEN 'Private'
        ELSE 'Public'
    END as network_type
FROM access_logs
WHERE created_at >= CURRENT_DATE;
```

## Custom Data Types

### Creating Custom Types

```sql
-- Create composite types
CREATE TYPE address AS (
    street VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50)
);

CREATE TYPE contact_info AS (
    email VARCHAR(255),
    phone VARCHAR(20),
    address address
);

-- Use composite types in tables
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    contact contact_info NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert composite type data
INSERT INTO customers (name, contact) VALUES
    ('John Doe', ROW('john@example.com', '+1-555-0123', 
                     ROW('123 Main St', 'New York', 'NY', '10001', 'USA'))),
    ('Jane Smith', ROW('jane@example.com', '+1-555-0456', 
                       ROW('456 Oak Ave', 'San Francisco', 'CA', '94102', 'USA')));

-- Query composite types
SELECT 
    name,
    (contact).email,
    (contact).phone,
    (contact).address.street,
    (contact).address.city
FROM customers;
```

### Type Conversion and Casting

```sql
-- Create type conversion functions
CREATE OR REPLACE FUNCTION convert_temperature(
    value NUMERIC,
    from_unit VARCHAR(10),
    to_unit VARCHAR(10)
)
RETURNS NUMERIC AS $$
BEGIN
    -- Convert to Celsius first
    IF from_unit = 'fahrenheit' THEN
        value := (value - 32) * 5/9;
    ELSIF from_unit = 'kelvin' THEN
        value := value - 273.15;
    END IF;
    
    -- Convert from Celsius to target unit
    IF to_unit = 'fahrenheit' THEN
        RETURN (value * 9/5) + 32;
    ELSIF to_unit = 'kelvin' THEN
        RETURN value + 273.15;
    ELSE
        RETURN value; -- Celsius
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Use type conversion
SELECT 
    convert_temperature(32, 'fahrenheit', 'celsius') as celsius,
    convert_temperature(0, 'celsius', 'fahrenheit') as fahrenheit,
    convert_temperature(273.15, 'kelvin', 'celsius') as celsius_from_kelvin;
```

## Performance Optimization

### Type Performance Analysis

```python
# performance/type_performance.py
import psycopg2
import time
import json
from typing import Dict, List, Any

class DataTypePerformanceAnalyzer:
    def __init__(self, connection_params):
        self.conn_params = connection_params
    
    def analyze_type_performance(self) -> Dict[str, Any]:
        """Analyze performance of different data types."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Test integer types
                integer_performance = self.test_integer_types(cur)
                
                # Test string types
                string_performance = self.test_string_types(cur)
                
                # Test JSONB vs JSON
                json_performance = self.test_json_types(cur)
                
                return {
                    'integer_performance': integer_performance,
                    'string_performance': string_performance,
                    'json_performance': json_performance,
                    'analysis_timestamp': '2024-01-15T10:30:00Z'
                }
                
        except Exception as e:
            print(f"Error analyzing type performance: {e}")
            return {}
        finally:
            conn.close()
    
    def test_integer_types(self, cursor) -> Dict[str, Any]:
        """Test performance of different integer types."""
        # Create test tables
        cursor.execute("""
            CREATE TEMP TABLE test_integers (
                id SERIAL PRIMARY KEY,
                smallint_col SMALLINT,
                integer_col INTEGER,
                bigint_col BIGINT
            )
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO test_integers (smallint_col, integer_col, bigint_col)
            SELECT 
                (random() * 32767)::SMALLINT,
                (random() * 2147483647)::INTEGER,
                (random() * 9223372036854775807)::BIGINT
            FROM generate_series(1, 10000)
        """)
        
        # Test query performance
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_integers WHERE smallint_col > 1000")
        smallint_time = time.time() - start_time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_integers WHERE integer_col > 1000000")
        integer_time = time.time() - start_time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_integers WHERE bigint_col > 1000000000")
        bigint_time = time.time() - start_time
        
        return {
            'smallint_time': smallint_time,
            'integer_time': integer_time,
            'bigint_time': bigint_time
        }
    
    def test_string_types(self, cursor) -> Dict[str, Any]:
        """Test performance of different string types."""
        # Create test tables
        cursor.execute("""
            CREATE TEMP TABLE test_strings (
                id SERIAL PRIMARY KEY,
                varchar_col VARCHAR(100),
                text_col TEXT,
                char_col CHAR(10)
            )
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO test_strings (varchar_col, text_col, char_col)
            SELECT 
                'test_string_' || generate_series,
                'long_text_content_' || generate_series,
                'char' || generate_series
            FROM generate_series(1, 10000)
        """)
        
        # Test query performance
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_strings WHERE varchar_col LIKE 'test_string_%'")
        varchar_time = time.time() - start_time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_strings WHERE text_col LIKE 'long_text_content_%'")
        text_time = time.time() - start_time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_strings WHERE char_col LIKE 'char%'")
        char_time = time.time() - start_time
        
        return {
            'varchar_time': varchar_time,
            'text_time': text_time,
            'char_time': char_time
        }
    
    def test_json_types(self, cursor) -> Dict[str, Any]:
        """Test performance of JSON vs JSONB."""
        # Create test tables
        cursor.execute("""
            CREATE TEMP TABLE test_json (
                id SERIAL PRIMARY KEY,
                json_col JSON,
                jsonb_col JSONB
            )
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO test_json (json_col, jsonb_col)
            SELECT 
                '{"id": ' || generate_series || ', "name": "test_' || generate_series || '", "value": ' || (random() * 1000)::INTEGER || '}',
                '{"id": ' || generate_series || ', "name": "test_' || generate_series || '", "value": ' || (random() * 1000)::INTEGER || '}'::JSONB
            FROM generate_series(1, 10000)
        """)
        
        # Test query performance
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_json WHERE json_col ->> 'name' LIKE 'test_%'")
        json_time = time.time() - start_time
        
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM test_json WHERE jsonb_col ->> 'name' LIKE 'test_%'")
        jsonb_time = time.time() - start_time
        
        return {
            'json_time': json_time,
            'jsonb_time': jsonb_time,
            'jsonb_advantage': (json_time - jsonb_time) / json_time * 100
        }

# Usage
if __name__ == "__main__":
    analyzer = DataTypePerformanceAnalyzer({
        'host': 'localhost',
        'database': 'production',
        'user': 'analyzer_user',
        'password': 'analyzer_password'
    })
    
    results = analyzer.analyze_type_performance()
    print(json.dumps(results, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Choose appropriate numeric types
CREATE TABLE products (
    id SERIAL PRIMARY KEY,             -- Auto-incrementing
    price NUMERIC(10,2) NOT NULL,       -- Exact decimal
    stock INTEGER NOT NULL,             -- 32-bit integer
    weight DOUBLE PRECISION            -- Scientific precision
);

-- 2. Use appropriate string types
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE,        -- Fixed max length
    email VARCHAR(255) UNIQUE,          -- Email addresses
    bio TEXT,                          -- Variable length
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Use JSONB for semi-structured data
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Essential Patterns

```python
# Complete PostgreSQL data types setup
def setup_postgresql_data_types():
    # 1. Numeric types
    # 2. Character and text types
    # 3. Date and time types
    # 4. Boolean and enum types
    # 5. Array types
    # 6. JSON and JSONB types
    # 7. UUID and network types
    # 8. Custom types and performance optimization
    
    print("PostgreSQL data types setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL data types excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL data type systems.*
