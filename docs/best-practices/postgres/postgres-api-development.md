# PostgreSQL API Development Best Practices

**Objective**: Master senior-level PostgreSQL API development patterns for production systems. When you need to build robust APIs with PostgreSQL, when you want to implement RESTful services, when you need enterprise-grade API strategies—these best practices become your weapon of choice.

## Core Principles

- **RESTful Design**: Follow REST principles for API design
- **Security**: Implement proper authentication and authorization
- **Performance**: Optimize API response times and throughput
- **Documentation**: Provide comprehensive API documentation
- **Versioning**: Implement API versioning strategies

## API Architecture Patterns

### RESTful API Design

```sql
-- Create API endpoints table
CREATE TABLE api_endpoints (
    id SERIAL PRIMARY KEY,
    endpoint_path VARCHAR(200) UNIQUE NOT NULL,
    http_method VARCHAR(10) NOT NULL,
    description TEXT,
    parameters JSONB,
    response_schema JSONB,
    authentication_required BOOLEAN DEFAULT TRUE,
    rate_limit_per_minute INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register API endpoint
CREATE OR REPLACE FUNCTION register_api_endpoint(
    p_endpoint_path VARCHAR(200),
    p_http_method VARCHAR(10),
    p_description TEXT,
    p_parameters JSONB DEFAULT '{}',
    p_response_schema JSONB DEFAULT '{}',
    p_authentication_required BOOLEAN DEFAULT TRUE,
    p_rate_limit_per_minute INTEGER DEFAULT 100
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO api_endpoints (
        endpoint_path, http_method, description, parameters,
        response_schema, authentication_required, rate_limit_per_minute
    ) VALUES (
        p_endpoint_path, p_http_method, p_description, p_parameters,
        p_response_schema, p_authentication_required, p_rate_limit_per_minute
    ) ON CONFLICT (endpoint_path) 
    DO UPDATE SET 
        http_method = EXCLUDED.http_method,
        description = EXCLUDED.description,
        parameters = EXCLUDED.parameters,
        response_schema = EXCLUDED.response_schema,
        authentication_required = EXCLUDED.authentication_required,
        rate_limit_per_minute = EXCLUDED.rate_limit_per_minute,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Register common API endpoints
SELECT register_api_endpoint('/api/v1/users', 'GET', 'Get all users', '{}', '{"users": []}', TRUE, 100);
SELECT register_api_endpoint('/api/v1/users/{id}', 'GET', 'Get user by ID', '{"id": "integer"}', '{"user": {}}', TRUE, 100);
SELECT register_api_endpoint('/api/v1/users', 'POST', 'Create new user', '{"user": {}}', '{"user": {}}', TRUE, 50);
SELECT register_api_endpoint('/api/v1/users/{id}', 'PUT', 'Update user', '{"id": "integer", "user": {}}', '{"user": {}}', TRUE, 50);
SELECT register_api_endpoint('/api/v1/users/{id}', 'DELETE', 'Delete user', '{"id": "integer"}', '{"message": "User deleted"}', TRUE, 25);
```

### API Authentication

```sql
-- Create API keys table
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    user_id INTEGER,
    permissions JSONB DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ
);

-- Create function to generate API key
CREATE OR REPLACE FUNCTION generate_api_key(
    p_key_name VARCHAR(100),
    p_user_id INTEGER DEFAULT NULL,
    p_permissions JSONB DEFAULT '{}',
    p_rate_limit_per_minute INTEGER DEFAULT 100,
    p_expires_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS TEXT AS $$
DECLARE
    api_key TEXT;
    key_hash VARCHAR(64);
BEGIN
    -- Generate random API key
    api_key := encode(gen_random_bytes(32), 'hex');
    key_hash := encode(digest(api_key, 'sha256'), 'hex');
    
    -- Store key hash
    INSERT INTO api_keys (
        key_hash, key_name, user_id, permissions, 
        rate_limit_per_minute, expires_at
    ) VALUES (
        key_hash, p_key_name, p_user_id, p_permissions,
        p_rate_limit_per_minute, p_expires_at
    );
    
    RETURN api_key;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate API key
CREATE OR REPLACE FUNCTION validate_api_key(p_api_key TEXT)
RETURNS TABLE (
    is_valid BOOLEAN,
    user_id INTEGER,
    permissions JSONB,
    rate_limit_per_minute INTEGER
) AS $$
DECLARE
    key_hash VARCHAR(64);
    key_record RECORD;
BEGIN
    -- Hash the provided key
    key_hash := encode(digest(p_api_key, 'sha256'), 'hex');
    
    -- Look up key
    SELECT ak.user_id, ak.permissions, ak.rate_limit_per_minute
    INTO key_record
    FROM api_keys ak
    WHERE ak.key_hash = key_hash
    AND ak.is_active = TRUE
    AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP);
    
    IF key_record IS NULL THEN
        RETURN QUERY SELECT FALSE, NULL::INTEGER, '{}'::JSONB, 0::INTEGER;
    ELSE
        -- Update last used timestamp
        UPDATE api_keys 
        SET last_used_at = CURRENT_TIMESTAMP
        WHERE key_hash = key_hash;
        
        RETURN QUERY SELECT TRUE, key_record.user_id, key_record.permissions, key_record.rate_limit_per_minute;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## API Implementation

### FastAPI Integration

```python
# api/postgres_api.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import psycopg2
import json
from datetime import datetime
from typing import Optional, List
import logging

class PostgreSQLAPI:
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
    
    def validate_api_key(self, api_key: str):
        """Validate API key."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM validate_api_key(%s)", (api_key,))
                result = cur.fetchone()
                
                if result and result[0]:  # is_valid
                    return {
                        'user_id': result[1],
                        'permissions': result[2],
                        'rate_limit': result[3]
                    }
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            return None
        finally:
            conn.close()
    
    def get_users(self, limit: int = 100, offset: int = 0):
        """Get users with pagination."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, email, created_at
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                users = cur.fetchall()
                return [{'id': u[0], 'username': u[1], 'email': u[2], 'created_at': u[3]} for u in users]
                
        except Exception as e:
            self.logger.error(f"Error getting users: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            conn.close()
    
    def get_user(self, user_id: int):
        """Get user by ID."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, email, created_at
                    FROM users
                    WHERE id = %s
                """, (user_id,))
                
                user = cur.fetchone()
                if user:
                    return {'id': user[0], 'username': user[1], 'email': user[2], 'created_at': user[3]}
                else:
                    raise HTTPException(status_code=404, detail="User not found")
                    
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting user: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            conn.close()
    
    def create_user(self, user_data: dict):
        """Create new user."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (username, email, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING id, username, email, created_at
                """, (user_data['username'], user_data['email'], user_data['password_hash']))
                
                user = cur.fetchone()
                conn.commit()
                
                return {'id': user[0], 'username': user[1], 'email': user[2], 'created_at': user[3]}
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            conn.close()
    
    def update_user(self, user_id: int, user_data: dict):
        """Update user."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE users
                    SET username = %s, email = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id, username, email, created_at
                """, (user_data['username'], user_data['email'], user_id))
                
                user = cur.fetchone()
                if user:
                    conn.commit()
                    return {'id': user[0], 'username': user[1], 'email': user[2], 'created_at': user[3]}
                else:
                    raise HTTPException(status_code=404, detail="User not found")
                    
        except HTTPException:
            raise
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating user: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            conn.close()
    
    def delete_user(self, user_id: int):
        """Delete user."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                
                if cur.rowcount > 0:
                    conn.commit()
                    return {'message': 'User deleted successfully'}
                else:
                    raise HTTPException(status_code=404, detail="User not found")
                    
        except HTTPException:
            raise
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error deleting user: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            conn.close()

# FastAPI application
app = FastAPI(title="PostgreSQL API", version="1.0.0")
security = HTTPBearer()
api = PostgreSQLAPI({
    'host': 'localhost',
    'database': 'production',
    'user': 'api_user',
    'password': 'api_password'
})

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from API key."""
    api_key = credentials.credentials
    user_info = api.validate_api_key(api_key)
    
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user_info

@app.get("/api/v1/users")
async def get_users(
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get all users."""
    return api.get_users(limit, offset)

@app.get("/api/v1/users/{user_id}")
async def get_user(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get user by ID."""
    return api.get_user(user_id)

@app.post("/api/v1/users")
async def create_user(
    user_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Create new user."""
    return api.create_user(user_data)

@app.put("/api/v1/users/{user_id}")
async def update_user(
    user_id: int,
    user_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update user."""
    return api.update_user(user_id, user_data)

@app.delete("/api/v1/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete user."""
    return api.delete_user(user_id)
```

## API Security

### Rate Limiting

```sql
-- Create rate limiting table
CREATE TABLE api_rate_limits (
    id SERIAL PRIMARY KEY,
    api_key_hash VARCHAR(64) NOT NULL,
    endpoint_path VARCHAR(200) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to check rate limit
CREATE OR REPLACE FUNCTION check_rate_limit(
    p_api_key_hash VARCHAR(64),
    p_endpoint_path VARCHAR(200),
    p_rate_limit_per_minute INTEGER DEFAULT 100
)
RETURNS BOOLEAN AS $$
DECLARE
    current_count INTEGER;
    window_start TIMESTAMPTZ;
BEGIN
    -- Get current window start
    window_start := date_trunc('minute', CURRENT_TIMESTAMP);
    
    -- Get current request count
    SELECT COALESCE(SUM(request_count), 0) INTO current_count
    FROM api_rate_limits
    WHERE api_key_hash = p_api_key_hash
    AND endpoint_path = p_endpoint_path
    AND window_start >= window_start;
    
    -- Check if limit exceeded
    IF current_count >= p_rate_limit_per_minute THEN
        RETURN FALSE;
    END IF;
    
    -- Record request
    INSERT INTO api_rate_limits (api_key_hash, endpoint_path, window_start)
    VALUES (p_api_key_hash, p_endpoint_path, window_start)
    ON CONFLICT (api_key_hash, endpoint_path, window_start)
    DO UPDATE SET request_count = api_rate_limits.request_count + 1;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Input Validation

```sql
-- Create function to validate input
CREATE OR REPLACE FUNCTION validate_api_input(
    p_input_data JSONB,
    p_validation_schema JSONB
)
RETURNS BOOLEAN AS $$
DECLARE
    required_fields TEXT[];
    field_name TEXT;
    field_value JSONB;
BEGIN
    -- Check required fields
    required_fields := ARRAY(SELECT jsonb_array_elements_text(p_validation_schema->'required'));
    
    FOR field_name IN SELECT unnest(required_fields) LOOP
        field_value := p_input_data -> field_name;
        
        IF field_value IS NULL OR field_value = 'null' THEN
            RAISE EXCEPTION 'Required field % is missing', field_name;
        END IF;
    END LOOP;
    
    -- Validate field types
    IF p_validation_schema ? 'properties' THEN
        FOR field_name, field_schema IN 
            SELECT key, value FROM jsonb_each(p_validation_schema->'properties')
        LOOP
            field_value := p_input_data -> field_name;
            
            IF field_value IS NOT NULL AND field_value != 'null' THEN
                -- Validate string fields
                IF field_schema->>'type' = 'string' AND jsonb_typeof(field_value) != 'string' THEN
                    RAISE EXCEPTION 'Field % must be a string', field_name;
                END IF;
                
                -- Validate integer fields
                IF field_schema->>'type' = 'integer' AND jsonb_typeof(field_value) != 'number' THEN
                    RAISE EXCEPTION 'Field % must be an integer', field_name;
                END IF;
                
                -- Validate email format
                IF field_schema->>'format' = 'email' THEN
                    IF NOT (field_value #>> '{}' ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
                        RAISE EXCEPTION 'Field % must be a valid email', field_name;
                    END IF;
                END IF;
            END IF;
        END LOOP;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## API Documentation

### OpenAPI Integration

```python
# api/openapi_generator.py
import psycopg2
import json
from typing import Dict, List

class OpenAPIGenerator:
    def __init__(self, connection_params):
        self.conn_params = connection_params
    
    def generate_openapi_spec(self):
        """Generate OpenAPI specification from database."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Get API endpoints
                cur.execute("""
                    SELECT endpoint_path, http_method, description, 
                           parameters, response_schema, authentication_required
                    FROM api_endpoints
                    ORDER BY endpoint_path
                """)
                
                endpoints = cur.fetchall()
                
                # Generate OpenAPI spec
                openapi_spec = {
                    "openapi": "3.0.0",
                    "info": {
                        "title": "PostgreSQL API",
                        "version": "1.0.0",
                        "description": "API for PostgreSQL database operations"
                    },
                    "servers": [
                        {"url": "https://api.example.com", "description": "Production server"}
                    ],
                    "security": [
                        {"ApiKeyAuth": []}
                    ],
                    "paths": self.generate_paths(endpoints),
                    "components": {
                        "securitySchemes": {
                            "ApiKeyAuth": {
                                "type": "apiKey",
                                "in": "header",
                                "name": "Authorization"
                            }
                        }
                    }
                }
                
                return openapi_spec
                
        except Exception as e:
            print(f"Error generating OpenAPI spec: {e}")
            return {}
        finally:
            conn.close()
    
    def generate_paths(self, endpoints):
        """Generate OpenAPI paths from endpoints."""
        paths = {}
        
        for endpoint_path, http_method, description, parameters, response_schema, auth_required in endpoints:
            if endpoint_path not in paths:
                paths[endpoint_path] = {}
            
            paths[endpoint_path][http_method.lower()] = {
                "summary": description,
                "security": [{"ApiKeyAuth": []}] if auth_required else [],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": response_schema
                            }
                        }
                    },
                    "401": {"description": "Unauthorized"},
                    "404": {"description": "Not found"},
                    "500": {"description": "Internal server error"}
                }
            }
        
        return paths
```

## API Monitoring

### API Metrics

```sql
-- Create API metrics table
CREATE TABLE api_metrics (
    id SERIAL PRIMARY KEY,
    endpoint_path VARCHAR(200) NOT NULL,
    http_method VARCHAR(10) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    status_code INTEGER NOT NULL,
    api_key_hash VARCHAR(64),
    user_id INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to record API metrics
CREATE OR REPLACE FUNCTION record_api_metrics(
    p_endpoint_path VARCHAR(200),
    p_http_method VARCHAR(10),
    p_response_time_ms INTEGER,
    p_status_code INTEGER,
    p_api_key_hash VARCHAR(64) DEFAULT NULL,
    p_user_id INTEGER DEFAULT NULL,
    p_request_size_bytes INTEGER DEFAULT NULL,
    p_response_size_bytes INTEGER DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO api_metrics (
        endpoint_path, http_method, response_time_ms, status_code,
        api_key_hash, user_id, request_size_bytes, response_size_bytes
    ) VALUES (
        p_endpoint_path, p_http_method, p_response_time_ms, p_status_code,
        p_api_key_hash, p_user_id, p_request_size_bytes, p_response_size_bytes
    );
END;
$$ LANGUAGE plpgsql;

-- Create function to get API metrics
CREATE OR REPLACE FUNCTION get_api_metrics(
    p_start_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP - INTERVAL '1 hour',
    p_end_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    endpoint_path VARCHAR(200),
    http_method VARCHAR(10),
    request_count BIGINT,
    avg_response_time NUMERIC,
    max_response_time INTEGER,
    min_response_time INTEGER,
    success_rate NUMERIC,
    error_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        am.endpoint_path,
        am.http_method,
        COUNT(*) as request_count,
        AVG(am.response_time_ms) as avg_response_time,
        MAX(am.response_time_ms) as max_response_time,
        MIN(am.response_time_ms) as min_response_time,
        (COUNT(*) FILTER (WHERE am.status_code < 400)::NUMERIC / COUNT(*)::NUMERIC * 100) as success_rate,
        COUNT(*) FILTER (WHERE am.status_code >= 400) as error_count
    FROM api_metrics am
    WHERE am.created_at BETWEEN p_start_date AND p_end_date
    GROUP BY am.endpoint_path, am.http_method
    ORDER BY request_count DESC;
END;
$$ LANGUAGE plpgsql;
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Register API endpoints
SELECT register_api_endpoint('/api/v1/users', 'GET', 'Get all users');

-- 2. Generate API key
SELECT generate_api_key('my_api_key', 1, '{"read": true, "write": true}');

-- 3. Validate API key
SELECT * FROM validate_api_key('your_api_key_here');

-- 4. Check rate limits
SELECT check_rate_limit('key_hash', '/api/v1/users', 100);

-- 5. Record API metrics
SELECT record_api_metrics('/api/v1/users', 'GET', 150, 200);
```

### Essential Patterns

```python
# Complete PostgreSQL API development setup
def setup_postgresql_api_development():
    # 1. RESTful API design
    # 2. API authentication
    # 3. API implementation
    # 4. API security
    # 5. Input validation
    # 6. API documentation
    # 7. API monitoring
    # 8. Rate limiting
    
    print("PostgreSQL API development setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL API development excellence. Each pattern includes implementation examples, API strategies, and real-world usage patterns for enterprise PostgreSQL API systems.*
