# PostgreSQL Development Environment Best Practices

**Objective**: Master senior-level PostgreSQL development environment setup for production systems. When you need to set up a robust development environment, when you want to optimize your PostgreSQL workflow, when you need enterprise-grade development patterns—these best practices become your weapon of choice.

## Core Principles

- **Version Management**: Use multiple PostgreSQL versions for testing
- **Extension Management**: Essential extensions for development
- **Configuration Optimization**: Tuned settings for development
- **Development Tools**: Modern tooling for PostgreSQL development
- **Testing Environment**: Isolated testing with proper data management

## Development Environment Setup

### PostgreSQL Installation

```bash
# Install PostgreSQL using official repository
sudo apt update
sudo apt install -y postgresql-16 postgresql-client-16 postgresql-contrib-16

# Or using Homebrew on macOS
brew install postgresql@16

# Or using Docker for development
docker run --name postgres-dev \
  -e POSTGRES_PASSWORD=devpassword \
  -e POSTGRES_DB=development \
  -p 5432:5432 \
  -d postgres:16
```

### Version Management

```bash
# Using pg_versions for multiple PostgreSQL versions
curl -sSL https://raw.githubusercontent.com/markw/pg_versions/master/install.sh | bash

# Install multiple versions
pg_versions install 14
pg_versions install 15
pg_versions install 16

# Switch between versions
pg_versions use 16
```

### Essential Extensions

```sql
-- Connect to your development database
\c development

-- Essential extensions for development
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "hstore";
CREATE EXTENSION IF NOT EXISTS "ltree";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- For geospatial development
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "postgis_topology";
CREATE EXTENSION IF NOT EXISTS "fuzzystrmatch";
CREATE EXTENSION IF NOT EXISTS "postgis_tiger_geocoder";

-- For full-text search
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- For JSON operations
CREATE EXTENSION IF NOT EXISTS "jsquery";

-- For time series (if needed)
CREATE EXTENSION IF NOT EXISTS "timescaledb";
```

## Development Configuration

### PostgreSQL Configuration

```bash
# postgresql.conf for development
# /etc/postgresql/16/main/postgresql.conf

# Memory settings (adjust based on your system)
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# WAL settings
wal_level = replica
max_wal_size = 1GB
min_wal_size = 80MB
checkpoint_completion_target = 0.9

# Logging for development
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Development-specific settings
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
max_parallel_workers_per_gather = 2
max_parallel_workers = 8
max_parallel_maintenance_workers = 2

# Security (development only)
ssl = off
```

### Development Database Setup

```sql
-- Create development databases
CREATE DATABASE development;
CREATE DATABASE testing;
CREATE DATABASE staging;

-- Create development user with appropriate privileges
CREATE USER dev_user WITH PASSWORD 'dev_password';
GRANT ALL PRIVILEGES ON DATABASE development TO dev_user;
GRANT ALL PRIVILEGES ON DATABASE testing TO dev_user;
GRANT ALL PRIVILEGES ON DATABASE staging TO dev_user;

-- Grant schema privileges
\c development
GRANT ALL ON SCHEMA public TO dev_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dev_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dev_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO dev_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO dev_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO dev_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO dev_user;
```

## Development Tools

### Database Management Tools

```bash
# Install pgAdmin (GUI)
sudo apt install pgadmin4

# Install DBeaver (Cross-platform GUI)
wget https://dbeaver.io/files/dbeaver-ce_latest_amd64.deb
sudo dpkg -i dbeaver-ce_latest_amd64.deb

# Install psql (command line - usually included)
psql --version
```

### Command Line Tools

```bash
# Install additional PostgreSQL tools
sudo apt install postgresql-client-common

# Install pg_dump and pg_restore utilities
sudo apt install postgresql-client-16

# Install pgbench for performance testing
sudo apt install postgresql-contrib-16
```

### Development Scripts

```bash
#!/bin/bash
# scripts/dev-setup.sh

# Create development environment
echo "Setting up PostgreSQL development environment..."

# Create databases
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS development;"
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS testing;"

# Create user
psql -U postgres -c "CREATE USER IF NOT EXISTS dev_user WITH PASSWORD 'dev_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE development TO dev_user;"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE testing TO dev_user;"

# Install extensions
psql -U dev_user -d development -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql -U dev_user -d development -c "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";"
psql -U dev_user -d development -c "CREATE EXTENSION IF NOT EXISTS \"pg_stat_statements\";"

echo "Development environment setup complete!"
```

## Testing Environment

### Test Database Setup

```sql
-- Create test database with isolated schema
CREATE DATABASE testing;

-- Connect to test database
\c testing

-- Create test schema
CREATE SCHEMA test_schema;

-- Create test user
CREATE USER test_user WITH PASSWORD 'test_password';
GRANT ALL PRIVILEGES ON DATABASE testing TO test_user;
GRANT ALL PRIVILEGES ON SCHEMA test_schema TO test_user;

-- Set up test data
CREATE TABLE test_schema.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert test data
INSERT INTO test_schema.users (username, email) VALUES
    ('testuser1', 'test1@example.com'),
    ('testuser2', 'test2@example.com'),
    ('testuser3', 'test3@example.com');
```

### Automated Testing Setup

```python
# tests/conftest.py
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

@pytest.fixture(scope="session")
def test_db():
    """Create test database for the session."""
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    with conn.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS test_db;")
        cur.execute("CREATE DATABASE test_db;")
    
    conn.close()
    
    # Return connection to test database
    test_conn = psycopg2.connect(
        host="localhost",
        database="test_db",
        user="postgres",
        password="postgres"
    )
    
    yield test_conn
    test_conn.close()
    
    # Cleanup
    cleanup_conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    cleanup_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with cleanup_conn.cursor() as cur:
        cur.execute("DROP DATABASE test_db;")
    cleanup_conn.close()

@pytest.fixture
def test_table(test_db):
    """Create test table for each test."""
    with test_db.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                value INTEGER
            );
        """)
        test_db.commit()
    
    yield test_db
    
    # Cleanup after each test
    with test_db.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS test_table;")
        test_db.commit()
```

## Development Workflow

### Database Migrations

```python
# migrations/001_initial_schema.sql
-- Initial schema creation
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200) NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_published ON posts(published);
CREATE INDEX idx_posts_created_at ON posts(created_at);
```

### Schema Versioning

```sql
-- Create schema versioning table
CREATE TABLE schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial version
INSERT INTO schema_migrations (version) VALUES ('001_initial_schema');
```

### Development Data Management

```python
# scripts/seed_data.py
import psycopg2
import uuid
from datetime import datetime, timedelta
import random

def seed_development_data():
    """Seed development database with sample data."""
    conn = psycopg2.connect(
        host="localhost",
        database="development",
        user="dev_user",
        password="dev_password"
    )
    
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("TRUNCATE TABLE posts, users RESTART IDENTITY CASCADE;")
        
        # Insert sample users
        users_data = [
            ('john_doe', 'john@example.com', 'hashed_password_1'),
            ('jane_smith', 'jane@example.com', 'hashed_password_2'),
            ('bob_wilson', 'bob@example.com', 'hashed_password_3'),
            ('alice_brown', 'alice@example.com', 'hashed_password_4'),
            ('charlie_davis', 'charlie@example.com', 'hashed_password_5')
        ]
        
        for username, email, password_hash in users_data:
            cur.execute("""
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s)
            """, (username, email, password_hash))
        
        # Insert sample posts
        post_titles = [
            'Getting Started with PostgreSQL',
            'Advanced SQL Techniques',
            'Database Design Best Practices',
            'Performance Optimization Tips',
            'Security Considerations',
            'Backup and Recovery Strategies',
            'Monitoring and Maintenance',
            'Scaling PostgreSQL',
            'Replication Setup',
            'Troubleshooting Common Issues'
        ]
        
        for i in range(50):
            user_id = random.randint(1, 5)
            title = random.choice(post_titles)
            content = f"This is sample content for post {i+1}. " * 10
            published = random.choice([True, False])
            created_at = datetime.now() - timedelta(days=random.randint(0, 30))
            
            cur.execute("""
                INSERT INTO posts (user_id, title, content, published, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, title, content, published, created_at))
        
        conn.commit()
        print("Development data seeded successfully!")
    
    conn.close()

if __name__ == "__main__":
    seed_development_data()
```

## Performance Monitoring

### Query Performance Analysis

```sql
-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- View table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### Development Monitoring

```python
# monitoring/dev_monitor.py
import psycopg2
import time
import json
from datetime import datetime

class PostgreSQLMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(**self.conn_params)
    
    def get_connection_stats(self):
        """Get connection statistics."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity;
            """)
            return cur.fetchone()
    
    def get_database_size(self):
        """Get database size."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database()));
            """)
            return cur.fetchone()[0]
    
    def get_table_sizes(self):
        """Get table sizes."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """)
            return cur.fetchall()
    
    def monitor_loop(self, interval=60):
        """Run monitoring loop."""
        while True:
            try:
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'connections': self.get_connection_stats(),
                    'database_size': self.get_database_size(),
                    'table_sizes': self.get_table_sizes()
                }
                
                print(json.dumps(stats, indent=2))
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)

# Usage
if __name__ == "__main__":
    monitor = PostgreSQLMonitor({
        'host': 'localhost',
        'database': 'development',
        'user': 'dev_user',
        'password': 'dev_password'
    })
    
    monitor.connect()
    monitor.monitor_loop(interval=30)
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Install PostgreSQL
sudo apt install postgresql-16 postgresql-client-16 postgresql-contrib-16

# 2. Create development databases
psql -U postgres -c "CREATE DATABASE development;"
psql -U postgres -c "CREATE DATABASE testing;"

# 3. Create development user
psql -U postgres -c "CREATE USER dev_user WITH PASSWORD 'dev_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE development TO dev_user;"

# 4. Install extensions
psql -U dev_user -d development -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql -U dev_user -d development -c "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";"
```

### Essential Patterns

```python
# Complete PostgreSQL development setup
def setup_postgresql_development():
    # 1. PostgreSQL installation
    # 2. Version management
    # 3. Essential extensions
    # 4. Development configuration
    # 5. Testing environment
    # 6. Development tools
    # 7. Performance monitoring
    # 8. Data management
    
    print("PostgreSQL development environment setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL development excellence. Each pattern includes implementation examples, configuration strategies, and real-world usage patterns for enterprise PostgreSQL development systems.*
