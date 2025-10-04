# Database Migrations & Schema Evolution

**Objective**: Databases are living systems. Schema changes are inevitable. Handle them without breaking prod, corrupting data, or waking ops in the night.

Databases are living systems. Schema changes are inevitable. Handle them without breaking prod, corrupting data, or waking ops in the night.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Schema evolution is continuous, not one-off**
   - Plan for ongoing changes
   - Design for evolution from day one
   - Never assume "this is the final schema"
   - Embrace change as a first-class concern

2. **Always test migrations in staging with prod-like data**
   - Use production data snapshots (sanitized)
   - Test with realistic data volumes
   - Validate performance impact
   - Never test migrations in production

3. **Favor additive changes over destructive ones**
   - Backward-compatible migrations
   - Add before you remove
   - Use feature flags for schema changes
   - Plan rollback strategies

4. **Write migrations as code, versioned in Git**
   - No manual schema changes in production
   - Version control everything
   - Review migrations like code
   - Track migration history

5. **Rollbacks must be possible, but roll-forwards are usually safer**
   - Design for forward migration
   - Plan rollback strategies
   - Test both directions
   - Document rollback procedures

**Why These Principles**: Database migrations require understanding schema evolution, data safety, and production operations. Understanding these patterns prevents data chaos and enables reliable database management.

## 1) Core Principles

### The Migration Reality

```yaml
# What you thought migrations were
migration_fantasy:
  "simplicity": "Just run ALTER TABLE and you're done"
  "speed": "Schema changes are instant"
  "safety": "Nothing can go wrong"
  "rollback": "Just run the reverse SQL"

# What migrations actually are
migration_reality:
  "simplicity": "Complex orchestration of multiple steps"
  "speed": "Can take hours on large tables"
  "safety": "One wrong move and you're toast"
  "rollback": "Often impossible or extremely expensive"
```

**Why Reality Checks Matter**: Understanding the true nature of database migrations enables proper planning and risk management. Understanding these patterns prevents data chaos and enables reliable database management.

### Migration Lifecycle

```markdown
## Migration Lifecycle

### Planning Phase
- [ ] Analyze current schema and data
- [ ] Design migration strategy
- [ ] Plan rollback strategy
- [ ] Estimate downtime and performance impact

### Development Phase
- [ ] Write migration scripts
- [ ] Test on development data
- [ ] Validate with staging data
- [ ] Review migration code

### Deployment Phase
- [ ] Backup production database
- [ ] Run migration in maintenance window
- [ ] Validate migration results
- [ ] Monitor system performance

### Post-Deployment Phase
- [ ] Verify application functionality
- [ ] Monitor for issues
- [ ] Clean up old schema elements
- [ ] Document lessons learned
```

**Why Lifecycle Management Matters**: Proper migration lifecycle enables safe schema evolution and reduces production risks. Understanding these patterns prevents data chaos and enables reliable database management.

## 2) Tooling

### Alembic (Python/SQLAlchemy)

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add your model's MetaData object here
from myapp.models import Base
target_metadata = Base.metadata

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Why Alembic Matters**: Alembic provides version-controlled database migrations with rollback support and dependency management. Understanding these patterns prevents migration chaos and enables reliable database management.

### Alembic Configuration

```ini
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://user:pass@localhost/mydb

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Why Alembic Configuration Matters**: Proper Alembic configuration enables consistent migration management and logging. Understanding these patterns prevents migration chaos and enables reliable database management.

### Flyway (Language-Agnostic)

```sql
-- V1__Create_users_table.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- V2__Add_user_profiles.sql
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    bio TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);

-- V3__Add_geospatial_support.sql
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Add geometry column to user_profiles
ALTER TABLE user_profiles 
ADD COLUMN location GEOMETRY(POINT, 4326);

-- Create spatial index
CREATE INDEX idx_user_profiles_location ON user_profiles 
USING GIST (location);

-- V4__Add_telemetry_table.sql
CREATE TABLE telemetry (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sensor_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create time-series optimized indexes
CREATE INDEX idx_telemetry_timestamp ON telemetry (timestamp DESC);
CREATE INDEX idx_telemetry_user_timestamp ON telemetry (user_id, timestamp DESC);
CREATE INDEX idx_telemetry_location ON telemetry USING GIST (location);

-- Partition by time for better performance
CREATE TABLE telemetry_2024_01 PARTITION OF telemetry
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

**Why Flyway Matters**: Flyway provides simple, version-controlled database migrations with rollback support. Understanding these patterns prevents migration chaos and enables reliable database management.

### Sqitch (Dependency-Aware)

```sql
-- sqitch.plan
%syntax-version=1.0.0
%project=myapp
%uri=https://github.com/myorg/myapp

users 2024-01-15T10:00:00Z "Create users table" [main]
user_profiles 2024-01-15T10:30:00Z "Add user profiles" [users]
geospatial 2024-01-15T11:00:00Z "Add geospatial support" [user_profiles]
telemetry 2024-01-15T11:30:00Z "Add telemetry table" [users geospatial]

-- deploy/users.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- revert/users.sql
DROP TABLE users CASCADE;

-- verify/users.sql
SELECT 1 FROM users LIMIT 1;
```

**Why Sqitch Matters**: Sqitch provides dependency-aware migrations with explicit rollback and verification steps. Understanding these patterns prevents migration chaos and enables reliable database management.

## 3) Zero-Downtime Strategies

### Additive Changes (Safe)

```python
# Migration: Add new column without breaking existing code
"""Add user preferences table

Revision ID: 001_add_user_preferences
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001_add_user_preferences'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Step 1: Add new table
    op.create_table('user_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('theme', sa.String(50), nullable=True, default='light'),
        sa.Column('notifications', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Step 2: Create indexes
    op.create_index('idx_user_preferences_user_id', 'user_preferences', ['user_id'])
    
    # Step 3: Add default preferences for existing users
    op.execute("""
        INSERT INTO user_preferences (user_id, theme, notifications, created_at, updated_at)
        SELECT id, 'light', true, NOW(), NOW()
        FROM users
        WHERE NOT EXISTS (
            SELECT 1 FROM user_preferences WHERE user_id = users.id
        )
    """)

def downgrade():
    # Step 1: Drop indexes
    op.drop_index('idx_user_preferences_user_id', table_name='user_preferences')
    
    # Step 2: Drop table
    op.drop_table('user_preferences')
```

**Why Additive Changes Matter**: Additive changes enable safe schema evolution without breaking existing functionality. Understanding these patterns prevents production chaos and enables reliable database management.

### Column Rename Strategy (Zero-Downtime)

```python
# Migration: Rename column with zero downtime
"""Rename username to user_handle

Revision ID: 002_rename_username_to_user_handle
Revises: 001_add_user_preferences
Create Date: 2024-01-15 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '002_rename_username_to_user_handle'
down_revision = '001_add_user_preferences'
branch_labels = None
depends_on = None

def upgrade():
    # Step 1: Add new column
    op.add_column('users', sa.Column('user_handle', sa.String(50), nullable=True))
    
    # Step 2: Copy data from old column to new column
    op.execute("UPDATE users SET user_handle = username WHERE user_handle IS NULL")
    
    # Step 3: Make new column NOT NULL (after data is copied)
    op.alter_column('users', 'user_handle', nullable=False)
    
    # Step 4: Create unique index on new column
    op.create_index('idx_users_user_handle', 'users', ['user_handle'], unique=True)
    
    # Note: Don't drop old column yet - wait for application code to be updated

def downgrade():
    # Step 1: Drop new column index
    op.drop_index('idx_users_user_handle', table_name='users')
    
    # Step 2: Drop new column
    op.drop_column('users', 'user_handle')
```

**Why Column Rename Strategy Matters**: Zero-downtime column renames enable safe schema evolution without breaking existing applications. Understanding these patterns prevents production chaos and enables reliable database management.

### View-Based Compatibility Layer

```sql
-- Create view for backward compatibility during column rename
CREATE OR REPLACE VIEW users_compat AS
SELECT 
    id,
    username,  -- Old column name
    user_handle,  -- New column name
    email,
    created_at,
    updated_at
FROM users;

-- Grant permissions to application
GRANT SELECT ON users_compat TO app_user;

-- Later, after application is updated to use new column:
-- Drop the compatibility view
DROP VIEW users_compat;
```

**Why View-Based Compatibility Matters**: Views provide backward compatibility during schema changes without duplicating data. Understanding these patterns prevents application chaos and enables reliable database management.

## 4) PostGIS / Geospatial Considerations

### Concurrent Index Creation

```sql
-- Bad: Blocks table during index creation
CREATE INDEX idx_telemetry_location ON telemetry USING GIST (location);

-- Good: Creates index without blocking table
CREATE INDEX CONCURRENTLY idx_telemetry_location ON telemetry USING GIST (location);

-- For large tables, monitor progress
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE indexname = 'idx_telemetry_location';
```

**Why Concurrent Index Creation Matters**: Concurrent index creation prevents table locks on large tables, enabling zero-downtime schema changes. Understanding these patterns prevents production chaos and enables reliable database management.

### Spatial Reference System (SRID) Validation

```sql
-- Check SRID before altering geometry columns
SELECT 
    f_table_name,
    f_geometry_column,
    srid,
    type
FROM geometry_columns 
WHERE f_table_name = 'telemetry';

-- Validate SRID before migration
SELECT 
    ST_SRID(location) as current_srid,
    COUNT(*) as feature_count
FROM telemetry 
WHERE location IS NOT NULL
GROUP BY ST_SRID(location);

-- Update SRID if needed (careful with this!)
-- UPDATE telemetry SET location = ST_SetSRID(location, 4326) WHERE ST_SRID(location) != 4326;
```

**Why SRID Validation Matters**: SRID validation prevents geospatial data corruption during schema changes. Understanding these patterns prevents data chaos and enables reliable geospatial database management.

### Partitioning for Large Geospatial Tables

```sql
-- Create partitioned table for time-series geospatial data
CREATE TABLE telemetry_partitioned (
    id SERIAL,
    user_id INTEGER NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE telemetry_2024_01 PARTITION OF telemetry_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE telemetry_2024_02 PARTITION OF telemetry_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Create spatial indexes on each partition
CREATE INDEX CONCURRENTLY idx_telemetry_2024_01_location 
ON telemetry_2024_01 USING GIST (location);

CREATE INDEX CONCURRENTLY idx_telemetry_2024_02_location 
ON telemetry_2024_02 USING GIST (location);
```

**Why Partitioning Matters**: Partitioning enables efficient management of large geospatial datasets with reduced lock times. Understanding these patterns prevents performance chaos and enables reliable geospatial database management.

### FDW Schema Evolution

```sql
-- For parquet_s3_fdw: Schema changes mean metadata updates, not file rewrites
-- Create foreign table
CREATE FOREIGN TABLE telemetry_parquet (
    id INTEGER,
    user_id INTEGER,
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326)
) SERVER parquet_s3_server
OPTIONS (
    filename 's3://my-bucket/telemetry/2024/01/',
    format 'parquet'
);

-- When schema changes in source files:
-- 1. Update foreign table definition
ALTER FOREIGN TABLE telemetry_parquet 
ADD COLUMN new_field VARCHAR(100);

-- 2. Update server options if needed
ALTER SERVER parquet_s3_server 
OPTIONS (SET filename 's3://my-bucket/telemetry/2024/01/v2/');

-- 3. Refresh metadata
IMPORT FOREIGN SCHEMA public FROM SERVER parquet_s3_server INTO public;
```

**Why FDW Schema Evolution Matters**: FDW schema changes require metadata updates rather than file modifications, enabling efficient schema evolution. Understanding these patterns prevents integration chaos and enables reliable federated database management.

## 5) Testing & Validation

### Migration Testing with Docker

```yaml
# docker-compose.test.yml
version: '3.9'
services:
  postgres:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    ports:
      - "5432:5432"
    volumes:
      - ./test-data:/docker-entrypoint-initdb.d

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://testuser:testpass@postgres:5432/testdb
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    command: python -m pytest tests/migrations/
```

**Why Migration Testing Matters**: Automated migration testing prevents production failures and ensures schema changes work correctly. Understanding these patterns prevents deployment chaos and enables reliable database management.

### Migration Validation Scripts

```python
# tests/migrations/test_migration_validation.py
import pytest
import psycopg2
from sqlalchemy import create_engine, text
from alembic import command
from alembic.config import Config

class TestMigrationValidation:
    """Test that migrations work correctly and don't break data."""
    
    @pytest.fixture
    def db_engine(self):
        """Create test database engine."""
        return create_engine("postgresql://testuser:testpass@localhost:5432/testdb")
    
    @pytest.fixture
    def alembic_cfg(self):
        """Create Alembic configuration."""
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", "postgresql://testuser:testpass@localhost:5432/testdb")
        return config
    
    def test_migration_upgrade(self, db_engine, alembic_cfg):
        """Test that migration upgrade works."""
        # Run migration
        command.upgrade(alembic_cfg, "head")
        
        # Validate schema
        with db_engine.connect() as conn:
            # Check tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            assert 'users' in tables
            assert 'user_preferences' in tables
            assert 'telemetry' in tables
    
    def test_migration_downgrade(self, db_engine, alembic_cfg):
        """Test that migration downgrade works."""
        # Upgrade first
        command.upgrade(alembic_cfg, "head")
        
        # Then downgrade
        command.downgrade(alembic_cfg, "base")
        
        # Validate schema is clean
        with db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            assert len(tables) == 0
    
    def test_data_integrity(self, db_engine, alembic_cfg):
        """Test that data integrity is maintained during migration."""
        # Insert test data before migration
        with db_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (username, email) 
                VALUES ('testuser', 'test@example.com')
            """))
            conn.commit()
        
        # Run migration
        command.upgrade(alembic_cfg, "head")
        
        # Validate data is still there
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            count = result.scalar()
            assert count == 1
    
    def test_geospatial_data(self, db_engine, alembic_cfg):
        """Test that geospatial data is handled correctly."""
        # Run migration
        command.upgrade(alembic_cfg, "head")
        
        # Insert geospatial test data
        with db_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO telemetry (user_id, sensor_id, timestamp, location)
                VALUES (1, 'sensor_001', NOW(), ST_GeomFromText('POINT(-122.4194 37.7749)', 4326))
            """))
            conn.commit()
        
        # Validate geospatial data
        with db_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT ST_AsText(location), ST_SRID(location)
                FROM telemetry
                WHERE sensor_id = 'sensor_001'
            """))
            row = result.fetchone()
            assert row[0] == 'POINT(-122.4194 37.7749)'
            assert row[1] == 4326
```

**Why Migration Validation Matters**: Comprehensive migration testing prevents data corruption and ensures schema changes work correctly. Understanding these patterns prevents production chaos and enables reliable database management.

## 6) Automation in CI/CD

### GitHub Actions Migration Pipeline

```yaml
# .github/workflows/migrations.yml
name: Database Migrations

on:
  push:
    branches: [ main, develop ]
    paths: [ 'alembic/**', 'migrations/**' ]
  pull_request:
    branches: [ main ]
    paths: [ 'alembic/**', 'migrations/**' ]

jobs:
  test-migrations:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgis/postgis:15-3.4
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run migration tests
      run: |
        pytest tests/migrations/ -v
      env:
        DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
    
    - name: Test migration dry-run
      run: |
        alembic upgrade head --sql > migration.sql
        echo "Migration SQL generated successfully"
    
    - name: Validate migration SQL
      run: |
        # Check for dangerous operations
        if grep -i "drop table" migration.sql; then
          echo "WARNING: DROP TABLE found in migration"
          exit 1
        fi
        
        if grep -i "alter table.*drop column" migration.sql; then
          echo "WARNING: DROP COLUMN found in migration"
          exit 1
        fi
        
        echo "Migration SQL validation passed"

  deploy-migrations:
    runs-on: ubuntu-latest
    needs: test-migrations
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run production migration
      run: |
        alembic upgrade head
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
    
    - name: Validate migration results
      run: |
        python scripts/validate_migration.py
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

**Why CI/CD Migration Automation Matters**: Automated migration testing and deployment prevents production failures and ensures consistent schema changes. Understanding these patterns prevents deployment chaos and enables reliable database management.

### Migration Safety Checks

```python
# scripts/validate_migration.py
import os
import sys
import psycopg2
from sqlalchemy import create_engine, text

def validate_migration():
    """Validate that migration completed successfully."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Check that all tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]
        expected_tables = ['users', 'user_preferences', 'telemetry']
        
        for table in expected_tables:
            if table not in tables:
                print(f"ERROR: Table {table} not found")
                sys.exit(1)
        
        # Check row counts
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.scalar()
        print(f"Users: {user_count}")
        
        # Check geospatial data integrity
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM telemetry 
            WHERE location IS NOT NULL
        """))
        geo_count = result.scalar()
        print(f"Geospatial records: {geo_count}")
        
        # Check SRID consistency
        result = conn.execute(text("""
            SELECT DISTINCT ST_SRID(location) as srid
            FROM telemetry 
            WHERE location IS NOT NULL
        """))
        srids = [row[0] for row in result]
        if len(srids) > 1:
            print(f"WARNING: Multiple SRIDs found: {srids}")
        else:
            print(f"SRID consistency: {srids[0] if srids else 'No geospatial data'}")
        
        print("Migration validation completed successfully")

if __name__ == "__main__":
    validate_migration()
```

**Why Migration Safety Checks Matter**: Post-migration validation ensures data integrity and schema consistency. Understanding these patterns prevents data chaos and enables reliable database management.

## 7) Anti-Patterns

### Common Migration Mistakes

```yaml
# What NOT to do
migration_anti_patterns:
  "manual_prod_changes": "Never apply manual schema changes in production",
  "no_version_control": "Never skip version control for migrations",
  "destructive_changes": "Never make destructive changes without planning",
  "peak_hour_changes": "Never apply schema changes during peak hours",
  "no_testing": "Never skip testing migrations",
  "no_rollback_plan": "Never deploy without rollback plan",
  "fdw_static": "Never treat FDW schemas as static",
  "no_backup": "Never skip database backup before migration",
  "no_monitoring": "Never skip monitoring during migration",
  "no_validation": "Never skip post-migration validation"
```

**Why Anti-Patterns Matter**: Understanding common mistakes prevents production failures and data corruption. Understanding these patterns prevents migration chaos and enables reliable database management.

### Migration Horror Stories

```markdown
## Migration Horror Stories

### The DROP TABLE Incident
**What happened**: Developer ran `DROP TABLE users` in production
**Result**: Lost all user data, 4-hour outage
**Lesson**: Always backup before destructive operations

### The Index Lock Disaster
**What happened**: Created index on 100M row table without CONCURRENTLY
**Result**: 2-hour table lock, application down
**Lesson**: Use CONCURRENTLY for large table indexes

### The SRID Mismatch
**What happened**: Changed SRID without updating existing data
**Result**: All geospatial queries returned wrong results
**Lesson**: Always validate SRID consistency

### The FDW Metadata Sync
**What happened**: Changed parquet schema without updating FDW metadata
**Result**: All federated queries failed
**Lesson**: FDW schemas need metadata updates, not file changes
```

**Why Horror Stories Matter**: Learning from others' mistakes prevents similar failures and improves migration practices. Understanding these patterns prevents production chaos and enables reliable database management.

## 8) TL;DR Runbook

### Essential Commands

```bash
# Create new migration
alembic revision -m "Add user preferences table"

# Run migration
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check migration status
alembic current

# Generate migration SQL (dry run)
alembic upgrade head --sql

# Test migration on staging
alembic upgrade head --sql > migration.sql
psql staging_db < migration.sql
```

### Essential Patterns

```yaml
# Essential migration patterns
migration_patterns:
  "version_control": "Version-control all migrations (Alembic/Flyway)",
  "additive_changes": "Prefer additive over destructive changes",
  "test_snapshots": "Test on prod-like data snapshots",
  "zero_downtime": "Use zero-downtime strategies (add → backfill → drop)",
  "concurrent_indexes": "Always index concurrently in PostGIS",
  "ci_cd_automation": "Automate in CI/CD with dry-runs + PR reviews",
  "backup_first": "Always backup before migration",
  "monitor_during": "Monitor during migration execution",
  "validate_after": "Validate results after migration",
  "rollback_ready": "Have rollback plan ready"
```

### Quick Reference

```markdown
## Emergency Migration Response

### If Migration Fails
1. **Stop the migration immediately**
2. **Check database state**
3. **Restore from backup if needed**
4. **Analyze failure cause**
5. **Fix and retry**

### If Data is Corrupted
1. **Stop all applications**
2. **Restore from backup**
3. **Investigate corruption cause**
4. **Fix migration script**
5. **Test thoroughly before retry**

### If Rollback is Needed
1. **Stop all applications**
2. **Run rollback migration**
3. **Validate data integrity**
4. **Restart applications**
5. **Monitor for issues**
```

**Why This Runbook**: These patterns cover 90% of migration needs. Master these before exploring advanced migration scenarios.

## 9) The Machine's Summary

Database migrations require understanding schema evolution, data safety, and production operations. When used correctly, effective migrations enable safe schema changes, prevent data corruption, and maintain system reliability. The key is understanding zero-downtime strategies, proper testing, and automation.

**The Dark Truth**: Without proper migration practices, your database becomes a liability. Database migrations are your weapon. Use them wisely.

**The Machine's Mantra**: "In the additive changes we trust, in the testing we find safety, and in the automation we find the path to reliable schema evolution."

**Why This Matters**: Database migrations enable schema evolution that can handle complex data systems, prevent data corruption, and provide insights into database design while ensuring technical accuracy and reliability.

---

*This guide provides the complete machinery for database migrations. The patterns scale from simple table changes to complex geospatial schema evolution, from basic Alembic usage to advanced zero-downtime strategies.*
