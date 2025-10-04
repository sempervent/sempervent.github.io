# PostgreSQL Database Design Best Practices

**Objective**: Master senior-level PostgreSQL database design patterns for production systems. When you need to design efficient schemas, when you want to optimize data relationships, when you need enterprise-grade database design strategies—these best practices become your weapon of choice.

## Core Principles

- **Normalization**: Balance between normalization and performance
- **Data Integrity**: Enforce constraints and relationships
- **Performance**: Design for query patterns and access patterns
- **Scalability**: Plan for growth and horizontal scaling
- **Maintainability**: Design for long-term maintenance and evolution

## Schema Design Patterns

### Entity-Relationship Design

```sql
-- Create core entities with proper relationships
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE organizations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    website VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_organizations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    is_active BOOLEAN DEFAULT TRUE,
    joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, organization_id)
);

CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    organization_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE project_members (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'contributor',
    permissions JSONB,
    joined_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, user_id)
);
```

### Audit Trail Design

```sql
-- Create audit trail table
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER NOT NULL,
    operation VARCHAR(20) NOT NULL, -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    changed_by INTEGER REFERENCES users(id),
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    client_ip INET,
    user_agent TEXT
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, operation, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, to_jsonb(NEW), current_setting('app.current_user_id')::integer);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, new_values, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, to_jsonb(OLD), to_jsonb(NEW), current_setting('app.current_user_id')::integer);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id, TG_OP, to_jsonb(OLD), current_setting('app.current_user_id')::integer);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers
CREATE TRIGGER users_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER projects_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON projects
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

## Data Modeling Patterns

### Hierarchical Data Design

```sql
-- Create hierarchical categories table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
    path LTREE,
    level INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to update path and level
CREATE OR REPLACE FUNCTION update_category_path()
RETURNS TRIGGER AS $$
DECLARE
    parent_path LTREE;
    parent_level INTEGER;
BEGIN
    IF NEW.parent_id IS NULL THEN
        NEW.path := NEW.id::text::ltree;
        NEW.level := 0;
    ELSE
        SELECT path, level INTO parent_path, parent_level
        FROM categories WHERE id = NEW.parent_id;
        
        NEW.path := parent_path || NEW.id::text;
        NEW.level := parent_level + 1;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update path
CREATE TRIGGER update_categories_path
    BEFORE INSERT OR UPDATE ON categories
    FOR EACH ROW EXECUTE FUNCTION update_category_path();

-- Insert sample hierarchical data
INSERT INTO categories (name, parent_id) VALUES
    ('Technology', NULL),
    ('Programming', 1),
    ('Web Development', 2),
    ('Frontend', 3),
    ('Backend', 3),
    ('Database', 2),
    ('PostgreSQL', 6),
    ('MySQL', 6);

-- Query hierarchical data
SELECT 
    c.id,
    c.name,
    c.path,
    c.level,
    REPEAT('  ', c.level) || c.name as indented_name
FROM categories c
ORDER BY c.path;
```

### Polymorphic Associations

```sql
-- Create polymorphic associations for comments
CREATE TABLE commentable_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

INSERT INTO commentable_types (name) VALUES ('post'), ('project'), ('user');

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    commentable_type VARCHAR(50) NOT NULL,
    commentable_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES comments(id) ON DELETE CASCADE,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(commentable_type, commentable_id, author_id, created_at)
);

-- Create function to validate polymorphic associations
CREATE OR REPLACE FUNCTION validate_commentable_reference()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate that the referenced record exists
    IF NEW.commentable_type = 'post' THEN
        IF NOT EXISTS (SELECT 1 FROM posts WHERE id = NEW.commentable_id) THEN
            RAISE EXCEPTION 'Referenced post does not exist';
        END IF;
    ELSIF NEW.commentable_type = 'project' THEN
        IF NOT EXISTS (SELECT 1 FROM projects WHERE id = NEW.commentable_id) THEN
            RAISE EXCEPTION 'Referenced project does not exist';
        END IF;
    ELSIF NEW.commentable_type = 'user' THEN
        IF NOT EXISTS (SELECT 1 FROM users WHERE id = NEW.commentable_id) THEN
            RAISE EXCEPTION 'Referenced user does not exist';
        END IF;
    ELSE
        RAISE EXCEPTION 'Invalid commentable_type: %', NEW.commentable_type;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_commentable_reference_trigger
    BEFORE INSERT OR UPDATE ON comments
    FOR EACH ROW EXECUTE FUNCTION validate_commentable_reference();
```

## Performance-Oriented Design

### Denormalization Strategies

```sql
-- Create denormalized views for performance
CREATE MATERIALIZED VIEW user_project_summary AS
SELECT 
    u.id as user_id,
    u.username,
    u.email,
    COUNT(p.id) as project_count,
    COUNT(pm.id) as member_count,
    MAX(p.created_at) as latest_project_created,
    ARRAY_AGG(DISTINCT o.name) as organization_names
FROM users u
LEFT JOIN projects p ON u.id = p.owner_id
LEFT JOIN project_members pm ON u.id = pm.user_id
LEFT JOIN organizations o ON p.organization_id = o.id
GROUP BY u.id, u.username, u.email;

-- Create index on materialized view
CREATE INDEX idx_user_project_summary_username ON user_project_summary (username);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_user_project_summary()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_project_summary;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-refresh materialized view
CREATE OR REPLACE FUNCTION trigger_refresh_user_project_summary()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM refresh_user_project_summary();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_user_project_summary_trigger
    AFTER INSERT OR UPDATE OR DELETE ON projects
    FOR EACH STATEMENT EXECUTE FUNCTION trigger_refresh_user_project_summary();
```

### Partitioning Design

```sql
-- Create partitioned table for events
CREATE TABLE events (
    id BIGSERIAL,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL,
    data JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Create indexes on partitions
CREATE INDEX idx_events_2024_01_user_id ON events_2024_01 (user_id);
CREATE INDEX idx_events_2024_01_event_type ON events_2024_01 (event_type);
CREATE INDEX idx_events_2024_01_created_at ON events_2024_01 (created_at);

-- Create function to auto-create partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
    
    -- Create indexes on new partition
    EXECUTE format('CREATE INDEX %I ON %I (user_id)', 
                   'idx_' || partition_name || '_user_id', partition_name);
    EXECUTE format('CREATE INDEX %I ON %I (event_type)', 
                   'idx_' || partition_name || '_event_type', partition_name);
    EXECUTE format('CREATE INDEX %I ON %I (created_at)', 
                   'idx_' || partition_name || '_created_at', partition_name);
END;
$$ LANGUAGE plpgsql;
```

## Data Integrity Patterns

### Constraint Design

```sql
-- Create comprehensive constraints
ALTER TABLE users ADD CONSTRAINT users_email_format 
    CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

ALTER TABLE users ADD CONSTRAINT users_username_length 
    CHECK (LENGTH(username) >= 3 AND LENGTH(username) <= 50);

ALTER TABLE projects ADD CONSTRAINT projects_status_valid 
    CHECK (status IN ('active', 'inactive', 'archived', 'deleted'));

ALTER TABLE project_members ADD CONSTRAINT project_members_role_valid 
    CHECK (role IN ('owner', 'admin', 'contributor', 'viewer'));

-- Create custom constraint for business rules
CREATE OR REPLACE FUNCTION validate_project_owner()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure project owner is a member of the organization
    IF NOT EXISTS (
        SELECT 1 FROM user_organizations uo
        WHERE uo.user_id = NEW.owner_id 
        AND uo.organization_id = NEW.organization_id
        AND uo.is_active = TRUE
    ) THEN
        RAISE EXCEPTION 'Project owner must be a member of the organization';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_project_owner_trigger
    BEFORE INSERT OR UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION validate_project_owner();
```

### Referential Integrity

```sql
-- Create soft delete pattern
CREATE TABLE soft_deletable (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    deleted_at TIMESTAMPTZ,
    deleted_by INTEGER REFERENCES users(id),
    is_deleted BOOLEAN GENERATED ALWAYS AS (deleted_at IS NOT NULL) STORED
);

-- Create function for soft delete
CREATE OR REPLACE FUNCTION soft_delete_record(
    table_name TEXT,
    record_id INTEGER,
    deleted_by_user_id INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format('UPDATE %I SET deleted_at = CURRENT_TIMESTAMP, deleted_by = %s WHERE id = %s',
                   table_name, deleted_by_user_id, record_id);
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Create view to exclude soft-deleted records
CREATE VIEW active_projects AS
SELECT * FROM projects WHERE deleted_at IS NULL;

-- Create function to restore soft-deleted records
CREATE OR REPLACE FUNCTION restore_record(
    table_name TEXT,
    record_id INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format('UPDATE %I SET deleted_at = NULL, deleted_by = NULL WHERE id = %s',
                   table_name, record_id);
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;
```

## Schema Evolution Patterns

### Migration Management

```sql
-- Create schema versioning table
CREATE TABLE schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Create function to track schema changes
CREATE OR REPLACE FUNCTION track_schema_change(
    change_description TEXT
)
RETURNS VOID AS $$
DECLARE
    current_version VARCHAR(50);
BEGIN
    current_version := to_char(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    
    INSERT INTO schema_migrations (version, description)
    VALUES (current_version, change_description);
    
    RAISE NOTICE 'Schema change tracked: % - %', current_version, change_description;
END;
$$ LANGUAGE plpgsql;

-- Example schema migration
DO $$
BEGIN
    -- Add new column
    ALTER TABLE users ADD COLUMN phone VARCHAR(20);
    
    -- Track the change
    PERFORM track_schema_change('Add phone column to users table');
END $$;
```

### Backward Compatibility

```sql
-- Create function to handle schema evolution gracefully
CREATE OR REPLACE FUNCTION migrate_user_data()
RETURNS VOID AS $$
BEGIN
    -- Add new columns with defaults
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'phone') THEN
        ALTER TABLE users ADD COLUMN phone VARCHAR(20);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'users' AND column_name = 'preferences') THEN
        ALTER TABLE users ADD COLUMN preferences JSONB DEFAULT '{}';
    END IF;
    
    -- Migrate existing data
    UPDATE users 
    SET preferences = '{"theme": "light", "notifications": true}'::jsonb
    WHERE preferences IS NULL;
    
    RAISE NOTICE 'User data migration completed';
END;
$$ LANGUAGE plpgsql;
```

## Design Validation

### Schema Validation Tools

```python
# validation/schema_validator.py
import psycopg2
import json
from typing import Dict, List, Any

class DatabaseSchemaValidator:
    def __init__(self, connection_params):
        self.conn_params = connection_params
    
    def validate_schema_design(self) -> Dict[str, Any]:
        """Validate database schema design."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Check for missing indexes
                missing_indexes = self.check_missing_indexes(cur)
                
                # Check for foreign key constraints
                fk_constraints = self.check_foreign_keys(cur)
                
                # Check for data types
                data_types = self.check_data_types(cur)
                
                # Check for naming conventions
                naming_issues = self.check_naming_conventions(cur)
                
                return {
                    'missing_indexes': missing_indexes,
                    'foreign_keys': fk_constraints,
                    'data_types': data_types,
                    'naming_issues': naming_issues,
                    'validation_timestamp': '2024-01-15T10:30:00Z'
                }
                
        except Exception as e:
            print(f"Error validating schema: {e}")
            return {}
        finally:
            conn.close()
    
    def check_missing_indexes(self, cursor) -> List[Dict[str, Any]]:
        """Check for missing indexes on foreign keys."""
        cursor.execute("""
            SELECT 
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND NOT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE tablename = tc.table_name 
                AND indexdef LIKE '%' || kcu.column_name || '%'
            )
        """)
        
        missing_indexes = []
        for row in cursor.fetchall():
            missing_indexes.append({
                'table': row[0],
                'column': row[1],
                'foreign_table': row[2],
                'foreign_column': row[3]
            })
        
        return missing_indexes
    
    def check_foreign_keys(self, cursor) -> List[Dict[str, Any]]:
        """Check foreign key constraints."""
        cursor.execute("""
            SELECT 
                tc.table_name,
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                rc.delete_rule,
                rc.update_rule
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            JOIN information_schema.referential_constraints AS rc
                ON tc.constraint_name = rc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)
        
        fk_constraints = []
        for row in cursor.fetchall():
            fk_constraints.append({
                'table': row[0],
                'constraint': row[1],
                'column': row[2],
                'foreign_table': row[3],
                'foreign_column': row[4],
                'delete_rule': row[5],
                'update_rule': row[6]
            })
        
        return fk_constraints
    
    def check_data_types(self, cursor) -> List[Dict[str, Any]]:
        """Check data type usage."""
        cursor.execute("""
            SELECT 
                table_name,
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        
        data_types = []
        for row in cursor.fetchall():
            data_types.append({
                'table': row[0],
                'column': row[1],
                'type': row[2],
                'max_length': row[3],
                'nullable': row[4],
                'default': row[5]
            })
        
        return data_types
    
    def check_naming_conventions(self, cursor) -> List[Dict[str, Any]]:
        """Check naming convention violations."""
        cursor.execute("""
            SELECT 
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND (
                column_name LIKE '% %' OR
                column_name LIKE '%-%' OR
                column_name ~ '[A-Z]'
            )
        """)
        
        naming_issues = []
        for row in cursor.fetchall():
            naming_issues.append({
                'table': row[0],
                'column': row[1],
                'type': row[2],
                'issue': 'Naming convention violation'
            })
        
        return naming_issues

# Usage
if __name__ == "__main__":
    validator = DatabaseSchemaValidator({
        'host': 'localhost',
        'database': 'production',
        'user': 'validator_user',
        'password': 'validator_password'
    })
    
    validation_results = validator.validate_schema_design()
    print(json.dumps(validation_results, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create core entities with proper relationships
CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL, email VARCHAR(100) UNIQUE NOT NULL);
CREATE TABLE organizations (id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL);
CREATE TABLE projects (id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL, organization_id INTEGER REFERENCES organizations(id));

-- 2. Add constraints and indexes
ALTER TABLE users ADD CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');
CREATE INDEX idx_projects_organization_id ON projects (organization_id);

-- 3. Create audit trail
CREATE TABLE audit_log (id BIGSERIAL PRIMARY KEY, table_name VARCHAR(100), operation VARCHAR(20), old_values JSONB, new_values JSONB, changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);
```

### Essential Patterns

```python
# Complete PostgreSQL database design setup
def setup_postgresql_database_design():
    # 1. Entity-relationship design
    # 2. Data modeling patterns
    # 3. Performance-oriented design
    # 4. Data integrity patterns
    # 5. Schema evolution
    # 6. Design validation
    # 7. Migration management
    # 8. Backward compatibility
    
    print("PostgreSQL database design setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL database design excellence. Each pattern includes implementation examples, design strategies, and real-world usage patterns for enterprise PostgreSQL database systems.*
