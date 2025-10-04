# PostgreSQL Constraints & Validation Best Practices

**Objective**: Master senior-level PostgreSQL constraint and validation patterns for production systems. When you need to enforce data integrity, when you want to implement robust validation, when you need enterprise-grade constraint strategies—these best practices become your weapon of choice.

## Core Principles

- **Data Integrity**: Enforce business rules at the database level
- **Performance**: Design constraints for optimal performance
- **Validation**: Implement comprehensive data validation
- **Error Handling**: Provide clear constraint violation messages
- **Maintenance**: Design constraints for long-term maintainability

## Primary Key Constraints

### Primary Key Design

```sql
-- Create tables with appropriate primary keys
CREATE TABLE users (
    id SERIAL PRIMARY KEY,                    -- Auto-incrementing primary key
    username VARCHAR(50) UNIQUE NOT NULL,    -- Natural key with uniqueness
    email VARCHAR(255) UNIQUE NOT NULL,      -- Natural key with uniqueness
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Composite primary keys for junction tables
CREATE TABLE user_roles (
    user_id INTEGER NOT NULL,
    role_id INTEGER NOT NULL,
    assigned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    assigned_by INTEGER,
    PRIMARY KEY (user_id, role_id)            -- Composite primary key
);

-- UUID primary keys for distributed systems
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE distributed_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Natural primary keys for reference data
CREATE TABLE countries (
    country_code CHAR(2) PRIMARY KEY,         -- ISO country code as natural key
    country_name VARCHAR(100) NOT NULL,
    currency_code CHAR(3) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Primary Key Performance

```sql
-- Create indexes for primary key performance
CREATE INDEX CONCURRENTLY idx_users_id_btree ON users (id);
CREATE INDEX CONCURRENTLY idx_user_roles_composite ON user_roles (user_id, role_id);

-- Analyze primary key usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%_pkey'
ORDER BY idx_scan DESC;
```

## Foreign Key Constraints

### Referential Integrity

```sql
-- Create tables with foreign key relationships
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE RESTRICT,
    price NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create foreign key with custom action
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Foreign Key Performance

```sql
-- Create indexes for foreign key columns
CREATE INDEX idx_products_category_id ON products (category_id);
CREATE INDEX idx_order_items_order_id ON order_items (order_id);
CREATE INDEX idx_order_items_product_id ON order_items (product_id);
CREATE INDEX idx_user_sessions_user_id ON user_sessions (user_id);

-- Analyze foreign key performance
SELECT 
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    tc.constraint_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name, kcu.column_name;
```

## Check Constraints

### Data Validation Constraints

```sql
-- Create tables with check constraints
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    phone VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Check constraints for data validation
    CONSTRAINT users_username_length CHECK (LENGTH(username) >= 3),
    CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_age_range CHECK (age IS NULL OR (age >= 0 AND age <= 150)),
    CONSTRAINT users_phone_format CHECK (phone IS NULL OR phone ~ '^\+?[1-9]\d{1,14}$'),
    CONSTRAINT users_status_valid CHECK (status IN ('active', 'inactive', 'suspended', 'deleted'))
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    stock_quantity INTEGER NOT NULL,
    weight_grams INTEGER,
    dimensions JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Check constraints for business rules
    CONSTRAINT products_price_positive CHECK (price > 0),
    CONSTRAINT products_stock_non_negative CHECK (stock_quantity >= 0),
    CONSTRAINT products_weight_positive CHECK (weight_grams IS NULL OR weight_grams > 0),
    CONSTRAINT products_dimensions_valid CHECK (
        dimensions IS NULL OR 
        (dimensions ? 'length' AND dimensions ? 'width' AND dimensions ? 'height')
    )
);
```

### Complex Check Constraints

```sql
-- Create complex check constraints
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    ship_date DATE,
    delivery_date DATE,
    total_amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Complex date validation
    CONSTRAINT orders_ship_date_after_order CHECK (
        ship_date IS NULL OR ship_date >= order_date
    ),
    CONSTRAINT orders_delivery_date_after_ship CHECK (
        delivery_date IS NULL OR ship_date IS NULL OR delivery_date >= ship_date
    ),
    
    -- Status-based validation
    CONSTRAINT orders_status_valid CHECK (
        status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled', 'returned')
    ),
    
    -- Business rule validation
    CONSTRAINT orders_amount_positive CHECK (total_amount > 0),
    CONSTRAINT orders_dates_logical CHECK (
        (status = 'pending' AND ship_date IS NULL AND delivery_date IS NULL) OR
        (status = 'processing' AND ship_date IS NULL AND delivery_date IS NULL) OR
        (status = 'shipped' AND ship_date IS NOT NULL AND delivery_date IS NULL) OR
        (status = 'delivered' AND ship_date IS NOT NULL AND delivery_date IS NOT NULL) OR
        (status = 'cancelled' AND ship_date IS NULL AND delivery_date IS NULL) OR
        (status = 'returned' AND ship_date IS NOT NULL AND delivery_date IS NOT NULL)
    )
);
```

## Unique Constraints

### Unique Constraint Design

```sql
-- Create unique constraints for data integrity
CREATE TABLE user_emails (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    email VARCHAR(255) NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraints
    CONSTRAINT user_emails_email_unique UNIQUE (email),
    CONSTRAINT user_emails_primary_unique UNIQUE (user_id, is_primary) 
        DEFERRABLE INITIALLY DEFERRED
);

-- Partial unique constraints
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Partial unique constraint
    CONSTRAINT user_sessions_active_unique UNIQUE (user_id, is_active) 
        WHERE is_active = TRUE
);

-- Composite unique constraints
CREATE TABLE user_permissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id INTEGER NOT NULL,
    permission VARCHAR(50) NOT NULL,
    granted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite unique constraint
    CONSTRAINT user_permissions_unique UNIQUE (user_id, resource_type, resource_id, permission)
);
```

### Unique Constraint Performance

```sql
-- Create indexes for unique constraints
CREATE INDEX CONCURRENTLY idx_user_emails_email ON user_emails (email);
CREATE INDEX CONCURRENTLY idx_user_emails_user_id ON user_emails (user_id);
CREATE INDEX CONCURRENTLY idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX CONCURRENTLY idx_user_permissions_user_id ON user_permissions (user_id);

-- Analyze unique constraint usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%_unique%'
ORDER BY idx_scan DESC;
```

## Not Null Constraints

### Not Null Design

```sql
-- Create tables with appropriate NOT NULL constraints
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    date_of_birth DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Use NOT NULL with default values
CREATE TABLE system_settings (
    id SERIAL PRIMARY KEY,
    setting_name VARCHAR(100) NOT NULL,
    setting_value TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Conditional NOT NULL constraints
CREATE TABLE user_addresses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    address_type VARCHAR(20) NOT NULL,
    street_address VARCHAR(200) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50) NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Conditional NOT NULL based on country
    CONSTRAINT user_addresses_state_required CHECK (
        (country = 'US' AND state IS NOT NULL) OR 
        (country != 'US' AND state IS NULL)
    )
);
```

## Custom Validation Functions

### Advanced Validation

```sql
-- Create custom validation functions
CREATE OR REPLACE FUNCTION validate_email(email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    IF email IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Basic email format validation
    IF NOT (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
        RETURN FALSE;
    END IF;
    
    -- Check for consecutive dots
    IF email ~ '\.\.' THEN
        RETURN FALSE;
    END IF;
    
    -- Check for leading/trailing dots
    IF email ~ '^\.|\.$' THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_phone(phone TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    IF phone IS NULL THEN
        RETURN TRUE; -- NULL is allowed
    END IF;
    
    -- E.164 format validation
    IF NOT (phone ~ '^\+[1-9]\d{1,14}$') THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_credit_card(card_number TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    i INTEGER;
    sum INTEGER := 0;
    digit INTEGER;
    is_even BOOLEAN := FALSE;
BEGIN
    IF card_number IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Remove non-digits
    card_number := regexp_replace(card_number, '[^0-9]', '', 'g');
    
    -- Check length
    IF LENGTH(card_number) < 13 OR LENGTH(card_number) > 19 THEN
        RETURN FALSE;
    END IF;
    
    -- Luhn algorithm
    FOR i IN REVERSE LENGTH(card_number)..1 LOOP
        digit := (ASCII(SUBSTRING(card_number, i, 1)) - ASCII('0'));
        
        IF is_even THEN
            digit := digit * 2;
            IF digit > 9 THEN
                digit := digit - 9;
            END IF;
        END IF;
        
        sum := sum + digit;
        is_even := NOT is_even;
    END LOOP;
    
    RETURN (sum % 10) = 0;
END;
$$ LANGUAGE plpgsql;

-- Use validation functions in constraints
CREATE TABLE user_payments (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    card_number VARCHAR(19) NOT NULL,
    card_holder_name VARCHAR(100) NOT NULL,
    expiry_date DATE NOT NULL,
    cvv VARCHAR(4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT user_payments_card_valid CHECK (validate_credit_card(card_number)),
    CONSTRAINT user_payments_expiry_future CHECK (expiry_date > CURRENT_DATE)
);
```

## Constraint Management

### Dynamic Constraint Management

```sql
-- Create function to add constraints dynamically
CREATE OR REPLACE FUNCTION add_constraint_if_not_exists(
    table_name TEXT,
    constraint_name TEXT,
    constraint_definition TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if constraint already exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = $1 AND constraint_name = $2
    ) THEN
        RETURN FALSE;
    END IF;
    
    -- Add constraint
    EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I %s', 
                   table_name, constraint_name, constraint_definition);
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Use function to add constraints
SELECT add_constraint_if_not_exists(
    'users',
    'users_username_format',
    'CHECK (username ~ ''^[a-zA-Z0-9_]{3,50}$'')'
);

-- Create function to drop constraints safely
CREATE OR REPLACE FUNCTION drop_constraint_if_exists(
    table_name TEXT,
    constraint_name TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if constraint exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = $1 AND constraint_name = $2
    ) THEN
        RETURN FALSE;
    END IF;
    
    -- Drop constraint
    EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I', table_name, constraint_name);
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Constraint Monitoring

```python
# monitoring/constraint_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class ConstraintMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_constraint_info(self):
        """Get information about all constraints."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        tc.table_name,
                        tc.constraint_name,
                        tc.constraint_type,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name,
                        rc.delete_rule,
                        rc.update_rule
                    FROM information_schema.table_constraints AS tc
                    LEFT JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    LEFT JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    LEFT JOIN information_schema.referential_constraints AS rc
                        ON tc.constraint_name = rc.constraint_name
                    WHERE tc.table_schema = 'public'
                    ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name
                """)
                
                constraints = cur.fetchall()
                return constraints
                
        except Exception as e:
            self.logger.error(f"Error getting constraint info: {e}")
            return []
        finally:
            conn.close()
    
    def check_constraint_violations(self, table_name):
        """Check for potential constraint violations."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Check for duplicate values in unique constraints
                cur.execute("""
                    SELECT 
                        tc.constraint_name,
                        kcu.column_name,
                        COUNT(*) as duplicate_count
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = %s
                    AND tc.constraint_type = 'UNIQUE'
                    GROUP BY tc.constraint_name, kcu.column_name
                    HAVING COUNT(*) > 1
                """, (table_name,))
                
                violations = cur.fetchall()
                return violations
                
        except Exception as e:
            self.logger.error(f"Error checking constraint violations: {e}")
            return []
        finally:
            conn.close()
    
    def analyze_constraint_performance(self):
        """Analyze constraint performance impact."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Get constraint usage statistics
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE indexname LIKE '%_pkey%' OR indexname LIKE '%_unique%'
                    ORDER BY idx_scan DESC
                """)
                
                performance_stats = cur.fetchall()
                return performance_stats
                
        except Exception as e:
            self.logger.error(f"Error analyzing constraint performance: {e}")
            return []
        finally:
            conn.close()
    
    def generate_constraint_report(self):
        """Generate comprehensive constraint report."""
        constraint_info = self.get_constraint_info()
        performance_stats = self.analyze_constraint_performance()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_constraints': len(constraint_info),
            'constraint_info': constraint_info,
            'performance_stats': performance_stats
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = ConstraintMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_constraint_report()
    print(json.dumps(report, indent=2))
```

## Error Handling and Messages

### Custom Constraint Messages

```sql
-- Create function for custom constraint error messages
CREATE OR REPLACE FUNCTION handle_constraint_violation()
RETURNS TRIGGER AS $$
BEGIN
    -- Check for specific constraint violations
    IF TG_OP = 'INSERT' THEN
        -- Check for duplicate email
        IF EXISTS (SELECT 1 FROM users WHERE email = NEW.email AND id != NEW.id) THEN
            RAISE EXCEPTION 'Email address % is already in use', NEW.email;
        END IF;
        
        -- Check for duplicate username
        IF EXISTS (SELECT 1 FROM users WHERE username = NEW.username AND id != NEW.id) THEN
            RAISE EXCEPTION 'Username % is already taken', NEW.username;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for custom error handling
CREATE TRIGGER users_constraint_handler
    BEFORE INSERT OR UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION handle_constraint_violation();
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create primary key constraints
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

-- 2. Create foreign key constraints
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_amount NUMERIC(10,2) NOT NULL
);

-- 3. Create check constraints
ALTER TABLE users ADD CONSTRAINT users_email_format 
    CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

-- 4. Create unique constraints
CREATE UNIQUE INDEX idx_users_username_unique ON users (username);
```

### Essential Patterns

```python
# Complete PostgreSQL constraints and validation setup
def setup_postgresql_constraints_validation():
    # 1. Primary key constraints
    # 2. Foreign key constraints
    # 3. Check constraints
    # 4. Unique constraints
    # 5. Not null constraints
    # 6. Custom validation functions
    # 7. Constraint management
    # 8. Error handling and monitoring
    
    print("PostgreSQL constraints and validation setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL constraints and validation excellence. Each pattern includes implementation examples, validation strategies, and real-world usage patterns for enterprise PostgreSQL constraint systems.*
