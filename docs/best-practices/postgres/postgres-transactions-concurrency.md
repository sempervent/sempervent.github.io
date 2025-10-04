# PostgreSQL Transactions & Concurrency Best Practices

**Objective**: Master senior-level PostgreSQL transaction and concurrency patterns for production systems. When you need to handle concurrent access, when you want to optimize transaction performance, when you need enterprise-grade concurrency strategies—these best practices become your weapon of choice.

## Core Principles

- **ACID Properties**: Ensure atomicity, consistency, isolation, and durability
- **Isolation Levels**: Choose appropriate isolation levels for your use case
- **Lock Management**: Minimize lock contention and deadlocks
- **Performance**: Optimize transaction performance and throughput
- **Error Handling**: Implement robust transaction error handling

## Transaction Isolation Levels

### Isolation Level Overview

```sql
-- Set transaction isolation levels
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- Your transaction code here
COMMIT;

-- Different isolation levels for different use cases
BEGIN TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;  -- Lowest isolation
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;    -- Default level
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;   -- Higher isolation
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;     -- Highest isolation

-- Check current isolation level
SELECT current_setting('transaction_isolation');
```

### Isolation Level Examples

```sql
-- Create test tables for isolation level demonstration
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(20) UNIQUE NOT NULL,
    balance NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    from_account_id INTEGER REFERENCES accounts(id),
    to_account_id INTEGER REFERENCES accounts(id),
    amount NUMERIC(10,2) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO accounts (account_number, balance) VALUES
    ('ACC001', 1000.00),
    ('ACC002', 500.00),
    ('ACC003', 2000.00);

-- READ COMMITTED example
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM accounts WHERE account_number = 'ACC001';
-- Another transaction can modify this data here
SELECT balance FROM accounts WHERE account_number = 'ACC001';
COMMIT;

-- REPEATABLE READ example
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE account_number = 'ACC001';
-- This will see the same data even if another transaction modifies it
SELECT balance FROM accounts WHERE account_number = 'ACC001';
COMMIT;

-- SERIALIZABLE example
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT balance FROM accounts WHERE account_number = 'ACC001';
-- This will fail if another transaction modifies the data
UPDATE accounts SET balance = balance - 100 WHERE account_number = 'ACC001';
COMMIT;
```

## Lock Management

### Row-Level Locking

```sql
-- Create tables for lock demonstration
CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    reserved_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount NUMERIC(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL
);

-- Insert sample data
INSERT INTO inventory (product_id, quantity) VALUES
    (1, 100),
    (2, 50),
    (3, 200);

-- Row-level locking examples
BEGIN;
-- Lock specific rows for update
SELECT * FROM inventory WHERE product_id = 1 FOR UPDATE;
-- Perform operations on locked rows
UPDATE inventory SET reserved_quantity = reserved_quantity + 10 WHERE product_id = 1;
COMMIT;

-- Lock for share (read lock)
BEGIN;
SELECT * FROM inventory WHERE product_id = 1 FOR SHARE;
-- Other transactions can read but not modify
COMMIT;
```

### Table-Level Locking

```sql
-- Table-level locking examples
BEGIN;
-- Lock table for exclusive access
LOCK TABLE inventory IN EXCLUSIVE MODE;
-- Perform bulk operations
UPDATE inventory SET quantity = quantity - 10;
COMMIT;

-- Lock table for share access
BEGIN;
LOCK TABLE inventory IN SHARE MODE;
-- Allow other transactions to read but not modify
SELECT * FROM inventory;
COMMIT;
```

### Deadlock Prevention

```sql
-- Create function to prevent deadlocks
CREATE OR REPLACE FUNCTION transfer_inventory(
    from_product_id INTEGER,
    to_product_id INTEGER,
    quantity INTEGER
)
RETURNS BOOLEAN AS $$
DECLARE
    from_quantity INTEGER;
    to_quantity INTEGER;
BEGIN
    -- Always lock in the same order to prevent deadlocks
    IF from_product_id < to_product_id THEN
        -- Lock from_product first
        SELECT quantity INTO from_quantity 
        FROM inventory WHERE product_id = from_product_id FOR UPDATE;
        
        SELECT quantity INTO to_quantity 
        FROM inventory WHERE product_id = to_product_id FOR UPDATE;
    ELSE
        -- Lock to_product first
        SELECT quantity INTO to_quantity 
        FROM inventory WHERE product_id = to_product_id FOR UPDATE;
        
        SELECT quantity INTO from_quantity 
        FROM inventory WHERE product_id = from_product_id FOR UPDATE;
    END IF;
    
    -- Check if sufficient inventory
    IF from_quantity < quantity THEN
        RAISE EXCEPTION 'Insufficient inventory: % available, % requested', from_quantity, quantity;
    END IF;
    
    -- Perform transfer
    UPDATE inventory SET quantity = quantity - quantity WHERE product_id = from_product_id;
    UPDATE inventory SET quantity = quantity + quantity WHERE product_id = to_product_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## Transaction Patterns

### Savepoints and Nested Transactions

```sql
-- Create function with savepoints
CREATE OR REPLACE FUNCTION process_order(
    customer_id INTEGER,
    order_items JSONB
)
RETURNS INTEGER AS $$
DECLARE
    order_id INTEGER;
    item JSONB;
    product_id INTEGER;
    quantity INTEGER;
    unit_price NUMERIC(10,2);
    total_amount NUMERIC(10,2) := 0;
BEGIN
    -- Start transaction
    BEGIN
        -- Create order
        INSERT INTO orders (customer_id, total_amount) 
        VALUES (customer_id, 0) RETURNING id INTO order_id;
        
        -- Process each order item
        FOR item IN SELECT * FROM jsonb_array_elements(order_items) LOOP
            product_id := (item->>'product_id')::INTEGER;
            quantity := (item->>'quantity')::INTEGER;
            unit_price := (item->>'unit_price')::NUMERIC(10,2);
            
            -- Use savepoint for each item
            SAVEPOINT item_processing;
            
            BEGIN
                -- Check inventory
                IF NOT check_inventory_availability(product_id, quantity) THEN
                    RAISE EXCEPTION 'Insufficient inventory for product %', product_id;
                END IF;
                
                -- Reserve inventory
                UPDATE inventory 
                SET reserved_quantity = reserved_quantity + quantity 
                WHERE product_id = product_id;
                
                -- Add order item
                INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                VALUES (order_id, product_id, quantity, unit_price);
                
                total_amount := total_amount + (quantity * unit_price);
                
            EXCEPTION
                WHEN OTHERS THEN
                    -- Rollback to savepoint
                    ROLLBACK TO SAVEPOINT item_processing;
                    RAISE;
            END;
        END LOOP;
        
        -- Update order total
        UPDATE orders SET total_amount = total_amount WHERE id = order_id;
        
        RETURN order_id;
        
    EXCEPTION
        WHEN OTHERS THEN
            -- Rollback entire transaction
            ROLLBACK;
            RAISE;
    END;
END;
$$ LANGUAGE plpgsql;
```

### Optimistic Locking

```sql
-- Create table with version column for optimistic locking
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    version INTEGER DEFAULT 1,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function for optimistic locking update
CREATE OR REPLACE FUNCTION update_product_optimistic(
    product_id INTEGER,
    new_name VARCHAR(200),
    new_price NUMERIC(10,2),
    expected_version INTEGER
)
RETURNS BOOLEAN AS $$
DECLARE
    current_version INTEGER;
    rows_affected INTEGER;
BEGIN
    -- Update with version check
    UPDATE products 
    SET name = new_name, 
        price = new_price, 
        version = version + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = product_id AND version = expected_version;
    
    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    
    IF rows_affected = 0 THEN
        RAISE EXCEPTION 'Product was modified by another transaction. Expected version: %, current version: %', 
            expected_version, (SELECT version FROM products WHERE id = product_id);
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Pessimistic Locking

```sql
-- Create function for pessimistic locking
CREATE OR REPLACE FUNCTION update_product_pessimistic(
    product_id INTEGER,
    new_name VARCHAR(200),
    new_price NUMERIC(10,2)
)
RETURNS BOOLEAN AS $$
DECLARE
    current_product RECORD;
BEGIN
    -- Lock the row for update
    SELECT * INTO current_product 
    FROM products 
    WHERE id = product_id 
    FOR UPDATE;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Product not found: %', product_id;
    END IF;
    
    -- Update the product
    UPDATE products 
    SET name = new_name, 
        price = new_price,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = product_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## Concurrency Control

### MVCC (Multi-Version Concurrency Control)

```sql
-- Create table to demonstrate MVCC
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- MVCC example: Multiple transactions can read without blocking
BEGIN;
-- Transaction 1: Read sessions
SELECT * FROM user_sessions WHERE is_active = TRUE;
-- This won't block other transactions

-- Transaction 2: Insert new session (runs concurrently)
INSERT INTO user_sessions (user_id, session_token, expires_at) 
VALUES (1, 'token123', CURRENT_TIMESTAMP + INTERVAL '1 hour');
COMMIT;
```

### Snapshot Isolation

```sql
-- Create function to demonstrate snapshot isolation
CREATE OR REPLACE FUNCTION get_user_balance_snapshot(user_id INTEGER)
RETURNS NUMERIC(10,2) AS $$
DECLARE
    balance NUMERIC(10,2);
BEGIN
    -- This will see a consistent snapshot of the data
    SELECT COALESCE(SUM(amount), 0) INTO balance
    FROM transactions 
    WHERE user_id = user_id;
    
    RETURN balance;
END;
$$ LANGUAGE plpgsql;
```

## Transaction Monitoring

### Transaction Performance Analysis

```python
# monitoring/transaction_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class TransactionMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_active_transactions(self):
        """Get information about active transactions."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        backend_start,
                        state,
                        query_start,
                        state_change,
                        wait_event_type,
                        wait_event,
                        query
                    FROM pg_stat_activity
                    WHERE state = 'active'
                    ORDER BY query_start
                """)
                
                active_transactions = cur.fetchall()
                return active_transactions
                
        except Exception as e:
            self.logger.error(f"Error getting active transactions: {e}")
            return []
        finally:
            conn.close()
    
    def get_lock_information(self):
        """Get information about current locks."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        l.locktype,
                        l.database,
                        l.relation,
                        l.page,
                        l.tuple,
                        l.virtualxid,
                        l.transactionid,
                        l.classid,
                        l.objid,
                        l.objsubid,
                        l.virtualtransaction,
                        l.pid,
                        l.mode,
                        l.granted,
                        a.usename,
                        a.query,
                        a.query_start,
                        a.state
                    FROM pg_locks l
                    LEFT JOIN pg_stat_activity a ON l.pid = a.pid
                    WHERE l.granted = false
                    ORDER BY l.pid
                """)
                
                lock_info = cur.fetchall()
                return lock_info
                
        except Exception as e:
            self.logger.error(f"Error getting lock information: {e}")
            return []
        finally:
            conn.close()
    
    def get_deadlock_information(self):
        """Get information about recent deadlocks."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        log_time,
                        user_name,
                        database_name,
                        process_id,
                        session_id,
                        session_line_num,
                        command_tag,
                        session_start_time,
                        virtual_transaction_id,
                        transaction_id,
                        error_severity,
                        sql_state_code,
                        message,
                        detail,
                        hint,
                        internal_query,
                        internal_query_pos,
                        context,
                        query,
                        query_pos,
                        location,
                        application_name
                    FROM pg_log
                    WHERE message LIKE '%deadlock%'
                    ORDER BY log_time DESC
                    LIMIT 10
                """)
                
                deadlock_info = cur.fetchall()
                return deadlock_info
                
        except Exception as e:
            self.logger.error(f"Error getting deadlock information: {e}")
            return []
        finally:
            conn.close()
    
    def get_transaction_statistics(self):
        """Get transaction statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        datname,
                        numbackends,
                        xact_commit,
                        xact_rollback,
                        blks_read,
                        blks_hit,
                        tup_returned,
                        tup_fetched,
                        tup_inserted,
                        tup_updated,
                        tup_deleted,
                        conflicts,
                        temp_files,
                        temp_bytes,
                        deadlocks,
                        blk_read_time,
                        blk_write_time
                    FROM pg_stat_database
                    WHERE datname = current_database()
                """)
                
                stats = cur.fetchone()
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting transaction statistics: {e}")
            return None
        finally:
            conn.close()
    
    def generate_transaction_report(self):
        """Generate comprehensive transaction report."""
        active_transactions = self.get_active_transactions()
        lock_info = self.get_lock_information()
        deadlock_info = self.get_deadlock_information()
        transaction_stats = self.get_transaction_statistics()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'active_transactions_count': len(active_transactions),
            'active_transactions': active_transactions,
            'lock_information': lock_info,
            'deadlock_information': deadlock_info,
            'transaction_statistics': transaction_stats
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = TransactionMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_transaction_report()
    print(json.dumps(report, indent=2))
```

## Error Handling and Recovery

### Transaction Error Handling

```sql
-- Create function with comprehensive error handling
CREATE OR REPLACE FUNCTION process_payment(
    user_id INTEGER,
    amount NUMERIC(10,2),
    payment_method VARCHAR(50)
)
RETURNS INTEGER AS $$
DECLARE
    transaction_id INTEGER;
    current_balance NUMERIC(10,2);
BEGIN
    -- Start transaction
    BEGIN
        -- Check user balance
        SELECT balance INTO current_balance 
        FROM user_accounts 
        WHERE user_id = user_id FOR UPDATE;
        
        IF current_balance < amount THEN
            RAISE EXCEPTION 'Insufficient funds: % available, % requested', current_balance, amount;
        END IF;
        
        -- Create transaction record
        INSERT INTO transactions (user_id, amount, transaction_type, payment_method)
        VALUES (user_id, amount, 'payment', payment_method)
        RETURNING id INTO transaction_id;
        
        -- Update user balance
        UPDATE user_accounts 
        SET balance = balance - amount 
        WHERE user_id = user_id;
        
        -- Log transaction
        INSERT INTO transaction_logs (transaction_id, action, timestamp)
        VALUES (transaction_id, 'payment_processed', CURRENT_TIMESTAMP);
        
        RETURN transaction_id;
        
    EXCEPTION
        WHEN insufficient_funds THEN
            -- Log the error
            INSERT INTO error_logs (user_id, error_type, error_message, timestamp)
            VALUES (user_id, 'insufficient_funds', SQLERRM, CURRENT_TIMESTAMP);
            RAISE;
            
        WHEN OTHERS THEN
            -- Log unexpected errors
            INSERT INTO error_logs (user_id, error_type, error_message, timestamp)
            VALUES (user_id, 'unexpected_error', SQLERRM, CURRENT_TIMESTAMP);
            RAISE;
    END;
END;
$$ LANGUAGE plpgsql;
```

### Transaction Recovery

```sql
-- Create function for transaction recovery
CREATE OR REPLACE FUNCTION recover_failed_transactions()
RETURNS INTEGER AS $$
DECLARE
    failed_transaction RECORD;
    recovery_count INTEGER := 0;
BEGIN
    -- Find failed transactions
    FOR failed_transaction IN 
        SELECT * FROM transactions 
        WHERE status = 'failed' 
        AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
    LOOP
        BEGIN
            -- Attempt to recover the transaction
            UPDATE transactions 
            SET status = 'recovered', 
                updated_at = CURRENT_TIMESTAMP 
            WHERE id = failed_transaction.id;
            
            recovery_count := recovery_count + 1;
            
        EXCEPTION
            WHEN OTHERS THEN
                -- Log recovery failure
                INSERT INTO error_logs (transaction_id, error_type, error_message, timestamp)
                VALUES (failed_transaction.id, 'recovery_failed', SQLERRM, CURRENT_TIMESTAMP);
        END;
    END LOOP;
    
    RETURN recovery_count;
END;
$$ LANGUAGE plpgsql;
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Set appropriate isolation levels
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- Your transaction code here
COMMIT;

-- 2. Use row-level locking
SELECT * FROM table_name WHERE id = 1 FOR UPDATE;

-- 3. Implement savepoints for nested transactions
SAVEPOINT my_savepoint;
-- Your code here
ROLLBACK TO SAVEPOINT my_savepoint;

-- 4. Handle errors in transactions
BEGIN;
-- Your code here
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
```

### Essential Patterns

```python
# Complete PostgreSQL transactions and concurrency setup
def setup_postgresql_transactions_concurrency():
    # 1. Transaction isolation levels
    # 2. Lock management
    # 3. Transaction patterns
    # 4. Concurrency control
    # 5. Transaction monitoring
    # 6. Error handling and recovery
    # 7. Performance optimization
    # 8. Deadlock prevention
    
    print("PostgreSQL transactions and concurrency setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL transactions and concurrency excellence. Each pattern includes implementation examples, concurrency strategies, and real-world usage patterns for enterprise PostgreSQL transaction systems.*
