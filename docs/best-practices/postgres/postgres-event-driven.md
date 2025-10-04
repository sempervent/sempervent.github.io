# PostgreSQL Event-Driven Architecture Best Practices

**Objective**: Master senior-level PostgreSQL event-driven patterns for production systems. When you need to implement event-driven architectures, when you want to build reactive systems, when you need enterprise-grade event processing—these best practices become your weapon of choice.

## Core Principles

- **Event Sourcing**: Store events as the source of truth
- **CQRS**: Separate command and query responsibilities
- **Event Streaming**: Implement real-time event processing
- **Eventual Consistency**: Design for distributed consistency
- **Reactive Systems**: Build responsive and resilient systems

## Event Sourcing Patterns

### Event Store Implementation

```sql
-- Create events table
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    event_metadata JSONB DEFAULT '{}',
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(aggregate_id, version)
);

-- Create indexes for performance
CREATE INDEX idx_events_aggregate_id ON events(aggregate_id);
CREATE INDEX idx_events_aggregate_type ON events(aggregate_type);
CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_created_at ON events(created_at);

-- Create function to append event
CREATE OR REPLACE FUNCTION append_event(
    p_aggregate_id UUID,
    p_aggregate_type VARCHAR(100),
    p_event_type VARCHAR(100),
    p_event_data JSONB,
    p_event_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    event_id UUID;
    next_version INTEGER;
BEGIN
    -- Get next version for aggregate
    SELECT COALESCE(MAX(version), 0) + 1 INTO next_version
    FROM events
    WHERE aggregate_id = p_aggregate_id;
    
    -- Insert event
    INSERT INTO events (
        aggregate_id, aggregate_type, event_type, 
        event_data, event_metadata, version
    ) VALUES (
        p_aggregate_id, p_aggregate_type, p_event_type,
        p_event_data, p_event_metadata, next_version
    ) RETURNING id INTO event_id;
    
    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to get events for aggregate
CREATE OR REPLACE FUNCTION get_events_for_aggregate(
    p_aggregate_id UUID,
    p_from_version INTEGER DEFAULT 0
)
RETURNS TABLE (
    id UUID,
    aggregate_id UUID,
    aggregate_type VARCHAR(100),
    event_type VARCHAR(100),
    event_data JSONB,
    event_metadata JSONB,
    version INTEGER,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id, e.aggregate_id, e.aggregate_type, e.event_type,
        e.event_data, e.event_metadata, e.version, e.created_at
    FROM events e
    WHERE e.aggregate_id = p_aggregate_id
    AND e.version > p_from_version
    ORDER BY e.version;
END;
$$ LANGUAGE plpgsql;
```

### Aggregate Root Management

```sql
-- Create aggregates table
CREATE TABLE aggregates (
    id UUID PRIMARY KEY,
    aggregate_type VARCHAR(100) NOT NULL,
    current_version INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to create aggregate
CREATE OR REPLACE FUNCTION create_aggregate(
    p_aggregate_id UUID,
    p_aggregate_type VARCHAR(100)
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO aggregates (id, aggregate_type)
    VALUES (p_aggregate_id, p_aggregate_type)
    ON CONFLICT (id) DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Create function to get aggregate version
CREATE OR REPLACE FUNCTION get_aggregate_version(p_aggregate_id UUID)
RETURNS INTEGER AS $$
DECLARE
    current_version INTEGER;
BEGIN
    SELECT COALESCE(MAX(version), 0) INTO current_version
    FROM events
    WHERE aggregate_id = p_aggregate_id;
    
    RETURN current_version;
END;
$$ LANGUAGE plpgsql;

-- Create function to update aggregate version
CREATE OR REPLACE FUNCTION update_aggregate_version(
    p_aggregate_id UUID,
    p_new_version INTEGER
)
RETURNS VOID AS $$
BEGIN
    UPDATE aggregates
    SET current_version = p_new_version,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_aggregate_id;
END;
$$ LANGUAGE plpgsql;
```

## CQRS Implementation

### Command Side

```sql
-- Create commands table
CREATE TABLE commands (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    command_type VARCHAR(100) NOT NULL,
    command_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ
);

-- Create function to create command
CREATE OR REPLACE FUNCTION create_command(
    p_aggregate_id UUID,
    p_command_type VARCHAR(100),
    p_command_data JSONB
)
RETURNS UUID AS $$
DECLARE
    command_id UUID;
BEGIN
    INSERT INTO commands (aggregate_id, command_type, command_data)
    VALUES (p_aggregate_id, p_command_type, p_command_data)
    RETURNING id INTO command_id;
    
    RETURN command_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to process command
CREATE OR REPLACE FUNCTION process_command(p_command_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    command_record RECORD;
    event_id UUID;
BEGIN
    -- Get command
    SELECT * INTO command_record
    FROM commands
    WHERE id = p_command_id AND status = 'pending';
    
    IF command_record IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Process command based on type
    CASE command_record.command_type
        WHEN 'create_user' THEN
            -- Create user event
            SELECT append_event(
                command_record.aggregate_id,
                'user',
                'user_created',
                command_record.command_data
            ) INTO event_id;
            
        WHEN 'update_user' THEN
            -- Update user event
            SELECT append_event(
                command_record.aggregate_id,
                'user',
                'user_updated',
                command_record.command_data
            ) INTO event_id;
            
        WHEN 'delete_user' THEN
            -- Delete user event
            SELECT append_event(
                command_record.aggregate_id,
                'user',
                'user_deleted',
                command_record.command_data
            ) INTO event_id;
            
        ELSE
            RAISE EXCEPTION 'Unknown command type: %', command_record.command_type;
    END CASE;
    
    -- Update command status
    UPDATE commands
    SET status = 'processed',
        processed_at = CURRENT_TIMESTAMP
    WHERE id = p_command_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Query Side Implementation

```sql
-- Create read models table
CREATE TABLE read_models (
    id UUID PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    model_data JSONB NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to update read model
CREATE OR REPLACE FUNCTION update_read_model(
    p_aggregate_id UUID,
    p_aggregate_type VARCHAR(100),
    p_model_data JSONB,
    p_version INTEGER
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO read_models (id, aggregate_id, aggregate_type, model_data, version)
    VALUES (gen_random_uuid(), p_aggregate_id, p_aggregate_type, p_model_data, p_version)
    ON CONFLICT (aggregate_id) 
    DO UPDATE SET 
        model_data = EXCLUDED.model_data,
        version = EXCLUDED.version,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create function to get read model
CREATE OR REPLACE FUNCTION get_read_model(p_aggregate_id UUID)
RETURNS TABLE (
    aggregate_id UUID,
    aggregate_type VARCHAR(100),
    model_data JSONB,
    version INTEGER,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rm.aggregate_id, rm.aggregate_type, rm.model_data,
        rm.version, rm.updated_at
    FROM read_models rm
    WHERE rm.aggregate_id = p_aggregate_id;
END;
$$ LANGUAGE plpgsql;
```

## Event Streaming

### Event Stream Processing

```sql
-- Create event streams table
CREATE TABLE event_streams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_name VARCHAR(100) NOT NULL,
    event_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    position BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stream_name, position)
);

-- Create function to append to stream
CREATE OR REPLACE FUNCTION append_to_stream(
    p_stream_name VARCHAR(100),
    p_event_id UUID,
    p_event_type VARCHAR(100),
    p_event_data JSONB
)
RETURNS BIGINT AS $$
DECLARE
    next_position BIGINT;
BEGIN
    -- Get next position
    SELECT COALESCE(MAX(position), 0) + 1 INTO next_position
    FROM event_streams
    WHERE stream_name = p_stream_name;
    
    -- Insert event
    INSERT INTO event_streams (stream_name, event_id, event_type, event_data, position)
    VALUES (p_stream_name, p_event_id, p_event_type, p_event_data, next_position);
    
    RETURN next_position;
END;
$$ LANGUAGE plpgsql;

-- Create function to read from stream
CREATE OR REPLACE FUNCTION read_from_stream(
    p_stream_name VARCHAR(100),
    p_from_position BIGINT DEFAULT 0,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    event_id UUID,
    event_type VARCHAR(100),
    event_data JSONB,
    position BIGINT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        es.event_id, es.event_type, es.event_data,
        es.position, es.created_at
    FROM event_streams es
    WHERE es.stream_name = p_stream_name
    AND es.position > p_from_position
    ORDER BY es.position
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Event Handlers

```sql
-- Create event handlers table
CREATE TABLE event_handlers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    handler_name VARCHAR(100) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    handler_function VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to register event handler
CREATE OR REPLACE FUNCTION register_event_handler(
    p_handler_name VARCHAR(100),
    p_event_type VARCHAR(100),
    p_handler_function VARCHAR(100)
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO event_handlers (handler_name, event_type, handler_function)
    VALUES (p_handler_name, p_event_type, p_handler_function)
    ON CONFLICT (handler_name) 
    DO UPDATE SET 
        event_type = EXCLUDED.event_type,
        handler_function = EXCLUDED.handler_function;
END;
$$ LANGUAGE plpgsql;

-- Create function to process event
CREATE OR REPLACE FUNCTION process_event(p_event_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    event_record RECORD;
    handler_record RECORD;
BEGIN
    -- Get event
    SELECT * INTO event_record
    FROM events
    WHERE id = p_event_id;
    
    IF event_record IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Get handlers for event type
    FOR handler_record IN 
        SELECT * FROM event_handlers
        WHERE event_type = event_record.event_type
        AND is_active = TRUE
    LOOP
        -- Execute handler function
        EXECUTE format('SELECT %s(%s, %s, %s)',
                       handler_record.handler_function,
                       event_record.id,
                       event_record.aggregate_id,
                       event_record.event_data);
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## Event Processing Implementation

### Python Event Processor

```python
# event_processing/postgres_event_processor.py
import psycopg2
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

class PostgreSQLEventProcessor:
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
    
    def append_event(self, aggregate_id: str, aggregate_type: str, 
                    event_type: str, event_data: dict, 
                    event_metadata: dict = None):
        """Append event to event store."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT append_event(%s, %s, %s, %s, %s)
                """, (aggregate_id, aggregate_type, event_type, 
                      json.dumps(event_data), json.dumps(event_metadata or {})))
                
                event_id = cur.fetchone()[0]
                conn.commit()
                
                self.logger.info(f"Event {event_id} appended for aggregate {aggregate_id}")
                return event_id
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error appending event: {e}")
            raise
        finally:
            conn.close()
    
    def get_events_for_aggregate(self, aggregate_id: str, from_version: int = 0):
        """Get events for aggregate."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM get_events_for_aggregate(%s, %s)
                """, (aggregate_id, from_version))
                
                events = cur.fetchall()
                return [{
                    'id': event[0],
                    'aggregate_id': event[1],
                    'aggregate_type': event[2],
                    'event_type': event[3],
                    'event_data': event[4],
                    'event_metadata': event[5],
                    'version': event[6],
                    'created_at': event[7]
                } for event in events]
                
        except Exception as e:
            self.logger.error(f"Error getting events: {e}")
            return []
        finally:
            conn.close()
    
    def process_command(self, aggregate_id: str, command_type: str, command_data: dict):
        """Process command."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT create_command(%s, %s, %s)
                """, (aggregate_id, command_type, json.dumps(command_data)))
                
                command_id = cur.fetchone()[0]
                
                # Process command
                cur.execute("SELECT process_command(%s)", (command_id,))
                result = cur.fetchone()[0]
                
                conn.commit()
                
                if result:
                    self.logger.info(f"Command {command_id} processed successfully")
                else:
                    self.logger.warning(f"Command {command_id} processing failed")
                
                return result
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error processing command: {e}")
            raise
        finally:
            conn.close()
    
    def update_read_model(self, aggregate_id: str, aggregate_type: str, 
                         model_data: dict, version: int):
        """Update read model."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT update_read_model(%s, %s, %s, %s)
                """, (aggregate_id, aggregate_type, json.dumps(model_data), version))
                
                conn.commit()
                self.logger.info(f"Read model updated for aggregate {aggregate_id}")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating read model: {e}")
            raise
        finally:
            conn.close()
    
    def get_read_model(self, aggregate_id: str):
        """Get read model."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM get_read_model(%s)", (aggregate_id,))
                
                result = cur.fetchone()
                if result:
                    return {
                        'aggregate_id': result[0],
                        'aggregate_type': result[1],
                        'model_data': result[2],
                        'version': result[3],
                        'updated_at': result[4]
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting read model: {e}")
            return None
        finally:
            conn.close()
    
    def append_to_stream(self, stream_name: str, event_id: str, 
                        event_type: str, event_data: dict):
        """Append event to stream."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT append_to_stream(%s, %s, %s, %s)
                """, (stream_name, event_id, event_type, json.dumps(event_data)))
                
                position = cur.fetchone()[0]
                conn.commit()
                
                self.logger.info(f"Event {event_id} appended to stream {stream_name} at position {position}")
                return position
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error appending to stream: {e}")
            raise
        finally:
            conn.close()
    
    def read_from_stream(self, stream_name: str, from_position: int = 0, limit: int = 100):
        """Read from stream."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM read_from_stream(%s, %s, %s)
                """, (stream_name, from_position, limit))
                
                events = cur.fetchall()
                return [{
                    'event_id': event[0],
                    'event_type': event[1],
                    'event_data': event[2],
                    'position': event[3],
                    'created_at': event[4]
                } for event in events]
                
        except Exception as e:
            self.logger.error(f"Error reading from stream: {e}")
            return []
        finally:
            conn.close()
    
    def register_event_handler(self, handler_name: str, event_type: str, 
                              handler_function: str):
        """Register event handler."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT register_event_handler(%s, %s, %s)
                """, (handler_name, event_type, handler_function))
                
                conn.commit()
                self.logger.info(f"Event handler {handler_name} registered for {event_type}")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error registering event handler: {e}")
            raise
        finally:
            conn.close()
    
    def process_event(self, event_id: str):
        """Process event."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT process_event(%s)", (event_id,))
                result = cur.fetchone()[0]
                
                conn.commit()
                
                if result:
                    self.logger.info(f"Event {event_id} processed successfully")
                else:
                    self.logger.warning(f"Event {event_id} processing failed")
                
                return result
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error processing event: {e}")
            raise
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    processor = PostgreSQLEventProcessor({
        'host': 'localhost',
        'database': 'production',
        'user': 'event_processor_user',
        'password': 'event_processor_password'
    })
    
    # Example usage
    event_id = processor.append_event(
        'user-123', 'user', 'user_created', 
        {'username': 'john_doe', 'email': 'john@example.com'}
    )
    
    print(f"Event created: {event_id}")
```

## Event-Driven Monitoring

### Event Metrics

```sql
-- Create event metrics table
CREATE TABLE event_metrics (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    event_count BIGINT DEFAULT 1,
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to record event metrics
CREATE OR REPLACE FUNCTION record_event_metrics(
    p_event_type VARCHAR(100),
    p_aggregate_type VARCHAR(100),
    p_processing_time_ms INTEGER DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO event_metrics (event_type, aggregate_type, processing_time_ms)
    VALUES (p_event_type, p_aggregate_type, p_processing_time_ms);
END;
$$ LANGUAGE plpgsql;

-- Create function to get event metrics
CREATE OR REPLACE FUNCTION get_event_metrics(
    p_start_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP - INTERVAL '1 hour',
    p_end_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    event_type VARCHAR(100),
    aggregate_type VARCHAR(100),
    event_count BIGINT,
    avg_processing_time NUMERIC,
    max_processing_time INTEGER,
    min_processing_time INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        em.event_type,
        em.aggregate_type,
        COUNT(*) as event_count,
        AVG(em.processing_time_ms) as avg_processing_time,
        MAX(em.processing_time_ms) as max_processing_time,
        MIN(em.processing_time_ms) as min_processing_time
    FROM event_metrics em
    WHERE em.created_at BETWEEN p_start_date AND p_end_date
    GROUP BY em.event_type, em.aggregate_type
    ORDER BY event_count DESC;
END;
$$ LANGUAGE plpgsql;
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Append event
SELECT append_event('user-123', 'user', 'user_created', '{"username": "john"}');

-- 2. Get events for aggregate
SELECT * FROM get_events_for_aggregate('user-123');

-- 3. Process command
SELECT create_command('user-123', 'update_user', '{"username": "john_doe"}');

-- 4. Update read model
SELECT update_read_model('user-123', 'user', '{"username": "john_doe"}', 1);

-- 5. Append to stream
SELECT append_to_stream('user_events', 'event-123', 'user_created', '{"username": "john"}');

-- 6. Read from stream
SELECT * FROM read_from_stream('user_events', 0, 100);
```

### Essential Patterns

```python
# Complete PostgreSQL event-driven architecture setup
def setup_postgresql_event_driven():
    # 1. Event sourcing patterns
    # 2. CQRS implementation
    # 3. Event streaming
    # 4. Event handlers
    # 5. Event processing
    # 6. Event metrics
    # 7. Eventual consistency
    # 8. Reactive systems
    
    print("PostgreSQL event-driven architecture setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL event-driven architecture excellence. Each pattern includes implementation examples, event processing strategies, and real-world usage patterns for enterprise PostgreSQL event-driven systems.*
