# Real-Time Data Processing with Kafka and TimescaleDB

**Objective**: Build a complete real-time data processing pipeline using Kafka for streaming and TimescaleDB for time-series storage.

Real-time data processing enables immediate insights and responsive applications. This tutorial covers building a production-ready streaming pipeline that handles geospatial time-series data at scale.

## 1) Architecture Overview

### System Components

```yaml
# docker-compose.yaml for real-time processing stack
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'

  timescaledb:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: timeseries
      POSTGRES_USER: timeseries
      POSTGRES_PASSWORD: timeseries
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on:
      - timescaledb

volumes:
  timescale_data:
```

**Why**: This stack provides a complete real-time processing environment with streaming, storage, and visualization capabilities.

## 2) Kafka Producer Setup

### Python Producer

```python
import json
import time
import random
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialDataProducer:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            retry_backoff_ms=100,
            request_timeout_ms=30000
        )
    
    def generate_geospatial_data(self):
        """Generate sample geospatial time-series data"""
        # Sample locations (lat, lon)
        locations = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437), # Los Angeles
            (41.8781, -87.6298),  # Chicago
            (29.7604, -95.3698),  # Houston
            (33.4484, -112.0740)  # Phoenix
        ]
        
        location = random.choice(locations)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location_id': f"loc_{random.randint(1, 1000)}",
            'latitude': location[0] + random.uniform(-0.01, 0.01),
            'longitude': location[1] + random.uniform(-0.01, 0.01),
            'temperature': random.uniform(15, 35),
            'humidity': random.uniform(30, 80),
            'pressure': random.uniform(1000, 1020),
            'wind_speed': random.uniform(0, 20),
            'wind_direction': random.uniform(0, 360)
        }
    
    def send_data(self, topic='geospatial-data'):
        """Send geospatial data to Kafka topic"""
        try:
            data = self.generate_geospatial_data()
            
            # Use location_id as key for partitioning
            key = data['location_id']
            
            future = self.producer.send(
                topic,
                key=key,
                value=data
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.info(f"Data sent to {record_metadata.topic} partition {record_metadata.partition}")
            
        except KafkaError as e:
            logger.error(f"Failed to send data: {e}")
    
    def run_continuous(self, topic='geospatial-data', interval=1):
        """Continuously send data at specified interval"""
        logger.info(f"Starting continuous data production to topic: {topic}")
        
        try:
            while True:
                self.send_data(topic)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopping data production")
        finally:
            self.producer.close()

# Usage
if __name__ == "__main__":
    producer = GeospatialDataProducer()
    producer.run_continuous(interval=0.5)  # Send data every 500ms
```

**Why**: Kafka producers ensure reliable data delivery with proper error handling and partitioning for scalable processing.

## 3) Kafka Consumer Setup

### Python Consumer

```python
from kafka import KafkaConsumer
import json
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialDataConsumer:
    def __init__(self, bootstrap_servers=['localhost:9092'], db_config=None):
        self.consumer = KafkaConsumer(
            'geospatial-data',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            group_id='geospatial-processors',
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000
        )
        
        # Database connection
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'timeseries',
            'user': 'timeseries',
            'password': 'timeseries'
        }
        
        self.conn = None
        self.connect_db()
    
    def connect_db(self):
        """Connect to TimescaleDB"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            logger.info("Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_tables(self):
        """Create TimescaleDB tables"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS geospatial_metrics (
            time TIMESTAMPTZ NOT NULL,
            location_id TEXT NOT NULL,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            temperature DOUBLE PRECISION,
            humidity DOUBLE PRECISION,
            pressure DOUBLE PRECISION,
            wind_speed DOUBLE PRECISION,
            wind_direction DOUBLE PRECISION
        );
        
        -- Create hypertable
        SELECT create_hypertable('geospatial_metrics', 'time', if_not_exists => TRUE);
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_geospatial_metrics_location_time 
        ON geospatial_metrics (location_id, time DESC);
        
        CREATE INDEX IF NOT EXISTS idx_geospatial_metrics_time 
        ON geospatial_metrics (time DESC);
        """
        
        with self.conn.cursor() as cursor:
            cursor.execute(create_table_sql)
            logger.info("Tables created successfully")
    
    def process_message(self, message):
        """Process individual Kafka message"""
        try:
            data = message.value
            key = message.key
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
            # Insert into TimescaleDB
            insert_sql = """
            INSERT INTO geospatial_metrics (
                time, location_id, latitude, longitude,
                temperature, humidity, pressure, wind_speed, wind_direction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                timestamp,
                data['location_id'],
                data['latitude'],
                data['longitude'],
                data['temperature'],
                data['humidity'],
                data['pressure'],
                data['wind_speed'],
                data['wind_direction']
            )
            
            with self.conn.cursor() as cursor:
                cursor.execute(insert_sql, values)
            
            logger.info(f"Processed message for location {key}")
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    def run_consumer(self):
        """Run Kafka consumer"""
        logger.info("Starting Kafka consumer")
        
        try:
            # Create tables if they don't exist
            self.create_tables()
            
            # Consume messages
            for message in self.consumer:
                self.process_message(message)
                
        except KeyboardInterrupt:
            logger.info("Stopping consumer")
        finally:
            self.consumer.close()
            if self.conn:
                self.conn.close()

# Usage
if __name__ == "__main__":
    consumer = GeospatialDataConsumer()
    consumer.run_consumer()
```

**Why**: Kafka consumers provide reliable message processing with automatic offset management and error handling.

## 4) TimescaleDB Optimization

### Hypertable Configuration

```sql
-- Create optimized hypertable
CREATE TABLE geospatial_metrics (
    time TIMESTAMPTZ NOT NULL,
    location_id TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION
);

-- Create hypertable with optimal chunk size
SELECT create_hypertable(
    'geospatial_metrics', 
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create compression policy
ALTER TABLE geospatial_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'location_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy (compress after 1 day)
SELECT add_compression_policy('geospatial_metrics', INTERVAL '1 day');

-- Create retention policy (keep data for 1 year)
SELECT add_retention_policy('geospatial_metrics', INTERVAL '1 year');
```

**Why**: Proper hypertable configuration optimizes storage, compression, and query performance for time-series data.

### Continuous Aggregates

```sql
-- Create continuous aggregate for hourly averages
CREATE MATERIALIZED VIEW hourly_metrics
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    location_id,
    AVG(temperature) as avg_temperature,
    AVG(humidity) as avg_humidity,
    AVG(pressure) as avg_pressure,
    AVG(wind_speed) as avg_wind_speed,
    COUNT(*) as record_count
FROM geospatial_metrics
GROUP BY hour, location_id;

-- Create continuous aggregate for daily summaries
CREATE MATERIALIZED VIEW daily_metrics
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    location_id,
    AVG(temperature) as avg_temperature,
    MAX(temperature) as max_temperature,
    MIN(temperature) as min_temperature,
    AVG(humidity) as avg_humidity,
    AVG(pressure) as avg_pressure,
    AVG(wind_speed) as avg_wind_speed,
    COUNT(*) as record_count
FROM geospatial_metrics
GROUP BY day, location_id;

-- Refresh policies
SELECT add_continuous_aggregate_policy('hourly_metrics',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('daily_metrics',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

**Why**: Continuous aggregates provide pre-computed metrics for fast analytical queries without impacting real-time ingestion.

## 5) Grafana Visualization

### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Real-Time Geospatial Metrics",
    "panels": [
      {
        "title": "Temperature by Location",
        "type": "timeseries",
        "targets": [
          {
            "expr": "SELECT time, location_id, temperature FROM geospatial_metrics WHERE time > NOW() - INTERVAL '1 hour'",
            "format": "table"
          }
        ]
      },
      {
        "title": "Wind Speed and Direction",
        "type": "stat",
        "targets": [
          {
            "expr": "SELECT AVG(wind_speed) as avg_wind_speed FROM geospatial_metrics WHERE time > NOW() - INTERVAL '5 minutes'"
          }
        ]
      },
      {
        "title": "Data Ingestion Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "SELECT COUNT(*) as records_per_minute FROM geospatial_metrics WHERE time > NOW() - INTERVAL '1 minute'"
          }
        ]
      }
    ]
  }
}
```

**Why**: Grafana provides real-time visualization of streaming data with interactive dashboards and alerting capabilities.

### Alerting Rules

```yaml
# Grafana alerting configuration
alerting:
  rules:
    - alert: HighTemperature
      expr: avg(temperature) > 30
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High temperature detected"
        description: "Average temperature is {{ $value }}°C"
    
    - alert: DataIngestionStopped
      expr: rate(geospatial_metrics[5m]) == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Data ingestion stopped"
        description: "No data received in the last 5 minutes"
```

**Why**: Alerting ensures proactive monitoring of the real-time processing pipeline and rapid response to issues.

## 6) Performance Optimization

### Kafka Optimization

```python
# Optimized Kafka producer configuration
producer_config = {
    'bootstrap_servers': ['localhost:9092'],
    'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
    'key_serializer': lambda k: k.encode('utf-8') if k else None,
    'acks': 'all',  # Wait for all replicas
    'retries': 3,
    'retry_backoff_ms': 100,
    'request_timeout_ms': 30000,
    'batch_size': 16384,  # Batch size in bytes
    'linger_ms': 10,  # Wait up to 10ms to batch messages
    'compression_type': 'snappy',  # Compress messages
    'max_in_flight_requests_per_connection': 5,
    'enable_idempotence': True  # Ensure exactly-once delivery
}

# Optimized consumer configuration
consumer_config = {
    'bootstrap_servers': ['localhost:9092'],
    'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
    'key_deserializer': lambda k: k.decode('utf-8') if k else None,
    'group_id': 'geospatial-processors',
    'auto_offset_reset': 'earliest',
    'enable_auto_commit': True,
    'auto_commit_interval_ms': 1000,
    'fetch_min_bytes': 1,
    'fetch_max_wait_ms': 500,
    'max_poll_records': 500  # Process up to 500 records per poll
}
```

**Why**: Optimized Kafka configuration improves throughput, reduces latency, and ensures reliable message delivery.

### Database Optimization

```sql
-- Optimize TimescaleDB for high-throughput ingestion
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_geospatial_metrics_location_time 
ON geospatial_metrics (location_id, time DESC);

CREATE INDEX CONCURRENTLY idx_geospatial_metrics_time 
ON geospatial_metrics (time DESC);

-- Create partial index for recent data
CREATE INDEX CONCURRENTLY idx_geospatial_metrics_recent 
ON geospatial_metrics (location_id, time DESC) 
WHERE time > NOW() - INTERVAL '1 day';
```

**Why**: Proper database optimization ensures high-throughput ingestion and fast query performance for real-time analytics.

## 7) Monitoring and Observability

### Metrics Collection

```python
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Prometheus metrics
messages_processed = Counter('messages_processed_total', 'Total messages processed')
processing_duration = Histogram('processing_duration_seconds', 'Message processing duration')
queue_size = Gauge('kafka_queue_size', 'Current queue size')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')

class MonitoredConsumer(GeospatialDataConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start Prometheus metrics server
        start_http_server(8000)
    
    def process_message(self, message):
        start_time = time.time()
        
        try:
            # Process message
            super().process_message(message)
            
            # Update metrics
            messages_processed.inc()
            processing_duration.observe(time.time() - start_time)
            memory_usage.set(psutil.Process().memory_info().rss)
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise
```

**Why**: Comprehensive monitoring provides visibility into system performance and enables proactive optimization.

## 8) TL;DR Quickstart

```bash
# 1. Start the stack
docker-compose up -d

# 2. Create Kafka topic
kafka-topics --create --topic geospatial-data --bootstrap-server localhost:9092

# 3. Run producer
python producer.py

# 4. Run consumer
python consumer.py

# 5. View in Grafana
# Open http://localhost:3000 (admin/admin)
```

## 9) Anti-Patterns to Avoid

- **Don't ignore message ordering**—use proper partitioning for ordered processing
- **Don't skip error handling**—implement dead letter queues for failed messages
- **Don't ignore monitoring**—real-time systems need comprehensive observability
- **Don't skip database optimization**—TimescaleDB needs proper configuration for performance
- **Don't ignore data retention**—implement proper retention policies to manage storage

**Why**: These anti-patterns lead to data loss, performance issues, and unreliable real-time processing systems.

---

*This tutorial provides the foundation for building production-ready real-time data processing systems with Kafka and TimescaleDB.*
