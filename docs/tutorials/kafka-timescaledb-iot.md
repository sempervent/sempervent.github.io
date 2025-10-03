# Streaming IoT Telemetry with Kafka + TimescaleDB (via Docker Compose)

**Objective**: Simulate thousands of device ticks (JSON/Avro), publish to Kafka, sink into Timescale hypertables using Kafka Connect JDBC Sink. Cover schemas, partitioning, retention, and performance.

This tutorial builds a complete IoT data streaming pipeline that ingests device telemetry, processes it through Kafka, and stores it in TimescaleDB for time-series analytics. You'll learn production patterns for high-throughput data ingestion and real-time analytics.

## 1) Prerequisites

```bash
# Required tools
docker --version          # >= 20.10
docker compose --version  # >= 2.0
psql --version           # For direct DB access
curl --version           # For API calls
jq --version             # For JSON parsing (optional but helpful)

# System requirements
# - 4-8 GB RAM recommended
# - 10+ GB free disk space
# - Docker Desktop or Docker Engine running
```

**Why**: These tools provide the foundation for running the complete streaming stack locally with proper monitoring and debugging capabilities.

## 2) Repository Layout

Create the following file structure:

```
kafka-timescale-iot/
├─ docker-compose.yaml
├─ connectors/
│  └─ jdbc-sink-timescale.json
├─ timescale/
│  ├─ init.sql
│  └─ postgresql.conf
├─ simulator/
│  ├─ requirements.txt
│  └─ simulator.py
└─ schemas/
   └─ iot-value.avsc
```

**Why**: This structure separates concerns—Docker orchestration, database initialization, data connectors, simulation logic, and schemas—making the system maintainable and debuggable.

## 3) Docker Compose with Profiles

Create `docker-compose.yaml`:

```yaml
version: '3.8'

services:
  # Zookeeper for Kafka coordination
  zookeeper:
    image: bitnami/zookeeper:latest
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
      ZOOKEEPER_SERVER_ID: 1
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/bitnami/zookeeper
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5
    profiles: ["kafka"]

  # Kafka broker
  kafka:
    image: bitnami/kafka:latest
    container_name: kafka
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_CFG_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: 'PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT'
      KAFKA_CFG_LISTENERS: 'PLAINTEXT://0.0.0.0:9092,PLAINTEXT_HOST://0.0.0.0:29092'
      KAFKA_CFG_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092'
      KAFKA_CFG_INTER_BROKER_LISTENER_NAME: 'PLAINTEXT'
      KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_CFG_NUM_PARTITIONS: 3
      KAFKA_CFG_DEFAULT_REPLICATION_FACTOR: 1
      KAFKA_CFG_LOG_RETENTION_HOURS: 24
      KAFKA_CFG_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_CFG_LOG_COMPRESSION_TYPE: 'zstd'
      KAFKA_CFG_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_CFG_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_CFG_TRANSACTION_STATE_LOG_MIN_ISR: 1
    ports:
      - "29092:29092"
    volumes:
      - kafka_data:/bitnami/kafka
    healthcheck:
      test: ["CMD", "kafka-topics.sh", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 10s
      timeout: 5s
      retries: 5
    profiles: ["kafka"]

  # Schema Registry (optional for Avro)
  schema-registry:
    image: confluentinc/cp-schema-registry:7.6.1
    container_name: schema-registry
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_LISTENERS: http://0.0.0.0:8081
    ports:
      - "8081:8081"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/subjects"]
      interval: 10s
      timeout: 5s
      retries: 5
    profiles: ["kafka", "avro"]

  # Kafka Connect
  kafka-connect:
    image: confluentinc/cp-kafka-connect:7.6.1
    container_name: kafka-connect
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      CONNECT_BOOTSTRAP_SERVERS: kafka:9092
      CONNECT_REST_ADVERTISED_HOST_NAME: kafka-connect
      CONNECT_REST_PORT: 8083
      CONNECT_GROUP_ID: connect-cluster
      CONNECT_CONFIG_STORAGE_TOPIC: connect-configs
      CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_OFFSET_STORAGE_TOPIC: connect-offsets
      CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_STATUS_STORAGE_TOPIC: connect-status
      CONNECT_STATUS_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_KEY_CONVERTER: org.apache.kafka.connect.storage.StringConverter
      CONNECT_VALUE_CONVERTER: org.apache.kafka.connect.json.JsonConverter
      CONNECT_VALUE_CONVERTER_SCHEMAS_ENABLE: false
      CONNECT_PLUGIN_PATH: "/usr/share/java,/usr/share/confluent-hub-components"
      CONNECT_LOG4J_LOGGERS: org.apache.zookeeper=ERROR,org.I0Itec.zkclient=ERROR,org.reflections=ERROR
    ports:
      - "8083:8083"
    volumes:
      - ./connectors:/connectors
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8083/connectors"]
      interval: 10s
      timeout: 5s
      retries: 5
    profiles: ["kafka"]

  # Kafka UI for monitoring
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    ports:
      - "8080:8080"
    profiles: ["kafka"]

  # TimescaleDB
  timescale:
    image: timescale/timescaledb:2.15.2-pg16
    container_name: timescale
    environment:
      POSTGRES_DB: iotdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./timescale/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
      - ./timescale/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    command: ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d iotdb"]
      interval: 5s
      timeout: 5s
      retries: 10
    profiles: ["db"]

  # IoT Simulator
  simulator:
    build:
      context: ./simulator
      dockerfile: Dockerfile
    container_name: simulator
    depends_on:
      kafka:
        condition: service_healthy
    environment:
      KAFKA_BROKER: kafka:9092
      TOPIC: iot.readings.json
      EVENTS_PER_SEC: 200
      DEVICE_COUNT: 5000
    profiles: ["sim"]

volumes:
  zookeeper_data:
  kafka_data:
  timescale_data:

# Profile definitions
# kafka: Core Kafka infrastructure
# db: TimescaleDB database
# json: JSON serialization (default)
# avro: Avro serialization with Schema Registry
# sim: IoT simulator
# all: Complete stack (kafka + db + json + sim)
```

**Why**: This Docker Compose setup provides a complete streaming infrastructure with health checks, proper networking, and configurable profiles for different deployment scenarios.

## 4) TimescaleDB Initialization

Create `timescale/init.sql`:

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create IoT readings table
CREATE TABLE IF NOT EXISTS iot_readings (
    ts          TIMESTAMPTZ       NOT NULL,
    device_id  TEXT              NOT NULL,
    sensor     TEXT              NOT NULL,
    value      DOUBLE PRECISION  NOT NULL,
    location   TEXT              NULL,
    meta       JSONB             NULL
);

-- Create hypertable with time partitioning
SELECT create_hypertable('iot_readings', 'ts', if_not_exists => TRUE);

-- Create composite index for common queries
CREATE INDEX IF NOT EXISTS ix_iot_ts_device ON iot_readings (ts DESC, device_id);
CREATE INDEX IF NOT EXISTS ix_iot_device_sensor ON iot_readings (device_id, sensor, ts DESC);

-- Enable compression for older data
ALTER TABLE iot_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',
    timescaledb.compress_orderby = 'ts DESC'
);

-- Add compression policy (compress after 7 days)
SELECT add_compression_policy('iot_readings', INTERVAL '7 days');

-- Add retention policy (drop data after 90 days)
SELECT add_retention_policy('iot_readings', INTERVAL '90 days');

-- Create continuous aggregate for 1-minute rollups
CREATE MATERIALIZED VIEW iot_rollup_1m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute', ts) AS bucket,
    device_id,
    sensor,
    avg(value) AS avg_value,
    min(value) AS min_value,
    max(value) AS max_value,
    count(*) AS reading_count
FROM iot_readings
GROUP BY 1, 2, 3;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('iot_rollup_1m',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- Create 5-minute rollup for dashboard queries
CREATE MATERIALIZED VIEW iot_rollup_5m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', ts) AS bucket,
    device_id,
    sensor,
    avg(value) AS avg_value,
    min(value) AS min_value,
    max(value) AS max_value,
    count(*) AS reading_count
FROM iot_readings
GROUP BY 1, 2, 3;

SELECT add_continuous_aggregate_policy('iot_rollup_5m',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
```

**Why**: This initialization script sets up TimescaleDB with proper hypertables, compression, retention policies, and continuous aggregates for efficient time-series analytics.

## 5) PostgreSQL Configuration

Create `timescale/postgresql.conf`:

```conf
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB

# WAL settings for high-throughput ingestion
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 1GB
min_wal_size = 80MB

# Connection settings
max_connections = 200
shared_preload_libraries = 'timescaledb'

# Logging for debugging
log_statement = 'none'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on

# TimescaleDB specific settings
timescaledb.max_background_workers = 8
```

**Why**: These settings optimize PostgreSQL for time-series workloads with high write throughput and efficient compression.

## 6) Kafka Connect JDBC Sink

Create `connectors/jdbc-sink-timescale.json`:

```json
{
  "name": "jdbc-sink-timescale",
  "config": {
    "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
    "tasks.max": "3",
    "topics": "iot.readings.json",
    "connection.url": "jdbc:postgresql://timescale:5432/iotdb",
    "connection.user": "postgres",
    "connection.password": "postgres",
    "auto.create": "false",
    "auto.evolve": "false",
    "insert.mode": "insert",
    "pk.mode": "none",
    "table.name.format": "iot_readings",
    "fields.whitelist": "ts,device_id,sensor,value,location,meta",
    "key.converter": "org.apache.kafka.connect.storage.StringConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "false",
    "batch.size": "5000",
    "max.retries": "10",
    "retry.backoff.ms": "1000",
    "connection.attempts": "10",
    "connection.backoff.ms": "1000",
    "poll.interval.ms": "1000",
    "flush.size": "5000",
    "errors.tolerance": "all",
    "errors.log.enable": "true",
    "errors.log.include.messages": "true"
  }
}
```

**Why**: This connector configuration optimizes for high-throughput ingestion with proper error handling, batching, and retry logic for production reliability.

## 7) IoT Simulator

Create `simulator/requirements.txt`:

```txt
kafka-python>=2.0.2
orjson>=3.10.0
python-dateutil>=2.9.0
faker>=24.0.0
```

Create `simulator/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "simulator.py"]
```

Create `simulator/simulator.py`:

```python
#!/usr/bin/env python3
"""
IoT Telemetry Simulator for Kafka + TimescaleDB Pipeline
Generates realistic device telemetry data and streams to Kafka
"""

import json
import time
import random
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import orjson
from kafka import KafkaProducer
from kafka.errors import KafkaError
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IoTDeviceSimulator:
    """Simulates IoT devices generating telemetry data"""
    
    def __init__(self, device_id: str, location: str):
        self.device_id = device_id
        self.location = location
        self.fake = Faker()
        
        # Device characteristics
        self.sensors = ['temperature', 'humidity', 'pressure', 'vibration', 'voltage']
        self.base_values = {
            'temperature': 20.0,  # Celsius
            'humidity': 50.0,      # Percentage
            'pressure': 1013.25,   # hPa
            'vibration': 0.1,      # g
            'voltage': 12.0        # Volts
        }
        self.value_ranges = {
            'temperature': (15.0, 35.0),
            'humidity': (30.0, 80.0),
            'pressure': (1000.0, 1020.0),
            'vibration': (0.05, 0.5),
            'voltage': (11.5, 12.5)
        }
    
    def generate_reading(self) -> Dict[str, Any]:
        """Generate a single sensor reading"""
        sensor = random.choice(self.sensors)
        base_value = self.base_values[sensor]
        min_val, max_val = self.value_ranges[sensor]
        
        # Add some drift and noise
        drift = random.uniform(-0.1, 0.1)
        noise = random.uniform(-0.05, 0.05)
        value = base_value + drift + noise
        
        # Clamp to realistic range
        value = max(min_val, min(max_val, value))
        
        return {
            'ts': datetime.now(timezone.utc).isoformat(),
            'device_id': self.device_id,
            'sensor': sensor,
            'value': round(value, 3),
            'location': self.location,
            'meta': {
                'firmware_version': f"v{random.randint(1, 3)}.{random.randint(0, 9)}",
                'battery_level': random.randint(20, 100),
                'signal_strength': random.randint(-100, -30)
            }
        }

class IoTTelemetryProducer:
    """Produces IoT telemetry data to Kafka"""
    
    def __init__(self, broker: str, topic: str, events_per_sec: int = 100):
        self.broker = broker
        self.topic = topic
        self.events_per_sec = events_per_sec
        self.interval = 1.0 / events_per_sec
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=[broker],
            value_serializer=lambda v: orjson.dumps(v),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            retry_backoff_ms=100,
            request_timeout_ms=30000,
            batch_size=16384,
            linger_ms=10,
            compression_type='zstd'
        )
        
        # Initialize device simulators
        self.devices = self._create_devices()
        logger.info(f"Created {len(self.devices)} device simulators")
    
    def _create_devices(self) -> List[IoTDeviceSimulator]:
        """Create device simulators"""
        devices = []
        fake = Faker()
        
        # Create devices with realistic locations
        locations = [
            'Building A - Floor 1', 'Building A - Floor 2', 'Building A - Floor 3',
            'Building B - Floor 1', 'Building B - Floor 2', 'Building B - Floor 3',
            'Warehouse - Zone 1', 'Warehouse - Zone 2', 'Warehouse - Zone 3',
            'Factory Floor - Line 1', 'Factory Floor - Line 2', 'Factory Floor - Line 3'
        ]
        
        for i in range(1, 5001):  # 5000 devices
            device_id = f"device-{i:06d}"
            location = random.choice(locations)
            devices.append(IoTDeviceSimulator(device_id, location))
        
        return devices
    
    def produce_telemetry(self):
        """Produce telemetry data at configured rate"""
        logger.info(f"Starting telemetry production at {self.events_per_sec} events/sec")
        
        try:
            while True:
                start_time = time.time()
                
                # Generate reading from random device
                device = random.choice(self.devices)
                reading = device.generate_reading()
                
                # Send to Kafka
                future = self.producer.send(
                    self.topic,
                    key=reading['device_id'],
                    value=reading
                )
                
                # Handle send result
                try:
                    record_metadata = future.get(timeout=1)
                    logger.debug(f"Sent to {record_metadata.topic} partition {record_metadata.partition}")
                except KafkaError as e:
                    logger.error(f"Failed to send message: {e}")
                
                # Rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Stopping telemetry production")
        finally:
            self.producer.close()
            logger.info("Producer closed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IoT Telemetry Simulator')
    parser.add_argument('--broker', default='localhost:9092', help='Kafka broker address')
    parser.add_argument('--topic', default='iot.readings.json', help='Kafka topic')
    parser.add_argument('--eps', type=int, default=200, help='Events per second')
    parser.add_argument('--devices', type=int, default=5000, help='Number of devices')
    
    args = parser.parse_args()
    
    # Override from environment if set
    import os
    broker = os.getenv('KAFKA_BROKER', args.broker)
    topic = os.getenv('TOPIC', args.topic)
    eps = int(os.getenv('EVENTS_PER_SEC', args.eps))
    
    logger.info(f"Starting IoT simulator: {broker}/{topic} @ {eps} eps")
    
    producer = IoTTelemetryProducer(broker, topic, eps)
    producer.produce_telemetry()

if __name__ == '__main__':
    main()
```

**Why**: This simulator generates realistic IoT telemetry data with proper Kafka integration, rate limiting, and error handling for production-like testing.

## 8) Avro Schema (Optional)

Create `schemas/iot-value.avsc`:

```json
{
  "type": "record",
  "name": "IoTReading",
  "namespace": "com.example.iot",
  "fields": [
    {
      "name": "ts",
      "type": "string",
      "doc": "ISO8601 timestamp"
    },
    {
      "name": "device_id",
      "type": "string",
      "doc": "Device identifier"
    },
    {
      "name": "sensor",
      "type": "string",
      "doc": "Sensor type"
    },
    {
      "name": "value",
      "type": "double",
      "doc": "Sensor reading value"
    },
    {
      "name": "location",
      "type": ["null", "string"],
      "default": null,
      "doc": "Device location"
    },
    {
      "name": "meta",
      "type": ["null", {
        "type": "map",
        "values": "string"
      }],
      "default": null,
      "doc": "Additional metadata"
    }
  ]
}
```

**Why**: Avro provides schema evolution and better serialization performance for high-throughput scenarios.

## 9) Bring It All Up

### Start the Infrastructure

```bash
# 1) Start core infrastructure (Kafka + TimescaleDB)
docker compose --profile kafka --profile db --profile json up -d

# 2) Wait for services to be healthy
docker compose ps

# 3) Check TimescaleDB is ready
docker compose exec timescale psql -U postgres -d iotdb -c "SELECT version();"
```

**Why**: Starting with core services ensures the database is ready before connecting data sources.

### Register Kafka Connect Sink

```bash
# 4) Register the JDBC sink connector
curl -s -X POST -H "Content-Type: application/json" \
  --data @connectors/jdbc-sink-timescale.json \
  http://localhost:8083/connectors | jq

# 5) Check connector status
curl -s http://localhost:8083/connectors/jdbc-sink-timescale/status | jq
```

**Why**: The connector bridges Kafka and TimescaleDB, handling serialization and batching automatically.

### Start Data Generation

```bash
# 6) Start the IoT simulator
docker compose --profile sim up -d simulator

# 7) Check simulator logs
docker compose logs -f simulator
```

**Why**: The simulator provides realistic data load for testing the complete pipeline.

## 10) Validation and Monitoring

### Verify Data Flow

```bash
# Check Kafka topics
docker compose exec kafka kafka-topics.sh --bootstrap-server kafka:9092 --list

# Consume a few messages from Kafka
docker compose exec kafka kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic iot.readings.json \
  --from-beginning \
  --max-messages 5

# Check data in TimescaleDB
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT count(*) as total_readings FROM iot_readings;
"

# Check recent data
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT device_id, sensor, avg(value) as avg_value, count(*) as readings
FROM iot_readings 
WHERE ts > now() - interval '5 minutes' 
GROUP BY device_id, sensor 
ORDER BY device_id, sensor 
LIMIT 10;
"
```

**Why**: These validation steps confirm the complete data pipeline is working correctly.

### Monitor Performance

```bash
# Check connector metrics
curl -s http://localhost:8083/connectors/jdbc-sink-timescale/status | jq '.tasks[0].metrics'

# Check TimescaleDB performance
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'iot_readings';
"

# Check continuous aggregates
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT * FROM iot_rollup_1m 
WHERE bucket > now() - interval '1 hour' 
ORDER BY bucket DESC 
LIMIT 5;
"
```

**Why**: Monitoring ensures the system is performing optimally and helps identify bottlenecks.

## 11) Query Patterns and Analytics

### Time-Series Queries

```sql
-- Recent readings by device
SELECT 
    device_id,
    sensor,
    time_bucket('1 minute', ts) as minute,
    avg(value) as avg_value,
    min(value) as min_value,
    max(value) as max_value,
    count(*) as reading_count
FROM iot_readings 
WHERE ts > now() - interval '1 hour'
GROUP BY device_id, sensor, minute
ORDER BY device_id, sensor, minute;

-- Device health summary
SELECT 
    device_id,
    count(*) as total_readings,
    count(DISTINCT sensor) as sensor_count,
    min(ts) as first_reading,
    max(ts) as last_reading
FROM iot_readings 
WHERE ts > now() - interval '1 day'
GROUP BY device_id
ORDER BY total_readings DESC
LIMIT 10;

-- Sensor anomaly detection
WITH sensor_stats AS (
    SELECT 
        sensor,
        avg(value) as mean_value,
        stddev(value) as std_value
    FROM iot_readings 
    WHERE ts > now() - interval '1 hour'
    GROUP BY sensor
)
SELECT 
    i.device_id,
    i.sensor,
    i.value,
    s.mean_value,
    s.std_value,
    abs(i.value - s.mean_value) / s.std_value as z_score
FROM iot_readings i
JOIN sensor_stats s ON i.sensor = s.sensor
WHERE i.ts > now() - interval '10 minutes'
    AND abs(i.value - s.mean_value) / s.std_value > 2
ORDER BY z_score DESC;
```

**Why**: These queries demonstrate common time-series analytics patterns for IoT data monitoring and alerting.

### Continuous Aggregates

```sql
-- Create additional rollup for dashboard
CREATE MATERIALIZED VIEW iot_dashboard_1h
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', ts) AS bucket,
    device_id,
    sensor,
    avg(value) AS avg_value,
    min(value) AS min_value,
    max(value) AS max_value,
    count(*) AS reading_count
FROM iot_readings
GROUP BY 1, 2, 3;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('iot_dashboard_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Query dashboard data
SELECT * FROM iot_dashboard_1h 
WHERE bucket > now() - interval '24 hours'
ORDER BY bucket DESC, device_id, sensor;
```

**Why**: Continuous aggregates provide pre-computed metrics for fast dashboard queries without impacting real-time ingestion.

## 12) Performance Tuning

### Kafka Optimization

```yaml
# Add to docker-compose.yaml for production
environment:
  KAFKA_CFG_NUM_PARTITIONS: 6
  KAFKA_CFG_DEFAULT_REPLICATION_FACTOR: 1
  KAFKA_CFG_LOG_RETENTION_HOURS: 168  # 7 days
  KAFKA_CFG_LOG_SEGMENT_BYTES: 1073741824  # 1GB
  KAFKA_CFG_LOG_COMPRESSION_TYPE: 'zstd'
  KAFKA_CFG_LOG_FLUSH_INTERVAL_MESSAGES: 10000
  KAFKA_CFG_LOG_FLUSH_INTERVAL_MS: 1000
  KAFKA_CFG_MESSAGE_MAX_BYTES: 10485760  # 10MB
  KAFKA_CFG_REPLICA_FETCH_MAX_BYTES: 10485760
```

### TimescaleDB Optimization

```sql
-- Optimize for high-throughput ingestion
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET wal_buffers = '32MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_wal_size = '2GB';
ALTER SYSTEM SET min_wal_size = '160MB';

-- Create additional indexes for common queries
CREATE INDEX CONCURRENTLY idx_iot_sensor_ts ON iot_readings (sensor, ts DESC);
CREATE INDEX CONCURRENTLY idx_iot_location_ts ON iot_readings (location, ts DESC) WHERE location IS NOT NULL;
```

### Connect Optimization

```json
{
  "batch.size": "10000",
  "flush.size": "10000",
  "poll.interval.ms": "500",
  "tasks.max": "6"
}
```

**Why**: These optimizations improve throughput, reduce latency, and ensure reliable data processing at scale.

## 13) Switching to Avro + Schema Registry

### Update Docker Compose

```bash
# Start with Avro profile
docker compose --profile kafka --profile db --profile avro up -d

# Update connector for Avro
curl -s -X PUT -H "Content-Type: application/json" \
  --data '{
    "name": "jdbc-sink-timescale",
    "config": {
      "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
      "tasks.max": "3",
      "topics": "iot.readings.avro",
      "connection.url": "jdbc:postgresql://timescale:5432/iotdb",
      "connection.user": "postgres",
      "connection.password": "postgres",
      "auto.create": "false",
      "auto.evolve": "false",
      "insert.mode": "insert",
      "pk.mode": "none",
      "table.name.format": "iot_readings",
      "fields.whitelist": "ts,device_id,sensor,value,location,meta",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "value.converter": "io.confluent.connect.avro.AvroConverter",
      "value.converter.schema.registry.url": "http://schema-registry:8081",
      "value.converter.schemas.enable": "true",
      "batch.size": "5000",
      "max.retries": "10",
      "retry.backoff.ms": "1000"
    }
  }' \
  http://localhost:8083/connectors/jdbc-sink-timescale/config | jq
```

**Why**: Avro provides schema evolution and better serialization performance for production systems.

## 14) Troubleshooting

### Common Issues

```bash
# Check connector status
curl -s http://localhost:8083/connectors/jdbc-sink-timescale/status | jq

# Check connector logs
docker compose logs kafka-connect

# Check Kafka consumer lag
docker compose exec kafka kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group connect-jdbc-sink-timescale \
  --describe

# Check TimescaleDB connections
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';
"

# Check for data quality issues
docker compose exec timescale psql -U postgres -d iotdb -c "
SELECT 
    device_id,
    count(*) as readings,
    min(ts) as first_reading,
    max(ts) as last_reading,
    max(ts) - min(ts) as duration
FROM iot_readings 
WHERE ts > now() - interval '1 hour'
GROUP BY device_id
HAVING count(*) < 100  -- Expected readings per hour
ORDER BY readings;
"
```

**Why**: These diagnostic commands help identify and resolve common issues in the streaming pipeline.

## 15) Clean Up

```bash
# Stop all services
docker compose down

# Remove volumes (deletes all data)
docker compose down -v

# Remove images (optional)
docker compose down --rmi all
```

**Why**: Proper cleanup prevents resource conflicts and ensures clean test environments.

## 16) TL;DR Quickstart

```bash
# 1) Clone and setup
git clone <repo> && cd kafka-timescale-iot

# 2) Start infrastructure
docker compose --profile kafka --profile db --profile json up -d

# 3) Register sink connector
curl -s -X POST -H "Content-Type: application/json" \
  --data @connectors/jdbc-sink-timescale.json \
  http://localhost:8083/connectors | jq

# 4) Start simulator
docker compose --profile sim up -d simulator

# 5) Verify data flow
docker compose exec timescale psql -U postgres -d iotdb -c "SELECT count(*) FROM iot_readings;"

# 6) Monitor in Kafka UI
open http://localhost:8080
```

## 17) Production Considerations

### Scaling Patterns

- **Horizontal scaling**: Increase `tasks.max` in connector config
- **Partitioning**: Use device_id as partition key for locality
- **Batching**: Optimize batch.size and flush.size for throughput
- **Monitoring**: Implement comprehensive observability
- **Security**: Add authentication and encryption
- **Backup**: Implement TimescaleDB backup strategies

### Anti-Patterns to Avoid

- **Don't ignore backpressure**—monitor consumer lag
- **Don't skip compression**—enable TimescaleDB compression
- **Don't ignore retention**—set appropriate data retention policies
- **Don't skip monitoring**—implement comprehensive observability
- **Don't ignore error handling**—configure proper retry and dead letter queues

**Why**: These patterns ensure production reliability and performance at scale.

---

*This tutorial provides a complete, production-ready IoT streaming pipeline that scales from development to production workloads.*
