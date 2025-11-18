# Event-Driven Microservices with Observability and Data Governance: A Complete Integration Tutorial

**Objective**: Build a production-ready event-driven microservices system that integrates event-driven architecture patterns, observability-driven development, data validation and contract governance, API gateway architecture, and service decomposition strategies. This tutorial demonstrates how these best practices work together in a real-world application.

This tutorial combines:
- **[Event-Driven Architecture](../best-practices/architecture-design/event-driven-architecture.md)** - Event patterns, messaging, and decoupling
- **[Observability-Driven Development](../best-practices/operations-monitoring/observability-driven-development.md)** - Telemetry-first coding and preemptive debugging
- **[Data Validation and Contract Governance](../best-practices/data-governance/data-validation-and-contract-governance.md)** - Schema validation and data contracts
- **[API Gateway Architecture](../best-practices/architecture-design/api-gateway-architecture.md)** - Unified API entry point and routing
- **[Service Decomposition Strategy](../best-practices/architecture-design/service-decomposition-strategy.md)** - Domain boundaries and service design

## 1) Prerequisites

```bash
# Required tools
docker --version          # >= 20.10
docker compose --version  # >= 2.0
python --version          # >= 3.10
go --version              # >= 1.21
psql --version           # For database access
curl --version           # For API testing
jq --version             # For JSON parsing

# System requirements
# - 8+ GB RAM recommended
# - 20+ GB free disk space
# - Docker Desktop or Docker Engine running
```

**Why**: This stack requires multiple languages (Python, Go) and infrastructure components (Kafka, Postgres, Redis) to demonstrate polyglot best practices integration.

## 2) Architecture Overview

We'll build an **Order Processing System** with the following services:

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Client]
        API_CLIENT[API Client]
    end

    subgraph "API Gateway"
        GATEWAY[Kong API Gateway]
        AUTH[Auth Service]
        RATE[Rate Limiter]
    end

    subgraph "Microservices"
        ORDER[Order Service<br/>Python/FastAPI]
        PAYMENT[Payment Service<br/>Go]
        INVENTORY[Inventory Service<br/>Python/FastAPI]
        NOTIFICATION[Notification Service<br/>Go]
    end

    subgraph "Event Bus"
        KAFKA[Kafka]
        SCHEMA[Schema Registry]
    end

    subgraph "Data Layer"
        POSTGRES[(PostgreSQL<br/>Orders DB)]
        REDIS[(Redis<br/>Cache)]
    end

    subgraph "Observability"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        LOKI[Loki]
        JAEGER[Jaeger]
    end

    WEB --> GATEWAY
    API_CLIENT --> GATEWAY
    GATEWAY --> AUTH
    GATEWAY --> RATE
    GATEWAY --> ORDER
    GATEWAY --> PAYMENT
    GATEWAY --> INVENTORY

    ORDER --> KAFKA
    PAYMENT --> KAFKA
    INVENTORY --> KAFKA
    NOTIFICATION --> KAFKA

    KAFKA --> ORDER
    KAFKA --> PAYMENT
    KAFKA --> INVENTORY
    KAFKA --> NOTIFICATION

    ORDER --> POSTGRES
    ORDER --> REDIS
    PAYMENT --> POSTGRES
    INVENTORY --> POSTGRES

    ORDER --> PROMETHEUS
    PAYMENT --> PROMETHEUS
    INVENTORY --> PROMETHEUS
    NOTIFICATION --> PROMETHEUS

    ORDER --> JAEGER
    PAYMENT --> JAEGER
    INVENTORY --> JAEGER
    NOTIFICATION --> JAEGER

    PROMETHEUS --> GRAFANA
    LOKI --> GRAFANA
    JAEGER --> GRAFANA
```

**Domain Boundaries**:
- **Order Domain**: Order creation, status tracking
- **Payment Domain**: Payment processing, refunds
- **Inventory Domain**: Stock management, reservations
- **Notification Domain**: Email, SMS, push notifications

## 3) Repository Layout

```
event-driven-microservices/
├── docker-compose.yaml
├── api-gateway/
│   ├── kong.yml
│   └── plugins/
│       └── rate-limiting.yml
├── services/
│   ├── order-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── models.py
│   │   │   ├── schemas.py
│   │   │   ├── events.py
│   │   │   ├── observability.py
│   │   │   └── handlers.py
│   │   └── tests/
│   ├── payment-service/
│   │   ├── Dockerfile
│   │   ├── go.mod
│   │   ├── main.go
│   │   ├── models.go
│   │   ├── events.go
│   │   ├── observability.go
│   │   └── handlers.go
│   ├── inventory-service/
│   │   └── [similar structure to order-service]
│   └── notification-service/
│       └── [similar structure to payment-service]
├── kafka/
│   ├── schemas/
│   │   ├── order-created.avsc
│   │   ├── payment-processed.avsc
│   │   └── inventory-updated.avsc
│   └── connectors/
├── observability/
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── loki/
│       └── loki-config.yml
└── scripts/
    ├── setup.sh
    └── test-events.sh
```

**Why**: This structure separates concerns—services, infrastructure, observability, and schemas—enabling independent development and deployment.

## 4) Docker Compose Infrastructure

Create `docker-compose.yaml`:

```yaml
version: '3.8'

services:
  # Zookeeper for Kafka
  zookeeper:
    image: bitnami/zookeeper:latest
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/bitnami/zookeeper
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5

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
      KAFKA_CFG_LISTENERS: 'PLAINTEXT://0.0.0.0:9092,PLAINTEXT_HOST://0.0.0.0:29092'
      KAFKA_CFG_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092'
      KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_CFG_NUM_PARTITIONS: 3
    ports:
      - "29092:29092"
    volumes:
      - kafka_data:/bitnami/kafka
    healthcheck:
      test: ["CMD", "kafka-topics.sh", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Schema Registry
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

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: orders_db
      POSTGRES_USER: orders_user
      POSTGRES_PASSWORD: orders_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U orders_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Kong API Gateway
  kong:
    image: kong:3.4
    container_name: kong
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/kong.yml
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG: /dev/stderr
      KONG_ADMIN_ERROR_LOG: /dev/stderr
      KONG_ADMIN_LISTEN: 0.0.0.0:8001
    ports:
      - "8000:8000"
      - "8443:8443"
      - "8001:8001"
      - "8444:8444"
    volumes:
      - ./api-gateway/kong.yml:/kong/kong.yml
    depends_on:
      - order-service
      - payment-service
      - inventory-service
    healthcheck:
      test: ["CMD", "kong", "health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Order Service (Python/FastAPI)
  order-service:
    build:
      context: ./services/order-service
      dockerfile: Dockerfile
    container_name: order-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres:5432/orders_db
      REDIS_URL: redis://redis:6379/0
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_URL: http://schema-registry:8081
      JAEGER_AGENT_HOST: jaeger
      JAEGER_AGENT_PORT: 6831
      PROMETHEUS_PORT: 9090
    ports:
      - "8002:8000"
      - "9090:9090"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      kafka:
        condition: service_healthy
      schema-registry:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Payment Service (Go)
  payment-service:
    build:
      context: ./services/payment-service
      dockerfile: Dockerfile
    container_name: payment-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres:5432/orders_db
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_URL: http://schema-registry:8081
      JAEGER_AGENT_HOST: jaeger
      JAEGER_AGENT_PORT: 6831
      PROMETHEUS_PORT: 9091
    ports:
      - "8003:8000"
      - "9091:9091"
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_healthy
      schema-registry:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Inventory Service (Python/FastAPI)
  inventory-service:
    build:
      context: ./services/inventory-service
      dockerfile: Dockerfile
    container_name: inventory-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres:5432/orders_db
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_URL: http://schema-registry:8081
      JAEGER_AGENT_HOST: jaeger
      JAEGER_AGENT_PORT: 6831
      PROMETHEUS_PORT: 9092
    ports:
      - "8004:8000"
      - "9092:9092"
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_healthy
      schema-registry:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Notification Service (Go)
  notification-service:
    build:
      context: ./services/notification-service
      dockerfile: Dockerfile
    container_name: notification-service
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_URL: http://schema-registry:8081
      JAEGER_AGENT_HOST: jaeger
      JAEGER_AGENT_PORT: 6831
      PROMETHEUS_PORT: 9093
    ports:
      - "8005:8000"
      - "9093:9093"
    depends_on:
      kafka:
        condition: service_healthy
      schema-registry:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9094:9090"
    volumes:
      - ./observability/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    depends_on:
      - order-service
      - payment-service
      - inventory-service
      - notification-service

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./observability/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  # Loki
  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./observability/loki/loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    environment:
      COLLECTOR_ZIPKIN_HOST_PORT: :9411
    ports:
      - "16686:16686"
      - "6831:6831/udp"
      - "6832:6832/udp"
    depends_on:
      - order-service
      - payment-service

volumes:
  zookeeper_data:
  kafka_data:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  loki_data:
```

## 5) Order Service Implementation (Python/FastAPI)

### 5.1) Service Structure with Observability

Create `services/order-service/app/observability.py`:

```python
"""Observability-driven development: telemetry from day one."""
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from functools import wraps

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - '
    'trace_id=%(trace_id)s span_id=%(span_id)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# OpenTelemetry setup
resource = Resource.create({"service.name": "order-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
orders_created = Counter(
    'orders_created_total',
    'Total number of orders created',
    ['status']
)
order_processing_duration = Histogram(
    'order_processing_duration_seconds',
    'Time spent processing orders',
    ['operation']
)
active_orders = Gauge(
    'active_orders',
    'Number of active orders'
)

# Instrumentation
Psycopg2Instrumentor().instrument()
RedisInstrumentor().instrument()


@contextmanager
def trace_operation(operation_name: str, **attributes):
    """Context manager for tracing operations with attributes."""
    span = tracer.start_span(operation_name)
    for key, value in attributes.items():
        span.set_attribute(key, str(value))
    try:
        yield span
    except Exception as e:
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


def observe_function(func):
    """Decorator for automatic observability."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        operation = f"{func.__module__}.{func.__name__}"
        start_time = time.time()
        
        with trace_operation(operation):
            logger.info(f"Starting {operation}", extra={
                "operation": operation,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                order_processing_duration.labels(operation=operation).observe(duration)
                
                logger.info(f"Completed {operation}", extra={
                    "operation": operation,
                    "duration_seconds": duration,
                    "success": True
                })
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation}", extra={
                    "operation": operation,
                    "duration_seconds": duration,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise
    
    return wrapper
```

### 5.2) Data Contracts and Validation

Create `services/order-service/app/schemas.py`:

```python
"""Data validation and contract governance."""
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.dataclasses import dataclass


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    """Order item with validation."""
    product_id: str = Field(..., min_length=1, max_length=100)
    quantity: int = Field(..., gt=0, le=1000)
    price: Decimal = Field(..., gt=0, decimal_places=2)
    
    @validator('product_id')
    def validate_product_id(cls, v):
        """Validate product ID format."""
        if not v.isalnum():
            raise ValueError("product_id must be alphanumeric")
        return v
    
    @property
    def total(self) -> Decimal:
        """Calculate item total."""
        return self.quantity * self.price


class OrderCreate(BaseModel):
    """Order creation contract."""
    customer_id: str = Field(..., min_length=1, max_length=100)
    items: List[OrderItem] = Field(..., min_items=1, max_items=100)
    shipping_address: str = Field(..., min_length=10, max_length=500)
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        """Validate customer ID format."""
        if not v.isalnum():
            raise ValueError("customer_id must be alphanumeric")
        return v
    
    @root_validator
    def validate_order_total(cls, values):
        """Validate order business rules."""
        items = values.get('items', [])
        if not items:
            raise ValueError("Order must have at least one item")
        
        total = sum(item.total for item in items)
        if total > Decimal('100000'):
            raise ValueError("Order total cannot exceed $100,000")
        
        return values


class OrderResponse(BaseModel):
    """Order response contract."""
    order_id: str
    customer_id: str
    status: OrderStatus
    items: List[OrderItem]
    total: Decimal
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
```

### 5.3) Event Publishing

Create `services/order-service/app/events.py`:

```python
"""Event-driven architecture: event publishing."""
import json
import logging
from datetime import datetime
from typing import Dict, Any

from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField

from app.schemas import OrderResponse, OrderStatus

logger = logging.getLogger(__name__)


class EventPublisher:
    """Publishes events to Kafka with schema validation."""
    
    def __init__(self, kafka_bootstrap_servers: str, schema_registry_url: str):
        self.producer = Producer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'max.in.flight.requests.per.connection': 1,  # Ensure ordering
        })
        
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        self._load_schemas()
    
    def _load_schemas(self):
        """Load Avro schemas for events."""
        # In production, load from schema registry or files
        self.order_created_schema = {
            "type": "record",
            "name": "OrderCreated",
            "fields": [
                {"name": "order_id", "type": "string"},
                {"name": "customer_id", "type": "string"},
                {"name": "total", "type": "double"},
                {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"}
            ]
        }
        
        self.order_serializer = AvroSerializer(
            self.schema_registry,
            json.dumps(self.order_created_schema),
            lambda order, ctx: {
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'total': float(order['total']),
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
        )
    
    def publish_order_created(self, order: OrderResponse):
        """Publish order created event."""
        try:
            event_data = {
                'order_id': order.order_id,
                'customer_id': order.customer_id,
                'total': order.total,
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            
            self.producer.produce(
                topic='order.created',
                value=self.order_serializer(
                    event_data,
                    SerializationContext('order.created', MessageField.VALUE)
                ),
                key=order.order_id,
                callback=self._delivery_callback
            )
            
            logger.info("Published order.created event", extra={
                "order_id": order.order_id,
                "customer_id": order.customer_id
            })
            
        except Exception as e:
            logger.error("Failed to publish order.created event", extra={
                "order_id": order.order_id,
                "error": str(e)
            }, exc_info=True)
            raise
    
    def _delivery_callback(self, err, msg):
        """Handle message delivery callback."""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
```

### 5.4) Main Application

Create `services/order-service/app/main.py`:

```python
"""Order service main application."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.observability import observe_function, orders_created, active_orders
from app.schemas import OrderCreate, OrderResponse, OrderStatus
from app.events import EventPublisher
from app.models import Order, OrderItem as OrderItemModel
from app.handlers import OrderHandler

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Event publisher
event_publisher = EventPublisher(
    kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    schema_registry_url=os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
)

# Order handler
order_handler = OrderHandler(SessionLocal, event_publisher)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    yield
    # Shutdown
    event_publisher.producer.flush()


app = FastAPI(
    title="Order Service",
    description="Order processing microservice with observability",
    lifespan=lifespan
)

# Instrument FastAPI
from app.observability import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "order-service"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from app.observability import generate_latest
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/orders", response_model=OrderResponse)
@observe_function
async def create_order(order_data: OrderCreate):
    """Create a new order with validation and event publishing."""
    try:
        order = await order_handler.create_order(order_data)
        orders_created.labels(status=order.status.value).inc()
        active_orders.inc()
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/orders/{order_id}", response_model=OrderResponse)
@observe_function
async def get_order(order_id: str):
    """Get order by ID."""
    order = await order_handler.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order
```

## 6) Payment Service Implementation (Go)

Create `services/payment-service/main.go`:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/IBM/sarama"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// Observability: Metrics
var (
	paymentsProcessed = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "payments_processed_total",
			Help: "Total number of payments processed",
		},
		[]string{"status"},
	)
	paymentProcessingDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "payment_processing_duration_seconds",
			Help:    "Time spent processing payments",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"operation"},
	)
)

func init() {
	prometheus.MustRegister(paymentsProcessed)
	prometheus.MustRegister(paymentProcessingDuration)
}

// Observability: Tracing
func setupTracing() func() {
	jaegerEndpoint := fmt.Sprintf("http://%s:14268/api/traces",
		os.Getenv("JAEGER_AGENT_HOST"))
	
	exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(
		jaeger.WithEndpoint(jaegerEndpoint),
	))
	if err != nil {
		log.Fatal(err)
	}

	tp := trace.NewTracerProvider(
		trace.WithBatcher(exporter),
		trace.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String("payment-service"),
		)),
	)
	otel.SetTracerProvider(tp)

	return func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := tp.Shutdown(ctx); err != nil {
			log.Fatal(err)
		}
	}
}

// Data Contracts
type PaymentRequest struct {
	OrderID   string  `json:"order_id" binding:"required"`
	Amount    float64 `json:"amount" binding:"required,gt=0"`
	Currency  string  `json:"currency" binding:"required,len=3"`
	PaymentMethod string `json:"payment_method" binding:"required"`
}

type PaymentResponse struct {
	PaymentID string    `json:"payment_id"`
	OrderID   string    `json:"order_id"`
	Status    string    `json:"status"`
	Amount    float64   `json:"amount"`
	ProcessedAt time.Time `json:"processed_at"`
}

// Event Publishing
type PaymentEvent struct {
	PaymentID string  `json:"payment_id"`
	OrderID   string  `json:"order_id"`
	Amount    float64 `json:"amount"`
	Status    string  `json:"status"`
	Timestamp int64   `json:"timestamp"`
}

func publishPaymentEvent(producer sarama.SyncProducer, event PaymentEvent) error {
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return err
	}

	msg := &sarama.ProducerMessage{
		Topic: "payment.processed",
		Key:   sarama.StringEncoder(event.PaymentID),
		Value: sarama.ByteEncoder(eventJSON),
	}

	_, _, err = producer.SendMessage(msg)
	return err
}

func main() {
	// Setup observability
	cleanup := setupTracing()
	defer cleanup()

	// Database
	dsn := os.Getenv("DATABASE_URL")
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// Kafka producer
	config := sarama.NewConfig()
	config.Producer.Return.Successes = true
	config.Producer.RequiredAcks = sarama.WaitForAll
	producer, err := sarama.NewSyncProducer(
		[]string{os.Getenv("KAFKA_BOOTSTRAP_SERVERS")},
		config,
	)
	if err != nil {
		log.Fatal(err)
	}
	defer producer.Close()

	// Router
	r := gin.Default()

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy", "service": "payment-service"})
	})

	// Metrics
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Process payment
	r.POST("/payments", func(c *gin.Context) {
		ctx := c.Request.Context()
		start := time.Now()
		
		// Start span
		tr := otel.Tracer("payment-service")
		ctx, span := tr.Start(ctx, "process_payment")
		defer span.End()

		var req PaymentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			span.RecordError(err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Process payment (simplified)
		paymentID := fmt.Sprintf("pay_%d", time.Now().UnixNano())
		status := "completed"

		// Publish event
		event := PaymentEvent{
			PaymentID: paymentID,
			OrderID:   req.OrderID,
			Amount:    req.Amount,
			Status:    status,
			Timestamp: time.Now().Unix(),
		}
		
		if err := publishPaymentEvent(producer, event); err != nil {
			span.RecordError(err)
			log.Printf("Failed to publish payment event: %v", err)
		}

		// Metrics
		paymentsProcessed.WithLabelValues(status).Inc()
		paymentProcessingDuration.WithLabelValues("process_payment").Observe(
			time.Since(start).Seconds(),
		)

		response := PaymentResponse{
			PaymentID:   paymentID,
			OrderID:     req.OrderID,
			Status:      status,
			Amount:      req.Amount,
			ProcessedAt: time.Now(),
		}

		c.JSON(http.StatusOK, response)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	r.Run(":" + port)
}
```

## 7) API Gateway Configuration

Create `api-gateway/kong.yml`:

```yaml
_format_version: "3.0"

services:
  - name: order-service
    url: http://order-service:8000
    routes:
      - name: order-routes
        paths:
          - /api/orders
        methods:
          - GET
          - POST
        plugins:
          - name: rate-limiting
            config:
              minute: 100
              hour: 1000
          - name: prometheus
            config:
              per_consumer: true

  - name: payment-service
    url: http://payment-service:8000
    routes:
      - name: payment-routes
        paths:
          - /api/payments
        methods:
          - GET
          - POST
        plugins:
          - name: rate-limiting
            config:
              minute: 50
              hour: 500

  - name: inventory-service
    url: http://inventory-service:8000
    routes:
      - name: inventory-routes
        paths:
          - /api/inventory
        methods:
          - GET
          - POST
          - PUT
        plugins:
          - name: rate-limiting
            config:
              minute: 200
              hour: 2000

plugins:
  - name: prometheus
    config:
      per_consumer: false
```

## 8) Observability Configuration

### 8.1) Prometheus Configuration

Create `observability/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service:9090']
  
  - job_name: 'payment-service'
    static_configs:
      - targets: ['payment-service:9091']
  
  - job_name: 'inventory-service'
    static_configs:
      - targets: ['inventory-service:9092']
  
  - job_name: 'notification-service'
    static_configs:
      - targets: ['notification-service:9093']
  
  - job_name: 'kong'
    static_configs:
      - targets: ['kong:8001']
```

## 9) Testing the System

### 9.1) Start the System

```bash
# Start all services
docker compose up -d

# Wait for services to be healthy
docker compose ps

# Check logs
docker compose logs -f order-service
```

### 9.2) Create an Order

```bash
# Create order via API Gateway
curl -X POST http://localhost:8000/api/orders \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust123",
    "items": [
      {
        "product_id": "prod456",
        "quantity": 2,
        "price": 29.99
      }
    ],
    "shipping_address": "123 Main St, City, State 12345"
  }' | jq
```

### 9.3) Verify Events

```bash
# Check Kafka topics
docker exec -it kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

# Consume events
docker exec -it kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic order.created \
  --from-beginning
```

### 9.4) View Observability

```bash
# Grafana: http://localhost:3000 (admin/admin)
# Jaeger: http://localhost:16686
# Prometheus: http://localhost:9094
```

## 10) Best Practices Integration Summary

This tutorial demonstrates:

1. **Event-Driven Architecture**: Services communicate via Kafka events, maintaining loose coupling
2. **Observability-Driven Development**: Every operation is traced, logged, and metered from day one
3. **Data Validation**: Pydantic schemas and Go structs enforce contracts at API boundaries
4. **API Gateway**: Kong provides unified entry point with rate limiting and routing
5. **Service Decomposition**: Clear domain boundaries (Order, Payment, Inventory, Notification)

**Key Integration Points**:
- Events carry trace context for distributed tracing
- Validation errors are logged with observability context
- API Gateway metrics feed into Prometheus
- Service health checks integrate with orchestration

## 11) Next Steps

- Add authentication/authorization to API Gateway
- Implement event sourcing for order state
- Add circuit breakers for service resilience
- Implement CQRS pattern for read/write separation
- Add data quality validation for events

---

*This tutorial demonstrates how multiple best practices integrate to create a production-ready, observable, event-driven microservices system.*

