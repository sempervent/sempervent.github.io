# Polyglot Streaming Platform with SAGA, CQRS, and Multi-Cloud Portability: A Complete Integration Tutorial

**Objective**: Build a production-ready polyglot streaming platform that integrates streaming architecture patterns (SAGA, CQRS, Outbox), polyglot interoperability design, multi-cloud federation and portability, and API governance with backward compatibility. This tutorial demonstrates how to build scalable, portable, event-driven systems that work seamlessly across multiple cloud providers and programming languages.

This tutorial combines:
- **[Streaming Architecture Patterns: SAGA, CQRS, and Outbox](../best-practices/architecture-design/streaming-architecture-patterns.md)** - Distributed transactions and read/write separation
- **[Polyglot Interoperability Design](../best-practices/architecture-design/polyglot-interoperability-design.md)** - Cross-language integration patterns
- **[Multi-Cloud Federation & Portability Architecture](../best-practices/architecture-design/multi-cloud-federation-portability.md)** - Vendor-independent cloud architecture
- **[API Governance, Backward Compatibility Rules, and Cross-Language Interface Stability](../best-practices/architecture-design/api-governance-interface-stability.md)** - API versioning and stability

## 1) Prerequisites

```bash
# Required tools
docker --version          # >= 20.10
docker compose --version  # >= 2.0
kubectl --version         # >= 1.28
python --version          # >= 3.10
go --version              # >= 1.21
rust --version            # >= 1.70
protoc --version          # Protocol Buffers compiler
kafka --version           # Apache Kafka

# Python packages
pip install kafka-python confluent-kafka \
    grpcio grpcio-tools protobuf \
    boto3 azure-storage google-cloud-storage \
    prometheus-client opentelemetry-api

# Go packages
go get github.com/IBM/sarama
go get google.golang.org/grpc
go get github.com/aws/aws-sdk-go
go get github.com/Azure/azure-sdk-for-go

# Rust packages (Cargo.toml)
# tokio = "1.0"
# rdkafka = "0.36"
# tonic = "0.10"
# aws-sdk-s3 = "1.0"
```

**Why**: Polyglot streaming requires messaging (Kafka), serialization (Protobuf), multi-cloud SDKs, and orchestration (Kubernetes) to enable portable, language-agnostic event processing.

## 2) Architecture Overview

We'll build a **Multi-Cloud E-Commerce Order Processing Platform** with comprehensive streaming patterns:

```mermaid
graph TB
    subgraph "Multi-Cloud Infrastructure"
        AWS[AWS EKS<br/>Kafka, Services]
        GCP[GCP GKE<br/>Kafka, Services]
        AZURE[Azure AKS<br/>Kafka, Services]
        ONPREM[On-Prem RKE2<br/>Kafka, Services]
    end

    subgraph "Federation Layer"
        FED_CONTROL[Federation Control Plane]
        IDENTITY[Unified Identity]
        OBSERVABILITY[Cross-Cloud Observability]
    end

    subgraph "Streaming Layer"
        KAFKA_AWS[Kafka Cluster AWS]
        KAFKA_GCP[Kafka Cluster GCP]
        KAFKA_AZURE[Kafka Cluster Azure]
        SCHEMA_REG[Schema Registry]
    end

    subgraph "Polyglot Services"
        ORDER_PY[Order Service<br/>Python/FastAPI]
        PAYMENT_GO[Payment Service<br/>Go]
        INVENTORY_RUST[Inventory Service<br/>Rust]
        NOTIFICATION_PY[Notification Service<br/>Python]
    end

    subgraph "SAGA Orchestration"
        SAGA_ORCH[SAGA Orchestrator<br/>Python]
        COMPENSATION[Compensation Handler]
    end

    subgraph "CQRS Implementation"
        WRITE_SIDE[Write Side<br/>PostgreSQL]
        READ_SIDE[Read Side<br/>DuckDB/Parquet]
        EVENT_STORE[Event Store]
    end

    subgraph "Outbox Pattern"
        OUTBOX_DB[Outbox Table<br/>PostgreSQL]
        OUTBOX_POLLER[Outbox Poller<br/>Go]
        EVENT_PUBLISHER[Event Publisher]
    end

    subgraph "API Gateway"
        GATEWAY[Kong Gateway]
        VERSIONING[API Versioning]
        COMPATIBILITY[Compatibility Layer]
    end

    FED_CONTROL --> AWS
    FED_CONTROL --> GCP
    FED_CONTROL --> AZURE
    FED_CONTROL --> ONPREM

    IDENTITY --> ORDER_PY
    IDENTITY --> PAYMENT_GO
    IDENTITY --> INVENTORY_RUST

    ORDER_PY --> KAFKA_AWS
    PAYMENT_GO --> KAFKA_GCP
    INVENTORY_RUST --> KAFKA_AZURE

    KAFKA_AWS --> SCHEMA_REG
    KAFKA_GCP --> SCHEMA_REG
    KAFKA_AZURE --> SCHEMA_REG

    ORDER_PY --> SAGA_ORCH
    PAYMENT_GO --> SAGA_ORCH
    INVENTORY_RUST --> SAGA_ORCH
    SAGA_ORCH --> COMPENSATION

    ORDER_PY --> WRITE_SIDE
    WRITE_SIDE --> EVENT_STORE
    EVENT_STORE --> READ_SIDE

    ORDER_PY --> OUTBOX_DB
    OUTBOX_POLLER --> OUTBOX_DB
    OUTBOX_POLLER --> EVENT_PUBLISHER
    EVENT_PUBLISHER --> KAFKA_AWS

    ORDER_PY --> GATEWAY
    PAYMENT_GO --> GATEWAY
    INVENTORY_RUST --> GATEWAY
    GATEWAY --> VERSIONING
    GATEWAY --> COMPATIBILITY
```

**Architecture Principles**:
1. **Multi-Cloud Portability**: Services run on any cloud provider
2. **Polyglot Interoperability**: Python, Go, Rust services communicate seamlessly
3. **SAGA Transactions**: Distributed transactions with compensation
4. **CQRS Separation**: Read/write separation for scalability
5. **Outbox Reliability**: Guaranteed event publishing
6. **API Versioning**: Backward-compatible API evolution

## 3) Repository Layout

```
polyglot-streaming-platform/
├── docker-compose.yaml
├── k8s/
│   ├── aws/
│   │   ├── kafka-cluster.yaml
│   │   └── services.yaml
│   ├── gcp/
│   │   ├── kafka-cluster.yaml
│   │   └── services.yaml
│   └── azure/
│       ├── kafka-cluster.yaml
│       └── services.yaml
├── services/
│   ├── order-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── saga.py
│   │   │   ├── cqrs.py
│   │   │   ├── outbox.py
│   │   │   ├── cloud_adapter.py
│   │   │   └── api.py
│   │   └── proto/
│   │       └── order.proto
│   ├── payment-service/
│   │   ├── Dockerfile
│   │   ├── go.mod
│   │   ├── main.go
│   │   ├── saga.go
│   │   ├── outbox.go
│   │   ├── cloud_adapter.go
│   │   └── proto/
│   │       └── payment.proto
│   └── inventory-service/
│       ├── Dockerfile
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs
│       │   ├── saga.rs
│       │   ├── cqrs.rs
│       │   └── cloud_adapter.rs
│       └── proto/
│           └── inventory.proto
├── federation/
│   ├── control_plane.py
│   ├── identity_federation.py
│   └── observability_federation.py
├── schemas/
│   ├── order-events.avsc
│   ├── payment-events.avsc
│   └── inventory-events.avsc
└── scripts/
    ├── setup-multicloud.sh
    └── deploy-federation.sh
```

## 4) Protocol Buffer Schemas for Polyglot Interoperability

### 4.1) Order Service Schema

Create `services/order-service/proto/order.proto`:

```protobuf
syntax = "proto3";

package order.v1;

option go_package = "github.com/company/order-service/proto/order/v1";
option python_package = "order.proto.v1";
option rust_package = "order::proto::v1";

// Order service API
service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (CreateOrderResponse);
  rpc GetOrder(GetOrderRequest) returns (GetOrderResponse);
  rpc CancelOrder(CancelOrderRequest) returns (CancelOrderResponse);
  
  // Streaming endpoint for real-time updates
  rpc StreamOrderUpdates(StreamOrderUpdatesRequest) returns (stream OrderUpdate);
}

// Order messages
message CreateOrderRequest {
  string customer_id = 1;
  repeated OrderItem items = 2;
  Address shipping_address = 3;
  PaymentMethod payment_method = 4;
}

message CreateOrderResponse {
  string order_id = 1;
  OrderStatus status = 2;
  int64 created_at = 3;
  string saga_id = 4;  // SAGA transaction ID
}

message GetOrderRequest {
  string order_id = 1;
}

message GetOrderResponse {
  Order order = 1;
}

message CancelOrderRequest {
  string order_id = 1;
  string reason = 2;
}

message CancelOrderResponse {
  bool success = 1;
  string compensation_id = 2;  // SAGA compensation ID
}

// Order domain model
message Order {
  string order_id = 1;
  string customer_id = 2;
  OrderStatus status = 3;
  repeated OrderItem items = 4;
  Money total_amount = 5;
  Address shipping_address = 6;
  int64 created_at = 7;
  int64 updated_at = 8;
  string saga_id = 9;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
  Money price = 3;
  Money subtotal = 4;
}

message Address {
  string street = 1;
  string city = 2;
  string state = 3;
  string zip_code = 4;
  string country = 5;
}

message Money {
  string currency = 1;
  int64 amount_cents = 2;  // Amount in cents for precision
}

enum OrderStatus {
  ORDER_STATUS_UNSPECIFIED = 0;
  ORDER_STATUS_PENDING = 1;
  ORDER_STATUS_CONFIRMED = 2;
  ORDER_STATUS_PROCESSING = 3;
  ORDER_STATUS_SHIPPED = 4;
  ORDER_STATUS_DELIVERED = 5;
  ORDER_STATUS_CANCELLED = 6;
}

enum PaymentMethod {
  PAYMENT_METHOD_UNSPECIFIED = 0;
  PAYMENT_METHOD_CREDIT_CARD = 1;
  PAYMENT_METHOD_PAYPAL = 2;
  PAYMENT_METHOD_BANK_TRANSFER = 3;
}

// Streaming messages
message StreamOrderUpdatesRequest {
  string order_id = 1;
}

message OrderUpdate {
  string order_id = 1;
  OrderStatus status = 2;
  int64 timestamp = 3;
  string event_type = 4;
  map<string, string> metadata = 5;
}

// SAGA messages
message SagaStep {
  string step_id = 1;
  string service_name = 2;
  string action = 3;
  bytes payload = 4;  // Service-specific payload
  SagaStepStatus status = 5;
}

enum SagaStepStatus {
  SAGA_STEP_STATUS_UNSPECIFIED = 0;
  SAGA_STEP_STATUS_PENDING = 1;
  SAGA_STEP_STATUS_IN_PROGRESS = 2;
  SAGA_STEP_STATUS_COMPLETED = 3;
  SAGA_STEP_STATUS_FAILED = 4;
  SAGA_STEP_STATUS_COMPENSATED = 5;
}

message SagaState {
  string saga_id = 1;
  string order_id = 2;
  repeated SagaStep steps = 3;
  SagaStatus status = 4;
  int64 started_at = 5;
  int64 completed_at = 6;
}

enum SagaStatus {
  SAGA_STATUS_UNSPECIFIED = 0;
  SAGA_STATUS_IN_PROGRESS = 1;
  SAGA_STATUS_COMPLETED = 2;
  SAGA_STATUS_FAILED = 3;
  SAGA_STATUS_COMPENSATING = 4;
  SAGA_STATUS_COMPENSATED = 5;
}
```

### 4.2) Payment Service Schema

Create `services/payment-service/proto/payment.proto`:

```protobuf
syntax = "proto3";

package payment.v1;

option go_package = "github.com/company/payment-service/proto/payment/v1";

service PaymentService {
  rpc ProcessPayment(ProcessPaymentRequest) returns (ProcessPaymentResponse);
  rpc RefundPayment(RefundPaymentRequest) returns (RefundPaymentResponse);
  rpc GetPaymentStatus(GetPaymentStatusRequest) returns (GetPaymentStatusResponse);
}

message ProcessPaymentRequest {
  string order_id = 1;
  Money amount = 2;
  PaymentMethod method = 3;
  PaymentDetails details = 4;
  string saga_id = 5;  // SAGA transaction ID
}

message ProcessPaymentResponse {
  string payment_id = 1;
  PaymentStatus status = 2;
  int64 processed_at = 3;
  string transaction_id = 4;
}

message RefundPaymentRequest {
  string payment_id = 1;
  Money amount = 2;
  string reason = 3;
  string saga_id = 4;  // SAGA compensation ID
}

message RefundPaymentResponse {
  string refund_id = 1;
  bool success = 2;
  int64 processed_at = 3;
}

message GetPaymentStatusRequest {
  string payment_id = 1;
}

message GetPaymentStatusResponse {
  Payment payment = 1;
}

message Payment {
  string payment_id = 1;
  string order_id = 2;
  Money amount = 3;
  PaymentStatus status = 4;
  PaymentMethod method = 5;
  int64 created_at = 6;
  int64 processed_at = 7;
  string transaction_id = 8;
}

message PaymentDetails {
  oneof details {
    CreditCardDetails credit_card = 1;
    PayPalDetails paypal = 2;
    BankTransferDetails bank_transfer = 3;
  }
}

message CreditCardDetails {
  string card_number = 1;  // Encrypted
  string expiry_month = 2;
  string expiry_year = 3;
  string cvv = 4;  // Encrypted
  string cardholder_name = 5;
}

message PayPalDetails {
  string email = 1;
  string payer_id = 2;
}

message BankTransferDetails {
  string account_number = 1;  // Encrypted
  string routing_number = 2;
  string account_name = 3;
}

enum PaymentStatus {
  PAYMENT_STATUS_UNSPECIFIED = 0;
  PAYMENT_STATUS_PENDING = 1;
  PAYMENT_STATUS_PROCESSING = 2;
  PAYMENT_STATUS_COMPLETED = 3;
  PAYMENT_STATUS_FAILED = 4;
  PAYMENT_STATUS_REFUNDED = 5;
}

enum PaymentMethod {
  PAYMENT_METHOD_UNSPECIFIED = 0;
  PAYMENT_METHOD_CREDIT_CARD = 1;
  PAYMENT_METHOD_PAYPAL = 2;
  PAYMENT_METHOD_BANK_TRANSFER = 3;
}

message Money {
  string currency = 1;
  int64 amount_cents = 2;
}
```

## 5) SAGA Orchestration (Python)

Create `services/order-service/app/saga.py`:

```python
"""SAGA pattern implementation for distributed transactions."""
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

import grpc
from prometheus_client import Counter, Histogram, Gauge

from proto.order_pb2 import (
    SagaState, SagaStep, SagaStatus, SagaStepStatus,
    CreateOrderRequest, ProcessPaymentRequest
)
from proto.order_pb2_grpc import OrderServiceStub
from proto.payment_pb2_grpc import PaymentServiceStub
from proto.inventory_pb2_grpc import InventoryServiceStub

saga_metrics = {
    "saga_started": Counter("saga_started_total", "SAGAs started", ["saga_type"]),
    "saga_completed": Counter("saga_completed_total", "SAGAs completed", ["saga_type", "status"]),
    "saga_duration": Histogram("saga_duration_seconds", "SAGA duration", ["saga_type"]),
    "saga_compensations": Counter("saga_compensations_total", "SAGA compensations", ["saga_type"]),
    "active_sagas": Gauge("active_sagas", "Active SAGAs", ["saga_type"]),
}


class SagaStepResult(Enum):
    """SAGA step execution result."""
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"


@dataclass
class SagaStepDefinition:
    """SAGA step definition."""
    step_id: str
    service_name: str
    action: str
    compensation_action: str
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaExecutionResult:
    """SAGA execution result."""
    saga_id: str
    status: SagaStatus
    completed_steps: List[str]
    failed_step: Optional[str] = None
    compensation_applied: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None


class SagaOrchestrator:
    """Orchestrates SAGA transactions across polyglot services."""
    
    def __init__(
        self,
        order_service_stub: OrderServiceStub,
        payment_service_stub: PaymentServiceStub,
        inventory_service_stub: InventoryServiceStub,
        state_store: Optional[Any] = None  # For persistence
    ):
        self.order_stub = order_service_stub
        self.payment_stub = payment_service_stub
        self.inventory_stub = inventory_service_stub
        self.state_store = state_store
        self.active_sagas: Dict[str, SagaState] = {}
    
    async def execute_saga(
        self,
        saga_type: str,
        steps: List[SagaStepDefinition],
        initial_payload: Dict[str, Any]
    ) -> SagaExecutionResult:
        """Execute a SAGA transaction."""
        saga_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        saga_metrics["saga_started"].labels(saga_type=saga_type).inc()
        saga_metrics["active_sagas"].labels(saga_type=saga_type).inc()
        
        # Initialize SAGA state
        saga_state = SagaState(
            saga_id=saga_id,
            order_id=initial_payload.get("order_id", ""),
            status=SagaStatus.SAGA_STATUS_IN_PROGRESS,
            started_at=int(start_time.timestamp() * 1000)
        )
        
        self.active_sagas[saga_id] = saga_state
        
        completed_steps = []
        failed_step = None
        
        try:
            # Execute steps sequentially
            for step_def in steps:
                step = SagaStep(
                    step_id=step_def.step_id,
                    service_name=step_def.service_name,
                    action=step_def.action,
                    status=SagaStepStatus.SAGA_STEP_STATUS_PENDING,
                    payload=json.dumps(step_def.payload).encode()
                )
                
                saga_state.steps.append(step)
                
                # Execute step with retries
                result = await self._execute_step_with_retry(
                    step_def,
                    initial_payload,
                    saga_id
                )
                
                if result == SagaStepResult.SUCCESS:
                    step.status = SagaStepStatus.SAGA_STEP_STATUS_COMPLETED
                    completed_steps.append(step_def.step_id)
                elif result == SagaStepResult.RETRY:
                    # Retry logic handled in _execute_step_with_retry
                    step.status = SagaStepStatus.SAGA_STEP_STATUS_FAILED
                    failed_step = step_def.step_id
                    break
                else:  # FAILURE
                    step.status = SagaStepStatus.SAGA_STEP_STATUS_FAILED
                    failed_step = step_def.step_id
                    break
            
            # Check if all steps completed
            if failed_step is None:
                saga_state.status = SagaStatus.SAGA_STATUS_COMPLETED
                saga_state.completed_at = int(datetime.utcnow().timestamp() * 1000)
                
                saga_metrics["saga_completed"].labels(
                    saga_type=saga_type,
                    status="success"
                ).inc()
            else:
                # Compensate for completed steps
                await self._compensate_saga(saga_state, saga_type)
                saga_state.status = SagaStatus.SAGA_STATUS_COMPENSATED
                
                saga_metrics["saga_compensations"].labels(saga_type=saga_type).inc()
                saga_metrics["saga_completed"].labels(
                    saga_type=saga_type,
                    status="compensated"
                ).inc()
        
        except Exception as e:
            logging.error(f"SAGA {saga_id} failed: {e}", exc_info=True)
            saga_state.status = SagaStatus.SAGA_STATUS_FAILED
            await self._compensate_saga(saga_state, saga_type)
            
            saga_metrics["saga_completed"].labels(
                saga_type=saga_type,
                status="failed"
            ).inc()
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            saga_metrics["saga_duration"].labels(saga_type=saga_type).observe(duration)
            saga_metrics["active_sagas"].labels(saga_type=saga_type).dec()
            
            # Persist state
            if self.state_store:
                await self._persist_saga_state(saga_state)
        
        return SagaExecutionResult(
            saga_id=saga_id,
            status=saga_state.status,
            completed_steps=completed_steps,
            failed_step=failed_step,
            compensation_applied=saga_state.status == SagaStatus.SAGA_STATUS_COMPENSATED,
            duration_seconds=duration
        )
    
    async def _execute_step_with_retry(
        self,
        step_def: SagaStepDefinition,
        payload: Dict[str, Any],
        saga_id: str
    ) -> SagaStepResult:
        """Execute a SAGA step with retry logic."""
        for attempt in range(step_def.retry_count):
            try:
                # Route to appropriate service based on service_name
                if step_def.service_name == "order-service":
                    result = await self._execute_order_step(step_def, payload, saga_id)
                elif step_def.service_name == "payment-service":
                    result = await self._execute_payment_step(step_def, payload, saga_id)
                elif step_def.service_name == "inventory-service":
                    result = await self._execute_inventory_step(step_def, payload, saga_id)
                else:
                    raise ValueError(f"Unknown service: {step_def.service_name}")
                
                if result:
                    return SagaStepResult.SUCCESS
                else:
                    if attempt < step_def.retry_count - 1:
                        await asyncio.sleep(step_def.retry_delay_seconds * (attempt + 1))
                        continue
                    return SagaStepResult.FAILURE
            
            except asyncio.TimeoutError:
                if attempt < step_def.retry_count - 1:
                    await asyncio.sleep(step_def.retry_delay_seconds * (attempt + 1))
                    continue
                return SagaStepResult.RETRY
            
            except Exception as e:
                logging.error(f"Step {step_def.step_id} failed: {e}", exc_info=True)
                if attempt < step_def.retry_count - 1:
                    await asyncio.sleep(step_def.retry_delay_seconds * (attempt + 1))
                    continue
                return SagaStepResult.FAILURE
        
        return SagaStepResult.FAILURE
    
    async def _execute_order_step(
        self,
        step_def: SagaStepDefinition,
        payload: Dict[str, Any],
        saga_id: str
    ) -> bool:
        """Execute order service step."""
        if step_def.action == "create_order":
            request = CreateOrderRequest(
                customer_id=payload["customer_id"],
                shipping_address=payload["shipping_address"],
                payment_method=payload["payment_method"]
            )
            # Add items from payload
            for item in payload.get("items", []):
                order_item = request.items.add()
                order_item.product_id = item["product_id"]
                order_item.quantity = item["quantity"]
                # Set price from item
            
            response = await asyncio.to_thread(
                self.order_stub.CreateOrder,
                request
            )
            return response.order_id != ""
        
        return False
    
    async def _execute_payment_step(
        self,
        step_def: SagaStepDefinition,
        payload: Dict[str, Any],
        saga_id: str
    ) -> bool:
        """Execute payment service step."""
        if step_def.action == "process_payment":
            from proto.payment_pb2 import ProcessPaymentRequest, Money
            
            request = ProcessPaymentRequest(
                order_id=payload["order_id"],
                amount=Money(
                    currency=payload["amount"]["currency"],
                    amount_cents=payload["amount"]["amount_cents"]
                ),
                payment_method=payload["payment_method"],
                saga_id=saga_id
            )
            
            response = await asyncio.to_thread(
                self.payment_stub.ProcessPayment,
                request
            )
            return response.status == 1  # COMPLETED
        
        elif step_def.action == "refund_payment":
            from proto.payment_pb2 import RefundPaymentRequest
            
            request = RefundPaymentRequest(
                payment_id=payload["payment_id"],
                amount=payload.get("amount"),
                reason=payload.get("reason", "SAGA compensation"),
                saga_id=saga_id
            )
            
            response = await asyncio.to_thread(
                self.payment_stub.RefundPayment,
                request
            )
            return response.success
        
        return False
    
    async def _execute_inventory_step(
        self,
        step_def: SagaStepDefinition,
        payload: Dict[str, Any],
        saga_id: str
    ) -> bool:
        """Execute inventory service step."""
        # Similar pattern for inventory service
        # In production, implement actual gRPC calls
        return True
    
    async def _compensate_saga(
        self,
        saga_state: SagaState,
        saga_type: str
    ):
        """Compensate a failed SAGA by executing compensation actions in reverse order."""
        saga_state.status = SagaStatus.SAGA_STATUS_COMPENSATING
        
        # Execute compensations in reverse order
        for step in reversed(saga_state.steps):
            if step.status == SagaStepStatus.SAGA_STEP_STATUS_COMPLETED:
                # Execute compensation action
                try:
                    await self._execute_compensation(step, saga_state.saga_id)
                    step.status = SagaStepStatus.SAGA_STEP_STATUS_COMPENSATED
                except Exception as e:
                    logging.error(
                        f"Compensation failed for step {step.step_id}: {e}",
                        exc_info=True
                    )
                    # Log but continue with other compensations
    
    async def _execute_compensation(
        self,
        step: SagaStep,
        saga_id: str
    ):
        """Execute compensation for a step."""
        # Route compensation to appropriate service
        if step.service_name == "payment-service":
            payload = json.loads(step.payload.decode())
            await self._execute_payment_step(
                SagaStepDefinition(
                    step_id=f"{step.step_id}_compensation",
                    service_name=step.service_name,
                    action="refund_payment",
                    compensation_action="",
                    payload=payload
                ),
                payload,
                saga_id
            )
        # Add other service compensations
    
    async def _persist_saga_state(self, saga_state: SagaState):
        """Persist SAGA state for recovery."""
        # In production, persist to database or event store
        pass


class OrderSagaOrchestrator:
    """Specialized orchestrator for order processing SAGAs."""
    
    def __init__(self, base_orchestrator: SagaOrchestrator):
        self.orchestrator = base_orchestrator
    
    async def create_order_saga(
        self,
        customer_id: str,
        items: List[Dict],
        shipping_address: Dict,
        payment_method: str
    ) -> SagaExecutionResult:
        """Execute order creation SAGA."""
        steps = [
            SagaStepDefinition(
                step_id="create_order",
                service_name="order-service",
                action="create_order",
                compensation_action="cancel_order",
                payload={
                    "customer_id": customer_id,
                    "items": items,
                    "shipping_address": shipping_address
                }
            ),
            SagaStepDefinition(
                step_id="reserve_inventory",
                service_name="inventory-service",
                action="reserve_items",
                compensation_action="release_reservation",
                payload={"items": items}
            ),
            SagaStepDefinition(
                step_id="process_payment",
                service_name="payment-service",
                action="process_payment",
                compensation_action="refund_payment",
                payload={
                    "order_id": "",  # Will be set by first step
                    "amount": self._calculate_total(items),
                    "payment_method": payment_method
                }
            ),
        ]
        
        initial_payload = {
            "customer_id": customer_id,
            "items": items,
            "shipping_address": shipping_address,
            "payment_method": payment_method
        }
        
        return await self.orchestrator.execute_saga(
            saga_type="create_order",
            steps=steps,
            initial_payload=initial_payload
        )
    
    def _calculate_total(self, items: List[Dict]) -> Dict:
        """Calculate order total."""
        total_cents = sum(
            item["quantity"] * item["price_cents"]
            for item in items
        )
        return {
            "currency": "USD",
            "amount_cents": total_cents
        }
```

## 6) CQRS Implementation (Python)

Create `services/order-service/app/cqrs.py`:

```python
"""CQRS pattern implementation for read/write separation."""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import asyncpg
import duckdb
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge

from proto.order_pb2 import Order, OrderStatus

cqrs_metrics = {
    "commands_executed": Counter("cqrs_commands_total", "Commands executed", ["command_type"]),
    "queries_executed": Counter("cqrs_queries_total", "Queries executed", ["query_type"]),
    "event_projections": Counter("cqrs_projections_total", "Event projections", ["projection_type"]),
    "read_model_lag": Gauge("cqrs_read_model_lag_seconds", "Read model lag", ["model"]),
    "write_latency": Histogram("cqrs_write_latency_seconds", "Write latency", ["command_type"]),
    "read_latency": Histogram("cqrs_read_latency_seconds", "Read latency", ["query_type"]),
}


@dataclass
class Command:
    """Command for write side."""
    command_id: str
    command_type: str
    payload: Dict[str, Any]
    timestamp: datetime


@dataclass
class Query:
    """Query for read side."""
    query_id: str
    query_type: str
    filters: Dict[str, Any]
    timestamp: datetime


class WriteSide:
    """CQRS write side - handles commands and generates events."""
    
    def __init__(self, database_url: str, event_store):
        self.engine = create_async_engine(database_url)
        self.Session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.event_store = event_store
    
    async def execute_command(self, command: Command) -> Dict[str, Any]:
        """Execute a command on the write side."""
        start_time = datetime.utcnow()
        
        try:
            async with self.Session() as session:
                if command.command_type == "create_order":
                    result = await self._create_order_command(session, command)
                elif command.command_type == "update_order_status":
                    result = await self._update_order_status_command(session, command)
                elif command.command_type == "cancel_order":
                    result = await self._cancel_order_command(session, command)
                else:
                    raise ValueError(f"Unknown command type: {command.command_type}")
                
                await session.commit()
                
                # Generate and store event
                event = self._generate_event(command, result)
                await self.event_store.append_event(event)
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                cqrs_metrics["write_latency"].labels(
                    command_type=command.command_type
                ).observe(duration)
                cqrs_metrics["commands_executed"].labels(
                    command_type=command.command_type
                ).inc()
                
                return result
        
        except Exception as e:
            logging.error(f"Command execution failed: {e}", exc_info=True)
            raise
    
    async def _create_order_command(
        self,
        session: AsyncSession,
        command: Command
    ) -> Dict[str, Any]:
        """Handle create order command."""
        # Insert into write database
        result = await session.execute(
            """
            INSERT INTO orders (order_id, customer_id, status, total_amount, created_at)
            VALUES (:order_id, :customer_id, :status, :total_amount, :created_at)
            RETURNING order_id
            """,
            {
                "order_id": command.payload["order_id"],
                "customer_id": command.payload["customer_id"],
                "status": "pending",
                "total_amount": command.payload["total_amount"],
                "created_at": datetime.utcnow()
            }
        )
        
        order_id = result.scalar()
        
        # Insert order items
        for item in command.payload.get("items", []):
            await session.execute(
                """
                INSERT INTO order_items (order_id, product_id, quantity, price)
                VALUES (:order_id, :product_id, :quantity, :price)
                """,
                {
                    "order_id": order_id,
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "price": item["price"]
                }
            )
        
        return {"order_id": order_id, "status": "pending"}
    
    async def _update_order_status_command(
        self,
        session: AsyncSession,
        command: Command
    ) -> Dict[str, Any]:
        """Handle update order status command."""
        await session.execute(
            """
            UPDATE orders
            SET status = :status, updated_at = :updated_at
            WHERE order_id = :order_id
            """,
            {
                "order_id": command.payload["order_id"],
                "status": command.payload["status"],
                "updated_at": datetime.utcnow()
            }
        )
        
        return {"order_id": command.payload["order_id"], "status": command.payload["status"]}
    
    async def _cancel_order_command(
        self,
        session: AsyncSession,
        command: Command
    ) -> Dict[str, Any]:
        """Handle cancel order command."""
        await session.execute(
            """
            UPDATE orders
            SET status = 'cancelled', cancelled_at = :cancelled_at, cancellation_reason = :reason
            WHERE order_id = :order_id
            """,
            {
                "order_id": command.payload["order_id"],
                "cancelled_at": datetime.utcnow(),
                "reason": command.payload.get("reason", "")
            }
        )
        
        return {"order_id": command.payload["order_id"], "status": "cancelled"}
    
    def _generate_event(self, command: Command, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate event from command result."""
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": f"{command.command_type}_completed",
            "aggregate_id": result.get("order_id"),
            "aggregate_type": "order",
            "payload": result,
            "timestamp": datetime.utcnow().isoformat(),
            "command_id": command.command_id
        }


class ReadSide:
    """CQRS read side - optimized for queries."""
    
    def __init__(self, parquet_path: str, duckdb_path: str = ":memory:"):
        self.parquet_path = parquet_path
        self.conn = duckdb.connect(duckdb_path)
        self._initialize_read_model()
    
    def _initialize_read_model(self):
        """Initialize read model from Parquet files."""
        # Create views from Parquet files
        self.conn.execute("""
            CREATE VIEW orders_read AS
            SELECT * FROM read_parquet('orders.parquet')
        """)
        
        self.conn.execute("""
            CREATE VIEW order_items_read AS
            SELECT * FROM read_parquet('order_items.parquet')
        """)
    
    async def execute_query(self, query: Query) -> List[Dict[str, Any]]:
        """Execute a query on the read side."""
        start_time = datetime.utcnow()
        
        try:
            if query.query_type == "get_order":
                result = await self._get_order_query(query)
            elif query.query_type == "list_orders":
                result = await self._list_orders_query(query)
            elif query.query_type == "get_order_history":
                result = await self._get_order_history_query(query)
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            cqrs_metrics["read_latency"].labels(
                query_type=query.query_type
            ).observe(duration)
            cqrs_metrics["queries_executed"].labels(
                query_type=query.query_type
            ).inc()
            
            return result
        
        except Exception as e:
            logging.error(f"Query execution failed: {e}", exc_info=True)
            raise
    
    async def _get_order_query(self, query: Query) -> List[Dict[str, Any]]:
        """Get order by ID."""
        order_id = query.filters.get("order_id")
        
        result = self.conn.execute("""
            SELECT 
                o.order_id,
                o.customer_id,
                o.status,
                o.total_amount,
                o.created_at,
                o.updated_at,
                json_agg(
                    json_object(
                        'product_id', oi.product_id,
                        'quantity', oi.quantity,
                        'price', oi.price
                    )
                ) as items
            FROM orders_read o
            LEFT JOIN order_items_read oi ON o.order_id = oi.order_id
            WHERE o.order_id = ?
            GROUP BY o.order_id, o.customer_id, o.status, o.total_amount, o.created_at, o.updated_at
        """, [order_id]).fetchall()
        
        return [dict(row) for row in result]
    
    async def _list_orders_query(self, query: Query) -> List[Dict[str, Any]]:
        """List orders with filters."""
        filters = query.filters
        where_clauses = []
        params = []
        
        if filters.get("customer_id"):
            where_clauses.append("customer_id = ?")
            params.append(filters["customer_id"])
        
        if filters.get("status"):
            where_clauses.append("status = ?")
            params.append(filters["status"])
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        result = self.conn.execute(f"""
            SELECT * FROM orders_read
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
        """, params + [filters.get("limit", 100)]).fetchall()
        
        return [dict(row) for row in result]
    
    async def _get_order_history_query(self, query: Query) -> List[Dict[str, Any]]:
        """Get order history with events."""
        order_id = query.filters.get("order_id")
        
        # Query from event store or read model
        result = self.conn.execute("""
            SELECT 
                event_type,
                payload,
                timestamp
            FROM order_events_read
            WHERE order_id = ?
            ORDER BY timestamp ASC
        """, [order_id]).fetchall()
        
        return [dict(row) for row in result]
    
    async def project_event(self, event: Dict[str, Any]):
        """Project event into read model."""
        event_type = event["event_type"]
        
        if event_type == "create_order_completed":
            await self._project_order_created(event)
        elif event_type == "update_order_status_completed":
            await self._project_order_status_updated(event)
        elif event_type == "cancel_order_completed":
            await self._project_order_cancelled(event)
        
        cqrs_metrics["event_projections"].labels(
            projection_type=event_type
        ).inc()
    
    async def _project_order_created(self, event: Dict[str, Any]):
        """Project order created event."""
        # In production, update Parquet files or materialized views
        # For demo, update DuckDB in-memory view
        pass
    
    async def _project_order_status_updated(self, event: Dict[str, Any]):
        """Project order status updated event."""
        pass
    
    async def _project_order_cancelled(self, event: Dict[str, Any]):
        """Project order cancelled event."""
        pass


class EventStore:
    """Event store for CQRS event sourcing."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url)
        self.Session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def append_event(self, event: Dict[str, Any]):
        """Append event to event store."""
        async with self.Session() as session:
            await session.execute(
                """
                INSERT INTO events (
                    event_id, event_type, aggregate_id, aggregate_type,
                    payload, timestamp, command_id
                ) VALUES (
                    :event_id, :event_type, :aggregate_id, :aggregate_type,
                    :payload, :timestamp, :command_id
                )
                """,
                {
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "aggregate_id": event["aggregate_id"],
                    "aggregate_type": event["aggregate_type"],
                    "payload": json.dumps(event["payload"]),
                    "timestamp": datetime.fromisoformat(event["timestamp"]),
                    "command_id": event.get("command_id")
                }
            )
            await session.commit()
    
    async def get_events(
        self,
        aggregate_id: str,
        aggregate_type: str,
        from_version: int = 0
    ) -> List[Dict[str, Any]]:
        """Get events for an aggregate."""
        async with self.Session() as session:
            result = await session.execute(
                """
                SELECT event_id, event_type, payload, timestamp, version
                FROM events
                WHERE aggregate_id = :aggregate_id
                  AND aggregate_type = :aggregate_type
                  AND version > :from_version
                ORDER BY version ASC
                """,
                {
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "from_version": from_version
                }
            )
            
            events = []
            for row in result:
                events.append({
                    "event_id": row.event_id,
                    "event_type": row.event_type,
                    "payload": json.loads(row.payload),
                    "timestamp": row.timestamp.isoformat(),
                    "version": row.version
                })
            
            return events
```

## 7) Outbox Pattern (Python)

Create `services/order-service/app/outbox.py`:

```python
"""Outbox pattern for reliable event publishing."""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import asyncpg
from confluent_kafka import Producer
from prometheus_client import Counter, Gauge, Histogram

outbox_metrics = {
    "outbox_events_created": Counter("outbox_events_created_total", "Outbox events created"),
    "outbox_events_published": Counter("outbox_events_published_total", "Outbox events published", ["status"]),
    "outbox_events_failed": Counter("outbox_events_failed_total", "Outbox events failed"),
    "outbox_lag": Gauge("outbox_lag_events", "Outbox lag (unpublished events)"),
    "outbox_publish_duration": Histogram("outbox_publish_duration_seconds", "Outbox publish duration"),
}


@dataclass
class OutboxEvent:
    """Outbox event record."""
    event_id: str
    aggregate_id: str
    aggregate_type: str
    event_type: str
    payload: Dict[str, Any]
    created_at: datetime
    published_at: Optional[datetime] = None
    retry_count: int = 0
    status: str = "pending"  # pending, published, failed


class OutboxManager:
    """Manages outbox table and event publishing."""
    
    def __init__(
        self,
        database_url: str,
        kafka_producer: Producer,
        kafka_topic: str,
        poll_interval_seconds: float = 1.0
    ):
        self.database_url = database_url
        self.producer = kafka_producer
        self.topic = kafka_topic
        self.poll_interval = poll_interval_seconds
        self.running = False
    
    async def create_outbox_event(
        self,
        aggregate_id: str,
        aggregate_type: str,
        event_type: str,
        payload: Dict[str, Any],
        transaction: Optional[Any] = None
    ) -> str:
        """Create outbox event within a transaction."""
        event_id = str(uuid.uuid4())
        
        if transaction:
            # Use existing transaction
            await transaction.execute(
                """
                INSERT INTO outbox (
                    event_id, aggregate_id, aggregate_type, event_type,
                    payload, created_at, status
                ) VALUES (
                    :event_id, :aggregate_id, :aggregate_type, :event_type,
                    :payload, :created_at, 'pending'
                )
                """,
                {
                    "event_id": event_id,
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "event_type": event_type,
                    "payload": json.dumps(payload),
                    "created_at": datetime.utcnow()
                }
            )
        else:
            # Create new connection
            conn = await asyncpg.connect(self.database_url)
            try:
                await conn.execute(
                    """
                    INSERT INTO outbox (
                        event_id, aggregate_id, aggregate_type, event_type,
                        payload, created_at, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'pending')
                    """,
                    event_id,
                    aggregate_id,
                    aggregate_type,
                    event_type,
                    json.dumps(payload),
                    datetime.utcnow()
                )
            finally:
                await conn.close()
        
        outbox_metrics["outbox_events_created"].inc()
        return event_id
    
    async def start_poller(self):
        """Start outbox event poller."""
        self.running = True
        
        while self.running:
            try:
                await self._poll_and_publish()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logging.error(f"Outbox poller error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval * 2)  # Back off on error
    
    async def stop_poller(self):
        """Stop outbox event poller."""
        self.running = False
    
    async def _poll_and_publish(self):
        """Poll outbox table and publish events."""
        conn = await asyncpg.connect(self.database_url)
        
        try:
            # Fetch unpublished events
            rows = await conn.fetch("""
                SELECT 
                    event_id, aggregate_id, aggregate_type, event_type,
                    payload, created_at, retry_count
                FROM outbox
                WHERE status = 'pending'
                  AND retry_count < 5
                ORDER BY created_at ASC
                LIMIT 100
                FOR UPDATE SKIP LOCKED
            """)
            
            if not rows:
                # Update lag metric
                lag_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM outbox WHERE status = 'pending'
                """)
                outbox_metrics["outbox_lag"].set(lag_count)
                return
            
            # Publish each event
            for row in rows:
                try:
                    await self._publish_event(row)
                    
                    # Mark as published
                    await conn.execute("""
                        UPDATE outbox
                        SET status = 'published',
                            published_at = $1
                        WHERE event_id = $2
                    """, datetime.utcnow(), row['event_id'])
                    
                    outbox_metrics["outbox_events_published"].labels(status="success").inc()
                
                except Exception as e:
                    logging.error(f"Failed to publish event {row['event_id']}: {e}")
                    
                    # Increment retry count
                    await conn.execute("""
                        UPDATE outbox
                        SET retry_count = retry_count + 1,
                            status = CASE 
                                WHEN retry_count + 1 >= 5 THEN 'failed'
                                ELSE 'pending'
                            END
                        WHERE event_id = $1
                    """, row['event_id'])
                    
                    outbox_metrics["outbox_events_published"].labels(status="failed").inc()
                    outbox_metrics["outbox_events_failed"].inc()
        
        finally:
            await conn.close()
    
    async def _publish_event(self, row: Dict[str, Any]):
        """Publish event to Kafka."""
        start_time = datetime.utcnow()
        
        event_data = {
            "event_id": row['event_id'],
            "aggregate_id": row['aggregate_id'],
            "aggregate_type": row['aggregate_type'],
            "event_type": row['event_type'],
            "payload": json.loads(row['payload']),
            "timestamp": row['created_at'].isoformat()
        }
        
        # Publish to Kafka
        future = self.producer.produce(
            topic=self.topic,
            key=row['aggregate_id'],
            value=json.dumps(event_data).encode('utf-8'),
            callback=self._delivery_callback
        )
        
        # Wait for delivery
        self.producer.poll(1)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        outbox_metrics["outbox_publish_duration"].observe(duration)
    
    def _delivery_callback(self, err, msg):
        """Kafka delivery callback."""
        if err:
            logging.error(f"Message delivery failed: {err}")
        else:
            logging.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")


class TransactionalOutbox:
    """Transactional outbox pattern implementation."""
    
    def __init__(self, database_url: str, outbox_manager: OutboxManager):
        self.database_url = database_url
        self.outbox_manager = outbox_manager
    
    async def execute_with_outbox(
        self,
        operation: Callable,
        aggregate_id: str,
        aggregate_type: str,
        event_type: str,
        event_payload: Dict[str, Any]
    ) -> Any:
        """Execute operation with outbox event in same transaction."""
        conn = await asyncpg.connect(self.database_url)
        tr = conn.transaction()
        
        try:
            await tr.start()
            
            # Execute business operation
            result = await operation(conn, tr)
            
            # Create outbox event in same transaction
            await self.outbox_manager.create_outbox_event(
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                event_type=event_type,
                payload=event_payload,
                transaction=tr
            )
            
            await tr.commit()
            return result
        
        except Exception as e:
            await tr.rollback()
            raise
        finally:
            await conn.close()
```

## 8) Multi-Cloud Federation

Create `federation/control_plane.py`:

```python
"""Multi-cloud federation control plane."""
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import boto3
from google.cloud import container_v1
from azure.mgmt.containerservice import ContainerServiceClient
from azure.identity import DefaultAzureCredential

from prometheus_client import Gauge, Counter

federation_metrics = {
    "cloud_resources": Gauge("federation_cloud_resources", "Resources per cloud", ["cloud", "resource_type"]),
    "cross_cloud_requests": Counter("federation_cross_cloud_requests_total", "Cross-cloud requests", ["from_cloud", "to_cloud"]),
    "federation_health": Gauge("federation_health", "Federation health", ["cloud"]),
}


class CloudProvider(Enum):
    """Cloud provider enumeration."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREM = "on_prem"


@dataclass
class CloudCluster:
    """Cloud cluster definition."""
    provider: CloudProvider
    cluster_name: str
    region: str
    endpoint: str
    credentials: Dict[str, str]
    kafka_bootstrap_servers: List[str]
    schema_registry_url: str


class FederationControlPlane:
    """Manages multi-cloud federation."""
    
    def __init__(self):
        self.clusters: Dict[CloudProvider, CloudCluster] = {}
        self.service_registry: Dict[str, CloudProvider] = {}
    
    def register_cluster(self, cluster: CloudCluster):
        """Register a cloud cluster."""
        self.clusters[cluster.provider] = cluster
        federation_metrics["federation_health"].labels(cloud=cluster.provider.value).set(1.0)
    
    def register_service(self, service_name: str, provider: CloudProvider):
        """Register a service to a cloud provider."""
        self.service_registry[service_name] = provider
    
    def get_service_cluster(self, service_name: str) -> Optional[CloudCluster]:
        """Get cluster for a service."""
        provider = self.service_registry.get(service_name)
        if provider:
            return self.clusters.get(provider)
        return None
    
    def route_request(
        self,
        service_name: str,
        request_data: Dict[str, Any],
        preferred_cloud: Optional[CloudProvider] = None
    ) -> Dict[str, Any]:
        """Route request to appropriate cloud."""
        target_cluster = self.get_service_cluster(service_name)
        
        if not target_cluster:
            # Fallback to preferred cloud or round-robin
            target_cluster = self._select_cluster(preferred_cloud)
        
        source_provider = preferred_cloud or CloudProvider.AWS
        target_provider = target_cluster.provider
        
        if source_provider != target_provider:
            federation_metrics["cross_cloud_requests"].labels(
                from_cloud=source_provider.value,
                to_cloud=target_provider.value
            ).inc()
        
        # Route request (in production, use service mesh or API gateway)
        return self._execute_request(target_cluster, service_name, request_data)
    
    def _select_cluster(self, preferred: Optional[CloudProvider]) -> CloudCluster:
        """Select cluster for routing."""
        if preferred and preferred in self.clusters:
            return self.clusters[preferred]
        
        # Round-robin or load-based selection
        return list(self.clusters.values())[0]
    
    def _execute_request(
        self,
        cluster: CloudCluster,
        service_name: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute request on target cluster."""
        # In production, use gRPC or HTTP client with proper authentication
        # For demo, return mock response
        return {"status": "success", "cluster": cluster.provider.value}
    
    def migrate_service(
        self,
        service_name: str,
        from_provider: CloudProvider,
        to_provider: CloudProvider
    ) -> bool:
        """Migrate service between clouds."""
        # Update service registry
        self.service_registry[service_name] = to_provider
        
        # In production, handle:
        # - DNS updates
        # - Load balancer updates
        # - Data migration
        # - Traffic shifting
        
        return True
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get federation status across all clouds."""
        status = {
            "clusters": {},
            "services": {},
            "health": {}
        }
        
        for provider, cluster in self.clusters.items():
            status["clusters"][provider.value] = {
                "name": cluster.cluster_name,
                "region": cluster.region,
                "endpoint": cluster.endpoint
            }
            
            # Check health
            health = self._check_cluster_health(cluster)
            status["health"][provider.value] = health
            federation_metrics["federation_health"].labels(cloud=provider.value).set(
                1.0 if health else 0.0
            )
        
        for service, provider in self.service_registry.items():
            status["services"][service] = provider.value
        
        return status
    
    def _check_cluster_health(self, cluster: CloudCluster) -> bool:
        """Check cluster health."""
        # In production, check:
        # - Kubernetes API availability
        # - Service endpoints
        # - Kafka connectivity
        return True
```

## 9) Cloud Adapter for Portability

Create `services/order-service/app/cloud_adapter.py`:

```python
"""Cloud adapter for multi-cloud portability."""
from typing import Dict, Any, Optional
from enum import Enum
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from federation.control_plane import CloudProvider


class CloudAdapter:
    """Abstracts cloud-specific operations for portability."""
    
    def __init__(self, provider: CloudProvider, credentials: Dict[str, str]):
        self.provider = provider
        self.credentials = credentials
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud-specific client."""
        if self.provider == CloudProvider.AWS:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                region_name=self.credentials.get('region', 'us-east-1')
            )
        
        elif self.provider == CloudProvider.GCP:
            self.gcs_client = gcs.Client(
                credentials=self.credentials.get('credentials')
            )
        
        elif self.provider == CloudProvider.AZURE:
            self.blob_client = BlobServiceClient(
                account_url=self.credentials.get('account_url'),
                credential=DefaultAzureCredential()
            )
    
    def upload_file(
        self,
        bucket_name: str,
        object_key: str,
        file_path: str
    ) -> str:
        """Upload file to object storage (portable interface)."""
        if self.provider == CloudProvider.AWS:
            self.s3_client.upload_file(file_path, bucket_name, object_key)
            return f"s3://{bucket_name}/{object_key}"
        
        elif self.provider == CloudProvider.GCP:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(object_key)
            blob.upload_from_filename(file_path)
            return f"gs://{bucket_name}/{object_key}"
        
        elif self.provider == CloudProvider.AZURE:
            container_client = self.blob_client.get_container_client(bucket_name)
            blob_client = container_client.get_blob_client(object_key)
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            return f"https://{bucket_name}.blob.core.windows.net/{object_key}"
    
    def download_file(
        self,
        bucket_name: str,
        object_key: str,
        local_path: str
    ):
        """Download file from object storage."""
        if self.provider == CloudProvider.AWS:
            self.s3_client.download_file(bucket_name, object_key, local_path)
        
        elif self.provider == CloudProvider.GCP:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(object_key)
            blob.download_to_filename(local_path)
        
        elif self.provider == CloudProvider.AZURE:
            container_client = self.blob_client.get_container_client(bucket_name)
            blob_client = container_client.get_blob_client(object_key)
            with open(local_path, 'wb') as data:
                data.write(blob_client.download_blob().readall())
    
    def get_kafka_bootstrap_servers(self) -> List[str]:
        """Get Kafka bootstrap servers (cloud-agnostic)."""
        # In production, use service discovery or configuration
        if self.provider == CloudProvider.AWS:
            return ["kafka.aws.internal:9092"]
        elif self.provider == CloudProvider.GCP:
            return ["kafka.gcp.internal:9092"]
        elif self.provider == CloudProvider.AZURE:
            return ["kafka.azure.internal:9092"]
        else:
            return ["kafka.onprem.internal:9092"]
```

## 10) API Versioning and Backward Compatibility

Create `services/order-service/app/api.py`:

```python
"""API with versioning and backward compatibility."""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime
import json

from proto.order_pb2 import (
    CreateOrderRequest, CreateOrderResponse,
    GetOrderRequest, GetOrderResponse,
    Order, OrderStatus
)
from proto.order_pb2_grpc import OrderServiceStub
import grpc

app = FastAPI(title="Order Service API", version="1.0.0")


class APIVersion:
    """API version management."""
    
    SUPPORTED_VERSIONS = ["v1", "v2"]
    DEFAULT_VERSION = "v1"
    DEPRECATED_VERSIONS = []
    
    @staticmethod
    def parse_version(version_header: Optional[str]) -> str:
        """Parse API version from header."""
        if not version_header:
            return APIVersion.DEFAULT_VERSION
        
        # Parse "application/vnd.api+json;version=v1"
        if "version=" in version_header:
            version = version_header.split("version=")[1].split(";")[0]
            if version in APIVersion.SUPPORTED_VERSIONS:
                return version
        
        return APIVersion.DEFAULT_VERSION
    
    @staticmethod
    def is_deprecated(version: str) -> bool:
        """Check if version is deprecated."""
        return version in APIVersion.DEPRECATED_VERSIONS


def get_api_version(
    accept: Optional[str] = Header(None, alias="Accept")
) -> str:
    """Dependency to extract API version."""
    version = APIVersion.parse_version(accept)
    
    if APIVersion.is_deprecated(version):
        # Add deprecation warning header
        pass
    
    return version


@app.post("/api/v1/orders", response_model=Dict)
@app.post("/api/v2/orders", response_model=Dict)
async def create_order(
    order_data: Dict,
    version: str = Depends(get_api_version)
):
    """Create order with version-aware handling."""
    # Validate version-specific requirements
    if version == "v2":
        # V2 requires additional fields
        if "metadata" not in order_data:
            raise HTTPException(
                status_code=400,
                detail="V2 API requires 'metadata' field"
            )
    
    # Convert to protobuf (version-agnostic core logic)
    request = CreateOrderRequest(
        customer_id=order_data["customer_id"],
        shipping_address=json.dumps(order_data["shipping_address"])
    )
    
    # Execute (same logic for all versions)
    # ... business logic ...
    
    # Format response based on version
    if version == "v1":
        return {
            "order_id": "123",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
    else:  # v2
        return {
            "order": {
                "id": "123",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            },
            "metadata": {
                "version": "v2",
                "saga_id": "saga-123"
            }
        }


@app.get("/api/v1/orders/{order_id}", response_model=Dict)
@app.get("/api/v2/orders/{order_id}", response_model=Dict)
async def get_order(
    order_id: str,
    version: str = Depends(get_api_version)
):
    """Get order with version-aware response."""
    # Fetch order (same for all versions)
    # ... fetch logic ...
    
    # Format based on version
    if version == "v1":
        return {
            "order_id": order_id,
            "status": "pending",
            "items": []
        }
    else:  # v2
        return {
            "order": {
                "id": order_id,
                "status": "pending",
                "items": []
            },
            "links": {
                "self": f"/api/v2/orders/{order_id}",
                "cancel": f"/api/v2/orders/{order_id}/cancel"
            }
        }
```

## 11) Payment Service (Go) with SAGA and Outbox

Create `services/payment-service/saga.go`:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/IBM/sarama"
	"google.golang.org/grpc"
	pb "github.com/company/payment-service/proto/payment/v1"
)

type SagaStep struct {
	StepID      string
	ServiceName string
	Action      string
	Status      string
	Payload     map[string]interface{}
}

type SagaOrchestrator struct {
	paymentClient pb.PaymentServiceClient
	kafkaProducer sarama.SyncProducer
}

func (so *SagaOrchestrator) ExecutePaymentStep(
	ctx context.Context,
	step SagaStep,
	sagaID string,
) error {
	if step.Action == "process_payment" {
		// Extract payload
		orderID := step.Payload["order_id"].(string)
		amount := step.Payload["amount"].(map[string]interface{})
		
		// Create gRPC request
		req := &pb.ProcessPaymentRequest{
			OrderId: orderID,
			Amount: &pb.Money{
				Currency:    amount["currency"].(string),
				AmountCents: int64(amount["amount_cents"].(float64)),
			},
			SagaId: sagaID,
		}
		
		// Execute payment
		resp, err := so.paymentClient.ProcessPayment(ctx, req)
		if err != nil {
			return fmt.Errorf("payment processing failed: %w", err)
		}
		
		if resp.Status != pb.PaymentStatus_PAYMENT_STATUS_COMPLETED {
			return fmt.Errorf("payment not completed: %s", resp.Status)
		}
		
		return nil
	}
	
	return fmt.Errorf("unknown action: %s", step.Action)
}

func (so *SagaOrchestrator) CompensatePayment(
	ctx context.Context,
	paymentID string,
	sagaID string,
) error {
	req := &pb.RefundPaymentRequest{
		PaymentId: paymentID,
		Reason:    "SAGA compensation",
		SagaId:    sagaID,
	}
	
	_, err := so.paymentClient.RefundPayment(ctx, req)
	return err
}
```

## 12) Inventory Service (Rust) with CQRS

Create `services/inventory-service/src/cqrs.rs`:

```rust
/// CQRS implementation in Rust
use serde::{Deserialize, Serialize};
use tokio_postgres::Client;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub command_id: String,
    pub command_type: String,
    pub payload: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub query_id: String,
    pub query_type: String,
    pub filters: HashMap<String, serde_json::Value>,
}

pub struct WriteSide {
    db_client: Client,
}

impl WriteSide {
    pub async fn execute_command(&self, command: Command) -> Result<HashMap<String, serde_json::Value>, String> {
        match command.command_type.as_str() {
            "reserve_items" => self.reserve_items(command).await,
            "release_reservation" => self.release_reservation(command).await,
            _ => Err(format!("Unknown command type: {}", command.command_type)),
        }
    }
    
    async fn reserve_items(&self, command: Command) -> Result<HashMap<String, serde_json::Value>, String> {
        // Execute reservation in database
        let items = command.payload.get("items")
            .and_then(|v| v.as_array())
            .ok_or("Missing items")?;
        
        for item in items {
            let product_id = item.get("product_id")
                .and_then(|v| v.as_str())
                .ok_or("Missing product_id")?;
            let quantity = item.get("quantity")
                .and_then(|v| v.as_i64())
                .ok_or("Missing quantity")? as i32;
            
            // Reserve in database
            self.db_client.execute(
                "UPDATE inventory SET reserved = reserved + $1 WHERE product_id = $2",
                &[&quantity, &product_id]
            ).await
            .map_err(|e| format!("Reservation failed: {}", e))?;
        }
        
        Ok(HashMap::new())
    }
    
    async fn release_reservation(&self, command: Command) -> Result<HashMap<String, serde_json::Value>, String> {
        // Release reservation (compensation)
        Ok(HashMap::new())
    }
}

pub struct ReadSide {
    // In production, use DuckDB or Parquet for read model
}

impl ReadSide {
    pub async fn execute_query(&self, query: Query) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        match query.query_type.as_str() {
            "get_inventory" => self.get_inventory(query).await,
            "list_products" => self.list_products(query).await,
            _ => Err(format!("Unknown query type: {}", query.query_type)),
        }
    }
    
    async fn get_inventory(&self, query: Query) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        // Query read model
        Ok(vec![])
    }
    
    async fn list_products(&self, query: Query) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        // Query read model
        Ok(vec![])
    }
}
```

## 13) Docker Compose for Multi-Cloud Simulation

Create `docker-compose.yaml`:

```yaml
version: '3.8'

services:
  # Kafka clusters (simulating different clouds)
  kafka-aws:
    image: bitnami/kafka:latest
    container_name: kafka-aws
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_CFG_LISTENERS: 'PLAINTEXT://0.0.0.0:9092'
      KAFKA_CFG_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka-aws:9092'
    ports:
      - "29092:9092"
    networks:
      - aws-network

  kafka-gcp:
    image: bitnami/kafka:latest
    container_name: kafka-gcp
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_CFG_LISTENERS: 'PLAINTEXT://0.0.0.0:9092'
      KAFKA_CFG_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka-gcp:9092'
    ports:
      - "29093:9092"
    networks:
      - gcp-network

  # Schema Registry
  schema-registry:
    image: confluentinc/cp-schema-registry:7.6.1
    container_name: schema-registry
    environment:
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka-aws:9092
    ports:
      - "8081:8081"
    depends_on:
      - kafka-aws

  # PostgreSQL (write side)
  postgres-write:
    image: postgres:16-alpine
    container_name: postgres-write
    environment:
      POSTGRES_DB: orders_db
      POSTGRES_USER: orders_user
      POSTGRES_PASSWORD: orders_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Order Service (Python)
  order-service:
    build:
      context: ./services/order-service
      dockerfile: Dockerfile
    container_name: order-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres-write:5432/orders_db
      KAFKA_BOOTSTRAP_SERVERS: kafka-aws:9092
      SCHEMA_REGISTRY_URL: http://schema-registry:8081
      CLOUD_PROVIDER: aws
    ports:
      - "8000:8000"
    depends_on:
      - postgres-write
      - kafka-aws
      - schema-registry

  # Payment Service (Go)
  payment-service:
    build:
      context: ./services/payment-service
      dockerfile: Dockerfile
    container_name: payment-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres-write:5432/orders_db
      KAFKA_BOOTSTRAP_SERVERS: kafka-gcp:9092
      CLOUD_PROVIDER: gcp
    ports:
      - "8001:8000"
    depends_on:
      - postgres-write
      - kafka-gcp

  # Inventory Service (Rust)
  inventory-service:
    build:
      context: ./services/inventory-service
      dockerfile: Dockerfile
    container_name: inventory-service
    environment:
      DATABASE_URL: postgresql://orders_user:orders_pass@postgres-write:5432/orders_db
      KAFKA_BOOTSTRAP_SERVERS: kafka-aws:9092
      CLOUD_PROVIDER: aws
    ports:
      - "8002:8000"
    depends_on:
      - postgres-write
      - kafka-aws

  # Federation Control Plane
  federation-control:
    build:
      context: ./federation
      dockerfile: Dockerfile
    container_name: federation-control
    environment:
      AWS_CLUSTER_ENDPOINT: http://kafka-aws:9092
      GCP_CLUSTER_ENDPOINT: http://kafka-gcp:9092
    ports:
      - "8080:8080"
    depends_on:
      - kafka-aws
      - kafka-gcp

networks:
  aws-network:
  gcp-network:

volumes:
  postgres_data:
```

## 14) Database Schema for Outbox and CQRS

Create `sql/init.sql`:

```sql
-- Write side tables
CREATE TABLE orders (
    order_id VARCHAR(100) PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    saga_id VARCHAR(100)
);

CREATE TABLE order_items (
    order_id VARCHAR(100) REFERENCES orders(order_id),
    product_id VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- Outbox table
CREATE TABLE outbox (
    event_id VARCHAR(100) PRIMARY KEY,
    aggregate_id VARCHAR(100) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    published_at TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_outbox_pending ON outbox(status, created_at) WHERE status = 'pending';
CREATE INDEX idx_outbox_aggregate ON outbox(aggregate_id, aggregate_type);

-- Event store
CREATE TABLE events (
    event_id VARCHAR(100) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    aggregate_id VARCHAR(100) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL,
    command_id VARCHAR(100)
);

CREATE INDEX idx_events_aggregate ON events(aggregate_id, aggregate_type, version);

-- SAGA state table
CREATE TABLE saga_state (
    saga_id VARCHAR(100) PRIMARY KEY,
    order_id VARCHAR(100),
    status VARCHAR(50) NOT NULL,
    steps JSONB NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE INDEX idx_saga_order ON saga_state(order_id);
```

## 15) Testing the System

### 15.1) Start Services

```bash
# Start all services
docker compose up -d

# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### 15.2) Create Order with SAGA

```python
from services.order_service.app.saga import OrderSagaOrchestrator, SagaOrchestrator

# Initialize orchestrator
orchestrator = OrderSagaOrchestrator(...)

# Execute order creation SAGA
result = await orchestrator.create_order_saga(
    customer_id="cust123",
    items=[
        {"product_id": "prod1", "quantity": 2, "price_cents": 2999},
        {"product_id": "prod2", "quantity": 1, "price_cents": 4999}
    ],
    shipping_address={
        "street": "123 Main St",
        "city": "City",
        "state": "State",
        "zip_code": "12345",
        "country": "USA"
    },
    payment_method="credit_card"
)

print(f"SAGA result: {result.status}")
```

### 15.3) Test Multi-Cloud Federation

```python
from federation.control_plane import FederationControlPlane, CloudCluster, CloudProvider

federation = FederationControlPlane()

# Register clusters
federation.register_cluster(CloudCluster(
    provider=CloudProvider.AWS,
    cluster_name="aws-cluster",
    region="us-east-1",
    endpoint="https://eks.us-east-1.amazonaws.com",
    credentials={},
    kafka_bootstrap_servers=["kafka-aws:9092"],
    schema_registry_url="http://schema-registry:8081"
))

# Register services
federation.register_service("order-service", CloudProvider.AWS)
federation.register_service("payment-service", CloudProvider.GCP)

# Route request
result = federation.route_request(
    "payment-service",
    {"order_id": "123", "amount": 10000},
    preferred_cloud=CloudProvider.AWS
)
```

## 16) Best Practices Integration Summary

This tutorial demonstrates:

1. **Streaming Architecture Patterns**: SAGA for distributed transactions, CQRS for read/write separation, Outbox for reliable events
2. **Polyglot Interoperability**: Python, Go, Rust services communicate via Protocol Buffers and gRPC
3. **Multi-Cloud Federation**: Services run on AWS, GCP, Azure with unified control plane
4. **API Governance**: Versioned APIs with backward compatibility and cross-language stability

**Key Integration Points**:
- Protocol Buffers enable polyglot communication
- SAGA orchestrator works across all languages
- CQRS read/write separation scales independently
- Outbox pattern ensures reliable event publishing
- Multi-cloud federation enables vendor independence
- API versioning maintains backward compatibility

## 17) Next Steps

- Add event sourcing for full audit trail
- Implement cross-cloud data replication
- Add service mesh for multi-cloud networking
- Implement automated cloud migration
- Add multi-cloud cost optimization

---

*This tutorial demonstrates how multiple best practices integrate to create a scalable, portable, polyglot streaming platform that works across multiple cloud providers.*

