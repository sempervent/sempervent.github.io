# Python Microservices Best Practices

**Objective**: Master senior-level Python microservices patterns for production systems. When you need to build scalable, distributed applications, when you want to implement service communication patterns, when you need enterprise-grade microservices strategies—these best practices become your weapon of choice.

## Core Principles

- **Service Independence**: Each service should be independently deployable
- **Data Isolation**: Services should own their data
- **Communication**: Use appropriate communication patterns
- **Resilience**: Implement circuit breakers and retry mechanisms
- **Observability**: Monitor and trace service interactions

## Service Architecture

### Service Discovery

```python
# python/01-service-architecture.py

"""
Microservices architecture patterns and service discovery
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import time
from datetime import datetime
import uuid

class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

@dataclass
class ServiceInfo:
    """Service information"""
    service_id: str
    name: str
    version: str
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    registered_at: datetime
    last_heartbeat: datetime

class ServiceRegistry:
    """Service registry for service discovery"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.health_checkers: Dict[str, Callable] = {}
        self.heartbeat_interval = 30  # seconds
        self.health_check_interval = 10  # seconds
    
    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register service in registry"""
        try:
            self.services[service_info.service_id] = service_info
            print(f"Service {service_info.name} registered with ID {service_info.service_id}")
            return True
        except Exception as e:
            print(f"Failed to register service: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister service from registry"""
        if service_id in self.services:
            del self.services[service_id]
            print(f"Service {service_id} unregistered")
            return True
        return False
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service by ID"""
        return self.services.get(service_id)
    
    def get_services_by_name(self, name: str) -> List[ServiceInfo]:
        """Get services by name"""
        return [service for service in self.services.values() if service.name == name]
    
    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get all healthy services"""
        return [service for service in self.services.values() if service.status == ServiceStatus.HEALTHY]
    
    def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status"""
        if service_id in self.services:
            self.services[service_id].status = status
            self.services[service_id].last_heartbeat = datetime.utcnow()
            return True
        return False
    
    def register_health_checker(self, service_id: str, checker: Callable) -> None:
        """Register health checker for service"""
        self.health_checkers[service_id] = checker
    
    async def perform_health_check(self, service_id: str) -> bool:
        """Perform health check for service"""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        checker = self.health_checkers.get(service_id)
        
        if checker:
            try:
                result = await checker()
                if result:
                    self.update_service_status(service_id, ServiceStatus.HEALTHY)
                else:
                    self.update_service_status(service_id, ServiceStatus.UNHEALTHY)
                return result
            except Exception as e:
                print(f"Health check failed for {service_id}: {e}")
                self.update_service_status(service_id, ServiceStatus.UNHEALTHY)
                return False
        
        return True
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all services"""
        while True:
            for service_id in list(self.services.keys()):
                await self.perform_health_check(service_id)
            await asyncio.sleep(self.health_check_interval)

class ServiceClient:
    """Service client for inter-service communication"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
        self.http_client = None  # Would be aiohttp.ClientSession in real implementation
    
    async def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover service by name"""
        services = self.registry.get_services_by_name(service_name)
        healthy_services = [s for s in services if s.status == ServiceStatus.HEALTHY]
        
        if not healthy_services:
            return None
        
        # Simple round-robin load balancing
        return healthy_services[0]
    
    async def call_service(self, service_name: str, endpoint: str, method: str = "GET", 
                          data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call service endpoint"""
        service = await self.discover_service(service_name)
        if not service:
            raise Exception(f"Service {service_name} not found or unhealthy")
        
        # In real implementation, this would make HTTP request
        url = f"http://{service.host}:{service.port}{endpoint}"
        print(f"Calling {method} {url}")
        
        # Simulate service call
        return {
            "status": "success",
            "service": service.name,
            "endpoint": endpoint,
            "data": data
        }

# Usage examples
async def example_service_discovery():
    """Example service discovery usage"""
    # Create service registry
    registry = ServiceRegistry()
    
    # Register services
    user_service = ServiceInfo(
        service_id=str(uuid.uuid4()),
        name="user-service",
        version="1.0.0",
        host="localhost",
        port=8001,
        status=ServiceStatus.HEALTHY,
        health_check_url="/health",
        metadata={"environment": "production"},
        registered_at=datetime.utcnow(),
        last_heartbeat=datetime.utcnow()
    )
    
    order_service = ServiceInfo(
        service_id=str(uuid.uuid4()),
        name="order-service",
        version="1.0.0",
        host="localhost",
        port=8002,
        status=ServiceStatus.HEALTHY,
        health_check_url="/health",
        metadata={"environment": "production"},
        registered_at=datetime.utcnow(),
        last_heartbeat=datetime.utcnow()
    )
    
    registry.register_service(user_service)
    registry.register_service(order_service)
    
    # Create service client
    client = ServiceClient(registry)
    
    # Call services
    try:
        result = await client.call_service("user-service", "/users", "GET")
        print(f"User service result: {result}")
    except Exception as e:
        print(f"Service call failed: {e}")
    
    # Start health monitoring
    asyncio.create_task(registry.start_health_monitoring())
```

### Service Communication

```python
# python/02-service-communication.py

"""
Service communication patterns including synchronous and asynchronous communication
"""

from typing import List, Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import time
from datetime import datetime
import uuid

class MessageType(Enum):
    """Message type enumeration"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"

@dataclass
class Message:
    """Message for service communication"""
    message_id: str
    message_type: MessageType
    source_service: str
    target_service: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

class MessageBus:
    """Message bus for service communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue: List[Message] = []
        self.message_handlers: Dict[str, Callable] = {}
    
    def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)
    
    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Unsubscribe from topic"""
        if topic in self.subscribers:
            if handler in self.subscribers[topic]:
                self.subscribers[topic].remove(handler)
    
    async def publish(self, topic: str, message: Message) -> None:
        """Publish message to topic"""
        if topic in self.subscribers:
            for handler in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    print(f"Error handling message: {e}")
    
    async def send_message(self, message: Message) -> None:
        """Send message to target service"""
        topic = f"service.{message.target_service}"
        await self.publish(topic, message)
    
    async def send_request(self, source_service: str, target_service: str, 
                          payload: Dict[str, Any], reply_to: str = None) -> str:
        """Send request message"""
        message_id = str(uuid.uuid4())
        message = Message(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            source_service=source_service,
            target_service=target_service,
            payload=payload,
            timestamp=datetime.utcnow(),
            reply_to=reply_to
        )
        
        await self.send_message(message)
        return message_id
    
    async def send_response(self, request_message: Message, payload: Dict[str, Any]) -> None:
        """Send response message"""
        response_message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            source_service=request_message.target_service,
            target_service=request_message.source_service,
            payload=payload,
            timestamp=datetime.utcnow(),
            correlation_id=request_message.message_id
        )
        
        await self.send_message(response_message)
    
    async def send_event(self, source_service: str, event_type: str, 
                        payload: Dict[str, Any]) -> None:
        """Send event message"""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVENT,
            source_service=source_service,
            target_service="*",  # Broadcast to all
            payload={
                "event_type": event_type,
                "data": payload
            },
            timestamp=datetime.utcnow()
        )
        
        # Publish to all services
        for topic in self.subscribers.keys():
            await self.publish(topic, message)

class ServiceGateway:
    """Service gateway for routing and load balancing"""
    
    def __init__(self, service_registry: ServiceRegistry, message_bus: MessageBus):
        self.registry = service_registry
        self.message_bus = message_bus
        self.routing_rules: Dict[str, str] = {}
        self.load_balancers: Dict[str, List[str]] = {}
    
    def add_route(self, path: str, service_name: str) -> None:
        """Add routing rule"""
        self.routing_rules[path] = service_name
    
    def add_load_balancer(self, service_name: str, service_ids: List[str]) -> None:
        """Add load balancer for service"""
        self.load_balancers[service_name] = service_ids
    
    async def route_request(self, path: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate service"""
        if path not in self.routing_rules:
            raise Exception(f"No route found for {path}")
        
        service_name = self.routing_rules[path]
        
        # Get available services
        services = self.registry.get_services_by_name(service_name)
        healthy_services = [s for s in services if s.status == ServiceStatus.HEALTHY]
        
        if not healthy_services:
            raise Exception(f"No healthy services found for {service_name}")
        
        # Simple round-robin load balancing
        service = healthy_services[0]
        
        # Send request to service
        request_id = await self.message_bus.send_request(
            source_service="gateway",
            target_service=service.name,
            payload={
                "path": path,
                "method": method,
                "data": data
            }
        )
        
        return {
            "request_id": request_id,
            "service": service.name,
            "status": "routed"
        }

class CircuitBreaker:
    """Circuit breaker for service resilience"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self) -> None:
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout
        }

# Usage examples
async def example_service_communication():
    """Example service communication usage"""
    # Create message bus
    message_bus = MessageBus()
    
    # Create service registry
    registry = ServiceRegistry()
    
    # Create service gateway
    gateway = ServiceGateway(registry, message_bus)
    
    # Add routing rules
    gateway.add_route("/users", "user-service")
    gateway.add_route("/orders", "order-service")
    
    # Subscribe to messages
    async def handle_user_request(message: Message):
        print(f"Handling user request: {message.payload}")
        # Process request and send response
        await message_bus.send_response(message, {"result": "success"})
    
    message_bus.subscribe("service.user-service", handle_user_request)
    
    # Send request through gateway
    try:
        result = await gateway.route_request("/users", "GET", {})
        print(f"Gateway result: {result}")
    except Exception as e:
        print(f"Gateway error: {e}")
    
    # Send event
    await message_bus.send_event(
        source_service="user-service",
        event_type="user_created",
        payload={"user_id": "123", "name": "John Doe"}
    )
```

### Data Consistency

```python
# python/03-data-consistency.py

"""
Data consistency patterns for microservices including saga pattern and event sourcing
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import time
from datetime import datetime
import uuid

class SagaStatus(Enum):
    """Saga status enumeration"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

class SagaStepStatus(Enum):
    """Saga step status enumeration"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    """Saga step definition"""
    step_id: str
    service_name: str
    action: str
    compensate_action: str
    payload: Dict[str, Any]
    status: SagaStepStatus
    executed_at: Optional[datetime] = None
    compensated_at: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class Saga:
    """Saga definition"""
    saga_id: str
    status: SagaStatus
    steps: List[SagaStep]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class SagaOrchestrator:
    """Saga orchestrator for managing distributed transactions"""
    
    def __init__(self):
        self.sagas: Dict[str, Saga] = {}
        self.step_handlers: Dict[str, Callable] = {}
        self.compensation_handlers: Dict[str, Callable] = {}
    
    def register_step_handler(self, service_name: str, action: str, handler: Callable) -> None:
        """Register step handler"""
        key = f"{service_name}.{action}"
        self.step_handlers[key] = handler
    
    def register_compensation_handler(self, service_name: str, action: str, handler: Callable) -> None:
        """Register compensation handler"""
        key = f"{service_name}.{action}"
        self.compensation_handlers[key] = handler
    
    async def create_saga(self, steps: List[Dict[str, Any]]) -> str:
        """Create new saga"""
        saga_id = str(uuid.uuid4())
        saga_steps = []
        
        for step_data in steps:
            step = SagaStep(
                step_id=str(uuid.uuid4()),
                service_name=step_data["service_name"],
                action=step_data["action"],
                compensate_action=step_data["compensate_action"],
                payload=step_data["payload"],
                status=SagaStepStatus.PENDING
            )
            saga_steps.append(step)
        
        saga = Saga(
            saga_id=saga_id,
            status=SagaStatus.STARTED,
            steps=saga_steps,
            created_at=datetime.utcnow()
        )
        
        self.sagas[saga_id] = saga
        return saga_id
    
    async def execute_saga(self, saga_id: str) -> bool:
        """Execute saga"""
        if saga_id not in self.sagas:
            return False
        
        saga = self.sagas[saga_id]
        
        try:
            # Execute steps in order
            for step in saga.steps:
                await self.execute_step(step)
            
            # Mark saga as completed
            saga.status = SagaStatus.COMPLETED
            saga.completed_at = datetime.utcnow()
            return True
        
        except Exception as e:
            # Start compensation
            saga.status = SagaStatus.FAILED
            saga.error = str(e)
            await self.compensate_saga(saga_id)
            return False
    
    async def execute_step(self, step: SagaStep) -> None:
        """Execute saga step"""
        step.status = SagaStepStatus.EXECUTING
        step.executed_at = datetime.utcnow()
        
        try:
            key = f"{step.service_name}.{step.action}"
            handler = self.step_handlers.get(key)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(step.payload)
                else:
                    handler(step.payload)
                
                step.status = SagaStepStatus.COMPLETED
            else:
                raise Exception(f"No handler found for {key}")
        
        except Exception as e:
            step.status = SagaStepStatus.FAILED
            step.error = str(e)
            raise e
    
    async def compensate_saga(self, saga_id: str) -> None:
        """Compensate saga"""
        if saga_id not in self.sagas:
            return
        
        saga = self.sagas[saga_id]
        saga.status = SagaStatus.COMPENSATING
        
        # Compensate steps in reverse order
        for step in reversed(saga.steps):
            if step.status == SagaStepStatus.COMPLETED:
                await self.compensate_step(step)
        
        saga.status = SagaStatus.COMPENSATED
    
    async def compensate_step(self, step: SagaStep) -> None:
        """Compensate saga step"""
        step.status = SagaStepStatus.COMPENSATING
        step.compensated_at = datetime.utcnow()
        
        try:
            key = f"{step.service_name}.{step.compensate_action}"
            handler = self.compensation_handlers.get(key)
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(step.payload)
                else:
                    handler(step.payload)
                
                step.status = SagaStepStatus.COMPENSATED
            else:
                raise Exception(f"No compensation handler found for {key}")
        
        except Exception as e:
            step.error = str(e)
            raise e
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Get saga by ID"""
        return self.sagas.get(saga_id)
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get saga status"""
        saga = self.get_saga(saga_id)
        if not saga:
            return None
        
        return {
            "saga_id": saga.saga_id,
            "status": saga.status.value,
            "steps": [
                {
                    "step_id": step.step_id,
                    "service_name": step.service_name,
                    "action": step.action,
                    "status": step.status.value,
                    "executed_at": step.executed_at.isoformat() if step.executed_at else None,
                    "compensated_at": step.compensated_at.isoformat() if step.compensated_at else None,
                    "error": step.error
                }
                for step in saga.steps
            ],
            "created_at": saga.created_at.isoformat(),
            "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
            "error": saga.error
        }

class EventStore:
    """Event store for event sourcing"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.aggregates: Dict[str, List[str]] = {}
    
    def append_event(self, aggregate_id: str, event_type: str, 
                    payload: Dict[str, Any], version: int) -> str:
        """Append event to store"""
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "payload": payload,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.events.append(event)
        
        if aggregate_id not in self.aggregates:
            self.aggregates[aggregate_id] = []
        self.aggregates[aggregate_id].append(event_id)
        
        return event_id
    
    def get_events(self, aggregate_id: str) -> List[Dict[str, Any]]:
        """Get events for aggregate"""
        if aggregate_id not in self.aggregates:
            return []
        
        event_ids = self.aggregates[aggregate_id]
        return [event for event in self.events if event["event_id"] in event_ids]
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get events by type"""
        return [event for event in self.events if event["event_type"] == event_type]
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events"""
        return self.events.copy()

# Usage examples
async def example_data_consistency():
    """Example data consistency usage"""
    # Create saga orchestrator
    orchestrator = SagaOrchestrator()
    
    # Register step handlers
    async def create_user(payload: Dict[str, Any]):
        print(f"Creating user: {payload}")
        # Simulate user creation
        await asyncio.sleep(0.1)
    
    async def create_order(payload: Dict[str, Any]):
        print(f"Creating order: {payload}")
        # Simulate order creation
        await asyncio.sleep(0.1)
    
    async def send_notification(payload: Dict[str, Any]):
        print(f"Sending notification: {payload}")
        # Simulate notification
        await asyncio.sleep(0.1)
    
    # Register compensation handlers
    async def delete_user(payload: Dict[str, Any]):
        print(f"Deleting user: {payload}")
        # Simulate user deletion
        await asyncio.sleep(0.1)
    
    async def cancel_order(payload: Dict[str, Any]):
        print(f"Cancelling order: {payload}")
        # Simulate order cancellation
        await asyncio.sleep(0.1)
    
    # Register handlers
    orchestrator.register_step_handler("user-service", "create", create_user)
    orchestrator.register_step_handler("order-service", "create", create_order)
    orchestrator.register_step_handler("notification-service", "send", send_notification)
    
    orchestrator.register_compensation_handler("user-service", "create", delete_user)
    orchestrator.register_compensation_handler("order-service", "create", cancel_order)
    
    # Create saga
    steps = [
        {
            "service_name": "user-service",
            "action": "create",
            "compensate_action": "delete",
            "payload": {"name": "John Doe", "email": "john@example.com"}
        },
        {
            "service_name": "order-service",
            "action": "create",
            "compensate_action": "cancel",
            "payload": {"user_id": "123", "items": ["item1", "item2"]}
        },
        {
            "service_name": "notification-service",
            "action": "send",
            "compensate_action": "cancel",
            "payload": {"user_id": "123", "message": "Order created"}
        }
    ]
    
    saga_id = await orchestrator.create_saga(steps)
    print(f"Created saga: {saga_id}")
    
    # Execute saga
    success = await orchestrator.execute_saga(saga_id)
    print(f"Saga execution success: {success}")
    
    # Get saga status
    status = orchestrator.get_saga_status(saga_id)
    print(f"Saga status: {status}")
    
    # Event store example
    event_store = EventStore()
    
    # Append events
    event_store.append_event("user-123", "user_created", {"name": "John Doe"}, 1)
    event_store.append_event("user-123", "user_updated", {"email": "john@example.com"}, 2)
    event_store.append_event("order-456", "order_created", {"user_id": "user-123"}, 1)
    
    # Get events
    user_events = event_store.get_events("user-123")
    print(f"User events: {user_events}")
    
    order_events = event_store.get_events("order-456")
    print(f"Order events: {order_events}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Service registry
registry = ServiceRegistry()
service_info = ServiceInfo(...)
registry.register_service(service_info)

# 2. Service communication
message_bus = MessageBus()
await message_bus.publish("topic", message)

# 3. Circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
if circuit_breaker.can_execute():
    # Execute service call
    circuit_breaker.record_success()

# 4. Saga orchestration
orchestrator = SagaOrchestrator()
saga_id = await orchestrator.create_saga(steps)
await orchestrator.execute_saga(saga_id)

# 5. Event sourcing
event_store = EventStore()
event_store.append_event(aggregate_id, event_type, payload, version)
```

### Essential Patterns

```python
# Complete microservices setup
def setup_microservices():
    """Setup complete microservices environment"""
    
    # Service registry
    registry = ServiceRegistry()
    
    # Message bus
    message_bus = MessageBus()
    
    # Service gateway
    gateway = ServiceGateway(registry, message_bus)
    
    # Circuit breaker
    circuit_breaker = CircuitBreaker()
    
    # Saga orchestrator
    orchestrator = SagaOrchestrator()
    
    # Event store
    event_store = EventStore()
    
    print("Microservices setup complete!")
```

---

*This guide provides the complete machinery for Python microservices. Each pattern includes implementation examples, communication strategies, and real-world usage patterns for enterprise microservices development.*
