# Rust Microservices Best Practices

**Objective**: Master senior-level Rust microservices patterns for production systems. When you need to build scalable microservices, when you want to leverage Rust's performance for distributed systems, when you need enterprise-grade microservice patterns—these best practices become your weapon of choice.

## Core Principles

- **Service Independence**: Each microservice should be independently deployable
- **Communication**: Use appropriate communication patterns between services
- **Resilience**: Implement circuit breakers and retry mechanisms
- **Observability**: Comprehensive logging, metrics, and tracing
- **Configuration**: Externalized configuration management

## Service Architecture Patterns

### Service Discovery

```rust
// rust/01-service-discovery.rs

/*
Service discovery patterns and best practices
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Service instance information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub id: Uuid,
    pub name: String,
    pub host: String,
    pub port: u16,
    pub health_check_url: String,
    pub metadata: HashMap<String, String>,
    pub tags: Vec<String>,
}

/// Service registry for managing service instances.
pub struct ServiceRegistry {
    services: Arc<RwLock<HashMap<String, Vec<ServiceInstance>>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register(&self, instance: ServiceInstance) -> Result<(), String> {
        let mut services = self.services.write().await;
        let service_name = instance.name.clone();
        
        services.entry(service_name)
            .or_insert_with(Vec::new)
            .push(instance);
        
        Ok(())
    }
    
    pub async fn deregister(&self, service_name: &str, instance_id: Uuid) -> Result<(), String> {
        let mut services = self.services.write().await;
        
        if let Some(instances) = services.get_mut(service_name) {
            instances.retain(|instance| instance.id != instance_id);
            
            if instances.is_empty() {
                services.remove(service_name);
            }
        }
        
        Ok(())
    }
    
    pub async fn discover(&self, service_name: &str) -> Result<Vec<ServiceInstance>, String> {
        let services = self.services.read().await;
        
        match services.get(service_name) {
            Some(instances) => Ok(instances.clone()),
            None => Err(format!("Service '{}' not found", service_name)),
        }
    }
    
    pub async fn get_healthy_instances(&self, service_name: &str) -> Result<Vec<ServiceInstance>, String> {
        let instances = self.discover(service_name).await?;
        
        // Filter healthy instances (in a real implementation, you'd check health)
        let healthy_instances: Vec<ServiceInstance> = instances
            .into_iter()
            .filter(|instance| self.is_healthy(instance).await)
            .collect();
        
        Ok(healthy_instances)
    }
    
    async fn is_healthy(&self, instance: &ServiceInstance) -> bool {
        // In a real implementation, you'd make an HTTP request to the health check URL
        // For now, we'll just return true
        true
    }
}

/// Service discovery client.
pub struct ServiceDiscoveryClient {
    registry: Arc<ServiceRegistry>,
    load_balancer: Box<dyn LoadBalancer>,
}

impl ServiceDiscoveryClient {
    pub fn new(registry: Arc<ServiceRegistry>, load_balancer: Box<dyn LoadBalancer>) -> Self {
        Self {
            registry,
            load_balancer,
        }
    }
    
    pub async fn get_service_instance(&self, service_name: &str) -> Result<ServiceInstance, String> {
        let instances = self.registry.get_healthy_instances(service_name).await?;
        
        if instances.is_empty() {
            return Err("No healthy instances available".to_string());
        }
        
        self.load_balancer.select_instance(instances)
    }
}

/// Load balancer trait.
pub trait LoadBalancer: Send + Sync {
    fn select_instance(&self, instances: Vec<ServiceInstance>) -> Result<ServiceInstance, String>;
}

/// Round-robin load balancer.
pub struct RoundRobinLoadBalancer {
    current_index: Arc<std::sync::atomic::AtomicUsize>,
}

impl RoundRobinLoadBalancer {
    pub fn new() -> Self {
        Self {
            current_index: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl LoadBalancer for RoundRobinLoadBalancer {
    fn select_instance(&self, instances: Vec<ServiceInstance>) -> Result<ServiceInstance, String> {
        if instances.is_empty() {
            return Err("No instances available".to_string());
        }
        
        let index = self.current_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let instance = instances[index % instances.len()].clone();
        
        Ok(instance)
    }
}

/// Random load balancer.
pub struct RandomLoadBalancer;

impl LoadBalancer for RandomLoadBalancer {
    fn select_instance(&self, instances: Vec<ServiceInstance>) -> Result<ServiceInstance, String> {
        if instances.is_empty() {
            return Err("No instances available".to_string());
        }
        
        let index = fastrand::usize(..instances.len());
        Ok(instances[index].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_registry() {
        let registry = ServiceRegistry::new();
        
        let instance = ServiceInstance {
            id: Uuid::new_v4(),
            name: "user-service".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            health_check_url: "/health".to_string(),
            metadata: HashMap::new(),
            tags: vec!["api".to_string()],
        };
        
        registry.register(instance.clone()).await.unwrap();
        
        let instances = registry.discover("user-service").await.unwrap();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].name, "user-service");
    }
    
    #[tokio::test]
    async fn test_load_balancer() {
        let registry = Arc::new(ServiceRegistry::new());
        let load_balancer = Box::new(RoundRobinLoadBalancer::new());
        let client = ServiceDiscoveryClient::new(registry, load_balancer);
        
        // This would fail in a real test since we don't have registered instances
        // but it demonstrates the API
        let result = client.get_service_instance("user-service").await;
        assert!(result.is_err());
    }
}
```

### Circuit Breaker Pattern

```rust
// rust/02-circuit-breaker.rs

/*
Circuit breaker patterns and best practices
*/

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Circuit breaker states.
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub timeout: Duration,
    pub retry_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            retry_timeout: Duration::from_secs(30),
        }
    }
}

/// Circuit breaker implementation.
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            config,
        }
    }
    
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        let state = self.state.read().await;
        
        match *state {
            CircuitState::Open => {
                if self.should_attempt_reset().await {
                    self.set_state(CircuitState::HalfOpen).await;
                } else {
                    return Err(CircuitBreakerError::CircuitOpen);
                }
            }
            CircuitState::HalfOpen => {
                // Allow one request to test if the service is back
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }
        
        drop(state);
        
        match operation() {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(error) => {
                self.on_failure().await;
                Err(CircuitBreakerError::OperationFailed(error.to_string()))
            }
        }
    }
    
    async fn should_attempt_reset(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;
        if let Some(time) = *last_failure {
            time.elapsed() >= self.config.retry_timeout
        } else {
            true
        }
    }
    
    async fn on_success(&self) {
        let mut state = self.state.write().await;
        let mut failure_count = self.failure_count.write().await;
        
        *failure_count = 0;
        *state = CircuitState::Closed;
    }
    
    async fn on_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        let mut last_failure_time = self.last_failure_time.write().await;
        
        *failure_count += 1;
        *last_failure_time = Some(Instant::now());
        
        if *failure_count >= self.config.failure_threshold {
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }
    
    async fn set_state(&self, new_state: CircuitState) {
        let mut state = self.state.write().await;
        *state = new_state;
    }
    
    pub async fn get_state(&self) -> CircuitState {
        let state = self.state.read().await;
        state.clone()
    }
}

/// Circuit breaker error types.
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    CircuitOpen,
    
    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Service client with circuit breaker.
pub struct ResilientServiceClient {
    circuit_breaker: Arc<CircuitBreaker>,
    base_url: String,
}

impl ResilientServiceClient {
    pub fn new(base_url: String, config: CircuitBreakerConfig) -> Self {
        Self {
            circuit_breaker: Arc::new(CircuitBreaker::new(config)),
            base_url,
        }
    }
    
    pub async fn call_service<T>(&self, endpoint: &str) -> Result<T, CircuitBreakerError>
    where
        T: for<'de> serde::Deserialize<'de>,
    {
        let url = format!("{}/{}", self.base_url, endpoint);
        
        self.circuit_breaker.call(|| {
            // In a real implementation, you'd make an HTTP request here
            // For now, we'll simulate a successful response
            Ok(serde_json::from_str("{}").unwrap())
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            timeout: Duration::from_secs(60),
            retry_timeout: Duration::from_secs(30),
        };
        
        let circuit_breaker = CircuitBreaker::new(config);
        
        // Test successful operation
        let result = circuit_breaker.call(|| Ok::<i32, String>(42)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        
        // Test failure handling
        let result = circuit_breaker.call(|| Err::<i32, String>("Test error".to_string())).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_resilient_service_client() {
        let config = CircuitBreakerConfig::default();
        let client = ResilientServiceClient::new("http://localhost:8080".to_string(), config);
        
        // This would make an actual HTTP request in a real implementation
        let result: Result<serde_json::Value, _> = client.call_service("test").await;
        assert!(result.is_ok());
    }
}
```

### Event-Driven Architecture

```rust
// rust/03-event-driven.rs

/*
Event-driven architecture patterns and best practices
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    UserCreated,
    UserUpdated,
    UserDeleted,
    OrderCreated,
    OrderUpdated,
    OrderDeleted,
}

/// Event structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub aggregate_id: String,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub version: u32,
}

/// Event handler trait.
pub trait EventHandler: Send + Sync {
    fn handle(&self, event: &Event) -> Result<(), String>;
    fn can_handle(&self, event_type: &EventType) -> bool;
}

/// Event bus for managing events.
pub struct EventBus {
    handlers: Vec<Box<dyn EventHandler>>,
    events: Vec<Event>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            events: Vec::new(),
        }
    }
    
    pub fn register_handler(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }
    
    pub async fn publish(&mut self, event: Event) -> Result<(), String> {
        // Store the event
        self.events.push(event.clone());
        
        // Notify all handlers
        for handler in &self.handlers {
            if handler.can_handle(&event.event_type) {
                if let Err(e) = handler.handle(&event) {
                    eprintln!("Handler error: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    pub fn get_events(&self) -> &[Event] {
        &self.events
    }
}

/// User event handler.
pub struct UserEventHandler {
    user_service: String,
}

impl UserEventHandler {
    pub fn new(user_service: String) -> Self {
        Self { user_service }
    }
}

impl EventHandler for UserEventHandler {
    fn handle(&self, event: &Event) -> Result<(), String> {
        match event.event_type {
            EventType::UserCreated => {
                println!("User created: {}", event.aggregate_id);
                // Handle user created event
            }
            EventType::UserUpdated => {
                println!("User updated: {}", event.aggregate_id);
                // Handle user updated event
            }
            EventType::UserDeleted => {
                println!("User deleted: {}", event.aggregate_id);
                // Handle user deleted event
            }
            _ => {}
        }
        Ok(())
    }
    
    fn can_handle(&self, event_type: &EventType) -> bool {
        matches!(event_type, EventType::UserCreated | EventType::UserUpdated | EventType::UserDeleted)
    }
}

/// Order event handler.
pub struct OrderEventHandler {
    order_service: String,
}

impl OrderEventHandler {
    pub fn new(order_service: String) -> Self {
        Self { order_service }
    }
}

impl EventHandler for OrderEventHandler {
    fn handle(&self, event: &Event) -> Result<(), String> {
        match event.event_type {
            EventType::OrderCreated => {
                println!("Order created: {}", event.aggregate_id);
                // Handle order created event
            }
            EventType::OrderUpdated => {
                println!("Order updated: {}", event.aggregate_id);
                // Handle order updated event
            }
            EventType::OrderDeleted => {
                println!("Order deleted: {}", event.aggregate_id);
                // Handle order deleted event
            }
            _ => {}
        }
        Ok(())
    }
    
    fn can_handle(&self, event_type: &EventType) -> bool {
        matches!(event_type, EventType::OrderCreated | EventType::OrderUpdated | EventType::OrderDeleted)
    }
}

/// Event store for persisting events.
pub struct EventStore {
    events: Vec<Event>,
}

impl EventStore {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }
    
    pub fn append(&mut self, event: Event) -> Result<(), String> {
        self.events.push(event);
        Ok(())
    }
    
    pub fn get_events(&self, aggregate_id: &str) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|event| event.aggregate_id == aggregate_id)
            .collect()
    }
    
    pub fn get_all_events(&self) -> &[Event] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_bus() {
        let mut event_bus = EventBus::new();
        
        let user_handler = Box::new(UserEventHandler::new("user-service".to_string()));
        let order_handler = Box::new(OrderEventHandler::new("order-service".to_string()));
        
        event_bus.register_handler(user_handler);
        event_bus.register_handler(order_handler);
        
        let event = Event {
            id: Uuid::new_v4(),
            event_type: EventType::UserCreated,
            aggregate_id: "user-123".to_string(),
            data: serde_json::Value::Null,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            version: 1,
        };
        
        // This would be async in a real implementation
        // event_bus.publish(event).await.unwrap();
    }
    
    #[test]
    fn test_event_store() {
        let mut event_store = EventStore::new();
        
        let event = Event {
            id: Uuid::new_v4(),
            event_type: EventType::UserCreated,
            aggregate_id: "user-123".to_string(),
            data: serde_json::Value::Null,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            version: 1,
        };
        
        event_store.append(event).unwrap();
        
        let events = event_store.get_events("user-123");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].aggregate_id, "user-123");
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Service discovery
let registry = ServiceRegistry::new();
let instance = ServiceInstance { /* ... */ };
registry.register(instance).await?;

// 2. Circuit breaker
let circuit_breaker = CircuitBreaker::new(config);
let result = circuit_breaker.call(|| operation()).await?;

// 3. Event-driven architecture
let mut event_bus = EventBus::new();
event_bus.register_handler(Box::new(MyEventHandler::new()));
event_bus.publish(event).await?;
```

### Essential Patterns

```rust
// Complete microservices setup
pub fn setup_rust_microservices() {
    // 1. Service discovery
    // 2. Circuit breakers
    // 3. Event-driven architecture
    // 4. Service communication
    // 5. Load balancing
    // 6. Resilience patterns
    // 7. Observability
    // 8. Configuration management
    
    println!("Rust microservices setup complete!");
}
```

---

*This guide provides the complete machinery for Rust microservices. Each pattern includes implementation examples, microservice strategies, and real-world usage patterns for enterprise distributed systems.*
