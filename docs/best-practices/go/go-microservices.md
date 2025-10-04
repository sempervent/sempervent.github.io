# Go Microservices Best Practices

**Objective**: Master senior-level Go microservices patterns for production systems. When you need to build robust, scalable distributed systems, when you want to follow proven methodologies, when you need enterprise-grade microservices patterns—these best practices become your weapon of choice.

## Core Principles

- **Service Independence**: Services should be independently deployable and scalable
- **Fault Tolerance**: Design for failure and implement circuit breakers
- **Observability**: Comprehensive logging, metrics, and tracing
- **API-First Design**: Well-defined service interfaces
- **Data Consistency**: Handle distributed data challenges

## Service Architecture

### Service Structure

```go
// internal/service/service.go
package service

import (
    "context"
    "fmt"
    "time"
)

// Service represents a microservice
type Service struct {
    name        string
    version     string
    port        int
    dependencies []Dependency
    healthCheck HealthCheck
    metrics     Metrics
    logger      Logger
}

// Dependency represents a service dependency
type Dependency struct {
    Name    string
    URL     string
    Timeout time.Duration
    Retries int
}

// HealthCheck represents a health check
type HealthCheck struct {
    Endpoint string
    Interval time.Duration
    Timeout  time.Duration
}

// NewService creates a new service
func NewService(name, version string, port int) *Service {
    return &Service{
        name:    name,
        version: version,
        port:    port,
        dependencies: make([]Dependency, 0),
        healthCheck: HealthCheck{
            Endpoint: "/health",
            Interval: 30 * time.Second,
            Timeout:  5 * time.Second,
        },
    }
}

// AddDependency adds a service dependency
func (s *Service) AddDependency(name, url string, timeout time.Duration, retries int) {
    s.dependencies = append(s.dependencies, Dependency{
        Name:    name,
        URL:     url,
        Timeout: timeout,
        Retries: retries,
    })
}

// Start starts the service
func (s *Service) Start(ctx context.Context) error {
    s.logger.Info("Starting service", map[string]interface{}{
        "name":    s.name,
        "version": s.version,
        "port":    s.port,
    })
    
    // Start health check
    go s.startHealthCheck(ctx)
    
    // Start metrics collection
    go s.startMetricsCollection(ctx)
    
    return nil
}

// startHealthCheck starts the health check routine
func (s *Service) startHealthCheck(ctx context.Context) {
    ticker := time.NewTicker(s.healthCheck.Interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            s.performHealthCheck()
        }
    }
}

// performHealthCheck performs a health check
func (s *Service) performHealthCheck() {
    for _, dep := range s.dependencies {
        if err := s.checkDependency(dep); err != nil {
            s.logger.Error("Dependency health check failed", map[string]interface{}{
                "dependency": dep.Name,
                "error":      err.Error(),
            })
        }
    }
}

// checkDependency checks a dependency's health
func (s *Service) checkDependency(dep Dependency) error {
    ctx, cancel := context.WithTimeout(context.Background(), dep.Timeout)
    defer cancel()
    
    // Implement health check logic
    return s.pingDependency(ctx, dep.URL)
}

// pingDependency pings a dependency
func (s *Service) pingDependency(ctx context.Context, url string) error {
    // Implement ping logic
    return nil
}
```

### Service Discovery

```go
// internal/discovery/discovery.go
package discovery

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// ServiceRegistry represents a service registry
type ServiceRegistry struct {
    services map[string][]ServiceInstance
    mutex    sync.RWMutex
    ttl      time.Duration
}

// ServiceInstance represents a service instance
type ServiceInstance struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Metadata map[string]string
    TTL      time.Time
}

// NewServiceRegistry creates a new service registry
func NewServiceRegistry(ttl time.Duration) *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string][]ServiceInstance),
        ttl:      ttl,
    }
}

// Register registers a service instance
func (sr *ServiceRegistry) Register(instance ServiceInstance) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instance.TTL = time.Now().Add(sr.ttl)
    sr.services[instance.Name] = append(sr.services[instance.Name], instance)
    
    return nil
}

// Deregister deregisters a service instance
func (sr *ServiceRegistry) Deregister(serviceName, instanceID string) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instances, exists := sr.services[serviceName]
    if !exists {
        return fmt.Errorf("service %s not found", serviceName)
    }
    
    for i, instance := range instances {
        if instance.ID == instanceID {
            sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
            return nil
        }
    }
    
    return fmt.Errorf("instance %s not found", instanceID)
}

// Discover discovers service instances
func (sr *ServiceRegistry) Discover(serviceName string) ([]ServiceInstance, error) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    instances, exists := sr.services[serviceName]
    if !exists {
        return nil, fmt.Errorf("service %s not found", serviceName)
    }
    
    // Filter expired instances
    var validInstances []ServiceInstance
    for _, instance := range instances {
        if time.Now().Before(instance.TTL) {
            validInstances = append(validInstances, instance)
        }
    }
    
    return validInstances, nil
}

// Cleanup removes expired instances
func (sr *ServiceRegistry) Cleanup() {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    for serviceName, instances := range sr.services {
        var validInstances []ServiceInstance
        for _, instance := range instances {
            if time.Now().Before(instance.TTL) {
                validInstances = append(validInstances, instance)
            }
        }
        sr.services[serviceName] = validInstances
    }
}
```

## Inter-Service Communication

### HTTP Client

```go
// internal/client/http_client.go
package client

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

// HTTPClient represents an HTTP client
type HTTPClient struct {
    baseURL    string
    timeout    time.Duration
    retries    int
    circuitBreaker *CircuitBreaker
    logger     Logger
}

// NewHTTPClient creates a new HTTP client
func NewHTTPClient(baseURL string, timeout time.Duration, retries int) *HTTPClient {
    return &HTTPClient{
        baseURL: baseURL,
        timeout: timeout,
        retries: retries,
        circuitBreaker: NewCircuitBreaker(5, time.Minute),
        logger:  NewLogger(),
    }
}

// Get performs a GET request
func (c *HTTPClient) Get(ctx context.Context, path string, result interface{}) error {
    return c.doRequest(ctx, "GET", path, nil, result)
}

// Post performs a POST request
func (c *HTTPClient) Post(ctx context.Context, path string, data interface{}, result interface{}) error {
    return c.doRequest(ctx, "POST", path, data, result)
}

// Put performs a PUT request
func (c *HTTPClient) Put(ctx context.Context, path string, data interface{}, result interface{}) error {
    return c.doRequest(ctx, "PUT", path, data, result)
}

// Delete performs a DELETE request
func (c *HTTPClient) Delete(ctx context.Context, path string) error {
    return c.doRequest(ctx, "DELETE", path, nil, nil)
}

// doRequest performs an HTTP request
func (c *HTTPClient) doRequest(ctx context.Context, method, path string, data interface{}, result interface{}) error {
    return c.circuitBreaker.Execute(ctx, func() error {
        return c.doRequestWithRetry(ctx, method, path, data, result)
    })
}

// doRequestWithRetry performs a request with retry logic
func (c *HTTPClient) doRequestWithRetry(ctx context.Context, method, path string, data interface{}, result interface{}) error {
    var lastErr error
    
    for i := 0; i <= c.retries; i++ {
        if i > 0 {
            time.Sleep(time.Duration(i) * time.Second)
        }
        
        err := c.performRequest(ctx, method, path, data, result)
        if err == nil {
            return nil
        }
        
        lastErr = err
        c.logger.Warn("Request failed, retrying", map[string]interface{}{
            "method": method,
            "path":   path,
            "attempt": i + 1,
            "error":  err.Error(),
        })
    }
    
    return fmt.Errorf("request failed after %d retries: %w", c.retries, lastErr)
}

// performRequest performs a single HTTP request
func (c *HTTPClient) performRequest(ctx context.Context, method, path string, data interface{}, result interface{}) error {
    url := c.baseURL + path
    
    var body []byte
    var err error
    if data != nil {
        body, err = json.Marshal(data)
        if err != nil {
            return fmt.Errorf("failed to marshal request body: %w", err)
        }
    }
    
    req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("User-Agent", "microservice-client/1.0")
    
    client := &http.Client{Timeout: c.timeout}
    resp, err := client.Do(req)
    if err != nil {
        return fmt.Errorf("failed to perform request: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode >= 400 {
        return fmt.Errorf("request failed with status %d", resp.StatusCode)
    }
    
    if result != nil {
        if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
            return fmt.Errorf("failed to decode response: %w", err)
        }
    }
    
    return nil
}
```

### gRPC Client

```go
// internal/client/grpc_client.go
package client

import (
    "context"
    "time"
    
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

// GRPCClient represents a gRPC client
type GRPCClient struct {
    conn   *grpc.ClientConn
    logger Logger
}

// NewGRPCClient creates a new gRPC client
func NewGRPCClient(address string) (*GRPCClient, error) {
    conn, err := grpc.Dial(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        return nil, fmt.Errorf("failed to connect to gRPC server: %w", err)
    }
    
    return &GRPCClient{
        conn:   conn,
        logger: NewLogger(),
    }, nil
}

// Close closes the gRPC connection
func (c *GRPCClient) Close() error {
    return c.conn.Close()
}

// Call performs a gRPC call
func (c *GRPCClient) Call(ctx context.Context, method string, req, resp interface{}) error {
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    
    return grpc.Invoke(ctx, method, req, resp, c.conn)
}
```

## Circuit Breaker Pattern

### Circuit Breaker Implementation

```go
// internal/circuitbreaker/circuitbreaker.go
package circuitbreaker

import (
    "context"
    "errors"
    "sync"
    "time"
)

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
    mu          sync.RWMutex
    state       CircuitState
    failures    int
    threshold   int
    timeout     time.Duration
    lastFailure time.Time
    successCount int
    halfOpenMaxCalls int
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        state:           StateClosed,
        threshold:       threshold,
        timeout:         timeout,
        halfOpenMaxCalls: 3,
    }
}

// Execute executes a function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
    if !cb.allowRequest() {
        return errors.New("circuit breaker is open")
    }
    
    err := fn()
    cb.recordResult(err)
    return err
}

// allowRequest checks if a request should be allowed
func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    switch cb.state {
    case StateClosed:
        return true
    case StateOpen:
        return time.Since(cb.lastFailure) > cb.timeout
    case StateHalfOpen:
        return cb.successCount < cb.halfOpenMaxCalls
    default:
        return false
    }
}

// recordResult records the result of a request
func (cb *CircuitBreaker) recordResult(err error) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        
        if cb.failures >= cb.threshold {
            cb.state = StateOpen
        }
    } else {
        cb.failures = 0
        cb.successCount++
        
        if cb.state == StateHalfOpen && cb.successCount >= cb.halfOpenMaxCalls {
            cb.state = StateClosed
            cb.successCount = 0
        }
    }
}

// GetState returns the current state
func (cb *CircuitBreaker) GetState() CircuitState {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    return cb.state
}
```

## Load Balancing

### Load Balancer

```go
// internal/loadbalancer/loadbalancer.go
package loadbalancer

import (
    "context"
    "math/rand"
    "sync"
    "time"
)

// LoadBalancer represents a load balancer
type LoadBalancer struct {
    instances []ServiceInstance
    strategy  LoadBalanceStrategy
    mutex     sync.RWMutex
}

// LoadBalanceStrategy represents a load balancing strategy
type LoadBalanceStrategy int

const (
    StrategyRoundRobin LoadBalanceStrategy = iota
    StrategyRandom
    StrategyLeastConnections
)

// NewLoadBalancer creates a new load balancer
func NewLoadBalancer(strategy LoadBalanceStrategy) *LoadBalancer {
    return &LoadBalancer{
        instances: make([]ServiceInstance, 0),
        strategy:  strategy,
    }
}

// AddInstance adds a service instance
func (lb *LoadBalancer) AddInstance(instance ServiceInstance) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    lb.instances = append(lb.instances, instance)
}

// RemoveInstance removes a service instance
func (lb *LoadBalancer) RemoveInstance(instanceID string) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, instance := range lb.instances {
        if instance.ID == instanceID {
            lb.instances = append(lb.instances[:i], lb.instances[i+1:]...)
            break
        }
    }
}

// GetInstance returns a service instance
func (lb *LoadBalancer) GetInstance() (ServiceInstance, error) {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    if len(lb.instances) == 0 {
        return ServiceInstance{}, errors.New("no instances available")
    }
    
    switch lb.strategy {
    case StrategyRoundRobin:
        return lb.getRoundRobinInstance()
    case StrategyRandom:
        return lb.getRandomInstance()
    case StrategyLeastConnections:
        return lb.getLeastConnectionsInstance()
    default:
        return lb.getRoundRobinInstance()
    }
}

// getRoundRobinInstance returns an instance using round-robin
func (lb *LoadBalancer) getRoundRobinInstance() ServiceInstance {
    // Implement round-robin logic
    return lb.instances[0]
}

// getRandomInstance returns a random instance
func (lb *LoadBalancer) getRandomInstance() ServiceInstance {
    rand.Seed(time.Now().UnixNano())
    return lb.instances[rand.Intn(len(lb.instances))]
}

// getLeastConnectionsInstance returns the instance with least connections
func (lb *LoadBalancer) getLeastConnectionsInstance() ServiceInstance {
    // Implement least connections logic
    return lb.instances[0]
}
```

## Event-Driven Architecture

### Event Bus

```go
// internal/events/event_bus.go
package events

import (
    "context"
    "sync"
    "time"
)

// Event represents an event
type Event struct {
    ID        string
    Type      string
    Source    string
    Data      interface{}
    Timestamp time.Time
    Version   string
}

// EventHandler represents an event handler
type EventHandler interface {
    Handle(ctx context.Context, event Event) error
}

// EventBus represents an event bus
type EventBus struct {
    handlers map[string][]EventHandler
    mutex    sync.RWMutex
    logger   Logger
}

// NewEventBus creates a new event bus
func NewEventBus() *EventBus {
    return &EventBus{
        handlers: make(map[string][]EventHandler),
        logger:   NewLogger(),
    }
}

// Subscribe subscribes to an event type
func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

// Publish publishes an event
func (eb *EventBus) Publish(ctx context.Context, event Event) error {
    eb.mutex.RLock()
    handlers := eb.handlers[event.Type]
    eb.mutex.RUnlock()
    
    for _, handler := range handlers {
        go func(h EventHandler) {
            if err := h.Handle(ctx, event); err != nil {
                eb.logger.Error("Event handler failed", map[string]interface{}{
                    "event_type": event.Type,
                    "error":      err.Error(),
                })
            }
        }(handler)
    }
    
    return nil
}
```

### Message Queue

```go
// internal/messaging/queue.go
package messaging

import (
    "context"
    "encoding/json"
    "time"
)

// Message represents a message
type Message struct {
    ID        string
    Type      string
    Data      interface{}
    Timestamp time.Time
    Retries   int
    MaxRetries int
}

// MessageQueue represents a message queue
type MessageQueue struct {
    name    string
    handler MessageHandler
    logger  Logger
}

// MessageHandler represents a message handler
type MessageHandler interface {
    Handle(ctx context.Context, message Message) error
}

// NewMessageQueue creates a new message queue
func NewMessageQueue(name string, handler MessageHandler) *MessageQueue {
    return &MessageQueue{
        name:    name,
        handler: handler,
        logger:  NewLogger(),
    }
}

// Publish publishes a message
func (mq *MessageQueue) Publish(ctx context.Context, messageType string, data interface{}) error {
    message := Message{
        ID:        generateID(),
        Type:      messageType,
        Data:      data,
        Timestamp: time.Now(),
        Retries:   0,
        MaxRetries: 3,
    }
    
    return mq.publishMessage(ctx, message)
}

// publishMessage publishes a message to the queue
func (mq *MessageQueue) publishMessage(ctx context.Context, message Message) error {
    // Implement message publishing logic
    return nil
}

// Consume consumes messages from the queue
func (mq *MessageQueue) Consume(ctx context.Context) error {
    // Implement message consumption logic
    return nil
}

// generateID generates a unique ID
func generateID() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}
```

## Distributed Tracing

### Tracing Implementation

```go
// internal/tracing/tracing.go
package tracing

import (
    "context"
    "fmt"
    "time"
)

// Trace represents a distributed trace
type Trace struct {
    ID        string
    ParentID  string
    Operation string
    StartTime time.Time
    EndTime   time.Time
    Tags      map[string]string
    Logs      []Log
}

// Log represents a trace log
type Log struct {
    Timestamp time.Time
    Fields    map[string]interface{}
}

// Tracer represents a tracer
type Tracer struct {
    serviceName string
    logger      Logger
}

// NewTracer creates a new tracer
func NewTracer(serviceName string) *Tracer {
    return &Tracer{
        serviceName: serviceName,
        logger:      NewLogger(),
    }
}

// StartSpan starts a new span
func (t *Tracer) StartSpan(ctx context.Context, operation string) (context.Context, *Span) {
    span := &Span{
        trace:     t,
        operation: operation,
        startTime: time.Now(),
        tags:      make(map[string]string),
        logs:      make([]Log, 0),
    }
    
    return context.WithValue(ctx, "span", span), span
}

// Span represents a span
type Span struct {
    trace     *Tracer
    operation string
    startTime time.Time
    endTime   time.Time
    tags      map[string]string
    logs      []Log
}

// SetTag sets a tag on the span
func (s *Span) SetTag(key, value string) {
    s.tags[key] = value
}

// Log logs an event on the span
func (s *Span) Log(fields map[string]interface{}) {
    s.logs = append(s.logs, Log{
        Timestamp: time.Now(),
        Fields:    fields,
    })
}

// Finish finishes the span
func (s *Span) Finish() {
    s.endTime = time.Now()
    
    trace := Trace{
        Operation: s.operation,
        StartTime: s.startTime,
        EndTime:   s.endTime,
        Tags:      s.tags,
        Logs:      s.logs,
    }
    
    s.trace.logger.Info("Span finished", map[string]interface{}{
        "operation": s.operation,
        "duration":  s.endTime.Sub(s.startTime),
        "tags":      s.tags,
    })
}
```

## Service Mesh Integration

### Service Mesh Client

```go
// internal/mesh/mesh_client.go
package mesh

import (
    "context"
    "fmt"
    "time"
)

// MeshClient represents a service mesh client
type MeshClient struct {
    serviceName string
    namespace   string
    logger      Logger
}

// NewMeshClient creates a new service mesh client
func NewMeshClient(serviceName, namespace string) *MeshClient {
    return &MeshClient{
        serviceName: serviceName,
        namespace:   namespace,
        logger:      NewLogger(),
    }
}

// CallService calls a service through the mesh
func (mc *MeshClient) CallService(ctx context.Context, serviceName, method string, data interface{}) (interface{}, error) {
    start := time.Now()
    
    // Implement service mesh call logic
    result, err := mc.performMeshCall(ctx, serviceName, method, data)
    
    duration := time.Since(start)
    mc.logger.Info("Service mesh call completed", map[string]interface{}{
        "service":  serviceName,
        "method":   method,
        "duration": duration,
        "error":    err != nil,
    })
    
    return result, err
}

// performMeshCall performs a call through the service mesh
func (mc *MeshClient) performMeshCall(ctx context.Context, serviceName, method string, data interface{}) (interface{}, error) {
    // Implement mesh call logic
    return nil, fmt.Errorf("mesh call not implemented")
}
```

## Testing Microservices

### Integration Testing

```go
// internal/testing/integration_test.go
package testing

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestMicroservice_Integration(t *testing.T) {
    // Arrange
    service := NewService("test-service", "1.0.0", 8080)
    service.AddDependency("user-service", "http://localhost:8081", 5*time.Second, 3)
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    // Act
    err := service.Start(ctx)
    
    // Assert
    require.NoError(t, err)
    
    // Test service functionality
    time.Sleep(100 * time.Millisecond)
    
    // Verify service is running
    assert.True(t, service.IsHealthy())
}

func TestCircuitBreaker_Integration(t *testing.T) {
    // Arrange
    cb := NewCircuitBreaker(3, time.Minute)
    
    // Act & Assert
    // Test circuit breaker behavior
    for i := 0; i < 5; i++ {
        err := cb.Execute(context.Background(), func() error {
            return fmt.Errorf("test error")
        })
        
        if i < 3 {
            assert.Error(t, err)
        } else {
            assert.Error(t, err)
            assert.Contains(t, err.Error(), "circuit breaker is open")
        }
    }
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Create a microservice
service := NewService("user-service", "1.0.0", 8080)
service.AddDependency("database", "postgres://localhost:5432", 5*time.Second, 3)

// 2. Start the service
ctx := context.Background()
service.Start(ctx)

// 3. Circuit breaker
cb := NewCircuitBreaker(5, time.Minute)
err := cb.Execute(ctx, func() error {
    return callExternalService()
})

// 4. Load balancer
lb := NewLoadBalancer(StrategyRoundRobin)
lb.AddInstance(ServiceInstance{ID: "1", Address: "localhost:8081"})
instance, err := lb.GetInstance()
```

### Essential Patterns

```go
// Service discovery
registry := NewServiceRegistry(30 * time.Second)
registry.Register(ServiceInstance{
    ID:      "user-service-1",
    Name:    "user-service",
    Address: "localhost:8080",
    Port:    8080,
})

// Event-driven communication
eventBus := NewEventBus()
eventBus.Subscribe("user.created", &UserCreatedHandler{})
eventBus.Publish(ctx, Event{
    Type: "user.created",
    Data: userData,
})

// Distributed tracing
tracer := NewTracer("user-service")
ctx, span := tracer.StartSpan(ctx, "create-user")
defer span.Finish()
```

---

*This guide provides the complete machinery for building production-ready microservices in Go applications. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*
