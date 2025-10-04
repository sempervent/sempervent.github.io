# Go Concurrency Patterns Best Practices

**Objective**: Master senior-level Go concurrency patterns for production systems. When you need to build robust, scalable concurrent applications, when you want to follow proven methodologies, when you need enterprise-grade concurrency patterns—these best practices become your weapon of choice.

## Core Principles

- **Communicate by Sharing Memory**: Use channels for communication, not shared memory
- **Don't Share Memory**: Avoid shared state, use message passing
- **Goroutine Lifecycle Management**: Proper creation, management, and cleanup
- **Context-Driven Cancellation**: Use context for graceful shutdown
- **Backpressure Handling**: Manage load and prevent resource exhaustion

## Goroutine Patterns

### Basic Goroutine Management

```go
// internal/concurrency/worker.go
package concurrency

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// WorkerPool manages a pool of workers
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultChan chan Result
    wg         sync.WaitGroup
    ctx        context.Context
    cancel     context.CancelFunc
}

// Job represents a unit of work
type Job struct {
    ID   string
    Data interface{}
}

// Result represents the result of a job
type Result struct {
    JobID string
    Data  interface{}
    Error error
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workers int, bufferSize int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &WorkerPool{
        workers:    workers,
        jobQueue:   make(chan Job, bufferSize),
        resultChan: make(chan Result, bufferSize),
        ctx:        ctx,
        cancel:     cancel,
    }
}

// Start begins processing jobs
func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

// worker processes jobs from the queue
func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()
    
    for {
        select {
        case job := <-wp.jobQueue:
            result := wp.processJob(job)
            wp.resultChan <- result
        case <-wp.ctx.Done():
            return
        }
    }
}

// processJob handles individual job processing
func (wp *WorkerPool) processJob(job Job) Result {
    // Simulate work
    time.Sleep(100 * time.Millisecond)
    
    return Result{
        JobID: job.ID,
        Data:  fmt.Sprintf("Processed: %v", job.Data),
        Error: nil,
    }
}

// Submit adds a job to the queue
func (wp *WorkerPool) Submit(job Job) error {
    select {
    case wp.jobQueue <- job:
        return nil
    case <-wp.ctx.Done():
        return fmt.Errorf("worker pool is shutting down")
    }
}

// Results returns the result channel
func (wp *WorkerPool) Results() <-chan Result {
    return wp.resultChan
}

// Stop gracefully shuts down the worker pool
func (wp *WorkerPool) Stop() {
    wp.cancel()
    close(wp.jobQueue)
    wp.wg.Wait()
    close(wp.resultChan)
}
```

### Fan-Out/Fan-In Pattern

```go
// internal/concurrency/fanout.go
package concurrency

import (
    "context"
    "sync"
)

// FanOut distributes work across multiple workers
func FanOut(ctx context.Context, input <-chan Job, workers int) []<-chan Result {
    outputs := make([]<-chan Result, workers)
    
    for i := 0; i < workers; i++ {
        output := make(chan Result)
        outputs[i] = output
        
        go func(out chan<- Result) {
            defer close(out)
            
            for {
                select {
                case job, ok := <-input:
                    if !ok {
                        return
                    }
                    result := processJob(job)
                    out <- result
                case <-ctx.Done():
                    return
                }
            }
        }(output)
    }
    
    return outputs
}

// FanIn combines multiple channels into one
func FanIn(ctx context.Context, inputs ...<-chan Result) <-chan Result {
    output := make(chan Result)
    var wg sync.WaitGroup
    
    for _, input := range inputs {
        wg.Add(1)
        go func(in <-chan Result) {
            defer wg.Done()
            
            for result := range in {
                select {
                case output <- result:
                case <-ctx.Done():
                    return
                }
            }
        }(input)
    }
    
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}

// processJob handles job processing
func processJob(job Job) Result {
    // Implementation here
    return Result{JobID: job.ID, Data: job.Data, Error: nil}
}
```

## Channel Patterns

### Pipeline Pattern

```go
// internal/concurrency/pipeline.go
package concurrency

import (
    "context"
    "fmt"
)

// Pipeline represents a processing pipeline
type Pipeline struct {
    stages []Stage
}

// Stage represents a pipeline stage
type Stage interface {
    Process(ctx context.Context, input <-chan interface{}) <-chan interface{}
}

// TransformStage transforms data
type TransformStage struct {
    transform func(interface{}) interface{}
}

// Process implements the Stage interface
func (ts *TransformStage) Process(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output)
        
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                transformed := ts.transform(data)
                output <- transformed
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

// FilterStage filters data
type FilterStage struct {
    predicate func(interface{}) bool
}

// Process implements the Stage interface
func (fs *FilterStage) Process(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output)
        
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                if fs.predicate(data) {
                    output <- data
                }
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

// NewPipeline creates a new pipeline
func NewPipeline(stages ...Stage) *Pipeline {
    return &Pipeline{stages: stages}
}

// Execute runs the pipeline
func (p *Pipeline) Execute(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    current := input
    
    for _, stage := range p.stages {
        current = stage.Process(ctx, current)
    }
    
    return current
}
```

### Timeout and Cancellation

```go
// internal/concurrency/timeout.go
package concurrency

import (
    "context"
    "time"
)

// WithTimeout creates a context with timeout
func WithTimeout(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
    return context.WithTimeout(parent, timeout)
}

// WithDeadline creates a context with deadline
func WithDeadline(parent context.Context, deadline time.Time) (context.Context, context.CancelFunc) {
    return context.WithDeadline(parent, deadline)
}

// DoWithTimeout executes a function with timeout
func DoWithTimeout(ctx context.Context, fn func() error, timeout time.Duration) error {
    ctx, cancel := WithTimeout(ctx, timeout)
    defer cancel()
    
    done := make(chan error, 1)
    
    go func() {
        done <- fn()
    }()
    
    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        return ctx.Err()
    }
}

// RetryWithBackoff retries a function with exponential backoff
func RetryWithBackoff(ctx context.Context, fn func() error, maxRetries int, initialDelay time.Duration) error {
    var err error
    
    for i := 0; i < maxRetries; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        err = fn()
        if err == nil {
            return nil
        }
        
        if i < maxRetries-1 {
            delay := time.Duration(1<<uint(i)) * initialDelay
            time.Sleep(delay)
        }
    }
    
    return err
}
```

## Synchronization Patterns

### Mutex Patterns

```go
// internal/concurrency/sync.go
package concurrency

import (
    "sync"
    "time"
)

// SafeCounter is a thread-safe counter
type SafeCounter struct {
    mu    sync.RWMutex
    count int
}

// Increment safely increments the counter
func (sc *SafeCounter) Increment() {
    sc.mu.Lock()
    defer sc.mu.Unlock()
    sc.count++
}

// Get safely gets the counter value
func (sc *SafeCounter) Get() int {
    sc.mu.RLock()
    defer sc.mu.RUnlock()
    return sc.count
}

// SafeMap is a thread-safe map
type SafeMap struct {
    mu sync.RWMutex
    m  map[string]interface{}
}

// NewSafeMap creates a new thread-safe map
func NewSafeMap() *SafeMap {
    return &SafeMap{
        m: make(map[string]interface{}),
    }
}

// Set safely sets a value
func (sm *SafeMap) Set(key string, value interface{}) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    sm.m[key] = value
}

// Get safely gets a value
func (sm *SafeMap) Get(key string) (interface{}, bool) {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    value, ok := sm.m[key]
    return value, ok
}

// Delete safely deletes a value
func (sm *SafeMap) Delete(key string) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    delete(sm.m, key)
}
```

### Once Pattern

```go
// internal/concurrency/once.go
package concurrency

import (
    "sync"
    "time"
)

// LazyInitializer provides lazy initialization
type LazyInitializer struct {
    once sync.Once
    data interface{}
    err  error
}

// Get initializes and returns data
func (li *LazyInitializer) Get(initFunc func() (interface{}, error)) (interface{}, error) {
    li.once.Do(func() {
        li.data, li.err = initFunc()
    })
    return li.data, li.err
}

// Singleton provides singleton pattern
type Singleton struct {
    once sync.Once
    instance *Singleton
}

// GetInstance returns the singleton instance
func (s *Singleton) GetInstance() *Singleton {
    s.once.Do(func() {
        s.instance = &Singleton{}
    })
    return s.instance
}
```

## Advanced Patterns

### Circuit Breaker

```go
// internal/concurrency/circuitbreaker.go
package concurrency

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
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        state:     StateClosed,
        threshold: threshold,
        timeout:   timeout,
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
        return true
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
        cb.state = StateClosed
    }
}
```

### Rate Limiter

```go
// internal/concurrency/ratelimiter.go
package concurrency

import (
    "context"
    "sync"
    "time"
)

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
    mu       sync.Mutex
    tokens   int
    capacity int
    rate     time.Duration
    lastTime time.Time
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(capacity int, rate time.Duration) *RateLimiter {
    return &RateLimiter{
        tokens:   capacity,
        capacity: capacity,
        rate:     rate,
        lastTime: time.Now(),
    }
}

// Allow checks if a request is allowed
func (rl *RateLimiter) Allow() bool {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    now := time.Now()
    elapsed := now.Sub(rl.lastTime)
    
    // Add tokens based on elapsed time
    tokensToAdd := int(elapsed / rl.rate)
    if tokensToAdd > 0 {
        rl.tokens = min(rl.capacity, rl.tokens+tokensToAdd)
        rl.lastTime = now
    }
    
    if rl.tokens > 0 {
        rl.tokens--
        return true
    }
    
    return false
}

// Wait waits for a token to become available
func (rl *RateLimiter) Wait(ctx context.Context) error {
    for {
        if rl.Allow() {
            return nil
        }
        
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(rl.rate):
            // Try again
        }
    }
}

// min returns the minimum of two integers
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

## Error Handling in Concurrency

```go
// internal/concurrency/errors.go
package concurrency

import (
    "context"
    "errors"
    "fmt"
    "sync"
)

// ConcurrencyError represents a concurrency-related error
type ConcurrencyError struct {
    Op  string
    Err error
}

// Error implements the error interface
func (ce *ConcurrencyError) Error() string {
    return fmt.Sprintf("concurrency error in %s: %v", ce.Op, ce.Err)
}

// Unwrap returns the underlying error
func (ce *ConcurrencyError) Unwrap() error {
    return ce.Err
}

// ErrorGroup manages a group of goroutines with error handling
type ErrorGroup struct {
    wg     sync.WaitGroup
    errs   []error
    mu     sync.Mutex
    ctx    context.Context
    cancel context.CancelFunc
}

// NewErrorGroup creates a new error group
func NewErrorGroup(ctx context.Context) *ErrorGroup {
    ctx, cancel := context.WithCancel(ctx)
    return &ErrorGroup{
        ctx:    ctx,
        cancel: cancel,
    }
}

// Go runs a function in a goroutine
func (eg *ErrorGroup) Go(fn func() error) {
    eg.wg.Add(1)
    
    go func() {
        defer eg.wg.Done()
        
        if err := fn(); err != nil {
            eg.mu.Lock()
            eg.errs = append(eg.errs, err)
            eg.mu.Unlock()
            eg.cancel()
        }
    }()
}

// Wait waits for all goroutines to complete
func (eg *ErrorGroup) Wait() error {
    eg.wg.Wait()
    eg.cancel()
    
    if len(eg.errs) > 0 {
        return fmt.Errorf("multiple errors: %v", eg.errs)
    }
    
    return nil
}
```

## Testing Concurrency

```go
// internal/concurrency/concurrency_test.go
package concurrency

import (
    "context"
    "testing"
    "time"
)

func TestWorkerPool(t *testing.T) {
    pool := NewWorkerPool(3, 10)
    pool.Start()
    defer pool.Stop()
    
    // Submit jobs
    for i := 0; i < 5; i++ {
        job := Job{
            ID:   fmt.Sprintf("job-%d", i),
            Data: fmt.Sprintf("data-%d", i),
        }
        err := pool.Submit(job)
        if err != nil {
            t.Fatalf("Failed to submit job: %v", err)
        }
    }
    
    // Collect results
    results := make([]Result, 0, 5)
    for i := 0; i < 5; i++ {
        select {
        case result := <-pool.Results():
            results = append(results, result)
        case <-time.After(5 * time.Second):
            t.Fatal("Timeout waiting for results")
        }
    }
    
    if len(results) != 5 {
        t.Errorf("Expected 5 results, got %d", len(results))
    }
}

func TestCircuitBreaker(t *testing.T) {
    cb := NewCircuitBreaker(2, 100*time.Millisecond)
    
    // Test successful execution
    err := cb.Execute(context.Background(), func() error {
        return nil
    })
    if err != nil {
        t.Errorf("Expected no error, got %v", err)
    }
    
    // Test failure threshold
    for i := 0; i < 3; i++ {
        err := cb.Execute(context.Background(), func() error {
            return errors.New("test error")
        })
        if i < 2 && err == nil {
            t.Error("Expected error for failed execution")
        }
        if i >= 2 && err == nil {
            t.Error("Expected circuit breaker to be open")
        }
    }
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Basic goroutine
go func() {
    // Your code here
}()

// 2. Worker pool
pool := NewWorkerPool(4, 100)
pool.Start()
defer pool.Stop()

// 3. Pipeline
pipeline := NewPipeline(
    &TransformStage{transform: func(x interface{}) interface{} { return x }},
    &FilterStage{predicate: func(x interface{}) bool { return true }},
)

// 4. Circuit breaker
cb := NewCircuitBreaker(5, time.Minute)
err := cb.Execute(ctx, func() error { return nil })
```

### Essential Patterns

```go
// Context cancellation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// Fan-out/fan-in
outputs := FanOut(ctx, input, 4)
result := FanIn(ctx, outputs...)

// Rate limiting
limiter := NewRateLimiter(100, time.Second)
if !limiter.Allow() {
    return errors.New("rate limited")
}
```

---

*This guide provides the complete machinery for building production-ready concurrent Go applications. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*
