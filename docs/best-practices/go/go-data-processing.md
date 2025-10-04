# Go Data Processing Best Practices

**Objective**: Master senior-level Go data processing patterns for production systems. When you need to build efficient data pipelines, when you want to process large datasets, when you need enterprise-grade data processing patterns—these best practices become your weapon of choice.

## Core Principles

- **Stream Processing**: Process data as it arrives
- **Batch Processing**: Process data in efficient batches
- **Memory Efficiency**: Minimize memory usage for large datasets
- **Parallel Processing**: Leverage concurrency for performance
- **Fault Tolerance**: Handle failures gracefully

## Stream Processing

### Stream Processor

```go
// internal/streaming/stream_processor.go
package streaming

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// StreamProcessor represents a stream processor
type StreamProcessor struct {
    input     <-chan Data
    output    chan ProcessedData
    processors []Processor
    wg        sync.WaitGroup
    ctx       context.Context
    cancel    context.CancelFunc
}

// Data represents input data
type Data struct {
    ID        string
    Timestamp time.Time
    Payload   interface{}
    Metadata  map[string]interface{}
}

// ProcessedData represents processed data
type ProcessedData struct {
    ID        string
    Timestamp time.Time
    Result    interface{}
    Metadata  map[string]interface{}
    Duration  time.Duration
}

// Processor represents a data processor
type Processor interface {
    Process(ctx context.Context, data Data) (ProcessedData, error)
    Name() string
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(input <-chan Data, bufferSize int) *StreamProcessor {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &StreamProcessor{
        input:     input,
        output:    make(chan ProcessedData, bufferSize),
        processors: make([]Processor, 0),
        ctx:       ctx,
        cancel:    cancel,
    }
}

// AddProcessor adds a processor to the stream
func (sp *StreamProcessor) AddProcessor(processor Processor) {
    sp.processors = append(sp.processors, processor)
}

// Start starts the stream processor
func (sp *StreamProcessor) Start() {
    for _, processor := range sp.processors {
        sp.wg.Add(1)
        go sp.runProcessor(processor)
    }
}

// runProcessor runs a processor
func (sp *StreamProcessor) runProcessor(processor Processor) {
    defer sp.wg.Done()
    
    for {
        select {
        case data := <-sp.input:
            start := time.Now()
            
            result, err := processor.Process(sp.ctx, data)
            if err != nil {
                // Handle error
                continue
            }
            
            result.Duration = time.Since(start)
            sp.output <- result
            
        case <-sp.ctx.Done():
            return
        }
    }
}

// GetOutput returns the output channel
func (sp *StreamProcessor) GetOutput() <-chan ProcessedData {
    return sp.output
}

// Stop stops the stream processor
func (sp *StreamProcessor) Stop() {
    sp.cancel()
    sp.wg.Wait()
    close(sp.output)
}

// Example processor implementations
type ValidationProcessor struct{}

func (vp *ValidationProcessor) Name() string {
    return "validation"
}

func (vp *ValidationProcessor) Process(ctx context.Context, data Data) (ProcessedData, error) {
    // Implement validation logic
    return ProcessedData{
        ID:        data.ID,
        Timestamp: data.Timestamp,
        Result:    "validated",
        Metadata:  data.Metadata,
    }, nil
}

type TransformationProcessor struct{}

func (tp *TransformationProcessor) Name() string {
    return "transformation"
}

func (tp *TransformationProcessor) Process(ctx context.Context, data Data) (ProcessedData, error) {
    // Implement transformation logic
    return ProcessedData{
        ID:        data.ID,
        Timestamp: data.Timestamp,
        Result:    "transformed",
        Metadata:  data.Metadata,
    }, nil
}
```

### Event Stream Processing

```go
// internal/streaming/event_stream.go
package streaming

import (
    "context"
    "encoding/json"
    "sync"
    "time"
)

// EventStream represents an event stream
type EventStream struct {
    events    chan Event
    handlers  map[string][]EventHandler
    mutex     sync.RWMutex
    ctx       context.Context
    cancel    context.CancelFunc
}

// Event represents an event
type Event struct {
    ID        string                 `json:"id"`
    Type      string                 `json:"type"`
    Source    string                 `json:"source"`
    Timestamp time.Time              `json:"timestamp"`
    Data      interface{}            `json:"data"`
    Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// EventHandler represents an event handler
type EventHandler interface {
    Handle(ctx context.Context, event Event) error
    CanHandle(eventType string) bool
}

// NewEventStream creates a new event stream
func NewEventStream(bufferSize int) *EventStream {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &EventStream{
        events:   make(chan Event, bufferSize),
        handlers: make(map[string][]EventHandler),
        ctx:      ctx,
        cancel:   cancel,
    }
}

// Publish publishes an event to the stream
func (es *EventStream) Publish(event Event) error {
    select {
    case es.events <- event:
        return nil
    case <-es.ctx.Done():
        return context.Canceled
    default:
        return fmt.Errorf("event stream is full")
    }
}

// Subscribe subscribes to events of a specific type
func (es *EventStream) Subscribe(eventType string, handler EventHandler) {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    es.handlers[eventType] = append(es.handlers[eventType], handler)
}

// Start starts the event stream
func (es *EventStream) Start() {
    go es.processEvents()
}

// processEvents processes events from the stream
func (es *EventStream) processEvents() {
    for {
        select {
        case event := <-es.events:
            es.handleEvent(event)
        case <-es.ctx.Done():
            return
        }
    }
}

// handleEvent handles a single event
func (es *EventStream) handleEvent(event Event) {
    es.mutex.RLock()
    handlers := es.handlers[event.Type]
    es.mutex.RUnlock()
    
    for _, handler := range handlers {
        if handler.CanHandle(event.Type) {
            go func(h EventHandler) {
                if err := h.Handle(es.ctx, event); err != nil {
                    // Handle error
                }
            }(handler)
        }
    }
}

// Stop stops the event stream
func (es *EventStream) Stop() {
    es.cancel()
    close(es.events)
}

// Example event handlers
type UserEventHandler struct{}

func (ueh *UserEventHandler) CanHandle(eventType string) bool {
    return eventType == "user.created" || eventType == "user.updated"
}

func (ueh *UserEventHandler) Handle(ctx context.Context, event Event) error {
    // Implement user event handling
    return nil
}

type OrderEventHandler struct{}

func (oeh *OrderEventHandler) CanHandle(eventType string) bool {
    return eventType == "order.created" || eventType == "order.updated"
}

func (oeh *OrderEventHandler) Handle(ctx context.Context, event Event) error {
    // Implement order event handling
    return nil
}
```

## Batch Processing

### Batch Processor

```go
// internal/batch/batch_processor.go
package batch

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// BatchProcessor represents a batch processor
type BatchProcessor struct {
    batchSize    int
    flushTimeout time.Duration
    input        <-chan Data
    output       chan []ProcessedData
    buffer       []Data
    mutex        sync.Mutex
    ctx          context.Context
    cancel       context.CancelFunc
    processor    Processor
}

// Processor represents a batch processor
type Processor interface {
    ProcessBatch(ctx context.Context, batch []Data) ([]ProcessedData, error)
    Name() string
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(batchSize int, flushTimeout time.Duration, input <-chan Data, processor Processor) *BatchProcessor {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &BatchProcessor{
        batchSize:    batchSize,
        flushTimeout: flushTimeout,
        input:        input,
        output:       make(chan []ProcessedData, 10),
        buffer:       make([]Data, 0, batchSize),
        ctx:          ctx,
        cancel:       cancel,
        processor:    processor,
    }
}

// Start starts the batch processor
func (bp *BatchProcessor) Start() {
    go bp.processBatches()
}

// processBatches processes data in batches
func (bp *BatchProcessor) processBatches() {
    ticker := time.NewTicker(bp.flushTimeout)
    defer ticker.Stop()
    
    for {
        select {
        case data := <-bp.input:
            bp.addToBuffer(data)
            if bp.shouldFlush() {
                bp.flushBatch()
            }
            
        case <-ticker.C:
            bp.flushBatch()
            
        case <-bp.ctx.Done():
            bp.flushBatch()
            return
        }
    }
}

// addToBuffer adds data to the buffer
func (bp *BatchProcessor) addToBuffer(data Data) {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()
    
    bp.buffer = append(bp.buffer, data)
}

// shouldFlush checks if the buffer should be flushed
func (bp *BatchProcessor) shouldFlush() bool {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()
    
    return len(bp.buffer) >= bp.batchSize
}

// flushBatch flushes the current batch
func (bp *BatchProcessor) flushBatch() {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()
    
    if len(bp.buffer) == 0 {
        return
    }
    
    // Create a copy of the buffer
    batch := make([]Data, len(bp.buffer))
    copy(batch, bp.buffer)
    
    // Clear the buffer
    bp.buffer = bp.buffer[:0]
    
    // Process the batch
    go func() {
        result, err := bp.processor.ProcessBatch(bp.ctx, batch)
        if err != nil {
            // Handle error
            return
        }
        
        bp.output <- result
    }()
}

// GetOutput returns the output channel
func (bp *BatchProcessor) GetOutput() <-chan []ProcessedData {
    return bp.output
}

// Stop stops the batch processor
func (bp *BatchProcessor) Stop() {
    bp.cancel()
    close(bp.output)
}

// Example batch processor implementations
type DataValidationProcessor struct{}

func (dvp *DataValidationProcessor) Name() string {
    return "data-validation"
}

func (dvp *DataValidationProcessor) ProcessBatch(ctx context.Context, batch []Data) ([]ProcessedData, error) {
    results := make([]ProcessedData, 0, len(batch))
    
    for _, data := range batch {
        // Implement validation logic
        result := ProcessedData{
            ID:        data.ID,
            Timestamp: data.Timestamp,
            Result:    "validated",
            Metadata:  data.Metadata,
        }
        results = append(results, result)
    }
    
    return results, nil
}

type DataTransformationProcessor struct{}

func (dtp *DataTransformationProcessor) Name() string {
    return "data-transformation"
}

func (dtp *DataTransformationProcessor) ProcessBatch(ctx context.Context, batch []Data) ([]ProcessedData, error) {
    results := make([]ProcessedData, 0, len(batch))
    
    for _, data := range batch {
        // Implement transformation logic
        result := ProcessedData{
            ID:        data.ID,
            Timestamp: data.Timestamp,
            Result:    "transformed",
            Metadata:  data.Metadata,
        }
        results = append(results, result)
    }
    
    return results, nil
}
```

## Data Pipeline

### Pipeline Builder

```go
// internal/pipeline/pipeline_builder.go
package pipeline

import (
    "context"
    "fmt"
    "sync"
)

// Pipeline represents a data processing pipeline
type Pipeline struct {
    stages []Stage
    input  <-chan Data
    output chan ProcessedData
    ctx    context.Context
    cancel context.CancelFunc
    wg     sync.WaitGroup
}

// Stage represents a pipeline stage
type Stage interface {
    Process(ctx context.Context, input <-chan Data) <-chan ProcessedData
    Name() string
}

// NewPipeline creates a new pipeline
func NewPipeline(input <-chan Data) *Pipeline {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &Pipeline{
        stages: make([]Stage, 0),
        input:  input,
        output: make(chan ProcessedData, 100),
        ctx:    ctx,
        cancel: cancel,
    }
}

// AddStage adds a stage to the pipeline
func (p *Pipeline) AddStage(stage Stage) {
    p.stages = append(p.stages, stage)
}

// Start starts the pipeline
func (p *Pipeline) Start() {
    current := p.input
    
    for i, stage := range p.stages {
        p.wg.Add(1)
        
        stageOutput := stage.Process(p.ctx, current)
        
        if i == len(p.stages)-1 {
            // Last stage, connect to output
            go p.forwardToOutput(stageOutput)
        } else {
            // Intermediate stage, connect to next stage
            current = stageOutput
        }
    }
}

// forwardToOutput forwards data to the output channel
func (p *Pipeline) forwardToOutput(input <-chan ProcessedData) {
    defer p.wg.Done()
    defer close(p.output)
    
    for {
        select {
        case data, ok := <-input:
            if !ok {
                return
            }
            p.output <- data
            
        case <-p.ctx.Done():
            return
        }
    }
}

// GetOutput returns the output channel
func (p *Pipeline) GetOutput() <-chan ProcessedData {
    return p.output
}

// Stop stops the pipeline
func (p *Pipeline) Stop() {
    p.cancel()
    p.wg.Wait()
}

// Example stage implementations
type ValidationStage struct{}

func (vs *ValidationStage) Name() string {
    return "validation"
}

func (vs *ValidationStage) Process(ctx context.Context, input <-chan Data) <-chan ProcessedData {
    output := make(chan ProcessedData, 100)
    
    go func() {
        defer close(output)
        
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                
                // Implement validation logic
                result := ProcessedData{
                    ID:        data.ID,
                    Timestamp: data.Timestamp,
                    Result:    "validated",
                    Metadata:  data.Metadata,
                }
                
                select {
                case output <- result:
                case <-ctx.Done():
                    return
                }
                
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

type TransformationStage struct{}

func (ts *TransformationStage) Name() string {
    return "transformation"
}

func (ts *TransformationStage) Process(ctx context.Context, input <-chan Data) <-chan ProcessedData {
    output := make(chan ProcessedData, 100)
    
    go func() {
        defer close(output)
        
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                
                // Implement transformation logic
                result := ProcessedData{
                    ID:        data.ID,
                    Timestamp: data.Timestamp,
                    Result:    "transformed",
                    Metadata:  data.Metadata,
                }
                
                select {
                case output <- result:
                case <-ctx.Done():
                    return
                }
                
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}
```

## Data Aggregation

### Aggregator

```go
// internal/aggregation/aggregator.go
package aggregation

import (
    "context"
    "sync"
    "time"
)

// Aggregator represents a data aggregator
type Aggregator struct {
    windowSize    time.Duration
    input         <-chan Data
    output        chan AggregatedData
    buffer        map[string][]Data
    mutex         sync.RWMutex
    ctx           context.Context
    cancel        context.CancelFunc
    aggregators   map[string]AggregationFunction
}

// AggregatedData represents aggregated data
type AggregatedData struct {
    Key       string                 `json:"key"`
    Timestamp time.Time              `json:"timestamp"`
    Count     int                    `json:"count"`
    Sum       float64                `json:"sum"`
    Average   float64                `json:"average"`
    Min       float64                `json:"min"`
    Max       float64                `json:"max"`
    Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// AggregationFunction represents an aggregation function
type AggregationFunction func([]Data) AggregatedData

// NewAggregator creates a new aggregator
func NewAggregator(windowSize time.Duration, input <-chan Data) *Aggregator {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &Aggregator{
        windowSize:  windowSize,
        input:       input,
        output:      make(chan AggregatedData, 100),
        buffer:      make(map[string][]Data),
        ctx:         ctx,
        cancel:      cancel,
        aggregators: make(map[string]AggregationFunction),
    }
}

// AddAggregator adds an aggregation function
func (a *Aggregator) AddAggregator(key string, fn AggregationFunction) {
    a.aggregators[key] = fn
}

// Start starts the aggregator
func (a *Aggregator) Start() {
    go a.processData()
    go a.flushPeriodically()
}

// processData processes incoming data
func (a *Aggregator) processData() {
    for {
        select {
        case data := <-a.input:
            a.addToBuffer(data)
            
        case <-a.ctx.Done():
            return
        }
    }
}

// addToBuffer adds data to the buffer
func (a *Aggregator) addToBuffer(data Data) {
    a.mutex.Lock()
    defer a.mutex.Unlock()
    
    key := a.getKey(data)
    a.buffer[key] = append(a.buffer[key], data)
}

// getKey gets the aggregation key for data
func (a *Aggregator) getKey(data Data) string {
    // Implement key generation logic
    return "default"
}

// flushPeriodically flushes data periodically
func (a *Aggregator) flushPeriodically() {
    ticker := time.NewTicker(a.windowSize)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            a.flush()
            
        case <-a.ctx.Done():
            a.flush()
            return
        }
    }
}

// flush flushes the current buffer
func (a *Aggregator) flush() {
    a.mutex.Lock()
    defer a.mutex.Unlock()
    
    for key, data := range a.buffer {
        if len(data) == 0 {
            continue
        }
        
        if fn, exists := a.aggregators[key]; exists {
            result := fn(data)
            a.output <- result
        }
        
        // Clear the buffer
        a.buffer[key] = a.buffer[key][:0]
    }
}

// GetOutput returns the output channel
func (a *Aggregator) GetOutput() <-chan AggregatedData {
    return a.output
}

// Stop stops the aggregator
func (a *Aggregator) Stop() {
    a.cancel()
    close(a.output)
}

// Example aggregation functions
func CountAggregation(data []Data) AggregatedData {
    return AggregatedData{
        Key:       "count",
        Timestamp: time.Now(),
        Count:     len(data),
    }
}

func SumAggregation(data []Data) AggregatedData {
    sum := 0.0
    for _, d := range data {
        if val, ok := d.Payload.(float64); ok {
            sum += val
        }
    }
    
    return AggregatedData{
        Key:       "sum",
        Timestamp: time.Now(),
        Sum:       sum,
        Count:     len(data),
    }
}

func AverageAggregation(data []Data) AggregatedData {
    sum := 0.0
    count := 0
    
    for _, d := range data {
        if val, ok := d.Payload.(float64); ok {
            sum += val
            count++
        }
    }
    
    average := 0.0
    if count > 0 {
        average = sum / float64(count)
    }
    
    return AggregatedData{
        Key:       "average",
        Timestamp: time.Now(),
        Sum:       sum,
        Count:     count,
        Average:   average,
    }
}
```

## Data Validation

### Validator

```go
// internal/validation/validator.go
package validation

import (
    "context"
    "fmt"
    "reflect"
    "regexp"
    "strings"
    "time"
)

// Validator represents a data validator
type Validator struct {
    rules map[string][]ValidationRule
}

// ValidationRule represents a validation rule
type ValidationRule interface {
    Validate(value interface{}) error
    Name() string
}

// NewValidator creates a new validator
func NewValidator() *Validator {
    return &Validator{
        rules: make(map[string][]ValidationRule),
    }
}

// AddRule adds a validation rule for a field
func (v *Validator) AddRule(field string, rule ValidationRule) {
    v.rules[field] = append(v.rules[field], rule)
}

// Validate validates data against rules
func (v *Validator) Validate(data map[string]interface{}) []ValidationError {
    var errors []ValidationError
    
    for field, rules := range v.rules {
        value, exists := data[field]
        if !exists {
            continue
        }
        
        for _, rule := range rules {
            if err := rule.Validate(value); err != nil {
                errors = append(errors, ValidationError{
                    Field: field,
                    Rule:  rule.Name(),
                    Error: err.Error(),
                })
            }
        }
    }
    
    return errors
}

// ValidationError represents a validation error
type ValidationError struct {
    Field string `json:"field"`
    Rule  string `json:"rule"`
    Error string `json:"error"`
}

// Example validation rules
type RequiredRule struct{}

func (rr *RequiredRule) Name() string {
    return "required"
}

func (rr *RequiredRule) Validate(value interface{}) error {
    if value == nil {
        return fmt.Errorf("field is required")
    }
    
    if str, ok := value.(string); ok && strings.TrimSpace(str) == "" {
        return fmt.Errorf("field cannot be empty")
    }
    
    return nil
}

type StringLengthRule struct {
    Min, Max int
}

func (slr *StringLengthRule) Name() string {
    return "string_length"
}

func (slr *StringLengthRule) Validate(value interface{}) error {
    str, ok := value.(string)
    if !ok {
        return fmt.Errorf("value must be a string")
    }
    
    length := len(str)
    if length < slr.Min {
        return fmt.Errorf("string must be at least %d characters", slr.Min)
    }
    
    if length > slr.Max {
        return fmt.Errorf("string must be at most %d characters", slr.Max)
    }
    
    return nil
}

type EmailRule struct{}

func (er *EmailRule) Name() string {
    return "email"
}

func (er *EmailRule) Validate(value interface{}) error {
    str, ok := value.(string)
    if !ok {
        return fmt.Errorf("value must be a string")
    }
    
    pattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
    matched, err := regexp.MatchString(pattern, str)
    if err != nil {
        return err
    }
    
    if !matched {
        return fmt.Errorf("invalid email format")
    }
    
    return nil
}

type NumericRangeRule struct {
    Min, Max float64
}

func (nrr *NumericRangeRule) Name() string {
    return "numeric_range"
}

func (nrr *NumericRangeRule) Validate(value interface{}) error {
    var num float64
    
    switch v := value.(type) {
    case int:
        num = float64(v)
    case int64:
        num = float64(v)
    case float32:
        num = float64(v)
    case float64:
        num = v
    default:
        return fmt.Errorf("value must be numeric")
    }
    
    if num < nrr.Min {
        return fmt.Errorf("value must be at least %f", nrr.Min)
    }
    
    if num > nrr.Max {
        return fmt.Errorf("value must be at most %f", nrr.Max)
    }
    
    return nil
}
```

## Data Transformation

### Transformer

```go
// internal/transformation/transformer.go
package transformation

import (
    "context"
    "encoding/json"
    "fmt"
    "reflect"
    "strings"
)

// Transformer represents a data transformer
type Transformer struct {
    rules map[string]TransformationRule
}

// TransformationRule represents a transformation rule
type TransformationRule interface {
    Transform(value interface{}) (interface{}, error)
    Name() string
}

// NewTransformer creates a new transformer
func NewTransformer() *Transformer {
    return &Transformer{
        rules: make(map[string]TransformationRule),
    }
}

// AddRule adds a transformation rule for a field
func (t *Transformer) AddRule(field string, rule TransformationRule) {
    t.rules[field] = append(t.rules[field], rule)
}

// Transform transforms data according to rules
func (t *Transformer) Transform(data map[string]interface{}) (map[string]interface{}, error) {
    result := make(map[string]interface{})
    
    for key, value := range data {
        if rules, exists := t.rules[key]; exists {
            transformed := value
            
            for _, rule := range rules {
                var err error
                transformed, err = rule.Transform(transformed)
                if err != nil {
                    return nil, fmt.Errorf("transformation failed for field %s: %w", key, err)
                }
            }
            
            result[key] = transformed
        } else {
            result[key] = value
        }
    }
    
    return result, nil
}

// Example transformation rules
type ToLowerCaseRule struct{}

func (tlcr *ToLowerCaseRule) Name() string {
    return "to_lowercase"
}

func (tlcr *ToLowerCaseRule) Transform(value interface{}) (interface{}, error) {
    str, ok := value.(string)
    if !ok {
        return value, nil
    }
    
    return strings.ToLower(str), nil
}

type ToUpperCaseRule struct{}

func (tucr *ToUpperCaseRule) Name() string {
    return "to_uppercase"
}

func (tucr *ToUpperCaseRule) Transform(value interface{}) (interface{}, error) {
    str, ok := value.(string)
    if !ok {
        return value, nil
    }
    
    return strings.ToUpper(str), nil
}

type TrimRule struct{}

func (tr *TrimRule) Name() string {
    return "trim"
}

func (tr *TrimRule) Transform(value interface{}) (interface{}, error) {
    str, ok := value.(string)
    if !ok {
        return value, nil
    }
    
    return strings.TrimSpace(str), nil
}

type MultiplyRule struct {
    Factor float64
}

func (mr *MultiplyRule) Name() string {
    return "multiply"
}

func (mr *MultiplyRule) Transform(value interface{}) (interface{}, error) {
    var num float64
    
    switch v := value.(type) {
    case int:
        num = float64(v)
    case int64:
        num = float64(v)
    case float32:
        num = float64(v)
    case float64:
        num = v
    default:
        return value, nil
    }
    
    return num * mr.Factor, nil
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Stream processing
processor := NewStreamProcessor(input, 100)
processor.AddProcessor(&ValidationProcessor{})
processor.AddProcessor(&TransformationProcessor{})
processor.Start()

// 2. Batch processing
batchProcessor := NewBatchProcessor(100, 5*time.Second, input, &DataValidationProcessor{})
batchProcessor.Start()

// 3. Data pipeline
pipeline := NewPipeline(input)
pipeline.AddStage(&ValidationStage{})
pipeline.AddStage(&TransformationStage{})
pipeline.Start()

// 4. Data aggregation
aggregator := NewAggregator(1*time.Minute, input)
aggregator.AddAggregator("count", CountAggregation)
aggregator.AddAggregator("sum", SumAggregation)
aggregator.Start()
```

### Essential Patterns

```go
// Data validation
validator := NewValidator()
validator.AddRule("email", &EmailRule{})
validator.AddRule("age", &NumericRangeRule{Min: 0, Max: 120})
errors := validator.Validate(data)

// Data transformation
transformer := NewTransformer()
transformer.AddRule("name", &ToLowerCaseRule{})
transformer.AddRule("price", &MultiplyRule{Factor: 1.2})
result, err := transformer.Transform(data)

// Event streaming
eventStream := NewEventStream(1000)
eventStream.Subscribe("user.created", &UserEventHandler{})
eventStream.Start()
```

---

*This guide provides the complete machinery for building efficient data processing pipelines in Go applications. Each pattern includes implementation examples, performance considerations, and real-world usage patterns for enterprise deployment.*
