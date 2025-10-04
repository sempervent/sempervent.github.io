# Go Monitoring & Observability Best Practices

**Objective**: Master senior-level Go monitoring and observability patterns for production systems. When you need to build comprehensive monitoring solutions, when you want to implement distributed tracing, when you need enterprise-grade observability patterns—these best practices become your weapon of choice.

## Core Principles

- **Three Pillars**: Metrics, Logs, and Traces
- **Structured Logging**: Use structured, machine-readable logs
- **Context Propagation**: Maintain context across service boundaries
- **Performance Monitoring**: Track application performance and health
- **Alerting**: Implement intelligent alerting based on meaningful metrics

## Structured Logging

### Logging Framework

```go
// internal/logging/logger.go
package logging

import (
    "context"
    "encoding/json"
    "fmt"
    "os"
    "runtime"
    "time"
)

// LogLevel represents log levels
type LogLevel int

const (
    DebugLevel LogLevel = iota
    InfoLevel
    WarnLevel
    ErrorLevel
    FatalLevel
)

// String returns the string representation of the log level
func (ll LogLevel) String() string {
    switch ll {
    case DebugLevel:
        return "DEBUG"
    case InfoLevel:
        return "INFO"
    case WarnLevel:
        return "WARN"
    case ErrorLevel:
        return "ERROR"
    case FatalLevel:
        return "FATAL"
    default:
        return "UNKNOWN"
    }
}

// LogEntry represents a log entry
type LogEntry struct {
    Timestamp time.Time              `json:"timestamp"`
    Level     LogLevel               `json:"level"`
    Message   string                 `json:"message"`
    Fields    map[string]interface{} `json:"fields,omitempty"`
    Caller    string                 `json:"caller,omitempty"`
    TraceID   string                 `json:"trace_id,omitempty"`
    SpanID    string                 `json:"span_id,omitempty"`
}

// Logger represents a structured logger
type Logger struct {
    level  LogLevel
    fields map[string]interface{}
    output *os.File
}

// NewLogger creates a new logger
func NewLogger(level LogLevel) *Logger {
    return &Logger{
        level:  level,
        fields: make(map[string]interface{}),
        output: os.Stdout,
    }
}

// WithFields adds fields to the logger
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
    newFields := make(map[string]interface{})
    for k, v := range l.fields {
        newFields[k] = v
    }
    for k, v := range fields {
        newFields[k] = v
    }
    
    return &Logger{
        level:  l.level,
        fields: newFields,
        output: l.output,
    }
}

// WithField adds a single field to the logger
func (l *Logger) WithField(key string, value interface{}) *Logger {
    return l.WithFields(map[string]interface{}{key: value})
}

// WithContext adds context fields to the logger
func (l *Logger) WithContext(ctx context.Context) *Logger {
    fields := make(map[string]interface{})
    
    if traceID := ctx.Value("trace_id"); traceID != nil {
        fields["trace_id"] = traceID
    }
    
    if spanID := ctx.Value("span_id"); spanID != nil {
        fields["span_id"] = spanID
    }
    
    if userID := ctx.Value("user_id"); userID != nil {
        fields["user_id"] = userID
    }
    
    return l.WithFields(fields)
}

// log logs a message with the specified level
func (l *Logger) log(level LogLevel, msg string, fields map[string]interface{}) {
    if level < l.level {
        return
    }
    
    entry := LogEntry{
        Timestamp: time.Now().UTC(),
        Level:     level,
        Message:   msg,
        Fields:    l.mergeFields(fields),
    }
    
    // Add caller information
    if _, file, line, ok := runtime.Caller(3); ok {
        entry.Caller = fmt.Sprintf("%s:%d", file, line)
    }
    
    // Add context fields
    if traceID, ok := entry.Fields["trace_id"].(string); ok {
        entry.TraceID = traceID
    }
    if spanID, ok := entry.Fields["span_id"].(string); ok {
        entry.SpanID = spanID
    }
    
    // Write to output
    json.NewEncoder(l.output).Encode(entry)
}

// mergeFields merges logger fields with entry fields
func (l *Logger) mergeFields(fields map[string]interface{}) map[string]interface{} {
    merged := make(map[string]interface{})
    
    for k, v := range l.fields {
        merged[k] = v
    }
    
    for k, v := range fields {
        merged[k] = v
    }
    
    return merged
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, fields ...map[string]interface{}) {
    l.log(DebugLevel, msg, l.mergeFieldsList(fields))
}

// Info logs an info message
func (l *Logger) Info(msg string, fields ...map[string]interface{}) {
    l.log(InfoLevel, msg, l.mergeFieldsList(fields))
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, fields ...map[string]interface{}) {
    l.log(WarnLevel, msg, l.mergeFieldsList(fields))
}

// Error logs an error message
func (l *Logger) Error(msg string, fields ...map[string]interface{}) {
    l.log(ErrorLevel, msg, l.mergeFieldsList(fields))
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(msg string, fields ...map[string]interface{}) {
    l.log(FatalLevel, msg, l.mergeFieldsList(fields))
    os.Exit(1)
}

// mergeFieldsList merges a list of field maps
func (l *Logger) mergeFieldsList(fields []map[string]interface{}) map[string]interface{} {
    merged := make(map[string]interface{})
    
    for _, fieldMap := range fields {
        for k, v := range fieldMap {
            merged[k] = v
        }
    }
    
    return merged
}
```

### Context-Aware Logging

```go
// internal/logging/context_logger.go
package logging

import (
    "context"
    "fmt"
    "time"
)

// ContextLogger provides context-aware logging
type ContextLogger struct {
    logger *Logger
    ctx    context.Context
}

// NewContextLogger creates a new context logger
func NewContextLogger(logger *Logger, ctx context.Context) *ContextLogger {
    return &ContextLogger{
        logger: logger.WithContext(ctx),
        ctx:    ctx,
    }
}

// WithFields adds fields to the context logger
func (cl *ContextLogger) WithFields(fields map[string]interface{}) *ContextLogger {
    return &ContextLogger{
        logger: cl.logger.WithFields(fields),
        ctx:    cl.ctx,
    }
}

// WithField adds a single field to the context logger
func (cl *ContextLogger) WithField(key string, value interface{}) *ContextLogger {
    return cl.WithFields(map[string]interface{}{key: value})
}

// Debug logs a debug message
func (cl *ContextLogger) Debug(msg string, fields ...map[string]interface{}) {
    cl.logger.Debug(msg, fields...)
}

// Info logs an info message
func (cl *ContextLogger) Info(msg string, fields ...map[string]interface{}) {
    cl.logger.Info(msg, fields...)
}

// Warn logs a warning message
func (cl *ContextLogger) Warn(msg string, fields ...map[string]interface{}) {
    cl.logger.Warn(msg, fields...)
}

// Error logs an error message
func (cl *ContextLogger) Error(msg string, fields ...map[string]interface{}) {
    cl.logger.Error(msg, fields...)
}

// Fatal logs a fatal message and exits
func (cl *ContextLogger) Fatal(msg string, fields ...map[string]interface{}) {
    cl.logger.Fatal(msg, fields...)
}

// LogRequest logs an HTTP request
func (cl *ContextLogger) LogRequest(method, path string, statusCode int, duration time.Duration) {
    cl.Info("HTTP request", map[string]interface{}{
        "method":      method,
        "path":        path,
        "status_code": statusCode,
        "duration_ms": duration.Milliseconds(),
    })
}

// LogError logs an error with context
func (cl *ContextLogger) LogError(err error, msg string, fields ...map[string]interface{}) {
    errorFields := map[string]interface{}{
        "error": err.Error(),
    }
    
    for _, fieldMap := range fields {
        for k, v := range fieldMap {
            errorFields[k] = v
        }
    }
    
    cl.Error(msg, errorFields)
}
```

## Metrics Collection

### Prometheus Metrics

```go
// internal/metrics/prometheus.go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics represents application metrics
type Metrics struct {
    // HTTP metrics
    HTTPRequestsTotal     *prometheus.CounterVec
    HTTPRequestDuration   *prometheus.HistogramVec
    HTTPRequestsInFlight  prometheus.Gauge
    
    // Business metrics
    UsersTotal           prometheus.Gauge
    OrdersTotal          prometheus.Counter
    OrderValue           prometheus.Histogram
    
    // System metrics
    GoRoutines           prometheus.Gauge
    MemoryUsage          prometheus.Gauge
    CPUUsage             prometheus.Gauge
}

// NewMetrics creates new metrics
func NewMetrics() *Metrics {
    return &Metrics{
        HTTPRequestsTotal: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_requests_total",
                Help: "Total number of HTTP requests",
            },
            []string{"method", "path", "status_code"},
        ),
        HTTPRequestDuration: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "http_request_duration_seconds",
                Help:    "HTTP request duration in seconds",
                Buckets: prometheus.DefBuckets,
            },
            []string{"method", "path"},
        ),
        HTTPRequestsInFlight: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "http_requests_in_flight",
                Help: "Number of HTTP requests currently being processed",
            },
        ),
        UsersTotal: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "users_total",
                Help: "Total number of users",
            },
        ),
        OrdersTotal: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "orders_total",
                Help: "Total number of orders",
            },
        ),
        OrderValue: promauto.NewHistogram(
            prometheus.HistogramOpts{
                Name:    "order_value",
                Help:    "Order value distribution",
                Buckets: prometheus.ExponentialBuckets(10, 2, 10),
            },
        ),
        GoRoutines: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "go_goroutines",
                Help: "Number of goroutines",
            },
        ),
        MemoryUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "memory_usage_bytes",
                Help: "Memory usage in bytes",
            },
        ),
        CPUUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "cpu_usage_percent",
                Help: "CPU usage percentage",
            },
        ),
    }
}

// RecordHTTPRequest records an HTTP request
func (m *Metrics) RecordHTTPRequest(method, path string, statusCode int, duration float64) {
    m.HTTPRequestsTotal.WithLabelValues(method, path, fmt.Sprintf("%d", statusCode)).Inc()
    m.HTTPRequestDuration.WithLabelValues(method, path).Observe(duration)
}

// RecordOrder records an order
func (m *Metrics) RecordOrder(value float64) {
    m.OrdersTotal.Inc()
    m.OrderValue.Observe(value)
}

// UpdateSystemMetrics updates system metrics
func (m *Metrics) UpdateSystemMetrics() {
    m.GoRoutines.Set(float64(runtime.NumGoroutine()))
    
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    m.MemoryUsage.Set(float64(m.Alloc))
}
```

### Custom Metrics

```go
// internal/metrics/custom_metrics.go
package metrics

import (
    "sync"
    "time"
)

// CustomMetrics represents custom application metrics
type CustomMetrics struct {
    mutex sync.RWMutex
    
    // Business metrics
    activeUsers    int64
    totalSessions  int64
    averageSession time.Duration
    
    // Performance metrics
    cacheHits      int64
    cacheMisses    int64
    databaseQueries int64
    databaseErrors  int64
    
    // Error metrics
    errorCounts    map[string]int64
    lastErrorTime  time.Time
}

// NewCustomMetrics creates new custom metrics
func NewCustomMetrics() *CustomMetrics {
    return &CustomMetrics{
        errorCounts: make(map[string]int64),
    }
}

// IncrementActiveUsers increments active users
func (cm *CustomMetrics) IncrementActiveUsers() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.activeUsers++
}

// DecrementActiveUsers decrements active users
func (cm *CustomMetrics) DecrementActiveUsers() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    if cm.activeUsers > 0 {
        cm.activeUsers--
    }
}

// RecordSession records a session
func (cm *CustomMetrics) RecordSession(duration time.Duration) {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    
    cm.totalSessions++
    cm.averageSession = time.Duration(
        (int64(cm.averageSession)*int64(cm.totalSessions-1) + int64(duration)) / int64(cm.totalSessions),
    )
}

// RecordCacheHit records a cache hit
func (cm *CustomMetrics) RecordCacheHit() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.cacheHits++
}

// RecordCacheMiss records a cache miss
func (cm *CustomMetrics) RecordCacheMiss() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.cacheMisses++
}

// RecordDatabaseQuery records a database query
func (cm *CustomMetrics) RecordDatabaseQuery() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.databaseQueries++
}

// RecordDatabaseError records a database error
func (cm *CustomMetrics) RecordDatabaseError() {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    cm.databaseErrors++
}

// RecordError records an error
func (cm *CustomMetrics) RecordError(errorType string) {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    
    cm.errorCounts[errorType]++
    cm.lastErrorTime = time.Now()
}

// GetMetrics returns current metrics
func (cm *CustomMetrics) GetMetrics() map[string]interface{} {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    
    cacheHitRate := float64(0)
    if cm.cacheHits+cm.cacheMisses > 0 {
        cacheHitRate = float64(cm.cacheHits) / float64(cm.cacheHits+cm.cacheMisses)
    }
    
    databaseErrorRate := float64(0)
    if cm.databaseQueries > 0 {
        databaseErrorRate = float64(cm.databaseErrors) / float64(cm.databaseQueries)
    }
    
    return map[string]interface{}{
        "active_users":        cm.activeUsers,
        "total_sessions":      cm.totalSessions,
        "average_session_ms":  cm.averageSession.Milliseconds(),
        "cache_hits":          cm.cacheHits,
        "cache_misses":       cm.cacheMisses,
        "cache_hit_rate":     cacheHitRate,
        "database_queries":   cm.databaseQueries,
        "database_errors":    cm.databaseErrors,
        "database_error_rate": databaseErrorRate,
        "error_counts":       cm.errorCounts,
        "last_error_time":    cm.lastErrorTime,
    }
}
```

## Distributed Tracing

### OpenTelemetry Integration

```go
// internal/tracing/opentelemetry.go
package tracing

import (
    "context"
    "fmt"
    "time"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
    "go.opentelemetry.io/otel/trace"
)

// Tracer represents a distributed tracer
type Tracer struct {
    tracer trace.Tracer
}

// NewTracer creates a new tracer
func NewTracer(serviceName, serviceVersion string) (*Tracer, error) {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint())
    if err != nil {
        return nil, fmt.Errorf("failed to create Jaeger exporter: %w", err)
    }
    
    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String(serviceName),
            semconv.ServiceVersionKey.String(serviceVersion),
        ),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create resource: %w", err)
    }
    
    // Create tracer provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(res),
    )
    
    // Set global tracer provider
    otel.SetTracerProvider(tp)
    
    // Set global propagator
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))
    
    return &Tracer{
        tracer: tp.Tracer(serviceName),
    }, nil
}

// StartSpan starts a new span
func (t *Tracer) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
    return t.tracer.Start(ctx, name, opts...)
}

// StartSpanWithAttributes starts a new span with attributes
func (t *Tracer) StartSpanWithAttributes(ctx context.Context, name string, attrs map[string]interface{}) (context.Context, trace.Span) {
    spanOpts := []trace.SpanStartOption{
        trace.WithAttributes(t.convertAttributes(attrs)...),
    }
    
    return t.tracer.Start(ctx, name, spanOpts...)
}

// convertAttributes converts map to attributes
func (t *Tracer) convertAttributes(attrs map[string]interface{}) []attribute.KeyValue {
    var attributes []attribute.KeyValue
    
    for k, v := range attrs {
        switch val := v.(type) {
        case string:
            attributes = append(attributes, attribute.String(k, val))
        case int:
            attributes = append(attributes, attribute.Int64(k, int64(val)))
        case int64:
            attributes = append(attributes, attribute.Int64(k, val))
        case float64:
            attributes = append(attributes, attribute.Float64(k, val))
        case bool:
            attributes = append(attributes, attribute.Bool(k, val))
        default:
            attributes = append(attributes, attribute.String(k, fmt.Sprintf("%v", val)))
        }
    }
    
    return attributes
}

// FinishSpan finishes a span with success
func (t *Tracer) FinishSpan(span trace.Span) {
    span.SetStatus(codes.Ok, "")
    span.End()
}

// FinishSpanWithError finishes a span with error
func (t *Tracer) FinishSpanWithError(span trace.Span, err error) {
    span.RecordError(err)
    span.SetStatus(codes.Error, err.Error())
    span.End()
}

// AddSpanAttributes adds attributes to a span
func (t *Tracer) AddSpanAttributes(span trace.Span, attrs map[string]interface{}) {
    span.SetAttributes(t.convertAttributes(attrs)...)
}

// AddSpanEvent adds an event to a span
func (t *Tracer) AddSpanEvent(span trace.Span, name string, attrs map[string]interface{}) {
    span.AddEvent(name, trace.WithAttributes(t.convertAttributes(attrs)...))
}
```

### HTTP Tracing Middleware

```go
// internal/tracing/http_middleware.go
package tracing

import (
    "net/http"
    "time"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/trace"
)

// TracingMiddleware provides HTTP tracing middleware
type TracingMiddleware struct {
    tracer trace.Tracer
}

// NewTracingMiddleware creates a new tracing middleware
func NewTracingMiddleware() *TracingMiddleware {
    return &TracingMiddleware{
        tracer: otel.Tracer("http-server"),
    }
}

// Middleware returns the HTTP middleware function
func (tm *TracingMiddleware) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract trace context from headers
        ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
        
        // Start span
        ctx, span := tm.tracer.Start(ctx, fmt.Sprintf("%s %s", r.Method, r.URL.Path))
        defer span.End()
        
        // Add span attributes
        span.SetAttributes(
            attribute.String("http.method", r.Method),
            attribute.String("http.url", r.URL.String()),
            attribute.String("http.user_agent", r.UserAgent()),
            attribute.String("http.remote_addr", r.RemoteAddr),
        )
        
        // Create response writer wrapper
        ww := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
        
        // Set trace context in response headers
        otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(ww.Header()))
        
        // Process request
        start := time.Now()
        next.ServeHTTP(ww, r.WithContext(ctx))
        duration := time.Since(start)
        
        // Add response attributes
        span.SetAttributes(
            attribute.Int("http.status_code", ww.statusCode),
            attribute.Int64("http.response_size", ww.size),
            attribute.Float64("http.duration_ms", float64(duration.Nanoseconds())/1e6),
        )
        
        // Set span status
        if ww.statusCode >= 400 {
            span.SetStatus(codes.Error, fmt.Sprintf("HTTP %d", ww.statusCode))
        }
    })
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
    http.ResponseWriter
    statusCode int
    size       int64
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
    n, err := rw.ResponseWriter.Write(b)
    rw.size += int64(n)
    return n, err
}
```

## Health Checks

### Health Check System

```go
// internal/health/health_check.go
package health

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// HealthStatus represents health status
type HealthStatus string

const (
    StatusHealthy   HealthStatus = "healthy"
    StatusUnhealthy HealthStatus = "unhealthy"
    StatusDegraded  HealthStatus = "degraded"
)

// HealthCheck represents a health check
type HealthCheck interface {
    Name() string
    Check(ctx context.Context) HealthResult
}

// HealthResult represents the result of a health check
type HealthResult struct {
    Name      string        `json:"name"`
    Status    HealthStatus `json:"status"`
    Message   string        `json:"message,omitempty"`
    Duration  time.Duration `json:"duration"`
    Timestamp time.Time     `json:"timestamp"`
    Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// HealthChecker represents a health checker
type HealthChecker struct {
    checks []HealthCheck
    mutex  sync.RWMutex
}

// NewHealthChecker creates a new health checker
func NewHealthChecker() *HealthChecker {
    return &HealthChecker{
        checks: make([]HealthCheck, 0),
    }
}

// AddCheck adds a health check
func (hc *HealthChecker) AddCheck(check HealthCheck) {
    hc.mutex.Lock()
    defer hc.mutex.Unlock()
    hc.checks = append(hc.checks, check)
}

// CheckAll runs all health checks
func (hc *HealthChecker) CheckAll(ctx context.Context) map[string]HealthResult {
    hc.mutex.RLock()
    checks := make([]HealthCheck, len(hc.checks))
    copy(checks, hc.checks)
    hc.mutex.RUnlock()
    
    results := make(map[string]HealthResult)
    
    for _, check := range checks {
        result := check.Check(ctx)
        results[check.Name()] = result
    }
    
    return results
}

// GetOverallStatus returns the overall health status
func (hc *HealthChecker) GetOverallStatus(ctx context.Context) HealthStatus {
    results := hc.CheckAll(ctx)
    
    hasUnhealthy := false
    hasDegraded := false
    
    for _, result := range results {
        switch result.Status {
        case StatusUnhealthy:
            hasUnhealthy = true
        case StatusDegraded:
            hasDegraded = true
        }
    }
    
    if hasUnhealthy {
        return StatusUnhealthy
    }
    if hasDegraded {
        return StatusDegraded
    }
    
    return StatusHealthy
}

// DatabaseHealthCheck represents a database health check
type DatabaseHealthCheck struct {
    name     string
    pingFunc func(context.Context) error
}

// NewDatabaseHealthCheck creates a new database health check
func NewDatabaseHealthCheck(name string, pingFunc func(context.Context) error) *DatabaseHealthCheck {
    return &DatabaseHealthCheck{
        name:     name,
        pingFunc: pingFunc,
    }
}

// Name returns the health check name
func (dhc *DatabaseHealthCheck) Name() string {
    return dhc.name
}

// Check performs the health check
func (dhc *DatabaseHealthCheck) Check(ctx context.Context) HealthResult {
    start := time.Now()
    
    err := dhc.pingFunc(ctx)
    duration := time.Since(start)
    
    result := HealthResult{
        Name:      dhc.name,
        Duration:  duration,
        Timestamp: time.Now(),
    }
    
    if err != nil {
        result.Status = StatusUnhealthy
        result.Message = err.Error()
    } else {
        result.Status = StatusHealthy
        result.Message = "Database connection successful"
    }
    
    return result
}

// HTTPHealthCheck represents an HTTP health check
type HTTPHealthCheck struct {
    name    string
    url     string
    client  *http.Client
    timeout time.Duration
}

// NewHTTPHealthCheck creates a new HTTP health check
func NewHTTPHealthCheck(name, url string, timeout time.Duration) *HTTPHealthCheck {
    return &HTTPHealthCheck{
        name:    name,
        url:     url,
        client:  &http.Client{Timeout: timeout},
        timeout: timeout,
    }
}

// Name returns the health check name
func (hhc *HTTPHealthCheck) Name() string {
    return hhc.name
}

// Check performs the health check
func (hhc *HTTPHealthCheck) Check(ctx context.Context) HealthResult {
    start := time.Now()
    
    req, err := http.NewRequestWithContext(ctx, "GET", hhc.url, nil)
    if err != nil {
        return HealthResult{
            Name:      hhc.name,
            Status:    StatusUnhealthy,
            Message:   fmt.Sprintf("Failed to create request: %v", err),
            Duration:  time.Since(start),
            Timestamp: time.Now(),
        }
    }
    
    resp, err := hhc.client.Do(req)
    duration := time.Since(start)
    
    result := HealthResult{
        Name:      hhc.name,
        Duration:  duration,
        Timestamp: time.Now(),
    }
    
    if err != nil {
        result.Status = StatusUnhealthy
        result.Message = err.Error()
    } else {
        resp.Body.Close()
        
        if resp.StatusCode >= 200 && resp.StatusCode < 300 {
            result.Status = StatusHealthy
            result.Message = "HTTP health check successful"
        } else {
            result.Status = StatusUnhealthy
            result.Message = fmt.Sprintf("HTTP status %d", resp.StatusCode)
        }
    }
    
    return result
}
```

## Alerting

### Alert Manager

```go
// internal/alerting/alert_manager.go
package alerting

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// AlertLevel represents alert levels
type AlertLevel int

const (
    InfoLevel AlertLevel = iota
    WarningLevel
    CriticalLevel
)

// String returns the string representation of the alert level
func (al AlertLevel) String() string {
    switch al {
    case InfoLevel:
        return "INFO"
    case WarningLevel:
        return "WARNING"
    case CriticalLevel:
        return "CRITICAL"
    default:
        return "UNKNOWN"
    }
}

// Alert represents an alert
type Alert struct {
    ID          string                 `json:"id"`
    Level       AlertLevel             `json:"level"`
    Title       string                 `json:"title"`
    Message     string                 `json:"message"`
    Source      string                 `json:"source"`
    Timestamp   time.Time              `json:"timestamp"`
    Metadata    map[string]interface{} `json:"metadata,omitempty"`
    Resolved    bool                   `json:"resolved"`
    ResolvedAt  *time.Time             `json:"resolved_at,omitempty"`
}

// AlertRule represents an alert rule
type AlertRule struct {
    ID          string
    Name        string
    Condition   func(metrics map[string]interface{}) bool
    Level       AlertLevel
    Title       string
    Message     string
    Cooldown    time.Duration
    lastTriggered time.Time
}

// AlertManager represents an alert manager
type AlertManager struct {
    rules    []AlertRule
    alerts   map[string]*Alert
    mutex    sync.RWMutex
    notifiers []Notifier
}

// NewAlertManager creates a new alert manager
func NewAlertManager() *AlertManager {
    return &AlertManager{
        rules:    make([]AlertRule, 0),
        alerts:   make(map[string]*Alert),
        notifiers: make([]Notifier, 0),
    }
}

// AddRule adds an alert rule
func (am *AlertManager) AddRule(rule AlertRule) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    am.rules = append(am.rules, rule)
}

// AddNotifier adds a notifier
func (am *AlertManager) AddNotifier(notifier Notifier) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    am.notifiers = append(am.notifiers, notifier)
}

// EvaluateRules evaluates all alert rules
func (am *AlertManager) EvaluateRules(ctx context.Context, metrics map[string]interface{}) {
    am.mutex.RLock()
    rules := make([]AlertRule, len(am.rules))
    copy(rules, am.rules)
    am.mutex.RUnlock()
    
    for _, rule := range rules {
        if rule.Condition(metrics) {
            am.triggerAlert(rule, metrics)
        }
    }
}

// triggerAlert triggers an alert
func (am *AlertManager) triggerAlert(rule AlertRule, metrics map[string]interface{}) {
    // Check cooldown
    if time.Since(rule.lastTriggered) < rule.Cooldown {
        return
    }
    
    alert := &Alert{
        ID:        fmt.Sprintf("%s-%d", rule.ID, time.Now().Unix()),
        Level:     rule.Level,
        Title:     rule.Title,
        Message:   rule.Message,
        Source:    "alert-manager",
        Timestamp: time.Now(),
        Metadata:  metrics,
        Resolved:  false,
    }
    
    am.mutex.Lock()
    am.alerts[alert.ID] = alert
    am.mutex.Unlock()
    
    // Notify
    for _, notifier := range am.notifiers {
        go notifier.Notify(ctx, alert)
    }
    
    // Update last triggered time
    rule.lastTriggered = time.Now()
}

// ResolveAlert resolves an alert
func (am *AlertManager) ResolveAlert(alertID string) {
    am.mutex.Lock()
    defer am.mutex.Unlock()
    
    if alert, exists := am.alerts[alertID]; exists {
        alert.Resolved = true
        now := time.Now()
        alert.ResolvedAt = &now
    }
}

// GetActiveAlerts returns active alerts
func (am *AlertManager) GetActiveAlerts() []Alert {
    am.mutex.RLock()
    defer am.mutex.RUnlock()
    
    var activeAlerts []Alert
    for _, alert := range am.alerts {
        if !alert.Resolved {
            activeAlerts = append(activeAlerts, *alert)
        }
    }
    
    return activeAlerts
}

// Notifier represents a notification interface
type Notifier interface {
    Notify(ctx context.Context, alert *Alert) error
}

// SlackNotifier represents a Slack notifier
type SlackNotifier struct {
    webhookURL string
    channel    string
}

// NewSlackNotifier creates a new Slack notifier
func NewSlackNotifier(webhookURL, channel string) *SlackNotifier {
    return &SlackNotifier{
        webhookURL: webhookURL,
        channel:    channel,
    }
}

// Notify sends a notification to Slack
func (sn *SlackNotifier) Notify(ctx context.Context, alert *Alert) error {
    // Implement Slack notification
    return nil
}

// EmailNotifier represents an email notifier
type EmailNotifier struct {
    smtpHost string
    smtpPort int
    username string
    password string
    from     string
    to       []string
}

// NewEmailNotifier creates a new email notifier
func NewEmailNotifier(smtpHost string, smtpPort int, username, password, from string, to []string) *EmailNotifier {
    return &EmailNotifier{
        smtpHost: smtpHost,
        smtpPort: smtpPort,
        username: username,
        password: password,
        from:     from,
        to:       to,
    }
}

// Notify sends an email notification
func (en *EmailNotifier) Notify(ctx context.Context, alert *Alert) error {
    // Implement email notification
    return nil
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Structured logging
logger := NewLogger(InfoLevel)
logger.Info("Application started", map[string]interface{}{
    "version": "1.0.0",
    "port":    8080,
})

// 2. Metrics collection
metrics := NewMetrics()
metrics.RecordHTTPRequest("GET", "/api/users", 200, 0.1)

// 3. Distributed tracing
tracer, err := NewTracer("myapp", "1.0.0")
ctx, span := tracer.StartSpan(ctx, "process-request")
defer tracer.FinishSpan(span)

// 4. Health checks
healthChecker := NewHealthChecker()
healthChecker.AddCheck(NewDatabaseHealthCheck("database", db.Ping))
```

### Essential Patterns

```go
// Context-aware logging
ctxLogger := NewContextLogger(logger, ctx)
ctxLogger.Info("Processing request", map[string]interface{}{
    "user_id": userID,
    "action":  "create",
})

// Custom metrics
customMetrics := NewCustomMetrics()
customMetrics.RecordSession(5 * time.Minute)

// Alert rules
alertManager.AddRule(AlertRule{
    ID: "high-error-rate",
    Condition: func(metrics map[string]interface{}) bool {
        return metrics["error_rate"].(float64) > 0.1
    },
    Level:   CriticalLevel,
    Title:   "High Error Rate",
    Message: "Error rate is above 10%",
})
```

---

*This guide provides the complete machinery for implementing comprehensive monitoring and observability in Go applications. Each pattern includes implementation examples, integration strategies, and real-world usage patterns for enterprise deployment.*
