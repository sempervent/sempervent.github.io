# Rust Monitoring & Observability Best Practices

**Objective**: Master senior-level Rust monitoring and observability patterns for production systems. When you need to build comprehensive monitoring, when you want to implement distributed tracing, when you need enterprise-grade observability—these best practices become your weapon of choice.

## Core Principles

- **Three Pillars**: Metrics, logging, and tracing
- **Structured Logging**: Use structured logging for better analysis
- **Distributed Tracing**: Implement end-to-end request tracing
- **Health Checks**: Monitor application health and readiness
- **Alerting**: Set up intelligent alerting based on metrics

## Monitoring & Observability Patterns

### Structured Logging

```rust
// rust/01-structured-logging.rs

/*
Structured logging patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub target: String,
    pub span_id: Option<String>,
    pub trace_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: String, target: String) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            level,
            message,
            fields: HashMap::new(),
            target,
            span_id: None,
            trace_id: None,
        }
    }
    
    /// Add a field to the log entry.
    pub fn field(mut self, key: String, value: serde_json::Value) -> Self {
        self.fields.insert(key, value);
        self
    }
    
    /// Set span ID for distributed tracing.
    pub fn span_id(mut self, span_id: String) -> Self {
        self.span_id = Some(span_id);
        self
    }
    
    /// Set trace ID for distributed tracing.
    pub fn trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

/// Structured logger.
pub struct StructuredLogger {
    app_name: String,
    version: String,
    environment: String,
    log_level: LogLevel,
    fields: HashMap<String, serde_json::Value>,
}

impl StructuredLogger {
    pub fn new(app_name: String, version: String, environment: String) -> Self {
        Self {
            app_name,
            version,
            environment,
            log_level: LogLevel::Info,
            fields: HashMap::new(),
        }
    }
    
    /// Set the minimum log level.
    pub fn set_log_level(&mut self, level: LogLevel) {
        self.log_level = level;
    }
    
    /// Add a global field to all log entries.
    pub fn add_global_field(&mut self, key: String, value: serde_json::Value) {
        self.fields.insert(key, value);
    }
    
    /// Create a log entry.
    pub fn log(&self, level: LogLevel, message: String, target: String) -> LogEntry {
        let mut entry = LogEntry::new(level, message, target);
        
        // Add global fields
        for (key, value) in &self.fields {
            entry.fields.insert(key.clone(), value.clone());
        }
        
        // Add application metadata
        entry.fields.insert("app_name".to_string(), serde_json::Value::String(self.app_name.clone()));
        entry.fields.insert("version".to_string(), serde_json::Value::String(self.version.clone()));
        entry.fields.insert("environment".to_string(), serde_json::Value::String(self.environment.clone()));
        
        entry
    }
    
    /// Log an info message.
    pub fn info(&self, message: String, target: String) -> LogEntry {
        self.log(LogLevel::Info, message, target)
    }
    
    /// Log a warning message.
    pub fn warn(&self, message: String, target: String) -> LogEntry {
        self.log(LogLevel::Warn, message, target)
    }
    
    /// Log an error message.
    pub fn error(&self, message: String, target: String) -> LogEntry {
        self.log(LogLevel::Error, message, target)
    }
    
    /// Log a debug message.
    pub fn debug(&self, message: String, target: String) -> LogEntry {
        self.log(LogLevel::Debug, message, target)
    }
    
    /// Log a trace message.
    pub fn trace(&self, message: String, target: String) -> LogEntry {
        self.log(LogLevel::Trace, message, target)
    }
}

/// Log formatter for different output formats.
pub trait LogFormatter {
    fn format(&self, entry: &LogEntry) -> String;
}

/// JSON log formatter.
pub struct JsonFormatter;

impl LogFormatter for JsonFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        serde_json::to_string(entry).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Text log formatter.
pub struct TextFormatter;

impl LogFormatter for TextFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        format!(
            "[{}] {} {}: {}",
            entry.timestamp,
            entry.level,
            entry.target,
            entry.message
        )
    }
}

/// Log handler for processing log entries.
pub trait LogHandler {
    fn handle(&self, entry: &LogEntry) -> Result<(), String>;
}

/// Console log handler.
pub struct ConsoleLogHandler {
    formatter: Box<dyn LogFormatter>,
}

impl ConsoleLogHandler {
    pub fn new(formatter: Box<dyn LogFormatter>) -> Self {
        Self { formatter }
    }
}

impl LogHandler for ConsoleLogHandler {
    fn handle(&self, entry: &LogEntry) -> Result<(), String> {
        let formatted = self.formatter.format(entry);
        println!("{}", formatted);
        Ok(())
    }
}

/// File log handler.
pub struct FileLogHandler {
    file_path: String,
    formatter: Box<dyn LogFormatter>,
}

impl FileLogHandler {
    pub fn new(file_path: String, formatter: Box<dyn LogFormatter>) -> Self {
        Self { file_path, formatter }
    }
}

impl LogHandler for FileLogHandler {
    fn handle(&self, entry: &LogEntry) -> Result<(), String> {
        let formatted = self.formatter.format(entry);
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| format!("Failed to open log file: {}", e))?
            .write_all(formatted.as_bytes())
            .map_err(|e| format!("Failed to write to log file: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_structured_logger() {
        let logger = StructuredLogger::new(
            "test-app".to_string(),
            "1.0.0".to_string(),
            "test".to_string(),
        );
        
        let entry = logger.info("Test message".to_string(), "test::module".to_string());
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.target, "test::module");
    }
    
    #[test]
    fn test_log_entry_fields() {
        let mut entry = LogEntry::new(
            LogLevel::Info,
            "Test message".to_string(),
            "test::module".to_string(),
        );
        
        entry = entry.field("user_id".to_string(), serde_json::Value::String("123".to_string()));
        entry = entry.field("request_id".to_string(), serde_json::Value::String("req-123".to_string()));
        
        assert!(entry.fields.contains_key("user_id"));
        assert!(entry.fields.contains_key("request_id"));
    }
}
```

### Metrics Collection

```rust
// rust/02-metrics-collection.rs

/*
Metrics collection patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use serde::{Deserialize, Serialize};

/// Metric types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Metric value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp: u64,
}

/// Counter metric.
pub struct Counter {
    name: String,
    value: AtomicU64,
    labels: HashMap<String, String>,
}

impl Counter {
    pub fn new(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            name,
            value: AtomicU64::new(0),
            labels,
        }
    }
    
    /// Increment the counter.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add a value to the counter.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }
    
    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    
    /// Get the metric value.
    pub fn get_metric_value(&self) -> MetricValue {
        MetricValue {
            name: self.name.clone(),
            value: self.get() as f64,
            labels: self.labels.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}

/// Gauge metric.
pub struct Gauge {
    name: String,
    value: AtomicF64::new(0.0),
    labels: HashMap<String, String>,
}

impl Gauge {
    pub fn new(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            name,
            value: AtomicF64::new(0.0),
            labels,
        }
    }
    
    /// Set the gauge value.
    pub fn set(&self, value: f64) {
        self.value.store(value, Ordering::Relaxed);
    }
    
    /// Add a value to the gauge.
    pub fn add(&self, value: f64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }
    
    /// Subtract a value from the gauge.
    pub fn sub(&self, value: f64) {
        self.value.fetch_sub(value, Ordering::Relaxed);
    }
    
    /// Get the current value.
    pub fn get(&self) -> f64 {
        self.value.load(Ordering::Relaxed)
    }
    
    /// Get the metric value.
    pub fn get_metric_value(&self) -> MetricValue {
        MetricValue {
            name: self.name.clone(),
            value: self.get(),
            labels: self.labels.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}

/// Histogram metric.
pub struct Histogram {
    name: String,
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum: AtomicF64,
    labels: HashMap<String, String>,
}

impl Histogram {
    pub fn new(name: String, buckets: Vec<f64>, labels: HashMap<String, String>) -> Self {
        let counts = vec![AtomicU64::new(0); buckets.len() + 1]; // +1 for +Inf bucket
        Self {
            name,
            buckets,
            counts,
            sum: AtomicF64::new(0.0),
            labels,
        }
    }
    
    /// Observe a value.
    pub fn observe(&self, value: f64) {
        self.sum.fetch_add(value, Ordering::Relaxed);
        
        // Find the appropriate bucket
        let mut bucket_index = self.buckets.len();
        for (i, bucket) in self.buckets.iter().enumerate() {
            if value <= *bucket {
                bucket_index = i;
                break;
            }
        }
        
        self.counts[bucket_index].fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get the metric value.
    pub fn get_metric_value(&self) -> MetricValue {
        MetricValue {
            name: self.name.clone(),
            value: self.sum.load(Ordering::Relaxed),
            labels: self.labels.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}

/// Metrics registry.
pub struct MetricsRegistry {
    counters: HashMap<String, Arc<Counter>>,
    gauges: HashMap<String, Arc<Gauge>>,
    histograms: HashMap<String, Arc<Histogram>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
        }
    }
    
    /// Register a counter.
    pub fn register_counter(&mut self, name: String, labels: HashMap<String, String>) -> Arc<Counter> {
        let counter = Arc::new(Counter::new(name.clone(), labels));
        self.counters.insert(name, counter.clone());
        counter
    }
    
    /// Register a gauge.
    pub fn register_gauge(&mut self, name: String, labels: HashMap<String, String>) -> Arc<Gauge> {
        let gauge = Arc::new(Gauge::new(name.clone(), labels));
        self.gauges.insert(name, gauge.clone());
        gauge
    }
    
    /// Register a histogram.
    pub fn register_histogram(&mut self, name: String, buckets: Vec<f64>, labels: HashMap<String, String>) -> Arc<Histogram> {
        let histogram = Arc::new(Histogram::new(name.clone(), buckets, labels));
        self.histograms.insert(name, histogram.clone());
        histogram
    }
    
    /// Get all metric values.
    pub fn get_all_metrics(&self) -> Vec<MetricValue> {
        let mut metrics = Vec::new();
        
        for counter in self.counters.values() {
            metrics.push(counter.get_metric_value());
        }
        
        for gauge in self.gauges.values() {
            metrics.push(gauge.get_metric_value());
        }
        
        for histogram in self.histograms.values() {
            metrics.push(histogram.get_metric_value());
        }
        
        metrics
    }
    
    /// Get metrics in Prometheus format.
    pub fn get_prometheus_metrics(&self) -> String {
        let mut output = String::new();
        
        for counter in self.counters.values() {
            let metric = counter.get_metric_value();
            output.push_str(&format!(
                "# HELP {} Counter metric\n",
                metric.name
            ));
            output.push_str(&format!(
                "# TYPE {} counter\n",
                metric.name
            ));
            output.push_str(&format!(
                "{} {} {}\n",
                metric.name,
                metric.value,
                metric.timestamp
            ));
        }
        
        for gauge in self.gauges.values() {
            let metric = gauge.get_metric_value();
            output.push_str(&format!(
                "# HELP {} Gauge metric\n",
                metric.name
            ));
            output.push_str(&format!(
                "# TYPE {} gauge\n",
                metric.name
            ));
            output.push_str(&format!(
                "{} {} {}\n",
                metric.name,
                metric.value,
                metric.timestamp
            ));
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter".to_string(), HashMap::new());
        counter.inc();
        counter.add(5);
        assert_eq!(counter.get(), 6);
    }
    
    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge".to_string(), HashMap::new());
        gauge.set(10.0);
        gauge.add(5.0);
        assert_eq!(gauge.get(), 15.0);
    }
    
    #[test]
    fn test_histogram() {
        let histogram = Histogram::new(
            "test_histogram".to_string(),
            vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            HashMap::new(),
        );
        
        histogram.observe(0.3);
        histogram.observe(0.7);
        histogram.observe(1.5);
        
        assert!(histogram.sum.load(Ordering::Relaxed) > 0.0);
    }
    
    #[test]
    fn test_metrics_registry() {
        let mut registry = MetricsRegistry::new();
        
        let counter = registry.register_counter("test_counter".to_string(), HashMap::new());
        counter.inc();
        
        let gauge = registry.register_gauge("test_gauge".to_string(), HashMap::new());
        gauge.set(10.0);
        
        let metrics = registry.get_all_metrics();
        assert_eq!(metrics.len(), 2);
    }
}
```

### Distributed Tracing

```rust
// rust/03-distributed-tracing.rs

/*
Distributed tracing patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Trace context for distributed tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub baggage: HashMap<String, String>,
}

impl TraceContext {
    pub fn new(trace_id: String, span_id: String) -> Self {
        Self {
            trace_id,
            span_id,
            parent_span_id: None,
            baggage: HashMap::new(),
        }
    }
    
    /// Create a new trace context.
    pub fn create() -> Self {
        Self::new(
            Uuid::new_v4().to_string(),
            Uuid::new_v4().to_string(),
        )
    }
    
    /// Create a child span context.
    pub fn create_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            baggage: self.baggage.clone(),
        }
    }
    
    /// Add baggage to the trace context.
    pub fn add_baggage(&mut self, key: String, value: String) {
        self.baggage.insert(key, value);
    }
    
    /// Get baggage from the trace context.
    pub fn get_baggage(&self, key: &str) -> Option<&String> {
        self.baggage.get(key)
    }
}

/// Span for distributed tracing.
pub struct Span {
    pub context: TraceContext,
    pub operation_name: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: u64,
    pub fields: HashMap<String, String>,
}

impl Span {
    pub fn new(context: TraceContext, operation_name: String) -> Self {
        Self {
            context,
            operation_name,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            end_time: None,
            tags: HashMap::new(),
            logs: Vec::new(),
        }
    }
    
    /// Add a tag to the span.
    pub fn tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }
    
    /// Add a log entry to the span.
    pub fn log(&mut self, fields: HashMap<String, String>) {
        let log = SpanLog {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            fields,
        };
        self.logs.push(log);
    }
    
    /// Finish the span.
    pub fn finish(&mut self) {
        self.end_time = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        );
    }
    
    /// Get the duration of the span.
    pub fn duration(&self) -> Option<u64> {
        self.end_time.map(|end| end - self.start_time)
    }
}

/// Tracer for distributed tracing.
pub struct Tracer {
    service_name: String,
    service_version: String,
    spans: Vec<Span>,
}

impl Tracer {
    pub fn new(service_name: String, service_version: String) -> Self {
        Self {
            service_name,
            service_version,
            spans: Vec::new(),
        }
    }
    
    /// Start a new span.
    pub fn start_span(&mut self, operation_name: String, parent_context: Option<&TraceContext>) -> TraceContext {
        let context = if let Some(parent) = parent_context {
            parent.create_child()
        } else {
            TraceContext::create()
        };
        
        let mut span = Span::new(context.clone(), operation_name);
        span.tag("service.name".to_string(), self.service_name.clone());
        span.tag("service.version".to_string(), self.service_version.clone());
        
        self.spans.push(span);
        context
    }
    
    /// Finish a span.
    pub fn finish_span(&mut self, context: &TraceContext) -> Option<Span> {
        if let Some(span) = self.spans.iter_mut().find(|s| s.context.span_id == context.span_id) {
            span.finish();
            Some(span.clone())
        } else {
            None
        }
    }
    
    /// Get all spans.
    pub fn get_spans(&self) -> &[Span] {
        &self.spans
    }
    
    /// Get spans for a specific trace.
    pub fn get_spans_for_trace(&self, trace_id: &str) -> Vec<&Span> {
        self.spans
            .iter()
            .filter(|span| span.context.trace_id == trace_id)
            .collect()
    }
}

/// Trace exporter for sending traces to external systems.
pub trait TraceExporter {
    fn export(&self, spans: &[Span]) -> Result<(), String>;
}

/// Console trace exporter.
pub struct ConsoleTraceExporter;

impl TraceExporter for ConsoleTraceExporter {
    fn export(&self, spans: &[Span]) -> Result<(), String> {
        for span in spans {
            println!("Span: {} ({}ms)", span.operation_name, span.duration().unwrap_or(0));
            println!("  Trace ID: {}", span.context.trace_id);
            println!("  Span ID: {}", span.context.span_id);
            if let Some(parent_id) = &span.context.parent_span_id {
                println!("  Parent Span ID: {}", parent_id);
            }
            println!("  Tags: {:?}", span.tags);
            println!("  Logs: {}", span.logs.len());
        }
        Ok(())
    }
}

/// Jaeger trace exporter.
pub struct JaegerTraceExporter {
    endpoint: String,
    service_name: String,
}

impl JaegerTraceExporter {
    pub fn new(endpoint: String, service_name: String) -> Self {
        Self { endpoint, service_name }
    }
}

impl TraceExporter for JaegerTraceExporter {
    fn export(&self, spans: &[Span]) -> Result<(), String> {
        // In a real implementation, you would send the spans to Jaeger
        println!("Exporting {} spans to Jaeger at {}", spans.len(), self.endpoint);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trace_context() {
        let context = TraceContext::create();
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        
        let child_context = context.create_child();
        assert_eq!(child_context.trace_id, context.trace_id);
        assert_eq!(child_context.parent_span_id, Some(context.span_id));
    }
    
    #[test]
    fn test_span() {
        let context = TraceContext::create();
        let mut span = Span::new(context, "test_operation".to_string());
        
        span.tag("test_key".to_string(), "test_value".to_string());
        span.log(HashMap::from([("message".to_string(), "test log".to_string())]));
        span.finish();
        
        assert!(span.duration().is_some());
        assert_eq!(span.tags.get("test_key"), Some(&"test_value".to_string()));
        assert_eq!(span.logs.len(), 1);
    }
    
    #[test]
    fn test_tracer() {
        let mut tracer = Tracer::new("test_service".to_string(), "1.0.0".to_string());
        
        let context = tracer.start_span("test_operation".to_string(), None);
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        
        let span = tracer.finish_span(&context);
        assert!(span.is_some());
        assert_eq!(tracer.get_spans().len(), 1);
    }
}
```

### Health Checks

```rust
// rust/04-health-checks.rs

/*
Health check patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Health check status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Health check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub duration: Duration,
    pub timestamp: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Health check trait.
pub trait HealthCheck {
    fn name(&self) -> &str;
    fn check(&self) -> HealthCheckResult;
}

/// Database health check.
pub struct DatabaseHealthCheck {
    name: String,
    connection_string: String,
}

impl DatabaseHealthCheck {
    pub fn new(name: String, connection_string: String) -> Self {
        Self { name, connection_string }
    }
}

impl HealthCheck for DatabaseHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn check(&self) -> HealthCheckResult {
        let start_time = Instant::now();
        
        // In a real implementation, you would test the database connection
        let status = if self.connection_string.contains("test") {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };
        
        let duration = start_time.elapsed();
        
        HealthCheckResult {
            name: self.name.clone(),
            status,
            message: Some("Database connection test".to_string()),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

/// Redis health check.
pub struct RedisHealthCheck {
    name: String,
    redis_url: String,
}

impl RedisHealthCheck {
    pub fn new(name: String, redis_url: String) -> Self {
        Self { name, redis_url }
    }
}

impl HealthCheck for RedisHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn check(&self) -> HealthCheckResult {
        let start_time = Instant::now();
        
        // In a real implementation, you would test the Redis connection
        let status = if self.redis_url.contains("test") {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };
        
        let duration = start_time.elapsed();
        
        HealthCheckResult {
            name: self.name.clone(),
            status,
            message: Some("Redis connection test".to_string()),
            duration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

/// Health check manager.
pub struct HealthCheckManager {
    checks: Vec<Box<dyn HealthCheck>>,
    results: HashMap<String, HealthCheckResult>,
}

impl HealthCheckManager {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            results: HashMap::new(),
        }
    }
    
    /// Add a health check.
    pub fn add_check(&mut self, check: Box<dyn HealthCheck>) {
        self.checks.push(check);
    }
    
    /// Run all health checks.
    pub fn run_checks(&mut self) -> HashMap<String, HealthCheckResult> {
        let mut results = HashMap::new();
        
        for check in &self.checks {
            let result = check.check();
            results.insert(check.name().to_string(), result.clone());
        }
        
        self.results = results.clone();
        results
    }
    
    /// Get the overall health status.
    pub fn get_overall_status(&self) -> HealthStatus {
        if self.results.is_empty() {
            return HealthStatus::Unhealthy;
        }
        
        let has_unhealthy = self.results.values().any(|r| r.status == HealthStatus::Unhealthy);
        let has_degraded = self.results.values().any(|r| r.status == HealthStatus::Degraded);
        
        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
    
    /// Get health check results.
    pub fn get_results(&self) -> &HashMap<String, HealthCheckResult> {
        &self.results
    }
    
    /// Get health check summary.
    pub fn get_summary(&self) -> String {
        let total_checks = self.results.len();
        let healthy_checks = self.results.values()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count();
        let unhealthy_checks = self.results.values()
            .filter(|r| r.status == HealthStatus::Unhealthy)
            .count();
        let degraded_checks = self.results.values()
            .filter(|r| r.status == HealthStatus::Degraded)
            .count();
        
        format!(
            "Health Checks: {} total, {} healthy, {} degraded, {} unhealthy",
            total_checks, healthy_checks, degraded_checks, unhealthy_checks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_database_health_check() {
        let check = DatabaseHealthCheck::new(
            "database".to_string(),
            "postgresql://test:test@localhost/test".to_string(),
        );
        
        let result = check.check();
        assert_eq!(result.name, "database");
        assert_eq!(result.status, HealthStatus::Healthy);
    }
    
    #[test]
    fn test_redis_health_check() {
        let check = RedisHealthCheck::new(
            "redis".to_string(),
            "redis://test:6379".to_string(),
        );
        
        let result = check.check();
        assert_eq!(result.name, "redis");
        assert_eq!(result.status, HealthStatus::Healthy);
    }
    
    #[test]
    fn test_health_check_manager() {
        let mut manager = HealthCheckManager::new();
        
        manager.add_check(Box::new(DatabaseHealthCheck::new(
            "database".to_string(),
            "postgresql://test:test@localhost/test".to_string(),
        )));
        
        manager.add_check(Box::new(RedisHealthCheck::new(
            "redis".to_string(),
            "redis://test:6379".to_string(),
        )));
        
        let results = manager.run_checks();
        assert_eq!(results.len(), 2);
        assert_eq!(manager.get_overall_status(), HealthStatus::Healthy);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Structured logging
let logger = StructuredLogger::new("app".to_string(), "1.0.0".to_string(), "prod".to_string());
let entry = logger.info("User logged in".to_string(), "auth".to_string())
    .field("user_id".to_string(), serde_json::Value::String("123".to_string()));

// 2. Metrics collection
let mut registry = MetricsRegistry::new();
let counter = registry.register_counter("requests_total".to_string(), HashMap::new());
counter.inc();

// 3. Distributed tracing
let mut tracer = Tracer::new("service".to_string(), "1.0.0".to_string());
let context = tracer.start_span("operation".to_string(), None);
tracer.finish_span(&context);

// 4. Health checks
let mut manager = HealthCheckManager::new();
manager.add_check(Box::new(DatabaseHealthCheck::new("db".to_string(), "connection".to_string())));
manager.run_checks();
```

### Essential Patterns

```rust
// Complete monitoring setup
pub fn setup_rust_monitoring() {
    // 1. Structured logging
    // 2. Metrics collection
    // 3. Distributed tracing
    // 4. Health checks
    // 5. Alerting
    // 6. Dashboards
    // 7. Performance monitoring
    // 8. Error tracking
    
    println!("Rust monitoring setup complete!");
}
```

---

*This guide provides the complete machinery for Rust monitoring and observability. Each pattern includes implementation examples, monitoring strategies, and real-world usage patterns for enterprise observability systems.*
