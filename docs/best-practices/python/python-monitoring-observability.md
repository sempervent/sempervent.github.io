# Python Monitoring & Observability Best Practices

**Objective**: Master senior-level Python monitoring and observability patterns for production systems. When you need to implement comprehensive monitoring, when you want to build distributed tracing, when you need enterprise-grade observability strategies—these best practices become your weapon of choice.

## Core Principles

- **Observability**: Implement the three pillars (metrics, logs, traces)
- **Alerting**: Set up intelligent alerting with proper thresholds
- **Performance**: Monitor application performance and bottlenecks
- **Reliability**: Ensure system reliability through monitoring
- **Debugging**: Enable effective debugging through observability

## Structured Logging

### Advanced Logging Patterns

```python
# python/01-structured-logging.py

"""
Structured logging patterns and observability implementation
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
import structlog
from contextvars import ContextVar
import traceback
import sys
from functools import wraps

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class StructuredLogger:
    """Advanced structured logger"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.metrics = {}
        self.log_context = {}
    
    def set_context(self, **kwargs) -> None:
        """Set logging context"""
        self.log_context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context"""
        self.log_context.clear()
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        self.logger.info(message, **self.log_context, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context"""
        self.logger.warning(message, **self.log_context, **kwargs)
    
    def error(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log error message with context and exception"""
        error_data = {
            "error_type": type(exception).__name__ if exception else None,
            "error_message": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None
        }
        self.logger.error(message, **self.log_context, **error_data, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context"""
        self.logger.critical(message, **self.log_context, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        self.logger.debug(message, **self.log_context, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            **self.log_context,
            **kwargs
        )
    
    def log_business_event(self, event: str, **kwargs) -> None:
        """Log business events"""
        self.logger.info(
            "Business event",
            event=event,
            timestamp=datetime.utcnow().isoformat(),
            **self.log_context,
            **kwargs
        )

class RequestLogger:
    """Request-scoped logging"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.request_start_time = None
        self.request_metrics = {}
    
    def start_request(self, request_id: str = None, user_id: str = None, 
                     correlation_id: str = None) -> str:
        """Start request logging"""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Set context variables
        request_id_var.set(request_id)
        if user_id:
            user_id_var.set(user_id)
        if correlation_id:
            correlation_id_var.set(correlation_id)
        
        self.request_start_time = time.time()
        self.request_metrics = {
            "request_id": request_id,
            "user_id": user_id,
            "correlation_id": correlation_id,
            "start_time": datetime.utcnow().isoformat()
        }
        
        self.logger.set_context(**self.request_metrics)
        self.logger.info("Request started", **self.request_metrics)
        
        return request_id
    
    def end_request(self, status_code: int = 200, **kwargs) -> None:
        """End request logging"""
        if self.request_start_time is None:
            return
        
        duration = time.time() - self.request_start_time
        
        end_metrics = {
            "status_code": status_code,
            "duration": duration,
            "end_time": datetime.utcnow().isoformat()
        }
        
        self.logger.log_performance("request", duration, **end_metrics, **kwargs)
        self.logger.info("Request completed", **end_metrics, **kwargs)
        
        # Clear context
        self.logger.clear_context()
        request_id_var.set('')
        user_id_var.set('')
        correlation_id_var.set('')

class LoggingMiddleware:
    """Logging middleware for web applications"""
    
    def __init__(self, app, logger: StructuredLogger):
        self.app = app
        self.logger = logger
        self.request_logger = RequestLogger(logger)
    
    def __call__(self, environ, start_response):
        """WSGI middleware"""
        request_id = environ.get('HTTP_X_REQUEST_ID', str(uuid.uuid4()))
        user_id = environ.get('HTTP_X_USER_ID')
        correlation_id = environ.get('HTTP_X_CORRELATION_ID')
        
        # Start request logging
        self.request_logger.start_request(request_id, user_id, correlation_id)
        
        # Log request details
        self.logger.info(
            "HTTP request",
            method=environ.get('REQUEST_METHOD'),
            path=environ.get('PATH_INFO'),
            query_string=environ.get('QUERY_STRING'),
            user_agent=environ.get('HTTP_USER_AGENT'),
            remote_addr=environ.get('REMOTE_ADDR')
        )
        
        # Process request
        status_code = 200
        try:
            response = self.app(environ, start_response)
            return response
        except Exception as e:
            status_code = 500
            self.logger.error("Request failed", exception=e)
            raise
        finally:
            # End request logging
            self.request_logger.end_request(status_code)

def log_execution_time(operation: str):
    """Decorator to log execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = structlog.get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    "Function executed",
                    function=func.__name__,
                    operation=operation,
                    duration=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    operation=operation,
                    duration=duration,
                    success=False,
                    exception=str(e)
                )
                raise
        
        return wrapper
    return decorator

class LogAggregator:
    """Log aggregation utilities"""
    
    def __init__(self):
        self.logs = []
        self.metrics = {}
    
    def add_log(self, log_entry: Dict[str, Any]) -> None:
        """Add log entry to aggregator"""
        self.logs.append(log_entry)
        
        # Update metrics
        level = log_entry.get('level', 'INFO')
        if level not in self.metrics:
            self.metrics[level] = 0
        self.metrics[level] += 1
    
    def get_logs_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get logs by level"""
        return [log for log in self.logs if log.get('level') == level]
    
    def get_logs_by_time_range(self, start_time: datetime, 
                              end_time: datetime) -> List[Dict[str, Any]]:
        """Get logs by time range"""
        return [
            log for log in self.logs
            if start_time <= datetime.fromisoformat(log.get('timestamp', '')) <= end_time
        ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        error_logs = self.get_logs_by_level('ERROR')
        critical_logs = self.get_logs_by_level('CRITICAL')
        
        return {
            "total_errors": len(error_logs),
            "total_critical": len(critical_logs),
            "error_rate": len(error_logs) / len(self.logs) if self.logs else 0,
            "recent_errors": error_logs[-10:] if error_logs else []
        }

# Usage examples
def example_structured_logging():
    """Example structured logging usage"""
    # Create structured logger
    logger = StructuredLogger("my_app")
    
    # Set context
    logger.set_context(service="user_service", version="1.0.0")
    
    # Log messages
    logger.info("Application started", port=8000, environment="production")
    logger.warning("High memory usage detected", memory_usage=85.5)
    logger.error("Database connection failed", exception=Exception("Connection timeout"))
    
    # Log performance
    logger.log_performance("database_query", 0.150, query="SELECT * FROM users")
    logger.log_business_event("user_registration", user_id="12345", email="user@example.com")
    
    # Request logging
    request_logger = RequestLogger(logger)
    request_id = request_logger.start_request(user_id="12345")
    
    # Simulate request processing
    time.sleep(0.1)
    
    request_logger.end_request(status_code=200, response_size=1024)
    
    # Log aggregation
    aggregator = LogAggregator()
    aggregator.add_log({"level": "INFO", "message": "Test log", "timestamp": datetime.utcnow().isoformat()})
    aggregator.add_log({"level": "ERROR", "message": "Test error", "timestamp": datetime.utcnow().isoformat()})
    
    error_summary = aggregator.get_error_summary()
    print(f"Error summary: {error_summary}")
```

### Metrics Collection

```python
# python/02-metrics-collection.py

"""
Metrics collection and monitoring patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import psutil
import requests
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from prometheus_client.core import CollectorRegistry
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Advanced metrics collector"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.custom_metrics = {}
        self.metrics_thread = None
        self.running = False
    
    def create_counter(self, name: str, description: str, labels: List[str] = None) -> Counter:
        """Create counter metric"""
        counter = Counter(
            name,
            description,
            labels or [],
            registry=self.registry
        )
        self.metrics[name] = counter
        return counter
    
    def create_histogram(self, name: str, description: str, 
                        buckets: List[float] = None, labels: List[str] = None) -> Histogram:
        """Create histogram metric"""
        histogram = Histogram(
            name,
            description,
            labels or [],
            buckets=buckets,
            registry=self.registry
        )
        self.metrics[name] = histogram
        return histogram
    
    def create_gauge(self, name: str, description: str, labels: List[str] = None) -> Gauge:
        """Create gauge metric"""
        gauge = Gauge(
            name,
            description,
            labels or [],
            registry=self.registry
        )
        self.metrics[name] = gauge
        return gauge
    
    def create_summary(self, name: str, description: str, labels: List[str] = None) -> Summary:
        """Create summary metric"""
        summary = Summary(
            name,
            description,
            labels or [],
            registry=self.registry
        )
        self.metrics[name] = summary
        return summary
    
    def increment_counter(self, name: str, value: float = 1, labels: Dict[str, str] = None) -> None:
        """Increment counter metric"""
        if name in self.metrics and isinstance(self.metrics[name], Counter):
            if labels:
                self.metrics[name].labels(**labels).inc(value)
            else:
                self.metrics[name].inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe histogram metric"""
        if name in self.metrics and isinstance(self.metrics[name], Histogram):
            if labels:
                self.metrics[name].labels(**labels).observe(value)
            else:
                self.metrics[name].observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set gauge metric value"""
        if name in self.metrics and isinstance(self.metrics[name], Gauge):
            if labels:
                self.metrics[name].labels(**labels).set(value)
            else:
                self.metrics[name].set(value)
    
    def observe_summary(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Observe summary metric"""
        if name in self.metrics and isinstance(self.metrics[name], Summary):
            if labels:
                self.metrics[name].labels(**labels).observe(value)
            else:
                self.metrics[name].observe(value)
    
    def start_metrics_server(self, port: int = 8000) -> None:
        """Start metrics HTTP server"""
        start_http_server(port, registry=self.registry)
        logger.info(f"Metrics server started on port {port}")
    
    def start_system_metrics_collection(self, interval: int = 30) -> None:
        """Start system metrics collection"""
        self.running = True
        self.metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval,),
            daemon=True
        )
        self.metrics_thread.start()
    
    def stop_system_metrics_collection(self) -> None:
        """Stop system metrics collection"""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join()
    
    def _collect_system_metrics(self, interval: int) -> None:
        """Collect system metrics"""
        while self.running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system_cpu_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.set_gauge("system_memory_percent", memory.percent)
                self.set_gauge("system_memory_used_bytes", memory.used)
                self.set_gauge("system_memory_available_bytes", memory.available)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.set_gauge("system_disk_percent", disk.percent)
                self.set_gauge("system_disk_used_bytes", disk.used)
                self.set_gauge("system_disk_free_bytes", disk.free)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.set_gauge("system_network_bytes_sent", network.bytes_sent)
                self.set_gauge("system_network_bytes_recv", network.bytes_recv)
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(interval)

class ApplicationMetrics:
    """Application-specific metrics"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.setup_metrics()
    
    def setup_metrics(self) -> None:
        """Setup application metrics"""
        # HTTP metrics
        self.collector.create_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"]
        )
        
        self.collector.create_histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Database metrics
        self.collector.create_counter(
            "database_queries_total",
            "Total database queries",
            ["operation", "table", "status"]
        )
        
        self.collector.create_histogram(
            "database_query_duration_seconds",
            "Database query duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        # Business metrics
        self.collector.create_counter(
            "business_events_total",
            "Total business events",
            ["event_type", "status"]
        )
        
        self.collector.create_gauge(
            "active_users",
            "Number of active users"
        )
        
        self.collector.create_gauge(
            "queue_size",
            "Queue size",
            ["queue_name"]
        )
    
    def record_http_request(self, method: str, endpoint: str, 
                           status_code: int, duration: float) -> None:
        """Record HTTP request metrics"""
        self.collector.increment_counter(
            "http_requests_total",
            labels={"method": method, "endpoint": endpoint, "status_code": str(status_code)}
        )
        self.collector.observe_histogram("http_request_duration_seconds", duration)
    
    def record_database_query(self, operation: str, table: str, 
                            status: str, duration: float) -> None:
        """Record database query metrics"""
        self.collector.increment_counter(
            "database_queries_total",
            labels={"operation": operation, "table": table, "status": status}
        )
        self.collector.observe_histogram("database_query_duration_seconds", duration)
    
    def record_business_event(self, event_type: str, status: str) -> None:
        """Record business event metrics"""
        self.collector.increment_counter(
            "business_events_total",
            labels={"event_type": event_type, "status": status}
        )
    
    def update_active_users(self, count: int) -> None:
        """Update active users count"""
        self.collector.set_gauge("active_users", count)
    
    def update_queue_size(self, queue_name: str, size: int) -> None:
        """Update queue size"""
        self.collector.set_gauge("queue_size", size, labels={"queue_name": queue_name})

class MetricsMiddleware:
    """Metrics middleware for web applications"""
    
    def __init__(self, app, metrics: ApplicationMetrics):
        self.app = app
        self.metrics = metrics
    
    def __call__(self, environ, start_response):
        """WSGI middleware for metrics collection"""
        start_time = time.time()
        
        # Process request
        response = self.app(environ, start_response)
        
        # Calculate metrics
        duration = time.time() - start_time
        method = environ.get('REQUEST_METHOD', 'UNKNOWN')
        path = environ.get('PATH_INFO', '/')
        status_code = 200  # This would be extracted from response
        
        # Record metrics
        self.metrics.record_http_request(method, path, status_code, duration)
        
        return response

class HealthChecker:
    """Health check utilities"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.setup_health_metrics()
    
    def setup_health_metrics(self) -> None:
        """Setup health check metrics"""
        self.metrics_collector.create_gauge(
            "health_check_status",
            "Health check status (1=healthy, 0=unhealthy)",
            ["check_name"]
        )
        
        self.metrics_collector.create_histogram(
            "health_check_duration_seconds",
            "Health check duration",
            ["check_name"]
        )
    
    def add_health_check(self, name: str, check_func: callable) -> None:
        """Add health check"""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            start_time = time.time()
            
            try:
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    "healthy": is_healthy,
                    "duration": duration,
                    "error": None
                }
                
                # Record metrics
                self.metrics_collector.set_gauge(
                    "health_check_status",
                    1 if is_healthy else 0,
                    labels={"check_name": name}
                )
                self.metrics_collector.observe_histogram(
                    "health_check_duration_seconds",
                    duration,
                    labels={"check_name": name}
                )
                
            except Exception as e:
                duration = time.time() - start_time
                results[name] = {
                    "healthy": False,
                    "duration": duration,
                    "error": str(e)
                }
                
                # Record metrics
                self.metrics_collector.set_gauge(
                    "health_check_status",
                    0,
                    labels={"check_name": name}
                )
                self.metrics_collector.observe_histogram(
                    "health_check_duration_seconds",
                    duration,
                    labels={"check_name": name}
                )
        
        return results

# Usage examples
def example_metrics_collection():
    """Example metrics collection usage"""
    # Create metrics collector
    collector = MetricsCollector("my_app")
    
    # Setup application metrics
    app_metrics = ApplicationMetrics(collector)
    
    # Start metrics server
    collector.start_metrics_server(port=8000)
    
    # Start system metrics collection
    collector.start_system_metrics_collection(interval=30)
    
    # Record some metrics
    app_metrics.record_http_request("GET", "/users", 200, 0.150)
    app_metrics.record_database_query("SELECT", "users", "success", 0.050)
    app_metrics.record_business_event("user_registration", "success")
    app_metrics.update_active_users(150)
    app_metrics.update_queue_size("email_queue", 25)
    
    # Setup health checks
    health_checker = HealthChecker(collector)
    
    def check_database():
        # Simulate database check
        return True
    
    def check_redis():
        # Simulate Redis check
        return True
    
    health_checker.add_health_check("database", check_database)
    health_checker.add_health_check("redis", check_redis)
    
    # Run health checks
    health_results = health_checker.run_health_checks()
    print(f"Health check results: {health_results}")
    
    # Stop metrics collection
    collector.stop_system_metrics_collection()
```

### Distributed Tracing

```python
# python/03-distributed-tracing.py

"""
Distributed tracing implementation and patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import time
import uuid
import json
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass, asdict
import threading
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)

# Context variables for tracing
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')
span_id_var: ContextVar[str] = ContextVar('span_id', default='')
parent_span_id_var: ContextVar[str] = ContextVar('parent_span_id', default='')

@dataclass
class Span:
    """Span definition for distributed tracing"""
    trace_id: str
    span_id: str
    parent_span_id: str = None
    operation_name: str = ""
    start_time: datetime = None
    end_time: datetime = None
    duration: float = 0.0
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    status: str = "ok"
    error: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
        if self.start_time is None:
            self.start_time = datetime.utcnow()
    
    def finish(self, status: str = "ok", error: str = None) -> None:
        """Finish the span"""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = status
        self.error = error
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, **kwargs) -> None:
        """Add log to span"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return asdict(self)

class Tracer:
    """Distributed tracer implementation"""
    
    def __init__(self, service_name: str, endpoint: str = None):
        self.service_name = service_name
        self.endpoint = endpoint
        self.spans = []
        self.active_spans = {}
        self.trace_context = {}
    
    def start_span(self, operation_name: str, parent_span_id: str = None, 
                  tags: Dict[str, Any] = None) -> Span:
        """Start a new span"""
        trace_id = trace_id_var.get() or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        if parent_span_id is None:
            parent_span_id = span_id_var.get()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        # Set context variables
        trace_id_var.set(trace_id)
        span_id_var.set(span_id)
        parent_span_id_var.set(parent_span_id)
        
        # Add service tag
        span.add_tag("service.name", self.service_name)
        
        # Store active span
        self.active_spans[span_id] = span
        
        return span
    
    def finish_span(self, span: Span, status: str = "ok", error: str = None) -> None:
        """Finish a span"""
        span.finish(status, error)
        
        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        # Add to completed spans
        self.spans.append(span)
        
        # Send to tracing backend if endpoint is configured
        if self.endpoint:
            self._send_span(span)
    
    def _send_span(self, span: Span) -> None:
        """Send span to tracing backend"""
        try:
            if self.endpoint:
                requests.post(
                    self.endpoint,
                    json=span.to_dict(),
                    timeout=5
                )
        except Exception as e:
            logger.error(f"Failed to send span to tracing backend: {e}")
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        return [span for span in self.spans if span.trace_id == trace_id]
    
    def get_active_spans(self) -> List[Span]:
        """Get all active spans"""
        return list(self.active_spans.values())
    
    def create_child_span(self, operation_name: str, tags: Dict[str, Any] = None) -> Span:
        """Create child span of current active span"""
        current_span_id = span_id_var.get()
        return self.start_span(operation_name, current_span_id, tags)

class TraceMiddleware:
    """Tracing middleware for web applications"""
    
    def __init__(self, app, tracer: Tracer):
        self.app = app
        self.tracer = tracer
    
    def __call__(self, environ, start_response):
        """WSGI middleware for tracing"""
        # Start root span
        span = self.tracer.start_span(
            f"{environ.get('REQUEST_METHOD')} {environ.get('PATH_INFO')}",
            tags={
                "http.method": environ.get('REQUEST_METHOD'),
                "http.url": environ.get('PATH_INFO'),
                "http.user_agent": environ.get('HTTP_USER_AGENT'),
                "http.remote_addr": environ.get('REMOTE_ADDR')
            }
        )
        
        try:
            # Process request
            response = self.app(environ, start_response)
            
            # Finish span successfully
            self.tracer.finish_span(span, status="ok")
            
            return response
        except Exception as e:
            # Finish span with error
            self.tracer.finish_span(span, status="error", error=str(e))
            raise

def trace_function(operation_name: str = None):
    """Decorator to trace function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = Tracer("my_app")
            span_name = operation_name or func.__name__
            
            span = tracer.start_span(span_name, tags={
                "function.name": func.__name__,
                "function.module": func.__module__
            })
            
            try:
                result = func(*args, **kwargs)
                tracer.finish_span(span, status="ok")
                return result
            except Exception as e:
                tracer.finish_span(span, status="error", error=str(e))
                raise
        
        return wrapper
    return decorator

class TraceAnalyzer:
    """Trace analysis utilities"""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """Analyze a trace"""
        spans = self.tracer.get_trace(trace_id)
        
        if not spans:
            return {"error": "Trace not found"}
        
        # Calculate trace duration
        start_times = [span.start_time for span in spans]
        end_times = [span.end_time for span in spans if span.end_time]
        
        if not end_times:
            return {"error": "Trace not completed"}
        
        trace_start = min(start_times)
        trace_end = max(end_times)
        trace_duration = (trace_end - trace_start).total_seconds()
        
        # Find root span
        root_spans = [span for span in spans if span.parent_span_id is None]
        root_span = root_spans[0] if root_spans else spans[0]
        
        # Analyze span hierarchy
        span_tree = self._build_span_tree(spans)
        
        # Find slow spans
        slow_spans = [span for span in spans if span.duration > 1.0]
        
        # Find error spans
        error_spans = [span for span in spans if span.status == "error"]
        
        return {
            "trace_id": trace_id,
            "duration": trace_duration,
            "span_count": len(spans),
            "root_operation": root_span.operation_name,
            "slow_spans": [{"operation": span.operation_name, "duration": span.duration} for span in slow_spans],
            "error_spans": [{"operation": span.operation_name, "error": span.error} for span in error_spans],
            "span_tree": span_tree
        }
    
    def _build_span_tree(self, spans: List[Span]) -> Dict[str, Any]:
        """Build span hierarchy tree"""
        span_dict = {span.span_id: span for span in spans}
        root_spans = [span for span in spans if span.parent_span_id is None]
        
        def build_tree(span: Span) -> Dict[str, Any]:
            children = [s for s in spans if s.parent_span_id == span.span_id]
            return {
                "span_id": span.span_id,
                "operation": span.operation_name,
                "duration": span.duration,
                "status": span.status,
                "children": [build_tree(child) for child in children]
            }
        
        return [build_tree(span) for span in root_spans]

# Usage examples
def example_distributed_tracing():
    """Example distributed tracing usage"""
    # Create tracer
    tracer = Tracer("my_app", endpoint="http://localhost:14268/api/traces")
    
    # Start root span
    root_span = tracer.start_span("user_registration", tags={"user_id": "12345"})
    
    # Create child span
    child_span = tracer.create_child_span("validate_user_data", tags={"validation_type": "email"})
    time.sleep(0.1)  # Simulate work
    tracer.finish_span(child_span)
    
    # Create another child span
    child_span2 = tracer.create_child_span("save_user", tags={"table": "users"})
    time.sleep(0.2)  # Simulate work
    tracer.finish_span(child_span2)
    
    # Finish root span
    tracer.finish_span(root_span)
    
    # Analyze trace
    analyzer = TraceAnalyzer(tracer)
    trace_analysis = analyzer.analyze_trace(root_span.trace_id)
    print(f"Trace analysis: {trace_analysis}")
    
    # Function tracing
    @trace_function("database_query")
    def query_database(query: str):
        time.sleep(0.1)
        return {"result": "success"}
    
    result = query_database("SELECT * FROM users")
    print(f"Database query result: {result}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Structured logging
logger = StructuredLogger("my_app")
logger.set_context(service="user_service", version="1.0.0")
logger.info("Application started", port=8000)

# 2. Metrics collection
collector = MetricsCollector("my_app")
collector.start_metrics_server(port=8000)
app_metrics = ApplicationMetrics(collector)
app_metrics.record_http_request("GET", "/users", 200, 0.150)

# 3. Distributed tracing
tracer = Tracer("my_app", endpoint="http://localhost:14268/api/traces")
span = tracer.start_span("user_registration")
tracer.finish_span(span)

# 4. Health checks
health_checker = HealthChecker(collector)
health_checker.add_health_check("database", check_database)
health_results = health_checker.run_health_checks()

# 5. Trace analysis
analyzer = TraceAnalyzer(tracer)
trace_analysis = analyzer.analyze_trace(trace_id)
```

### Essential Patterns

```python
# Complete monitoring setup
def setup_monitoring_observability():
    """Setup complete monitoring and observability environment"""
    
    # Structured logger
    logger = StructuredLogger("my_app")
    
    # Metrics collector
    collector = MetricsCollector("my_app")
    
    # Application metrics
    app_metrics = ApplicationMetrics(collector)
    
    # Distributed tracer
    tracer = Tracer("my_app")
    
    # Health checker
    health_checker = HealthChecker(collector)
    
    # Trace analyzer
    analyzer = TraceAnalyzer(tracer)
    
    print("Monitoring and observability setup complete!")
```

---

*This guide provides the complete machinery for Python monitoring and observability. Each pattern includes implementation examples, monitoring strategies, and real-world usage patterns for enterprise observability management.*
