# Performance Monitoring Best Practices

This document establishes comprehensive performance monitoring strategies for geospatial systems, covering application monitoring, metrics collection, alerting, and observability patterns.

## Application Performance Monitoring

### Custom Metrics Collection

```python
import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any
import json

class PerformanceMonitor:
    """Performance monitoring for geospatial applications"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def track_performance(self, operation_name: str):
        """Decorator to track function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success metrics
                    self.record_metric(operation_name, {
                        'status': 'success',
                        'execution_time': time.time() - start_time,
                        'memory_usage': psutil.Process().memory_info().rss - start_memory,
                        'timestamp': time.time()
                    })
                    
                    return result
                    
                except Exception as e:
                    # Record error metrics
                    self.record_metric(operation_name, {
                        'status': 'error',
                        'execution_time': time.time() - start_time,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    raise
            
            return wrapper
        return decorator
    
    def record_metric(self, operation: str, metrics: Dict[str, Any]):
        """Record performance metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(metrics)
        
        # Log metrics
        self.logger.info(f"Performance metric: {operation} - {json.dumps(metrics)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, metrics in self.metrics.items():
            if metrics:
                execution_times = [m['execution_time'] for m in metrics if 'execution_time' in m]
                success_count = len([m for m in metrics if m.get('status') == 'success'])
                error_count = len([m for m in metrics if m.get('status') == 'error'])
                
                summary[operation] = {
                    'total_executions': len(metrics),
                    'success_count': success_count,
                    'error_count': error_count,
                    'success_rate': success_count / len(metrics) if metrics else 0,
                    'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                    'min_execution_time': min(execution_times) if execution_times else 0,
                    'max_execution_time': max(execution_times) if execution_times else 0
                }
        
        return summary

# Usage example
monitor = PerformanceMonitor()

@monitor.track_performance('spatial_analysis')
def perform_spatial_analysis(data):
    """Example function with performance monitoring"""
    # Implementation
    time.sleep(0.1)  # Simulate processing
    return {"result": "analysis_complete"}

# Get performance summary
summary = monitor.get_performance_summary()
print(json.dumps(summary, indent=2))
```

**Why:** Custom metrics provide application-specific insights. Decorators enable non-intrusive performance tracking. Historical metrics enable trend analysis and capacity planning.

### Prometheus Metrics Integration

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('geospatial_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('geospatial_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('geospatial_active_connections', 'Active connections')
SPATIAL_QUERIES = Counter('geospatial_spatial_queries_total', 'Spatial queries', ['operation'])

def track_request(func):
    """Decorator to track request metrics"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

# Start metrics server
start_http_server(8001)
```

**Why:** Prometheus provides standardized metrics collection and querying. Histograms enable percentile analysis and SLA monitoring.

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Geospatial API Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(geospatial_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(geospatial_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Spatial Query Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(geospatial_spatial_queries_total[5m])",
            "legendFormat": "{{operation}}"
          }
        ]
      }
    ]
  }
}
```

**Why:** Grafana provides rich visualization of metrics data. Dashboards enable real-time monitoring and historical analysis.

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=14268,
)

# Add span processor
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Custom tracing
@tracer.start_as_current_span("spatial_analysis")
def perform_spatial_analysis(data):
    """Perform spatial analysis with tracing"""
    with tracer.start_as_current_span("data_validation"):
        validate_data(data)
    
    with tracer.start_as_current_span("spatial_processing"):
        result = process_spatial_data(data)
    
    with tracer.start_as_current_span("result_formatting"):
        formatted_result = format_result(result)
    
    return formatted_result
```

**Why:** Distributed tracing provides end-to-end visibility across microservices. OpenTelemetry provides vendor-neutral observability.

### Custom Span Attributes

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

def trace_spatial_operation(operation_name: str):
    """Decorator to trace spatial operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with trace.get_tracer(__name__).start_as_current_span(operation_name) as span:
                try:
                    # Add custom attributes
                    span.set_attribute("operation.type", "spatial")
                    span.set_attribute("operation.name", operation_name)
                    
                    # Add input parameters
                    if args:
                        span.set_attribute("input.arg_count", len(args))
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Add result attributes
                    if isinstance(result, dict):
                        span.set_attribute("result.keys", list(result.keys()))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator

# Usage
@trace_spatial_operation("buffer_analysis")
def perform_buffer_analysis(point, distance):
    """Perform buffer analysis with tracing"""
    # Implementation
    pass
```

**Why:** Custom attributes provide context-specific information. Error tracking enables debugging and issue resolution.

## Logging and Structured Data

### Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured logging for geospatial applications"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Configure JSON formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_structured(self, level: str, message: str, **kwargs):
        """Log structured data"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))
    
    def log_spatial_operation(self, operation: str, **kwargs):
        """Log spatial operation"""
        self.log_structured('INFO', f"Spatial operation: {operation}", 
                          operation=operation, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.log_structured('INFO', f"Performance: {operation}", 
                          operation=operation, duration=duration, **kwargs)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.log_structured('ERROR', f"Error: {str(error)}", 
                          error_type=type(error).__name__,
                          error_message=str(error),
                          context=context or {})

# Usage
logger = StructuredLogger(__name__)

def process_spatial_data(data):
    """Process spatial data with structured logging"""
    logger.log_spatial_operation("data_processing", 
                                input_size=len(data),
                                data_type=type(data).__name__)
    
    try:
        # Process data
        result = perform_processing(data)
        
        logger.log_performance("data_processing", 
                              duration=0.5,
                              output_size=len(result))
        
        return result
        
    except Exception as e:
        logger.log_error(e, context={"input_data": str(data)[:100]})
        raise
```

**Why:** Structured logging enables log aggregation and analysis. JSON format facilitates parsing and querying.

### Log Aggregation and Analysis

```python
import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path

class LogAnalyzer:
    """Analyze application logs"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.logs = []
    
    def load_logs(self):
        """Load logs from file"""
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    self.logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    def analyze_error_patterns(self):
        """Analyze error patterns in logs"""
        error_logs = [log for log in self.logs if log.get('level') == 'ERROR']
        
        error_counts = {}
        for log in error_logs:
            error_type = log.get('error_type', 'Unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
    def analyze_performance_trends(self):
        """Analyze performance trends"""
        performance_logs = [log for log in self.logs if 'duration' in log]
        
        if not performance_logs:
            return {}
        
        durations = [log['duration'] for log in performance_logs]
        
        return {
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'total_operations': len(performance_logs)
        }
    
    def generate_report(self):
        """Generate analysis report"""
        self.load_logs()
        
        error_patterns = self.analyze_error_patterns()
        performance_trends = self.analyze_performance_trends()
        
        report = {
            'total_logs': len(self.logs),
            'error_patterns': error_patterns,
            'performance_trends': performance_trends,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        return report

# Usage
analyzer = LogAnalyzer('application.log')
report = analyzer.generate_report()
print(json.dumps(report, indent=2))
```

**Why:** Log analysis identifies patterns and trends. Automated analysis enables proactive issue detection.

## Alerting and Notification

### Alert Configuration

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from typing import Dict, Any, List

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add alert rule"""
        self.alert_rules.append(rule)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if self.evaluate_rule(rule, metrics):
                triggered_alerts.append(rule)
        
        return triggered_alerts
    
    def evaluate_rule(self, rule: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Evaluate alert rule"""
        metric_name = rule['metric']
        threshold = rule['threshold']
        operator = rule['operator']
        
        if metric_name not in metrics:
            return False
        
        value = metrics[metric_name]
        
        if operator == 'gt':
            return value > threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'eq':
            return value == threshold
        elif operator == 'ne':
            return value != threshold
        
        return False
    
    def send_alert(self, alert: Dict[str, Any], metrics: Dict[str, Any]):
        """Send alert notification"""
        if self.config.get('email', {}).get('enabled'):
            self.send_email_alert(alert, metrics)
        
        if self.config.get('slack', {}).get('enabled'):
            self.send_slack_alert(alert, metrics)
    
    def send_email_alert(self, alert: Dict[str, Any], metrics: Dict[str, Any]):
        """Send email alert"""
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = alert.get('recipients', email_config['to'])
        msg['Subject'] = f"Alert: {alert['name']}"
        
        body = f"""
        Alert: {alert['name']}
        Description: {alert.get('description', '')}
        
        Current Metrics:
        {json.dumps(metrics, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
    
    def send_slack_alert(self, alert: Dict[str, Any], metrics: Dict[str, Any]):
        """Send Slack alert"""
        slack_config = self.config['slack']
        
        payload = {
            'text': f"Alert: {alert['name']}",
            'attachments': [
                {
                    'color': 'danger',
                    'fields': [
                        {'title': 'Description', 'value': alert.get('description', ''), 'short': False},
                        {'title': 'Current Value', 'value': str(metrics.get(alert['metric'], 'N/A')), 'short': True},
                        {'title': 'Threshold', 'value': str(alert['threshold']), 'short': True}
                    ]
                }
            ]
        }
        
        requests.post(slack_config['webhook_url'], json=payload)

# Usage
alert_manager = AlertManager({
    'email': {
        'enabled': True,
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-password',
        'from': 'alerts@yourcompany.com',
        'to': 'admin@yourcompany.com'
    },
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/...'
    }
})

# Add alert rules
alert_manager.add_alert_rule({
    'name': 'High Error Rate',
    'metric': 'error_rate',
    'threshold': 0.05,
    'operator': 'gt',
    'description': 'Error rate exceeds 5%',
    'recipients': ['admin@yourcompany.com']
})

alert_manager.add_alert_rule({
    'name': 'Slow Response Time',
    'metric': 'avg_response_time',
    'threshold': 5.0,
    'operator': 'gt',
    'description': 'Average response time exceeds 5 seconds'
})

# Check alerts
current_metrics = {
    'error_rate': 0.03,
    'avg_response_time': 2.1,
    'request_count': 1000
}

triggered_alerts = alert_manager.check_alerts(current_metrics)
for alert in triggered_alerts:
    alert_manager.send_alert(alert, current_metrics)
```

**Why:** Automated alerting enables proactive issue detection. Multi-channel notifications ensure alerts reach the right people.

## Health Checks and Monitoring

### Comprehensive Health Checks

```python
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import psutil
import time

class HealthChecker:
    """Comprehensive health checking"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register health check"""
        self.checks[name] = check_func
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Test database connection
            # Implementation depends on your database
            return {'status': 'healthy', 'response_time': 0.1}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            # Test Redis connection
            # Implementation depends on your Redis setup
            return {'status': 'healthy', 'response_time': 0.05}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        status = 'healthy'
        if cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
            status = 'unhealthy'
        
        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = 'healthy'
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result
                
                if result.get('status') != 'healthy':
                    overall_status = 'unhealthy'
                    
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
                overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'checks': results
        }

# Usage
health_checker = HealthChecker()
health_checker.register_check('database', health_checker.check_database_health)
health_checker.register_check('redis', health_checker.check_redis_health)
health_checker.register_check('system', health_checker.check_system_resources)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = health_checker.run_all_checks()
    
    if health_status['overall_status'] != 'healthy':
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status
```

**Why:** Health checks provide system status visibility. Comprehensive checks enable targeted troubleshooting and capacity planning.
