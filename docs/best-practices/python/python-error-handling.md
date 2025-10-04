# Python Error Handling Best Practices

**Objective**: Master senior-level Python error handling patterns for production systems. When you need to build robust applications that handle failures gracefully, when you want to implement comprehensive error recovery, when you need enterprise-grade error handling strategies—these best practices become your weapon of choice.

## Core Principles

- **Fail Fast**: Detect errors early and fail fast
- **Graceful Degradation**: Provide fallback mechanisms when possible
- **Comprehensive Logging**: Log errors with sufficient context for debugging
- **Error Recovery**: Implement retry mechanisms and circuit breakers
- **User Experience**: Provide meaningful error messages to users

## Exception Handling Patterns

### Custom Exception Classes

```python
# python/01-exception-handling.py

"""
Advanced exception handling patterns and custom exception classes
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Callable, Type
from enum import Enum
from dataclasses import dataclass
import functools
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Error context information"""
    timestamp: float
    function_name: str
    line_number: int
    error_code: str
    severity: ErrorSeverity
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

class BaseApplicationError(Exception):
    """Base class for application-specific errors"""
    
    def __init__(self, message: str, error_code: str = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[ErrorContext] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context
        self.cause = cause
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "context": self.context.__dict__ if self.context else None,
            "cause": str(self.cause) if self.cause else None
        }

class ValidationError(BaseApplicationError):
    """Validation error"""
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.LOW, **kwargs)
        self.field = field
        self.value = value

class BusinessLogicError(BaseApplicationError):
    """Business logic error"""
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.operation = operation

class ExternalServiceError(BaseApplicationError):
    """External service error"""
    def __init__(self, message: str, service_name: str = None, status_code: int = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.service_name = service_name
        self.status_code = status_code

class SystemError(BaseApplicationError):
    """System error"""
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)
        self.component = component

class ErrorHandler:
    """Advanced error handler"""
    
    def __init__(self):
        self.error_handlers = {}
        self.default_handler = self._default_handler
        self.error_metrics = {}
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register error handler for specific exception type"""
        self.error_handlers[exception_type] = handler
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """Handle error with appropriate handler"""
        # Get handler for exception type
        handler = self.error_handlers.get(type(error), self.default_handler)
        
        # Create error context if not provided
        if not context:
            context = self._create_error_context(error)
        
        # Update error metrics
        self._update_error_metrics(error)
        
        # Call handler
        try:
            return handler(error, context)
        except Exception as handler_error:
            logger.error(f"Error in error handler: {handler_error}")
            return self._default_handler(error, context)
    
    def _default_handler(self, error: Exception, context: ErrorContext) -> None:
        """Default error handler"""
        logger.error(f"Unhandled error: {error}", exc_info=True)
        
        # Log error details
        error_details = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context.__dict__ if context else None,
            "traceback": traceback.format_exc()
        }
        
        logger.error(f"Error details: {error_details}")
    
    def _create_error_context(self, error: Exception) -> ErrorContext:
        """Create error context from exception"""
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            frame = tb[-1]
            return ErrorContext(
                timestamp=time.time(),
                function_name=frame.name,
                line_number=frame.lineno,
                error_code=type(error).__name__,
                severity=ErrorSeverity.MEDIUM
            )
        return ErrorContext(
            timestamp=time.time(),
            function_name="unknown",
            line_number=0,
            error_code=type(error).__name__,
            severity=ErrorSeverity.MEDIUM
        )
    
    def _update_error_metrics(self, error: Exception) -> None:
        """Update error metrics"""
        error_type = type(error).__name__
        if error_type not in self.error_metrics:
            self.error_metrics[error_type] = 0
        self.error_metrics[error_type] += 1
    
    def get_error_metrics(self) -> Dict[str, int]:
        """Get error metrics"""
        return self.error_metrics.copy()

# Usage examples
def example_custom_exceptions():
    """Example custom exception usage"""
    # Validation error
    try:
        if not isinstance("invalid", int):
            raise ValidationError("Expected integer", field="value", value="invalid")
    except ValidationError as e:
        print(f"Validation error: {e.to_dict()}")
    
    # Business logic error
    try:
        if 10 > 5:
            raise BusinessLogicError("Business rule violated", operation="validation")
    except BusinessLogicError as e:
        print(f"Business logic error: {e.to_dict()}")
    
    # External service error
    try:
        raise ExternalServiceError("API call failed", service_name="payment_api", status_code=500)
    except ExternalServiceError as e:
        print(f"External service error: {e.to_dict()}")

def example_error_handler():
    """Example error handler usage"""
    handler = ErrorHandler()
    
    # Register custom handlers
    def handle_validation_error(error: ValidationError, context: ErrorContext):
        print(f"Handling validation error: {error.message}")
        return {"status": "validation_failed", "field": error.field}
    
    def handle_business_error(error: BusinessLogicError, context: ErrorContext):
        print(f"Handling business logic error: {error.message}")
        return {"status": "business_rule_violated", "operation": error.operation}
    
    handler.register_handler(ValidationError, handle_validation_error)
    handler.register_handler(BusinessLogicError, handle_business_error)
    
    # Handle errors
    try:
        raise ValidationError("Invalid input", field="email")
    except ValidationError as e:
        result = handler.handle_error(e)
        print(f"Handler result: {result}")
    
    # Get error metrics
    metrics = handler.get_error_metrics()
    print(f"Error metrics: {metrics}")
```

### Error Recovery Patterns

```python
# python/02-error-recovery.py

"""
Advanced error recovery patterns including retry mechanisms and circuit breakers
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Callable, Type
from enum import Enum
from dataclasses import dataclass
import functools
import logging

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    exceptions: tuple = (Exception,)

class RetryManager:
    """Advanced retry manager"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempts = 0
        self.last_error = None
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        else:
            delay = self.config.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if error should be retried"""
        if self.attempts >= self.config.max_attempts:
            return False
        
        if not isinstance(error, self.config.exceptions):
            return False
        
        return True
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry function with configured strategy"""
        self.attempts = 0
        self.last_error = None
        
        while self.attempts < self.config.max_attempts:
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.last_error = e
                self.attempts += 1
                
                if not self.should_retry(e):
                    raise e
                
                if self.attempts < self.config.max_attempts:
                    delay = self.calculate_delay(self.attempts - 1)
                    logger.warning(f"Attempt {self.attempts} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
                    raise e
        
        raise self.last_error

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, 
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.state != CircuitBreakerState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout
        }

class FallbackManager:
    """Fallback mechanism manager"""
    
    def __init__(self):
        self.fallbacks = {}
        self.default_fallback = None
    
    def register_fallback(self, exception_type: Type[Exception], fallback_func: Callable):
        """Register fallback function for exception type"""
        self.fallbacks[exception_type] = fallback_func
    
    def set_default_fallback(self, fallback_func: Callable):
        """Set default fallback function"""
        self.default_fallback = fallback_func
    
    def execute_with_fallback(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            fallback_func = self.fallbacks.get(type(e), self.default_fallback)
            if fallback_func:
                logger.warning(f"Using fallback for {type(e).__name__}: {e}")
                return fallback_func(*args, **kwargs)
            else:
                raise e

# Usage examples
def example_retry_mechanism():
    """Example retry mechanism usage"""
    def unreliable_function():
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Random failure")
        return "Success"
    
    # Configure retry
    retry_config = RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        backoff_multiplier=2.0,
        jitter=True
    )
    
    retry_manager = RetryManager(retry_config)
    
    try:
        result = retry_manager.retry(unreliable_function)
        print(f"Retry result: {result}")
    except Exception as e:
        print(f"Retry failed: {e}")

def example_circuit_breaker():
    """Example circuit breaker usage"""
    def failing_function():
        raise Exception("Service unavailable")
    
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=10.0)
    
    # Simulate multiple failures
    for i in range(5):
        try:
            result = circuit_breaker.call(failing_function)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: {e}")
            print(f"Circuit breaker state: {circuit_breaker.get_state()}")

def example_fallback_mechanism():
    """Example fallback mechanism usage"""
    def primary_function():
        raise Exception("Primary service failed")
    
    def fallback_function():
        return "Fallback response"
    
    def default_fallback():
        return "Default fallback response"
    
    fallback_manager = FallbackManager()
    fallback_manager.register_fallback(Exception, fallback_function)
    fallback_manager.set_default_fallback(default_fallback)
    
    result = fallback_manager.execute_with_fallback(primary_function)
    print(f"Fallback result: {result}")
```

### Async Error Handling

```python
# python/03-async-error-handling.py

"""
Advanced async error handling patterns
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Coroutine
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)

class AsyncErrorHandler:
    """Async error handler"""
    
    def __init__(self):
        self.error_handlers = {}
        self.retry_configs = {}
        self.circuit_breakers = {}
    
    def register_async_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register async error handler"""
        self.error_handlers[exception_type] = handler
    
    async def handle_async_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle async error"""
        handler = self.error_handlers.get(type(error))
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return await handler(error, context)
            else:
                return handler(error, context)
        else:
            logger.error(f"Unhandled async error: {error}", exc_info=True)
            raise error
    
    async def execute_with_retry(self, coro: Coroutine, max_attempts: int = 3, 
                                delay: float = 1.0) -> Any:
        """Execute coroutine with retry"""
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                return await coro
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_attempts} attempts failed")
        
        raise last_error
    
    async def execute_with_circuit_breaker(self, coro: Coroutine, breaker_name: str = "default") -> Any:
        """Execute coroutine with circuit breaker"""
        if breaker_name not in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = CircuitBreaker()
        
        breaker = self.circuit_breakers[breaker_name]
        return breaker.call(lambda: asyncio.run(coro))

class AsyncFallbackManager:
    """Async fallback manager"""
    
    def __init__(self):
        self.fallbacks = {}
        self.default_fallback = None
    
    def register_async_fallback(self, exception_type: Type[Exception], fallback_coro: Coroutine):
        """Register async fallback"""
        self.fallbacks[exception_type] = fallback_coro
    
    def set_default_async_fallback(self, fallback_coro: Coroutine):
        """Set default async fallback"""
        self.default_fallback = fallback_coro
    
    async def execute_with_async_fallback(self, coro: Coroutine) -> Any:
        """Execute coroutine with async fallback"""
        try:
            return await coro
        except Exception as e:
            fallback_coro = self.fallbacks.get(type(e), self.default_fallback)
            if fallback_coro:
                logger.warning(f"Using async fallback for {type(e).__name__}: {e}")
                return await fallback_coro
            else:
                raise e

@asynccontextmanager
async def async_error_context(handler: AsyncErrorHandler):
    """Async error context manager"""
    try:
        yield handler
    except Exception as e:
        await handler.handle_async_error(e)

# Usage examples
async def example_async_error_handling():
    """Example async error handling usage"""
    async def unreliable_async_function():
        if random.random() < 0.5:
            raise Exception("Async random failure")
        return "Async success"
    
    async def async_fallback():
        return "Async fallback response"
    
    # Setup async error handler
    async_handler = AsyncErrorHandler()
    async_handler.register_async_handler(Exception, async_fallback)
    
    # Execute with retry
    try:
        result = await async_handler.execute_with_retry(
            unreliable_async_function(), max_attempts=3, delay=0.5
        )
        print(f"Async retry result: {result}")
    except Exception as e:
        print(f"Async retry failed: {e}")
    
    # Execute with fallback
    async_fallback_manager = AsyncFallbackManager()
    async_fallback_manager.set_default_async_fallback(async_fallback)
    
    try:
        result = await async_fallback_manager.execute_with_async_fallback(
            unreliable_async_function()
        )
        print(f"Async fallback result: {result}")
    except Exception as e:
        print(f"Async fallback failed: {e}")

async def example_async_error_context():
    """Example async error context usage"""
    async_handler = AsyncErrorHandler()
    
    async def async_fallback(error: Exception, context: Optional[Dict[str, Any]] = None):
        return f"Handled async error: {error}"
    
    async_handler.register_async_handler(Exception, async_fallback)
    
    async with async_error_context(async_handler) as handler:
        # This will raise an exception
        raise Exception("Test async error")
```

## Error Logging and Monitoring

### Structured Error Logging

```python
# python/04-error-logging.py

"""
Advanced error logging and monitoring patterns
"""

import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid
import sys

class StructuredErrorLogger:
    """Structured error logger"""
    
    def __init__(self, logger_name: str = "error_logger"):
        self.logger = logging.getLogger(logger_name)
        self.error_id_generator = self._generate_error_id()
        self.error_metrics = {}
    
    def _generate_error_id(self):
        """Generate unique error ID"""
        while True:
            yield str(uuid.uuid4())
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                  severity: str = "ERROR") -> str:
        """Log structured error"""
        error_id = next(self.error_id_generator)
        
        error_data = {
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "context": context or {},
            "traceback": traceback.format_exc(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Log structured error
        self.logger.error(json.dumps(error_data, indent=2))
        
        # Update metrics
        self._update_error_metrics(error)
        
        return error_id
    
    def _update_error_metrics(self, error: Exception):
        """Update error metrics"""
        error_type = type(error).__name__
        if error_type not in self.error_metrics:
            self.error_metrics[error_type] = 0
        self.error_metrics[error_type] += 1
    
    def get_error_metrics(self) -> Dict[str, int]:
        """Get error metrics"""
        return self.error_metrics.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        total_errors = sum(self.error_metrics.values())
        return {
            "total_errors": total_errors,
            "error_types": self.error_metrics,
            "most_common_error": max(self.error_metrics.items(), key=lambda x: x[1]) if self.error_metrics else None
        }

class ErrorMonitor:
    """Error monitoring and alerting"""
    
    def __init__(self, alert_threshold: int = 10, time_window: float = 300.0):
        self.alert_threshold = alert_threshold
        self.time_window = time_window
        self.error_counts = {}
        self.alerts_sent = set()
        self.alert_handlers = []
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record error and check for alerts"""
        error_type = type(error).__name__
        current_time = time.time()
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = []
        
        self.error_counts[error_type].append(current_time)
        
        # Clean old errors outside time window
        self.error_counts[error_type] = [
            timestamp for timestamp in self.error_counts[error_type]
            if current_time - timestamp <= self.time_window
        ]
        
        # Check for alert threshold
        if len(self.error_counts[error_type]) >= self.alert_threshold:
            self._send_alert(error_type, len(self.error_counts[error_type]))
    
    def _send_alert(self, error_type: str, count: int):
        """Send alert for error threshold"""
        alert_key = f"{error_type}_{count}"
        if alert_key in self.alerts_sent:
            return
        
        self.alerts_sent.add(alert_key)
        
        alert_data = {
            "error_type": error_type,
            "count": count,
            "threshold": self.alert_threshold,
            "time_window": self.time_window,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        current_time = time.time()
        stats = {}
        
        for error_type, timestamps in self.error_counts.items():
            recent_errors = [
                ts for ts in timestamps
                if current_time - ts <= self.time_window
            ]
            stats[error_type] = {
                "count": len(recent_errors),
                "rate": len(recent_errors) / (self.time_window / 60)  # errors per minute
            }
        
        return stats

# Usage examples
def example_structured_logging():
    """Example structured logging usage"""
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    
    error_logger = StructuredErrorLogger()
    
    # Log errors
    try:
        raise ValueError("Invalid input")
    except ValueError as e:
        error_id = error_logger.log_error(e, {"user_id": "123", "operation": "validation"})
        print(f"Error logged with ID: {error_id}")
    
    # Get error metrics
    metrics = error_logger.get_error_metrics()
    print(f"Error metrics: {metrics}")
    
    summary = error_logger.get_error_summary()
    print(f"Error summary: {summary}")

def example_error_monitoring():
    """Example error monitoring usage"""
    def alert_handler(alert_data: Dict[str, Any]):
        print(f"ALERT: {alert_data['error_type']} - {alert_data['count']} errors in {alert_data['time_window']}s")
    
    monitor = ErrorMonitor(alert_threshold=3, time_window=60.0)
    monitor.add_alert_handler(alert_handler)
    
    # Simulate errors
    for i in range(5):
        try:
            raise ValueError(f"Error {i}")
        except ValueError as e:
            monitor.record_error(e)
            time.sleep(0.1)
    
    # Get error stats
    stats = monitor.get_error_stats()
    print(f"Error stats: {stats}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Custom exceptions
class ValidationError(BaseApplicationError):
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.LOW, **kwargs)
        self.field = field

# 2. Error handling
try:
    # Your code here
    pass
except ValidationError as e:
    error_handler.handle_error(e)
except Exception as e:
    error_handler.handle_error(e)

# 3. Retry mechanism
retry_config = RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
retry_manager = RetryManager(retry_config)
result = retry_manager.retry(unreliable_function)

# 4. Circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
result = circuit_breaker.call(risky_function)

# 5. Fallback mechanism
fallback_manager = FallbackManager()
fallback_manager.register_fallback(Exception, fallback_function)
result = fallback_manager.execute_with_fallback(primary_function)
```

### Essential Patterns

```python
# Complete error handling setup
def setup_error_handling():
    """Setup complete error handling environment"""
    
    # Error handler
    error_handler = ErrorHandler()
    
    # Retry manager
    retry_config = RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
    retry_manager = RetryManager(retry_config)
    
    # Circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
    
    # Fallback manager
    fallback_manager = FallbackManager()
    
    # Error logger
    error_logger = StructuredErrorLogger()
    
    # Error monitor
    error_monitor = ErrorMonitor(alert_threshold=10, time_window=300.0)
    
    print("Error handling setup complete!")
```

---

*This guide provides the complete machinery for Python error handling. Each pattern includes implementation examples, recovery strategies, and real-world usage patterns for enterprise error management.*
