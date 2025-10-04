# Go Error Handling Best Practices

**Objective**: Master senior-level Go error handling patterns for production systems. When you need to build robust, maintainable applications, when you want to follow proven methodologies, when you need enterprise-grade error handling patterns—these best practices become your weapon of choice.

## Core Principles

- **Errors are Values**: Treat errors as first-class values, not exceptions
- **Explicit Error Handling**: Always handle errors explicitly, never ignore them
- **Error Wrapping**: Use error wrapping to provide context and preserve error chains
- **Custom Error Types**: Create meaningful error types for different error conditions
- **Error Logging**: Log errors with appropriate context and severity levels

## Basic Error Handling

### Standard Error Patterns

```go
// internal/errors/basic.go
package errors

import (
    "errors"
    "fmt"
    "io"
    "os"
)

// Basic error handling
func ReadFile(filename string) ([]byte, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file %s: %w", filename, err)
    }
    defer file.Close()
    
    data, err := io.ReadAll(file)
    if err != nil {
        return nil, fmt.Errorf("failed to read file %s: %w", filename, err)
    }
    
    return data, nil
}

// Error checking with early return
func ProcessData(data []byte) error {
    if len(data) == 0 {
        return errors.New("data cannot be empty")
    }
    
    if len(data) > 1024*1024 {
        return fmt.Errorf("data too large: %d bytes", len(data))
    }
    
    // Process data...
    return nil
}
```

### Error Wrapping and Unwrapping

```go
// internal/errors/wrapping.go
package errors

import (
    "errors"
    "fmt"
    "io"
)

// WrapError wraps an error with additional context
func WrapError(err error, message string) error {
    if err == nil {
        return nil
    }
    return fmt.Errorf("%s: %w", message, err)
}

// WrapErrorf wraps an error with formatted context
func WrapErrorf(err error, format string, args ...interface{}) error {
    if err == nil {
        return nil
    }
    return fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), err)
}

// IsError checks if the error chain contains a specific error
func IsError(err, target error) bool {
    return errors.Is(err, target)
}

// AsError extracts a specific error type from the error chain
func AsError(err error, target interface{}) bool {
    return errors.As(err, target)
}

// Example usage
func ProcessFile(filename string) error {
    data, err := ReadFile(filename)
    if err != nil {
        return WrapErrorf(err, "failed to process file %s", filename)
    }
    
    if err := ProcessData(data); err != nil {
        return WrapError(err, "data processing failed")
    }
    
    return nil
}
```

## Custom Error Types

### Structured Error Types

```go
// internal/errors/custom.go
package errors

import (
    "fmt"
    "time"
)

// ErrorCode represents different types of errors
type ErrorCode int

const (
    ErrorCodeValidation ErrorCode = iota
    ErrorCodeNotFound
    ErrorCodeUnauthorized
    ErrorCodeInternal
    ErrorCodeTimeout
    ErrorCodeRateLimit
)

// AppError represents an application error
type AppError struct {
    Code      ErrorCode
    Message   string
    Details   map[string]interface{}
    Timestamp time.Time
    Cause     error
}

// Error implements the error interface
func (ae *AppError) Error() string {
    if ae.Cause != nil {
        return fmt.Sprintf("[%s] %s: %v", ae.Code.String(), ae.Message, ae.Cause)
    }
    return fmt.Sprintf("[%s] %s", ae.Code.String(), ae.Message)
}

// Unwrap returns the underlying cause
func (ae *AppError) Unwrap() error {
    return ae.Cause
}

// String returns the string representation of the error code
func (ec ErrorCode) String() string {
    switch ec {
    case ErrorCodeValidation:
        return "VALIDATION_ERROR"
    case ErrorCodeNotFound:
        return "NOT_FOUND"
    case ErrorCodeUnauthorized:
        return "UNAUTHORIZED"
    case ErrorCodeInternal:
        return "INTERNAL_ERROR"
    case ErrorCodeTimeout:
        return "TIMEOUT"
    case ErrorCodeRateLimit:
        return "RATE_LIMIT"
    default:
        return "UNKNOWN_ERROR"
    }
}

// NewAppError creates a new application error
func NewAppError(code ErrorCode, message string, cause error) *AppError {
    return &AppError{
        Code:      code,
        Message:   message,
        Details:   make(map[string]interface{}),
        Timestamp: time.Now(),
        Cause:     cause,
    }
}

// WithDetail adds a detail to the error
func (ae *AppError) WithDetail(key string, value interface{}) *AppError {
    ae.Details[key] = value
    return ae
}
```

### Validation Errors

```go
// internal/errors/validation.go
package errors

import (
    "fmt"
    "strings"
)

// ValidationError represents a validation error
type ValidationError struct {
    Field   string
    Value   interface{}
    Message string
}

// Error implements the error interface
func (ve *ValidationError) Error() string {
    return fmt.Sprintf("validation error for field '%s': %s", ve.Field, ve.Message)
}

// ValidationErrors represents multiple validation errors
type ValidationErrors struct {
    Errors []ValidationError
}

// Error implements the error interface
func (ve *ValidationErrors) Error() string {
    if len(ve.Errors) == 0 {
        return "no validation errors"
    }
    
    var messages []string
    for _, err := range ve.Errors {
        messages = append(messages, err.Error())
    }
    
    return fmt.Sprintf("validation errors: %s", strings.Join(messages, "; "))
}

// Add adds a validation error
func (ve *ValidationErrors) Add(field string, value interface{}, message string) {
    ve.Errors = append(ve.Errors, ValidationError{
        Field:   field,
        Value:   value,
        Message: message,
    })
}

// HasErrors returns true if there are validation errors
func (ve *ValidationErrors) HasErrors() bool {
    return len(ve.Errors) > 0
}

// NewValidationError creates a new validation error
func NewValidationError(field string, value interface{}, message string) *ValidationError {
    return &ValidationError{
        Field:   field,
        Value:   value,
        Message: message,
    }
}
```

## Error Handling Patterns

### Error Recovery

```go
// internal/errors/recovery.go
package errors

import (
    "fmt"
    "log"
    "runtime"
)

// RecoverPanic recovers from a panic and returns an error
func RecoverPanic() error {
    if r := recover(); r != nil {
        return fmt.Errorf("panic recovered: %v", r)
    }
    return nil
}

// SafeExecute executes a function with panic recovery
func SafeExecute(fn func() error) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic recovered: %v", r)
        }
    }()
    
    return fn()
}

// LogPanic logs panic information
func LogPanic() {
    if r := recover(); r != nil {
        stack := make([]byte, 4096)
        length := runtime.Stack(stack, false)
        log.Printf("panic recovered: %v\nstack trace:\n%s", r, stack[:length])
    }
}

// Example usage
func ProcessWithRecovery(data interface{}) error {
    defer LogPanic()
    
    // Risky operation that might panic
    return processData(data)
}
```

### Error Aggregation

```go
// internal/errors/aggregation.go
package errors

import (
    "fmt"
    "strings"
)

// MultiError represents multiple errors
type MultiError struct {
    Errors []error
}

// Error implements the error interface
func (me *MultiError) Error() string {
    if len(me.Errors) == 0 {
        return "no errors"
    }
    
    var messages []string
    for _, err := range me.Errors {
        messages = append(messages, err.Error())
    }
    
    return fmt.Sprintf("multiple errors: %s", strings.Join(messages, "; "))
}

// Add adds an error to the collection
func (me *MultiError) Add(err error) {
    if err != nil {
        me.Errors = append(me.Errors, err)
    }
}

// HasErrors returns true if there are errors
func (me *MultiError) HasErrors() bool {
    return len(me.Errors) > 0
}

// NewMultiError creates a new multi-error
func NewMultiError() *MultiError {
    return &MultiError{
        Errors: make([]error, 0),
    }
}
```

## Context-Aware Error Handling

### Error with Context

```go
// internal/errors/context.go
package errors

import (
    "context"
    "fmt"
    "time"
)

// ContextError represents an error with context information
type ContextError struct {
    Operation string
    Context   map[string]interface{}
    Timestamp time.Time
    Cause     error
}

// Error implements the error interface
func (ce *ContextError) Error() string {
    return fmt.Sprintf("error in %s: %v", ce.Operation, ce.Cause)
}

// Unwrap returns the underlying cause
func (ce *ContextError) Unwrap() error {
    return ce.Cause
}

// WithContext creates an error with context
func WithContext(operation string, err error, context map[string]interface{}) *ContextError {
    return &ContextError{
        Operation: operation,
        Context:   context,
        Timestamp: time.Now(),
        Cause:     err,
    }
}

// HandleContextError handles context cancellation
func HandleContextError(ctx context.Context, operation string) error {
    select {
    case <-ctx.Done():
        return WithContext(operation, ctx.Err(), map[string]interface{}{
            "cancelled": true,
            "deadline":  ctx.Deadline(),
        })
    default:
        return nil
    }
}
```

## Error Logging and Monitoring

### Structured Error Logging

```go
// internal/errors/logging.go
package errors

import (
    "encoding/json"
    "fmt"
    "log"
    "os"
    "time"
)

// ErrorLogger provides structured error logging
type ErrorLogger struct {
    logger *log.Logger
}

// NewErrorLogger creates a new error logger
func NewErrorLogger() *ErrorLogger {
    return &ErrorLogger{
        logger: log.New(os.Stderr, "[ERROR] ", log.LstdFlags),
    }
}

// LogError logs an error with structured information
func (el *ErrorLogger) LogError(err error, context map[string]interface{}) {
    errorInfo := map[string]interface{}{
        "error":     err.Error(),
        "timestamp": time.Now().UTC(),
        "context":   context,
    }
    
    if wrappedErr, ok := err.(*AppError); ok {
        errorInfo["code"] = wrappedErr.Code.String()
        errorInfo["details"] = wrappedErr.Details
    }
    
    jsonData, _ := json.Marshal(errorInfo)
    el.logger.Printf("%s", jsonData)
}

// LogPanic logs panic information
func (el *ErrorLogger) LogPanic(panic interface{}, stack []byte, context map[string]interface{}) {
    panicInfo := map[string]interface{}{
        "panic":     fmt.Sprintf("%v", panic),
        "stack":     string(stack),
        "timestamp": time.Now().UTC(),
        "context":   context,
    }
    
    jsonData, _ := json.Marshal(panicInfo)
    el.logger.Printf("%s", jsonData)
}
```

### Error Metrics

```go
// internal/errors/metrics.go
package errors

import (
    "sync"
    "time"
)

// ErrorMetrics tracks error statistics
type ErrorMetrics struct {
    mu           sync.RWMutex
    errorCounts  map[string]int
    errorRates   map[string]float64
    lastReset    time.Time
}

// NewErrorMetrics creates new error metrics
func NewErrorMetrics() *ErrorMetrics {
    return &ErrorMetrics{
        errorCounts: make(map[string]int),
        errorRates:  make(map[string]float64),
        lastReset:   time.Now(),
    }
}

// RecordError records an error occurrence
func (em *ErrorMetrics) RecordError(errorType string) {
    em.mu.Lock()
    defer em.mu.Unlock()
    
    em.errorCounts[errorType]++
}

// GetErrorCount returns the count for an error type
func (em *ErrorMetrics) GetErrorCount(errorType string) int {
    em.mu.RLock()
    defer em.mu.RUnlock()
    
    return em.errorCounts[errorType]
}

// GetErrorRate returns the rate for an error type
func (em *ErrorMetrics) GetErrorRate(errorType string) float64 {
    em.mu.RLock()
    defer em.mu.RUnlock()
    
    return em.errorRates[errorType]
}

// Reset resets the metrics
func (em *ErrorMetrics) Reset() {
    em.mu.Lock()
    defer em.mu.Unlock()
    
    em.errorCounts = make(map[string]int)
    em.errorRates = make(map[string]float64)
    em.lastReset = time.Now()
}
```

## Testing Error Handling

### Error Testing Patterns

```go
// internal/errors/errors_test.go
package errors

import (
    "errors"
    "testing"
)

func TestAppError(t *testing.T) {
    cause := errors.New("underlying error")
    appErr := NewAppError(ErrorCodeValidation, "validation failed", cause)
    
    if appErr.Code != ErrorCodeValidation {
        t.Errorf("Expected code %v, got %v", ErrorCodeValidation, appErr.Code)
    }
    
    if appErr.Message != "validation failed" {
        t.Errorf("Expected message 'validation failed', got '%s'", appErr.Message)
    }
    
    if appErr.Cause != cause {
        t.Errorf("Expected cause %v, got %v", cause, appErr.Cause)
    }
}

func TestValidationErrors(t *testing.T) {
    ve := &ValidationErrors{}
    
    ve.Add("email", "invalid-email", "invalid email format")
    ve.Add("age", -1, "age must be positive")
    
    if !ve.HasErrors() {
        t.Error("Expected validation errors")
    }
    
    if len(ve.Errors) != 2 {
        t.Errorf("Expected 2 errors, got %d", len(ve.Errors))
    }
}

func TestErrorWrapping(t *testing.T) {
    originalErr := errors.New("original error")
    wrappedErr := WrapError(originalErr, "context")
    
    if !errors.Is(wrappedErr, originalErr) {
        t.Error("Expected wrapped error to contain original error")
    }
    
    var targetErr error
    if !errors.As(wrappedErr, &targetErr) {
        t.Error("Expected to extract error from wrapped error")
    }
}
```

## Best Practices Summary

### Do's

```go
// ✅ Good: Explicit error handling
func ProcessData(data []byte) error {
    if len(data) == 0 {
        return errors.New("data cannot be empty")
    }
    
    if err := validateData(data); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    return nil
}

// ✅ Good: Error wrapping with context
func ReadConfig(filename string) (*Config, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to read config file %s: %w", filename, err)
    }
    
    config, err := ParseConfig(data)
    if err != nil {
        return nil, fmt.Errorf("failed to parse config: %w", err)
    }
    
    return config, nil
}

// ✅ Good: Custom error types
type DatabaseError struct {
    Operation string
    Query     string
    Cause     error
}

func (de *DatabaseError) Error() string {
    return fmt.Sprintf("database error in %s: %v", de.Operation, de.Cause)
}
```

### Don'ts

```go
// ❌ Bad: Ignoring errors
func BadExample() {
    file, _ := os.Open("file.txt")  // Don't ignore errors
    defer file.Close()
}

// ❌ Bad: Generic error messages
func BadExample2() error {
    return errors.New("error")  // Too generic
}

// ❌ Bad: Panic for control flow
func BadExample3() {
    if condition {
        panic("something went wrong")  // Don't use panic for control flow
    }
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Basic error handling
if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}

// 2. Custom error types
type MyError struct {
    Code    int
    Message string
    Cause   error
}

func (me *MyError) Error() string {
    return fmt.Sprintf("[%d] %s: %v", me.Code, me.Message, me.Cause)
}

// 3. Error wrapping
wrappedErr := fmt.Errorf("context: %w", originalErr)

// 4. Error checking
if errors.Is(err, targetErr) {
    // Handle specific error
}
```

### Essential Patterns

```go
// Error aggregation
multiErr := NewMultiError()
multiErr.Add(err1)
multiErr.Add(err2)
if multiErr.HasErrors() {
    return multiErr
}

// Context-aware errors
ctxErr := WithContext("operation", err, map[string]interface{}{
    "user_id": 123,
    "action":  "create",
})

// Error logging
logger := NewErrorLogger()
logger.LogError(err, map[string]interface{}{
    "operation": "process_data",
    "user_id":   123,
})
```

---

*This guide provides the complete machinery for building production-ready error handling in Go applications. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*
