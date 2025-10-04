# Go Web Services Best Practices

**Objective**: Master senior-level Go web service patterns for production systems. When you need to build robust, scalable HTTP services, when you want to follow proven methodologies, when you need enterprise-grade web service patterns—these best practices become your weapon of choice.

## Core Principles

- **HTTP-First Design**: Leverage HTTP semantics and status codes
- **Middleware Architecture**: Use middleware for cross-cutting concerns
- **Graceful Shutdown**: Handle shutdown signals properly
- **Request/Response Validation**: Validate all inputs and outputs
- **Security by Default**: Implement security measures from the start

## HTTP Server Setup

### Basic Server Structure

```go
// internal/server/server.go
package server

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

// Server represents an HTTP server
type Server struct {
    httpServer *http.Server
    router     *http.ServeMux
    config     *Config
}

// Config holds server configuration
type Config struct {
    Port         string
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    IdleTimeout  time.Duration
}

// NewServer creates a new HTTP server
func NewServer(config *Config) *Server {
    router := http.NewServeMux()
    
    server := &Server{
        router: router,
        config: config,
    }
    
    server.setupRoutes()
    
    return server
}

// setupRoutes configures the HTTP routes
func (s *Server) setupRoutes() {
    // Health check
    s.router.HandleFunc("/health", s.healthHandler)
    
    // API routes
    s.router.HandleFunc("/api/users", s.usersHandler)
    s.router.HandleFunc("/api/users/", s.userHandler)
    
    // Static files
    s.router.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
}

// Start starts the HTTP server
func (s *Server) Start() error {
    s.httpServer = &http.Server{
        Addr:         s.config.Port,
        Handler:      s.router,
        ReadTimeout:  s.config.ReadTimeout,
        WriteTimeout: s.config.WriteTimeout,
        IdleTimeout:  s.config.IdleTimeout,
    }
    
    log.Printf("Starting server on %s", s.config.Port)
    return s.httpServer.ListenAndServe()
}

// StartTLS starts the HTTPS server
func (s *Server) StartTLS(certFile, keyFile string) error {
    s.httpServer = &http.Server{
        Addr:         s.config.Port,
        Handler:      s.router,
        ReadTimeout:  s.config.ReadTimeout,
        WriteTimeout: s.config.WriteTimeout,
        IdleTimeout:  s.config.IdleTimeout,
    }
    
    log.Printf("Starting TLS server on %s", s.config.Port)
    return s.httpServer.ListenAndServeTLS(certFile, keyFile)
}

// Stop gracefully stops the server
func (s *Server) Stop(ctx context.Context) error {
    log.Println("Shutting down server...")
    return s.httpServer.Shutdown(ctx)
}

// healthHandler handles health check requests
func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"status":"healthy","timestamp":"%s"}`, time.Now().UTC().Format(time.RFC3339))
}
```

### Graceful Shutdown

```go
// internal/server/graceful.go
package server

import (
    "context"
    "log"
    "os"
    "os/signal"
    "syscall"
    "time"
)

// RunWithGracefulShutdown runs the server with graceful shutdown
func (s *Server) RunWithGracefulShutdown() {
    // Start server in a goroutine
    go func() {
        if err := s.Start(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server failed to start: %v", err)
        }
    }()
    
    // Wait for interrupt signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    // Graceful shutdown with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := s.Stop(ctx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }
    
    log.Println("Server exited")
}
```

## Middleware Architecture

### Middleware Interface

```go
// internal/middleware/middleware.go
package middleware

import (
    "context"
    "net/http"
    "time"
)

// Middleware represents a middleware function
type Middleware func(http.Handler) http.Handler

// Chain chains multiple middlewares
func Chain(middlewares ...Middleware) Middleware {
    return func(next http.Handler) http.Handler {
        for i := len(middlewares) - 1; i >= 0; i-- {
            next = middlewares[i](next)
        }
        return next
    }
}
```

### Logging Middleware

```go
// internal/middleware/logging.go
package middleware

import (
    "log"
    "net/http"
    "time"
)

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap the ResponseWriter to capture status code
        wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
        
        next.ServeHTTP(wrapped, r)
        
        duration := time.Since(start)
        log.Printf("%s %s %d %v", r.Method, r.URL.Path, wrapped.statusCode, duration)
    })
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}
```

### CORS Middleware

```go
// internal/middleware/cors.go
package middleware

import (
    "net/http"
)

// CORSMiddleware handles CORS headers
func CORSMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if r.Method == http.MethodOptions {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}
```

### Authentication Middleware

```go
// internal/middleware/auth.go
package middleware

import (
    "context"
    "net/http"
    "strings"
)

// AuthMiddleware handles authentication
func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        authHeader := r.Header.Get("Authorization")
        if authHeader == "" {
            http.Error(w, "Authorization header required", http.StatusUnauthorized)
            return
        }
        
        token := strings.TrimPrefix(authHeader, "Bearer ")
        if token == authHeader {
            http.Error(w, "Invalid authorization format", http.StatusUnauthorized)
            return
        }
        
        // Validate token (implement your validation logic)
        userID, err := validateToken(token)
        if err != nil {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }
        
        // Add user ID to context
        ctx := context.WithValue(r.Context(), "userID", userID)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// validateToken validates a JWT token
func validateToken(token string) (string, error) {
    // Implement JWT validation logic
    // Return user ID and error
    return "user123", nil
}
```

### Rate Limiting Middleware

```go
// internal/middleware/ratelimit.go
package middleware

import (
    "net/http"
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

// RateLimitMiddleware implements rate limiting
func RateLimitMiddleware(limiter *RateLimiter) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            if !limiter.Allow() {
                http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
                return
            }
            
            next.ServeHTTP(w, r)
        })
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

## Request/Response Handling

### JSON Request/Response

```go
// internal/handler/json.go
package handler

import (
    "encoding/json"
    "net/http"
)

// JSONResponse represents a JSON response
type JSONResponse struct {
    Data    interface{} `json:"data,omitempty"`
    Error   string      `json:"error,omitempty"`
    Message string      `json:"message,omitempty"`
}

// WriteJSON writes a JSON response
func WriteJSON(w http.ResponseWriter, statusCode int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    
    response := JSONResponse{Data: data}
    json.NewEncoder(w).Encode(response)
}

// WriteError writes an error response
func WriteError(w http.ResponseWriter, statusCode int, message string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    
    response := JSONResponse{Error: message}
    json.NewEncoder(w).Encode(response)
}

// ReadJSON reads JSON from request body
func ReadJSON(r *http.Request, v interface{}) error {
    if r.Header.Get("Content-Type") != "application/json" {
        return fmt.Errorf("content-type must be application/json")
    }
    
    return json.NewDecoder(r.Body).Decode(v)
}
```

### Request Validation

```go
// internal/handler/validation.go
package handler

import (
    "fmt"
    "net/http"
    "strings"
)

// Validator represents a request validator
type Validator interface {
    Validate() error
}

// UserRequest represents a user creation request
type UserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

// Validate validates the user request
func (ur *UserRequest) Validate() error {
    if strings.TrimSpace(ur.Name) == "" {
        return fmt.Errorf("name is required")
    }
    
    if strings.TrimSpace(ur.Email) == "" {
        return fmt.Errorf("email is required")
    }
    
    if !isValidEmail(ur.Email) {
        return fmt.Errorf("invalid email format")
    }
    
    return nil
}

// isValidEmail validates email format
func isValidEmail(email string) bool {
    return strings.Contains(email, "@") && strings.Contains(email, ".")
}
```

## Handler Patterns

### RESTful Handlers

```go
// internal/handler/user.go
package handler

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
)

// UserHandler handles user-related HTTP requests
type UserHandler struct {
    userService UserService
}

// NewUserHandler creates a new user handler
func NewUserHandler(userService UserService) *UserHandler {
    return &UserHandler{
        userService: userService,
    }
}

// CreateUser handles user creation
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
        return
    }
    
    var req UserRequest
    if err := ReadJSON(r, &req); err != nil {
        WriteError(w, http.StatusBadRequest, "Invalid JSON")
        return
    }
    
    if err := req.Validate(); err != nil {
        WriteError(w, http.StatusBadRequest, err.Error())
        return
    }
    
    user, err := h.userService.CreateUser(&req)
    if err != nil {
        WriteError(w, http.StatusInternalServerError, "Failed to create user")
        return
    }
    
    WriteJSON(w, http.StatusCreated, user)
}

// GetUser handles user retrieval
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
        return
    }
    
    userID := extractUserID(r.URL.Path)
    if userID == "" {
        WriteError(w, http.StatusBadRequest, "User ID required")
        return
    }
    
    user, err := h.userService.GetUser(userID)
    if err != nil {
        WriteError(w, http.StatusNotFound, "User not found")
        return
    }
    
    WriteJSON(w, http.StatusOK, user)
}

// UpdateUser handles user updates
func (h *UserHandler) UpdateUser(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPut {
        WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
        return
    }
    
    userID := extractUserID(r.URL.Path)
    if userID == "" {
        WriteError(w, http.StatusBadRequest, "User ID required")
        return
    }
    
    var req UserRequest
    if err := ReadJSON(r, &req); err != nil {
        WriteError(w, http.StatusBadRequest, "Invalid JSON")
        return
    }
    
    if err := req.Validate(); err != nil {
        WriteError(w, http.StatusBadRequest, err.Error())
        return
    }
    
    user, err := h.userService.UpdateUser(userID, &req)
    if err != nil {
        WriteError(w, http.StatusInternalServerError, "Failed to update user")
        return
    }
    
    WriteJSON(w, http.StatusOK, user)
}

// DeleteUser handles user deletion
func (h *UserHandler) DeleteUser(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodDelete {
        WriteError(w, http.StatusMethodNotAllowed, "Method not allowed")
        return
    }
    
    userID := extractUserID(r.URL.Path)
    if userID == "" {
        WriteError(w, http.StatusBadRequest, "User ID required")
        return
    }
    
    if err := h.userService.DeleteUser(userID); err != nil {
        WriteError(w, http.StatusInternalServerError, "Failed to delete user")
        return
    }
    
    w.WriteHeader(http.StatusNoContent)
}

// extractUserID extracts user ID from URL path
func extractUserID(path string) string {
    parts := strings.Split(path, "/")
    if len(parts) >= 3 {
        return parts[3]
    }
    return ""
}
```

### Error Handling

```go
// internal/handler/error.go
package handler

import (
    "fmt"
    "net/http"
)

// ErrorHandler handles errors
type ErrorHandler struct {
    logger Logger
}

// NewErrorHandler creates a new error handler
func NewErrorHandler(logger Logger) *ErrorHandler {
    return &ErrorHandler{
        logger: logger,
    }
}

// HandleError handles application errors
func (eh *ErrorHandler) HandleError(w http.ResponseWriter, r *http.Request, err error) {
    eh.logger.Error("Request failed", map[string]interface{}{
        "method": r.Method,
        "path":   r.URL.Path,
        "error":  err.Error(),
    })
    
    switch err.(type) {
    case *ValidationError:
        WriteError(w, http.StatusBadRequest, err.Error())
    case *NotFoundError:
        WriteError(w, http.StatusNotFound, err.Error())
    case *UnauthorizedError:
        WriteError(w, http.StatusUnauthorized, err.Error())
    default:
        WriteError(w, http.StatusInternalServerError, "Internal server error")
    }
}

// Custom error types
type ValidationError struct {
    Message string
}

func (ve *ValidationError) Error() string {
    return ve.Message
}

type NotFoundError struct {
    Resource string
}

func (nfe *NotFoundError) Error() string {
    return fmt.Sprintf("%s not found", nfe.Resource)
}

type UnauthorizedError struct {
    Message string
}

func (ue *UnauthorizedError) Error() string {
    return ue.Message
}
```

## Security Best Practices

### HTTPS Configuration

```go
// internal/server/tls.go
package server

import (
    "crypto/tls"
    "net/http"
)

// TLSConfig holds TLS configuration
type TLSConfig struct {
    CertFile string
    KeyFile  string
    MinTLS   uint16
}

// ConfigureTLS configures TLS settings
func (s *Server) ConfigureTLS(config *TLSConfig) {
    tlsConfig := &tls.Config{
        MinVersion: config.MinTLS,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
            tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        },
    }
    
    s.httpServer.TLSConfig = tlsConfig
}
```

### Security Headers

```go
// internal/middleware/security.go
package middleware

import "net/http"

// SecurityHeadersMiddleware adds security headers
func SecurityHeadersMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("X-Content-Type-Options", "nosniff")
        w.Header().Set("X-Frame-Options", "DENY")
        w.Header().Set("X-XSS-Protection", "1; mode=block")
        w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        w.Header().Set("Content-Security-Policy", "default-src 'self'")
        
        next.ServeHTTP(w, r)
    })
}
```

## Performance Optimization

### Connection Pooling

```go
// internal/server/pool.go
package server

import (
    "net"
    "net/http"
    "time"
)

// ConfigureConnectionPool configures HTTP connection pooling
func (s *Server) ConfigureConnectionPool() {
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:    90 * time.Second,
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
    }
    
    s.httpServer.Transport = transport
}
```

### Compression Middleware

```go
// internal/middleware/compression.go
package middleware

import (
    "compress/gzip"
    "net/http"
    "strings"
)

// CompressionMiddleware adds gzip compression
func CompressionMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
            next.ServeHTTP(w, r)
            return
        }
        
        w.Header().Set("Content-Encoding", "gzip")
        gz := gzip.NewWriter(w)
        defer gz.Close()
        
        gzw := &gzipResponseWriter{ResponseWriter: w, Writer: gz}
        next.ServeHTTP(gzw, r)
    })
}

// gzipResponseWriter wraps http.ResponseWriter with gzip
type gzipResponseWriter struct {
    http.ResponseWriter
    *gzip.Writer
}
```

## Testing Web Services

### HTTP Testing

```go
// internal/handler/user_test.go
package handler

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestUserHandler_CreateUser(t *testing.T) {
    // Arrange
    mockService := &MockUserService{}
    handler := NewUserHandler(mockService)
    
    userReq := UserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    jsonData, err := json.Marshal(userReq)
    require.NoError(t, err)
    
    req := httptest.NewRequest("POST", "/api/users", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusCreated, w.Code)
    assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
    
    var response JSONResponse
    err = json.Unmarshal(w.Body.Bytes(), &response)
    require.NoError(t, err)
    assert.NotNil(t, response.Data)
}

func TestUserHandler_CreateUser_InvalidJSON(t *testing.T) {
    // Arrange
    mockService := &MockUserService{}
    handler := NewUserHandler(mockService)
    
    req := httptest.NewRequest("POST", "/api/users", bytes.NewBufferString("invalid json"))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusBadRequest, w.Code)
    
    var response JSONResponse
    err := json.Unmarshal(w.Body.Bytes(), &response)
    require.NoError(t, err)
    assert.Equal(t, "Invalid JSON", response.Error)
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Basic HTTP server
func main() {
    server := NewServer(&Config{
        Port:         ":8080",
        ReadTimeout:  30 * time.Second,
        WriteTimeout:   30 * time.Second,
        IdleTimeout:   120 * time.Second,
    })
    
    server.RunWithGracefulShutdown()
}

// 2. Middleware chain
middleware := Chain(
    LoggingMiddleware,
    CORSMiddleware,
    AuthMiddleware,
    SecurityHeadersMiddleware,
)

// 3. Handler with validation
func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req UserRequest
    if err := ReadJSON(r, &req); err != nil {
        WriteError(w, http.StatusBadRequest, "Invalid JSON")
        return
    }
    
    if err := req.Validate(); err != nil {
        WriteError(w, http.StatusBadRequest, err.Error())
        return
    }
    
    // Process request...
}
```

### Essential Patterns

```go
// Error handling
func (h *Handler) HandleRequest(w http.ResponseWriter, r *http.Request) {
    if err := h.processRequest(r); err != nil {
        h.errorHandler.HandleError(w, r, err)
        return
    }
    
    WriteJSON(w, http.StatusOK, result)
}

// Middleware usage
router.Handle("/api/users", middleware(handler.CreateUser))

// Graceful shutdown
server.RunWithGracefulShutdown()
```

---

*This guide provides the complete machinery for building production-ready web services in Go applications. Each pattern includes implementation examples, security considerations, and real-world usage patterns for enterprise deployment.*
