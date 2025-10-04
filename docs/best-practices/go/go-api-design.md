# Go API Design Best Practices

**Objective**: Master senior-level Go API design patterns for production systems. When you need to build robust, scalable RESTful APIs, when you want to follow proven methodologies, when you need enterprise-grade API design patterns—these best practices become your weapon of choice.

## Core Principles

- **RESTful Design**: Follow REST principles and HTTP semantics
- **Consistent Interface**: Maintain consistent API patterns across endpoints
- **Versioning Strategy**: Implement proper API versioning
- **Documentation First**: Design APIs with documentation in mind
- **Security by Design**: Implement security measures from the start

## API Design Patterns

### RESTful Resource Design

```go
// internal/api/design.go
package api

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
)

// Resource represents a RESTful resource
type Resource struct {
    ID        string    `json:"id"`
    Type      string    `json:"type"`
    Attributes interface{} `json:"attributes"`
    Links     Links     `json:"links,omitempty"`
    Meta      Meta      `json:"meta,omitempty"`
}

// Links represents resource links
type Links struct {
    Self    string `json:"self,omitempty"`
    Related string `json:"related,omitempty"`
    Next    string `json:"next,omitempty"`
    Prev    string `json:"prev,omitempty"`
}

// Meta represents metadata
type Meta struct {
    TotalCount int       `json:"total_count,omitempty"`
    Page       int       `json:"page,omitempty"`
    PerPage    int       `json:"per_page,omitempty"`
    Timestamp  time.Time `json:"timestamp"`
}

// Response represents a standard API response
type Response struct {
    Data     interface{} `json:"data,omitempty"`
    Errors   []Error     `json:"errors,omitempty"`
    Meta     Meta        `json:"meta,omitempty"`
    Links    Links       `json:"links,omitempty"`
    Included []Resource  `json:"included,omitempty"`
}

// Error represents an API error
type Error struct {
    ID     string `json:"id,omitempty"`
    Status string `json:"status"`
    Code   string `json:"code"`
    Title  string `json:"title"`
    Detail string `json:"detail"`
    Source Source `json:"source,omitempty"`
}

// Source represents error source
type Source struct {
    Pointer   string `json:"pointer,omitempty"`
    Parameter string `json:"parameter,omitempty"`
}
```

### HTTP Status Codes

```go
// internal/api/status.go
package api

import "net/http"

// StatusCode represents HTTP status codes
type StatusCode int

const (
    StatusOK                  StatusCode = http.StatusOK
    StatusCreated            StatusCode = http.StatusCreated
    StatusAccepted           StatusCode = http.StatusAccepted
    StatusNoContent          StatusCode = http.StatusNoContent
    StatusBadRequest         StatusCode = http.StatusBadRequest
    StatusUnauthorized       StatusCode = http.StatusUnauthorized
    StatusForbidden          StatusCode = http.StatusForbidden
    StatusNotFound           StatusCode = http.StatusNotFound
    StatusMethodNotAllowed   StatusCode = http.StatusMethodNotAllowed
    StatusConflict           StatusCode = http.StatusConflict
    StatusUnprocessableEntity StatusCode = http.StatusUnprocessableEntity
    StatusTooManyRequests   StatusCode = http.StatusTooManyRequests
    StatusInternalServerError StatusCode = http.StatusInternalServerError
)

// GetStatusText returns the status text for a status code
func GetStatusText(code StatusCode) string {
    return http.StatusText(int(code))
}
```

### Request/Response Models

```go
// internal/api/models.go
package api

import (
    "time"
    "github.com/google/uuid"
)

// User represents a user resource
type User struct {
    ID        string    `json:"id"`
    Type      string    `json:"type"`
    Attributes UserAttributes `json:"attributes"`
    Links     Links     `json:"links"`
}

// UserAttributes represents user attributes
type UserAttributes struct {
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

// CreateUserRequest represents a user creation request
type CreateUserRequest struct {
    Data User `json:"data"`
}

// UpdateUserRequest represents a user update request
type UpdateUserRequest struct {
    Data User `json:"data"`
}

// ListUsersRequest represents a user list request
type ListUsersRequest struct {
    Page     int    `json:"page,omitempty"`
    PerPage  int    `json:"per_page,omitempty"`
    Sort     string `json:"sort,omitempty"`
    Filter   string `json:"filter,omitempty"`
    Include  string `json:"include,omitempty"`
}

// ListUsersResponse represents a user list response
type ListUsersResponse struct {
    Data     []User `json:"data"`
    Meta     Meta   `json:"meta"`
    Links    Links  `json:"links"`
    Included []Resource `json:"included,omitempty"`
}
```

## API Versioning

### URL Versioning

```go
// internal/api/versioning.go
package api

import (
    "net/http"
    "strings"
)

// Version represents an API version
type Version string

const (
    VersionV1 Version = "v1"
    VersionV2 Version = "v2"
)

// VersionHandler handles API versioning
type VersionHandler struct {
    versions map[Version]http.Handler
    defaultVersion Version
}

// NewVersionHandler creates a new version handler
func NewVersionHandler() *VersionHandler {
    return &VersionHandler{
        versions: make(map[Version]http.Handler),
        defaultVersion: VersionV1,
    }
}

// RegisterVersion registers a handler for a version
func (vh *VersionHandler) RegisterVersion(version Version, handler http.Handler) {
    vh.versions[version] = handler
}

// ServeHTTP implements http.Handler
func (vh *VersionHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    version := vh.extractVersion(r)
    
    handler, exists := vh.versions[version]
    if !exists {
        vh.handleVersionNotFound(w, r)
        return
    }
    
    handler.ServeHTTP(w, r)
}

// extractVersion extracts the version from the request
func (vh *VersionHandler) extractVersion(r *http.Request) Version {
    // Extract from URL path: /api/v1/users
    path := r.URL.Path
    if strings.HasPrefix(path, "/api/") {
        parts := strings.Split(path, "/")
        if len(parts) >= 3 {
            return Version(parts[2])
        }
    }
    
    // Extract from header
    if version := r.Header.Get("API-Version"); version != "" {
        return Version(version)
    }
    
    // Extract from query parameter
    if version := r.URL.Query().Get("version"); version != "" {
        return Version(version)
    }
    
    return vh.defaultVersion
}

// handleVersionNotFound handles version not found
func (vh *VersionHandler) handleVersionNotFound(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusNotFound)
    
    response := Response{
        Errors: []Error{
            {
                Status: "404",
                Code:   "VERSION_NOT_FOUND",
                Title:  "API Version Not Found",
                Detail: "The requested API version is not supported",
            },
        },
    }
    
    json.NewEncoder(w).Encode(response)
}
```

### Header Versioning

```go
// internal/api/header_versioning.go
package api

import (
    "net/http"
    "strings"
)

// HeaderVersioningMiddleware handles versioning via headers
func HeaderVersioningMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        version := r.Header.Get("API-Version")
        if version == "" {
            version = "v1" // Default version
        }
        
        // Add version to context
        ctx := context.WithValue(r.Context(), "api_version", version)
        r = r.WithContext(ctx)
        
        next.ServeHTTP(w, r)
    })
}

// GetVersionFromContext extracts version from context
func GetVersionFromContext(ctx context.Context) string {
    if version, ok := ctx.Value("api_version").(string); ok {
        return version
    }
    return "v1"
}
```

## Request Validation

### Input Validation

```go
// internal/api/validation.go
package api

import (
    "fmt"
    "net/http"
    "regexp"
    "strings"
    "time"
)

// Validator represents a request validator
type Validator interface {
    Validate() error
}

// UserValidator validates user requests
type UserValidator struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

// Validate validates the user request
func (uv *UserValidator) Validate() error {
    var errors []string
    
    if strings.TrimSpace(uv.Name) == "" {
        errors = append(errors, "name is required")
    } else if len(uv.Name) < 2 {
        errors = append(errors, "name must be at least 2 characters")
    } else if len(uv.Name) > 100 {
        errors = append(errors, "name must be less than 100 characters")
    }
    
    if strings.TrimSpace(uv.Email) == "" {
        errors = append(errors, "email is required")
    } else if !isValidEmail(uv.Email) {
        errors = append(errors, "email must be a valid email address")
    }
    
    if len(errors) > 0 {
        return fmt.Errorf("validation failed: %s", strings.Join(errors, ", "))
    }
    
    return nil
}

// isValidEmail validates email format
func isValidEmail(email string) bool {
    pattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
    matched, _ := regexp.MatchString(pattern, email)
    return matched
}

// PaginationValidator validates pagination parameters
type PaginationValidator struct {
    Page    int `json:"page"`
    PerPage int `json:"per_page"`
}

// Validate validates pagination parameters
func (pv *PaginationValidator) Validate() error {
    if pv.Page < 1 {
        return fmt.Errorf("page must be greater than 0")
    }
    
    if pv.PerPage < 1 {
        return fmt.Errorf("per_page must be greater than 0")
    }
    
    if pv.PerPage > 100 {
        return fmt.Errorf("per_page must be less than or equal to 100")
    }
    
    return nil
}
```

### Schema Validation

```go
// internal/api/schema.go
package api

import (
    "encoding/json"
    "fmt"
    "net/http"
)

// SchemaValidator validates JSON schema
type SchemaValidator struct {
    schema map[string]interface{}
}

// NewSchemaValidator creates a new schema validator
func NewSchemaValidator(schema map[string]interface{}) *SchemaValidator {
    return &SchemaValidator{schema: schema}
}

// Validate validates JSON against schema
func (sv *SchemaValidator) Validate(data []byte) error {
    var jsonData interface{}
    if err := json.Unmarshal(data, &jsonData); err != nil {
        return fmt.Errorf("invalid JSON: %w", err)
    }
    
    // Implement schema validation logic
    // This is a simplified example
    return sv.validateObject(jsonData)
}

// validateObject validates a JSON object
func (sv *SchemaValidator) validateObject(obj interface{}) error {
    objMap, ok := obj.(map[string]interface{})
    if !ok {
        return fmt.Errorf("expected object")
    }
    
    // Check required fields
    if required, ok := sv.schema["required"].([]string); ok {
        for _, field := range required {
            if _, exists := objMap[field]; !exists {
                return fmt.Errorf("required field '%s' is missing", field)
            }
        }
    }
    
    return nil
}
```

## Response Formatting

### Standard Response Format

```go
// internal/api/response.go
package api

import (
    "encoding/json"
    "net/http"
    "time"
)

// ResponseBuilder builds API responses
type ResponseBuilder struct {
    response Response
}

// NewResponseBuilder creates a new response builder
func NewResponseBuilder() *ResponseBuilder {
    return &ResponseBuilder{
        response: Response{
            Meta: Meta{
                Timestamp: time.Now(),
            },
        },
    }
}

// WithData sets the response data
func (rb *ResponseBuilder) WithData(data interface{}) *ResponseBuilder {
    rb.response.Data = data
    return rb
}

// WithError adds an error to the response
func (rb *ResponseBuilder) WithError(err Error) *ResponseBuilder {
    rb.response.Errors = append(rb.response.Errors, err)
    return rb
}

// WithMeta sets the response metadata
func (rb *ResponseBuilder) WithMeta(meta Meta) *ResponseBuilder {
    rb.response.Meta = meta
    return rb
}

// WithLinks sets the response links
func (rb *ResponseBuilder) WithLinks(links Links) *ResponseBuilder {
    rb.response.Links = links
    return rb
}

// Build builds the response
func (rb *ResponseBuilder) Build() Response {
    return rb.response
}

// WriteResponse writes a response to the HTTP response writer
func WriteResponse(w http.ResponseWriter, statusCode int, response Response) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    json.NewEncoder(w).Encode(response)
}

// WriteSuccess writes a success response
func WriteSuccess(w http.ResponseWriter, data interface{}) {
    response := NewResponseBuilder().WithData(data).Build()
    WriteResponse(w, http.StatusOK, response)
}

// WriteCreated writes a created response
func WriteCreated(w http.ResponseWriter, data interface{}) {
    response := NewResponseBuilder().WithData(data).Build()
    WriteResponse(w, http.StatusCreated, response)
}

// WriteError writes an error response
func WriteError(w http.ResponseWriter, statusCode int, code, title, detail string) {
    response := NewResponseBuilder().WithError(Error{
        Status: fmt.Sprintf("%d", statusCode),
        Code:   code,
        Title:  title,
        Detail: detail,
    }).Build()
    
    WriteResponse(w, statusCode, response)
}
```

## Pagination

### Cursor-Based Pagination

```go
// internal/api/pagination.go
package api

import (
    "encoding/base64"
    "encoding/json"
    "fmt"
    "strconv"
    "time"
)

// Cursor represents a pagination cursor
type Cursor struct {
    ID        string    `json:"id"`
    Timestamp time.Time `json:"timestamp"`
    Direction string    `json:"direction"`
}

// EncodeCursor encodes a cursor to a string
func EncodeCursor(cursor Cursor) (string, error) {
    data, err := json.Marshal(cursor)
    if err != nil {
        return "", err
    }
    
    return base64.URLEncoding.EncodeToString(data), nil
}

// DecodeCursor decodes a cursor from a string
func DecodeCursor(cursorStr string) (Cursor, error) {
    data, err := base64.URLEncoding.DecodeString(cursorStr)
    if err != nil {
        return Cursor{}, err
    }
    
    var cursor Cursor
    err = json.Unmarshal(data, &cursor)
    return cursor, err
}

// PaginationParams represents pagination parameters
type PaginationParams struct {
    Cursor  string `json:"cursor,omitempty"`
    Limit   int    `json:"limit,omitempty"`
    Order   string `json:"order,omitempty"`
}

// Validate validates pagination parameters
func (pp *PaginationParams) Validate() error {
    if pp.Limit < 1 {
        pp.Limit = 20 // Default limit
    }
    
    if pp.Limit > 100 {
        return fmt.Errorf("limit must be less than or equal to 100")
    }
    
    if pp.Order != "" && pp.Order != "asc" && pp.Order != "desc" {
        return fmt.Errorf("order must be 'asc' or 'desc'")
    }
    
    return nil
}
```

### Offset-Based Pagination

```go
// internal/api/offset_pagination.go
package api

import (
    "fmt"
    "strconv"
)

// OffsetPaginationParams represents offset-based pagination parameters
type OffsetPaginationParams struct {
    Page    int `json:"page"`
    PerPage int `json:"per_page"`
}

// Validate validates offset pagination parameters
func (opp *OffsetPaginationParams) Validate() error {
    if opp.Page < 1 {
        opp.Page = 1
    }
    
    if opp.PerPage < 1 {
        opp.PerPage = 20
    }
    
    if opp.PerPage > 100 {
        return fmt.Errorf("per_page must be less than or equal to 100")
    }
    
    return nil
}

// GetOffset calculates the offset from page and per_page
func (opp *OffsetPaginationParams) GetOffset() int {
    return (opp.Page - 1) * opp.PerPage
}

// ParseOffsetPagination parses offset pagination from query parameters
func ParseOffsetPagination(queryParams map[string]string) OffsetPaginationParams {
    params := OffsetPaginationParams{
        Page:    1,
        PerPage: 20,
    }
    
    if pageStr, exists := queryParams["page"]; exists {
        if page, err := strconv.Atoi(pageStr); err == nil && page > 0 {
            params.Page = page
        }
    }
    
    if perPageStr, exists := queryParams["per_page"]; exists {
        if perPage, err := strconv.Atoi(perPageStr); err == nil && perPage > 0 {
            params.PerPage = perPage
        }
    }
    
    return params
}
```

## Error Handling

### API Error Types

```go
// internal/api/errors.go
package api

import (
    "fmt"
    "net/http"
)

// APIError represents an API error
type APIError struct {
    Status  int    `json:"status"`
    Code    string `json:"code"`
    Title   string `json:"title"`
    Detail  string `json:"detail"`
    Source  Source `json:"source,omitempty"`
}

// Error implements the error interface
func (ae *APIError) Error() string {
    return fmt.Sprintf("[%d] %s: %s", ae.Status, ae.Title, ae.Detail)
}

// NewAPIError creates a new API error
func NewAPIError(status int, code, title, detail string) *APIError {
    return &APIError{
        Status: status,
        Code:   code,
        Title:  title,
        Detail: detail,
    }
}

// Common API errors
var (
    ErrBadRequest = NewAPIError(
        http.StatusBadRequest,
        "BAD_REQUEST",
        "Bad Request",
        "The request is invalid",
    )
    
    ErrUnauthorized = NewAPIError(
        http.StatusUnauthorized,
        "UNAUTHORIZED",
        "Unauthorized",
        "Authentication required",
    )
    
    ErrForbidden = NewAPIError(
        http.StatusForbidden,
        "FORBIDDEN",
        "Forbidden",
        "Access denied",
    )
    
    ErrNotFound = NewAPIError(
        http.StatusNotFound,
        "NOT_FOUND",
        "Not Found",
        "The requested resource was not found",
    )
    
    ErrConflict = NewAPIError(
        http.StatusConflict,
        "CONFLICT",
        "Conflict",
        "The request conflicts with the current state",
    )
    
    ErrUnprocessableEntity = NewAPIError(
        http.StatusUnprocessableEntity,
        "UNPROCESSABLE_ENTITY",
        "Unprocessable Entity",
        "The request is well-formed but contains semantic errors",
    )
    
    ErrInternalServerError = NewAPIError(
        http.StatusInternalServerError,
        "INTERNAL_SERVER_ERROR",
        "Internal Server Error",
        "An unexpected error occurred",
    )
)
```

### Error Handler

```go
// internal/api/error_handler.go
package api

import (
    "encoding/json"
    "log"
    "net/http"
)

// ErrorHandler handles API errors
type ErrorHandler struct {
    logger Logger
}

// NewErrorHandler creates a new error handler
func NewErrorHandler(logger Logger) *ErrorHandler {
    return &ErrorHandler{logger: logger}
}

// HandleError handles an error
func (eh *ErrorHandler) HandleError(w http.ResponseWriter, r *http.Request, err error) {
    eh.logger.Error("API error", map[string]interface{}{
        "method": r.Method,
        "path":   r.URL.Path,
        "error":  err.Error(),
    })
    
    var apiErr *APIError
    switch err := err.(type) {
    case *APIError:
        apiErr = err
    case *ValidationError:
        apiErr = NewAPIError(
            http.StatusBadRequest,
            "VALIDATION_ERROR",
            "Validation Error",
            err.Error(),
        )
    case *NotFoundError:
        apiErr = NewAPIError(
            http.StatusNotFound,
            "NOT_FOUND",
            "Not Found",
            err.Error(),
        )
    default:
        apiErr = ErrInternalServerError
    }
    
    eh.writeErrorResponse(w, apiErr)
}

// writeErrorResponse writes an error response
func (eh *ErrorHandler) writeErrorResponse(w http.ResponseWriter, apiErr *APIError) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(apiErr.Status)
    
    response := Response{
        Errors: []Error{
            {
                Status: fmt.Sprintf("%d", apiErr.Status),
                Code:   apiErr.Code,
                Title:  apiErr.Title,
                Detail: apiErr.Detail,
            },
        },
    }
    
    json.NewEncoder(w).Encode(response)
}
```

## OpenAPI Integration

### OpenAPI Specification

```go
// internal/api/openapi.go
package api

import (
    "encoding/json"
    "net/http"
)

// OpenAPISpec represents an OpenAPI specification
type OpenAPISpec struct {
    OpenAPI string                 `json:"openapi"`
    Info    Info                   `json:"info"`
    Servers []Server               `json:"servers,omitempty"`
    Paths   map[string]PathItem    `json:"paths"`
    Components Components          `json:"components,omitempty"`
}

// Info represents API information
type Info struct {
    Title       string `json:"title"`
    Version     string `json:"version"`
    Description string `json:"description,omitempty"`
}

// Server represents a server
type Server struct {
    URL         string `json:"url"`
    Description string `json:"description,omitempty"`
}

// PathItem represents a path item
type PathItem struct {
    Get    *Operation `json:"get,omitempty"`
    Post   *Operation `json:"post,omitempty"`
    Put    *Operation `json:"put,omitempty"`
    Delete *Operation `json:"delete,omitempty"`
}

// Operation represents an operation
type Operation struct {
    Summary     string              `json:"summary,omitempty"`
    Description string              `json:"description,omitempty"`
    Parameters  []Parameter         `json:"parameters,omitempty"`
    RequestBody *RequestBody        `json:"requestBody,omitempty"`
    Responses   map[string]Response `json:"responses"`
    Tags        []string            `json:"tags,omitempty"`
}

// Parameter represents a parameter
type Parameter struct {
    Name        string `json:"name"`
    In          string `json:"in"`
    Description string `json:"description,omitempty"`
    Required    bool   `json:"required,omitempty"`
    Schema      Schema `json:"schema,omitempty"`
}

// RequestBody represents a request body
type RequestBody struct {
    Description string              `json:"description,omitempty"`
    Content     map[string]Content `json:"content"`
    Required    bool                `json:"required,omitempty"`
}

// Response represents a response
type Response struct {
    Description string              `json:"description"`
    Content     map[string]Content  `json:"content,omitempty"`
}

// Content represents content
type Content struct {
    Schema Schema `json:"schema,omitempty"`
}

// Schema represents a schema
type Schema struct {
    Type       string            `json:"type,omitempty"`
    Properties map[string]Schema `json:"properties,omitempty"`
    Required   []string          `json:"required,omitempty"`
}

// Components represents components
type Components struct {
    Schemas map[string]Schema `json:"schemas,omitempty"`
}

// OpenAPIHandler handles OpenAPI requests
type OpenAPIHandler struct {
    spec OpenAPISpec
}

// NewOpenAPIHandler creates a new OpenAPI handler
func NewOpenAPIHandler() *OpenAPIHandler {
    return &OpenAPIHandler{
        spec: OpenAPISpec{
            OpenAPI: "3.0.0",
            Info: Info{
                Title:   "My API",
                Version: "1.0.0",
            },
            Paths: make(map[string]PathItem),
        },
    }
}

// ServeHTTP implements http.Handler
func (h *OpenAPIHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(h.spec)
}
```

## Testing API Design

### API Testing

```go
// internal/api/api_test.go
package api

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestAPI_CreateUser(t *testing.T) {
    // Arrange
    handler := NewUserHandler()
    userReq := CreateUserRequest{
        Data: User{
            Type: "user",
            Attributes: UserAttributes{
                Name:  "John Doe",
                Email: "john@example.com",
            },
        },
    }
    
    jsonData, err := json.Marshal(userReq)
    require.NoError(t, err)
    
    req := httptest.NewRequest("POST", "/api/v1/users", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusCreated, w.Code)
    assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
    
    var response Response
    err = json.Unmarshal(w.Body.Bytes(), &response)
    require.NoError(t, err)
    assert.NotNil(t, response.Data)
}

func TestAPI_ValidationError(t *testing.T) {
    // Arrange
    handler := NewUserHandler()
    invalidReq := CreateUserRequest{
        Data: User{
            Type: "user",
            Attributes: UserAttributes{
                Name:  "", // Invalid: empty name
                Email: "invalid-email", // Invalid: malformed email
            },
        },
    }
    
    jsonData, err := json.Marshal(invalidReq)
    require.NoError(t, err)
    
    req := httptest.NewRequest("POST", "/api/v1/users", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusBadRequest, w.Code)
    
    var response Response
    err = json.Unmarshal(w.Body.Bytes(), &response)
    require.NoError(t, err)
    assert.NotEmpty(t, response.Errors)
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Basic API response
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    user, err := h.userService.GetUser(userID)
    if err != nil {
        WriteError(w, http.StatusNotFound, "NOT_FOUND", "User not found", err.Error())
        return
    }
    
    WriteSuccess(w, user)
}

// 2. Request validation
func (h *Handler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    if err := ReadJSON(r, &req); err != nil {
        WriteError(w, http.StatusBadRequest, "BAD_REQUEST", "Invalid JSON", err.Error())
        return
    }
    
    if err := req.Validate(); err != nil {
        WriteError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Validation failed", err.Error())
        return
    }
    
    // Process request...
}

// 3. API versioning
versionHandler := NewVersionHandler()
versionHandler.RegisterVersion("v1", v1Handler)
versionHandler.RegisterVersion("v2", v2Handler)
```

### Essential Patterns

```go
// Error handling
func (h *Handler) HandleRequest(w http.ResponseWriter, r *http.Request) {
    if err := h.processRequest(r); err != nil {
        h.errorHandler.HandleError(w, r, err)
        return
    }
    
    WriteSuccess(w, result)
}

// Pagination
func (h *Handler) ListUsers(w http.ResponseWriter, r *http.Request) {
    params := ParseOffsetPagination(r.URL.Query())
    if err := params.Validate(); err != nil {
        WriteError(w, http.StatusBadRequest, "BAD_REQUEST", "Invalid pagination", err.Error())
        return
    }
    
    users, total, err := h.userService.ListUsers(params)
    if err != nil {
        WriteError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to list users", err.Error())
        return
    }
    
    response := NewResponseBuilder().
        WithData(users).
        WithMeta(Meta{TotalCount: total, Page: params.Page, PerPage: params.PerPage}).
        Build()
    
    WriteResponse(w, http.StatusOK, response)
}
```

---

*This guide provides the complete machinery for building production-ready RESTful APIs in Go applications. Each pattern includes implementation examples, validation strategies, and real-world usage patterns for enterprise deployment.*
