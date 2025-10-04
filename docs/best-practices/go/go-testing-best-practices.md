# Go Testing Best Practices

**Objective**: Master senior-level Go testing patterns for production systems. When you need to build robust, maintainable test suites, when you want to follow proven methodologies, when you need enterprise-grade testing patterns—these best practices become your weapon of choice.

## Core Principles

- **Test-Driven Development**: Write tests before implementation
- **Comprehensive Coverage**: Test all code paths and edge cases
- **Fast Feedback**: Keep tests fast and focused
- **Isolation**: Tests should be independent and repeatable
- **Readability**: Tests should be self-documenting and easy to understand

## Testing Framework Setup

### Basic Testing Structure

```go
// internal/testing/setup.go
package testing

import (
    "context"
    "database/sql"
    "testing"
    "time"

    "github.com/DATA-DOG/go-sqlmock"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// TestSuite provides common testing utilities
type TestSuite struct {
    t      *testing.T
    db     *sql.DB
    mock   sqlmock.Sqlmock
    ctx    context.Context
    cancel context.CancelFunc
}

// NewTestSuite creates a new test suite
func NewTestSuite(t *testing.T) *TestSuite {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    
    return &TestSuite{
        t:      t,
        ctx:    ctx,
        cancel: cancel,
    }
}

// SetupDB sets up a mock database
func (ts *TestSuite) SetupDB() {
    db, mock, err := sqlmock.New()
    require.NoError(ts.t, err)
    
    ts.db = db
    ts.mock = mock
}

// Teardown cleans up test resources
func (ts *TestSuite) Teardown() {
    if ts.db != nil {
        ts.db.Close()
    }
    ts.cancel()
}

// AssertNoError is a helper for common error assertions
func (ts *TestSuite) AssertNoError(err error) {
    ts.t.Helper()
    assert.NoError(ts.t, err)
}

// AssertError is a helper for error assertions
func (ts *TestSuite) AssertError(err error) {
    ts.t.Helper()
    assert.Error(ts.t, err)
}
```

### Test Data Management

```go
// internal/testing/testdata.go
package testing

import (
    "encoding/json"
    "os"
    "path/filepath"
)

// TestDataManager manages test data
type TestDataManager struct {
    basePath string
}

// NewTestDataManager creates a new test data manager
func NewTestDataManager() *TestDataManager {
    return &TestDataManager{
        basePath: "testdata",
    }
}

// LoadJSON loads JSON test data
func (tdm *TestDataManager) LoadJSON(filename string, target interface{}) error {
    path := filepath.Join(tdm.basePath, filename)
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    
    return json.Unmarshal(data, target)
}

// LoadText loads text test data
func (tdm *TestDataManager) LoadText(filename string) (string, error) {
    path := filepath.Join(tdm.basePath, filename)
    data, err := os.ReadFile(path)
    if err != nil {
        return "", err
    }
    
    return string(data), nil
}

// CreateTestData creates test data
func (tdm *TestDataManager) CreateTestData(filename string, data interface{}) error {
    path := filepath.Join(tdm.basePath, filename)
    
    jsonData, err := json.MarshalIndent(data, "", "  ")
    if err != nil {
        return err
    }
    
    return os.WriteFile(path, jsonData, 0644)
}
```

## Unit Testing Patterns

### Basic Unit Tests

```go
// internal/service/user_test.go
package service

import (
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestUserService_CreateUser(t *testing.T) {
    // Arrange
    service := NewUserService()
    user := &User{
        Name:     "John Doe",
        Email:    "john@example.com",
        CreatedAt: time.Now(),
    }
    
    // Act
    result, err := service.CreateUser(user)
    
    // Assert
    require.NoError(t, err)
    assert.NotNil(t, result)
    assert.Equal(t, user.Name, result.Name)
    assert.Equal(t, user.Email, result.Email)
    assert.NotZero(t, result.ID)
}

func TestUserService_CreateUser_ValidationError(t *testing.T) {
    // Arrange
    service := NewUserService()
    user := &User{
        Name:  "", // Invalid: empty name
        Email: "invalid-email", // Invalid: malformed email
    }
    
    // Act
    result, err := service.CreateUser(user)
    
    // Assert
    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "validation failed")
}

func TestUserService_GetUser(t *testing.T) {
    // Arrange
    service := NewUserService()
    user := &User{
        Name:  "Jane Doe",
        Email: "jane@example.com",
    }
    
    created, err := service.CreateUser(user)
    require.NoError(t, err)
    
    // Act
    result, err := service.GetUser(created.ID)
    
    // Assert
    require.NoError(t, err)
    assert.Equal(t, created.ID, result.ID)
    assert.Equal(t, created.Name, result.Name)
    assert.Equal(t, created.Email, result.Email)
}

func TestUserService_GetUser_NotFound(t *testing.T) {
    // Arrange
    service := NewUserService()
    nonExistentID := "non-existent-id"
    
    // Act
    result, err := service.GetUser(nonExistentID)
    
    // Assert
    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "not found")
}
```

### Table-Driven Tests

```go
// internal/service/user_test.go
func TestUserService_ValidateEmail(t *testing.T) {
    tests := []struct {
        name     string
        email    string
        expected bool
    }{
        {
            name:     "valid email",
            email:    "user@example.com",
            expected: true,
        },
        {
            name:     "invalid email - no @",
            email:    "userexample.com",
            expected: false,
        },
        {
            name:     "invalid email - no domain",
            email:    "user@",
            expected: false,
        },
        {
            name:     "invalid email - empty",
            email:    "",
            expected: false,
        },
        {
            name:     "invalid email - spaces",
            email:    "user @example.com",
            expected: false,
        },
    }
    
    service := NewUserService()
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := service.ValidateEmail(tt.email)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

### Mock Testing

```go
// internal/service/user_test.go
func TestUserService_CreateUser_WithMock(t *testing.T) {
    // Arrange
    mockDB := &MockDatabase{}
    mockEmailService := &MockEmailService{}
    
    service := &UserService{
        db:           mockDB,
        emailService: mockEmailService,
    }
    
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    // Set up mock expectations
    mockDB.On("Save", user).Return(nil)
    mockEmailService.On("SendWelcomeEmail", user.Email).Return(nil)
    
    // Act
    result, err := service.CreateUser(user)
    
    // Assert
    require.NoError(t, err)
    assert.NotNil(t, result)
    
    // Verify mock calls
    mockDB.AssertExpectations(t)
    mockEmailService.AssertExpectations(t)
}

// Mock interfaces
type MockDatabase struct {
    mock.Mock
}

func (m *MockDatabase) Save(user *User) error {
    args := m.Called(user)
    return args.Error(0)
}

type MockEmailService struct {
    mock.Mock
}

func (m *MockEmailService) SendWelcomeEmail(email string) error {
    args := m.Called(email)
    return args.Error(0)
}
```

## Integration Testing

### Database Integration Tests

```go
// internal/service/user_integration_test.go
package service

import (
    "context"
    "database/sql"
    "testing"
    "time"

    "github.com/DATA-DOG/go-sqlmock"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestUserService_Integration_CreateUser(t *testing.T) {
    // Arrange
    db, mock, err := sqlmock.New()
    require.NoError(t, err)
    defer db.Close()
    
    service := &UserService{db: db}
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    // Set up mock expectations
    mock.ExpectBegin()
    mock.ExpectQuery("INSERT INTO users").
        WithArgs(user.Name, user.Email).
        WillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(1))
    mock.ExpectCommit()
    
    // Act
    result, err := service.CreateUser(user)
    
    // Assert
    require.NoError(t, err)
    assert.NotNil(t, result)
    assert.Equal(t, int64(1), result.ID)
    
    // Verify all expectations were met
    assert.NoError(t, mock.ExpectationsWereMet())
}

func TestUserService_Integration_GetUser(t *testing.T) {
    // Arrange
    db, mock, err := sqlmock.New()
    require.NoError(t, err)
    defer db.Close()
    
    service := &UserService{db: db}
    userID := int64(1)
    
    // Set up mock expectations
    rows := sqlmock.NewRows([]string{"id", "name", "email", "created_at"}).
        AddRow(userID, "John Doe", "john@example.com", time.Now())
    
    mock.ExpectQuery("SELECT id, name, email, created_at FROM users WHERE id = ?").
        WithArgs(userID).
        WillReturnRows(rows)
    
    // Act
    result, err := service.GetUser(userID)
    
    // Assert
    require.NoError(t, err)
    assert.NotNil(t, result)
    assert.Equal(t, userID, result.ID)
    assert.Equal(t, "John Doe", result.Name)
    assert.Equal(t, "john@example.com", result.Email)
    
    // Verify all expectations were met
    assert.NoError(t, mock.ExpectationsWereMet())
}
```

### HTTP Integration Tests

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
    handler := NewUserHandler()
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    jsonData, err := json.Marshal(user)
    require.NoError(t, err)
    
    req := httptest.NewRequest("POST", "/users", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusCreated, w.Code)
    
    var response User
    err = json.Unmarshal(w.Body.Bytes(), &response)
    require.NoError(t, err)
    assert.Equal(t, user.Name, response.Name)
    assert.Equal(t, user.Email, response.Email)
    assert.NotZero(t, response.ID)
}

func TestUserHandler_CreateUser_InvalidJSON(t *testing.T) {
    // Arrange
    handler := NewUserHandler()
    invalidJSON := `{"name": "John", "email": "invalid-email"`
    
    req := httptest.NewRequest("POST", "/users", bytes.NewBufferString(invalidJSON))
    req.Header.Set("Content-Type", "application/json")
    w := httptest.NewRecorder()
    
    // Act
    handler.CreateUser(w, req)
    
    // Assert
    assert.Equal(t, http.StatusBadRequest, w.Code)
    assert.Contains(t, w.Body.String(), "invalid JSON")
}
```

## Performance Testing

### Benchmark Tests

```go
// internal/service/user_benchmark_test.go
package service

import (
    "testing"
    "time"
)

func BenchmarkUserService_CreateUser(b *testing.B) {
    service := NewUserService()
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := service.CreateUser(user)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkUserService_GetUser(b *testing.B) {
    service := NewUserService()
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    created, err := service.CreateUser(user)
    if err != nil {
        b.Fatal(err)
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := service.GetUser(created.ID)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkUserService_ConcurrentCreateUser(b *testing.B) {
    service := NewUserService()
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            user := &User{
                Name:  "John Doe",
                Email: "john@example.com",
            }
            _, err := service.CreateUser(user)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}
```

## Test Coverage

### Coverage Analysis

```go
// internal/service/coverage_test.go
package service

import (
    "testing"
    "time"
)

func TestUserService_AllPaths(t *testing.T) {
    service := NewUserService()
    
    // Test successful creation
    user := &User{
        Name:  "John Doe",
        Email: "john@example.com",
    }
    
    result, err := service.CreateUser(user)
    require.NoError(t, err)
    assert.NotNil(t, result)
    
    // Test validation error
    invalidUser := &User{
        Name:  "", // Invalid
        Email: "invalid-email",
    }
    
    _, err = service.CreateUser(invalidUser)
    assert.Error(t, err)
    
    // Test duplicate email
    duplicateUser := &User{
        Name:  "Jane Doe",
        Email: "john@example.com", // Duplicate
    }
    
    _, err = service.CreateUser(duplicateUser)
    assert.Error(t, err)
    
    // Test successful retrieval
    retrieved, err := service.GetUser(result.ID)
    require.NoError(t, err)
    assert.Equal(t, result.ID, retrieved.ID)
    
    // Test not found
    _, err = service.GetUser("non-existent")
    assert.Error(t, err)
}
```

### Coverage Reporting

```bash
# Run tests with coverage
go test -coverprofile=coverage.out ./...

# View coverage report
go tool cover -html=coverage.out

# Coverage by function
go tool cover -func=coverage.out
```

## Test Utilities

### Test Helpers

```go
// internal/testing/helpers.go
package testing

import (
    "crypto/rand"
    "encoding/hex"
    "testing"
    "time"
)

// RandomString generates a random string
func RandomString(length int) string {
    bytes := make([]byte, length/2)
    rand.Read(bytes)
    return hex.EncodeToString(bytes)
}

// RandomEmail generates a random email
func RandomEmail() string {
    return RandomString(8) + "@example.com"
}

// RandomUser creates a random user
func RandomUser() *User {
    return &User{
        Name:     RandomString(10),
        Email:    RandomEmail(),
        CreatedAt: time.Now(),
    }
}

// AssertTimeEqual asserts that two times are equal within a tolerance
func AssertTimeEqual(t *testing.T, expected, actual time.Time, tolerance time.Duration) {
    t.Helper()
    diff := expected.Sub(actual)
    if diff < 0 {
        diff = -diff
    }
    if diff > tolerance {
        t.Errorf("expected time %v, got %v (diff: %v)", expected, actual, diff)
    }
}
```

### Test Fixtures

```go
// internal/testing/fixtures.go
package testing

import (
    "encoding/json"
    "os"
    "path/filepath"
)

// LoadFixture loads a test fixture
func LoadFixture(filename string, target interface{}) error {
    path := filepath.Join("testdata", "fixtures", filename)
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    
    return json.Unmarshal(data, target)
}

// SaveFixture saves a test fixture
func SaveFixture(filename string, data interface{}) error {
    path := filepath.Join("testdata", "fixtures", filename)
    jsonData, err := json.MarshalIndent(data, "", "  ")
    if err != nil {
        return err
    }
    
    return os.WriteFile(path, jsonData, 0644)
}
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Cache Go modules
      uses: actions/cache@v3
      with:
        path: ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-
    
    - name: Download dependencies
      run: go mod download
    
    - name: Run tests
      run: go test -v -race -coverprofile=coverage.out ./...
    
    - name: Run benchmarks
      run: go test -bench=. -benchmem ./...
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Run tests
go test ./...

# 2. Run tests with coverage
go test -cover ./...

# 3. Run benchmarks
go test -bench=. ./...

# 4. Run tests with race detection
go test -race ./...
```

### Essential Commands

```bash
# Test specific package
go test ./internal/service

# Test specific function
go test -run TestUserService_CreateUser

# Benchmark specific function
go test -bench=BenchmarkUserService_CreateUser

# Coverage report
go tool cover -html=coverage.out
```

### Test Structure

```go
// Test function naming
func TestPackageName_FunctionName_Scenario(t *testing.T) {
    // Arrange
    // Act
    // Assert
}

// Benchmark function naming
func BenchmarkPackageName_FunctionName(b *testing.B) {
    // Setup
    b.ResetTimer()
    // Benchmark code
}
```

---

*This guide provides the complete machinery for building production-ready test suites in Go applications. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise deployment.*
