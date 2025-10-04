# Go Development Environment Best Practices

**Objective**: Master senior-level Go development environment setup and operation across macOS, Linux, and Windows (WSL and native). Copy-paste runnable, auditable, and production-ready.

## Core Principles

- **Reproducible Builds**: Lock Go versions, dependencies, and toolchains
- **Fast Feedback Loops**: Hot reloading, instant testing, and rapid iteration
- **Cross-Platform Consistency**: Works identically on macOS, Linux, and Windows
- **Security First**: Secure dependency management and supply chain integrity
- **Performance Optimized**: Fast compilation, efficient tooling, and minimal overhead

## Environment Setup

### Go Version Management

```bash
# Install g (Go version manager)
curl -sSL https://git.io/g-install | sh -s

# Install latest Go
g install latest

# Use specific version for project
g use 1.21.5

# Verify installation
go version
```

### Project Structure

```
my-go-project/
├── cmd/                    # Application entry points
│   ├── server/
│   │   └── main.go
│   └── worker/
│       └── main.go
├── internal/              # Private application code
│   ├── api/
│   ├── config/
│   └── service/
├── pkg/                   # Public library code
│   ├── client/
│   └── utils/
├── api/                    # API definitions
│   └── openapi.yaml
├── scripts/               # Build and deployment scripts
├── testdata/              # Test fixtures
├── go.mod
├── go.sum
├── Makefile
├── .golangci.yml
├── .gitignore
└── README.md
```

### Essential Tools

```bash
# Install essential Go tools
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/air-verse/air@latest
go install github.com/golang/mock/mockgen@latest
go install github.com/swaggo/swag/cmd/swag@latest
go install github.com/cosmtrek/air@latest
go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
```

## Development Workflow

### Hot Reloading with Air

```yaml
# .air.toml
root = "."
testdata_dir = "testdata"
tmp_dir = "tmp"

[build]
  args_bin = []
  bin = "./tmp/main"
  cmd = "go build -o ./tmp/main ./cmd/server"
  delay = 1000
  exclude_dir = ["assets", "tmp", "vendor", "testdata"]
  exclude_file = []
  exclude_regex = ["_test.go"]
  exclude_unchanged = false
  follow_symlink = false
  full_bin = ""
  include_dir = []
  include_ext = ["go", "tpl", "tmpl", "html"]
  include_file = []
  kill_delay = "0s"
  log = "build-errors.log"
  poll = false
  poll_interval = 0
  rerun = false
  rerun_delay = 500
  send_interrupt = false
  stop_on_root = false

[color]
  app = ""
  build = "yellow"
  main = "magenta"
  runner = "green"
  watcher = "cyan"

[log]
  main_only = false
  time = false

[misc]
  clean_on_exit = false

[screen]
  clear_on_rebuild = false
  keep_scroll = true
```

### Linting Configuration

```yaml
# .golangci.yml
run:
  timeout: 5m
  issues-exit-code: 1
  tests: true
  skip-dirs:
    - vendor
    - testdata
  skip-files:
    - ".*\\.pb\\.go$"

linters-settings:
  gocyclo:
    min-complexity: 15
  goconst:
    min-len: 3
    min-occurrences: 3
  gocritic:
    enabled-tags:
      - diagnostic
      - experimental
      - opinionated
      - performance
      - style
  govet:
    check-shadowing: true
  gosec:
    severity: medium
    confidence: medium
  misspell:
    locale: US
  lll:
    line-length: 120
  funlen:
    lines: 100
    statements: 50
  gocognit:
    min-complexity: 20

linters:
  enable:
    - bodyclose
    - deadcode
    - depguard
    - dogsled
    - dupl
    - errcheck
    - exportloopref
    - exhaustive
    - funlen
    - gochecknoinits
    - goconst
    - gocritic
    - gocyclo
    - gofmt
    - goimports
    - gomnd
    - goprintffuncname
    - gosec
    - gosimple
    - govet
    - ineffassign
    - lll
    - misspell
    - nakedret
    - noctx
    - nolintlint
    - rowserrcheck
    - staticcheck
    - structcheck
    - stylecheck
    - typecheck
    - unconvert
    - unparam
    - unused
    - varcheck
    - whitespace

issues:
  exclude-rules:
    - path: _test\.go
      linters:
        - gomnd
        - funlen
        - goconst
```

### Testing Configuration

```go
// internal/testing/testutil.go
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

// TestDB creates a mock database for testing
func TestDB(t *testing.T) (*sql.DB, sqlmock.Sqlmock) {
    db, mock, err := sqlmock.New()
    require.NoError(t, err)
    
    t.Cleanup(func() {
        db.Close()
    })
    
    return db, mock
}

// TestContext creates a test context with timeout
func TestContext(t *testing.T) context.Context {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    t.Cleanup(cancel)
    return ctx
}

// AssertNoError is a helper for common error assertions
func AssertNoError(t *testing.T, err error) {
    t.Helper()
    assert.NoError(t, err)
}

// AssertError is a helper for error assertions
func AssertError(t *testing.T, err error) {
    t.Helper()
    assert.Error(t, err)
}
```

## Build and Deployment

### Makefile

```makefile
# Makefile
.PHONY: help build test lint clean docker

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build the application
	@echo "Building application..."
	go build -o bin/server ./cmd/server
	go build -o bin/worker ./cmd/worker

build-linux: ## Build for Linux
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 go build -o bin/server-linux ./cmd/server
	GOOS=linux GOARCH=amd64 go build -o bin/worker-linux ./cmd/worker

# Testing
test: ## Run tests
	@echo "Running tests..."
	go test -v -race -coverprofile=coverage.out ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	go test -v -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

benchmark: ## Run benchmarks
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Linting
lint: ## Run linters
	@echo "Running linters..."
	golangci-lint run

lint-fix: ## Run linters with auto-fix
	@echo "Running linters with auto-fix..."
	golangci-lint run --fix

# Security
security: ## Run security checks
	@echo "Running security checks..."
	gosec ./...

# Development
dev: ## Start development server with hot reload
	@echo "Starting development server..."
	air

# Docker
docker: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t my-go-app:latest .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8080:8080 my-go-app:latest

# Cleanup
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -f coverage.out coverage.html
	go clean

# Dependencies
deps: ## Download dependencies
	@echo "Downloading dependencies..."
	go mod download
	go mod tidy

deps-update: ## Update dependencies
	@echo "Updating dependencies..."
	go get -u ./...
	go mod tidy

# Code generation
generate: ## Generate code
	@echo "Generating code..."
	go generate ./...

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	swag init -g cmd/server/main.go -o api/docs
```

### Docker Configuration

```dockerfile
# Dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Expose port
EXPOSE 8080

# Run the application
CMD ["./main"]
```

### Multi-stage Docker Build

```dockerfile
# Dockerfile.multi
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# Test stage
FROM builder AS tester

# Run tests
RUN go test -v -race -coverprofile=coverage.out ./...

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Expose port
EXPOSE 8080

# Run the application
CMD ["./main"]
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

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
    
    - name: Verify dependencies
      run: go mod verify
    
    - name: Run tests
      run: go test -v -race -coverprofile=coverage.out ./...
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
    
    - name: Run linters
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest
    
    - name: Run security checks
      uses: securecodewarrior/github-action-gosec@master
      with:
        args: './...'
    
    - name: Build
      run: go build -o bin/server ./cmd/server
    
    - name: Build Docker image
      run: docker build -t my-go-app:${{ github.sha }} .
```

## Performance Optimization

### Profiling Configuration

```go
// cmd/server/main.go
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof"
    "os"
    "os/signal"
    "syscall"
)

func main() {
    // Enable pprof endpoints
    go func() {
        log.Println("Starting pprof server on :6060")
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // Graceful shutdown
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)
    
    // Start your application
    // ...
    
    <-c
    log.Println("Shutting down gracefully...")
}
```

### Benchmarking

```go
// internal/service/benchmark_test.go
package service

import (
    "testing"
)

func BenchmarkProcessData(b *testing.B) {
    service := NewService()
    data := generateTestData(1000)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        service.ProcessData(data)
    }
}

func BenchmarkConcurrentProcess(b *testing.B) {
    service := NewService()
    data := generateTestData(1000)
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            service.ProcessData(data)
        }
    })
}
```

## Security Best Practices

### Dependency Security

```bash
# Check for vulnerabilities
go list -json -deps ./... | nancy sleuth

# Update dependencies
go get -u ./...
go mod tidy

# Audit dependencies
gosec ./...
```

### Secure Configuration

```go
// internal/config/config.go
package config

import (
    "os"
    "strconv"
    "time"
)

type Config struct {
    Server   ServerConfig
    Database DatabaseConfig
    Security SecurityConfig
}

type ServerConfig struct {
    Port         string
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
}

type DatabaseConfig struct {
    Host     string
    Port     int
    Username string
    Password string
    SSLMode  string
}

type SecurityConfig struct {
    JWTSecret     string
    BcryptCost    int
    RateLimitRPS  int
}

func Load() (*Config, error) {
    return &Config{
        Server: ServerConfig{
            Port:         getEnv("SERVER_PORT", "8080"),
            ReadTimeout:  getDuration("SERVER_READ_TIMEOUT", 30*time.Second),
            WriteTimeout: getDuration("SERVER_WRITE_TIMEOUT", 30*time.Second),
        },
        Database: DatabaseConfig{
            Host:     getEnv("DB_HOST", "localhost"),
            Port:     getInt("DB_PORT", 5432),
            Username: getEnv("DB_USERNAME", "postgres"),
            Password: getEnv("DB_PASSWORD", ""),
            SSLMode:  getEnv("DB_SSL_MODE", "require"),
        },
        Security: SecurityConfig{
            JWTSecret:    getEnv("JWT_SECRET", ""),
            BcryptCost:   getInt("BCRYPT_COST", 12),
            RateLimitRPS: getInt("RATE_LIMIT_RPS", 100),
        },
    }, nil
}

func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func getInt(key string, defaultValue int) int {
    if value := os.Getenv(key); value != "" {
        if intValue, err := strconv.Atoi(value); err == nil {
            return intValue
        }
    }
    return defaultValue
}

func getDuration(key string, defaultValue time.Duration) time.Duration {
    if value := os.Getenv(key); value != "" {
        if duration, err := time.ParseDuration(value); err == nil {
            return duration
        }
    }
    return defaultValue
}
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Install Go version manager
curl -sSL https://git.io/g-install | sh -s

# 2. Install latest Go
g install latest

# 3. Install essential tools
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/air-verse/air@latest

# 4. Create new project
mkdir my-go-project && cd my-go-project
go mod init github.com/username/my-go-project

# 5. Start development
air
```

### Essential Commands

```bash
# Development
make dev          # Start with hot reload
make test         # Run tests
make lint         # Run linters
make build        # Build application

# Production
make docker       # Build Docker image
make security     # Run security checks
make benchmark    # Run benchmarks
```

---

*This guide provides the complete machinery for setting up a production-ready Go development environment. Each pattern includes configuration examples, tooling setup, and real-world implementation strategies for enterprise deployment.*
