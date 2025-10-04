# Go Containerization Best Practices

**Objective**: Master senior-level Go containerization patterns for production systems. When you need to build efficient, secure containers, when you want to optimize container performance, when you need enterprise-grade containerization patterns—these best practices become your weapon of choice.

## Core Principles

- **Multi-Stage Builds**: Optimize image size and build time
- **Security First**: Minimize attack surface and vulnerabilities
- **Layer Optimization**: Reduce image layers and improve caching
- **Runtime Optimization**: Configure containers for production workloads
- **Observability**: Include monitoring and logging capabilities

## Multi-Stage Docker Builds

### Basic Multi-Stage Build

```dockerfile
# Dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

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

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Change ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./main"]
```

### Advanced Multi-Stage Build

```dockerfile
# Dockerfile.advanced
# Dependencies stage
FROM golang:1.21-alpine AS deps
RUN apk add --no-cache git ca-certificates tzdata
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

# Test stage
FROM deps AS test
COPY . .
RUN go test -v -race -coverprofile=coverage.out ./...

# Build stage
FROM deps AS builder
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags='-w -s -extldflags "-static"' \
    -o main ./cmd/server

# Security scan stage
FROM builder AS security
RUN apk add --no-cache security-scan-tool
RUN security-scan-tool /app/main

# Final stage
FROM scratch AS final

# Copy ca-certificates
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy the binary
COPY --from=builder /app/main /main

# Set timezone
ENV TZ=UTC

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/main", "health"]

# Run the application
ENTRYPOINT ["/main"]
```

## Security Best Practices

### Security-Hardened Dockerfile

```dockerfile
# Dockerfile.secure
# Build stage
FROM golang:1.21-alpine AS builder

# Install security tools
RUN apk add --no-cache \
    git \
    ca-certificates \
    tzdata \
    gosu \
    dumb-init

# Create build user
RUN addgroup -g 1001 -S buildgroup && \
    adduser -u 1001 -S builduser -G buildgroup

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build with security flags
RUN CGO_ENABLED=0 GOOS=linux go build \
    -a -installsuffix cgo \
    -ldflags='-w -s -extldflags "-static"' \
    -buildmode=pie \
    -o main ./cmd/server

# Security scan
RUN apk add --no-cache security-scan-tool
RUN security-scan-tool /app/main

# Final stage
FROM alpine:latest

# Install security packages
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    gosu \
    dumb-init \
    && rm -rf /var/cache/apk/*

# Create application user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy the binary
COPY --from=builder /app/main .

# Change ownership
RUN chown -R appuser:appgroup /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appgroup /app/logs /app/data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Run the application
CMD ["./main"]
```

### Security Configuration

```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.secure
      args:
        - BUILDKIT_INLINE_CACHE=1
    image: myapp:latest
    container_name: myapp
    restart: unless-stopped
    
    # Security settings
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    
    # Network settings
    networks:
      - app-network
    
    # Environment variables
    environment:
      - GIN_MODE=release
      - LOG_LEVEL=info
      - TZ=UTC
    
    # Health check
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
    # Volumes
    volumes:
      - app-logs:/app/logs
      - app-data:/app/data
    
    # Ports
    ports:
      - "8080:8080"

networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  app-logs:
    driver: local
  app-data:
    driver: local
```

## Build Optimization

### BuildKit Configuration

```dockerfile
# syntax=docker/dockerfile:1
# Dockerfile.buildkit
FROM golang:1.21-alpine AS base

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Build stage
FROM base AS builder
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# Test stage
FROM base AS test
COPY . .
RUN go test -v -race -coverprofile=coverage.out ./...

# Security scan stage
FROM builder AS security
RUN apk add --no-cache security-scan-tool
RUN security-scan-tool /app/main

# Final stage
FROM alpine:latest
RUN apk --no-cache add ca-certificates tzdata
WORKDIR /app
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
```

### Build Script

```bash
#!/bin/bash
# build.sh

set -e

# Configuration
IMAGE_NAME="myapp"
TAG="${1:-latest}"
REGISTRY="${REGISTRY:-localhost:5000}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build function
build_image() {
    log_info "Building image: ${IMAGE_NAME}:${TAG}"
    
    # Build with BuildKit
    DOCKER_BUILDKIT=1 docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from ${REGISTRY}/${IMAGE_NAME}:cache \
        --tag ${IMAGE_NAME}:${TAG} \
        --tag ${REGISTRY}/${IMAGE_NAME}:${TAG} \
        --file ${DOCKERFILE} \
        ${BUILD_CONTEXT}
    
    log_info "Image built successfully"
}

# Security scan function
security_scan() {
    log_info "Running security scan"
    
    # Run Trivy security scan
    if command -v trivy &> /dev/null; then
        trivy image --exit-code 1 --severity HIGH,CRITICAL ${IMAGE_NAME}:${TAG}
    else
        log_warn "Trivy not found, skipping security scan"
    fi
}

# Push function
push_image() {
    log_info "Pushing image to registry"
    
    docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
    
    log_info "Image pushed successfully"
}

# Main execution
main() {
    log_info "Starting build process"
    
    # Build image
    build_image
    
    # Security scan
    security_scan
    
    # Push image
    if [ "${PUSH:-false}" = "true" ]; then
        push_image
    fi
    
    log_info "Build process completed"
}

# Run main function
main "$@"
```

## Runtime Configuration

### Container Runtime Settings

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.secure
    image: myapp:latest
    container_name: myapp
    restart: unless-stopped
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '1.0'
          memory: 512M
    
    # Security settings
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    
    # Capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    
    # Read-only root filesystem
    read_only: true
    tmpfs:
      - /tmp
      - /var/run
    
    # Environment variables
    environment:
      - GIN_MODE=release
      - LOG_LEVEL=info
      - TZ=UTC
      - GOMAXPROCS=2
    
    # Health check
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
    # Volumes
    volumes:
      - app-logs:/app/logs
      - app-data:/app/data
    
    # Network
    networks:
      - app-network
    
    # Ports
    ports:
      - "8080:8080"

networks:
  app-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  app-logs:
    driver: local
  app-data:
    driver: local
```

### Container Health Monitoring

```go
// internal/health/container_health.go
package health

import (
    "context"
    "fmt"
    "net/http"
    "os"
    "runtime"
    "time"
)

// ContainerHealth represents container health status
type ContainerHealth struct {
    Status      string                 `json:"status"`
    Timestamp   time.Time             `json:"timestamp"`
    Version     string                `json:"version"`
    Uptime      time.Duration         `json:"uptime"`
    Memory       MemoryStats          `json:"memory"`
    Goroutines  int                   `json:"goroutines"`
    Environment map[string]string     `json:"environment"`
}

// MemoryStats represents memory statistics
type MemoryStats struct {
    Alloc      uint64 `json:"alloc"`
    TotalAlloc uint64 `json:"total_alloc"`
    Sys        uint64 `json:"sys"`
    NumGC      uint32 `json:"num_gc"`
}

// HealthChecker represents a health checker
type HealthChecker struct {
    startTime time.Time
    version   string
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(version string) *HealthChecker {
    return &HealthChecker{
        startTime: time.Now(),
        version:   version,
    }
}

// CheckHealth checks the health of the container
func (hc *HealthChecker) CheckHealth() ContainerHealth {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    return ContainerHealth{
        Status:     "healthy",
        Timestamp:  time.Now(),
        Version:    hc.version,
        Uptime:     time.Since(hc.startTime),
        Memory: MemoryStats{
            Alloc:      m.Alloc,
            TotalAlloc: m.TotalAlloc,
            Sys:        m.Sys,
            NumGC:      m.NumGC,
        },
        Goroutines: runtime.NumGoroutine(),
        Environment: map[string]string{
            "GOOS":   runtime.GOOS,
            "GOARCH": runtime.GOARCH,
            "GOMAXPROCS": fmt.Sprintf("%d", runtime.GOMAXPROCS(0)),
        },
    }
}

// HealthHandler handles health check requests
func (hc *HealthChecker) HealthHandler(w http.ResponseWriter, r *http.Request) {
    health := hc.CheckHealth()
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    
    json.NewEncoder(w).Encode(health)
}

// ReadinessHandler handles readiness check requests
func (hc *HealthChecker) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
    // Check if the application is ready to serve traffic
    if hc.isReady() {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("ready"))
    } else {
        w.WriteHeader(http.StatusServiceUnavailable)
        w.Write([]byte("not ready"))
    }
}

// LivenessHandler handles liveness check requests
func (hc *HealthChecker) LivenessHandler(w http.ResponseWriter, r *http.Request) {
    // Check if the application is alive
    if hc.isAlive() {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("alive"))
    } else {
        w.WriteHeader(http.StatusInternalServerError)
        w.Write([]byte("not alive"))
    }
}

// isReady checks if the application is ready
func (hc *HealthChecker) isReady() bool {
    // Implement readiness checks
    return true
}

// isAlive checks if the application is alive
func (hc *HealthChecker) isAlive() bool {
    // Implement liveness checks
    return true
}
```

## Container Orchestration

### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    image: myapp:dev
    container_name: myapp-dev
    restart: unless-stopped
    
    # Development settings
    environment:
      - GIN_MODE=debug
      - LOG_LEVEL=debug
      - TZ=UTC
    
    # Volumes for development
    volumes:
      - .:/app
      - /app/vendor
      - app-logs:/app/logs
    
    # Ports
    ports:
      - "8080:8080"
      - "2345:2345"  # Delve debugger
    
    # Network
    networks:
      - dev-network
    
    # Health check
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Database for development
  postgres:
    image: postgres:15-alpine
    container_name: postgres-dev
    restart: unless-stopped
    
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=myapp
      - POSTGRES_PASSWORD=myapp
    
    volumes:
      - postgres-data:/var/lib/postgresql/data
    
    ports:
      - "5432:5432"
    
    networks:
      - dev-network

  # Redis for development
  redis:
    image: redis:7-alpine
    container_name: redis-dev
    restart: unless-stopped
    
    volumes:
      - redis-data:/data
    
    ports:
      - "6379:6379"
    
    networks:
      - dev-network

networks:
  dev-network:
    driver: bridge

volumes:
  app-logs:
    driver: local
  postgres-data:
    driver: local
  redis-data:
    driver: local
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8080
        env:
        - name: GIN_MODE
          value: "release"
        - name: LOG_LEVEL
          value: "info"
        - name: TZ
          value: "UTC"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          runAsGroup: 1001
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

## Testing Containers

### Container Tests

```go
// internal/testing/container_test.go
package testing

import (
    "context"
    "testing"
    "time"
    
    "github.com/docker/docker/api/types"
    "github.com/docker/docker/client"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestContainerBuild(t *testing.T) {
    ctx := context.Background()
    
    // Create Docker client
    cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
    require.NoError(t, err)
    defer cli.Close()
    
    // Build image
    buildCtx := createBuildContext()
    resp, err := cli.ImageBuild(ctx, buildCtx, types.ImageBuildOptions{
        Tags: []string{"myapp:test"},
    })
    require.NoError(t, err)
    defer resp.Body.Close()
    
    // Verify image exists
    images, err := cli.ImageList(ctx, types.ImageListOptions{})
    require.NoError(t, err)
    
    var found bool
    for _, image := range images {
        for _, tag := range image.RepoTags {
            if tag == "myapp:test" {
                found = true
                break
            }
        }
    }
    assert.True(t, found, "Image should exist")
}

func TestContainerRun(t *testing.T) {
    ctx := context.Background()
    
    // Create Docker client
    cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
    require.NoError(t, err)
    defer cli.Close()
    
    // Run container
    resp, err := cli.ContainerCreate(ctx, &types.ContainerConfig{
        Image: "myapp:test",
        ExposedPorts: map[string]struct{}{
            "8080/tcp": {},
        },
    }, &types.HostConfig{
        PortBindings: map[string][]types.PortBinding{
            "8080/tcp": {{HostPort: "8080"}},
        },
    }, nil, nil, "myapp-test")
    require.NoError(t, err)
    
    // Start container
    err = cli.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{})
    require.NoError(t, err)
    defer cli.ContainerStop(ctx, resp.ID, nil)
    
    // Wait for container to be ready
    time.Sleep(5 * time.Second)
    
    // Test health check
    health, err := cli.ContainerInspect(ctx, resp.ID)
    require.NoError(t, err)
    assert.Equal(t, "running", health.State.Status)
}

func createBuildContext() io.Reader {
    // Create build context
    return nil
}
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Build image
docker build -t myapp:latest .

# 2. Run container
docker run -d -p 8080:8080 --name myapp myapp:latest

# 3. Health check
docker exec myapp wget --spider http://localhost:8080/health

# 4. View logs
docker logs myapp

# 5. Stop container
docker stop myapp
```

### Essential Patterns

```dockerfile
# Multi-stage build
FROM golang:1.21-alpine AS builder
# ... build steps
FROM alpine:latest
COPY --from=builder /app/main .
CMD ["./main"]
```

```yaml
# Docker Compose
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GIN_MODE=release
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8080/health"]
```

---

*This guide provides the complete machinery for containerizing Go applications efficiently and securely. Each pattern includes implementation examples, security considerations, and real-world usage patterns for enterprise deployment.*
