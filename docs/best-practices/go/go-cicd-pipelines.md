# Go CI/CD Pipelines Best Practices

**Objective**: Master senior-level Go CI/CD pipeline patterns for production systems. When you need to build robust, automated deployment pipelines, when you want to ensure code quality and security, when you need enterprise-grade CI/CD patterns—these best practices become your weapon of choice.

## Core Principles

- **Fast Feedback**: Provide quick feedback on code changes
- **Quality Gates**: Enforce quality standards at every stage
- **Security First**: Integrate security scanning and compliance
- **Reproducible Builds**: Ensure consistent, reproducible builds
- **Automated Deployment**: Deploy with confidence and rollback capability

## GitHub Actions Workflows

### Basic CI Pipeline

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
    
    strategy:
      matrix:
        go-version: ['1.20', '1.21', '1.22']
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ matrix.go-version }}
    
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
    
    - name: Run benchmarks
      run: go test -bench=. -benchmem ./...
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true
```

### Advanced CI Pipeline

```yaml
# .github/workflows/advanced-ci.yml
name: Advanced CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly security scan

env:
  GO_VERSION: '1.21'
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Quality checks
  quality:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ env.GO_VERSION }}
    
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
    
    - name: Run linters
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest
        args: --timeout=5m
    
    - name: Run security scan
      uses: securecodewarrior/github-action-gosec@master
      with:
        args: './...'
    
    - name: Run tests
      run: go test -v -race -coverprofile=coverage.out ./...
    
    - name: Run benchmarks
      run: go test -bench=. -benchmem ./...
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  # Build and test
  build:
    runs-on: ubuntu-latest
    needs: quality
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ env.GO_VERSION }}
    
    - name: Cache Go modules
      uses: actions/cache@v3
      with:
        path: ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-
    
    - name: Download dependencies
      run: go mod download
    
    - name: Build application
      run: |
        go build -o bin/server ./cmd/server
        go build -o bin/worker ./cmd/worker
    
    - name: Test build artifacts
      run: |
        ./bin/server --version
        ./bin/worker --version
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: bin/

  # Security scanning
  security:
    runs-on: ubuntu-latest
    needs: quality
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run OWASP Dependency Check
      uses: dependency-check/Dependency-Check_Action@main
      with:
        project: 'myapp'
        path: '.'
        format: 'SARIF'
        out: 'dependency-check-report.sarif'
    
    - name: Upload dependency check results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'dependency-check-report.sarif'

  # Docker build and test
  docker:
    runs-on: ubuntu-latest
    needs: [quality, build]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.DOCKER_REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### CD Pipeline

```yaml
# .github/workflows/cd.yml
name: CD

on:
  push:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Deploy to staging
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/myapp myapp=${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        kubectl rollout status deployment/myapp
    
    - name: Run smoke tests
      run: |
        kubectl get pods -l app=myapp
        kubectl get services -l app=myapp

  # Deploy to production
  deploy-production:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    needs: deploy-staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/myapp myapp=${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        kubectl rollout status deployment/myapp
    
    - name: Run smoke tests
      run: |
        kubectl get pods -l app=myapp
        kubectl get services -l app=myapp
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## GitLab CI/CD

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - quality
  - build
  - test
  - security
  - deploy

variables:
  GO_VERSION: "1.21"
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

# Quality checks
quality:
  stage: quality
  image: golang:${GO_VERSION}-alpine
  before_script:
    - apk add --no-cache git ca-certificates tzdata
    - go mod download
  script:
    - go mod verify
    - go vet ./...
    - go fmt ./...
    - golangci-lint run
    - gosec ./...
  artifacts:
    reports:
      junit: junit.xml
    paths:
      - coverage.out
    expire_in: 1 hour

# Build
build:
  stage: build
  image: golang:${GO_VERSION}-alpine
  before_script:
    - apk add --no-cache git ca-certificates tzdata
    - go mod download
  script:
    - go build -o bin/server ./cmd/server
    - go build -o bin/worker ./cmd/worker
  artifacts:
    paths:
      - bin/
    expire_in: 1 hour

# Test
test:
  stage: test
  image: golang:${GO_VERSION}-alpine
  before_script:
    - apk add --no-cache git ca-certificates tzdata
    - go mod download
  script:
    - go test -v -race -coverprofile=coverage.out ./...
    - go test -bench=. -benchmem ./...
  coverage: '/coverage: \d+\.\d+%/'
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - coverage.out
    expire_in: 1 hour

# Security scan
security:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy fs --format json --output trivy-results.json .
  artifacts:
    reports:
      sast: trivy-results.json
    expire_in: 1 hour

# Docker build
docker-build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - develop

# Deploy to staging
deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - kubectl set image deployment/myapp myapp=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/myapp
  only:
    - develop

# Deploy to production
deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://example.com
  script:
    - kubectl set image deployment/myapp myapp=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/myapp
  when: manual
  only:
    - main
```

## Jenkins Pipeline

### Jenkinsfile

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        GO_VERSION = '1.21'
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_NAME = 'myapp'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Quality') {
            parallel {
                stage('Lint') {
                    steps {
                        sh 'golangci-lint run'
                    }
                }
                stage('Security') {
                    steps {
                        sh 'gosec ./...'
                    }
                }
                stage('Dependencies') {
                    steps {
                        sh 'go mod verify'
                        sh 'go mod tidy'
                    }
                }
            }
        }
        
        stage('Test') {
            steps {
                sh 'go test -v -race -coverprofile=coverage.out ./...'
                sh 'go test -bench=. -benchmem ./...'
            }
            post {
                always {
                    publishCoverage adapters: [
                        goCoberturaAdapter('coverage.out')
                    ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }
        
        stage('Build') {
            steps {
                sh 'go build -o bin/server ./cmd/server'
                sh 'go build -o bin/worker ./cmd/worker'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'bin/*', fingerprint: true
                }
            }
        }
        
        stage('Docker Build') {
            when {
                branch 'main'
            }
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push("latest")
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    sh """
                        kubectl set image deployment/myapp myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                        kubectl rollout status deployment/myapp
                    """
                }
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                script {
                    sh """
                        kubectl set image deployment/myapp myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                        kubectl rollout status deployment/myapp
                    """
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            slackSend channel: '#deployments', color: 'good', message: "Build ${BUILD_NUMBER} succeeded"
        }
        failure {
            slackSend channel: '#deployments', color: 'danger', message: "Build ${BUILD_NUMBER} failed"
        }
    }
}
```

## Build Scripts

### Makefile

```makefile
# Makefile
.PHONY: help build test lint security docker clean

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

# Docker
docker: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t myapp:latest .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8080:8080 myapp:latest

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
	godoc -http=:6060
```

### Build Script

```bash
#!/bin/bash
# build.sh

set -e

# Configuration
APP_NAME="myapp"
VERSION="${1:-latest}"
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
build_app() {
    log_info "Building application: ${APP_NAME}"
    
    # Build for multiple platforms
    GOOS=linux GOARCH=amd64 go build -o bin/server-linux ./cmd/server
    GOOS=linux GOARCH=arm64 go build -o bin/server-arm64 ./cmd/server
    GOOS=windows GOARCH=amd64 go build -o bin/server-windows.exe ./cmd/server
    GOOS=darwin GOARCH=amd64 go build -o bin/server-darwin ./cmd/server
    
    log_info "Application built successfully"
}

# Test function
run_tests() {
    log_info "Running tests"
    
    go test -v -race -coverprofile=coverage.out ./...
    
    log_info "Tests completed"
}

# Lint function
run_lint() {
    log_info "Running linters"
    
    golangci-lint run
    
    log_info "Linting completed"
}

# Security scan function
run_security() {
    log_info "Running security scan"
    
    gosec ./...
    
    log_info "Security scan completed"
}

# Docker build function
build_docker() {
    log_info "Building Docker image: ${APP_NAME}:${VERSION}"
    
    docker build -t ${APP_NAME}:${VERSION} -f ${DOCKERFILE} ${BUILD_CONTEXT}
    
    log_info "Docker image built successfully"
}

# Push function
push_docker() {
    log_info "Pushing Docker image to registry"
    
    docker tag ${APP_NAME}:${VERSION} ${REGISTRY}/${APP_NAME}:${VERSION}
    docker push ${REGISTRY}/${APP_NAME}:${VERSION}
    
    log_info "Docker image pushed successfully"
}

# Main execution
main() {
    log_info "Starting build process"
    
    # Run quality checks
    run_tests
    run_lint
    run_security
    
    # Build application
    build_app
    
    # Build Docker image
    build_docker
    
    # Push image
    if [ "${PUSH:-false}" = "true" ]; then
        push_docker
    fi
    
    log_info "Build process completed"
}

# Run main function
main "$@"
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Run tests
make test

# 2. Run linters
make lint

# 3. Run security checks
make security

# 4. Build application
make build

# 5. Build Docker image
make docker
```

### Essential Patterns

```yaml
# GitHub Actions
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-go@v4
    - run: go test ./...
```

```groovy
// Jenkins
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'go test ./...'
            }
        }
    }
}
```

---

*This guide provides the complete machinery for building robust CI/CD pipelines for Go applications. Each pattern includes implementation examples, security considerations, and real-world usage patterns for enterprise deployment.*
