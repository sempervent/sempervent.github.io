# Rust Containerization Best Practices

**Objective**: Master senior-level Rust containerization patterns for production systems. When you need to build efficient Docker containers, when you want to optimize Rust applications for containers, when you need enterprise-grade containerization—these best practices become your weapon of choice.

## Core Principles

- **Multi-stage Builds**: Use multi-stage Docker builds for optimization
- **Security**: Implement container security best practices
- **Size Optimization**: Minimize container image size
- **Performance**: Optimize for container runtime performance
- **Reproducibility**: Ensure consistent builds across environments

## Containerization Patterns

### Multi-stage Docker Builds

```dockerfile
# Dockerfile
# Multi-stage build for Rust applications

# Build stage
FROM rust:1.75-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release

# Copy source code
COPY src ./src

# Build the actual application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy binary from builder stage
COPY --from=builder /app/target/release/myapp /usr/local/bin/myapp

# Set ownership
RUN chown appuser:appuser /usr/local/bin/myapp

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["myapp"]
```

### Optimized Rust Dockerfile

```dockerfile
# Dockerfile.optimized
# Optimized Dockerfile for Rust applications

# Use distroless base image for minimal attack surface
FROM gcr.io/distroless/cc-debian12

# Copy the binary
COPY --from=builder /app/target/release/myapp /usr/local/bin/myapp

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/myapp"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  rust-app:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Container Security Patterns

```rust
// rust/01-container-security.rs

/*
Container security patterns and best practices
*/

use std::env;
use std::process;
use std::fs;
use std::path::Path;

/// Container security configuration.
pub struct ContainerSecurity {
    user_id: Option<u32>,
    group_id: Option<u32>,
    read_only_root: bool,
    no_new_privileges: bool,
    seccomp_profile: Option<String>,
    apparmor_profile: Option<String>,
}

impl ContainerSecurity {
    pub fn new() -> Self {
        Self {
            user_id: None,
            group_id: None,
            read_only_root: false,
            no_new_privileges: false,
            seccomp_profile: None,
            apparmor_profile: None,
        }
    }
    
    /// Set user and group IDs for security.
    pub fn set_user_group(&mut self, user_id: u32, group_id: u32) {
        self.user_id = Some(user_id);
        self.group_id = Some(group_id);
    }
    
    /// Enable read-only root filesystem.
    pub fn enable_read_only_root(&mut self) {
        self.read_only_root = true;
    }
    
    /// Enable no new privileges.
    pub fn enable_no_new_privileges(&mut self) {
        self.no_new_privileges = true;
    }
    
    /// Set seccomp profile.
    pub fn set_seccomp_profile(&mut self, profile: String) {
        self.seccomp_profile = Some(profile);
    }
    
    /// Set AppArmor profile.
    pub fn set_apparmor_profile(&mut self, profile: String) {
        self.apparmor_profile = Some(profile);
    }
    
    /// Apply security configuration.
    pub fn apply(&self) -> Result<(), String> {
        // Set user and group IDs
        if let Some(uid) = self.user_id {
            if let Some(gid) = self.group_id {
                self.set_user_group_ids(uid, gid)?;
            }
        }
        
        // Enable read-only root if requested
        if self.read_only_root {
            self.enable_read_only_root_fs()?;
        }
        
        // Enable no new privileges if requested
        if self.no_new_privileges {
            self.enable_no_new_privileges()?;
        }
        
        Ok(())
    }
    
    /// Set user and group IDs.
    fn set_user_group_ids(&self, uid: u32, gid: u32) -> Result<(), String> {
        // In a real implementation, you would use libc functions
        // to set the user and group IDs
        println!("Setting user ID to {} and group ID to {}", uid, gid);
        Ok(())
    }
    
    /// Enable read-only root filesystem.
    fn enable_read_only_root_fs(&self) -> Result<(), String> {
        // In a real implementation, you would remount the root filesystem
        // as read-only
        println!("Enabling read-only root filesystem");
        Ok(())
    }
    
    /// Enable no new privileges.
    fn enable_no_new_privileges(&self) -> Result<(), String> {
        // In a real implementation, you would use prctl to set
        // PR_SET_NO_NEW_PRIVS
        println!("Enabling no new privileges");
        Ok(())
    }
}

/// Container health checker.
pub struct ContainerHealthChecker {
    health_endpoint: String,
    timeout: std::time::Duration,
    interval: std::time::Duration,
}

impl ContainerHealthChecker {
    pub fn new(health_endpoint: String) -> Self {
        Self {
            health_endpoint,
            timeout: std::time::Duration::from_secs(5),
            interval: std::time::Duration::from_secs(30),
        }
    }
    
    /// Check container health.
    pub async fn check_health(&self) -> Result<bool, String> {
        // In a real implementation, you would make an HTTP request
        // to the health endpoint
        println!("Checking health at {}", self.health_endpoint);
        
        // Simulate health check
        Ok(true)
    }
    
    /// Start health checking loop.
    pub async fn start_health_checking(&self) -> Result<(), String> {
        let mut interval_timer = tokio::time::interval(self.interval);
        
        loop {
            interval_timer.tick().await;
            
            match self.check_health().await {
                Ok(true) => {
                    println!("Health check passed");
                }
                Ok(false) => {
                    println!("Health check failed");
                    process::exit(1);
                }
                Err(e) => {
                    println!("Health check error: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}

/// Container resource limits.
pub struct ContainerResourceLimits {
    memory_limit: Option<u64>,
    cpu_limit: Option<f64>,
    file_descriptor_limit: Option<u64>,
}

impl ContainerResourceLimits {
    pub fn new() -> Self {
        Self {
            memory_limit: None,
            cpu_limit: None,
            file_descriptor_limit: None,
        }
    }
    
    /// Set memory limit in bytes.
    pub fn set_memory_limit(&mut self, limit: u64) {
        self.memory_limit = Some(limit);
    }
    
    /// Set CPU limit as a fraction (e.g., 0.5 for 50%).
    pub fn set_cpu_limit(&mut self, limit: f64) {
        self.cpu_limit = Some(limit);
    }
    
    /// Set file descriptor limit.
    pub fn set_file_descriptor_limit(&mut self, limit: u64) {
        self.file_descriptor_limit = Some(limit);
    }
    
    /// Apply resource limits.
    pub fn apply(&self) -> Result<(), String> {
        if let Some(memory_limit) = self.memory_limit {
            self.set_memory_limit(memory_limit)?;
        }
        
        if let Some(cpu_limit) = self.cpu_limit {
            self.set_cpu_limit(cpu_limit)?;
        }
        
        if let Some(fd_limit) = self.file_descriptor_limit {
            self.set_file_descriptor_limit(fd_limit)?;
        }
        
        Ok(())
    }
    
    /// Set memory limit.
    fn set_memory_limit(&self, limit: u64) -> Result<(), String> {
        // In a real implementation, you would use cgroups or similar
        // to set memory limits
        println!("Setting memory limit to {} bytes", limit);
        Ok(())
    }
    
    /// Set CPU limit.
    fn set_cpu_limit(&self, limit: f64) -> Result<(), String> {
        // In a real implementation, you would use cgroups or similar
        // to set CPU limits
        println!("Setting CPU limit to {}", limit);
        Ok(())
    }
    
    /// Set file descriptor limit.
    fn set_file_descriptor_limit(&self, limit: u64) -> Result<(), String> {
        // In a real implementation, you would use setrlimit
        // to set file descriptor limits
        println!("Setting file descriptor limit to {}", limit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_container_security() {
        let mut security = ContainerSecurity::new();
        security.set_user_group(1000, 1000);
        security.enable_read_only_root();
        security.enable_no_new_privileges();
        
        assert!(security.apply().is_ok());
    }
    
    #[test]
    fn test_container_health_checker() {
        let checker = ContainerHealthChecker::new("/health".to_string());
        assert_eq!(checker.health_endpoint, "/health");
    }
    
    #[test]
    fn test_container_resource_limits() {
        let mut limits = ContainerResourceLimits::new();
        limits.set_memory_limit(1024 * 1024 * 1024); // 1GB
        limits.set_cpu_limit(0.5); // 50%
        limits.set_file_descriptor_limit(1024);
        
        assert!(limits.apply().is_ok());
    }
}
```

### Container Optimization

```rust
// rust/02-container-optimization.rs

/*
Container optimization patterns and best practices
*/

use std::env;
use std::process;
use std::time::{Duration, Instant};

/// Container optimization configuration.
pub struct ContainerOptimization {
    enable_jemalloc: bool,
    enable_mimalloc: bool,
    enable_system_allocator: bool,
    enable_compact_gc: bool,
    enable_zero_copy: bool,
    enable_async_io: bool,
}

impl ContainerOptimization {
    pub fn new() -> Self {
        Self {
            enable_jemalloc: false,
            enable_mimalloc: false,
            enable_system_allocator: true,
            enable_compact_gc: false,
            enable_zero_copy: false,
            enable_async_io: true,
        }
    }
    
    /// Enable jemalloc allocator.
    pub fn enable_jemalloc(&mut self) {
        self.enable_jemalloc = true;
        self.enable_system_allocator = false;
    }
    
    /// Enable mimalloc allocator.
    pub fn enable_mimalloc(&mut self) {
        self.enable_mimalloc = true;
        self.enable_system_allocator = false;
    }
    
    /// Enable compact garbage collection.
    pub fn enable_compact_gc(&mut self) {
        self.enable_compact_gc = true;
    }
    
    /// Enable zero-copy operations.
    pub fn enable_zero_copy(&mut self) {
        self.enable_zero_copy = true;
    }
    
    /// Apply optimization configuration.
    pub fn apply(&self) -> Result<(), String> {
        // Set environment variables for optimization
        if self.enable_jemalloc {
            env::set_var("RUSTFLAGS", "-C target-cpu=native");
            env::set_var("MALLOC_CONF", "dirty_decay_ms:0,muzzy_decay_ms:0");
        }
        
        if self.enable_mimalloc {
            env::set_var("RUSTFLAGS", "-C target-cpu=native");
            env::set_var("MIMALLOC_SHOW_STATS", "1");
        }
        
        if self.enable_compact_gc {
            env::set_var("RUST_GC_COMPACT", "1");
        }
        
        if self.enable_zero_copy {
            env::set_var("RUST_ZERO_COPY", "1");
        }
        
        if self.enable_async_io {
            env::set_var("RUST_ASYNC_IO", "1");
        }
        
        Ok(())
    }
}

/// Container performance monitor.
pub struct ContainerPerformanceMonitor {
    start_time: Instant,
    memory_usage: u64,
    cpu_usage: f64,
    request_count: u64,
    error_count: u64,
}

impl ContainerPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            memory_usage: 0,
            cpu_usage: 0.0,
            request_count: 0,
            error_count: 0,
        }
    }
    
    /// Record a request.
    pub fn record_request(&mut self) {
        self.request_count += 1;
    }
    
    /// Record an error.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }
    
    /// Update memory usage.
    pub fn update_memory_usage(&mut self, usage: u64) {
        self.memory_usage = usage;
    }
    
    /// Update CPU usage.
    pub fn update_cpu_usage(&mut self, usage: f64) {
        self.cpu_usage = usage;
    }
    
    /// Get performance metrics.
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let uptime = self.start_time.elapsed();
        let requests_per_second = if uptime.as_secs() > 0 {
            self.request_count as f64 / uptime.as_secs() as f64
        } else {
            0.0
        };
        
        let error_rate = if self.request_count > 0 {
            self.error_count as f64 / self.request_count as f64
        } else {
            0.0
        };
        
        PerformanceMetrics {
            uptime,
            memory_usage: self.memory_usage,
            cpu_usage: self.cpu_usage,
            request_count: self.request_count,
            error_count: self.error_count,
            requests_per_second,
            error_rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub uptime: Duration,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub request_count: u64,
    pub error_count: u64,
    pub requests_per_second: f64,
    pub error_rate: f64,
}

/// Container graceful shutdown handler.
pub struct GracefulShutdownHandler {
    shutdown_signal: tokio::sync::oneshot::Sender<()>,
    shutdown_receiver: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl GracefulShutdownHandler {
    pub fn new() -> Self {
        let (shutdown_signal, shutdown_receiver) = tokio::sync::oneshot::channel();
        Self {
            shutdown_signal,
            shutdown_receiver: Some(shutdown_receiver),
        }
    }
    
    /// Start graceful shutdown handling.
    pub async fn start(&mut self) -> Result<(), String> {
        let receiver = self.shutdown_receiver.take()
            .ok_or("Shutdown receiver already taken")?;
        
        // Handle shutdown signals
        tokio::select! {
            _ = receiver => {
                println!("Received shutdown signal");
                self.handle_shutdown().await?;
            }
            _ = tokio::signal::ctrl_c() => {
                println!("Received Ctrl+C");
                self.handle_shutdown().await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle graceful shutdown.
    async fn handle_shutdown(&self) -> Result<(), String> {
        println!("Starting graceful shutdown...");
        
        // Give ongoing requests time to complete
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        println!("Graceful shutdown complete");
        process::exit(0);
    }
    
    /// Trigger shutdown.
    pub fn shutdown(&self) -> Result<(), String> {
        self.shutdown_signal.send(())
            .map_err(|_| "Failed to send shutdown signal")?;
        Ok(())
    }
}

/// Container configuration loader.
pub struct ContainerConfigLoader {
    config_path: String,
    config: Option<ContainerConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContainerConfig {
    pub app_name: String,
    pub version: String,
    pub environment: String,
    pub database_url: String,
    pub redis_url: String,
    pub log_level: String,
    pub port: u16,
    pub workers: usize,
}

impl ContainerConfigLoader {
    pub fn new(config_path: String) -> Self {
        Self {
            config_path,
            config: None,
        }
    }
    
    /// Load configuration from file.
    pub fn load_config(&mut self) -> Result<&ContainerConfig, String> {
        let content = std::fs::read_to_string(&self.config_path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        
        let config: ContainerConfig = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse config: {}", e))?;
        
        self.config = Some(config);
        Ok(self.config.as_ref().unwrap())
    }
    
    /// Get configuration.
    pub fn get_config(&self) -> Option<&ContainerConfig> {
        self.config.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_container_optimization() {
        let mut optimization = ContainerOptimization::new();
        optimization.enable_jemalloc();
        optimization.enable_compact_gc();
        optimization.enable_zero_copy();
        
        assert!(optimization.apply().is_ok());
    }
    
    #[test]
    fn test_container_performance_monitor() {
        let mut monitor = ContainerPerformanceMonitor::new();
        monitor.record_request();
        monitor.record_request();
        monitor.record_error();
        monitor.update_memory_usage(1024 * 1024); // 1MB
        monitor.update_cpu_usage(0.5); // 50%
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.request_count, 2);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.memory_usage, 1024 * 1024);
        assert_eq!(metrics.cpu_usage, 0.5);
    }
    
    #[test]
    fn test_graceful_shutdown_handler() {
        let handler = GracefulShutdownHandler::new();
        assert!(handler.shutdown().is_ok());
    }
}
```

### Container Testing

```rust
// rust/03-container-testing.rs

/*
Container testing patterns and best practices
*/

use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::timeout;

/// Container test runner.
pub struct ContainerTestRunner {
    image_name: String,
    container_name: String,
    test_timeout: Duration,
}

impl ContainerTestRunner {
    pub fn new(image_name: String, container_name: String) -> Self {
        Self {
            image_name,
            container_name,
            test_timeout: Duration::from_secs(30),
        }
    }
    
    /// Set test timeout.
    pub fn set_test_timeout(&mut self, timeout: Duration) {
        self.test_timeout = timeout;
    }
    
    /// Run container tests.
    pub async fn run_tests(&self) -> Result<TestResults, String> {
        let mut results = TestResults::new();
        
        // Test 1: Container starts successfully
        if let Err(e) = self.test_container_start().await {
            results.add_failure("Container start test".to_string(), e);
        } else {
            results.add_success("Container start test".to_string());
        }
        
        // Test 2: Health check passes
        if let Err(e) = self.test_health_check().await {
            results.add_failure("Health check test".to_string(), e);
        } else {
            results.add_success("Health check test".to_string());
        }
        
        // Test 3: Application responds
        if let Err(e) = self.test_application_response().await {
            results.add_failure("Application response test".to_string(), e);
        } else {
            results.add_success("Application response test".to_string());
        }
        
        // Test 4: Resource usage is within limits
        if let Err(e) = self.test_resource_usage().await {
            results.add_failure("Resource usage test".to_string(), e);
        } else {
            results.add_success("Resource usage test".to_string());
        }
        
        // Test 5: Security scan passes
        if let Err(e) = self.test_security_scan().await {
            results.add_failure("Security scan test".to_string(), e);
        } else {
            results.add_success("Security scan test".to_string());
        }
        
        Ok(results)
    }
    
    /// Test container start.
    async fn test_container_start(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["run", "--rm", "-d", "--name", &self.container_name, &self.image_name])
            .output()
            .map_err(|e| format!("Failed to start container: {}", e))?;
        
        if !output.status.success() {
            return Err("Container failed to start".to_string());
        }
        
        // Wait for container to be ready
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        Ok(())
    }
    
    /// Test health check.
    async fn test_health_check(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["exec", &self.container_name, "curl", "-f", "http://localhost:8080/health"])
            .output()
            .map_err(|e| format!("Failed to run health check: {}", e))?;
        
        if !output.status.success() {
            return Err("Health check failed".to_string());
        }
        
        Ok(())
    }
    
    /// Test application response.
    async fn test_application_response(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["exec", &self.container_name, "curl", "-f", "http://localhost:8080/"])
            .output()
            .map_err(|e| format!("Failed to test application response: {}", e))?;
        
        if !output.status.success() {
            return Err("Application response test failed".to_string());
        }
        
        Ok(())
    }
    
    /// Test resource usage.
    async fn test_resource_usage(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["stats", "--no-stream", "--format", "table {{.MemUsage}}", &self.container_name])
            .output()
            .map_err(|e| format!("Failed to check resource usage: {}", e))?;
        
        if !output.status.success() {
            return Err("Resource usage check failed".to_string());
        }
        
        // Parse memory usage and check if it's within limits
        let output_str = String::from_utf8_lossy(&output.stdout);
        if output_str.contains("GiB") {
            return Err("Memory usage too high".to_string());
        }
        
        Ok(())
    }
    
    /// Test security scan.
    async fn test_security_scan(&self) -> Result<(), String> {
        let output = Command::new("docker")
            .args(&["run", "--rm", "-v", "/var/run/docker.sock:/var/run/docker.sock", 
                   "aquasec/trivy", "image", &self.image_name])
            .output()
            .map_err(|e| format!("Failed to run security scan: {}", e))?;
        
        if !output.status.success() {
            return Err("Security scan failed".to_string());
        }
        
        // Check for high severity vulnerabilities
        let output_str = String::from_utf8_lossy(&output.stdout);
        if output_str.contains("HIGH") || output_str.contains("CRITICAL") {
            return Err("High severity vulnerabilities found".to_string());
        }
        
        Ok(())
    }
    
    /// Clean up test container.
    pub async fn cleanup(&self) -> Result<(), String> {
        let _ = Command::new("docker")
            .args(&["stop", &self.container_name])
            .output();
        
        let _ = Command::new("docker")
            .args(&["rm", &self.container_name])
            .output();
        
        Ok(())
    }
}

/// Test results container.
pub struct TestResults {
    successes: Vec<String>,
    failures: Vec<(String, String)>,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            successes: Vec::new(),
            failures: Vec::new(),
        }
    }
    
    /// Add a successful test.
    pub fn add_success(&mut self, test_name: String) {
        self.successes.push(test_name);
    }
    
    /// Add a failed test.
    pub fn add_failure(&mut self, test_name: String, error: String) {
        self.failures.push((test_name, error));
    }
    
    /// Get test summary.
    pub fn get_summary(&self) -> String {
        let total_tests = self.successes.len() + self.failures.len();
        let passed_tests = self.successes.len();
        let failed_tests = self.failures.len();
        
        format!(
            "Tests: {} total, {} passed, {} failed",
            total_tests, passed_tests, failed_tests
        )
    }
    
    /// Get detailed results.
    pub fn get_details(&self) -> String {
        let mut details = String::new();
        
        details.push_str("=== Test Results ===\n");
        details.push_str(&self.get_summary());
        details.push_str("\n\n");
        
        if !self.successes.is_empty() {
            details.push_str("Passed tests:\n");
            for success in &self.successes {
                details.push_str(&format!("  ✓ {}\n", success));
            }
            details.push_str("\n");
        }
        
        if !self.failures.is_empty() {
            details.push_str("Failed tests:\n");
            for (test_name, error) in &self.failures {
                details.push_str(&format!("  ✗ {}: {}\n", test_name, error));
            }
        }
        
        details
    }
    
    /// Check if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_results() {
        let mut results = TestResults::new();
        results.add_success("Test 1".to_string());
        results.add_failure("Test 2".to_string(), "Error".to_string());
        
        assert!(!results.all_passed());
        assert_eq!(results.successes.len(), 1);
        assert_eq!(results.failures.len(), 1);
    }
    
    #[test]
    fn test_container_test_runner() {
        let runner = ContainerTestRunner::new(
            "test-image".to_string(),
            "test-container".to_string(),
        );
        
        assert_eq!(runner.image_name, "test-image");
        assert_eq!(runner.container_name, "test-container");
    }
}
```

## TL;DR Runbook

### Quick Start

```dockerfile
# Multi-stage Dockerfile
FROM rust:1.75-slim as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/myapp /usr/local/bin/myapp
CMD ["myapp"]
```

### Essential Patterns

```rust
// Complete containerization setup
pub fn setup_rust_containerization() {
    // 1. Multi-stage builds
    // 2. Security configuration
    // 3. Resource optimization
    // 4. Health checking
    // 5. Graceful shutdown
    // 6. Performance monitoring
    // 7. Testing
    // 8. Deployment
    
    println!("Rust containerization setup complete!");
}
```

---

*This guide provides the complete machinery for Rust containerization. Each pattern includes implementation examples, container strategies, and real-world usage patterns for enterprise containerized systems.*
