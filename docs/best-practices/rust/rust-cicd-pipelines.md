# Rust CI/CD Pipelines Best Practices

**Objective**: Master senior-level Rust CI/CD pipeline patterns for production systems. When you need to build automated testing and deployment pipelines, when you want to ensure code quality and reliability, when you need enterprise-grade CI/CD—these best practices become your weapon of choice.

## Core Principles

- **Automation**: Automate all testing, building, and deployment processes
- **Quality Gates**: Implement quality checks at every stage
- **Security**: Integrate security scanning and vulnerability detection
- **Performance**: Optimize pipeline performance and resource usage
- **Reliability**: Ensure consistent and reliable deployments

## CI/CD Pipeline Patterns

### GitHub Actions Workflow

```yaml
# .github/workflows/rust-ci.yml
name: Rust CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rust-version: [stable, beta, nightly]
        target: [x86_64-unknown-linux-gnu, x86_64-unknown-linux-musl]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: ${{ matrix.rust-version }}
        targets: ${{ matrix.target }}
        
    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Cache cargo index
      uses: actions/cache@v3
      with:
        path: ~/.cargo/git
        key: ${{ runner.os }}-cargo-index-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Cache cargo build
      uses: actions/cache@v3
      with:
        path: target
        key: ${{ runner.os }}-cargo-build-${{ matrix.target }}-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libssl-dev pkg-config
        
    - name: Run tests
      run: |
        cargo test --verbose --target ${{ matrix.target }}
        
    - name: Run clippy
      run: |
        cargo clippy --verbose --target ${{ matrix.target }} -- -D warnings
        
    - name: Run rustfmt
      run: |
        cargo fmt --all -- --check
        
    - name: Run cargo audit
      run: |
        cargo install cargo-audit
        cargo audit
        
    - name: Run cargo deny
      run: |
        cargo install cargo-deny
        cargo deny check
        
    - name: Run cargo outdated
      run: |
        cargo install cargo-outdated
        cargo outdated --exit-code 1

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: test
    
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

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        toolchain: stable
        
    - name: Cache cargo build
      uses: actions/cache@v3
      with:
        path: target
        key: ${{ runner.os }}-cargo-build-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Build release
      run: |
        cargo build --release
        
    - name: Build for multiple targets
      run: |
        rustup target add x86_64-unknown-linux-musl
        cargo build --release --target x86_64-unknown-linux-musl
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: rust-binaries
        path: target/release/

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/rust-app:latest
          ${{ secrets.DOCKER_USERNAME }}/rust-app:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: [build, docker]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add your deployment commands here
        
    - name: Run smoke tests
      run: |
        echo "Running smoke tests"
        # Add your smoke test commands here
        
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add your production deployment commands here
```

### GitLab CI/CD Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - security
  - build
  - deploy

variables:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

test:
  stage: test
  image: rust:1.75-slim
  script:
    - apt-get update && apt-get install -y libssl-dev pkg-config
    - cargo test --verbose
    - cargo clippy -- -D warnings
    - cargo fmt --all -- --check
    - cargo audit
    - cargo deny check
    - cargo outdated --exit-code 1
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - target/
      - .cargo/

security:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy fs --format sarif --output trivy-results.sarif .
  artifacts:
    reports:
      sarif: trivy-results.sarif

build:
  stage: build
  image: rust:1.75-slim
  script:
    - cargo build --release
    - cargo build --release --target x86_64-unknown-linux-musl
  artifacts:
    paths:
      - target/release/
    expire_in: 1 week

docker:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to production"
    # Add your deployment commands here
  only:
    - main
  when: manual
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        CARGO_TERM_COLOR = 'always'
        RUST_BACKTRACE = '1'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'cargo test --verbose'
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh 'cargo test --verbose --test integration'
                    }
                }
                
                stage('Clippy') {
                    steps {
                        sh 'cargo clippy -- -D warnings'
                    }
                }
                
                stage('Rustfmt') {
                    steps {
                        sh 'cargo fmt --all -- --check'
                    }
                }
            }
        }
        
        stage('Security') {
            steps {
                sh 'cargo audit'
                sh 'cargo deny check'
                sh 'cargo outdated --exit-code 1'
            }
        }
        
        stage('Build') {
            steps {
                sh 'cargo build --release'
                sh 'cargo build --release --target x86_64-unknown-linux-musl'
            }
        }
        
        stage('Docker') {
            steps {
                script {
                    def image = docker.build("rust-app:${env.BUILD_NUMBER}")
                    docker.withRegistry('https://registry.example.com', 'docker-credentials') {
                        image.push("${env.BUILD_NUMBER}")
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'echo "Deploying to production"'
                // Add your deployment commands here
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'target/release/*', fingerprint: true
            publishTestResults testResultsPattern: 'target/test-results/*.xml'
        }
        
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check the console output for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### CI/CD Pipeline Configuration

```rust
// rust/01-cicd-configuration.rs

/*
CI/CD pipeline configuration and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::process::Command;

/// CI/CD pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub version: String,
    pub stages: Vec<PipelineStage>,
    pub variables: HashMap<String, String>,
    pub artifacts: Vec<String>,
    pub notifications: Vec<NotificationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub commands: Vec<String>,
    pub dependencies: Vec<String>,
    pub timeout: Option<u64>,
    pub retry_count: Option<u32>,
    pub parallel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub event: String,
    pub webhook_url: String,
    pub template: String,
}

impl PipelineConfig {
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            stages: Vec::new(),
            variables: HashMap::new(),
            artifacts: Vec::new(),
            notifications: Vec::new(),
        }
    }
    
    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }
    
    /// Set a pipeline variable.
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }
    
    /// Add an artifact to the pipeline.
    pub fn add_artifact(&mut self, artifact: String) {
        self.artifacts.push(artifact);
    }
    
    /// Add a notification configuration.
    pub fn add_notification(&mut self, notification: NotificationConfig) {
        self.notifications.push(notification);
    }
}

/// Pipeline executor.
pub struct PipelineExecutor {
    config: PipelineConfig,
    results: HashMap<String, StageResult>,
}

#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub status: StageStatus,
    pub duration: std::time::Duration,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageStatus {
    Success,
    Failure,
    Skipped,
    Running,
}

impl PipelineExecutor {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }
    
    /// Execute the pipeline.
    pub async fn execute(&mut self) -> Result<(), String> {
        println!("Starting pipeline: {}", self.config.name);
        
        for stage in &self.config.stages {
            println!("Executing stage: {}", stage.name);
            
            let start_time = std::time::Instant::now();
            let result = self.execute_stage(stage).await;
            let duration = start_time.elapsed();
            
            let stage_result = StageResult {
                stage_name: stage.name.clone(),
                status: if result.is_ok() { StageStatus::Success } else { StageStatus::Failure },
                duration,
                output: result.as_ref().unwrap_or(&String::new()).clone(),
                error: result.err(),
            };
            
            self.results.insert(stage.name.clone(), stage_result);
            
            if result.is_err() {
                println!("Stage {} failed: {:?}", stage.name, result.err());
                return Err(format!("Stage {} failed", stage.name));
            }
        }
        
        println!("Pipeline completed successfully");
        Ok(())
    }
    
    /// Execute a single stage.
    async fn execute_stage(&self, stage: &PipelineStage) -> Result<String, String> {
        let mut output = String::new();
        
        for command in &stage.commands {
            let result = self.execute_command(command).await?;
            output.push_str(&result);
            output.push('\n');
        }
        
        Ok(output)
    }
    
    /// Execute a single command.
    async fn execute_command(&self, command: &str) -> Result<String, String> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| format!("Failed to execute command: {}", e))?;
        
        if !output.status.success() {
            return Err(format!("Command failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
    
    /// Get stage results.
    pub fn get_results(&self) -> &HashMap<String, StageResult> {
        &self.results
    }
    
    /// Get pipeline summary.
    pub fn get_summary(&self) -> String {
        let total_stages = self.results.len();
        let successful_stages = self.results.values()
            .filter(|r| r.status == StageStatus::Success)
            .count();
        let failed_stages = self.results.values()
            .filter(|r| r.status == StageStatus::Failure)
            .count();
        
        format!(
            "Pipeline Summary: {} total stages, {} successful, {} failed",
            total_stages, successful_stages, failed_stages
        )
    }
}

/// Pipeline builder for creating complex pipelines.
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new(name: String, version: String) -> Self {
        Self {
            config: PipelineConfig::new(name, version),
        }
    }
    
    /// Add a test stage.
    pub fn add_test_stage(mut self) -> Self {
        let stage = PipelineStage {
            name: "test".to_string(),
            commands: vec![
                "cargo test --verbose".to_string(),
                "cargo clippy -- -D warnings".to_string(),
                "cargo fmt --all -- --check".to_string(),
            ],
            dependencies: Vec::new(),
            timeout: Some(300),
            retry_count: Some(2),
            parallel: false,
        };
        self.config.add_stage(stage);
        self
    }
    
    /// Add a security stage.
    pub fn add_security_stage(mut self) -> Self {
        let stage = PipelineStage {
            name: "security".to_string(),
            commands: vec![
                "cargo audit".to_string(),
                "cargo deny check".to_string(),
                "cargo outdated --exit-code 1".to_string(),
            ],
            dependencies: vec!["test".to_string()],
            timeout: Some(180),
            retry_count: Some(1),
            parallel: false,
        };
        self.config.add_stage(stage);
        self
    }
    
    /// Add a build stage.
    pub fn add_build_stage(mut self) -> Self {
        let stage = PipelineStage {
            name: "build".to_string(),
            commands: vec![
                "cargo build --release".to_string(),
                "cargo build --release --target x86_64-unknown-linux-musl".to_string(),
            ],
            dependencies: vec!["test".to_string(), "security".to_string()],
            timeout: Some(600),
            retry_count: Some(1),
            parallel: false,
        };
        self.config.add_stage(stage);
        self
    }
    
    /// Add a deploy stage.
    pub fn add_deploy_stage(mut self) -> Self {
        let stage = PipelineStage {
            name: "deploy".to_string(),
            commands: vec![
                "echo 'Deploying to production'".to_string(),
                "docker build -t rust-app:latest .".to_string(),
                "docker push rust-app:latest".to_string(),
            ],
            dependencies: vec!["build".to_string()],
            timeout: Some(900),
            retry_count: Some(1),
            parallel: false,
        };
        self.config.add_stage(stage);
        self
    }
    
    /// Set pipeline variables.
    pub fn set_variables(mut self, variables: HashMap<String, String>) -> Self {
        for (key, value) in variables {
            self.config.set_variable(key, value);
        }
        self
    }
    
    /// Add artifacts.
    pub fn add_artifacts(mut self, artifacts: Vec<String>) -> Self {
        for artifact in artifacts {
            self.config.add_artifact(artifact);
        }
        self
    }
    
    /// Build the pipeline configuration.
    pub fn build(self) -> PipelineConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_config() {
        let mut config = PipelineConfig::new("test-pipeline".to_string(), "1.0.0".to_string());
        
        let stage = PipelineStage {
            name: "test".to_string(),
            commands: vec!["cargo test".to_string()],
            dependencies: Vec::new(),
            timeout: Some(300),
            retry_count: Some(2),
            parallel: false,
        };
        
        config.add_stage(stage);
        assert_eq!(config.stages.len(), 1);
    }
    
    #[test]
    fn test_pipeline_builder() {
        let config = PipelineBuilder::new("test-pipeline".to_string(), "1.0.0".to_string())
            .add_test_stage()
            .add_security_stage()
            .add_build_stage()
            .add_deploy_stage()
            .build();
        
        assert_eq!(config.stages.len(), 4);
        assert_eq!(config.name, "test-pipeline");
    }
}
```

### Quality Gates

```rust
// rust/02-quality-gates.rs

/*
Quality gates and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Quality gate configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub name: String,
    pub metrics: Vec<QualityMetric>,
    pub thresholds: HashMap<String, f64>,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetric {
    pub name: String,
    pub value: f64,
    pub threshold: f64,
    pub operator: ComparisonOperator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl QualityGate {
    pub fn new(name: String) -> Self {
        Self {
            name,
            metrics: Vec::new(),
            thresholds: HashMap::new(),
            required: true,
        }
    }
    
    /// Add a quality metric.
    pub fn add_metric(&mut self, metric: QualityMetric) {
        self.metrics.push(metric);
    }
    
    /// Set a threshold for a metric.
    pub fn set_threshold(&mut self, metric_name: String, threshold: f64) {
        self.thresholds.insert(metric_name, threshold);
    }
    
    /// Check if the quality gate passes.
    pub fn check(&self) -> QualityGateResult {
        let mut results = Vec::new();
        let mut passed = true;
        
        for metric in &self.metrics {
            let result = self.check_metric(metric);
            results.push(result.clone());
            
            if !result.passed {
                passed = false;
            }
        }
        
        QualityGateResult {
            gate_name: self.name.clone(),
            passed,
            results,
        }
    }
    
    /// Check a single metric.
    fn check_metric(&self, metric: &QualityMetric) -> MetricResult {
        let passed = match metric.operator {
            ComparisonOperator::LessThan => metric.value < metric.threshold,
            ComparisonOperator::LessThanOrEqual => metric.value <= metric.threshold,
            ComparisonOperator::GreaterThan => metric.value > metric.threshold,
            ComparisonOperator::GreaterThanOrEqual => metric.value >= metric.threshold,
            ComparisonOperator::Equal => metric.value == metric.threshold,
            ComparisonOperator::NotEqual => metric.value != metric.threshold,
        };
        
        MetricResult {
            metric_name: metric.name.clone(),
            value: metric.value,
            threshold: metric.threshold,
            passed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityGateResult {
    pub gate_name: String,
    pub passed: bool,
    pub results: Vec<MetricResult>,
}

#[derive(Debug, Clone)]
pub struct MetricResult {
    pub metric_name: String,
    pub value: f64,
    pub threshold: f64,
    pub passed: bool,
}

/// Quality gate manager.
pub struct QualityGateManager {
    gates: Vec<QualityGate>,
    results: HashMap<String, QualityGateResult>,
}

impl QualityGateManager {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            results: HashMap::new(),
        }
    }
    
    /// Add a quality gate.
    pub fn add_gate(&mut self, gate: QualityGate) {
        self.gates.push(gate);
    }
    
    /// Check all quality gates.
    pub fn check_all(&mut self) -> HashMap<String, QualityGateResult> {
        let mut results = HashMap::new();
        
        for gate in &self.gates {
            let result = gate.check();
            results.insert(gate.name.clone(), result.clone());
        }
        
        self.results = results.clone();
        results
    }
    
    /// Get quality gate results.
    pub fn get_results(&self) -> &HashMap<String, QualityGateResult> {
        &self.results
    }
    
    /// Check if all quality gates pass.
    pub fn all_passed(&self) -> bool {
        self.results.values().all(|result| result.passed)
    }
    
    /// Get quality gate summary.
    pub fn get_summary(&self) -> String {
        let total_gates = self.results.len();
        let passed_gates = self.results.values()
            .filter(|result| result.passed)
            .count();
        let failed_gates = total_gates - passed_gates;
        
        format!(
            "Quality Gates: {} total, {} passed, {} failed",
            total_gates, passed_gates, failed_gates
        )
    }
}

/// Code quality analyzer.
pub struct CodeQualityAnalyzer {
    metrics: HashMap<String, f64>,
}

impl CodeQualityAnalyzer {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    /// Analyze code quality.
    pub fn analyze(&mut self) -> Result<(), String> {
        // Analyze test coverage
        self.analyze_test_coverage()?;
        
        // Analyze code complexity
        self.analyze_complexity()?;
        
        // Analyze maintainability
        self.analyze_maintainability()?;
        
        // Analyze security
        self.analyze_security()?;
        
        Ok(())
    }
    
    /// Analyze test coverage.
    fn analyze_test_coverage(&mut self) -> Result<(), String> {
        // In a real implementation, you would run cargo tarpaulin
        // or similar tool to get test coverage
        self.metrics.insert("test_coverage".to_string(), 85.0);
        Ok(())
    }
    
    /// Analyze code complexity.
    fn analyze_complexity(&mut self) -> Result<(), String> {
        // In a real implementation, you would run cargo clippy
        // or similar tool to get complexity metrics
        self.metrics.insert("cyclomatic_complexity".to_string(), 5.2);
        Ok(())
    }
    
    /// Analyze maintainability.
    fn analyze_maintainability(&mut self) -> Result<(), String> {
        // In a real implementation, you would run various tools
        // to get maintainability metrics
        self.metrics.insert("maintainability_index".to_string(), 75.0);
        Ok(())
    }
    
    /// Analyze security.
    fn analyze_security(&mut self) -> Result<(), String> {
        // In a real implementation, you would run cargo audit
        // or similar tool to get security metrics
        self.metrics.insert("security_score".to_string(), 90.0);
        Ok(())
    }
    
    /// Get quality metrics.
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_gate() {
        let mut gate = QualityGate::new("test-gate".to_string());
        
        let metric = QualityMetric {
            name: "test_coverage".to_string(),
            value: 85.0,
            threshold: 80.0,
            operator: ComparisonOperator::GreaterThanOrEqual,
        };
        
        gate.add_metric(metric);
        
        let result = gate.check();
        assert!(result.passed);
    }
    
    #[test]
    fn test_quality_gate_manager() {
        let mut manager = QualityGateManager::new();
        
        let mut gate = QualityGate::new("test-gate".to_string());
        let metric = QualityMetric {
            name: "test_coverage".to_string(),
            value: 85.0,
            threshold: 80.0,
            operator: ComparisonOperator::GreaterThanOrEqual,
        };
        gate.add_metric(metric);
        
        manager.add_gate(gate);
        let results = manager.check_all();
        
        assert_eq!(results.len(), 1);
        assert!(manager.all_passed());
    }
    
    #[test]
    fn test_code_quality_analyzer() {
        let mut analyzer = CodeQualityAnalyzer::new();
        analyzer.analyze().unwrap();
        
        let metrics = analyzer.get_metrics();
        assert!(metrics.contains_key("test_coverage"));
        assert!(metrics.contains_key("cyclomatic_complexity"));
    }
}
```

## TL;DR Runbook

### Quick Start

```yaml
# GitHub Actions workflow
name: Rust CI/CD
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --all -- --check
```

### Essential Patterns

```rust
// Complete CI/CD setup
pub fn setup_rust_cicd() {
    // 1. GitHub Actions
    // 2. GitLab CI/CD
    // 3. Jenkins pipelines
    // 4. Quality gates
    // 5. Security scanning
    // 6. Automated testing
    // 7. Build optimization
    // 8. Deployment automation
    
    println!("Rust CI/CD setup complete!");
}
```

---

*This guide provides the complete machinery for Rust CI/CD pipelines. Each pattern includes implementation examples, pipeline strategies, and real-world usage patterns for enterprise CI/CD systems.*
