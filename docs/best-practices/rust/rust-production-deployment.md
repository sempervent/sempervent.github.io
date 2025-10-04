# Rust Production Deployment Best Practices

**Objective**: Master senior-level Rust production deployment patterns for enterprise systems. When you need to deploy Rust applications to production, when you want to ensure high availability and reliability, when you need enterprise-grade deployment strategies—these best practices become your weapon of choice.

## Core Principles

- **Zero Downtime**: Deploy without service interruption
- **Rolling Updates**: Gradual deployment of new versions
- **Blue-Green Deployment**: Maintain two identical environments
- **Canary Releases**: Test new versions with a subset of users
- **Rollback Strategy**: Quick recovery from failed deployments

## Production Deployment Patterns

### Blue-Green Deployment

```rust
// rust/01-blue-green-deployment.rs

/*
Blue-green deployment patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Blue-green deployment manager.
pub struct BlueGreenDeployment {
    blue_environment: Environment,
    green_environment: Environment,
    active_environment: EnvironmentType,
    deployment_config: DeploymentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub name: String,
    pub version: String,
    pub status: EnvironmentStatus,
    pub health_check_url: String,
    pub load_balancer_weight: u32,
    pub instances: Vec<Instance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentStatus {
    Healthy,
    Unhealthy,
    Deploying,
    Deployed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    Blue,
    Green,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub status: InstanceStatus,
    pub health_check_url: String,
    pub load_balancer_weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceStatus {
    Healthy,
    Unhealthy,
    Starting,
    Stopping,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub health_check_timeout: Duration,
    pub health_check_interval: Duration,
    pub max_health_check_attempts: u32,
    pub rollback_timeout: Duration,
    pub load_balancer_switch_delay: Duration,
}

impl BlueGreenDeployment {
    pub fn new(config: DeploymentConfig) -> Self {
        Self {
            blue_environment: Environment {
                name: "blue".to_string(),
                version: "1.0.0".to_string(),
                status: EnvironmentStatus::Healthy,
                health_check_url: "http://blue.example.com/health".to_string(),
                load_balancer_weight: 100,
                instances: Vec::new(),
            },
            green_environment: Environment {
                name: "green".to_string(),
                version: "1.0.0".to_string(),
                status: EnvironmentStatus::Healthy,
                health_check_url: "http://green.example.com/health".to_string(),
                load_balancer_weight: 0,
                instances: Vec::new(),
            },
            active_environment: EnvironmentType::Blue,
            deployment_config: config,
        }
    }
    
    /// Deploy a new version using blue-green deployment.
    pub async fn deploy(&mut self, new_version: String) -> Result<DeploymentResult, String> {
        let start_time = Instant::now();
        
        // Determine which environment to deploy to
        let target_environment = match self.active_environment {
            EnvironmentType::Blue => &mut self.green_environment,
            EnvironmentType::Green => &mut self.blue_environment,
        };
        
        // Deploy to inactive environment
        target_environment.status = EnvironmentStatus::Deploying;
        target_environment.version = new_version.clone();
        
        // Deploy instances
        self.deploy_instances(target_environment).await?;
        
        // Wait for health checks
        self.wait_for_health_checks(target_environment).await?;
        
        // Switch traffic to new environment
        self.switch_traffic().await?;
        
        // Update active environment
        self.active_environment = match self.active_environment {
            EnvironmentType::Blue => EnvironmentType::Green,
            EnvironmentType::Green => EnvironmentType::Blue,
        };
        
        let duration = start_time.elapsed();
        
        Ok(DeploymentResult {
            success: true,
            duration,
            new_version,
            active_environment: self.active_environment.clone(),
        })
    }
    
    /// Deploy instances to an environment.
    async fn deploy_instances(&self, environment: &mut Environment) -> Result<(), String> {
        // In a real implementation, you would deploy instances using
        // Kubernetes, Docker Swarm, or other orchestration platform
        println!("Deploying instances to {} environment", environment.name);
        
        // Simulate deployment
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Create instances
        environment.instances = vec![
            Instance {
                id: "instance-1".to_string(),
                status: InstanceStatus::Starting,
                health_check_url: format!("{}/health", environment.health_check_url),
                load_balancer_weight: 50,
            },
            Instance {
                id: "instance-2".to_string(),
                status: InstanceStatus::Starting,
                health_check_url: format!("{}/health", environment.health_check_url),
                load_balancer_weight: 50,
            },
        ];
        
        Ok(())
    }
    
    /// Wait for health checks to pass.
    async fn wait_for_health_checks(&self, environment: &mut Environment) -> Result<(), String> {
        let mut attempts = 0;
        
        while attempts < self.deployment_config.max_health_check_attempts {
            // Check health of all instances
            let mut all_healthy = true;
            for instance in &mut environment.instances {
                if self.check_instance_health(instance).await? {
                    instance.status = InstanceStatus::Healthy;
                } else {
                    instance.status = InstanceStatus::Unhealthy;
                    all_healthy = false;
                }
            }
            
            if all_healthy {
                environment.status = EnvironmentStatus::Healthy;
                return Ok(());
            }
            
            attempts += 1;
            tokio::time::sleep(self.deployment_config.health_check_interval).await;
        }
        
        environment.status = EnvironmentStatus::Failed;
        Err("Health checks failed".to_string())
    }
    
    /// Check health of a single instance.
    async fn check_instance_health(&self, instance: &Instance) -> Result<bool, String> {
        // In a real implementation, you would make an HTTP request
        // to the health check URL
        println!("Checking health of instance {}", instance.id);
        
        // Simulate health check
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // For this example, assume all instances are healthy
        Ok(true)
    }
    
    /// Switch traffic to the new environment.
    async fn switch_traffic(&mut self) -> Result<(), String> {
        // In a real implementation, you would update load balancer
        // configuration to route traffic to the new environment
        
        let (blue_weight, green_weight) = match self.active_environment {
            EnvironmentType::Blue => (100, 0),
            EnvironmentType::Green => (0, 100),
        };
        
        self.blue_environment.load_balancer_weight = blue_weight;
        self.green_environment.load_balancer_weight = green_weight;
        
        // Wait for load balancer to switch
        tokio::time::sleep(self.deployment_config.load_balancer_switch_delay).await;
        
        Ok(())
    }
    
    /// Rollback to the previous environment.
    pub async fn rollback(&mut self) -> Result<DeploymentResult, String> {
        let start_time = Instant::now();
        
        // Switch back to the previous environment
        self.active_environment = match self.active_environment {
            EnvironmentType::Blue => EnvironmentType::Green,
            EnvironmentType::Green => EnvironmentType::Blue,
        };
        
        // Switch traffic back
        self.switch_traffic().await?;
        
        let duration = start_time.elapsed();
        
        Ok(DeploymentResult {
            success: true,
            duration,
            new_version: "rollback".to_string(),
            active_environment: self.active_environment.clone(),
        })
    }
    
    /// Get deployment status.
    pub fn get_status(&self) -> DeploymentStatus {
        DeploymentStatus {
            active_environment: self.active_environment.clone(),
            blue_environment: self.blue_environment.clone(),
            green_environment: self.green_environment.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentResult {
    pub success: bool,
    pub duration: Duration,
    pub new_version: String,
    pub active_environment: EnvironmentType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentStatus {
    pub active_environment: EnvironmentType,
    pub blue_environment: Environment,
    pub green_environment: Environment,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_blue_green_deployment() {
        let config = DeploymentConfig {
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
            rollback_timeout: Duration::from_secs(60),
            load_balancer_switch_delay: Duration::from_secs(10),
        };
        
        let mut deployment = BlueGreenDeployment::new(config);
        let result = deployment.deploy("2.0.0".to_string()).await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
    
    #[tokio::test]
    async fn test_rollback() {
        let config = DeploymentConfig {
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
            rollback_timeout: Duration::from_secs(60),
            load_balancer_switch_delay: Duration::from_secs(10),
        };
        
        let mut deployment = BlueGreenDeployment::new(config);
        let result = deployment.rollback().await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
}
```

### Canary Deployment

```rust
// rust/02-canary-deployment.rs

/*
Canary deployment patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Canary deployment manager.
pub struct CanaryDeployment {
    stable_environment: Environment,
    canary_environment: Environment,
    canary_percentage: u32,
    deployment_config: DeploymentConfig,
    metrics: CanaryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryMetrics {
    pub error_rate: f64,
    pub response_time: Duration,
    pub throughput: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
}

impl CanaryDeployment {
    pub fn new(config: DeploymentConfig) -> Self {
        Self {
            stable_environment: Environment {
                name: "stable".to_string(),
                version: "1.0.0".to_string(),
                status: EnvironmentStatus::Healthy,
                health_check_url: "http://stable.example.com/health".to_string(),
                load_balancer_weight: 100,
                instances: Vec::new(),
            },
            canary_environment: Environment {
                name: "canary".to_string(),
                version: "1.0.0".to_string(),
                status: EnvironmentStatus::Healthy,
                health_check_url: "http://canary.example.com/health".to_string(),
                load_balancer_weight: 0,
                instances: Vec::new(),
            },
            canary_percentage: 0,
            deployment_config: config,
            metrics: CanaryMetrics {
                error_rate: 0.0,
                response_time: Duration::from_millis(100),
                throughput: 1000.0,
                cpu_usage: 50.0,
                memory_usage: 60.0,
            },
        }
    }
    
    /// Deploy a new version using canary deployment.
    pub async fn deploy(&mut self, new_version: String, initial_percentage: u32) -> Result<DeploymentResult, String> {
        let start_time = Instant::now();
        
        // Deploy to canary environment
        self.canary_environment.status = EnvironmentStatus::Deploying;
        self.canary_environment.version = new_version.clone();
        
        // Deploy instances
        self.deploy_instances(&mut self.canary_environment).await?;
        
        // Wait for health checks
        self.wait_for_health_checks(&mut self.canary_environment).await?;
        
        // Start with initial canary percentage
        self.canary_percentage = initial_percentage;
        self.update_load_balancer_weights().await?;
        
        // Monitor canary metrics
        self.monitor_canary_metrics().await?;
        
        let duration = start_time.elapsed();
        
        Ok(DeploymentResult {
            success: true,
            duration,
            new_version,
            active_environment: EnvironmentType::Blue, // Not applicable for canary
        })
    }
    
    /// Deploy instances to an environment.
    async fn deploy_instances(&self, environment: &mut Environment) -> Result<(), String> {
        // In a real implementation, you would deploy instances using
        // Kubernetes, Docker Swarm, or other orchestration platform
        println!("Deploying instances to {} environment", environment.name);
        
        // Simulate deployment
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Create instances
        environment.instances = vec![
            Instance {
                id: "instance-1".to_string(),
                status: InstanceStatus::Starting,
                health_check_url: format!("{}/health", environment.health_check_url),
                load_balancer_weight: 50,
            },
        ];
        
        Ok(())
    }
    
    /// Wait for health checks to pass.
    async fn wait_for_health_checks(&self, environment: &mut Environment) -> Result<(), String> {
        let mut attempts = 0;
        
        while attempts < self.deployment_config.max_health_check_attempts {
            // Check health of all instances
            let mut all_healthy = true;
            for instance in &mut environment.instances {
                if self.check_instance_health(instance).await? {
                    instance.status = InstanceStatus::Healthy;
                } else {
                    instance.status = InstanceStatus::Unhealthy;
                    all_healthy = false;
                }
            }
            
            if all_healthy {
                environment.status = EnvironmentStatus::Healthy;
                return Ok(());
            }
            
            attempts += 1;
            tokio::time::sleep(self.deployment_config.health_check_interval).await;
        }
        
        environment.status = EnvironmentStatus::Failed;
        Err("Health checks failed".to_string())
    }
    
    /// Check health of a single instance.
    async fn check_instance_health(&self, instance: &Instance) -> Result<bool, String> {
        // In a real implementation, you would make an HTTP request
        // to the health check URL
        println!("Checking health of instance {}", instance.id);
        
        // Simulate health check
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // For this example, assume all instances are healthy
        Ok(true)
    }
    
    /// Update load balancer weights based on canary percentage.
    async fn update_load_balancer_weights(&mut self) -> Result<(), String> {
        let stable_weight = 100 - self.canary_percentage;
        let canary_weight = self.canary_percentage;
        
        self.stable_environment.load_balancer_weight = stable_weight;
        self.canary_environment.load_balancer_weight = canary_weight;
        
        println!("Updated load balancer weights: stable={}%, canary={}%", stable_weight, canary_weight);
        
        Ok(())
    }
    
    /// Monitor canary metrics and make decisions.
    async fn monitor_canary_metrics(&mut self) -> Result<(), String> {
        // In a real implementation, you would collect metrics from
        // monitoring systems like Prometheus, DataDog, etc.
        
        // Simulate metric collection
        self.metrics.error_rate = 0.01; // 1% error rate
        self.metrics.response_time = Duration::from_millis(150);
        self.metrics.throughput = 1200.0;
        self.metrics.cpu_usage = 55.0;
        self.metrics.memory_usage = 65.0;
        
        // Check if metrics are within acceptable ranges
        if self.metrics.error_rate > 0.05 { // 5% error rate threshold
            return Err("Canary deployment failed: error rate too high".to_string());
        }
        
        if self.metrics.response_time > Duration::from_millis(500) {
            return Err("Canary deployment failed: response time too high".to_string());
        }
        
        if self.metrics.cpu_usage > 80.0 {
            return Err("Canary deployment failed: CPU usage too high".to_string());
        }
        
        if self.metrics.memory_usage > 90.0 {
            return Err("Canary deployment failed: memory usage too high".to_string());
        }
        
        Ok(())
    }
    
    /// Increase canary percentage.
    pub async fn increase_canary_percentage(&mut self, percentage: u32) -> Result<(), String> {
        self.canary_percentage = (self.canary_percentage + percentage).min(100);
        self.update_load_balancer_weights().await?;
        
        // Monitor metrics after increase
        self.monitor_canary_metrics().await?;
        
        Ok(())
    }
    
    /// Promote canary to stable.
    pub async fn promote_canary(&mut self) -> Result<(), String> {
        // Switch all traffic to canary
        self.canary_percentage = 100;
        self.update_load_balancer_weights().await?;
        
        // Wait for traffic to stabilize
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // Update stable environment
        self.stable_environment.version = self.canary_environment.version.clone();
        self.stable_environment.instances = self.canary_environment.instances.clone();
        
        // Reset canary
        self.canary_percentage = 0;
        self.update_load_balancer_weights().await?;
        
        Ok(())
    }
    
    /// Rollback canary deployment.
    pub async fn rollback_canary(&mut self) -> Result<(), String> {
        // Set canary percentage to 0
        self.canary_percentage = 0;
        self.update_load_balancer_weights().await?;
        
        // Stop canary instances
        for instance in &mut self.canary_environment.instances {
            instance.status = InstanceStatus::Stopped;
        }
        
        Ok(())
    }
    
    /// Get canary metrics.
    pub fn get_metrics(&self) -> &CanaryMetrics {
        &self.metrics
    }
    
    /// Get canary status.
    pub fn get_status(&self) -> CanaryStatus {
        CanaryStatus {
            canary_percentage: self.canary_percentage,
            stable_environment: self.stable_environment.clone(),
            canary_environment: self.canary_environment.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryStatus {
    pub canary_percentage: u32,
    pub stable_environment: Environment,
    pub canary_environment: Environment,
    pub metrics: CanaryMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_canary_deployment() {
        let config = DeploymentConfig {
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
            rollback_timeout: Duration::from_secs(60),
            load_balancer_switch_delay: Duration::from_secs(10),
        };
        
        let mut deployment = CanaryDeployment::new(config);
        let result = deployment.deploy("2.0.0".to_string(), 10).await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
    
    #[tokio::test]
    async fn test_increase_canary_percentage() {
        let config = DeploymentConfig {
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
            rollback_timeout: Duration::from_secs(60),
            load_balancer_switch_delay: Duration::from_secs(10),
        };
        
        let mut deployment = CanaryDeployment::new(config);
        deployment.deploy("2.0.0".to_string(), 10).await.unwrap();
        
        let result = deployment.increase_canary_percentage(20).await;
        assert!(result.is_ok());
        assert_eq!(deployment.canary_percentage, 30);
    }
    
    #[tokio::test]
    async fn test_promote_canary() {
        let config = DeploymentConfig {
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
            rollback_timeout: Duration::from_secs(60),
            load_balancer_switch_delay: Duration::from_secs(10),
        };
        
        let mut deployment = CanaryDeployment::new(config);
        deployment.deploy("2.0.0".to_string(), 10).await.unwrap();
        
        let result = deployment.promote_canary().await;
        assert!(result.is_ok());
        assert_eq!(deployment.canary_percentage, 0);
    }
}
```

### Rolling Updates

```rust
// rust/03-rolling-updates.rs

/*
Rolling update patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Rolling update manager.
pub struct RollingUpdateManager {
    instances: Vec<Instance>,
    new_version: String,
    update_config: UpdateConfig,
    update_status: UpdateStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    pub max_unavailable: u32,
    pub max_surge: u32,
    pub update_timeout: Duration,
    pub health_check_timeout: Duration,
    pub health_check_interval: Duration,
    pub max_health_check_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStatus {
    NotStarted,
    InProgress,
    Completed,
    Failed,
    RollingBack,
}

impl RollingUpdateManager {
    pub fn new(instances: Vec<Instance>, new_version: String, config: UpdateConfig) -> Self {
        Self {
            instances,
            new_version,
            update_config: config,
            update_status: UpdateStatus::NotStarted,
        }
    }
    
    /// Start rolling update.
    pub async fn start_update(&mut self) -> Result<UpdateResult, String> {
        let start_time = Instant::now();
        self.update_status = UpdateStatus::InProgress;
        
        // Calculate update strategy
        let strategy = self.calculate_update_strategy()?;
        
        // Update instances in batches
        for batch in strategy.batches {
            self.update_batch(batch).await?;
        }
        
        self.update_status = UpdateStatus::Completed;
        let duration = start_time.elapsed();
        
        Ok(UpdateResult {
            success: true,
            duration,
            updated_instances: self.instances.len(),
            failed_instances: 0,
        })
    }
    
    /// Calculate update strategy.
    fn calculate_update_strategy(&self) -> Result<UpdateStrategy, String> {
        let total_instances = self.instances.len() as u32;
        let max_unavailable = self.update_config.max_unavailable;
        let max_surge = self.update_config.max_surge;
        
        // Calculate batch size
        let batch_size = if max_unavailable > 0 {
            (total_instances - max_unavailable).min(max_surge)
        } else {
            max_surge
        };
        
        if batch_size == 0 {
            return Err("Invalid update configuration".to_string());
        }
        
        // Create batches
        let mut batches = Vec::new();
        let mut remaining_instances = total_instances;
        
        while remaining_instances > 0 {
            let current_batch_size = batch_size.min(remaining_instances);
            batches.push(current_batch_size);
            remaining_instances -= current_batch_size;
        }
        
        Ok(UpdateStrategy { batches })
    }
    
    /// Update a batch of instances.
    async fn update_batch(&mut self, batch_size: u32) -> Result<(), String> {
        let mut updated_count = 0;
        
        for instance in &mut self.instances {
            if updated_count >= batch_size {
                break;
            }
            
            if instance.status == InstanceStatus::Healthy {
                // Update instance
                self.update_instance(instance).await?;
                updated_count += 1;
            }
        }
        
        // Wait for batch to be healthy
        self.wait_for_batch_health().await?;
        
        Ok(())
    }
    
    /// Update a single instance.
    async fn update_instance(&self, instance: &mut Instance) -> Result<(), String> {
        println!("Updating instance {}", instance.id);
        
        // In a real implementation, you would:
        // 1. Deploy new version to instance
        // 2. Update load balancer configuration
        // 3. Wait for instance to be ready
        
        // Simulate update
        instance.status = InstanceStatus::Starting;
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Check health
        if self.check_instance_health(instance).await? {
            instance.status = InstanceStatus::Healthy;
        } else {
            instance.status = InstanceStatus::Unhealthy;
            return Err("Instance health check failed".to_string());
        }
        
        Ok(())
    }
    
    /// Check health of a single instance.
    async fn check_instance_health(&self, instance: &Instance) -> Result<bool, String> {
        // In a real implementation, you would make an HTTP request
        // to the health check URL
        println!("Checking health of instance {}", instance.id);
        
        // Simulate health check
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // For this example, assume all instances are healthy
        Ok(true)
    }
    
    /// Wait for batch to be healthy.
    async fn wait_for_batch_health(&self) -> Result<(), String> {
        let mut attempts = 0;
        
        while attempts < self.update_config.max_health_check_attempts {
            let mut all_healthy = true;
            
            for instance in &self.instances {
                if instance.status == InstanceStatus::Starting {
                    if !self.check_instance_health(instance).await? {
                        all_healthy = false;
                        break;
                    }
                }
            }
            
            if all_healthy {
                return Ok(());
            }
            
            attempts += 1;
            tokio::time::sleep(self.update_config.health_check_interval).await;
        }
        
        Err("Batch health check failed".to_string())
    }
    
    /// Rollback update.
    pub async fn rollback(&mut self) -> Result<UpdateResult, String> {
        let start_time = Instant::now();
        self.update_status = UpdateStatus::RollingBack;
        
        // Rollback all instances
        for instance in &mut self.instances {
            if instance.status == InstanceStatus::Healthy {
                self.rollback_instance(instance).await?;
            }
        }
        
        self.update_status = UpdateStatus::Completed;
        let duration = start_time.elapsed();
        
        Ok(UpdateResult {
            success: true,
            duration,
            updated_instances: 0,
            failed_instances: 0,
        })
    }
    
    /// Rollback a single instance.
    async fn rollback_instance(&self, instance: &mut Instance) -> Result<(), String> {
        println!("Rolling back instance {}", instance.id);
        
        // In a real implementation, you would:
        // 1. Deploy previous version to instance
        // 2. Update load balancer configuration
        // 3. Wait for instance to be ready
        
        // Simulate rollback
        instance.status = InstanceStatus::Starting;
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Check health
        if self.check_instance_health(instance).await? {
            instance.status = InstanceStatus::Healthy;
        } else {
            instance.status = InstanceStatus::Unhealthy;
            return Err("Instance rollback failed".to_string());
        }
        
        Ok(())
    }
    
    /// Get update status.
    pub fn get_status(&self) -> &UpdateStatus {
        &self.update_status
    }
    
    /// Get instance status.
    pub fn get_instance_status(&self) -> Vec<&Instance> {
        self.instances.iter().collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStrategy {
    pub batches: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    pub success: bool,
    pub duration: Duration,
    pub updated_instances: usize,
    pub failed_instances: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_rolling_update() {
        let instances = vec![
            Instance {
                id: "instance-1".to_string(),
                status: InstanceStatus::Healthy,
                health_check_url: "http://instance-1.example.com/health".to_string(),
                load_balancer_weight: 50,
            },
            Instance {
                id: "instance-2".to_string(),
                status: InstanceStatus::Healthy,
                health_check_url: "http://instance-2.example.com/health".to_string(),
                load_balancer_weight: 50,
            },
        ];
        
        let config = UpdateConfig {
            max_unavailable: 1,
            max_surge: 1,
            update_timeout: Duration::from_secs(300),
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
        };
        
        let mut manager = RollingUpdateManager::new(instances, "2.0.0".to_string(), config);
        let result = manager.start_update().await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
    
    #[tokio::test]
    async fn test_rollback() {
        let instances = vec![
            Instance {
                id: "instance-1".to_string(),
                status: InstanceStatus::Healthy,
                health_check_url: "http://instance-1.example.com/health".to_string(),
                load_balancer_weight: 50,
            },
        ];
        
        let config = UpdateConfig {
            max_unavailable: 1,
            max_surge: 1,
            update_timeout: Duration::from_secs(300),
            health_check_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(5),
            max_health_check_attempts: 6,
        };
        
        let mut manager = RollingUpdateManager::new(instances, "2.0.0".to_string(), config);
        let result = manager.rollback().await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().success);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Blue-green deployment
let config = DeploymentConfig { /* ... */ };
let mut deployment = BlueGreenDeployment::new(config);
deployment.deploy("2.0.0".to_string()).await?;

// 2. Canary deployment
let mut canary = CanaryDeployment::new(config);
canary.deploy("2.0.0".to_string(), 10).await?;
canary.increase_canary_percentage(20).await?;

// 3. Rolling updates
let mut manager = RollingUpdateManager::new(instances, "2.0.0".to_string(), config);
manager.start_update().await?;
```

### Essential Patterns

```rust
// Complete production deployment setup
pub fn setup_rust_production_deployment() {
    // 1. Blue-green deployment
    // 2. Canary deployment
    // 3. Rolling updates
    // 4. Health checks
    // 5. Rollback strategies
    // 6. Load balancing
    // 7. Monitoring
    // 8. Alerting
    
    println!("Rust production deployment setup complete!");
}
```

---

*This guide provides the complete machinery for Rust production deployment. Each pattern includes implementation examples, deployment strategies, and real-world usage patterns for enterprise production systems.*
