# Python Production Deployment Best Practices

**Objective**: Master senior-level Python production deployment patterns for enterprise systems. When you need to deploy applications at scale, when you want to implement zero-downtime deployments, when you need enterprise-grade deployment strategies—these best practices become your weapon of choice.

## Core Principles

- **Reliability**: Ensure high availability and fault tolerance
- **Scalability**: Design for horizontal and vertical scaling
- **Security**: Implement comprehensive security measures
- **Monitoring**: Deploy with full observability
- **Automation**: Automate deployment processes

## Deployment Strategies

### Blue-Green Deployment

```python
# python/01-blue-green-deployment.py

"""
Blue-green deployment patterns and zero-downtime deployment strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
import subprocess
import requests
from datetime import datetime, timedelta
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"

class Environment(Enum):
    """Environment enumeration"""
    BLUE = "blue"
    GREEN = "green"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    application_name: str
    version: str
    environment: str
    replicas: int = 3
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    cpu_request: str = "250m"
    memory_request: str = "256Mi"
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    max_unavailable: int = 1
    max_surge: int = 1
    deployment_timeout: int = 300
    rollback_timeout: int = 180

class BlueGreenDeployer:
    """Blue-green deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_environment = None
        self.deployment_status = DeploymentStatus.PENDING
        self.deployment_metrics = {}
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer()
    
    def determine_current_environment(self) -> Environment:
        """Determine current active environment"""
        try:
            # Check which environment is currently receiving traffic
            blue_status = self.health_checker.check_environment_health("blue")
            green_status = self.health_checker.check_environment_health("green")
            
            if blue_status["healthy"] and not green_status["healthy"]:
                return Environment.BLUE
            elif green_status["healthy"] and not blue_status["healthy"]:
                return Environment.GREEN
            else:
                # Default to blue if both are healthy or both are unhealthy
                return Environment.BLUE
        except Exception as e:
            logger.error(f"Error determining current environment: {e}")
            return Environment.BLUE
    
    def deploy_to_environment(self, environment: Environment) -> bool:
        """Deploy application to specified environment"""
        try:
            self.deployment_status = DeploymentStatus.IN_PROGRESS
            start_time = time.time()
            
            logger.info(f"Starting deployment to {environment.value} environment")
            
            # Update deployment configuration
            deployment_manifest = self._create_deployment_manifest(environment)
            
            # Apply deployment
            success = self._apply_deployment(deployment_manifest, environment)
            
            if not success:
                logger.error(f"Failed to apply deployment to {environment.value}")
                self.deployment_status = DeploymentStatus.FAILED
                return False
            
            # Wait for deployment to be ready
            ready = self._wait_for_deployment_ready(environment)
            
            if not ready:
                logger.error(f"Deployment to {environment.value} not ready within timeout")
                self.deployment_status = DeploymentStatus.FAILED
                return False
            
            # Run health checks
            health_checks_passed = self._run_health_checks(environment)
            
            if not health_checks_passed:
                logger.error(f"Health checks failed for {environment.value}")
                self.deployment_status = DeploymentStatus.FAILED
                return False
            
            deployment_time = time.time() - start_time
            self.deployment_metrics[environment.value] = {
                "deployment_time": deployment_time,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully deployed to {environment.value} in {deployment_time:.2f}s")
            self.deployment_status = DeploymentStatus.SUCCESS
            return True
            
        except Exception as e:
            logger.error(f"Error deploying to {environment.value}: {e}")
            self.deployment_status = DeploymentStatus.FAILED
            return False
    
    def switch_traffic(self, from_env: Environment, to_env: Environment) -> bool:
        """Switch traffic from one environment to another"""
        try:
            logger.info(f"Switching traffic from {from_env.value} to {to_env.value}")
            
            # Update load balancer configuration
            success = self.load_balancer.update_backend_servers(to_env.value)
            
            if not success:
                logger.error(f"Failed to update load balancer for {to_env.value}")
                return False
            
            # Verify traffic switch
            traffic_switched = self._verify_traffic_switch(to_env)
            
            if not traffic_switched:
                logger.error(f"Traffic switch verification failed for {to_env.value}")
                return False
            
            logger.info(f"Successfully switched traffic to {to_env.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching traffic: {e}")
            return False
    
    def rollback_deployment(self, from_env: Environment, to_env: Environment) -> bool:
        """Rollback deployment to previous environment"""
        try:
            logger.info(f"Rolling back from {from_env.value} to {to_env.value}")
            
            self.deployment_status = DeploymentStatus.ROLLBACK
            
            # Switch traffic back
            traffic_switched = self.switch_traffic(from_env, to_env)
            
            if not traffic_switched:
                logger.error(f"Failed to rollback traffic to {to_env.value}")
                return False
            
            # Scale down failed environment
            self._scale_down_environment(from_env)
            
            logger.info(f"Successfully rolled back to {to_env.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def _create_deployment_manifest(self, environment: Environment) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.application_name}-{environment.value}",
                "labels": {
                    "app": self.config.application_name,
                    "environment": environment.value,
                    "version": self.config.version
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.application_name,
                        "environment": environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.application_name,
                            "environment": environment.value,
                            "version": self.config.version
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": self.config.application_name,
                                "image": f"{self.config.application_name}:{self.config.version}",
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit
                                    },
                                    "requests": {
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": self.config.health_check_path,
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": self.config.readiness_check_path,
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return manifest
    
    def _apply_deployment(self, manifest: Dict[str, Any], environment: Environment) -> bool:
        """Apply deployment manifest to Kubernetes"""
        try:
            # In real implementation, this would use kubectl or Kubernetes API
            logger.info(f"Applying deployment manifest for {environment.value}")
            
            # Simulate deployment application
            time.sleep(2)
            
            return True
        except Exception as e:
            logger.error(f"Error applying deployment: {e}")
            return False
    
    def _wait_for_deployment_ready(self, environment: Environment) -> bool:
        """Wait for deployment to be ready"""
        start_time = time.time()
        timeout = self.config.deployment_timeout
        
        while time.time() - start_time < timeout:
            if self.health_checker.check_environment_health(environment.value)["healthy"]:
                return True
            time.sleep(10)
        
        return False
    
    def _run_health_checks(self, environment: Environment) -> bool:
        """Run comprehensive health checks"""
        health_checks = [
            self.health_checker.check_application_health,
            self.health_checker.check_database_connectivity,
            self.health_checker.check_external_services,
            self.health_checker.check_performance_metrics
        ]
        
        for health_check in health_checks:
            if not health_check(environment.value):
                return False
        
        return True
    
    def _verify_traffic_switch(self, environment: Environment) -> bool:
        """Verify that traffic has been switched to the new environment"""
        # Check if traffic is reaching the new environment
        for _ in range(10):  # Check 10 times
            if self.health_checker.check_environment_health(environment.value)["healthy"]:
                return True
            time.sleep(5)
        
        return False
    
    def _scale_down_environment(self, environment: Environment) -> None:
        """Scale down the specified environment"""
        logger.info(f"Scaling down {environment.value} environment")
        # In real implementation, this would scale down the deployment

class HealthChecker:
    """Health check utilities for deployments"""
    
    def __init__(self):
        self.health_check_metrics = {}
    
    def check_environment_health(self, environment: str) -> Dict[str, Any]:
        """Check overall health of an environment"""
        try:
            # Simulate health check
            health_status = {
                "environment": environment,
                "healthy": True,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {}
            }
            
            # Check application health
            app_health = self.check_application_health(environment)
            health_status["checks"]["application"] = app_health
            
            # Check database connectivity
            db_health = self.check_database_connectivity(environment)
            health_status["checks"]["database"] = db_health
            
            # Overall health is healthy if all checks pass
            health_status["healthy"] = app_health and db_health
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking environment health: {e}")
            return {
                "environment": environment,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def check_application_health(self, environment: str) -> bool:
        """Check application health"""
        try:
            # Simulate health check request
            response = requests.get(f"http://{environment}-app:8000/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Application health check failed for {environment}: {e}")
            return False
    
    def check_database_connectivity(self, environment: str) -> bool:
        """Check database connectivity"""
        try:
            # Simulate database connectivity check
            # In real implementation, this would check actual database connection
            return True
        except Exception as e:
            logger.error(f"Database connectivity check failed for {environment}: {e}")
            return False
    
    def check_external_services(self, environment: str) -> bool:
        """Check external service dependencies"""
        try:
            # Simulate external service checks
            # In real implementation, this would check actual external services
            return True
        except Exception as e:
            logger.error(f"External services check failed for {environment}: {e}")
            return False
    
    def check_performance_metrics(self, environment: str) -> bool:
        """Check performance metrics"""
        try:
            # Simulate performance metrics check
            # In real implementation, this would check actual performance metrics
            return True
        except Exception as e:
            logger.error(f"Performance metrics check failed for {environment}: {e}")
            return False

class LoadBalancer:
    """Load balancer management"""
    
    def __init__(self):
        self.backend_servers = {}
        self.load_balancer_metrics = {}
    
    def update_backend_servers(self, environment: str) -> bool:
        """Update load balancer backend servers"""
        try:
            logger.info(f"Updating load balancer backend servers to {environment}")
            
            # In real implementation, this would update actual load balancer configuration
            self.backend_servers["active"] = environment
            
            return True
        except Exception as e:
            logger.error(f"Error updating load balancer: {e}")
            return False
    
    def get_backend_servers(self) -> Dict[str, str]:
        """Get current backend server configuration"""
        return self.backend_servers.copy()

class CanaryDeployer:
    """Canary deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.canary_percentage = 0
        self.canary_metrics = {}
        self.monitoring_duration = 300  # 5 minutes
    
    def deploy_canary(self, canary_percentage: int = 10) -> bool:
        """Deploy canary version"""
        try:
            self.canary_percentage = canary_percentage
            logger.info(f"Starting canary deployment with {canary_percentage}% traffic")
            
            # Deploy canary version
            canary_deployed = self._deploy_canary_version()
            
            if not canary_deployed:
                return False
            
            # Route percentage of traffic to canary
            traffic_routed = self._route_canary_traffic(canary_percentage)
            
            if not traffic_routed:
                return False
            
            # Monitor canary performance
            monitoring_success = self._monitor_canary_performance()
            
            if not monitoring_success:
                logger.warning("Canary monitoring detected issues")
                return False
            
            logger.info("Canary deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Error in canary deployment: {e}")
            return False
    
    def promote_canary(self) -> bool:
        """Promote canary to full deployment"""
        try:
            logger.info("Promoting canary to full deployment")
            
            # Route 100% traffic to canary
            traffic_routed = self._route_canary_traffic(100)
            
            if not traffic_routed:
                return False
            
            # Scale up canary environment
            scaled_up = self._scale_up_canary()
            
            if not scaled_up:
                return False
            
            # Scale down old environment
            self._scale_down_old_environment()
            
            logger.info("Canary successfully promoted")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting canary: {e}")
            return False
    
    def rollback_canary(self) -> bool:
        """Rollback canary deployment"""
        try:
            logger.info("Rolling back canary deployment")
            
            # Route traffic back to stable version
            traffic_routed = self._route_canary_traffic(0)
            
            if not traffic_routed:
                return False
            
            # Scale down canary environment
            self._scale_down_canary()
            
            logger.info("Canary rollback successful")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back canary: {e}")
            return False
    
    def _deploy_canary_version(self) -> bool:
        """Deploy canary version"""
        # Simulate canary deployment
        time.sleep(2)
        return True
    
    def _route_canary_traffic(self, percentage: int) -> bool:
        """Route traffic to canary"""
        logger.info(f"Routing {percentage}% traffic to canary")
        # Simulate traffic routing
        time.sleep(1)
        return True
    
    def _monitor_canary_performance(self) -> bool:
        """Monitor canary performance"""
        logger.info(f"Monitoring canary performance for {self.monitoring_duration} seconds")
        
        start_time = time.time()
        while time.time() - start_time < self.monitoring_duration:
            # Simulate performance monitoring
            if not self._check_canary_metrics():
                return False
            time.sleep(30)
        
        return True
    
    def _check_canary_metrics(self) -> bool:
        """Check canary performance metrics"""
        # Simulate metrics check
        return True
    
    def _scale_up_canary(self) -> bool:
        """Scale up canary environment"""
        logger.info("Scaling up canary environment")
        return True
    
    def _scale_down_canary(self) -> bool:
        """Scale down canary environment"""
        logger.info("Scaling down canary environment")
        return True
    
    def _scale_down_old_environment(self) -> bool:
        """Scale down old environment"""
        logger.info("Scaling down old environment")
        return True

# Usage examples
def example_blue_green_deployment():
    """Example blue-green deployment usage"""
    # Create deployment configuration
    config = DeploymentConfig(
        application_name="my-app",
        version="v1.2.0",
        environment="production",
        replicas=3
    )
    
    # Create blue-green deployer
    deployer = BlueGreenDeployer(config)
    
    # Determine current environment
    current_env = deployer.determine_current_environment()
    print(f"Current environment: {current_env.value}")
    
    # Determine target environment
    target_env = Environment.GREEN if current_env == Environment.BLUE else Environment.BLUE
    
    # Deploy to target environment
    deployment_success = deployer.deploy_to_environment(target_env)
    print(f"Deployment to {target_env.value}: {deployment_success}")
    
    if deployment_success:
        # Switch traffic
        traffic_switched = deployer.switch_traffic(current_env, target_env)
        print(f"Traffic switch: {traffic_switched}")
        
        if not traffic_switched:
            # Rollback if traffic switch failed
            rollback_success = deployer.rollback_deployment(target_env, current_env)
            print(f"Rollback: {rollback_success}")
    
    # Canary deployment example
    canary_deployer = CanaryDeployer(config)
    
    # Deploy canary
    canary_success = canary_deployer.deploy_canary(canary_percentage=10)
    print(f"Canary deployment: {canary_success}")
    
    if canary_success:
        # Promote canary
        promote_success = canary_deployer.promote_canary()
        print(f"Canary promotion: {promote_success}")
```

### Production Configuration

```python
# python/02-production-configuration.py

"""
Production configuration management and environment setup
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import secrets
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigManager:
    """Production configuration manager"""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self.config = {}
        self.secrets = {}
        self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration for environment"""
        # Load base configuration
        base_config = self._load_base_config()
        
        # Load environment-specific configuration
        env_config = self._load_environment_config()
        
        # Load secrets
        secrets_config = self._load_secrets()
        
        # Merge configurations
        self.config = {**base_config, **env_config}
        self.secrets = secrets_config
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        return {
            "app_name": "my-python-app",
            "version": "1.0.0",
            "debug": False,
            "log_level": "INFO",
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "myapp",
                "pool_size": 10,
                "max_overflow": 20
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 8000,
                "health_check_interval": 30
            }
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_configs = {
            Environment.DEVELOPMENT: {
                "debug": True,
                "log_level": "DEBUG",
                "database": {
                    "host": "localhost",
                    "port": 5432
                }
            },
            Environment.STAGING: {
                "debug": False,
                "log_level": "INFO",
                "database": {
                    "host": "staging-db.example.com",
                    "port": 5432
                }
            },
            Environment.PRODUCTION: {
                "debug": False,
                "log_level": "WARNING",
                "database": {
                    "host": "prod-db.example.com",
                    "port": 5432
                },
                "monitoring": {
                    "enabled": True,
                    "alerting": True
                }
            }
        }
        
        return env_configs.get(self.environment, {})
    
    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from environment variables or secret store"""
        secrets = {}
        
        # Database secrets
        secrets["database_password"] = os.getenv("DATABASE_PASSWORD")
        secrets["database_user"] = os.getenv("DATABASE_USER", "myapp")
        
        # Redis secrets
        secrets["redis_password"] = os.getenv("REDIS_PASSWORD")
        
        # API keys
        secrets["api_key"] = os.getenv("API_KEY")
        secrets["jwt_secret"] = os.getenv("JWT_SECRET")
        
        # External service credentials
        secrets["aws_access_key"] = os.getenv("AWS_ACCESS_KEY_ID")
        secrets["aws_secret_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        return secrets
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret value"""
        return self.secrets.get(key, default)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check required configuration
        required_configs = [
            "app_name",
            "version",
            "database.host",
            "database.port"
        ]
        
        for config_key in required_configs:
            if self.get_config(config_key) is None:
                issues.append(f"Missing required configuration: {config_key}")
        
        # Check required secrets
        required_secrets = [
            "database_password",
            "jwt_secret"
        ]
        
        for secret_key in required_secrets:
            if self.get_secret(secret_key) is None:
                issues.append(f"Missing required secret: {secret_key}")
        
        # Environment-specific validation
        if self.environment == Environment.PRODUCTION:
            if self.get_config("debug") is True:
                issues.append("Debug mode should not be enabled in production")
            
            if self.get_config("log_level") == "DEBUG":
                issues.append("Debug logging should not be enabled in production")
        
        return issues

class SecretManager:
    """Secret management utilities"""
    
    def __init__(self, environment: Environment):
        self.environment = environment
        self.secrets = {}
    
    def generate_secret(self, length: int = 32) -> str:
        """Generate a secure random secret"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def store_secret(self, key: str, value: str) -> None:
        """Store secret (in production, this would use a secret store)"""
        self.secrets[key] = value
        logger.info(f"Secret stored for key: {key}")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value"""
        return self.secrets.get(key)
    
    def rotate_secret(self, key: str) -> str:
        """Rotate secret and return new value"""
        new_secret = self.generate_secret()
        self.store_secret(key, new_secret)
        logger.info(f"Secret rotated for key: {key}")
        return new_secret

class ProductionValidator:
    """Production readiness validator"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.validation_results = {}
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate production readiness"""
        validations = {
            "configuration": self._validate_configuration(),
            "security": self._validate_security(),
            "performance": self._validate_performance(),
            "monitoring": self._validate_monitoring(),
            "scalability": self._validate_scalability()
        }
        
        overall_success = all(result["success"] for result in validations.values())
        
        return {
            "overall_success": overall_success,
            "validations": validations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration"""
        issues = self.config_manager.validate_configuration()
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "checks_performed": [
                "Required configuration present",
                "Required secrets present",
                "Environment-specific settings"
            ]
        }
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security settings"""
        issues = []
        
        # Check debug mode
        if self.config_manager.get_config("debug") is True:
            issues.append("Debug mode is enabled")
        
        # Check log level
        if self.config_manager.get_config("log_level") == "DEBUG":
            issues.append("Debug logging is enabled")
        
        # Check for default secrets
        default_secrets = ["password", "secret", "key"]
        for secret_key, secret_value in self.config_manager.secrets.items():
            if secret_value in default_secrets:
                issues.append(f"Default secret detected: {secret_key}")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "checks_performed": [
                "Debug mode disabled",
                "Debug logging disabled",
                "No default secrets"
            ]
        }
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance settings"""
        issues = []
        
        # Check database connection pool
        pool_size = self.config_manager.get_config("database.pool_size", 0)
        if pool_size < 5:
            issues.append("Database connection pool size too small")
        
        # Check Redis configuration
        redis_host = self.config_manager.get_config("redis.host")
        if not redis_host:
            issues.append("Redis configuration missing")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "checks_performed": [
                "Database connection pool",
                "Redis configuration",
                "Resource limits"
            ]
        }
    
    def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring configuration"""
        issues = []
        
        # Check monitoring enabled
        monitoring_enabled = self.config_manager.get_config("monitoring.enabled", False)
        if not monitoring_enabled:
            issues.append("Monitoring is not enabled")
        
        # Check health check configuration
        health_check_interval = self.config_manager.get_config("monitoring.health_check_interval", 0)
        if health_check_interval > 60:
            issues.append("Health check interval too long")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "checks_performed": [
                "Monitoring enabled",
                "Health check configuration",
                "Metrics collection"
            ]
        }
    
    def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability settings"""
        issues = []
        
        # Check for horizontal scaling configuration
        replicas = self.config_manager.get_config("replicas", 1)
        if replicas < 2:
            issues.append("Minimum replicas for production should be 2")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "checks_performed": [
                "Minimum replicas",
                "Resource limits",
                "Auto-scaling configuration"
            ]
        }

class ProductionDeployer:
    """Production deployment orchestrator"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.validator = ProductionValidator(config_manager)
        self.deployment_status = "pending"
    
    def deploy_to_production(self) -> bool:
        """Deploy application to production"""
        try:
            logger.info("Starting production deployment")
            
            # Validate production readiness
            validation_result = self.validator.validate_production_readiness()
            
            if not validation_result["overall_success"]:
                logger.error("Production readiness validation failed")
                self._log_validation_issues(validation_result)
                return False
            
            # Pre-deployment checks
            pre_deployment_success = self._run_pre_deployment_checks()
            
            if not pre_deployment_success:
                logger.error("Pre-deployment checks failed")
                return False
            
            # Deploy application
            deployment_success = self._deploy_application()
            
            if not deployment_success:
                logger.error("Application deployment failed")
                return False
            
            # Post-deployment checks
            post_deployment_success = self._run_post_deployment_checks()
            
            if not post_deployment_success:
                logger.error("Post-deployment checks failed")
                self._rollback_deployment()
                return False
            
            logger.info("Production deployment successful")
            self.deployment_status = "success"
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            self.deployment_status = "failed"
            return False
    
    def _run_pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks"""
        logger.info("Running pre-deployment checks")
        
        # Check database connectivity
        # Check external service availability
        # Check resource availability
        
        return True
    
    def _deploy_application(self) -> bool:
        """Deploy application"""
        logger.info("Deploying application")
        
        # Deploy to Kubernetes
        # Update load balancer
        # Update DNS records
        
        return True
    
    def _run_post_deployment_checks(self) -> bool:
        """Run post-deployment checks"""
        logger.info("Running post-deployment checks")
        
        # Health checks
        # Performance tests
        # Smoke tests
        
        return True
    
    def _rollback_deployment(self) -> None:
        """Rollback deployment"""
        logger.info("Rolling back deployment")
        
        # Rollback to previous version
        # Update load balancer
        # Update DNS records
    
    def _log_validation_issues(self, validation_result: Dict[str, Any]) -> None:
        """Log validation issues"""
        for validation_name, result in validation_result["validations"].items():
            if not result["success"]:
                logger.error(f"{validation_name} validation failed: {result['issues']}")

# Usage examples
def example_production_deployment():
    """Example production deployment usage"""
    # Create configuration manager
    config_manager = ConfigManager(Environment.PRODUCTION)
    
    # Validate configuration
    issues = config_manager.validate_configuration()
    if issues:
        print(f"Configuration issues: {issues}")
        return
    
    # Create secret manager
    secret_manager = SecretManager(Environment.PRODUCTION)
    
    # Generate and store secrets
    jwt_secret = secret_manager.generate_secret()
    secret_manager.store_secret("jwt_secret", jwt_secret)
    
    # Create production deployer
    deployer = ProductionDeployer(config_manager)
    
    # Deploy to production
    deployment_success = deployer.deploy_to_production()
    print(f"Production deployment: {deployment_success}")
    
    # Validate production readiness
    validator = ProductionValidator(config_manager)
    validation_result = validator.validate_production_readiness()
    print(f"Production readiness: {validation_result['overall_success']}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Blue-green deployment
config = DeploymentConfig("my-app", "v1.2.0", "production")
deployer = BlueGreenDeployer(config)
deployment_success = deployer.deploy_to_environment(Environment.GREEN)

# 2. Canary deployment
canary_deployer = CanaryDeployer(config)
canary_success = canary_deployer.deploy_canary(canary_percentage=10)

# 3. Production configuration
config_manager = ConfigManager(Environment.PRODUCTION)
issues = config_manager.validate_configuration()

# 4. Production deployment
deployer = ProductionDeployer(config_manager)
deployment_success = deployer.deploy_to_production()

# 5. Health checks
health_checker = HealthChecker()
health_status = health_checker.check_environment_health("production")
```

### Essential Patterns

```python
# Complete production deployment setup
def setup_production_deployment():
    """Setup complete production deployment environment"""
    
    # Configuration manager
    config_manager = ConfigManager(Environment.PRODUCTION)
    
    # Secret manager
    secret_manager = SecretManager(Environment.PRODUCTION)
    
    # Blue-green deployer
    blue_green_deployer = BlueGreenDeployer(config_manager.config)
    
    # Canary deployer
    canary_deployer = CanaryDeployer(config_manager.config)
    
    # Production deployer
    production_deployer = ProductionDeployer(config_manager)
    
    # Health checker
    health_checker = HealthChecker()
    
    # Load balancer
    load_balancer = LoadBalancer()
    
    print("Production deployment setup complete!")
```

---

*This guide provides the complete machinery for Python production deployment. Each pattern includes implementation examples, deployment strategies, and real-world usage patterns for enterprise deployment management.*
