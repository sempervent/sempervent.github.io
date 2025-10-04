# Python Containerization Best Practices

**Objective**: Master senior-level Python containerization patterns for production systems. When you need to build efficient Docker images, when you want to implement multi-stage builds, when you need enterprise-grade container optimization—these best practices become your weapon of choice.

## Core Principles

- **Efficiency**: Minimize image size and build time
- **Security**: Implement security best practices
- **Reproducibility**: Ensure consistent builds across environments
- **Performance**: Optimize for runtime performance
- **Maintainability**: Keep containers maintainable and debuggable

## Docker Image Optimization

### Multi-Stage Builds

```python
# python/01-multi-stage-builds.py

"""
Multi-stage Docker build patterns and optimization strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
import json
import subprocess
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildStage(Enum):
    """Docker build stage enumeration"""
    BASE = "base"
    DEPENDENCIES = "dependencies"
    APPLICATION = "application"
    RUNTIME = "runtime"
    TESTING = "testing"

class DockerImageBuilder:
    """Docker image builder with optimization patterns"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.build_config = {}
        self.build_metrics = {}
    
    def create_multi_stage_dockerfile(self, python_version: str = "3.11",
                                    base_image: str = "python:3.11-slim") -> str:
        """Create optimized multi-stage Dockerfile"""
        dockerfile_content = f"""# Multi-stage Dockerfile for {self.project_name}

# Stage 1: Base image with system dependencies
FROM {base_image} as base
LABEL stage="base"
LABEL maintainer="your-email@example.com"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies
LABEL stage="dependencies"

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt && \\
    pip install -r requirements-dev.txt

# Stage 3: Application
FROM dependencies as application
LABEL stage="application"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Install application in development mode
RUN pip install -e .

# Stage 4: Runtime
FROM {base_image} as runtime
LABEL stage="runtime"

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser --from=application /app .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Testing (optional)
FROM application as testing
LABEL stage="testing"

# Install testing dependencies
RUN pip install pytest pytest-cov pytest-asyncio

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=app", "--cov-report=html"]
"""
        return dockerfile_content
    
    def create_requirements_files(self) -> Tuple[str, str]:
        """Create optimized requirements files"""
        requirements_txt = """# Production dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
redis==5.0.1
celery==5.3.4
prometheus-client==0.19.0
structlog==23.2.0
"""
        
        requirements_dev_txt = """# Development dependencies
-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Code quality
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Development tools
ipython==8.17.2
jupyter==1.0.0
pre-commit==3.6.0
"""
        
        return requirements_txt, requirements_dev_txt
    
    def create_dockerignore(self) -> str:
        """Create optimized .dockerignore file"""
        dockerignore_content = """# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Virtual environments
venv/
ENV/
env/
.venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Documentation
docs/_build/
*.md
!README.md

# Testing
tests/
.pytest_cache/
htmlcov/

# Development
.env
.env.local
.env.development
.env.test
.env.production

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
"""
        return dockerignore_content
    
    def build_image(self, tag: str, build_args: Dict[str, str] = None) -> bool:
        """Build Docker image with optimization"""
        try:
            # Build command with optimization flags
            cmd = [
                "docker", "build",
                "--tag", tag,
                "--build-arg", "BUILDKIT_INLINE_CACHE=1",
                "--progress", "plain"
            ]
            
            if build_args:
                for key, value in build_args.items():
                    cmd.extend(["--build-arg", f"{key}={value}"])
            
            cmd.append(".")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                self.build_metrics[tag] = {
                    "build_time": build_time,
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.info(f"Successfully built image {tag} in {build_time:.2f}s")
                return True
            else:
                logger.error(f"Failed to build image {tag}: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error building image {tag}: {e}")
            return False
    
    def optimize_image_size(self, image_tag: str) -> Dict[str, Any]:
        """Analyze and optimize image size"""
        try:
            # Get image size
            size_cmd = ["docker", "images", "--format", "table {{.Size}}", image_tag]
            size_result = subprocess.run(size_cmd, capture_output=True, text=True)
            
            # Get image layers
            history_cmd = ["docker", "history", "--format", "table {{.Size}}\\t{{.CreatedBy}}", image_tag]
            history_result = subprocess.run(history_cmd, capture_output=True, text=True)
            
            optimization_suggestions = []
            
            # Analyze layers for optimization opportunities
            if "apt-get update" in history_result.stdout:
                optimization_suggestions.append("Consider combining RUN commands to reduce layers")
            
            if "pip install" in history_result.stdout:
                optimization_suggestions.append("Use multi-stage build to separate build and runtime dependencies")
            
            if "COPY ." in history_result.stdout:
                optimization_suggestions.append("Use .dockerignore to exclude unnecessary files")
            
            return {
                "image_size": size_result.stdout.strip(),
                "optimization_suggestions": optimization_suggestions,
                "layer_count": len(history_result.stdout.strip().split('\n')) - 1
            }
        
        except Exception as e:
            logger.error(f"Error analyzing image {image_tag}: {e}")
            return {}

class ContainerSecurity:
    """Container security best practices"""
    
    def __init__(self):
        self.security_metrics = {}
    
    def create_secure_dockerfile(self, base_image: str = "python:3.11-slim") -> str:
        """Create security-hardened Dockerfile"""
        secure_dockerfile = f"""# Security-hardened Dockerfile
FROM {base_image} as base

# Set security environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    PIP_ROOT_USER_ACTION=ignore

# Install security updates
RUN apt-get update && \\
    apt-get upgrade -y && \\
    apt-get install -y --no-install-recommends \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Create non-root user with specific UID/GID
RUN groupadd -r appuser -g 1000 && \\
    useradd -r -g appuser -u 1000 -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \\
    find /app -name "__pycache__" -delete

# Set proper permissions
RUN chown -R appuser:appuser /app && \\
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Use exec form for better signal handling
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return secure_dockerfile
    
    def scan_image_vulnerabilities(self, image_tag: str) -> Dict[str, Any]:
        """Scan Docker image for vulnerabilities"""
        try:
            # Use trivy for vulnerability scanning
            scan_cmd = ["trivy", "image", "--format", "json", image_tag]
            result = subprocess.run(scan_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                scan_data = json.loads(result.stdout)
                vulnerabilities = scan_data.get("Results", [])
                
                high_severity = 0
                medium_severity = 0
                low_severity = 0
                
                for result_item in vulnerabilities:
                    for vuln in result_item.get("Vulnerabilities", []):
                        severity = vuln.get("Severity", "").lower()
                        if severity == "high":
                            high_severity += 1
                        elif severity == "medium":
                            medium_severity += 1
                        elif severity == "low":
                            low_severity += 1
                
                return {
                    "total_vulnerabilities": high_severity + medium_severity + low_severity,
                    "high_severity": high_severity,
                    "medium_severity": medium_severity,
                    "low_severity": low_severity,
                    "scan_successful": True
                }
            else:
                return {"scan_successful": False, "error": result.stderr}
        
        except Exception as e:
            logger.error(f"Error scanning image {image_tag}: {e}")
            return {"scan_successful": False, "error": str(e)}
    
    def create_security_policy(self) -> str:
        """Create container security policy"""
        policy = """# Container Security Policy

## Image Security
- Use official base images from trusted registries
- Regularly update base images for security patches
- Scan images for vulnerabilities before deployment
- Use multi-stage builds to minimize attack surface

## Runtime Security
- Run containers as non-root user
- Use read-only filesystems where possible
- Limit container capabilities
- Use security contexts and pod security policies

## Network Security
- Use network policies to restrict traffic
- Encrypt traffic between containers
- Use service mesh for advanced networking

## Secrets Management
- Never store secrets in images
- Use external secret management systems
- Rotate secrets regularly
- Use least privilege principle

## Monitoring and Logging
- Monitor container behavior
- Log security events
- Use runtime security tools
- Implement audit logging
"""
        return policy

class ContainerOrchestration:
    """Container orchestration patterns"""
    
    def __init__(self):
        self.orchestration_metrics = {}
    
    def create_docker_compose(self, services: List[Dict[str, Any]]) -> str:
        """Create optimized Docker Compose configuration"""
        compose_content = """version: '3.8'

services:
"""
        
        for service in services:
            service_name = service["name"]
            service_config = service["config"]
            
            compose_content += f"""  {service_name}:
    build:
      context: {service_config.get('context', '.')}
      dockerfile: {service_config.get('dockerfile', 'Dockerfile')}
    image: {service_config.get('image', f'{service_name}:latest')}
    container_name: {service_name}
    restart: {service_config.get('restart', 'unless-stopped')}
    environment:
"""
            
            # Add environment variables
            for env_var in service_config.get('environment', []):
                compose_content += f"      - {env_var}\n"
            
            # Add ports
            if 'ports' in service_config:
                compose_content += "    ports:\n"
                for port in service_config['ports']:
                    compose_content += f"      - {port}\n"
            
            # Add volumes
            if 'volumes' in service_config:
                compose_content += "    volumes:\n"
                for volume in service_config['volumes']:
                    compose_content += f"      - {volume}\n"
            
            # Add networks
            if 'networks' in service_config:
                compose_content += "    networks:\n"
                for network in service_config['networks']:
                    compose_content += f"      - {network}\n"
            
            # Add health check
            if 'healthcheck' in service_config:
                healthcheck = service_config['healthcheck']
                compose_content += f"""    healthcheck:
      test: {healthcheck['test']}
      interval: {healthcheck.get('interval', '30s')}
      timeout: {healthcheck.get('timeout', '10s')}
      retries: {healthcheck.get('retries', 3)}
"""
            
            compose_content += "\n"
        
        # Add networks
        compose_content += """networks:
  default:
    driver: bridge

volumes:
  app_data:
    driver: local
"""
        
        return compose_content
    
    def create_kubernetes_manifests(self, app_name: str, 
                                  image: str, 
                                  replicas: int = 3) -> Dict[str, str]:
        """Create Kubernetes deployment manifests"""
        
        # Deployment manifest
        deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {image}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
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
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
"""
        
        # Service manifest
        service = f"""apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
  labels:
    app: {app_name}
spec:
  selector:
    app: {app_name}
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
"""
        
        # Ingress manifest
        ingress = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: {app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}-service
            port:
              number: 80
"""
        
        return {
            "deployment.yaml": deployment,
            "service.yaml": service,
            "ingress.yaml": ingress
        }
    
    def create_helm_chart(self, app_name: str) -> Dict[str, str]:
        """Create Helm chart for application"""
        
        # Chart.yaml
        chart_yaml = f"""apiVersion: v2
name: {app_name}
description: A Helm chart for {app_name}
type: application
version: 0.1.0
appVersion: "1.0.0"
"""
        
        # values.yaml
        values_yaml = f"""replicaCount: 3

image:
  repository: {app_name}
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
  hosts:
    - host: {app_name}.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

nodeSelector: {{}}

tolerations: []

affinity: {{}}
"""
        
        # templates/deployment.yaml
        deployment_template = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ include "{app_name}.fullname" . }}}}
  labels:
    {{{{ include "{app_name}.labels" . }}}}
spec:
  {{{{ if not .Values.autoscaling.enabled }}}}
  replicas: {{{{ .Values.replicaCount }}}}
  {{{{ end }}}}
  selector:
    matchLabels:
      {{{{ include "{app_name}.selectorLabels" . }}}}
  template:
    metadata:
      {{{{ include "{app_name}.podAnnotations" . }}}}
      labels:
        {{{{ include "{app_name}.podLabels" . }}}}
    spec:
      containers:
        - name: {{{{ .Chart.Name }}}}
          image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag | default .Chart.AppVersion }}}}"
          imagePullPolicy: {{{{ .Values.image.pullPolicy }}}}
          ports:
            - name: http
              containerPort: {{{{ .Values.service.targetPort }}}}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
          readinessProbe:
            httpGet:
              path: /ready
              port: http
          resources:
            {{{{ toYaml .Values.resources | nindent 12 }}}}
      {{{{ with .Values.nodeSelector }}}}
      nodeSelector:
        {{{{ toYaml . | nindent 8 }}}}
      {{{{ end }}}}
      {{{{ with .Values.affinity }}}}
      affinity:
        {{{{ toYaml . | nindent 8 }}}}
      {{{{ end }}}}
      {{{{ with .Values.tolerations }}}}
      tolerations:
        {{{{ toYaml . | nindent 8 }}}}
      {{{{ end }}}}
"""
        
        return {
            "Chart.yaml": chart_yaml,
            "values.yaml": values_yaml,
            "templates/deployment.yaml": deployment_template
        }

# Usage examples
def example_containerization():
    """Example containerization usage"""
    # Create Docker image builder
    builder = DockerImageBuilder("my-python-app")
    
    # Create multi-stage Dockerfile
    dockerfile = builder.create_multi_stage_dockerfile()
    print("Multi-stage Dockerfile created")
    
    # Create requirements files
    requirements, requirements_dev = builder.create_requirements_files()
    print("Requirements files created")
    
    # Create .dockerignore
    dockerignore = builder.create_dockerignore()
    print(".dockerignore created")
    
    # Build image
    build_success = builder.build_image("my-python-app:latest")
    print(f"Image build successful: {build_success}")
    
    # Optimize image size
    optimization = builder.optimize_image_size("my-python-app:latest")
    print(f"Image optimization suggestions: {optimization}")
    
    # Container security
    security = ContainerSecurity()
    
    # Create secure Dockerfile
    secure_dockerfile = security.create_secure_dockerfile()
    print("Secure Dockerfile created")
    
    # Scan for vulnerabilities
    vulnerabilities = security.scan_image_vulnerabilities("my-python-app:latest")
    print(f"Vulnerability scan results: {vulnerabilities}")
    
    # Container orchestration
    orchestration = ContainerOrchestration()
    
    # Create Docker Compose
    services = [
        {
            "name": "app",
            "config": {
                "context": ".",
                "ports": ["8000:8000"],
                "environment": ["ENV=production"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
        }
    ]
    
    compose_config = orchestration.create_docker_compose(services)
    print("Docker Compose configuration created")
    
    # Create Kubernetes manifests
    k8s_manifests = orchestration.create_kubernetes_manifests("my-app", "my-app:latest")
    print("Kubernetes manifests created")
    
    # Create Helm chart
    helm_chart = orchestration.create_helm_chart("my-app")
    print("Helm chart created")
```

### Container Monitoring

```python
# python/02-container-monitoring.py

"""
Container monitoring and observability patterns
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import json
import time
import subprocess
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ContainerMonitor:
    """Container monitoring utilities"""
    
    def __init__(self):
        self.monitoring_metrics = {}
    
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get container statistics"""
        try:
            # Get container stats
            stats_cmd = ["docker", "stats", "--no-stream", "--format", "json", container_name]
            result = subprocess.run(stats_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                stats_data = json.loads(result.stdout)
                
                # Parse memory usage
                memory_usage = stats_data.get("MemUsage", "0B / 0B")
                memory_used, memory_limit = memory_usage.split(" / ")
                
                # Parse CPU usage
                cpu_percent = stats_data.get("CPUPerc", "0%").replace("%", "")
                
                return {
                    "container_name": container_name,
                    "cpu_percent": float(cpu_percent),
                    "memory_used": memory_used,
                    "memory_limit": memory_limit,
                    "memory_percent": stats_data.get("MemPerc", "0%"),
                    "network_io": stats_data.get("NetIO", "0B / 0B"),
                    "block_io": stats_data.get("BlockIO", "0B / 0B"),
                    "pids": int(stats_data.get("PIDs", 0)),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"error": f"Failed to get stats: {result.stderr}"}
        
        except Exception as e:
            logger.error(f"Error getting container stats: {e}")
            return {"error": str(e)}
    
    def get_container_logs(self, container_name: str, 
                          lines: int = 100) -> List[str]:
        """Get container logs"""
        try:
            logs_cmd = ["docker", "logs", "--tail", str(lines), container_name]
            result = subprocess.run(logs_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.split('\n')
            else:
                return [f"Error getting logs: {result.stderr}"]
        
        except Exception as e:
            logger.error(f"Error getting container logs: {e}")
            return [f"Error: {str(e)}"]
    
    def check_container_health(self, container_name: str) -> Dict[str, Any]:
        """Check container health status"""
        try:
            # Get container inspect
            inspect_cmd = ["docker", "inspect", container_name]
            result = subprocess.run(inspect_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                inspect_data = json.loads(result.stdout)[0]
                
                # Get health status
                health_status = inspect_data.get("State", {}).get("Health", {})
                status = health_status.get("Status", "unknown")
                
                # Get container state
                container_state = inspect_data.get("State", {})
                running = container_state.get("Running", False)
                restart_count = container_state.get("RestartCount", 0)
                
                return {
                    "container_name": container_name,
                    "status": status,
                    "running": running,
                    "restart_count": restart_count,
                    "health_status": health_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"error": f"Failed to inspect container: {result.stderr}"}
        
        except Exception as e:
            logger.error(f"Error checking container health: {e}")
            return {"error": str(e)}
    
    def monitor_container_performance(self, container_name: str, 
                                    duration: int = 60) -> Dict[str, Any]:
        """Monitor container performance over time"""
        start_time = time.time()
        performance_data = []
        
        while time.time() - start_time < duration:
            stats = self.get_container_stats(container_name)
            if "error" not in stats:
                performance_data.append(stats)
            
            time.sleep(10)  # Sample every 10 seconds
        
        if performance_data:
            # Calculate averages
            avg_cpu = sum(item["cpu_percent"] for item in performance_data) / len(performance_data)
            avg_memory = sum(float(item["memory_percent"].replace("%", "")) 
                           for item in performance_data) / len(performance_data)
            
            return {
                "container_name": container_name,
                "monitoring_duration": duration,
                "samples": len(performance_data),
                "average_cpu": avg_cpu,
                "average_memory": avg_memory,
                "performance_data": performance_data
            }
        else:
            return {"error": "No performance data collected"}

class ContainerOptimizer:
    """Container optimization utilities"""
    
    def __init__(self):
        self.optimization_metrics = {}
    
    def analyze_image_layers(self, image_tag: str) -> Dict[str, Any]:
        """Analyze Docker image layers for optimization"""
        try:
            # Get image history
            history_cmd = ["docker", "history", "--format", "json", image_tag]
            result = subprocess.run(history_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                layers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        layer_data = json.loads(line)
                        layers.append(layer_data)
                
                # Analyze layers
                total_size = 0
                optimization_suggestions = []
                
                for layer in layers:
                    size_str = layer.get("Size", "0B")
                    if size_str != "0B":
                        # Parse size (simplified)
                        size_value = int(size_str.replace("B", "").replace("MB", "000000").replace("KB", "000"))
                        total_size += size_value
                
                # Check for optimization opportunities
                for layer in layers:
                    created_by = layer.get("CreatedBy", "")
                    if "apt-get update" in created_by and "apt-get install" in created_by:
                        optimization_suggestions.append("Combine apt-get commands to reduce layers")
                    if "pip install" in created_by and "pip install" in created_by:
                        optimization_suggestions.append("Combine pip install commands")
                    if "COPY ." in created_by:
                        optimization_suggestions.append("Use .dockerignore to exclude unnecessary files")
                
                return {
                    "total_layers": len(layers),
                    "total_size": total_size,
                    "optimization_suggestions": optimization_suggestions,
                    "layers": layers
                }
            else:
                return {"error": f"Failed to analyze layers: {result.stderr}"}
        
        except Exception as e:
            logger.error(f"Error analyzing image layers: {e}")
            return {"error": str(e)}
    
    def optimize_container_resources(self, container_name: str) -> Dict[str, Any]:
        """Optimize container resource usage"""
        try:
            # Get current resource usage
            stats = self.get_container_stats(container_name)
            
            if "error" in stats:
                return stats
            
            # Analyze resource usage
            cpu_percent = stats["cpu_percent"]
            memory_percent = float(stats["memory_percent"].replace("%", ""))
            
            recommendations = []
            
            if cpu_percent > 80:
                recommendations.append("High CPU usage detected - consider scaling or optimization")
            elif cpu_percent < 20:
                recommendations.append("Low CPU usage - consider reducing CPU limits")
            
            if memory_percent > 80:
                recommendations.append("High memory usage detected - consider increasing memory limits")
            elif memory_percent < 20:
                recommendations.append("Low memory usage - consider reducing memory limits")
            
            return {
                "container_name": container_name,
                "current_cpu": cpu_percent,
                "current_memory": memory_percent,
                "recommendations": recommendations,
                "optimization_score": self._calculate_optimization_score(cpu_percent, memory_percent)
            }
        
        except Exception as e:
            logger.error(f"Error optimizing container resources: {e}")
            return {"error": str(e)}
    
    def _calculate_optimization_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate optimization score (0-100)"""
        # Ideal usage is around 50-70%
        cpu_score = 100 - abs(cpu_percent - 60) * 2
        memory_score = 100 - abs(memory_percent - 60) * 2
        
        return max(0, min(100, (cpu_score + memory_score) / 2))

# Usage examples
def example_container_monitoring():
    """Example container monitoring usage"""
    # Create container monitor
    monitor = ContainerMonitor()
    
    # Get container stats
    stats = monitor.get_container_stats("my-container")
    print(f"Container stats: {stats}")
    
    # Get container logs
    logs = monitor.get_container_logs("my-container", lines=50)
    print(f"Container logs: {len(logs)} lines")
    
    # Check container health
    health = monitor.check_container_health("my-container")
    print(f"Container health: {health}")
    
    # Monitor performance
    performance = monitor.monitor_container_performance("my-container", duration=60)
    print(f"Performance monitoring: {performance}")
    
    # Container optimization
    optimizer = ContainerOptimizer()
    
    # Analyze image layers
    layer_analysis = optimizer.analyze_image_layers("my-app:latest")
    print(f"Layer analysis: {layer_analysis}")
    
    # Optimize resources
    resource_optimization = optimizer.optimize_container_resources("my-container")
    print(f"Resource optimization: {resource_optimization}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Multi-stage Docker build
builder = DockerImageBuilder("my-app")
dockerfile = builder.create_multi_stage_dockerfile()
build_success = builder.build_image("my-app:latest")

# 2. Container security
security = ContainerSecurity()
secure_dockerfile = security.create_secure_dockerfile()
vulnerabilities = security.scan_image_vulnerabilities("my-app:latest")

# 3. Container orchestration
orchestration = ContainerOrchestration()
compose_config = orchestration.create_docker_compose(services)
k8s_manifests = orchestration.create_kubernetes_manifests("my-app", "my-app:latest")

# 4. Container monitoring
monitor = ContainerMonitor()
stats = monitor.get_container_stats("my-container")
health = monitor.check_container_health("my-container")

# 5. Container optimization
optimizer = ContainerOptimizer()
layer_analysis = optimizer.analyze_image_layers("my-app:latest")
resource_optimization = optimizer.optimize_container_resources("my-container")
```

### Essential Patterns

```python
# Complete containerization setup
def setup_containerization():
    """Setup complete containerization environment"""
    
    # Docker image builder
    builder = DockerImageBuilder("my-app")
    
    # Container security
    security = ContainerSecurity()
    
    # Container orchestration
    orchestration = ContainerOrchestration()
    
    # Container monitoring
    monitor = ContainerMonitor()
    
    # Container optimization
    optimizer = ContainerOptimizer()
    
    print("Containerization setup complete!")
```

---

*This guide provides the complete machinery for Python containerization. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise container management.*
