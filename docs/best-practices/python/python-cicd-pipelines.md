# Python CI/CD Pipelines Best Practices

**Objective**: Master senior-level Python CI/CD pipeline patterns for production systems. When you need to build automated testing and deployment pipelines, when you want to implement comprehensive quality gates, when you need enterprise-grade CI/CD strategies—these best practices become your weapon of choice.

## Core Principles

- **Automation**: Automate all repetitive tasks
- **Quality**: Implement comprehensive quality gates
- **Speed**: Optimize for fast feedback loops
- **Reliability**: Ensure consistent and reliable deployments
- **Security**: Integrate security scanning and compliance

## GitHub Actions Workflows

### Comprehensive CI Pipeline

```python
# python/01-github-actions-ci.py

"""
GitHub Actions CI/CD pipeline patterns and automation
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    LINT = "lint"
    TEST = "test"
    BUILD = "build"
    SECURITY = "security"
    DEPLOY = "deploy"

class GitHubActionsBuilder:
    """GitHub Actions workflow builder"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.workflow_config = {}
        self.pipeline_metrics = {}
    
    def create_ci_workflow(self, python_version: str = "3.11") -> str:
        """Create comprehensive CI workflow"""
        workflow = {
            "name": "CI Pipeline",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main", "develop"]}
            },
            "env": {
                "PYTHON_VERSION": python_version,
                "POETRY_VERSION": "1.6.1"
            },
            "jobs": {
                "lint": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "${{ env.PYTHON_VERSION }}"}},
                        {"name": "Install Poetry", "uses": "snok/install-poetry@v1", "with": {"version": "${{ env.POETRY_VERSION }}"}},
                        {"name": "Configure Poetry", "run": "poetry config virtualenvs.create false"},
                        {"name": "Install dependencies", "run": "poetry install --no-interaction --no-ansi"},
                        {"name": "Run Black", "run": "poetry run black --check --diff ."},
                        {"name": "Run isort", "run": "poetry run isort --check-only --diff ."},
                        {"name": "Run Flake8", "run": "poetry run flake8 ."},
                        {"name": "Run MyPy", "run": "poetry run mypy ."}
                    ]
                },
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {"matrix": {"python-version": ["3.9", "3.10", "3.11"]}},
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "${{ matrix.python-version }}"}},
                        {"name": "Install Poetry", "uses": "snok/install-poetry@v1", "with": {"version": "${{ env.POETRY_VERSION }}"}},
                        {"name": "Configure Poetry", "run": "poetry config virtualenvs.create false"},
                        {"name": "Install dependencies", "run": "poetry install --no-interaction --no-ansi"},
                        {"name": "Run tests", "run": "poetry run pytest --cov=app --cov-report=xml --cov-report=html"},
                        {"name": "Upload coverage", "uses": "codecov/codecov-action@v3", "with": {"file": "./coverage.xml"}}
                    ]
                },
                "security": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "${{ env.PYTHON_VERSION }}"}},
                        {"name": "Install dependencies", "run": "pip install safety bandit"},
                        {"name": "Run Safety", "run": "safety check"},
                        {"name": "Run Bandit", "run": "bandit -r . -f json -o bandit-report.json"},
                        {"name": "Upload security report", "uses": "actions/upload-artifact@v3", "with": {"name": "security-report", "path": "bandit-report.json"}}
                    ]
                },
                "build": {
                    "runs-on": "ubuntu-latest",
                    "needs": ["lint", "test", "security"],
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Docker Buildx", "uses": "docker/setup-buildx-action@v3"},
                        {"name": "Login to Docker Hub", "uses": "docker/login-action@v3", "with": {"username": "${{ secrets.DOCKER_USERNAME }}", "password": "${{ secrets.DOCKER_PASSWORD }}"}},
                        {"name": "Build and push", "uses": "docker/build-push-action@v5", "with": {
                            "context": ".",
                            "push": True,
                            "tags": "${{ github.repository }}:${{ github.sha }},${{ github.repository }}:latest",
                            "cache-from": "type=gha",
                            "cache-to": "type=gha,mode=max"
                        }}
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def create_cd_workflow(self, environment: str = "production") -> str:
        """Create CD workflow for deployment"""
        workflow = {
            "name": f"CD Pipeline - {environment.title()}",
            "on": {
                "push": {"branches": ["main"]},
                "workflow_dispatch": {"inputs": {"environment": {"description": "Environment to deploy to", "required": True, "default": environment}}}
            },
            "env": {
                "ENVIRONMENT": environment,
                "REGISTRY": "ghcr.io"
            },
            "jobs": {
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "environment": environment,
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up kubectl", "uses": "azure/setup-kubectl@v3", "with": {"version": "latest"}},
                        {"name": "Configure AWS credentials", "uses": "aws-actions/configure-aws-credentials@v4", "with": {
                            "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                            "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                            "aws-region": "us-west-2"
                        }},
                        {"name": "Update kubeconfig", "run": "aws eks update-kubeconfig --region us-west-2 --name ${{ secrets.EKS_CLUSTER_NAME }}"},
                        {"name": "Deploy to Kubernetes", "run": f"kubectl apply -f k8s/{environment}/"},
                        {"name": "Wait for deployment", "run": "kubectl rollout status deployment/${{ github.event.repository.name }} -n ${{ env.ENVIRONMENT }}"},
                        {"name": "Run smoke tests", "run": "python scripts/smoke_tests.py --environment ${{ env.ENVIRONMENT }}"}
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def create_release_workflow(self) -> str:
        """Create release workflow"""
        workflow = {
            "name": "Release Pipeline",
            "on": {
                "push": {"tags": ["v*"]},
                "workflow_dispatch": {"inputs": {"version": {"description": "Version to release", "required": True, "type": "string"}}}
            },
            "jobs": {
                "release": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "3.11"}},
                        {"name": "Install Poetry", "uses": "snok/install-poetry@v1"},
                        {"name": "Configure Poetry", "run": "poetry config virtualenvs.create false"},
                        {"name": "Install dependencies", "run": "poetry install --no-interaction --no-ansi"},
                        {"name": "Build package", "run": "poetry build"},
                        {"name": "Publish to PyPI", "run": "poetry publish", "env": {"POETRY_PYPI_TOKEN_PYPI": "${{ secrets.PYPI_TOKEN }}"}},
                        {"name": "Create GitHub Release", "uses": "actions/create-release@v1", "env": {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}, "with": {
                            "tag_name": "${{ github.ref }}",
                            "release_name": "Release ${{ github.ref }}",
                            "draft": False,
                            "prerelease": False
                        }}
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def create_quality_gates(self) -> str:
        """Create quality gates workflow"""
        workflow = {
            "name": "Quality Gates",
            "on": {"pull_request": {"branches": ["main"]}},
            "jobs": {
                "quality-gates": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "3.11"}},
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {"name": "Run tests", "run": "pytest --cov=app --cov-fail-under=80"},
                        {"name": "Run security scan", "run": "bandit -r . -f json -o bandit-report.json"},
                        {"name": "Check dependencies", "run": "safety check"},
                        {"name": "Run performance tests", "run": "pytest tests/performance/ -v"},
                        {"name": "Upload quality report", "uses": "actions/upload-artifact@v3", "with": {"name": "quality-report", "path": "bandit-report.json"}}
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

class GitLabCIPipeline:
    """GitLab CI/CD pipeline builder"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.pipeline_config = {}
    
    def create_gitlab_ci(self) -> str:
        """Create GitLab CI configuration"""
        gitlab_ci = {
            "stages": ["lint", "test", "security", "build", "deploy"],
            "variables": {
                "PYTHON_VERSION": "3.11",
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.cache/pip"
            },
            "cache": {
                "paths": [".cache/pip", "venv/"]
            },
            "before_script": [
                "python -m venv venv",
                "source venv/bin/activate",
                "pip install --upgrade pip"
            ],
            "lint": {
                "stage": "lint",
                "script": [
                    "pip install black isort flake8 mypy",
                    "black --check --diff .",
                    "isort --check-only --diff .",
                    "flake8 .",
                    "mypy ."
                ],
                "artifacts": {"reports": {"junit": "junit.xml"}}
            },
            "test": {
                "stage": "test",
                "script": [
                    "pip install -r requirements.txt",
                    "pytest --cov=app --cov-report=xml --cov-report=html --junitxml=junit.xml"
                ],
                "coverage": "/TOTAL.*\\s+(\\d+%)$/",
                "artifacts": {
                    "reports": {
                        "junit": "junit.xml",
                        "cobertura": "coverage.xml"
                    },
                    "paths": ["htmlcov/"]
                }
            },
            "security": {
                "stage": "security",
                "script": [
                    "pip install safety bandit",
                    "safety check",
                    "bandit -r . -f json -o bandit-report.json"
                ],
                "artifacts": {"reports": {"sast": "bandit-report.json"}}
            },
            "build": {
                "stage": "build",
                "script": [
                    "docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .",
                    "docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA"
                ],
                "only": ["main", "develop"]
            },
            "deploy:staging": {
                "stage": "deploy",
                "script": [
                    "kubectl apply -f k8s/staging/",
                    "kubectl rollout status deployment/$CI_PROJECT_NAME -n staging"
                ],
                "environment": {"name": "staging", "url": "https://staging.example.com"},
                "only": ["develop"]
            },
            "deploy:production": {
                "stage": "deploy",
                "script": [
                    "kubectl apply -f k8s/production/",
                    "kubectl rollout status deployment/$CI_PROJECT_NAME -n production"
                ],
                "environment": {"name": "production", "url": "https://example.com"},
                "only": ["main"],
                "when": "manual"
            }
        }
        
        return yaml.dump(gitlab_ci, default_flow_style=False, sort_keys=False)

class JenkinsPipeline:
    """Jenkins pipeline builder"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.jenkins_config = {}
    
    def create_jenkinsfile(self) -> str:
        """Create Jenkinsfile for pipeline"""
        jenkinsfile = """pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
        DOCKER_REGISTRY = 'your-registry.com'
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Lint') {
            steps {
                sh 'python -m venv venv'
                sh 'source venv/bin/activate && pip install black isort flake8 mypy'
                sh 'source venv/bin/activate && black --check --diff .'
                sh 'source venv/bin/activate && isort --check-only --diff .'
                sh 'source venv/bin/activate && flake8 .'
                sh 'source venv/bin/activate && mypy .'
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'source venv/bin/activate && pytest tests/unit/ --cov=app --cov-report=xml'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'source venv/bin/activate && pytest tests/integration/ --cov=app --cov-report=xml'
                    }
                }
            }
            post {
                always {
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }
        
        stage('Security') {
            steps {
                sh 'source venv/bin/activate && pip install safety bandit'
                sh 'source venv/bin/activate && safety check'
                sh 'source venv/bin/activate && bandit -r . -f json -o bandit-report.json'
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'bandit-report.json',
                        reportName: 'Security Report'
                    ])
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${env.JOB_NAME}:${env.BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'kubectl apply -f k8s/production/'
                sh 'kubectl rollout status deployment/${JOB_NAME} -n production'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            emailext (
                subject: "Build Successful: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build ${env.BUILD_NUMBER} completed successfully.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build ${env.BUILD_NUMBER} failed. Please check the logs.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
"""
        return jenkinsfile

class PipelineOptimizer:
    """Pipeline optimization utilities"""
    
    def __init__(self):
        self.optimization_metrics = {}
    
    def analyze_pipeline_performance(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline performance metrics"""
        total_duration = 0
        stage_durations = {}
        
        for stage, data in pipeline_data.items():
            if isinstance(data, dict) and 'duration' in data:
                duration = data['duration']
                stage_durations[stage] = duration
                total_duration += duration
        
        # Calculate optimization opportunities
        optimization_suggestions = []
        
        if total_duration > 1800:  # 30 minutes
            optimization_suggestions.append("Consider parallelizing stages to reduce total duration")
        
        if any(duration > 600 for duration in stage_durations.values()):  # 10 minutes
            optimization_suggestions.append("Optimize slow stages with caching or better tooling")
        
        return {
            "total_duration": total_duration,
            "stage_durations": stage_durations,
            "optimization_suggestions": optimization_suggestions,
            "performance_score": self._calculate_performance_score(total_duration, stage_durations)
        }
    
    def _calculate_performance_score(self, total_duration: int, stage_durations: Dict[str, int]) -> float:
        """Calculate pipeline performance score (0-100)"""
        # Ideal total duration is under 10 minutes
        duration_score = max(0, 100 - (total_duration / 60) * 2)
        
        # Check for balanced stage durations
        if stage_durations:
            avg_duration = sum(stage_durations.values()) / len(stage_durations)
            variance = sum((duration - avg_duration) ** 2 for duration in stage_durations.values()) / len(stage_durations)
            balance_score = max(0, 100 - (variance / avg_duration) * 10)
        else:
            balance_score = 100
        
        return (duration_score + balance_score) / 2
    
    def optimize_pipeline_caching(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline with caching strategies"""
        optimized_config = pipeline_config.copy()
        
        # Add caching for dependencies
        if "jobs" in optimized_config:
            for job_name, job_config in optimized_config["jobs"].items():
                if "steps" in job_config:
                    # Add cache step for Python dependencies
                    cache_step = {
                        "name": "Cache dependencies",
                        "uses": "actions/cache@v3",
                        "with": {
                            "path": "~/.cache/pip",
                            "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}",
                            "restore-keys": "${{ runner.os }}-pip-"
                        }
                    }
                    
                    # Insert cache step after checkout
                    steps = job_config["steps"]
                    for i, step in enumerate(steps):
                        if step.get("uses") == "actions/checkout@v4":
                            steps.insert(i + 1, cache_step)
                            break
        
        return optimized_config
    
    def create_parallel_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create parallel job configuration"""
        parallel_jobs = {}
        
        for job in jobs:
            job_name = job["name"]
            job_config = job["config"]
            
            # Add parallel execution
            if "strategy" not in job_config:
                job_config["strategy"] = {"matrix": {"python-version": ["3.9", "3.10", "3.11"]}}
            
            parallel_jobs[job_name] = job_config
        
        return parallel_jobs

# Usage examples
def example_cicd_pipelines():
    """Example CI/CD pipeline usage"""
    # GitHub Actions
    github_builder = GitHubActionsBuilder("my-python-app")
    
    # Create CI workflow
    ci_workflow = github_builder.create_ci_workflow()
    print("GitHub Actions CI workflow created")
    
    # Create CD workflow
    cd_workflow = github_builder.create_cd_workflow("production")
    print("GitHub Actions CD workflow created")
    
    # Create release workflow
    release_workflow = github_builder.create_release_workflow()
    print("GitHub Actions release workflow created")
    
    # GitLab CI
    gitlab_builder = GitLabCIPipeline("my-python-app")
    gitlab_ci = gitlab_builder.create_gitlab_ci()
    print("GitLab CI configuration created")
    
    # Jenkins
    jenkins_builder = JenkinsPipeline("my-python-app")
    jenkinsfile = jenkins_builder.create_jenkinsfile()
    print("Jenkins pipeline created")
    
    # Pipeline optimization
    optimizer = PipelineOptimizer()
    
    # Analyze performance
    pipeline_data = {
        "lint": {"duration": 120},
        "test": {"duration": 300},
        "build": {"duration": 180},
        "deploy": {"duration": 240}
    }
    
    performance_analysis = optimizer.analyze_pipeline_performance(pipeline_data)
    print(f"Pipeline performance analysis: {performance_analysis}")
    
    # Optimize caching
    optimized_config = optimizer.optimize_pipeline_caching({"jobs": {}})
    print("Pipeline caching optimized")
```

### Advanced Pipeline Patterns

```python
# python/02-advanced-pipeline-patterns.py

"""
Advanced CI/CD pipeline patterns and automation strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import yaml
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdvancedPipelineBuilder:
    """Advanced pipeline builder with enterprise patterns"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.pipeline_templates = {}
    
    def create_multi_environment_pipeline(self) -> str:
        """Create multi-environment deployment pipeline"""
        pipeline = {
            "name": "Multi-Environment Pipeline",
            "on": {
                "push": {"branches": ["main", "develop", "feature/*"]},
                "pull_request": {"branches": ["main", "develop"]}
            },
            "jobs": {
                "determine-environment": {
                    "runs-on": "ubuntu-latest",
                    "outputs": {
                        "environment": "${{ steps.env.outputs.environment }}",
                        "deploy": "${{ steps.env.outputs.deploy }}"
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Determine Environment", "id": "env", "run": """
                            if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
                                echo "environment=production" >> $GITHUB_OUTPUT
                                echo "deploy=true" >> $GITHUB_OUTPUT
                            elif [[ "${{ github.ref }}" == "refs/heads/develop" ]]; then
                                echo "environment=staging" >> $GITHUB_OUTPUT
                                echo "deploy=true" >> $GITHUB_OUTPUT
                            else
                                echo "environment=development" >> $GITHUB_OUTPUT
                                echo "deploy=false" >> $GITHUB_OUTPUT
                            fi
                        """}
                    ]
                },
                "quality-gates": {
                    "runs-on": "ubuntu-latest",
                    "needs": "determine-environment",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "3.11"}},
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {"name": "Run tests", "run": "pytest --cov=app --cov-fail-under=80"},
                        {"name": "Run security scan", "run": "bandit -r . -f json -o bandit-report.json"},
                        {"name": "Run performance tests", "run": "pytest tests/performance/ -v"}
                    ]
                },
                "build-and-push": {
                    "runs-on": "ubuntu-latest",
                    "needs": ["determine-environment", "quality-gates"],
                    "if": "needs.determine-environment.outputs.deploy == 'true'",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up Docker Buildx", "uses": "docker/setup-buildx-action@v3"},
                        {"name": "Login to Registry", "uses": "docker/login-action@v3", "with": {
                            "registry": "ghcr.io",
                            "username": "${{ github.actor }}",
                            "password": "${{ secrets.GITHUB_TOKEN }}"
                        }},
                        {"name": "Build and push", "uses": "docker/build-push-action@v5", "with": {
                            "context": ".",
                            "push": True,
                            "tags": f"ghcr.io/${{{{ github.repository }}}}:${{{{ needs.determine-environment.outputs.environment }}}}-${{{{ github.sha }}}}}",
                            "cache-from": "type=gha",
                            "cache-to": "type=gha,mode=max"
                        }}
                    ]
                },
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "needs": ["build-and-push"],
                    "if": "needs.determine-environment.outputs.deploy == 'true'",
                    "environment": "${{ needs.determine-environment.outputs.environment }}",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up kubectl", "uses": "azure/setup-kubectl@v3"},
                        {"name": "Configure AWS credentials", "uses": "aws-actions/configure-aws-credentials@v4", "with": {
                            "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                            "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                            "aws-region": "us-west-2"
                        }},
                        {"name": "Update kubeconfig", "run": "aws eks update-kubeconfig --region us-west-2 --name ${{ secrets.EKS_CLUSTER_NAME }}"},
                        {"name": "Deploy", "run": f"kubectl apply -f k8s/${{{{ needs.determine-environment.outputs.environment }}}}/"},
                        {"name": "Wait for rollout", "run": "kubectl rollout status deployment/${{ github.event.repository.name }} -n ${{ needs.determine-environment.outputs.environment }}"},
                        {"name": "Run smoke tests", "run": "python scripts/smoke_tests.py --environment ${{ needs.determine-environment.outputs.environment }}"}
                    ]
                }
            }
        }
        
        return yaml.dump(pipeline, default_flow_style=False, sort_keys=False)
    
    def create_blue_green_deployment(self) -> str:
        """Create blue-green deployment pipeline"""
        pipeline = {
            "name": "Blue-Green Deployment",
            "on": {"workflow_dispatch": {"inputs": {"environment": {"description": "Environment", "required": True, "default": "production"}}}},
            "jobs": {
                "blue-green-deploy": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up kubectl", "uses": "azure/setup-kubectl@v3"},
                        {"name": "Configure AWS credentials", "uses": "aws-actions/configure-aws-credentials@v4", "with": {
                            "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                            "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                            "aws-region": "us-west-2"
                        }},
                        {"name": "Update kubeconfig", "run": "aws eks update-kubeconfig --region us-west-2 --name ${{ secrets.EKS_CLUSTER_NAME }}"},
                        {"name": "Determine current color", "id": "current-color", "run": """
                            CURRENT_COLOR=$(kubectl get service ${{ github.event.repository.name }}-service -o jsonpath='{.spec.selector.color}' || echo "blue")
                            echo "color=$CURRENT_COLOR" >> $GITHUB_OUTPUT
                            if [[ "$CURRENT_COLOR" == "blue" ]]; then
                                echo "new-color=green" >> $GITHUB_OUTPUT
                            else
                                echo "new-color=blue" >> $GITHUB_OUTPUT
                            fi
                        """},
                        {"name": "Deploy new version", "run": """
                            NEW_COLOR=${{ steps.current-color.outputs.new-color }}
                            kubectl apply -f k8s/${{ inputs.environment }}/deployment-$NEW_COLOR.yaml
                            kubectl rollout status deployment/${{ github.event.repository.name }}-$NEW_COLOR -n ${{ inputs.environment }}
                        """},
                        {"name": "Switch traffic", "run": """
                            NEW_COLOR=${{ steps.current-color.outputs.new-color }}
                            kubectl patch service ${{ github.event.repository.name }}-service -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'
                        """},
                        {"name": "Run health checks", "run": "python scripts/health_checks.py --environment ${{ inputs.environment }}"},
                        {"name": "Cleanup old version", "run": """
                            OLD_COLOR=${{ steps.current-color.outputs.color }}
                            kubectl delete deployment ${{ github.event.repository.name }}-$OLD_COLOR -n ${{ inputs.environment }}
                        """}
                    ]
                }
            }
        }
        
        return yaml.dump(pipeline, default_flow_style=False, sort_keys=False)
    
    def create_canary_deployment(self) -> str:
        """Create canary deployment pipeline"""
        pipeline = {
            "name": "Canary Deployment",
            "on": {"workflow_dispatch": {"inputs": {"environment": {"description": "Environment", "required": True, "default": "production"}}}},
            "jobs": {
                "canary-deploy": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {"name": "Set up kubectl", "uses": "azure/setup-kubectl@v3"},
                        {"name": "Configure AWS credentials", "uses": "aws-actions/configure-aws-credentials@v4", "with": {
                            "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                            "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                            "aws-region": "us-west-2"
                        }},
                        {"name": "Update kubeconfig", "run": "aws eks update-kubeconfig --region us-west-2 --name ${{ secrets.EKS_CLUSTER_NAME }}"},
                        {"name": "Deploy canary", "run": """
                            kubectl apply -f k8s/${{ inputs.environment }}/canary-deployment.yaml
                            kubectl rollout status deployment/${{ github.event.repository.name }}-canary -n ${{ inputs.environment }}
                        """},
                        {"name": "Run canary tests", "run": "python scripts/canary_tests.py --environment ${{ inputs.environment }}"},
                        {"name": "Monitor canary metrics", "run": "python scripts/monitor_canary.py --environment ${{ inputs.environment }} --duration 300"},
                        {"name": "Promote canary", "run": """
                            kubectl apply -f k8s/${{ inputs.environment }}/production-deployment.yaml
                            kubectl rollout status deployment/${{ github.event.repository.name }} -n ${{ inputs.environment }}
                        """},
                        {"name": "Cleanup canary", "run": "kubectl delete deployment ${{ github.event.repository.name }}-canary -n ${{ inputs.environment }}"}
                    ]
                }
            }
        }
        
        return yaml.dump(pipeline, default_flow_style=False, sort_keys=False)

class PipelineMonitoring:
    """Pipeline monitoring and alerting"""
    
    def __init__(self):
        self.monitoring_config = {}
    
    def create_monitoring_dashboard(self) -> str:
        """Create monitoring dashboard configuration"""
        dashboard = {
            "dashboard": {
                "title": "CI/CD Pipeline Monitoring",
                "panels": [
                    {
                        "title": "Pipeline Success Rate",
                        "type": "stat",
                        "targets": [
                            {"expr": "sum(rate(pipeline_runs_total{status=\"success\"}[5m])) / sum(rate(pipeline_runs_total[5m])) * 100"}
                        ]
                    },
                    {
                        "title": "Average Build Duration",
                        "type": "graph",
                        "targets": [
                            {"expr": "avg(pipeline_duration_seconds)"}
                        ]
                    },
                    {
                        "title": "Failed Pipelines",
                        "type": "table",
                        "targets": [
                            {"expr": "pipeline_runs_total{status=\"failure\"}"}
                        ]
                    }
                ]
            }
        }
        
        return json.dumps(dashboard, indent=2)
    
    def create_alerting_rules(self) -> str:
        """Create alerting rules for pipeline monitoring"""
        rules = {
            "groups": [
                {
                    "name": "pipeline_alerts",
                    "rules": [
                        {
                            "alert": "PipelineFailureRate",
                            "expr": "sum(rate(pipeline_runs_total{status=\"failure\"}[5m])) / sum(rate(pipeline_runs_total[5m])) > 0.1",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "High pipeline failure rate detected"}
                        },
                        {
                            "alert": "PipelineDurationHigh",
                            "expr": "avg(pipeline_duration_seconds) > 1800",
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "Pipeline duration is unusually high"}
                        }
                    ]
                }
            ]
        }
        
        return yaml.dump(rules, default_flow_style=False, sort_keys=False)

# Usage examples
def example_advanced_pipelines():
    """Example advanced pipeline usage"""
    # Advanced pipeline builder
    advanced_builder = AdvancedPipelineBuilder("my-python-app")
    
    # Multi-environment pipeline
    multi_env_pipeline = advanced_builder.create_multi_environment_pipeline()
    print("Multi-environment pipeline created")
    
    # Blue-green deployment
    blue_green_pipeline = advanced_builder.create_blue_green_deployment()
    print("Blue-green deployment pipeline created")
    
    # Canary deployment
    canary_pipeline = advanced_builder.create_canary_deployment()
    print("Canary deployment pipeline created")
    
    # Pipeline monitoring
    monitoring = PipelineMonitoring()
    
    # Monitoring dashboard
    dashboard = monitoring.create_monitoring_dashboard()
    print("Monitoring dashboard created")
    
    # Alerting rules
    alerting_rules = monitoring.create_alerting_rules()
    print("Alerting rules created")
```

## TL;DR Runbook

### Quick Start

```python
# 1. GitHub Actions CI
github_builder = GitHubActionsBuilder("my-app")
ci_workflow = github_builder.create_ci_workflow()
cd_workflow = github_builder.create_cd_workflow("production")

# 2. GitLab CI
gitlab_builder = GitLabCIPipeline("my-app")
gitlab_ci = gitlab_builder.create_gitlab_ci()

# 3. Jenkins Pipeline
jenkins_builder = JenkinsPipeline("my-app")
jenkinsfile = jenkins_builder.create_jenkinsfile()

# 4. Advanced patterns
advanced_builder = AdvancedPipelineBuilder("my-app")
multi_env_pipeline = advanced_builder.create_multi_environment_pipeline()
blue_green_pipeline = advanced_builder.create_blue_green_deployment()

# 5. Pipeline optimization
optimizer = PipelineOptimizer()
performance_analysis = optimizer.analyze_pipeline_performance(pipeline_data)
optimized_config = optimizer.optimize_pipeline_caching(pipeline_config)
```

### Essential Patterns

```python
# Complete CI/CD setup
def setup_cicd_pipelines():
    """Setup complete CI/CD pipeline environment"""
    
    # GitHub Actions builder
    github_builder = GitHubActionsBuilder("my-app")
    
    # GitLab CI builder
    gitlab_builder = GitLabCIPipeline("my-app")
    
    # Jenkins builder
    jenkins_builder = JenkinsPipeline("my-app")
    
    # Advanced pipeline builder
    advanced_builder = AdvancedPipelineBuilder("my-app")
    
    # Pipeline optimizer
    optimizer = PipelineOptimizer()
    
    # Pipeline monitoring
    monitoring = PipelineMonitoring()
    
    print("CI/CD pipeline setup complete!")
```

---

*This guide provides the complete machinery for Python CI/CD pipelines. Each pattern includes implementation examples, automation strategies, and real-world usage patterns for enterprise pipeline management.*
