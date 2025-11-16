# Repository Standardization, Templates, and Lifecycle Governance: Best Practices for Polyglot Ecosystems

**Objective**: Master production-grade repository standardization across Python, Go, Rust, Docker, Kubernetes, and data pipelines. When you need to eliminate snowflake repos, ensure consistency, and enable automated governance—this guide provides complete patterns and implementations.

## Introduction

Standardized repository architecture is the foundation of scalable, maintainable software ecosystems. Without standards, every repository becomes a unique snowflake, onboarding slows, security hardens inconsistently, and technical debt accumulates. This guide provides a complete framework for standardizing repositories across languages, frameworks, and deployment targets.

**What This Guide Covers**:
- Philosophy and rationale for standardization
- Repository taxonomy and canonical layouts
- Template designs for all project types
- Unified versioning strategies
- Cross-project governance automation
- Lifecycle governance and aging control
- Multi-environment release management
- Cross-repo observability
- Anti-patterns and recovery strategies
- End-to-end examples with agentic LLM integration

**Prerequisites**:
- Understanding of Git, CI/CD, and software development workflows
- Familiarity with multiple programming languages and frameworks
- Experience with containerization and infrastructure as code

## Philosophy: Why Standard Repo Architecture Matters

### Reproducibility

**Problem**: Without standards, each repository has unique setup procedures, making it impossible to reliably reproduce environments.

**Solution**: Standardized repositories enable:
- Automated environment setup
- Consistent development workflows
- Predictable CI/CD pipelines
- Reproducible builds across teams

### Developer Onboarding

**Problem**: New developers spend days understanding unique repository structures.

**Solution**: Standard layouts mean:
- Instant familiarity with any repository
- Reduced onboarding time from days to hours
- Consistent tooling and commands
- Predictable file locations

### Security Hardening

**Problem**: Inconsistent security practices across repositories.

**Solution**: Standardized security means:
- Uniform dependency scanning
- Consistent secret management
- Standardized SBOM generation
- Automated security policy enforcement

### Reducing Configuration Drift

**Problem**: Repositories drift from intended structure over time.

**Solution**: Standardization enables:
- Automated drift detection
- Template-based repository creation
- Continuous compliance checking
- Automated remediation

### Eliminating One-Off Snowflake Repos

**Problem**: Every repository is unique, making automation impossible.

**Solution**: Standardization provides:
- Reusable automation scripts
- Template-based generation
- Consistent tooling
- Predictable structures

### Automated Documentation Generation

**Problem**: Documentation is inconsistent or missing.

**Solution**: Standardized repositories enable:
- Automated API documentation
- Consistent README generation
- Standardized changelog formats
- Automated dependency documentation

```mermaid
graph TB
    subgraph Problems["Problems Without Standards"]
        P1["Unique Setup<br/>Procedures"]
        P2["Long Onboarding"]
        P3["Inconsistent<br/>Security"]
        P4["Configuration<br/>Drift"]
        P5["Snowflake<br/>Repos"]
        P6["Missing Docs"]
    end
    
    subgraph Solutions["Solutions With Standards"]
        S1["Automated Setup"]
        S2["Fast Onboarding"]
        S3["Uniform Security"]
        S4["Drift Detection"]
        S5["Template-Based"]
        S6["Auto-Generated Docs"]
    end
    
    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4
    P5 --> S5
    P6 --> S6
    
    style Problems fill:#ffebee
    style Solutions fill:#e8f5e9
```

## Core Repository Taxonomy

### Application Service Repos

**Structure**:
```
service-name/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── cmd/
│   └── service-name/
│       └── main.go          # or main.py, main.rs
├── internal/
│   ├── handlers/
│   ├── models/
│   └── services/
├── pkg/                      # Public packages (Go) or src/ (Rust)
├── api/
│   └── openapi.yaml
├── migrations/               # Database migrations
├── deployments/
│   ├── k8s/
│   ├── helm/
│   └── docker-compose.yml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   └── README.md
├── .pre-commit-config.yaml
├── Makefile
├── Dockerfile
├── docker-bake.hcl
├── go.mod                    # or requirements.txt, Cargo.toml
├── CHANGES.md
└── README.md
```

**When to Use**: Microservices, APIs, backend services

### Library/Package Repos

**Structure**:
```
package-name/
├── .github/
│   └── workflows/
│       └── release.yml
├── src/                      # or lib/, pkg/
│   └── package_name/
│       ├── __init__.py
│       └── core.py
├── tests/
│   └── test_core.py
├── docs/
│   ├── api/
│   └── examples/
├── examples/
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml            # or go.mod, Cargo.toml
├── CHANGES.md
└── README.md
```

**When to Use**: Reusable libraries, shared utilities, common packages

### Data/ETL Repos

**Structure**:
```
etl-pipeline-name/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── pipelines/
│   ├── extract/
│   ├── transform/
│   └── load/
├── schemas/
│   ├── input/
│   ├── output/
│   └── contracts/
├── tests/
│   ├── fixtures/
│   └── test_pipelines.py
├── prefect/
│   └── flows/
├── dbt/                     # If using dbt
│   └── models/
├── data/
│   └── samples/
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── CHANGES.md
└── README.md
```

**When to Use**: ETL pipelines, data transformations, Prefect/Dask workflows

### Infrastructure/IaC Repos

**Structure**:
```
infra-name/
├── .github/
│   └── workflows/
│       └── terraform.yml
├── terraform/                # or ansible/, pulumi/
│   ├── modules/
│   ├── environments/
│   │   ├── dev/
│   │   ├── stage/
│   │   └── prod/
│   └── main.tf
├── ansible/
│   ├── roles/
│   ├── playbooks/
│   └── inventory/
├── k8s/
│   └── base/
├── helm/
│   └── charts/
├── .pre-commit-config.yaml
├── Makefile
├── CHANGES.md
└── README.md
```

**When to Use**: Infrastructure as code, Kubernetes configs, Ansible playbooks

### ML Pipelines & Model Repos

**Structure**:
```
ml-project-name/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── training/
├── experiments/
│   └── notebooks/
├── models/
│   └── .gitkeep
├── tests/
│   ├── unit/
│   └── integration/
├── mlflow/
│   └── tracking/
├── schemas/
│   ├── input/
│   └── output/
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── CHANGES.md
└── README.md
```

**When to Use**: ML training pipelines, model repositories, feature engineering

### Postgres Schema/Migration Repos

**Structure**:
```
schema-name/
├── .github/
│   └── workflows/
│       └── migration-check.yml
├── migrations/
│   ├── versions/
│   │   ├── 001_initial_schema.py
│   │   └── 002_add_users.py
│   └── alembic.ini
├── schemas/
│   ├── base.sql
│   └── extensions.sql
├── tests/
│   └── test_migrations.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── CHANGES.md
└── README.md
```

**When to Use**: Database schemas, migration management, schema versioning

### UI Frontend Repos

**Structure**:
```
ui-app-name/
├── .github/
│   └── workflows/
│       └── ui-build.yml
├── src/
│   ├── components/
│   ├── pages/
│   └── utils/
├── assets/
│   ├── css/
│   └── images/
├── tests/
│   └── test_components.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── CHANGES.md
└── README.md
```

**When to Use**: NiceGUI apps, web frontends, UI components

### Cross-Language Mono-Repos

**Structure**:
```
monorepo-name/
├── .github/
│   └── workflows/
│       └── monorepo-ci.yml
├── services/
│   ├── api-go/
│   ├── worker-python/
│   └── processor-rust/
├── packages/
│   ├── shared-go/
│   └── common-python/
├── infrastructure/
│   ├── terraform/
│   └── k8s/
├── docs/
├── .pre-commit-config.yaml
├── Makefile
├── CHANGES.md
└── README.md
```

**When to Use**: Related services, shared codebases, coordinated releases

## Standard Repository Template Designs

### Python Package Template

**File Tree**:
```
python-package-template/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   ├── api/
│   └── examples/
├── .pre-commit-config.yaml
├── .ruff.toml
├── Makefile
├── pyproject.toml
├── README.md
└── CHANGES.md
```

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "package-name"
version = "0.1.0"
description = "Package description"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
```

**Makefile**:
```makefile
.PHONY: install test lint format type-check build release

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=package_name --cov-report=html

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

build:
	python -m build

release: test lint type-check
	python -m build
	twine upload dist/*
```

### Go Microservice Template

**File Tree**:
```
go-service-template/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── cmd/
│   └── service/
│       └── main.go
├── internal/
│   ├── handlers/
│   ├── models/
│   └── services/
├── pkg/
│   └── public/
├── api/
│   └── openapi.yaml
├── migrations/
├── deployments/
│   └── k8s/
├── .pre-commit-config.yaml
├── Makefile
├── go.mod
├── go.sum
├── Dockerfile
├── docker-bake.hcl
├── README.md
└── CHANGES.md
```

**go.mod**:
```go
module github.com/org/service-name

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)
```

**Makefile**:
```makefile
.PHONY: build test lint fmt vet docker-build

build:
	go build -o bin/service ./cmd/service

test:
	go test -v -cover ./...

lint:
	golangci-lint run

fmt:
	go fmt ./...

vet:
	go vet ./...

docker-build:
	docker buildx bake
```

**docker-bake.hcl**:
```hcl
group "default" {
  targets = ["service"]
}

target "service" {
  dockerfile = "Dockerfile"
  tags = ["service:latest"]
  platforms = ["linux/amd64", "linux/arm64"]
}
```

### Rust Service Template

**File Tree**:
```
rust-service-template/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── src/
│   ├── main.rs
│   ├── handlers.rs
│   ├── models.rs
│   └── services.rs
├── tests/
│   └── integration_test.rs
├── .pre-commit-config.yaml
├── Makefile
├── Cargo.toml
├── Dockerfile
├── docker-bake.hcl
├── README.md
└── CHANGES.md
```

**Cargo.toml**:
```toml
[package]
name = "service-name"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
tokio-test = "0.4"
```

**Makefile**:
```makefile
.PHONY: build test lint fmt clippy docker-build

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --check

docker-build:
	docker buildx bake
```

### Docker Image Template

**File Tree**:
```
docker-image-template/
├── .github/
│   └── workflows/
│       └── build.yml
├── Dockerfile
├── Dockerfile.dev
├── docker-bake.hcl
├── .dockerignore
├── Makefile
├── README.md
└── CHANGES.md
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Run application
CMD ["python", "-m", "src.main"]
```

**docker-bake.hcl**:
```hcl
group "default" {
  targets = ["app"]
}

target "app" {
  dockerfile = "Dockerfile"
  tags = [
    "app:latest",
    "app:${BAKE_GIT_COMMIT}",
    "registry.example.com/app:${BAKE_GIT_TAG}"
  ]
  platforms = ["linux/amd64", "linux/arm64"]
  args = {
    BUILDKIT_INLINE_CACHE = "1"
  }
}
```

**Makefile**:
```makefile
.PHONY: build push multiarch

build:
	docker buildx bake

push:
	docker buildx bake --push

multiarch:
	docker buildx build --platform linux/amd64,linux/arm64 -t app:latest .
```

### Helm Chart Template

**File Tree**:
```
helm-chart-template/
├── .github/
│   └── workflows/
│       └── chart-release.yml
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
├── tests/
│   └── test-chart.sh
├── .pre-commit-config.yaml
├── Makefile
├── README.md
└── CHANGES.md
```

**Chart.yaml**:
```yaml
apiVersion: v2
name: app-chart
description: Helm chart for application
type: application
version: 0.1.0
appVersion: "1.0.0"
```

**Makefile**:
```makefile
.PHONY: lint test package

lint:
	helm lint .

test:
	helm test .

package:
	helm package .
```

### NiceGUI UI Template

**File Tree**:
```
nicegui-app-template/
├── .github/
│   └── workflows/
│       └── ui-build.yml
├── src/
│   ├── main.py
│   ├── pages/
│   │   ├── __init__.py
│   │   └── dashboard.py
│   ├── components/
│   │   └── widgets.py
│   └── utils/
│       └── helpers.py
├── static/
│   ├── css/
│   └── images/
├── tests/
│   └── test_pages.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── Dockerfile
├── README.md
└── CHANGES.md
```

**requirements.txt**:
```txt
nicegui>=1.4.0
fastapi>=0.104.0
pydantic>=2.0.0
```

**Makefile**:
```makefile
.PHONY: run test lint format

run:
	python -m src.main

test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
```

### ML Experiment Template

**File Tree**:
```
ml-experiment-template/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── src/
│   ├── data/
│   │   └── loaders.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   └── trainer.py
│   └── training/
│       └── pipeline.py
├── experiments/
│   └── notebooks/
├── models/
│   └── .gitkeep
├── mlflow/
│   └── .gitkeep
├── schemas/
│   ├── input.json
│   └── output.json
├── tests/
│   └── test_pipeline.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── README.md
└── CHANGES.md
```

**Makefile**:
```makefile
.PHONY: train test export-onnx

train:
	python -m src.training.pipeline

test:
	pytest tests/

export-onnx:
	python -m src.models.export_onnx
```

### ETL Template (DuckDB, Prefect, Postgres)

**File Tree**:
```
etl-template/
├── .github/
│   └── workflows/
│       └── etl-pipeline.yml
├── pipelines/
│   ├── extract/
│   │   └── extract.py
│   ├── transform/
│   │   └── transform.py
│   └── load/
│       └── load.py
├── schemas/
│   ├── input/
│   ├── output/
│   └── contracts/
├── prefect/
│   └── flows/
│       └── main_flow.py
├── tests/
│   └── test_pipelines.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── README.md
└── CHANGES.md
```

**prefect/flows/main_flow.py**:
```python
from prefect import flow, task
import duckdb

@task
def extract():
    """Extract data"""
    pass

@task
def transform(data):
    """Transform data"""
    pass

@task
def load(data):
    """Load data"""
    pass

@flow
def etl_flow():
    """Main ETL flow"""
    data = extract()
    transformed = transform(data)
    load(transformed)
```

### Postgres Migration Repo Template

**File Tree**:
```
migration-template/
├── .github/
│   └── workflows/
│       └── migration-check.yml
├── migrations/
│   ├── versions/
│   │   ├── 001_initial_schema.py
│   │   └── 002_add_users.py
│   └── alembic.ini
├── schemas/
│   ├── base.sql
│   └── extensions.sql
├── tests/
│   └── test_migrations.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── README.md
└── CHANGES.md
```

**migrations/alembic.ini**:
```ini
[alembic]
script_location = migrations
sqlalchemy.url = postgresql://user:pass@localhost/db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

**Makefile**:
```makefile
.PHONY: migrate upgrade downgrade create-migration

migrate:
	alembic upgrade head

upgrade:
	alembic upgrade +1

downgrade:
	alembic downgrade -1

create-migration:
	alembic revision --autogenerate -m "$(MSG)"
```

## Unified Versioning Strategy

### Semantic Versioning

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: Backward-compatible additions
- **PATCH**: Bug fixes

**Examples**:
- `1.0.0` - Initial release
- `1.1.0` - Added feature (backward compatible)
- `1.1.1` - Bug fix
- `2.0.0` - Breaking change

### Branching Model

**GitFlow**:
```
main (production)
├── develop (integration)
│   ├── feature/* (feature branches)
│   └── hotfix/* (urgent fixes)
└── release/* (release candidates)
```

**Trunk-Based**:
```
main (trunk)
├── feature/* (short-lived)
└── release/* (release branches)
```

### Tagging Conventions

```bash
# Version tags
v1.2.3

# Pre-release tags
v1.2.3-alpha.1
v1.2.3-beta.1
v1.2.3-rc.1

# Build metadata
v1.2.3+abc123
```

### Container Tagging Best Practices

```yaml
# Container tags
tags:
  - "app:latest"                    # Latest
  - "app:v1.2.3"                    # Version
  - "app:v1.2.3-abc123"             # Version + commit
  - "app:abc123"                    # Commit SHA
  - "app:v1.2"                     # Minor version
  - "app:v1"                       # Major version
```

### Dataset Versioning

**DVC**:
```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - data/raw
    outs:
      - data/prepared
```

**LakeFS**:
```bash
# LakeFS versioning
lakectl branch create data/main
lakectl commit -m "Add new dataset" data/main
```

### ML Model Versioning

```python
# MLflow model versioning
import mlflow

mlflow.set_experiment("user-prediction")
with mlflow.start_run():
    mlflow.log_model(model, "model", registered_model_name="user-predictor")
    # Creates version 1, 2, 3, etc.
```

### Schema Versioning

```python
# Schema versioning
schema_version = "1.2.0"  # SemVer for schemas

# Migration versioning
migration_version = "002"  # Sequential
```

## Cross-Project Governance Automation

### Repo Creation via Templates

**Cookiecutter Template**:
```yaml
# cookiecutter.json
{
    "project_name": "My Project",
    "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '-') }}",
    "language": ["python", "go", "rust"],
    "framework": ["fastapi", "gin", "actix"],
    "license": ["MIT", "Apache-2.0"]
}
```

**Template Generation Script**:
```python
# scripts/generate_repo.py
import cookiecutter.main

def generate_repo(template_path: str, output_dir: str, context: dict):
    """Generate repository from template"""
    cookiecutter.main.cookiecutter(
        template_path,
        output_dir=output_dir,
        extra_context=context,
        no_input=True
    )
```

### Taskfile Standard Commands

**Taskfile.yml**:
```yaml
version: '3'

tasks:
  default:
    desc: Show available tasks
    cmds:
      - task --list

  install:
    desc: Install dependencies
    cmds:
      - pip install -r requirements.txt

  test:
    desc: Run tests
    cmds:
      - pytest tests/

  lint:
    desc: Lint code
    cmds:
      - ruff check src/

  format:
    desc: Format code
    cmds:
      - ruff format src/

  build:
    desc: Build artifacts
    cmds:
      - docker buildx bake

  release:
    desc: Release new version
    cmds:
      - task: test
      - task: lint
      - task: build
      - git tag v{{.VERSION}}
      - git push --tags
```

### Pre-Commit Hooks

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Standard CI/CD Pipeline Templates

**GitHub Actions Template**:
```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: make install
      - run: make test
      - run: make lint

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
      - run: syft packages . -o spdx > sbom.spdx

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - run: make build
```

### Automated CHANGES.md Generation

```python
# scripts/generate_changelog.py
import subprocess
from datetime import datetime

class ChangelogGenerator:
    def generate(self, since_tag: str = None):
        """Generate CHANGES.md from git commits"""
        if since_tag:
            commits = subprocess.run(
                ["git", "log", f"{since_tag}..HEAD", "--pretty=format:%s"],
                capture_output=True,
                text=True
            ).stdout.split("\n")
        else:
            commits = subprocess.run(
                ["git", "log", "--pretty=format:%s"],
                capture_output=True,
                text=True
            ).stdout.split("\n")
        
        # Categorize commits
        changes = {
            "Added": [],
            "Changed": [],
            "Fixed": [],
            "Removed": []
        }
        
        for commit in commits:
            if commit.startswith("feat:"):
                changes["Added"].append(commit[5:].strip())
            elif commit.startswith("fix:"):
                changes["Fixed"].append(commit[4:].strip())
            elif commit.startswith("refactor:"):
                changes["Changed"].append(commit[8:].strip())
        
        # Generate markdown
        changelog = f"# Changelog\n\n## {datetime.now().strftime('%Y-%m-%d')}\n\n"
        for category, items in changes.items():
            if items:
                changelog += f"### {category}\n\n"
                for item in items:
                    changelog += f"- {item}\n"
                changelog += "\n"
        
        return changelog
```

## Lifecycle Governance & Aging Control

### Deprecation Workflows

```yaml
# deprecation-policy.yaml
deprecation:
  notice_period_days: 90
  stages:
    - name: "deprecated"
      duration_days: 90
      actions:
        - add_deprecation_warning
        - update_documentation
        - notify_users
    
    - name: "sunset"
      duration_days: 30
      actions:
        - disable_new_usage
        - migrate_existing_usage
        - final_notice
    
    - name: "archived"
      actions:
        - archive_repository
        - update_documentation
        - redirect_to_replacement
```

### Archival Strategy

```python
# scripts/archive_repo.py
class RepoArchiver:
    def archive_repo(self, repo_name: str, reason: str):
        """Archive repository"""
        # 1. Add archival notice
        self.add_archival_notice(repo_name, reason)
        
        # 2. Move to archive org
        self.move_to_archive_org(repo_name)
        
        # 3. Update documentation
        self.update_documentation(repo_name)
        
        # 4. Notify dependents
        self.notify_dependents(repo_name)
```

### Sunset Schedules

```yaml
# sunset-schedule.yaml
sunset:
  repos:
    - name: "old-service"
      sunset_date: "2024-12-31"
      replacement: "new-service"
      migration_guide: "docs/migration.md"
    
    - name: "legacy-package"
      sunset_date: "2025-06-30"
      replacement: "modern-package"
      migration_guide: "docs/migration.md"
```

### Automated Stale-Repo Detection

```python
# scripts/detect_stale_repos.py
class StaleRepoDetector:
    def detect_stale_repos(self, days_inactive: int = 180):
        """Detect stale repositories"""
        repos = self.get_all_repos()
        
        stale_repos = []
        for repo in repos:
            last_commit = self.get_last_commit_date(repo)
            days_since = (datetime.now() - last_commit).days
            
            if days_since > days_inactive:
                stale_repos.append({
                    "repo": repo,
                    "days_inactive": days_since,
                    "last_commit": last_commit
                })
        
        return stale_repos
```

### Dependency Upgrade Cadences

```yaml
# dependency-upgrade-policy.yaml
upgrade_cadence:
  security:
    frequency: "weekly"
    automation: true
  
  minor:
    frequency: "monthly"
    automation: true
  
  major:
    frequency: "quarterly"
    automation: false
    requires_approval: true
```

### Container Base Image Rotation

```yaml
# base-image-rotation.yaml
base_images:
  python:
    current: "python:3.11-slim"
    rotation_frequency: "monthly"
    security_scan: true
  
  go:
    current: "golang:1.21"
    rotation_frequency: "monthly"
    security_scan: true
```

## Multi-Environment Release Management

### Dev → Staging → Prod Flows

```mermaid
graph LR
    Dev["Dev<br/>(Auto Deploy)"]
    Stage["Stage<br/>(Auto + Approval)"]
    Prod["Prod<br/>(Manual Approval)"]
    
    Dev -->|"Validate"| Validate["Validation"]
    Validate -->|"Promote"| Stage
    Stage -->|"Validate"| Validate2["Validation"]
    Validate2 -->|"Approve"| Approval["Manual Approval"]
    Approval -->|"Promote"| Prod
    
    style Dev fill:#e8f5e9
    style Stage fill:#fff4e1
    style Prod fill:#ffebee
```

### Multi-Cluster Deployments

```yaml
# multi-cluster-deployment.yaml
clusters:
  - name: dev-cluster
    environment: dev
    auto_deploy: true
  
  - name: stage-cluster
    environment: stage
    auto_deploy: true
    require_approval: false
  
  - name: prod-cluster-us-east
    environment: prod
    region: us-east-1
    auto_deploy: false
    require_approval: true
  
  - name: prod-cluster-us-west
    environment: prod
    region: us-west-2
    auto_deploy: false
    require_approval: true
```

### Image Promotion Pipelines

```yaml
# .github/workflows/promote-image.yml
name: Promote Image
on:
  workflow_dispatch:
    inputs:
      source_env:
        description: 'Source environment'
        required: true
        type: choice
        options:
          - dev
          - stage
      target_env:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - stage
          - prod

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - name: Promote image
        run: |
          # Tag image for target environment
          docker tag app:${{ github.sha }} app:${{ inputs.target_env }}
          docker push app:${{ inputs.target_env }}
```

### Artifact Integrity Verification

```python
# scripts/verify_artifacts.py
import hashlib
import json

class ArtifactVerifier:
    def verify_artifact(self, artifact_path: str, expected_hash: str) -> bool:
        """Verify artifact integrity"""
        with open(artifact_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        
        return actual_hash == expected_hash
    
    def generate_sbom(self, artifact_path: str) -> dict:
        """Generate SBOM for artifact"""
        import syft
        sbom = syft.scan(artifact_path, output="json")
        return json.loads(sbom)
```

### SBOM & VEX Integration

```yaml
# sbom-vex-policy.yaml
sbom:
  generation:
    required: true
    format: "spdx"
    include_dependencies: true
  
  storage:
    location: "artifacts/sbom/"
    versioned: true
  
vex:
  generation:
    required: true
    format: "vex-json"
  
  updates:
    frequency: "daily"
    automation: true
```

### Air-Gapped Release Bundles

```bash
# scripts/create_airgap_bundle.sh
#!/bin/bash

BUNDLE_NAME="release-bundle-$(date +%Y%m%d).tar.gz"

# Create bundle
tar -czf "$BUNDLE_NAME" \
    images/ \
    charts/ \
    manifests/ \
    sboms/ \
    checksums.txt

# Sign bundle
gpg --sign "$BUNDLE_NAME"

# Create manifest
cat > manifest.json <<EOF
{
  "bundle_name": "$BUNDLE_NAME",
  "created_at": "$(date -Iseconds)",
  "checksum": "$(sha256sum $BUNDLE_NAME | cut -d' ' -f1)"
}
EOF
```

### Reproducible Build Requirements

```dockerfile
# Reproducible Dockerfile
FROM python:3.11-slim@sha256:abc123...  # Pinned base

# Pin all dependencies
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Reproducible build
ENV SOURCE_DATE_EPOCH=1234567890
```

## Cross-Repo Observability

### Dependency Graph Visualizations

```python
# scripts/generate_dependency_graph.py
import networkx as nx
import matplotlib.pyplot as plt

class DependencyGraphGenerator:
    def generate_graph(self, repos: List[str]):
        """Generate dependency graph"""
        G = nx.DiGraph()
        
        for repo in repos:
            deps = self.get_dependencies(repo)
            G.add_node(repo)
            for dep in deps:
                G.add_edge(repo, dep)
        
        # Visualize
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.savefig("dependency_graph.png")
```

### Repo Health Dashboards

**Grafana Dashboard**:
```json
{
  "dashboard": {
    "title": "Repository Health",
    "panels": [
      {
        "title": "Repo Activity",
        "targets": [
          {
            "expr": "repo_commits_total",
            "legendFormat": "{{repo}}"
          }
        ]
      },
      {
        "title": "Open PRs",
        "targets": [
          {
            "expr": "repo_open_prs",
            "legendFormat": "{{repo}}"
          }
        ]
      },
      {
        "title": "Test Coverage",
        "targets": [
          {
            "expr": "repo_test_coverage",
            "legendFormat": "{{repo}}"
          }
        ]
      }
    ]
  }
}
```

### Automated Drift Detection

```python
# scripts/detect_repo_drift.py
class RepoDriftDetector:
    def detect_drift(self, repo: str, template: str):
        """Detect drift from template"""
        repo_structure = self.get_repo_structure(repo)
        template_structure = self.get_template_structure(template)
        
        drift = {
            "missing_files": [],
            "extra_files": [],
            "modified_files": []
        }
        
        # Compare structures
        template_files = set(template_structure.keys())
        repo_files = set(repo_structure.keys())
        
        drift["missing_files"] = list(template_files - repo_files)
        drift["extra_files"] = list(repo_files - template_files)
        
        # Check modified files
        for file in template_files & repo_files:
            if template_structure[file] != repo_structure[file]:
                drift["modified_files"].append(file)
        
        return drift
```

### Repo Complexity Scoring

```python
# scripts/calculate_complexity.py
class ComplexityScorer:
    def calculate_complexity(self, repo: str) -> dict:
        """Calculate repository complexity"""
        # Cyclomatic complexity
        cyclomatic = self.calculate_cyclomatic_complexity(repo)
        
        # Lines of code
        sloc = self.count_sloc(repo)
        
        # Churn rate
        churn = self.calculate_churn(repo)
        
        # Dependency count
        deps = self.count_dependencies(repo)
        
        # Complexity score
        score = (
            cyclomatic * 0.3 +
            (sloc / 1000) * 0.2 +
            churn * 0.3 +
            (deps / 10) * 0.2
        )
        
        return {
            "cyclomatic": cyclomatic,
            "sloc": sloc,
            "churn": churn,
            "dependencies": deps,
            "score": score
        }
```

## Anti-Patterns

### Unversioned Repos

**Symptom**: Repositories without version tags or releases.

**Fix**: Implement semantic versioning, automated tagging.

**Prevention**: Require version tags in CI/CD.

### Folder Chaos

**Symptom**: Inconsistent folder structures across repos.

**Fix**: Use templates, enforce structure in CI.

**Prevention**: Template-based repo creation.

### "Hidden Dockerfile" Syndrome

**Symptom**: Dockerfiles in unexpected locations.

**Fix**: Standardize Dockerfile location (root or `docker/`).

**Prevention**: Template enforcement, CI checks.

### One-Off ML Notebooks Becoming Production

**Symptom**: Jupyter notebooks used as production code.

**Fix**: Refactor to proper Python modules, add tests.

**Prevention**: Prohibit notebooks in production, require refactoring.

### Undocumented Migrations

**Symptom**: Database migrations without documentation.

**Fix**: Require migration descriptions, link to tickets.

**Prevention**: Enforce migration documentation in CI.

### Mutable Shared Scripts

**Symptom**: Scripts copied between repos, diverging.

**Fix**: Extract to shared package, version properly.

**Prevention**: Use shared packages, avoid copying.

## End-to-End Example

### Complete System Architecture

```mermaid
graph TB
    subgraph Repos["Repository Ecosystem"]
        UI["nicegui-app<br/>(Python)"]
        API["geo-service<br/>(Go)"]
        DB["schema-repo<br/>(Migrations)"]
        Events["event-lib<br/>(Rust)"]
    end
    
    subgraph CI["CI/CD"]
        Build["Build & Test"]
        Scan["Security Scan"]
        SBOM["SBOM Generation"]
    end
    
    subgraph Registry["Artifact Registry"]
        Images["Container Images"]
        Charts["Helm Charts"]
    end
    
    subgraph K8s["Kubernetes Cluster"]
        UI_Pod["NiceGUI Pod"]
        API_Pod["Go Service Pod"]
        DB_Pod["Postgres Pod"]
    end
    
    subgraph Data["Data Layer"]
        Postgres["Postgres DB"]
        Redis["Redis Streams"]
    end
    
    UI --> CI
    API --> CI
    DB --> CI
    Events --> CI
    
    CI --> Build
    Build --> Scan
    Scan --> SBOM
    SBOM --> Registry
    
    Registry --> Images
    Registry --> Charts
    
    Images --> K8s
    Charts --> K8s
    
    K8s --> UI_Pod
    K8s --> API_Pod
    K8s --> DB_Pod
    
    UI_Pod --> API_Pod
    API_Pod --> Postgres
    API_Pod --> Redis
    
    style Repos fill:#e1f5ff
    style CI fill:#fff4e1
    style Registry fill:#e8f5e9
    style K8s fill:#f3e5f5
    style Data fill:#ffebee
```

### NiceGUI App Repository

**Structure**:
```
nicegui-geo-app/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── src/
│   ├── main.py
│   ├── pages/
│   │   └── dashboard.py
│   └── components/
│       └── map_widget.py
├── tests/
│   └── test_pages.py
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
├── Dockerfile
├── docker-bake.hcl
├── README.md
└── CHANGES.md
```

**src/main.py**:
```python
from nicegui import ui
from src.pages.dashboard import DashboardPage

@ui.page("/")
async def index():
    dashboard = DashboardPage()
    await dashboard.render()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8080)
```

### Go Geospatial Service Repository

**Structure**:
```
geo-service/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── cmd/
│   └── service/
│       └── main.go
├── internal/
│   ├── handlers/
│   │   └── geo_handler.go
│   └── services/
│       └── geo_service.go
├── pkg/
│   └── geo/
│       └── spatial.go
├── api/
│   └── openapi.yaml
├── tests/
│   └── integration_test.go
├── .pre-commit-config.yaml
├── Makefile
├── go.mod
├── Dockerfile
├── docker-bake.hcl
├── README.md
└── CHANGES.md
```

**cmd/service/main.go**:
```go
package main

import (
    "github.com/gin-gonic/gin"
    "github.com/org/geo-service/internal/handlers"
)

func main() {
    r := gin.Default()
    
    geoHandler := handlers.NewGeoHandler()
    r.GET("/api/geo/intersect", geoHandler.Intersect)
    r.POST("/api/geo/query", geoHandler.Query)
    
    r.Run(":8080")
}
```

### Postgres Schema Repository

**Structure**:
```
geo-schema/
├── .github/
│   └── workflows/
│       └── migration-check.yml
├── migrations/
│   ├── versions/
│   │   ├── 001_initial_schema.py
│   │   └── 002_add_spatial_indexes.py
│   └── alembic.ini
├── schemas/
│   └── base.sql
├── tests/
│   └── test_migrations.py
├── Makefile
├── requirements.txt
├── README.md
└── CHANGES.md
```

### Integration Flow

```mermaid
sequenceDiagram
    participant UI as NiceGUI App
    participant API as Go Service
    participant DB as Postgres
    participant Redis as Redis Streams
    participant Loki as Loki
    
    UI->>API: HTTP Request
    API->>DB: Spatial Query
    DB-->>API: Results
    API->>Redis: Publish Event
    API-->>UI: JSON Response
    UI->>Loki: Log Event
    Redis->>UI: WebSocket Update
```

## Agentic LLM Integration Hooks

### Repo Generation Automation

```python
# llm/repo_generator.py
from openai import OpenAI

class LLMRepoGenerator:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_repo(self, description: str, language: str) -> dict:
        """Generate repository structure using LLM"""
        prompt = f"""
        Generate a standardized repository structure for:
        Description: {description}
        Language: {language}
        
        Include:
        1. Folder structure
        2. Key files (Makefile, Dockerfile, etc.)
        3. CI/CD configuration
        4. Pre-commit hooks
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a repository architecture expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
```

### Template Population

```python
# llm/template_populator.py
class LLMTemplatePopulator:
    def populate_template(self, template_path: str, context: dict) -> str:
        """Populate template with LLM-generated content"""
        # Read template
        with open(template_path) as f:
            template = f.read()
        
        # Generate content for placeholders
        for key, value in context.items():
            if key.startswith("llm_"):
                generated = self.llm_generate(value)
                template = template.replace(f"{{{{ {key} }}}}", generated)
        
        return template
```

### Semantic Version Bump Suggestions

```python
# llm/version_bumper.py
class LLMVersionBumper:
    def suggest_version_bump(self, changes: List[str]) -> str:
        """Suggest version bump based on changes"""
        prompt = f"""
        Analyze these changes and suggest semantic version bump:
        
        Changes:
        {json.dumps(changes, indent=2)}
        
        Return: MAJOR, MINOR, or PATCH
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a semantic versioning expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
```

### Dependency Upgrade PR Generation

```python
# llm/dependency_upgrader.py
class LLMDependencyUpgrader:
    def generate_upgrade_pr(self, repo: str, dependency: str, new_version: str):
        """Generate PR for dependency upgrade"""
        # Analyze impact
        impact = self.analyze_impact(repo, dependency, new_version)
        
        # Generate PR description
        pr_description = self.llm_generate_pr_description(
            dependency, new_version, impact
        )
        
        # Create PR
        from github import Github
        g = Github(token)
        repo_obj = g.get_repo(repo)
        
        pr = repo_obj.create_pull(
            title=f"Upgrade {dependency} to {new_version}",
            body=pr_description,
            head="upgrade-dependency",
            base="main"
        )
        
        return pr
```

### Cross-Repo Consistency Scans

```python
# llm/consistency_scanner.py
class LLMConsistencyScanner:
    def scan_consistency(self, repos: List[str]) -> dict:
        """Scan repositories for consistency"""
        # Get structures
        structures = {repo: self.get_structure(repo) for repo in repos}
        
        # Analyze with LLM
        prompt = f"""
        Analyze these repository structures for consistency:
        
        {json.dumps(structures, indent=2)}
        
        Identify:
        1. Inconsistencies
        2. Missing standard files
        3. Deviations from templates
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a repository standardization expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
```

### Long-Term Health Scoring

```python
# llm/health_scorer.py
class LLMHealthScorer:
    def score_repo_health(self, repo: str) -> dict:
        """Score repository long-term health"""
        metrics = {
            "activity": self.get_activity(repo),
            "test_coverage": self.get_test_coverage(repo),
            "documentation": self.get_documentation_score(repo),
            "dependencies": self.get_dependency_health(repo),
            "complexity": self.get_complexity(repo)
        }
        
        # LLM analysis
        prompt = f"""
        Score repository health based on:
        
        {json.dumps(metrics, indent=2)}
        
        Provide:
        1. Overall health score (0-100)
        2. Risk factors
        3. Recommendations
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a repository health expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)
```

### Automated Documentation Ingestion

```python
# llm/docs_ingester.py
class LLMDocsIngester:
    def ingest_to_mkdocs(self, repo: str, docs_path: str):
        """Ingest repository docs into MkDocs"""
        # Extract docs
        docs = self.extract_docs(repo, docs_path)
        
        # Generate MkDocs structure
        mkdocs_structure = self.llm_generate_mkdocs_structure(docs)
        
        # Write to MkDocs
        self.write_to_mkdocs(mkdocs_structure)
```

## Checklists

### Repository Creation Checklist

- [ ] Choose appropriate template
- [ ] Generate from template
- [ ] Configure CI/CD
- [ ] Set up pre-commit hooks
- [ ] Initialize versioning
- [ ] Create initial documentation
- [ ] Set up dependency scanning
- [ ] Configure security policies

### Repository Maintenance Checklist

- [ ] Update dependencies regularly
- [ ] Review and update documentation
- [ ] Run security scans
- [ ] Update CI/CD pipelines
- [ ] Review and merge dependabot PRs
- [ ] Update base images
- [ ] Review complexity metrics
- [ ] Check for drift from template

### Repository Deprecation Checklist

- [ ] Create deprecation notice
- [ ] Update documentation
- [ ] Notify users
- [ ] Create migration guide
- [ ] Set sunset date
- [ ] Archive repository
- [ ] Update dependent systems

## See Also

- **[ADR and Technical Decision Governance](adr-decision-governance.md)** - Decision recording
- **[Configuration Management](../operations-monitoring/configuration-management.md)** - Config governance
- **[Release Management](../operations-monitoring/release-management-and-progressive-delivery.md)** - Deployment practices

---

*This guide provides a complete framework for repository standardization. Start with templates, enforce consistency, automate governance, and monitor health. The goal is a unified, maintainable repository ecosystem that scales with your organization.*

