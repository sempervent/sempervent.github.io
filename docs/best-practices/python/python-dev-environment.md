# Python Development Environment Best Practices

**Objective**: Master senior-level Python development environment setup for production systems. When you need to manage Python versions, dependencies, and tooling efficiently, when you want to ensure reproducible development environments, when you need enterprise-grade development workflows—these best practices become your weapon of choice.

## Core Principles

- **Version Management**: Use modern Python version managers
- **Dependency Management**: Lock dependencies for reproducibility
- **Tooling Integration**: Integrate essential development tools
- **Environment Isolation**: Isolate project environments
- **Automation**: Automate common development tasks

## Python Version Management

### Modern Python Version Managers

```python
# python/01-version-management.py

"""
Modern Python version management with pyenv and uv
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

class PythonVersionManager:
    """Manage Python versions and environments"""
    
    def __init__(self, version_manager: str = "pyenv"):
        self.version_manager = version_manager
        self.available_versions = self._get_available_versions()
    
    def _get_available_versions(self) -> List[str]:
        """Get available Python versions"""
        try:
            if self.version_manager == "pyenv":
                result = subprocess.run(
                    ["pyenv", "install", "--list"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                versions = [line.strip() for line in result.stdout.split('\n') 
                           if line.strip() and not line.startswith('Available')]
                return [v for v in versions if v.startswith(('3.8', '3.9', '3.10', '3.11', '3.12'))]
            else:
                return []
        except subprocess.CalledProcessError:
            return []
    
    def install_python_version(self, version: str) -> bool:
        """Install a specific Python version"""
        try:
            if self.version_manager == "pyenv":
                subprocess.run(
                    ["pyenv", "install", version], 
                    check=True
                )
                return True
            return False
        except subprocess.CalledProcessError:
            return False
    
    def set_local_version(self, version: str, project_path: Path) -> bool:
        """Set local Python version for a project"""
        try:
            if self.version_manager == "pyenv":
                subprocess.run(
                    ["pyenv", "local", version], 
                    cwd=project_path, 
                    check=True
                )
                return True
            return False
        except subprocess.CalledProcessError:
            return False
    
    def get_current_version(self) -> Optional[str]:
        """Get current Python version"""
        try:
            result = subprocess.run(
                ["python", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

# Usage example
def setup_python_environment(project_path: Path, python_version: str = "3.11"):
    """Setup Python environment for a project"""
    manager = PythonVersionManager()
    
    # Install Python version if not available
    if python_version not in manager.available_versions:
        print(f"Installing Python {python_version}...")
        manager.install_python_version(python_version)
    
    # Set local version
    manager.set_local_version(python_version, project_path)
    
    print(f"Python environment setup complete: {manager.get_current_version()}")
```

### UV Package Manager

```python
# python/02-uv-package-manager.py

"""
Modern Python package management with uv
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional

class UVPackageManager:
    """Manage Python packages with uv"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.pyproject_toml = project_path / "pyproject.toml"
        self.uv_lock = project_path / "uv.lock"
    
    def init_project(self, name: str, version: str = "0.1.0") -> bool:
        """Initialize a new Python project"""
        try:
            subprocess.run(
                ["uv", "init", name, "--python", "3.11"], 
                cwd=self.project_path, 
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def add_dependency(self, package: str, version: Optional[str] = None) -> bool:
        """Add a dependency to the project"""
        try:
            cmd = ["uv", "add", package]
            if version:
                cmd.append(f"=={version}")
            
            subprocess.run(cmd, cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def add_dev_dependency(self, package: str, version: Optional[str] = None) -> bool:
        """Add a development dependency"""
        try:
            cmd = ["uv", "add", "--dev", package]
            if version:
                cmd.append(f"=={version}")
            
            subprocess.run(cmd, cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def install_dependencies(self) -> bool:
        """Install project dependencies"""
        try:
            subprocess.run(["uv", "sync"], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_command(self, command: str) -> bool:
        """Run a command in the project environment"""
        try:
            subprocess.run(["uv", "run", command], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get project dependencies"""
        if not self.pyproject_toml.exists():
            return {}
        
        try:
            with open(self.pyproject_toml, 'r') as f:
                content = f.read()
                # Parse TOML content (simplified)
                dependencies = {}
                in_deps = False
                for line in content.split('\n'):
                    if line.strip() == '[dependencies]':
                        in_deps = True
                    elif line.startswith('[') and line.strip() != '[dependencies]':
                        in_deps = False
                    elif in_deps and '=' in line:
                        key, value = line.split('=', 1)
                        dependencies[key.strip()] = value.strip().strip('"')
                return dependencies
        except Exception:
            return {}

# Usage example
def setup_project_with_uv(project_path: Path, project_name: str):
    """Setup a new project with uv"""
    uv_manager = UVPackageManager(project_path)
    
    # Initialize project
    uv_manager.init_project(project_name)
    
    # Add core dependencies
    uv_manager.add_dependency("fastapi")
    uv_manager.add_dependency("uvicorn")
    uv_manager.add_dependency("pydantic")
    
    # Add development dependencies
    uv_manager.add_dev_dependency("pytest")
    uv_manager.add_dev_dependency("black")
    uv_manager.add_dev_dependency("ruff")
    uv_manager.add_dev_dependency("mypy")
    
    # Install dependencies
    uv_manager.install_dependencies()
    
    print("Project setup complete with uv")
```

## Project Structure

### Modern Python Project Layout

```python
# python/03-project-structure.py

"""
Modern Python project structure and organization
"""

from pathlib import Path
from typing import Dict, List
import json

class PythonProjectStructure:
    """Manage Python project structure"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.src_path = project_path / "src"
        self.tests_path = project_path / "tests"
        self.docs_path = project_path / "docs"
    
    def create_project_structure(self, project_name: str) -> bool:
        """Create standard Python project structure"""
        try:
            # Create directories
            directories = [
                self.src_path / project_name,
                self.tests_path,
                self.docs_path,
                self.project_path / "scripts",
                self.project_path / "data",
                self.project_path / "notebooks"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files
            (self.src_path / project_name / "__init__.py").touch()
            (self.tests_path / "__init__.py").touch()
            
            return True
        except Exception:
            return False
    
    def create_pyproject_toml(self, project_name: str, version: str = "0.1.0") -> bool:
        """Create pyproject.toml configuration"""
        try:
            pyproject_content = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{project_name}"
version = "{version}"
description = "A Python project"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
{project_name} = "{project_name}.cli:main"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
'''
            
            with open(self.project_path / "pyproject.toml", 'w') as f:
                f.write(pyproject_content)
            
            return True
        except Exception:
            return False
    
    def create_gitignore(self) -> bool:
        """Create .gitignore file"""
        try:
            gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
data/
*.log
*.sqlite
*.db
'''
            
            with open(self.project_path / ".gitignore", 'w') as f:
                f.write(gitignore_content)
            
            return True
        except Exception:
            return False

# Usage example
def setup_complete_project(project_path: Path, project_name: str):
    """Setup a complete Python project"""
    structure = PythonProjectStructure(project_path)
    
    # Create project structure
    structure.create_project_structure(project_name)
    
    # Create configuration files
    structure.create_pyproject_toml(project_name)
    structure.create_gitignore()
    
    print(f"Project {project_name} structure created successfully")
```

## Development Tools

### Essential Development Tools

```python
# python/04-development-tools.py

"""
Essential Python development tools and configuration
"""

import subprocess
from pathlib import Path
from typing import Dict, List

class DevelopmentTools:
    """Manage development tools and configuration"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
    
    def setup_pre_commit(self) -> bool:
        """Setup pre-commit hooks"""
        try:
            pre_commit_config = '''repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
'''
            
            with open(self.project_path / ".pre-commit-config.yaml", 'w') as f:
                f.write(pre_commit_config)
            
            # Install pre-commit
            subprocess.run(["pre-commit", "install"], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def setup_vscode_settings(self) -> bool:
        """Setup VS Code settings"""
        try:
            vscode_dir = self.project_path / ".vscode"
            vscode_dir.mkdir(exist_ok=True)
            
            settings = {
                "python.defaultInterpreterPath": "./.venv/bin/python",
                "python.linting.enabled": True,
                "python.linting.pylintEnabled": False,
                "python.linting.flake8Enabled": False,
                "python.linting.mypyEnabled": True,
                "python.formatting.provider": "black",
                "python.formatting.blackArgs": ["--line-length", "88"],
                "editor.formatOnSave": True,
                "editor.codeActionsOnSave": {
                    "source.organizeImports": True
                }
            }
            
            with open(vscode_dir / "settings.json", 'w') as f:
                json.dump(settings, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def setup_makefile(self) -> bool:
        """Create Makefile for common tasks"""
        try:
            makefile_content = '''.PHONY: help install test lint format clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv sync

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	uv run ruff check .
	uv run mypy src/

format: ## Format code
	uv run black src/ tests/
	uv run ruff check --fix .

clean: ## Clean up
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
'''
            
            with open(self.project_path / "Makefile", 'w') as f:
                f.write(makefile_content)
            
            return True
        except Exception:
            return False

# Usage example
def setup_development_environment(project_path: Path):
    """Setup complete development environment"""
    tools = DevelopmentTools(project_path)
    
    # Setup tools
    tools.setup_pre_commit()
    tools.setup_vscode_settings()
    tools.setup_makefile()
    
    print("Development environment setup complete")
```

## Environment Management

### Environment Configuration

```python
# python/05-environment-management.py

"""
Environment management and configuration
"""

import os
from pathlib import Path
from typing import Dict, Optional
import json

class EnvironmentManager:
    """Manage development environments"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.env_file = project_path / ".env"
        self.env_example = project_path / ".env.example"
    
    def create_env_example(self, variables: Dict[str, str]) -> bool:
        """Create .env.example file"""
        try:
            with open(self.env_example, 'w') as f:
                for key, description in variables.items():
                    f.write(f"# {description}\n")
                    f.write(f"{key}=\n\n")
            return True
        except Exception:
            return False
    
    def load_env_variables(self) -> Dict[str, str]:
        """Load environment variables from .env file"""
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    
    def setup_environment_config(self) -> bool:
        """Setup environment configuration"""
        try:
            # Create environment variables template
            env_variables = {
                "DEBUG": "Enable debug mode",
                "SECRET_KEY": "Application secret key",
                "DATABASE_URL": "Database connection string",
                "REDIS_URL": "Redis connection string",
                "API_KEY": "External API key",
                "LOG_LEVEL": "Logging level (DEBUG, INFO, WARNING, ERROR)"
            }
            
            self.create_env_example(env_variables)
            
            # Create environment-specific configurations
            environments = ["development", "testing", "staging", "production"]
            for env in environments:
                config = {
                    "environment": env,
                    "debug": env == "development",
                    "log_level": "DEBUG" if env == "development" else "INFO"
                }
                
                config_file = self.project_path / f"config_{env}.json"
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            
            return True
        except Exception:
            return False

# Usage example
def setup_environment_management(project_path: Path):
    """Setup environment management"""
    env_manager = EnvironmentManager(project_path)
    env_manager.setup_environment_config()
    
    print("Environment management setup complete")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Setup Python version management
from pathlib import Path
project_path = Path("my-project")
project_path.mkdir(exist_ok=True)

# 2. Setup project with uv
from python.dev_environment import setup_project_with_uv
setup_project_with_uv(project_path, "my-project")

# 3. Setup complete project structure
from python.project_structure import setup_complete_project
setup_complete_project(project_path, "my-project")

# 4. Setup development tools
from python.development_tools import setup_development_environment
setup_development_environment(project_path)

# 5. Setup environment management
from python.environment_management import setup_environment_management
setup_environment_management(project_path)
```

### Essential Patterns

```python
# Complete Python development environment setup
def create_python_development_environment(project_path: Path, project_name: str):
    """Create complete Python development environment"""
    
    # Setup Python version management
    manager = PythonVersionManager()
    manager.install_python_version("3.11")
    manager.set_local_version("3.11", project_path)
    
    # Setup project with uv
    uv_manager = UVPackageManager(project_path)
    uv_manager.init_project(project_name)
    
    # Add essential dependencies
    uv_manager.add_dependency("fastapi")
    uv_manager.add_dependency("uvicorn")
    uv_manager.add_dev_dependency("pytest")
    uv_manager.add_dev_dependency("black")
    uv_manager.add_dev_dependency("ruff")
    
    # Create project structure
    structure = PythonProjectStructure(project_path)
    structure.create_project_structure(project_name)
    structure.create_pyproject_toml(project_name)
    structure.create_gitignore()
    
    # Setup development tools
    tools = DevelopmentTools(project_path)
    tools.setup_pre_commit()
    tools.setup_vscode_settings()
    tools.setup_makefile()
    
    # Setup environment management
    env_manager = EnvironmentManager(project_path)
    env_manager.setup_environment_config()
    
    print(f"Python development environment for {project_name} setup complete!")
```

---

*This guide provides the complete machinery for setting up Python development environments. Each pattern includes implementation examples, configuration strategies, and real-world usage patterns for enterprise Python development.*
