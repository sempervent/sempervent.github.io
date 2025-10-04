# Python Package Development Best Practices

**Objective**: Master senior-level Python package development patterns for production systems. When you need to build and ship Python packages, when you want to ensure reproducibility and maintainability, when you need enterprise-grade packaging workflows—these best practices become your weapon of choice.

## Core Principles

- **Modern Tooling**: Use modern packaging tools like `uv` and `hatchling`
- **Reproducibility**: Lock dependencies and ensure consistent builds
- **Quality**: Implement comprehensive testing and code quality checks
- **Documentation**: Maintain excellent documentation and examples
- **Distribution**: Publish to PyPI and private repositories

## Modern Package Structure

### Project Layout

```python
# python/01-package-structure.py

"""
Modern Python package structure and organization
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import toml

class PythonPackageStructure:
    """Manage Python package structure and configuration"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.src_path = package_path / "src"
        self.tests_path = package_path / "tests"
        self.docs_path = package_path / "docs"
    
    def create_package_structure(self, package_name: str) -> bool:
        """Create standard Python package structure"""
        try:
            # Create main directories
            directories = [
                self.src_path / package_name,
                self.tests_path,
                self.docs_path,
                self.package_path / "scripts",
                self.package_path / "examples",
                self.package_path / "benchmarks"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py files
            (self.src_path / package_name / "__init__.py").touch()
            (self.tests_path / "__init__.py").touch()
            
            # Create package structure
            self._create_package_files(package_name)
            self._create_test_structure(package_name)
            self._create_documentation_structure()
            
            return True
        except Exception:
            return False
    
    def _create_package_files(self, package_name: str):
        """Create essential package files"""
        # Create main module
        main_module = self.src_path / package_name / "main.py"
        main_module.write_text('''"""
Main module for {package_name}
"""

__version__ = "0.1.0"

def hello_world() -> str:
    """Return a hello world message"""
    return "Hello, World!"

def get_version() -> str:
    """Get package version"""
    return __version__
''')
        
        # Create CLI module
        cli_module = self.src_path / package_name / "cli.py"
        cli_module.write_text('''"""
Command-line interface for {package_name}
"""

import argparse
import sys
from typing import List, Optional

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="A Python package",
        prog="{package_name}"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.verbose:
        print("Verbose mode enabled")
    
    print("Hello from {package_name}!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
''')
    
    def _create_test_structure(self, package_name: str):
        """Create test structure"""
        # Create test files
        test_main = self.tests_path / "test_main.py"
        test_main.write_text('''"""
Tests for main module
"""

import pytest
from {package_name}.main import hello_world, get_version

def test_hello_world():
    """Test hello_world function"""
    result = hello_world()
    assert result == "Hello, World!"

def test_get_version():
    """Test get_version function"""
    version = get_version()
    assert version == "0.1.0"
''')
        
        # Create conftest.py
        conftest = self.tests_path / "conftest.py"
        conftest.write_text('''"""
Pytest configuration
"""

import pytest
from typing import Generator

@pytest.fixture
def sample_data():
    """Sample data fixture"""
    return {"key": "value", "number": 42}

@pytest.fixture
def temp_file(tmp_path):
    """Temporary file fixture"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    return file_path
''')
    
    def _create_documentation_structure(self):
        """Create documentation structure"""
        # Create README
        readme = self.package_path / "README.md"
        readme.write_text('''# Python Package

A modern Python package built with best practices.

## Installation

```bash
pip install {package_name}
```

## Usage

```python
from {package_name} import hello_world

print(hello_world())
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run linting
ruff check .
black .
```

## License

MIT
''')
        
        # Create CHANGELOG
        changelog = self.package_path / "CHANGELOG.md"
        changelog.write_text('''# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Basic functionality
''')

# Usage example
def create_python_package(package_path: Path, package_name: str):
    """Create a complete Python package"""
    structure = PythonPackageStructure(package_path)
    structure.create_package_structure(package_name)
    
    print(f"Python package {package_name} created successfully")
```

### PyProject.toml Configuration

```python
# python/02-pyproject-config.py

"""
Modern pyproject.toml configuration
"""

from pathlib import Path
from typing import Dict, List, Optional
import toml

class PyProjectConfig:
    """Manage pyproject.toml configuration"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.pyproject_file = package_path / "pyproject.toml"
    
    def create_pyproject_toml(self, package_name: str, version: str = "0.1.0") -> bool:
        """Create comprehensive pyproject.toml"""
        try:
            config = {
                "build-system": {
                    "requires": ["hatchling"],
                    "build-backend": "hatchling.build"
                },
                "project": {
                    "name": package_name,
                    "version": version,
                    "description": "A modern Python package",
                    "readme": "README.md",
                    "requires-python": ">=3.11",
                    "license": {"text": "MIT"},
                    "authors": [
                        {"name": "Your Name", "email": "your.email@example.com"}
                    ],
                    "keywords": ["python", "package"],
                    "classifiers": [
                        "Development Status :: 3 - Alpha",
                        "Intended Audience :: Developers",
                        "License :: OSI Approved :: MIT License",
                        "Programming Language :: Python :: 3",
                        "Programming Language :: Python :: 3.11",
                        "Programming Language :: Python :: 3.12",
                    ],
                    "dependencies": [
                        "pydantic>=2.0.0",
                        "typer>=0.9.0",
                        "rich>=13.0.0"
                    ],
                    "optional-dependencies": {
                        "dev": [
                            "pytest>=7.0.0",
                            "pytest-cov>=4.0.0",
                            "pytest-mock>=3.10.0",
                            "black>=23.0.0",
                            "ruff>=0.1.0",
                            "mypy>=1.0.0",
                            "pre-commit>=3.0.0",
                            "tox>=4.0.0"
                        ],
                        "docs": [
                            "mkdocs>=1.5.0",
                            "mkdocs-material>=9.0.0",
                            "mkdocstrings[python]>=0.20.0"
                        ],
                        "test": [
                            "pytest>=7.0.0",
                            "pytest-cov>=4.0.0",
                            "pytest-xdist>=3.0.0"
                        ]
                    },
                    "scripts": {
                        f"{package_name}": f"{package_name}.cli:main"
                    }
                },
                "tool": {
                    "hatch": {
                        "build": {
                            "targets": ["wheel"]
                        },
                        "version": {
                            "path": f"src/{package_name}/__init__.py"
                        }
                    },
                    "black": {
                        "line-length": 88,
                        "target-version": ["py311"]
                    },
                    "ruff": {
                        "line-length": 88,
                        "target-version": "py311",
                        "select": [
                            "E",  # pycodestyle errors
                            "W",  # pycodestyle warnings
                            "F",  # pyflakes
                            "I",  # isort
                            "B",  # flake8-bugbear
                            "C4", # flake8-comprehensions
                            "UP", # pyupgrade
                        ],
                        "ignore": [
                            "E501",  # line too long, handled by black
                            "B008",  # do not perform function calls in argument defaults
                            "C901",  # too complex
                        ]
                    },
                    "mypy": {
                        "python_version": "3.11",
                        "warn_return_any": True,
                        "warn_unused_configs": True,
                        "disallow_untyped_defs": True,
                        "disallow_incomplete_defs": True,
                        "check_untyped_defs": True,
                        "disallow_untyped_decorators": True,
                        "no_implicit_optional": True,
                        "warn_redundant_casts": True,
                        "warn_unused_ignores": True,
                        "warn_no_return": True,
                        "warn_unreachable": True,
                        "strict_equality": True
                    },
                    "pytest": {
                        "ini_options": {
                            "testpaths": ["tests"],
                            "python_files": ["test_*.py"],
                            "python_classes": ["Test*"],
                            "python_functions": ["test_*"],
                            "addopts": [
                                "--strict-markers",
                                "--strict-config",
                                "--cov=src",
                                "--cov-report=term-missing",
                                "--cov-report=html",
                                "--cov-report=xml"
                            ]
                        }
                    },
                    "coverage": {
                        "run": {
                            "source": ["src"],
                            "omit": ["tests/*", "*/tests/*"]
                        },
                        "report": {
                            "exclude_lines": [
                                "pragma: no cover",
                                "def __repr__",
                                "raise AssertionError",
                                "raise NotImplementedError"
                            ]
                        }
                    }
                }
            }
            
            with open(self.pyproject_file, 'w') as f:
                toml.dump(config, f)
            
            return True
        except Exception:
            return False
    
    def add_dependency(self, package: str, version: Optional[str] = None) -> bool:
        """Add a dependency to pyproject.toml"""
        try:
            if self.pyproject_file.exists():
                with open(self.pyproject_file, 'r') as f:
                    config = toml.load(f)
            else:
                config = {"project": {"dependencies": []}}
            
            if "project" not in config:
                config["project"] = {}
            if "dependencies" not in config["project"]:
                config["project"]["dependencies"] = []
            
            dependency = f"{package}>={version}" if version else package
            if dependency not in config["project"]["dependencies"]:
                config["project"]["dependencies"].append(dependency)
            
            with open(self.pyproject_file, 'w') as f:
                toml.dump(config, f)
            
            return True
        except Exception:
            return False

# Usage example
def setup_pyproject_config(package_path: Path, package_name: str):
    """Setup pyproject.toml configuration"""
    config = PyProjectConfig(package_path)
    config.create_pyproject_toml(package_name)
    
    print("pyproject.toml configuration created successfully")
```

## Testing Framework

### Comprehensive Testing Setup

```python
# python/03-testing-framework.py

"""
Comprehensive testing framework for Python packages
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import pytest

class TestingFramework:
    """Manage testing framework and configuration"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.tests_path = package_path / "tests"
    
    def create_test_configuration(self) -> bool:
        """Create comprehensive test configuration"""
        try:
            # Create pytest.ini
            pytest_ini = self.package_path / "pytest.ini"
            pytest_ini.write_text('''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
''')
            
            # Create tox.ini
            tox_ini = self.package_path / "tox.ini"
            tox_ini.write_text('''[tox]
envlist = py311, py312, lint, docs
isolated_build = True

[testenv]
deps = 
    pytest>=7.0.0
    pytest-cov>=4.0.0
    pytest-mock>=3.10.0
commands = 
    pytest {posargs:tests}

[testenv:lint]
deps = 
    black>=23.0.0
    ruff>=0.1.0
    mypy>=1.0.0
commands = 
    black --check src tests
    ruff check src tests
    mypy src

[testenv:docs]
deps = 
    mkdocs>=1.5.0
    mkdocs-material>=9.0.0
    mkdocstrings[python]>=0.20.0
commands = 
    mkdocs build
''')
            
            return True
        except Exception:
            return False
    
    def create_test_utilities(self) -> bool:
        """Create test utilities and fixtures"""
        try:
            # Create test utilities
            test_utils = self.tests_path / "utils.py"
            test_utils.write_text('''"""
Test utilities and helpers
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
import pytest

class TestData:
    """Test data utilities"""
    
    @staticmethod
    def create_temp_file(content: str = "test content") -> Path:
        """Create a temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_temp_directory() -> Path:
        """Create a temporary directory"""
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def sample_dict() -> Dict[str, Any]:
        """Return sample dictionary for testing"""
        return {
            "string": "test",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }

@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Temporary file fixture"""
    temp_file = TestData.create_temp_file()
    yield temp_file
    temp_file.unlink()

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory fixture"""
    temp_dir = TestData.create_temp_directory()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Sample data fixture"""
    return TestData.sample_dict()
''')
            
            # Create test fixtures
            test_fixtures = self.tests_path / "fixtures.py"
            test_fixtures.write_text('''"""
Test fixtures for common test scenarios
"""

import pytest
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_api_client() -> Mock:
    """Mock API client fixture"""
    mock_client = Mock()
    mock_client.get.return_value = {"status": "success", "data": []}
    mock_client.post.return_value = {"status": "created", "id": 1}
    return mock_client

@pytest.fixture
def mock_database() -> Mock:
    """Mock database fixture"""
    mock_db = Mock()
    mock_db.query.return_value.all.return_value = []
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    return mock_db

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration fixture"""
    return {
        "database_url": "sqlite:///test.db",
        "api_key": "test_key",
        "debug": True,
        "log_level": "DEBUG"
    }
''')
            
            return True
        except Exception:
            return False
    
    def run_tests(self, args: Optional[List[str]] = None) -> bool:
        """Run tests with pytest"""
        try:
            cmd = ["python", "-m", "pytest"]
            if args:
                cmd.extend(args)
            
            result = subprocess.run(cmd, cwd=self.package_path, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

# Usage example
def setup_testing_framework(package_path: Path):
    """Setup comprehensive testing framework"""
    testing = TestingFramework(package_path)
    testing.create_test_configuration()
    testing.create_test_utilities()
    
    print("Testing framework setup complete")
```

## Code Quality

### Linting and Formatting

```python
# python/04-code-quality.py

"""
Code quality tools and configuration
"""

import subprocess
from pathlib import Path
from typing import List, Optional

class CodeQuality:
    """Manage code quality tools and configuration"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
    
    def setup_pre_commit_hooks(self) -> bool:
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
      - id: check-toml
      - id: check-json
  
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
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        types: [markdown, yaml]
'''
            
            with open(self.package_path / ".pre-commit-config.yaml", 'w') as f:
                f.write(pre_commit_config)
            
            # Install pre-commit hooks
            subprocess.run(["pre-commit", "install"], cwd=self.package_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_black(self, check_only: bool = False) -> bool:
        """Run Black code formatter"""
        try:
            cmd = ["black"]
            if check_only:
                cmd.append("--check")
            cmd.extend(["src", "tests"])
            
            subprocess.run(cmd, cwd=self.package_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_ruff(self, fix: bool = False) -> bool:
        """Run Ruff linter"""
        try:
            cmd = ["ruff", "check"]
            if fix:
                cmd.append("--fix")
            cmd.extend(["src", "tests"])
            
            subprocess.run(cmd, cwd=self.package_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_mypy(self) -> bool:
        """Run MyPy type checker"""
        try:
            subprocess.run(["mypy", "src"], cwd=self.package_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_all_quality_checks(self) -> bool:
        """Run all code quality checks"""
        try:
            # Run Black
            if not self.run_black(check_only=True):
                print("Black formatting issues found")
                return False
            
            # Run Ruff
            if not self.run_ruff():
                print("Ruff linting issues found")
                return False
            
            # Run MyPy
            if not self.run_mypy():
                print("MyPy type checking issues found")
                return False
            
            print("All code quality checks passed")
            return True
        except Exception:
            return False

# Usage example
def setup_code_quality(package_path: Path):
    """Setup code quality tools"""
    quality = CodeQuality(package_path)
    quality.setup_pre_commit_hooks()
    
    print("Code quality setup complete")
```

## Documentation

### Modern Documentation Setup

```python
# python/05-documentation.py

"""
Modern documentation setup with MkDocs
"""

from pathlib import Path
from typing import Dict, List
import yaml

class DocumentationSetup:
    """Manage documentation setup and configuration"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.docs_path = package_path / "docs"
    
    def create_mkdocs_config(self) -> bool:
        """Create MkDocs configuration"""
        try:
            mkdocs_config = {
                "site_name": "Python Package Documentation",
                "site_description": "Modern Python package documentation",
                "site_url": "https://your-package.readthedocs.io/",
                "repo_url": "https://github.com/your-username/your-package",
                "repo_name": "your-username/your-package",
                "edit_uri": "edit/main/docs/",
                "theme": {
                    "name": "material",
                    "palette": {
                        "primary": "blue",
                        "accent": "blue"
                    },
                    "features": [
                        "navigation.tabs",
                        "navigation.sections",
                        "navigation.top",
                        "search.highlight",
                        "search.share"
                    ]
                },
                "markdown_extensions": [
                    "admonition",
                    "codehilite",
                    "fenced_code",
                    "toc",
                    "tables",
                    "mkdocstrings",
                    "pymdownx.superfences",
                    "pymdownx.tabbed"
                ],
                "plugins": [
                    "mkdocstrings",
                    "mkdocs-material"
                ],
                "nav": [
                    {"Home": "index.md"},
                    {"API Reference": "api.md"},
                    {"Examples": "examples.md"},
                    {"Contributing": "contributing.md"}
                ]
            }
            
            with open(self.package_path / "mkdocs.yml", 'w') as f:
                yaml.dump(mkdocs_config, f, default_flow_style=False)
            
            return True
        except Exception:
            return False
    
    def create_documentation_structure(self) -> bool:
        """Create documentation structure"""
        try:
            # Create documentation files
            docs_files = {
                "index.md": """# Python Package

A modern Python package built with best practices.

## Quick Start

```python
from your_package import hello_world

print(hello_world())
```

## Installation

```bash
pip install your-package
```

## Features

- Modern Python packaging
- Comprehensive testing
- Type hints
- Documentation
""",
                "api.md": """# API Reference

## Main Module

::: your_package.main
    options:
      show_source: true
      show_root_heading: true
""",
                "examples.md": """# Examples

## Basic Usage

```python
from your_package import hello_world

# Simple usage
message = hello_world()
print(message)
```

## Advanced Usage

```python
from your_package import get_version

# Get version
version = get_version()
print(f"Version: {version}")
```
""",
                "contributing.md": """# Contributing

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -e .[dev]`
3. Run tests: `pytest`
4. Run linting: `ruff check .`
5. Format code: `black .`

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request
"""
            }
            
            for filename, content in docs_files.items():
                with open(self.docs_path / filename, 'w') as f:
                    f.write(content)
            
            return True
        except Exception:
            return False

# Usage example
def setup_documentation(package_path: Path):
    """Setup documentation"""
    docs = DocumentationSetup(package_path)
    docs.create_mkdocs_config()
    docs.create_documentation_structure()
    
    print("Documentation setup complete")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Create package structure
from pathlib import Path
package_path = Path("my-package")
package_path.mkdir(exist_ok=True)

# 2. Setup package structure
from python.package_structure import create_python_package
create_python_package(package_path, "my_package")

# 3. Setup pyproject.toml
from python.pyproject_config import setup_pyproject_config
setup_pyproject_config(package_path, "my_package")

# 4. Setup testing
from python.testing_framework import setup_testing_framework
setup_testing_framework(package_path)

# 5. Setup code quality
from python.code_quality import setup_code_quality
setup_code_quality(package_path)

# 6. Setup documentation
from python.documentation import setup_documentation
setup_documentation(package_path)
```

### Essential Patterns

```python
# Complete Python package development setup
def create_complete_python_package(package_path: Path, package_name: str):
    """Create complete Python package with all best practices"""
    
    # Create package structure
    structure = PythonPackageStructure(package_path)
    structure.create_package_structure(package_name)
    
    # Setup pyproject.toml
    config = PyProjectConfig(package_path)
    config.create_pyproject_toml(package_name)
    
    # Setup testing
    testing = TestingFramework(package_path)
    testing.create_test_configuration()
    testing.create_test_utilities()
    
    # Setup code quality
    quality = CodeQuality(package_path)
    quality.setup_pre_commit_hooks()
    
    # Setup documentation
    docs = DocumentationSetup(package_path)
    docs.create_mkdocs_config()
    docs.create_documentation_structure()
    
    print(f"Complete Python package {package_name} created successfully!")
```

---

*This guide provides the complete machinery for Python package development. Each pattern includes implementation examples, configuration strategies, and real-world usage patterns for enterprise Python packaging.*
