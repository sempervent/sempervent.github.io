# Python Code Quality Best Practices

**Objective**: Master senior-level Python code quality patterns for production systems. When you need to maintain high code standards, when you want to implement comprehensive quality checks, when you need enterprise-grade code quality workflows—these best practices become your weapon of choice.

## Core Principles

- **Consistency**: Maintain consistent code style and patterns
- **Readability**: Write code that is easy to read and understand
- **Maintainability**: Design code that is easy to modify and extend
- **Automation**: Automate quality checks and enforcement
- **Continuous Improvement**: Continuously improve code quality standards

## Linting and Formatting

### Modern Linting Tools

```python
# python/01-linting-formatting.py

"""
Modern Python linting and formatting tools
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

class CodeQualityTools:
    """Manage code quality tools and configuration"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.config_files = {
            "ruff": project_path / "pyproject.toml",
            "black": project_path / "pyproject.toml",
            "mypy": project_path / "pyproject.toml",
            "isort": project_path / "pyproject.toml"
        }
    
    def setup_ruff_config(self) -> bool:
        """Setup Ruff linter configuration"""
        try:
            ruff_config = '''[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
    "N",  # pep8-naming
    "S",  # flake8-bandit
    "T20", # flake8-print
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    "ARG", # flake8-unused-arguments
    "PTH", # flake8-use-pathlib
    "ERA", # eradicate
    "PD",  # pandas-vet
    "PGH", # pygrep-hooks
    "PL",  # pylint
    "TRY", # tryceratops
    "FLY", # flynt
    "NPY", # numpy
    "RUF", # ruff-specific rules
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
    "PLR0913", # too many arguments
    "PLR0912", # too many branches
    "PLR0915", # too many statements
    "S101",  # use of assert
    "S104",  # hardcoded bind all interfaces
    "S108",  # hardcoded temp file
    "S311",  # standard pseudo-random generators
    "S603",  # subprocess call
    "S607",  # starting a process with a partial executable path
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*.py" = ["S101", "PLR2004", "S106", "S108"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
force-single-line = false

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pylint]
max-args = 5
max-branches = 12
max-returns = 6
max-statements = 50
'''
            
            # Update pyproject.toml with ruff config
            self._update_pyproject_toml("ruff", ruff_config)
            return True
        except Exception:
            return False
    
    def setup_black_config(self) -> bool:
        """Setup Black formatter configuration"""
        try:
            black_config = '''[tool.black]
line-length = 88
target-version = ['py311']
include = '\\.pyi?$'
extend-exclude = ''
'''
            # Update pyproject.toml with black config
            self._update_pyproject_toml("black", black_config)
            return True
        except Exception:
            return False
    
    def setup_mypy_config(self) -> bool:
        """Setup MyPy type checker configuration"""
        try:
            mypy_config = '''[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
show_error_codes = true
show_column_numbers = true
show_error_context = true
pretty = true
color_output = true
error_summary = true
'''
            
            # Update pyproject.toml with mypy config
            self._update_pyproject_toml("mypy", mypy_config)
            return True
        except Exception:
            return False
    
    def _update_pyproject_toml(self, tool: str, config: str) -> bool:
        """Update pyproject.toml with tool configuration"""
        try:
            pyproject_file = self.project_path / "pyproject.toml"
            
            if pyproject_file.exists():
                with open(pyproject_file, 'r') as f:
                    content = f.read()
            else:
                content = ""
            
            # Add tool configuration
            if f"[tool.{tool}]" not in content:
                content += f"\n{config}\n"
            
            with open(pyproject_file, 'w') as f:
                f.write(content)
            
            return True
        except Exception:
            return False
    
    def run_ruff_check(self, fix: bool = False) -> bool:
        """Run Ruff linter"""
        try:
            cmd = ["ruff", "check"]
            if fix:
                cmd.append("--fix")
            cmd.extend(["src", "tests"])
            
            result = subprocess.run(cmd, cwd=self.project_path, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def run_black_check(self, check_only: bool = False) -> bool:
        """Run Black formatter"""
        try:
            cmd = ["black"]
            if check_only:
                cmd.append("--check")
            cmd.extend(["src", "tests"])
            
            result = subprocess.run(cmd, cwd=self.project_path, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def run_mypy_check(self) -> bool:
        """Run MyPy type checker"""
        try:
            result = subprocess.run(["mypy", "src"], cwd=self.project_path, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def run_all_quality_checks(self) -> Dict[str, bool]:
        """Run all code quality checks"""
        results = {
            "ruff": self.run_ruff_check(),
            "black": self.run_black_check(check_only=True),
            "mypy": self.run_mypy_check()
        }
        
        return results

# Usage example
def setup_code_quality_tools(project_path: Path):
    """Setup comprehensive code quality tools"""
    tools = CodeQualityTools(project_path)
    
    # Setup configurations
    tools.setup_ruff_config()
    tools.setup_black_config()
    tools.setup_mypy_config()
    
    # Run quality checks
    results = tools.run_all_quality_checks()
    
    print("Code quality tools setup complete")
    print(f"Quality check results: {results}")
```

### Pre-commit Hooks

```python
# python/02-pre-commit-hooks.py

"""
Pre-commit hooks for automated code quality
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Any

class PreCommitSetup:
    """Setup and manage pre-commit hooks"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.pre_commit_config = project_path / ".pre-commit-config.yaml"
    
    def create_pre_commit_config(self) -> bool:
        """Create comprehensive pre-commit configuration"""
        try:
            config_content = '''repos:
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
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first
      - id: requirements-txt-fixer
      - id: mixed-line-ending
      - id: check-case-conflict
      - id: check-merge-conflict
      - id: check-symlinks
      - id: check-ast
      - id: check-builtin-literals
      - id: check-json
      - id: check-toml
      - id: check-xml
      - id: check-yaml
      - id: detect-private-key
      - id: end-of-file-fixer
      - id: fix-byte-order-marker
      - id: mixed-line-ending
      - id: trailing-whitespace

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

  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.0.0
    hooks:
      - id: eslint
        files: \\.(js|jsx|ts|tsx)$
        types: [file]

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        types: [css, html, javascript, json, markdown, yaml]
'''
            
            with open(self.pre_commit_config, 'w') as f:
                f.write(config_content)
            
            return True
        except Exception:
            return False
    
    def install_pre_commit_hooks(self) -> bool:
        """Install pre-commit hooks"""
        try:
            subprocess.run(["pre-commit", "install"], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_pre_commit_hooks(self) -> bool:
        """Run pre-commit hooks on all files"""
        try:
            subprocess.run(["pre-commit", "run", "--all-files"], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def update_pre_commit_hooks(self) -> bool:
        """Update pre-commit hooks"""
        try:
            subprocess.run(["pre-commit", "autoupdate"], cwd=self.project_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

# Usage example
def setup_pre_commit_quality(project_path: Path):
    """Setup pre-commit hooks for code quality"""
    pre_commit = PreCommitSetup(project_path)
    
    # Create configuration
    pre_commit.create_pre_commit_config()
    
    # Install hooks
    pre_commit.install_pre_commit_hooks()
    
    # Run hooks
    pre_commit.run_pre_commit_hooks()
    
    print("Pre-commit hooks setup complete")
```

## Code Review Standards

### Review Checklist

```python
# python/03-code-review.py

"""
Code review standards and checklist
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ReviewStatus(Enum):
    """Review status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"

@dataclass
class CodeReviewItem:
    """Code review item"""
    category: str
    description: str
    severity: str  # "critical", "major", "minor", "suggestion"
    line_number: Optional[int] = None
    file_path: Optional[str] = None

class CodeReviewChecklist:
    """Code review checklist and standards"""
    
    def __init__(self):
        self.checklist_items = self._create_checklist()
    
    def _create_checklist(self) -> List[CodeReviewItem]:
        """Create comprehensive code review checklist"""
        return [
            # Code Quality
            CodeReviewItem(
                category="Code Quality",
                description="Code follows PEP 8 style guidelines",
                severity="major"
            ),
            CodeReviewItem(
                category="Code Quality",
                description="Functions and classes have appropriate docstrings",
                severity="major"
            ),
            CodeReviewItem(
                category="Code Quality",
                description="Variable and function names are descriptive",
                severity="major"
            ),
            CodeReviewItem(
                category="Code Quality",
                description="Code is properly formatted with Black",
                severity="major"
            ),
            CodeReviewItem(
                category="Code Quality",
                description="No unused imports or variables",
                severity="minor"
            ),
            
            # Type Hints
            CodeReviewItem(
                category="Type Hints",
                description="All function parameters have type hints",
                severity="major"
            ),
            CodeReviewItem(
                category="Type Hints",
                description="Function return types are specified",
                severity="major"
            ),
            CodeReviewItem(
                category="Type Hints",
                description="Type hints are accurate and specific",
                severity="major"
            ),
            
            # Testing
            CodeReviewItem(
                category="Testing",
                description="New code has corresponding tests",
                severity="critical"
            ),
            CodeReviewItem(
                category="Testing",
                description="Tests cover edge cases and error conditions",
                severity="major"
            ),
            CodeReviewItem(
                category="Testing",
                description="Test names are descriptive and follow naming conventions",
                severity="minor"
            ),
            CodeReviewItem(
                category="Testing",
                description="Tests are independent and can run in any order",
                severity="major"
            ),
            
            # Security
            CodeReviewItem(
                category="Security",
                description="No hardcoded secrets or credentials",
                severity="critical"
            ),
            CodeReviewItem(
                category="Security",
                description="Input validation is implemented where needed",
                severity="major"
            ),
            CodeReviewItem(
                category="Security",
                description="SQL injection prevention measures are in place",
                severity="critical"
            ),
            CodeReviewItem(
                category="Security",
                description="Authentication and authorization are properly implemented",
                severity="critical"
            ),
            
            # Performance
            CodeReviewItem(
                category="Performance",
                description="No obvious performance bottlenecks",
                severity="major"
            ),
            CodeReviewItem(
                category="Performance",
                description="Database queries are optimized",
                severity="major"
            ),
            CodeReviewItem(
                category="Performance",
                description="Memory usage is reasonable",
                severity="major"
            ),
            
            # Documentation
            CodeReviewItem(
                category="Documentation",
                description="README is updated if needed",
                severity="minor"
            ),
            CodeReviewItem(
                category="Documentation",
                description="API documentation is updated",
                severity="minor"
            ),
            CodeReviewItem(
                category="Documentation",
                description="Code comments explain complex logic",
                severity="minor"
            ),
            
            # Architecture
            CodeReviewItem(
                category="Architecture",
                description="Code follows established patterns and conventions",
                severity="major"
            ),
            CodeReviewItem(
                category="Architecture",
                description="Separation of concerns is maintained",
                severity="major"
            ),
            CodeReviewItem(
                category="Architecture",
                description="Dependencies are minimal and justified",
                severity="minor"
            )
        ]
    
    def review_code(self, file_path: str, code_content: str) -> List[CodeReviewItem]:
        """Review code against checklist"""
        issues = []
        
        # Basic checks
        if "TODO" in code_content or "FIXME" in code_content:
            issues.append(CodeReviewItem(
                category="Code Quality",
                description="Code contains TODO or FIXME comments",
                severity="minor",
                file_path=file_path
            ))
        
        if "print(" in code_content:
            issues.append(CodeReviewItem(
                category="Code Quality",
                description="Code contains print statements (use logging instead)",
                severity="minor",
                file_path=file_path
            ))
        
        if "import *" in code_content:
            issues.append(CodeReviewItem(
                category="Code Quality",
                description="Code uses wildcard imports",
                severity="major",
                file_path=file_path
            ))
        
        return issues
    
    def generate_review_report(self, issues: List[CodeReviewItem]) -> Dict[str, Any]:
        """Generate code review report"""
        report = {
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.severity == "critical"]),
            "major_issues": len([i for i in issues if i.severity == "major"]),
            "minor_issues": len([i for i in issues if i.severity == "minor"]),
            "suggestions": len([i for i in issues if i.severity == "suggestion"]),
            "issues_by_category": {},
            "issues_by_severity": {}
        }
        
        # Group by category
        for issue in issues:
            if issue.category not in report["issues_by_category"]:
                report["issues_by_category"][issue.category] = []
            report["issues_by_category"][issue.category].append(issue)
        
        # Group by severity
        for issue in issues:
            if issue.severity not in report["issues_by_severity"]:
                report["issues_by_severity"][issue.severity] = []
            report["issues_by_severity"][issue.severity].append(issue)
        
        return report

# Usage example
def perform_code_review(file_path: str, code_content: str):
    """Perform code review"""
    reviewer = CodeReviewChecklist()
    
    # Review code
    issues = reviewer.review_code(file_path, code_content)
    
    # Generate report
    report = reviewer.generate_review_report(issues)
    
    print(f"Code review completed for {file_path}")
    print(f"Total issues: {report['total_issues']}")
    print(f"Critical: {report['critical_issues']}, Major: {report['major_issues']}")
    
    return report
```

## Automated Quality Gates

### CI/CD Integration

```python
# python/04-quality-gates.py

"""
Automated quality gates for CI/CD pipelines
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class QualityGate:
    """Quality gate configuration"""
    name: str
    command: List[str]
    threshold: float
    required: bool = True

class QualityGateManager:
    """Manage quality gates for CI/CD"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.quality_gates = self._create_quality_gates()
    
    def _create_quality_gates(self) -> List[QualityGate]:
        """Create quality gates"""
        return [
            QualityGate(
                name="Linting",
                command=["ruff", "check", "src", "tests"],
                threshold=0.0,
                required=True
            ),
            QualityGate(
                name="Formatting",
                command=["black", "--check", "src", "tests"],
                threshold=0.0,
                required=True
            ),
            QualityGate(
                name="Type Checking",
                command=["mypy", "src"],
                threshold=0.0,
                required=True
            ),
            QualityGate(
                name="Security Scan",
                command=["bandit", "-r", "src"],
                threshold=0.0,
                required=True
            ),
            QualityGate(
                name="Test Coverage",
                command=["pytest", "--cov=src", "--cov-report=json"],
                threshold=80.0,
                required=True
            )
        ]
    
    def run_quality_gate(self, gate: QualityGate) -> Dict[str, Any]:
        """Run a single quality gate"""
        try:
            result = subprocess.run(
                gate.command,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "name": gate.name,
                "status": "passed",
                "output": result.stdout,
                "threshold": gate.threshold,
                "required": gate.required
            }
        except subprocess.CalledProcessError as e:
            return {
                "name": gate.name,
                "status": "failed",
                "output": e.stdout,
                "error": e.stderr,
                "threshold": gate.threshold,
                "required": gate.required
            }
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates"""
        results = {
            "gates": [],
            "overall_status": "passed",
            "required_failed": 0,
            "total_gates": len(self.quality_gates)
        }
        
        for gate in self.quality_gates:
            result = self.run_quality_gate(gate)
            results["gates"].append(result)
            
            if result["status"] == "failed" and gate.required:
                results["required_failed"] += 1
                results["overall_status"] = "failed"
        
        return results
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate quality report"""
        report = f"""
# Quality Gate Report

## Overall Status: {results['overall_status'].upper()}

### Summary
- Total Gates: {results['total_gates']}
- Required Failures: {results['required_failed']}

### Gate Results
"""
        
        for gate in results["gates"]:
            status_emoji = "✅" if gate["status"] == "passed" else "❌"
            report += f"- {status_emoji} **{gate['name']}**: {gate['status']}\n"
        
        return report

# Usage example
def run_quality_gates(project_path: Path):
    """Run all quality gates"""
    manager = QualityGateManager(project_path)
    
    # Run quality gates
    results = manager.run_all_quality_gates()
    
    # Generate report
    report = manager.generate_quality_report(results)
    
    print(report)
    
    # Exit with appropriate code
    if results["overall_status"] == "failed":
        exit(1)
    else:
        exit(0)
```

## TL;DR Runbook

### Quick Start

```python
# 1. Setup code quality tools
from pathlib import Path
project_path = Path("my-project")

from python.code_quality import setup_code_quality_tools
setup_code_quality_tools(project_path)

# 2. Setup pre-commit hooks
from python.pre_commit_hooks import setup_pre_commit_quality
setup_pre_commit_quality(project_path)

# 3. Run quality gates
from python.quality_gates import run_quality_gates
run_quality_gates(project_path)
```

### Essential Patterns

```python
# Complete code quality setup
def setup_complete_code_quality(project_path: Path):
    """Setup complete code quality framework"""
    
    # Setup quality tools
    tools = CodeQualityTools(project_path)
    tools.setup_ruff_config()
    tools.setup_black_config()
    tools.setup_mypy_config()
    
    # Setup pre-commit hooks
    pre_commit = PreCommitSetup(project_path)
    pre_commit.create_pre_commit_config()
    pre_commit.install_pre_commit_hooks()
    
    # Setup quality gates
    manager = QualityGateManager(project_path)
    
    print("Complete code quality framework setup!")
```

---

*This guide provides the complete machinery for Python code quality. Each pattern includes implementation examples, automation strategies, and real-world usage patterns for enterprise code quality management.*
