# Python Package Best Practices

This document establishes the definitive approach to building and shipping Python packages that refuse to break. Every command is copy-paste runnable, every configuration is auditable, and every practice eliminates "works on my machine" entropy. We enforce locked environments, typed code, fast tests, reproducible builds, and CI that refuses to smile.

## 1. Environment & Tooling (use uv + pyproject.toml)

### Install uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv --version
```

**Why:** uv provides deterministic, fast installs with single-source metadata and fewer moving parts than traditional pip/poetry workflows.

### Bootstrap project

```bash
uv init your_package
cd your_package
uv python pin 3.12
uv add -d ruff mypy pytest pytest-asyncio pytest-cov hypothesis
uv add -d nox pip-audit build twine
```

**Why:** Pin the interpreter to prevent version drift, lock dev tools for consistency, and maintain speed without sacrificing sanity.

### pyproject.toml minimal

```toml
[project]
name = "your-package"
version = "0.1.0"
description = "Short, sharp purpose."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "Your Name" }]
dependencies = ["pydantic>=2.7"]

[project.optional-dependencies]
dev = ["ruff", "pytest", "pytest-asyncio", "pytest-cov", "mypy", "hypothesis"]
docs = ["mkdocs-material", "mkdocstrings[python]", "pymdown-extensions"]

[tool.uv]
default-groups = ["dev"]

[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"
```

**Why:** hatchling keeps builds simple and predictable. Optional dependencies fuel extras without bloat. uv dev group makes local development frictionless.

## 2. Virtual Envs, Locking, and Reproducibility

### Create & enter env

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### Lock & export

```bash
uv lock
uv export --no-hashes -o requirements.lock.txt
```

### Pin Python

Create `.python-version`:
```
3.12
```

**Why:** Locked dependencies plus pinned Python equals reproducible installs on hostile machines. The repository owns the interpreter; no global drift allowed.

## 3. Code Layout & Packaging

### Project structure

```
your_package/
  src/your_package/__init__.py
  src/your_package/core.py
  tests/
  pyproject.toml
  README.md
```

### Public API in __init__.py

```python
# src/your_package/__init__.py
from .core import add, multiply

__all__ = ["add", "multiply"]
__version__ = "0.1.0"
```

**Why:** src layout prevents tests from importing the working tree by accident. Explicit __all__ defines the public API contract and prevents implicit namespace pollution.

## 4. Linting, Formatting, Types (Ruff + Mypy)

### Ruff configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E","F","I","UP","B","SIM","ASYNC","PIE","TID","PTH","RUF"]
ignore = ["E501"]  # rely on line-length; trim as needed
```

**Commands:**
```bash
uv run ruff format .
uv run ruff check . --fix
```

### Mypy configuration

```ini
# mypy.ini
[mypy]
python_version = 3.12
warn_unused_ignores = True
disallow_untyped_defs = True
no_implicit_optional = True
strict_equality = True
```

**Run:**
```bash
uv run mypy src
```

**Why:** Fast feedback with typed interfaces prevents nocturnal regrets. Ruff replaces multiple tools with a single, blazing-fast linter and formatter.

## 5. Tests: Fast, Parallel, Measurable

### pytest skeleton

```python
# tests/test_core.py
from your_package.core import add

def test_add():
    assert add(1, 2) == 3
```

### Parallel & coverage

```bash
uv run pytest -q
uv run pytest -n auto --dist=loadfile --maxfail=1 --cov=your_package --cov-report=xml
```

### Property-based testing with Hypothesis

```python
from hypothesis import given, strategies as st
from your_package.core import add

@given(st.integers(), st.integers())
def test_add_holds(a, b):
    assert add(a, b) == add(b, a)
```

**Why:** Parallel tests gate performance regressions. Hypothesis catches the gremlins you didn't imagine with property-based testing that explores edge cases automatically.

## 6. Nox Automation (one command to rule them all)

```python
# noxfile.py
import nox

@nox.session
def lint(session):
    session.install("uv")
    session.run("uv", "run", "ruff", "format", "--check", ".")
    session.run("uv", "run", "ruff", "check", ".")

@nox.session
def types(session):
    session.run("uv", "run", "mypy", "src")

@nox.session
def tests(session):
    session.run("uv", "run", "pytest", "-n", "auto", "--cov=your_package", "--cov-report=term-missing")
```

**Why:** Standard entry points for CI and development eliminate bike-shedding about which incantation to use. Nox provides reproducible environments across different machines.

## 7. Security & Supply Chain

### Advisories

```bash
uv run pip-audit
```

### Runtime hygiene

- Never import from tests
- Kill wildcard imports
- Minimal __all__ exports
- .env for dev only (use real secret managers in prod)
- Never commit tokens

**Why:** The adversary is time and your past self. Audit early, often, and automatically to prevent supply chain vulnerabilities.

## 8. Docs (MkDocs + mkdocstrings)

### Add doc dependencies

```bash
uv add -g docs mkdocs-material mkdocstrings[python] pymdown-extensions
```

### Minimal mkdocs.yml

```yaml
theme:
  name: material
plugins:
  - mkdocstrings:
      default_handler: python
markdown_extensions:
  - admonition
  - toc:
      permalink: true
nav:
  - Reference:
    - API: reference.md
```

### Autodoc page

```markdown
# reference.md
# API
::: your_package.core
```

**Why:** Documentation rots slower when generated from signatures and docstrings. Automated API docs stay in sync with code changes.

## 9. Versioning & Changelog Discipline

### Semantic Versioning

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Release from CI only
- Keep __version__ in one place or use dynamic versioning via VCS tags

### Changelog generation

```bash
# Install git-cliff
cargo install git-cliff

# Generate changelog
git cliff --init
git cliff --unreleased
```

**Why:** Traceable releases beat folklore. Automated changelog generation ensures consistent documentation of changes.

## 10. Building, Publishing, and Artifacts

### Build sdist + wheel

```bash
uv run python -m build
```

### Check artifacts

```bash
twine check dist/*
```

### Upload (optional)

```bash
twine upload dist/*
```

**Why:** Wheels and sdists must be verifiable and boring. Build artifacts in CI only; no local heroics allowed.

## 11. Docker (slim, cache-friendly)

```dockerfile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (as needed)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Build layer
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip build && python -m build

# Runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=base /app/dist/*.whl /tmp/pkg.whl
RUN pip install --no-cache-dir /tmp/pkg.whl && rm /tmp/pkg.whl
CMD ["python", "-c", "import your_package; print('ok')"]
```

**Why:** Multi-stage builds keep runtime lean. Wheels ensure repeatability and faster container startup times.

## 12. GitHub Actions CI (matrix, cache, gates)

```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync env
        run: |
          uv venv
          . .venv/bin/activate
          uv sync
      - name: Lint & Types
        run: |
          . .venv/bin/activate
          uv run ruff format --check .
          uv run ruff check .
          uv run mypy src
      - name: Tests
        run: |
          . .venv/bin/activate
          uv run pytest -n auto --cov=your_package --cov-report=xml
      - name: Build
        run: |
          . .venv/bin/activate
          uv run python -m build
      - name: Upload coverage
        if: always()
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: false
```

**Why:** Interpreter matrix catches version rot. CI enforces the same rituals as local development with no exceptions.

## 13. Pre-commit Hooks (stop nonsense at the gate)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.8
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: ["types-requests"]
```

**Enable:**
```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

**Why:** It's cheaper to refuse bad code than to fix it later. Pre-commit hooks catch issues before they reach the repository.

## 14. Troubleshooting Playbook (short, lethal)

### Import succeeds in tests but fails in prod
→ Missing src/ layout or wheel not built; build and install the wheel locally to verify.

### Mysterious type errors
→ Check mypy.ini vs. runtime; stub missing? Add types-<pkg>.

### Slow installs
→ Lock with uv lock; avoid editable installs for production images; cache pip dir in CI.

### Native build errors
→ Missing build-essential, pkg-config, or headers; inspect pip debug -v.

**Why:** These solutions address 90% of Python packaging issues. The key is understanding that reproducible builds require consistent environments.

## 15. TL;DR (Zero → Green)

```bash
# 1) Tooling
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2) Project
uv init mylib && cd mylib
uv python pin 3.12
uv add -d ruff mypy pytest pytest-asyncio pytest-cov hypothesis nox pip-audit build twine

# 3) Sanity
uv run ruff format .
uv run ruff check . --fix
uv run mypy src
uv run pytest -n auto --maxfail=1

# 4) Package
uv run python -m build
twine check dist/*
```

**Why:** This sequence establishes a working Python package environment with all essential tools in under 5 minutes. Each command builds on the previous, ensuring a deterministic setup that matches production CI environments.
