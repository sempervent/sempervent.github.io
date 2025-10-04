# Pytest Best Practices: Coverage, Plugins, Speed, CI

**Objective**: Master pytest for production-grade testing that's fast, reliable, and maintainable. When you need comprehensive test coverage, when you want to prevent regressions, when you're building scalable test suites—pytest becomes your weapon of choice.

Pytest is the foundation of modern Python testing. Proper pytest configuration enables fast, reliable testing that prevents bugs and maintains code quality. This guide shows you how to wield pytest with the precision of a battle-tested backend engineer, covering everything from basic configuration to advanced CI integration.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand test organization**
   - Project layout and naming conventions
   - Test collection and discovery rules
   - Fixture scope and dependency management
   - Marker usage and test categorization

2. **Master core pytest features**
   - Fixtures, parametrization, and monkeypatching
   - Async testing and timeout handling
   - Assert rewriting and error reporting
   - Skip and xfail patterns

3. **Know your coverage patterns**
   - Line and branch coverage requirements
   - Coverage thresholds and reporting
   - HTML reports and gap analysis
   - Coverage exclusions and pragmas

4. **Validate everything**
   - Test isolation and parallel execution
   - CI integration and artifact generation
   - Performance benchmarking and monitoring
   - Flaky test detection and quarantine

5. **Plan for production**
   - Plugin management and version pinning
   - CI caching and optimization
   - Test data factories and isolation
   - Documentation and maintenance

**Why These Principles**: Pytest requires understanding both testing mechanics and CI integration patterns. Understanding these patterns prevents flaky tests and enables reliable test automation.

## 1) Project Layout and Collection

### Standard Test Organization

```
repo/
├── src/
│   └── yourpkg/
│       ├── __init__.py
│       ├── core.py
│       └── api.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_core.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── e2e/
│       └── test_workflows.py
├── pyproject.toml
└── .coveragerc
```

**Why Layout Matters**: Organized test structure enables maintainability and clear separation of concerns. Understanding these patterns prevents test confusion and enables efficient test discovery.

### Naming Conventions

```python
# Test files: test_*.py
test_core.py
test_api.py
test_database.py

# Test functions: test_*
def test_user_creation():
    pass

def test_api_endpoint():
    pass

# Test classes: Test* (no __init__)
class TestUserModel:
    def test_validation(self):
        pass
```

**Why Naming Matters**: Consistent naming enables automatic test discovery and clear test organization. Understanding these patterns prevents test collection issues and enables efficient test execution.

## 2) Minimal, Opinionated Configuration

### pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
minversion = "7.4"
addopts = """
  -ra
  -q
  --strict-markers
  --strict-config
  --maxfail=1
  --disable-warnings
  --cov=yourpkg
  --cov-branch
  --cov-report=term-missing
  --cov-report=html
  --no-cov-on-fail
"""
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

markers = [
  "unit: fast, pure in-memory tests",
  "integration: touches DB/network/filesystem",
  "e2e: slow end-to-end tests",
  "slow: long-running; excluded by default",
]

filterwarnings = [
  "error::DeprecationWarning",
  "ignore::ResourceWarning",
]
```

**Why Configuration Matters**: Strict configuration prevents typos and ensures consistent test execution. Understanding these patterns prevents configuration drift and enables reliable test automation.

### pytest.ini Alternative

```ini
[tool:pytest]
minversion = 7.4
addopts = -ra -q --strict-markers --strict-config --maxfail=1
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: fast, pure in-memory tests
    integration: touches DB/network/filesystem
    e2e: slow end-to-end tests
    slow: long-running; excluded by default

filterwarnings =
    error::DeprecationWarning
    ignore::ResourceWarning
```

**Why Alternative Config Matters**: Some teams prefer pytest.ini for simplicity. Understanding these patterns prevents configuration conflicts and enables team flexibility.

## 3) Core Plugins and Dependencies

### requirements-dev.txt

```txt
# Core testing
pytest==8.3.*
pytest-cov==5.0.*
pytest-xdist==3.6.*
pytest-randomly==3.15.*

# Async and mocking
pytest-asyncio==0.23.*
pytest-mock==3.14.*

# Time and randomness
freezegun==1.5.*

# Property-based testing
hypothesis==6.112.*

# Performance and timeouts
pytest-benchmark==4.0.*
pytest-timeout==2.3.*

# Network mocking
responses==0.25.*
vcrpy==6.0.*

# Snapshots
syrupy==4.6.*

# Flaky test handling
pytest-rerunfailures==2.2.*
```

**Why Plugin Management Matters**: Pinned versions prevent CI whiplash and ensure reproducible test environments. Understanding these patterns prevents dependency conflicts and enables reliable test execution.

## 4) conftest.py: Global Fixtures and Configuration

### Global Fixtures

```python
# tests/conftest.py
import os
import pathlib
import pytest
from typing import Generator

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end tests")
    config.addinivalue_line("markers", "slow: long-running tests")

@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Get project root directory."""
    return pathlib.Path(__file__).resolve().parents[1]

@pytest.fixture
def tmp_txt(tmp_path):
    """Create temporary text file."""
    p = tmp_path / "data.txt"
    p.write_text("hello")
    return p

@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "settings": {"theme": "dark", "notifications": True}
    }
```

**Why Global Fixtures Matter**: Centralized fixtures reduce duplication and provide consistent test data. Understanding these patterns prevents fixture soup and enables maintainable test code.

### Database Fixtures

```python
# tests/conftest.py (continued)
import tempfile
import sqlite3
from pathlib import Path

@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    yield conn
    conn.close()

@pytest.fixture
def db_with_data(temp_db):
    """Database with sample data."""
    cursor = temp_db.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob')")
    temp_db.commit()
    return temp_db
```

**Why Database Fixtures Matter**: Isolated database fixtures prevent test contamination and enable reliable database testing. Understanding these patterns prevents test flakiness and enables consistent database testing.

## 5) Core Pytest Features

### Fixtures and Parametrization

```python
# tests/unit/test_math.py
import pytest

@pytest.fixture
def calculator():
    """Provide calculator instance."""
    from yourpkg.calculator import Calculator
    return Calculator()

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5)
])
def test_add(calculator, a, b, expected):
    """Test addition with various inputs."""
    assert calculator.add(a, b) == expected

@pytest.fixture
def env_home(monkeypatch, tmp_path):
    """Set HOME environment variable."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home

def test_file_creation(env_home):
    """Test file creation in custom home."""
    from yourpkg.file_utils import create_config
    config_path = create_config()
    assert config_path.exists()
    assert str(config_path).startswith(str(env_home))
```

**Why Core Features Matter**: Proper fixture usage and parametrization enable comprehensive test coverage with minimal code duplication. Understanding these patterns prevents test bloat and enables efficient test maintenance.

### Async Testing

```python
# tests/unit/test_async.py
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_fetch():
    """Test async HTTP fetch."""
    from yourpkg.http_client import fetch_url
    data = await fetch_url("https://httpbin.org/json")
    assert "origin" in data

@pytest.mark.asyncio
async def test_async_timeout():
    """Test async timeout handling."""
    from yourpkg.http_client import fetch_with_timeout
    with pytest.raises(asyncio.TimeoutError):
        await fetch_with_timeout("https://httpbin.org/delay/5", timeout=1.0)
```

**Why Async Testing Matters**: Modern Python applications use async code extensively. Understanding these patterns prevents async test failures and enables reliable async testing.

### Time and Determinism

```python
# tests/unit/test_time.py
from freezegun import freeze_time
import datetime

@freeze_time("2025-01-01 12:00:00")
def test_timestamp_generation():
    """Test timestamp generation with frozen time."""
    from yourpkg.utils import get_timestamp
    assert get_timestamp() == "2025-01-01T12:00:00Z"

def test_random_seed():
    """Test with fixed random seed."""
    import random
    random.seed(42)
    values = [random.randint(1, 100) for _ in range(5)]
    assert values == [82, 15, 3, 35, 59]  # Deterministic with seed 42
```

**Why Time Control Matters**: Deterministic tests prevent flakiness and enable reliable test execution. Understanding these patterns prevents time-dependent test failures and enables consistent test results.

## 6) Coverage Configuration

### .coveragerc Configuration

```ini
# .coveragerc
[run]
branch = True
source = yourpkg
omit = 
    */tests/*
    */venv/*
    */migrations/*
    setup.py

[report]
fail_under = 90
show_missing = True
skip_covered = False
exclude_lines =
    pragma: no cover
    if TYPE_CHECKING:
    if __name__ == .__main__.:
    class .*:
        def __repr__
        def __str__
    raise NotImplementedError
    raise AssertionError
```

**Why Coverage Config Matters**: Proper coverage configuration ensures meaningful coverage metrics and prevents false positives. Understanding these patterns prevents coverage gaming and enables accurate quality measurement.

### Coverage Commands

```bash
# Run with coverage
pytest --cov --cov-report=term-missing

# Generate HTML report
pytest --cov --cov-report=html
open htmlcov/index.html

# Fail on coverage threshold
pytest --cov --cov-fail-under=90

# Coverage with branch analysis
pytest --cov --cov-branch --cov-report=term-missing
```

**Why Coverage Commands Matter**: Coverage reporting provides visibility into test quality and identifies untested code paths. Understanding these patterns prevents coverage regressions and enables quality improvement.

## 7) Popular Plugins

### Parallel Execution (pytest-xdist)

```bash
# Run tests in parallel
pytest -n auto

# Use specific number of workers
pytest -n 4

# Use loadfile distribution
pytest -n auto --dist loadfile
```

**Why Parallel Execution Matters**: Parallel testing reduces test execution time and enables faster feedback loops. Understanding these patterns prevents test bottlenecks and enables efficient CI pipelines.

### Random Test Order (pytest-randomly)

```bash
# Run tests in random order
pytest --randomly

# Use specific seed for reproducibility
pytest --randomly-seed=12345

# Show seed for reproducibility
pytest --randomly --randomly-seed=12345 -v
```

**Why Random Order Matters**: Random test order reveals hidden dependencies and improves test isolation. Understanding these patterns prevents test coupling and enables more robust test suites.

### Property-Based Testing (Hypothesis)

```python
# tests/unit/test_property.py
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=0, max_size=100))
def test_list_roundtrip(lst):
    """Test list encoding/decoding roundtrip."""
    from yourpkg.codec import encode, decode
    assert decode(encode(lst)) == lst

@given(st.text(min_size=1, max_size=100))
def test_string_validation(text):
    """Test string validation with various inputs."""
    from yourpkg.validators import is_valid_username
    # Should not crash on any string input
    result = is_valid_username(text)
    assert isinstance(result, bool)
```

**Why Property Testing Matters**: Property-based testing catches edge cases that manual testing might miss. Understanding these patterns prevents subtle bugs and enables more comprehensive test coverage.

### Network Mocking (responses)

```python
# tests/unit/test_http.py
import responses
import pytest

@responses.activate
def test_api_call_success():
    """Test successful API call with mocked response."""
    responses.add(
        responses.GET,
        "https://api.example.com/users/1",
        json={"id": 1, "name": "Alice"},
        status=200
    )
    
    from yourpkg.api_client import get_user
    user = get_user(1)
    assert user["name"] == "Alice"

@responses.activate
def test_api_call_failure():
    """Test API call failure handling."""
    responses.add(
        responses.GET,
        "https://api.example.com/users/999",
        status=404
    )
    
    from yourpkg.api_client import get_user
    with pytest.raises(Exception):
        get_user(999)
```

**Why Network Mocking Matters**: Network mocking prevents external dependencies and enables reliable testing. Understanding these patterns prevents test flakiness and enables consistent test execution.

### Snapshots (syrupy)

```python
# tests/unit/test_snapshots.py
def test_render_template(snapshot):
    """Test template rendering with snapshot."""
    from yourpkg.templates import render_user_card
    html = render_user_card({"name": "Alice", "email": "alice@example.com"})
    assert html == snapshot

def test_data_serialization(snapshot):
    """Test data serialization with snapshot."""
    from yourpkg.serializers import serialize_user
    data = serialize_user({"id": 1, "name": "Alice", "active": True})
    assert data == snapshot
```

**Why Snapshots Matter**: Snapshots prevent output format regressions and enable visual diffing. Understanding these patterns prevents output format bugs and enables reliable output validation.

## 8) Performance and Benchmarking

### Benchmarking (pytest-benchmark)

```python
# tests/performance/test_benchmarks.py
def test_fast_operation(benchmark):
    """Benchmark fast operation."""
    from yourpkg.core import fast_calculation
    result = benchmark(fast_calculation, 1000)
    assert result > 0

def test_slow_operation(benchmark):
    """Benchmark slow operation with timeout."""
    from yourpkg.core import slow_calculation
    result = benchmark(slow_calculation, 10000)
    assert result > 0
```

**Why Benchmarking Matters**: Performance benchmarks prevent performance regressions and enable optimization tracking. Understanding these patterns prevents performance degradation and enables performance monitoring.

### Timeout Handling (pytest-timeout)

```python
# tests/unit/test_timeouts.py
@pytest.mark.timeout(2)
def test_quick_operation():
    """Test that completes quickly."""
    from yourpkg.core import quick_task
    result = quick_task()
    assert result is not None

@pytest.mark.timeout(10)
def test_slow_operation():
    """Test that might take longer."""
    from yourpkg.core import slow_task
    result = slow_task()
    assert result is not None
```

**Why Timeout Handling Matters**: Timeout handling prevents hung tests and enables reliable CI execution. Understanding these patterns prevents test timeouts and enables efficient test execution.

## 9) CI Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  pytest:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/pip
            .pytest_cache
            .hypothesis
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
      
      - name: Install dependencies
        run: |
          pip install -U pip
          pip install -e .[dev]
      
      - name: Run tests
        run: |
          pytest -n auto --cov --cov-report=xml --junitxml=reports/junit.xml
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
      
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: reports/junit.xml
```

**Why CI Integration Matters**: Automated testing prevents regressions and enables continuous quality assurance. Understanding these patterns prevents deployment issues and enables reliable software delivery.

### Tox Configuration

```ini
# tox.ini
[tox]
envlist = py{310,311,312}
isolated_build = True

[testenv]
deps =
  -r requirements-dev.txt
commands =
  pytest -n auto --cov --cov-report=term-missing

[testenv:lint]
deps =
  flake8
  black
  isort
commands =
  flake8 src/ tests/
  black --check src/ tests/
  isort --check-only src/ tests/
```

**Why Tox Matters**: Multi-environment testing ensures compatibility across Python versions. Understanding these patterns prevents version-specific bugs and enables reliable cross-platform support.

## 10) Anti-Patterns and Red Flags

### Common Anti-Patterns

```python
# DON'T: Tests that depend on external services
def test_api_call():
    import requests
    response = requests.get("https://api.example.com/data")  # Flaky!
    assert response.status_code == 200

# DO: Mock external dependencies
@responses.activate
def test_api_call():
    responses.add(responses.GET, "https://api.example.com/data", json={"data": "test"})
    from yourpkg.api import get_data
    data = get_data()
    assert data == {"data": "test"}

# DON'T: Tests with hidden global state
class TestDatabase:
    def test_create_user(self):
        # Uses global database connection
        pass
    
    def test_delete_user(self):
        # Depends on previous test
        pass

# DO: Isolated tests with fixtures
class TestDatabase:
    def test_create_user(self, db_fixture):
        # Each test gets fresh database
        pass
    
    def test_delete_user(self, db_fixture):
        # Independent test
        pass
```

**Why Anti-Patterns Matter**: Common mistakes lead to flaky tests and unreliable CI. Understanding these patterns prevents test failures and enables maintainable test suites.

### Flaky Test Quarantine

```python
# Use sparingly - prefer fixing root cause
@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_network_operation():
    """Temporarily flaky test - fix root cause."""
    from yourpkg.network import fetch_data
    data = fetch_data()
    assert data is not None
```

**Why Quarantine Matters**: Flaky tests should be fixed, not normalized. Understanding these patterns prevents test quality degradation and enables reliable test execution.

## 11) TL;DR Runbook

### Essential Commands

```bash
# Fast local development
pytest -q

# Parallel with coverage
pytest -n auto --cov --cov-report=term-missing

# Deterministic with seed
pytest --randomly-seed=12345 -x

# Only unit tests
pytest -m unit

# Generate HTML coverage
pytest --cov --cov-report=html
open htmlcov/index.html

# Run specific test file
pytest tests/unit/test_core.py

# Run with verbose output
pytest -v -s
```

### Essential Patterns

```python
# Essential pytest patterns
pytest_patterns = {
    "fixtures": "Use fixtures for test data and setup",
    "parametrize": "Use parametrize for multiple test cases",
    "markers": "Use markers to categorize tests",
    "coverage": "Always run with coverage enabled",
    "parallel": "Use -n auto for parallel execution",
    "isolation": "Keep tests isolated and independent"
}
```

### Quick Reference

```bash
# Development workflow
pytest -q                    # Quick feedback
pytest -n auto              # Parallel execution
pytest --cov               # Coverage analysis
pytest -m unit             # Unit tests only
pytest --randomly          # Random order
pytest -x                  # Stop on first failure
```

**Why This Runbook**: These patterns cover 90% of pytest usage. Master these before exploring advanced features.

## 12) The Machine's Summary

Pytest requires understanding both testing mechanics and CI integration patterns. When used correctly, pytest enables fast, reliable testing that prevents regressions and maintains code quality. The key is understanding fixture management, mastering coverage patterns, and following CI integration best practices.

**The Dark Truth**: Without proper pytest understanding, your test suite is fragile and unreliable. Pytest is your weapon. Use it wisely.

**The Machine's Mantra**: "In fixtures we trust, in coverage we measure, and in the tests we find the path to reliable software."

**Why This Matters**: Pytest enables efficient testing that can handle complex test scenarios, maintain high coverage, and provide reliable CI integration while ensuring performance and maintainability.

---

*This guide provides the complete machinery for pytest testing. The patterns scale from simple unit tests to complex integration testing, from basic coverage to advanced CI automation.*
