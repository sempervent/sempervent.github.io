# Python Testing Best Practices

**Objective**: Master senior-level Python testing patterns for production systems. When you need to ensure code reliability and maintainability, when you want to implement comprehensive testing strategies, when you need enterprise-grade testing workflows—these best practices become your weapon of choice.

## Core Principles

- **Test Coverage**: Aim for high test coverage with meaningful tests
- **Test Isolation**: Ensure tests are independent and repeatable
- **Test Performance**: Write fast, efficient tests
- **Test Maintainability**: Keep tests readable and maintainable
- **Test Automation**: Integrate testing into CI/CD pipelines

## Testing Framework Setup

### Pytest Configuration

```python
# python/01-testing-setup.py

"""
Comprehensive pytest setup and configuration
"""

import pytest
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import json

class PytestSetup:
    """Setup and configure pytest for Python projects"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.tests_path = project_path / "tests"
        self.conftest_path = self.tests_path / "conftest.py"
    
    def create_pytest_config(self) -> bool:
        """Create comprehensive pytest configuration"""
        try:
            # Create pytest.ini
            pytest_ini = self.project_path / "pytest.ini"
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
    --durations=10
    --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    api: marks tests as API tests
    database: marks tests as database tests
    network: marks tests as network tests
''')
            
            # Create conftest.py
            conftest_content = '''"""
Global pytest configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import Mock, MagicMock
import json

# Test data utilities
class TestData:
    """Test data utilities and helpers"""
    
    @staticmethod
    def create_temp_file(content: str = "test content", suffix: str = ".txt") -> Path:
        """Create a temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_temp_directory() -> Path:
        """Create a temporary directory"""
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def sample_json_data() -> Dict[str, Any]:
        """Return sample JSON data for testing"""
        return {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "metadata": {
                "total": 2,
                "page": 1,
                "per_page": 10
            }
        }
    
    @staticmethod
    def sample_csv_data() -> str:
        """Return sample CSV data for testing"""
        return """id,name,email,age
1,Alice,alice@example.com,25
2,Bob,bob@example.com,30
3,Charlie,charlie@example.com,35"""

# Global fixtures
@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Session-scoped temporary directory"""
    temp_dir = TestData.create_temp_directory()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Temporary file fixture"""
    temp_file = TestData.create_temp_file()
    yield temp_file
    temp_file.unlink()

@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Sample data fixture"""
    return TestData.sample_json_data()

@pytest.fixture
def mock_api_client() -> Mock:
    """Mock API client fixture"""
    mock_client = Mock()
    mock_client.get.return_value = {"status": "success", "data": []}
    mock_client.post.return_value = {"status": "created", "id": 1}
    mock_client.put.return_value = {"status": "updated"}
    mock_client.delete.return_value = {"status": "deleted"}
    return mock_client

@pytest.fixture
def mock_database() -> Mock:
    """Mock database fixture"""
    mock_db = Mock()
    mock_db.query.return_value.all.return_value = []
    mock_db.query.return_value.first.return_value = None
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    return mock_db

@pytest.fixture
def mock_redis() -> Mock:
    """Mock Redis fixture"""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    return mock_redis

# Configuration fixtures
@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration fixture"""
    return {
        "database_url": "sqlite:///test.db",
        "redis_url": "redis://localhost:6379/0",
        "api_key": "test_key",
        "debug": True,
        "log_level": "DEBUG"
    }

# Performance fixtures
@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture"""
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    print(f"Test duration: {duration:.3f} seconds")
'''
            
            with open(self.conftest_path, 'w') as f:
                f.write(conftest_content)
            
            return True
        except Exception:
            return False
    
    def create_test_utilities(self) -> bool:
        """Create test utilities and helpers"""
        try:
            # Create test utilities module
            test_utils = self.tests_path / "utils.py"
            test_utils.write_text('''"""
Test utilities and helper functions
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pytest
from unittest.mock import Mock, patch

class TestHelpers:
    """Helper functions for testing"""
    
    @staticmethod
    def assert_json_equal(actual: str, expected: Dict[str, Any]) -> None:
        """Assert that JSON string equals expected dictionary"""
        actual_dict = json.loads(actual)
        assert actual_dict == expected
    
    @staticmethod
    def create_mock_response(status_code: int = 200, data: Any = None) -> Mock:
        """Create a mock HTTP response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data or {}
        mock_response.text = json.dumps(data) if data else ""
        return mock_response
    
    @staticmethod
    def create_temp_json_file(data: Dict[str, Any]) -> Path:
        """Create a temporary JSON file with data"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(data, temp_file)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def assert_file_exists(file_path: Path) -> None:
        """Assert that a file exists"""
        assert file_path.exists(), f"File {file_path} does not exist"
    
    @staticmethod
    def assert_file_content(file_path: Path, expected_content: str) -> None:
        """Assert that file contains expected content"""
        assert file_path.exists(), f"File {file_path} does not exist"
        actual_content = file_path.read_text()
        assert actual_content == expected_content

class DatabaseTestHelpers:
    """Database testing helpers"""
    
    @staticmethod
    def create_test_database_url() -> str:
        """Create a test database URL"""
        return "sqlite:///test.db"
    
    @staticmethod
    def setup_test_database():
        """Setup test database"""
        # Implementation depends on your database setup
        pass
    
    @staticmethod
    def cleanup_test_database():
        """Cleanup test database"""
        # Implementation depends on your database setup
        pass

class APITestHelpers:
    """API testing helpers"""
    
    @staticmethod
    def create_test_client():
        """Create test API client"""
        # Implementation depends on your API framework
        pass
    
    @staticmethod
    def assert_response_status(response, expected_status: int) -> None:
        """Assert response status code"""
        assert response.status_code == expected_status
    
    @staticmethod
    def assert_response_json(response, expected_data: Dict[str, Any]) -> None:
        """Assert response JSON data"""
        actual_data = response.json()
        assert actual_data == expected_data
''')
            
            return True
        except Exception:
            return False

# Usage example
def setup_pytest_framework(project_path: Path):
    """Setup comprehensive pytest framework"""
    pytest_setup = PytestSetup(project_path)
    pytest_setup.create_pytest_config()
    pytest_setup.create_test_utilities()
    
    print("Pytest framework setup complete")
```

### Test Categories and Markers

```python
# python/02-test-categories.py

"""
Test categories and markers for organized testing
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

class TestCategories:
    """Organize tests by categories and complexity"""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality - unit test"""
        # Test basic functionality
        assert True
    
    @pytest.mark.integration
    def test_database_integration(self, mock_database):
        """Test database integration - integration test"""
        # Test database operations
        assert mock_database is not None
    
    @pytest.mark.api
    def test_api_endpoint(self, mock_api_client):
        """Test API endpoint - API test"""
        # Test API functionality
        response = mock_api_client.get("/test")
        assert response["status"] == "success"
    
    @pytest.mark.slow
    def test_performance_intensive(self):
        """Test performance intensive operation - slow test"""
        # Test that takes time
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True
    
    @pytest.mark.network
    def test_external_api_call(self):
        """Test external API call - network test"""
        # Test that requires network access
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {"status": "success"}
            # Test external API call
            assert True

class TestDataDriven:
    """Data-driven testing patterns"""
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (0, 0),
        (-1, -2)
    ])
    def test_double_function(self, input_value, expected):
        """Test doubling function with multiple inputs"""
        def double(x):
            return x * 2
        
        result = double(input_value)
        assert result == expected
    
    @pytest.mark.parametrize("test_case", [
        {"input": "hello", "expected": "HELLO"},
        {"input": "world", "expected": "WORLD"},
        {"input": "test", "expected": "TEST"}
    ])
    def test_uppercase_function(self, test_case):
        """Test uppercase function with test cases"""
        def uppercase(text):
            return text.upper()
        
        result = uppercase(test_case["input"])
        assert result == test_case["expected"]

class TestFixtures:
    """Advanced fixture patterns"""
    
    @pytest.fixture(scope="class")
    def class_fixture(self):
        """Class-scoped fixture"""
        return {"data": "class_data"}
    
    @pytest.fixture(scope="module")
    def module_fixture(self):
        """Module-scoped fixture"""
        return {"data": "module_data"}
    
    @pytest.fixture(autouse=True)
    def auto_fixture(self):
        """Auto-use fixture"""
        print("Auto fixture setup")
        yield
        print("Auto fixture teardown")
    
    @pytest.fixture(params=["option1", "option2", "option3"])
    def parametrized_fixture(self, request):
        """Parametrized fixture"""
        return {"option": request.param}
    
    def test_with_fixtures(self, class_fixture, module_fixture, parametrized_fixture):
        """Test using multiple fixtures"""
        assert class_fixture["data"] == "class_data"
        assert module_fixture["data"] == "module_data"
        assert parametrized_fixture["option"] in ["option1", "option2", "option3"]
```

### Mocking and Patching

```python
# python/03-mocking-patching.py

"""
Advanced mocking and patching patterns
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List

class MockingPatterns:
    """Advanced mocking patterns for testing"""
    
    def test_basic_mocking(self):
        """Test basic mocking patterns"""
        # Create a mock object
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked_value"
        
        # Use the mock
        result = mock_obj.method()
        assert result == "mocked_value"
        
        # Verify method was called
        mock_obj.method.assert_called_once()
    
    def test_mock_with_side_effect(self):
        """Test mock with side effects"""
        mock_obj = Mock()
        mock_obj.method.side_effect = [1, 2, 3]
        
        # First call returns 1, second returns 2, third returns 3
        assert mock_obj.method() == 1
        assert mock_obj.method() == 2
        assert mock_obj.method() == 3
    
    def test_mock_with_exception(self):
        """Test mock that raises exceptions"""
        mock_obj = Mock()
        mock_obj.method.side_effect = ValueError("Test exception")
        
        with pytest.raises(ValueError, match="Test exception"):
            mock_obj.method()
    
    @patch('builtins.open')
    def test_file_operations(self, mock_open):
        """Test file operations with mocking"""
        mock_file = MagicMock()
        mock_file.read.return_value = "file content"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test file reading
        with open("test.txt") as f:
            content = f.read()
        
        assert content == "file content"
        mock_open.assert_called_once_with("test.txt")
    
    @patch('requests.get')
    def test_http_requests(self, mock_get):
        """Test HTTP requests with mocking"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test HTTP request
        import requests
        response = requests.get("https://api.example.com/data")
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_get.assert_called_once_with("https://api.example.com/data")
    
    def test_database_mocking(self, mock_database):
        """Test database operations with mocking"""
        # Setup mock database
        mock_database.query.return_value.filter.return_value.first.return_value = {
            "id": 1, "name": "Test User"
        }
        
        # Test database query
        result = mock_database.query().filter().first()
        assert result["id"] == 1
        assert result["name"] == "Test User"
    
    def test_async_mocking(self):
        """Test async function mocking"""
        import asyncio
        from unittest.mock import AsyncMock
        
        async def async_function():
            return "async result"
        
        # Mock async function
        with patch('__main__.async_function', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = "mocked async result"
            
            # Test async function
            result = asyncio.run(async_function())
            assert result == "mocked async result"

class PatchingPatterns:
    """Advanced patching patterns"""
    
    @patch('os.environ')
    def test_environment_variables(self, mock_environ):
        """Test environment variable patching"""
        mock_environ.get.return_value = "test_value"
        
        import os
        value = os.environ.get("TEST_VAR")
        assert value == "test_value"
    
    @patch('datetime.datetime')
    def test_datetime_mocking(self, mock_datetime):
        """Test datetime mocking"""
        from datetime import datetime
        
        # Mock datetime.now()
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        
        now = datetime.now()
        assert now.year == 2024
        assert now.month == 1
        assert now.day == 1
    
    def test_context_manager_patching(self):
        """Test context manager patching"""
        with patch('builtins.open') as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.read.return_value = "test content"
            mock_open.return_value = mock_file
            
            with open("test.txt") as f:
                content = f.read()
            
            assert content == "test content"
    
    def test_multiple_patches(self):
        """Test multiple patches"""
        with patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json_load:
            
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_json_load.return_value = {"key": "value"}
            
            # Test multiple patches
            with open("test.json") as f:
                import json
                data = json.load(f)
            
            assert data == {"key": "value"}
```

### Performance Testing

```python
# python/04-performance-testing.py

"""
Performance testing patterns and benchmarks
"""

import pytest
import time
import cProfile
import pstats
from typing import List, Dict, Any
from unittest.mock import Mock

class PerformanceTesting:
    """Performance testing patterns"""
    
    def test_execution_time(self):
        """Test execution time of a function"""
        def slow_function():
            time.sleep(0.1)  # Simulate slow operation
            return "result"
        
        start_time = time.time()
        result = slow_function()
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert result == "result"
        assert execution_time < 0.2  # Should complete within 200ms
    
    def test_memory_usage(self):
        """Test memory usage of a function"""
        import sys
        
        def memory_intensive_function():
            data = [i for i in range(10000)]
            return len(data)
        
        initial_memory = sys.getsizeof([])
        result = memory_intensive_function()
        final_memory = sys.getsizeof([])
        
        assert result == 10000
        # Memory usage should be reasonable
        assert final_memory - initial_memory < 1000000  # 1MB
    
    def test_algorithm_complexity(self):
        """Test algorithm time complexity"""
        def linear_search(items: List[int], target: int) -> int:
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        
        # Test with different input sizes
        test_cases = [
            (list(range(100)), 50),
            (list(range(1000)), 500),
            (list(range(10000)), 5000)
        ]
        
        for items, target in test_cases:
            start_time = time.time()
            result = linear_search(items, target)
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert result == target
            assert execution_time < 0.1  # Should be fast
    
    @pytest.mark.benchmark
    def test_benchmark_function(self, benchmark):
        """Benchmark a function using pytest-benchmark"""
        def benchmark_function():
            return sum(range(1000))
        
        result = benchmark(benchmark_function)
        assert result == 499500
    
    def test_profiling(self):
        """Test function profiling"""
        def profiled_function():
            total = 0
            for i in range(1000):
                total += i * i
            return total
        
        # Profile the function
        profiler = cProfile.Profile()
        profiler.enable()
        result = profiled_function()
        profiler.disable()
        
        # Get profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        assert result == 332833500
        # Verify function was profiled
        assert stats.total_calls > 0

class LoadTesting:
    """Load testing patterns"""
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        import threading
        import queue
        
        def worker(q, results):
            """Worker function for concurrent testing"""
            while True:
                try:
                    item = q.get(timeout=1)
                    # Simulate work
                    time.sleep(0.01)
                    results.append(f"processed_{item}")
                    q.task_done()
                except queue.Empty:
                    break
        
        # Create queue and results
        work_queue = queue.Queue()
        results = []
        
        # Add work items
        for i in range(10):
            work_queue.put(i)
        
        # Start worker threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker, args=(work_queue, results))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        work_queue.join()
        
        # Stop threads
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert all(result.startswith("processed_") for result in results)
    
    def test_memory_leak_detection(self):
        """Test for memory leaks"""
        import gc
        
        def create_objects():
            """Create objects that might leak memory"""
            objects = []
            for i in range(1000):
                obj = {"id": i, "data": "x" * 100}
                objects.append(obj)
            return objects
        
        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy objects
        for _ in range(10):
            objects = create_objects()
            del objects
            gc.collect()
        
        # Check final memory
        final_objects = len(gc.get_objects())
        
        # Memory should not grow significantly
        assert final_objects - initial_objects < 1000
```

## TL;DR Runbook

### Quick Start

```python
# 1. Setup pytest framework
from pathlib import Path
project_path = Path("my-project")

from python.testing_setup import setup_pytest_framework
setup_pytest_framework(project_path)

# 2. Run tests
import subprocess
subprocess.run(["pytest", "tests/"], cwd=project_path)

# 3. Run with coverage
subprocess.run(["pytest", "--cov=src", "tests/"], cwd=project_path)

# 4. Run specific test categories
subprocess.run(["pytest", "-m", "unit", "tests/"], cwd=project_path)
subprocess.run(["pytest", "-m", "integration", "tests/"], cwd=project_path)
```

### Essential Patterns

```python
# Complete testing setup
def setup_comprehensive_testing(project_path: Path):
    """Setup comprehensive testing framework"""
    
    # Setup pytest configuration
    pytest_setup = PytestSetup(project_path)
    pytest_setup.create_pytest_config()
    pytest_setup.create_test_utilities()
    
    # Create test categories
    test_categories = TestCategories()
    
    # Setup mocking patterns
    mocking_patterns = MockingPatterns()
    
    # Setup performance testing
    performance_testing = PerformanceTesting()
    
    print("Comprehensive testing framework setup complete!")
```

---

*This guide provides the complete machinery for Python testing. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise Python testing.*
