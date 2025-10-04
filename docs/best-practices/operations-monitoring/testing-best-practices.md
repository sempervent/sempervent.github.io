# Testing Best Practices

This document establishes comprehensive testing strategies for geospatial systems, covering unit testing, integration testing, performance testing, and test automation.

## Comprehensive Test Suite

### Unit Testing

```python
import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from unittest.mock import Mock, patch

class TestGeospatialProcessor:
    """Test suite for geospatial processing functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_points = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2)
            ]
        })
        
        self.test_polygons = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
            ]
        })
    
    def test_spatial_join(self):
        """Test spatial join functionality"""
        result = spatial_join_optimized(self.test_points, self.test_polygons)
        
        assert len(result) > 0
        assert all('left_idx' in item for item in result)
        assert all('right_idx' in item for item in result)
    
    def test_buffer_analysis(self):
        """Test buffer analysis"""
        point = Point(0, 0)
        buffer = point.buffer(1.0)
        
        assert buffer.area > 0
        assert buffer.contains(point)
    
    @patch('geopandas.read_file')
    def test_load_geospatial_data(self, mock_read_file):
        """Test loading geospatial data with mocking"""
        mock_read_file.return_value = self.test_points
        
        result = load_geospatial_data('test.geojson')
        
        assert len(result) == 3
        mock_read_file.assert_called_once_with('test.geojson')
    
    def test_crs_transformation(self):
        """Test coordinate reference system transformation"""
        gdf = self.test_points.copy()
        gdf.crs = 'EPSG:4326'
        
        transformed = gdf.to_crs('EPSG:3857')
        
        assert transformed.crs == 'EPSG:3857'
        assert len(transformed) == len(gdf)
    
    def test_spatial_index_creation(self):
        """Test spatial index creation"""
        index = create_spatial_index(self.test_points)
        
        assert index is not None
        assert len(list(index.intersection((0, 0, 1, 1)))) > 0

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_spatial_query_endpoint(self, client):
        """Test spatial query API endpoint"""
        query_data = {
            "geometry": {
                "type": "Point",
                "coordinates": [0, 0]
            },
            "buffer_distance": 1000
        }
        
        response = client.post("/spatial/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_count" in data
        assert "processing_time" in data
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

# Pytest fixtures
@pytest.fixture
def client():
    """Test client fixture"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)

@pytest.fixture
def sample_geospatial_data():
    """Sample geospatial data fixture"""
    return gpd.GeoDataFrame({
        'id': [1, 2, 3],
        'name': ['Point A', 'Point B', 'Point C'],
        'geometry': [
            Point(0, 0),
            Point(1, 1),
            Point(2, 2)
        ]
    })
```

**Why:** Unit tests ensure individual functions work correctly. Integration tests verify API endpoints and data flow. Fixtures provide reusable test data and reduce test setup complexity.

### Test Data Management

```python
import pytest
import tempfile
import os
from pathlib import Path

class TestDataManager:
    """Manages test data for geospatial testing"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_files = {}
    
    def setup_test_data(self):
        """Setup test data files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test GeoJSON
        test_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"id": 1, "name": "Test Point"}
                }
            ]
        }
        
        geojson_path = os.path.join(self.temp_dir, "test.geojson")
        with open(geojson_path, 'w') as f:
            json.dump(test_geojson, f)
        
        self.test_files['geojson'] = geojson_path
        
        # Create test Parquet
        test_gdf = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        })
        
        parquet_path = os.path.join(self.temp_dir, "test.parquet")
        test_gdf.to_parquet(parquet_path)
        self.test_files['parquet'] = parquet_path
    
    def cleanup_test_data(self):
        """Cleanup test data"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

@pytest.fixture
def test_data_manager():
    """Test data manager fixture"""
    manager = TestDataManager()
    manager.setup_test_data()
    yield manager
    manager.cleanup_test_data()

def test_geospatial_data_loading(test_data_manager):
    """Test loading geospatial data from files"""
    geojson_path = test_data_manager.test_files['geojson']
    gdf = gpd.read_file(geojson_path)
    
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].geom_type == 'Point'
    assert gdf.iloc[0]['name'] == 'Test Point'
```

**Why:** Test data management ensures consistent test environments. Temporary directories prevent test data pollution. Fixtures provide automatic cleanup.

## Performance Testing

### Load Testing

```python
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

class LoadTester:
    """Load testing for geospatial APIs"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session: aiohttp.ClientSession, endpoint: str, payload: dict = None):
        """Execute a single request"""
        start_time = time.time()
        
        try:
            if payload:
                async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                    await response.text()
            else:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
            
            end_time = time.time()
            return {
                'status': 'success',
                'response_time': end_time - start_time,
                'status_code': response.status
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'status': 'error',
                'response_time': end_time - start_time,
                'error': str(e)
            }
    
    async def run_load_test(self, endpoint: str, concurrent_users: int, requests_per_user: int, payload: dict = None):
        """Run load test with specified parameters"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    task = self.single_request(session, endpoint, payload)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results.extend(results)
    
    def analyze_results(self):
        """Analyze load test results"""
        successful_requests = [r for r in self.results if r.get('status') == 'success']
        failed_requests = [r for r in self.results if r.get('status') == 'error']
        
        if not successful_requests:
            return {'error': 'No successful requests'}
        
        response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'total_requests': len(self.results),
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(self.results),
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)],
            'p99_response_time': sorted(response_times)[int(len(response_times) * 0.99)],
            'min_response_time': min(response_times),
            'max_response_time': max(response_times)
        }

# Usage example
async def run_spatial_api_load_test():
    """Run load test on spatial API"""
    tester = LoadTester("http://localhost:8000")
    
    # Test spatial query endpoint
    spatial_query_payload = {
        "geometry": {
            "type": "Point",
            "coordinates": [0, 0]
        },
        "buffer_distance": 1000
    }
    
    await tester.run_load_test(
        endpoint="/spatial/query",
        concurrent_users=10,
        requests_per_user=100,
        payload=spatial_query_payload
    )
    
    results = tester.analyze_results()
    print(f"Load test results: {results}")

# Run load test
asyncio.run(run_spatial_api_load_test())
```

**Why:** Load testing identifies performance bottlenecks and capacity limits. Concurrent testing simulates real-world usage patterns.

### Stress Testing

```python
import psutil
import time
from threading import Thread
import requests

class StressTester:
    """Stress testing for geospatial systems"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.monitoring = False
        self.system_metrics = []
    
    def monitor_system_resources(self):
        """Monitor system resources during stress test"""
        while self.monitoring:
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()._asdict(),
                'network_io': psutil.net_io_counters()._asdict()
            }
            self.system_metrics.append(metrics)
            time.sleep(1)
    
    def run_stress_test(self, duration_minutes: int, requests_per_second: int):
        """Run stress test for specified duration"""
        self.monitoring = True
        
        # Start system monitoring
        monitor_thread = Thread(target=self.monitor_system_resources)
        monitor_thread.start()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        request_interval = 1.0 / requests_per_second
        request_count = 0
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                request_count += 1
                
                if response.status_code != 200:
                    print(f"Request {request_count} failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Request {request_count} failed with error: {e}")
            
            # Maintain request rate
            elapsed = time.time() - request_start
            sleep_time = max(0, request_interval - elapsed)
            time.sleep(sleep_time)
        
        self.monitoring = False
        monitor_thread.join()
        
        return {
            'total_requests': request_count,
            'duration': time.time() - start_time,
            'system_metrics': self.system_metrics
        }

# Usage example
def run_stress_test():
    """Run stress test on geospatial API"""
    tester = StressTester("http://localhost:8000")
    
    results = tester.run_stress_test(
        duration_minutes=10,
        requests_per_second=50
    )
    
    print(f"Stress test completed: {results['total_requests']} requests in {results['duration']:.2f}s")
    
    # Analyze system metrics
    if results['system_metrics']:
        cpu_usage = [m['cpu_percent'] for m in results['system_metrics']]
        memory_usage = [m['memory_percent'] for m in results['system_metrics']]
        
        print(f"Average CPU usage: {statistics.mean(cpu_usage):.2f}%")
        print(f"Average memory usage: {statistics.mean(memory_usage):.2f}%")
        print(f"Peak CPU usage: {max(cpu_usage):.2f}%")
        print(f"Peak memory usage: {max(memory_usage):.2f}%")

run_stress_test()
```

**Why:** Stress testing identifies system limits and failure points. System monitoring provides insights into resource utilization patterns.

## Test Automation

### Continuous Integration Testing

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgis/postgis:15-3.3
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_geospatial
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev
    
    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --durations=10
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

**Why:** CI/CD ensures consistent testing across environments. Service containers provide isolated test environments.

### Test Data Generation

```python
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import random

class TestDataGenerator:
    """Generate test data for geospatial testing"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_random_points(self, count: int, bounds: tuple = (-180, -90, 180, 90)) -> gpd.GeoDataFrame:
        """Generate random points within bounds"""
        minx, miny, maxx, maxy = bounds
        
        points = []
        for _ in range(count):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            points.append(Point(x, y))
        
        return gpd.GeoDataFrame({
            'id': range(count),
            'geometry': points,
            'value': np.random.random(count)
        })
    
    def generate_random_polygons(self, count: int, bounds: tuple = (-180, -90, 180, 90)) -> gpd.GeoDataFrame:
        """Generate random polygons within bounds"""
        minx, miny, maxx, maxy = bounds
        
        polygons = []
        for _ in range(count):
            # Generate random polygon
            center_x = random.uniform(minx, maxx)
            center_y = random.uniform(miny, maxy)
            size = random.uniform(0.1, 1.0)
            
            coords = []
            for _ in range(4):  # Square polygon
                x = center_x + random.uniform(-size, size)
                y = center_y + random.uniform(-size, size)
                coords.append((x, y))
            
            # Close polygon
            coords.append(coords[0])
            polygons.append(Polygon(coords))
        
        return gpd.GeoDataFrame({
            'id': range(count),
            'geometry': polygons,
            'area': [p.area for p in polygons]
        })
    
    def generate_test_dataset(self, points_count: int = 1000, polygons_count: int = 100) -> dict:
        """Generate comprehensive test dataset"""
        return {
            'points': self.generate_random_points(points_count),
            'polygons': self.generate_random_polygons(polygons_count),
            'lines': self.generate_random_lines(50)
        }
    
    def generate_random_lines(self, count: int, bounds: tuple = (-180, -90, 180, 90)) -> gpd.GeoDataFrame:
        """Generate random lines within bounds"""
        minx, miny, maxx, maxy = bounds
        
        lines = []
        for _ in range(count):
            # Generate random line
            start_x = random.uniform(minx, maxx)
            start_y = random.uniform(miny, maxy)
            end_x = random.uniform(minx, maxx)
            end_y = random.uniform(miny, maxy)
            
            lines.append(LineString([(start_x, start_y), (end_x, end_y)]))
        
        return gpd.GeoDataFrame({
            'id': range(count),
            'geometry': lines,
            'length': [l.length for l in lines]
        })

# Usage in tests
@pytest.fixture
def test_data_generator():
    """Test data generator fixture"""
    return TestDataGenerator(seed=42)

@pytest.fixture
def sample_test_data(test_data_generator):
    """Sample test data fixture"""
    return test_data_generator.generate_test_dataset(
        points_count=100,
        polygons_count=10
    )

def test_spatial_operations_with_generated_data(sample_test_data):
    """Test spatial operations with generated data"""
    points = sample_test_data['points']
    polygons = sample_test_data['polygons']
    
    # Test spatial join
    result = gpd.sjoin(points, polygons, how='inner', predicate='intersects')
    
    assert len(result) > 0
    assert 'index_right' in result.columns
```

**Why:** Automated test data generation ensures consistent test environments. Configurable data generation enables testing edge cases and boundary conditions.

## Test Reporting and Metrics

### Test Coverage Analysis

```python
import coverage
import pytest
from pathlib import Path

class TestCoverageAnalyzer:
    """Analyze test coverage for geospatial code"""
    
    def __init__(self, source_dir: str, test_dir: str):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_data = {}
    
    def run_coverage_analysis(self):
        """Run coverage analysis on test suite"""
        cov = coverage.Coverage(source=[str(self.source_dir)])
        cov.start()
        
        # Run tests
        pytest.main([str(self.test_dir), '-v'])
        
        cov.stop()
        cov.save()
        
        # Generate coverage report
        self.coverage_data = cov.get_data()
        
        return {
            'coverage_percent': cov.report(),
            'missing_lines': self.get_missing_lines(),
            'branch_coverage': self.get_branch_coverage()
        }
    
    def get_missing_lines(self):
        """Get lines not covered by tests"""
        missing_lines = {}
        
        for filename in self.coverage_data.measured_files():
            if str(self.source_dir) in filename:
                relative_path = Path(filename).relative_to(self.source_dir)
                missing_lines[str(relative_path)] = self.coverage_data.lines_not_covered(filename)
        
        return missing_lines
    
    def get_branch_coverage(self):
        """Get branch coverage information"""
        branch_data = {}
        
        for filename in self.coverage_data.measured_files():
            if str(self.source_dir) in filename:
                relative_path = Path(filename).relative_to(self.source_dir)
                branch_data[str(relative_path)] = {
                    'total_branches': len(self.coverage_data.branches(filename)),
                    'covered_branches': len(self.coverage_data.branches(filename)) - len(self.coverage_data.branches_not_covered(filename))
                }
        
        return branch_data

# Usage
analyzer = TestCoverageAnalyzer('src/', 'tests/')
coverage_results = analyzer.run_coverage_analysis()
print(f"Coverage: {coverage_results['coverage_percent']:.2f}%")
```

**Why:** Coverage analysis identifies untested code paths. Branch coverage ensures conditional logic is properly tested.

### Test Performance Metrics

```python
import time
import statistics
from typing import List, Dict, Any

class TestPerformanceMetrics:
    """Collect and analyze test performance metrics"""
    
    def __init__(self):
        self.test_results = []
        self.performance_data = {}
    
    def record_test_execution(self, test_name: str, execution_time: float, status: str, memory_usage: float = None):
        """Record test execution metrics"""
        self.test_results.append({
            'test_name': test_name,
            'execution_time': execution_time,
            'status': status,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        })
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze test performance"""
        if not self.test_results:
            return {}
        
        execution_times = [r['execution_time'] for r in self.test_results]
        successful_tests = [r for r in self.test_results if r['status'] == 'passed']
        failed_tests = [r for r in self.test_results if r['status'] == 'failed']
        
        return {
            'total_tests': len(self.test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(self.test_results),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'slowest_tests': sorted(self.test_results, key=lambda x: x['execution_time'], reverse=True)[:5],
            'fastest_tests': sorted(self.test_results, key=lambda x: x['execution_time'])[:5]
        }
    
    def identify_slow_tests(self, threshold_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """Identify tests that exceed execution time threshold"""
        return [r for r in self.test_results if r['execution_time'] > threshold_seconds]
    
    def generate_performance_report(self) -> str:
        """Generate performance report"""
        analysis = self.analyze_performance()
        
        report = f"""
Test Performance Report
======================

Total Tests: {analysis['total_tests']}
Successful: {analysis['successful_tests']}
Failed: {analysis['failed_tests']}
Success Rate: {analysis['success_rate']:.2%}

Execution Time Statistics:
- Average: {analysis['avg_execution_time']:.3f}s
- Median: {analysis['median_execution_time']:.3f}s

Slowest Tests:
"""
        
        for test in analysis['slowest_tests']:
            report += f"- {test['test_name']}: {test['execution_time']:.3f}s\n"
        
        return report

# Usage in tests
performance_metrics = TestPerformanceMetrics()

@pytest.fixture(autouse=True)
def record_test_performance(request):
    """Record test performance automatically"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    yield
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    performance_metrics.record_test_execution(
        test_name=request.node.name,
        execution_time=end_time - start_time,
        status='passed' if request.node.rep_call.passed else 'failed',
        memory_usage=end_memory - start_memory
    )
```

**Why:** Performance metrics identify slow tests and resource usage patterns. Automated performance tracking enables continuous optimization.
