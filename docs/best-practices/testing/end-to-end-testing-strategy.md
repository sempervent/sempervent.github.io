# End-to-End Testing, Integration Validation, and QA Strategy for Distributed Systems

**Objective**: Master production-grade testing strategies for distributed, multi-stack systems. When you need to ensure correctness across microservices, validate data pipelines, test ML models, verify GIS computations, and maintain quality in air-gapped environments—this guide provides the complete testing framework.

## Introduction

Testing distributed systems is fundamentally different from testing monolithic applications. Services communicate across networks, data flows through multiple systems, and failures can cascade unpredictably. This guide provides a complete framework for testing distributed systems end-to-end.

**What This Guide Covers**:
- Testing pyramid for distributed systems
- Unit, integration, and E2E testing patterns
- Contract testing (API, data, ML, GIS)
- Test data strategies
- Performance and chaos testing
- CI/CD integration
- Air-gapped testing patterns
- Observability-driven QA

**Prerequisites**:
- Understanding of distributed systems, microservices, and data pipelines
- Familiarity with testing frameworks (pytest, Go testing, Rust cargo test)
- Experience with CI/CD and containerization

## Why Testing is Hard in Distributed Systems

### The Challenges

**Network Dependencies**:
- Services communicate over networks (latency, failures, partitions)
- Tests must account for network conditions
- Mocking network calls is insufficient for integration testing

**Dynamic Scheduling**:
- RKE2/Kubernetes pods are scheduled dynamically
- Tests must handle pod restarts, migrations, scaling
- Service discovery and load balancing add complexity

**Shared Infrastructure**:
- Postgres, Redis, object stores are shared across services
- Tests must isolate data and state
- Concurrent tests can interfere with each other

**Large, Versioned Artifacts**:
- Data pipelines produce large Parquet files
- Schema versions must be tested across versions
- Artifact storage and retrieval adds complexity

**Nondeterminism**:
- ML models introduce randomness
- Floating-point precision varies across platforms
- Timing-dependent behavior is hard to test

**Spatial Correctness**:
- GIS computations require spatial validation
- CRS transformations must be verified
- Geometry correctness is non-trivial

**Air-Gapped Constraints**:
- No external service access
- Test data must be packaged
- CI/CD pipelines must work offline

**Latency-Bound Guarantees**:
- ONNX inference must meet latency SLAs
- Streaming analytics have real-time constraints
- Performance regressions are critical

### System-Wide Testing Architecture

```mermaid
graph TB
    subgraph TestLayers["Test Layers"]
        Unit["Unit Tests<br/>(Go, Python, Rust)"]
        Integration["Integration Tests<br/>(Postgres, Redis, NGINX)"]
        Contract["Contract Tests<br/>(API, Data, ML, GIS)"]
        E2E["End-to-End Tests<br/>(Full Workflows)"]
        Performance["Performance Tests<br/>(Load, Latency)"]
        Chaos["Chaos Tests<br/>(Fault Injection)"]
    end
    
    subgraph Systems["System Under Test"]
        API["API Services"]
        PG["PostgreSQL"]
        Redis["Redis"]
        ETL["ETL Pipelines"]
        ML["ML Services"]
        GIS["GIS Services"]
        NGINX["NGINX"]
    end
    
    subgraph Infrastructure["Test Infrastructure"]
        TestDB["Test Databases<br/>(Ephemeral)"]
        TestCache["Test Redis<br/>(Isolated)"]
        TestData["Test Data<br/>(Synthetic, Golden)"]
        Observability["Observability<br/>(Logs, Metrics, Traces)"]
    end
    
    Unit --> API
    Unit --> ETL
    Unit --> ML
    
    Integration --> PG
    Integration --> Redis
    Integration --> NGINX
    
    Contract --> API
    Contract --> ETL
    Contract --> ML
    Contract --> GIS
    
    E2E --> API
    E2E --> PG
    E2E --> Redis
    E2E --> ETL
    E2E --> ML
    E2E --> GIS
    
    Performance --> API
    Performance --> PG
    Performance --> Redis
    
    Chaos --> Systems
    
    TestDB --> PG
    TestCache --> Redis
    TestData --> ETL
    TestData --> ML
    TestData --> GIS
    
    Observability --> Systems
```

## Testing Pyramid (Distributed Systems Edition)

### Multi-Layer Pyramid

```mermaid
graph TD
    subgraph Top["Top: E2E & System Tests"]
        E2E["End-to-End Tests<br/>(10-20%)"]
        UserJourney["User Journey Tests"]
        CrossService["Cross-Service Tests"]
        Ops["Ops Tests<br/>(Scaling, DR)"]
        Performance["Performance Tests"]
        Chaos["Chaos Tests"]
    end
    
    subgraph Middle["Middle: Integration Tests"]
        Integration["Integration Tests<br/>(30-40%)"]
        APIContract["API Contract Tests"]
        ETL["ETL Transformation Tests"]
        ML["ML Input/Output Tests"]
        GIS["GIS Pipeline Tests"]
        DB["Database Tests"]
    end
    
    subgraph Bottom["Bottom: Unit Tests"]
        Unit["Unit Tests<br/>(50-60%)"]
        Schema["Schema Tests"]
        Contract["Data Contract Tests"]
        Model["Model Tests"]
    end
    
    Top --> Middle
    Middle --> Bottom
```

### Bottom Layers: Unit Tests

**Purpose**: Fast, isolated tests of individual components.

**Coverage**:
- **Unit tests**: 50-60% of test suite
- **Schema tests**: Pydantic models, SQL constraints
- **Data contract tests**: Schema validation, type checking
- **Model tests**: ONNX models, ML model logic

**Characteristics**:
- **Speed**: < 1ms per test
- **Isolation**: No external dependencies
- **Determinism**: Fully deterministic
- **Coverage**: High code coverage (80%+)

### Middle Layers: Integration Tests

**Purpose**: Test interactions between components.

**Coverage**:
- **Integration tests**: 30-40% of test suite
- **API contract tests**: OpenAPI validation
- **ETL tests**: Transformation correctness
- **ML tests**: Feature validation, inference
- **GIS tests**: Spatial correctness

**Characteristics**:
- **Speed**: 10ms-1s per test
- **Dependencies**: Real services (Postgres, Redis)
- **Isolation**: Per-test database/cache
- **Coverage**: Critical paths

### Top Layers: End-to-End Tests

**Purpose**: Test complete workflows across systems.

**Coverage**:
- **E2E tests**: 10-20% of test suite
- **User journey tests**: Full user workflows
- **Cross-service tests**: Multi-service interactions
- **Ops tests**: Scaling, failover, DR
- **Performance tests**: Load, latency, throughput
- **Chaos tests**: Fault injection, recovery

**Characteristics**:
- **Speed**: 1s-60s per test
- **Dependencies**: Full stack
- **Isolation**: Test environments
- **Coverage**: Critical user paths

## Unit Testing Best Practices

### Python pytest Patterns

**Basic Structure**:

```python
# tests/unit/test_user_service.py
import pytest
from unittest.mock import Mock, patch
from services.user_service import UserService

class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()
    
    def test_get_user_by_id(self, user_service):
        """Test getting user by ID"""
        user = user_service.get_user(123)
        assert user.id == 123
        assert user.name is not None
    
    def test_get_user_not_found(self, user_service):
        """Test getting non-existent user"""
        with pytest.raises(UserNotFoundError):
            user_service.get_user(999)
    
    @patch('services.user_service.db')
    def test_create_user(self, mock_db, user_service):
        """Test creating user with mocked database"""
        mock_db.insert.return_value = {"id": 123, "name": "Test User"}
        
        user = user_service.create_user("Test User")
        
        assert user.id == 123
        assert user.name == "Test User"
        mock_db.insert.assert_called_once()
```

**Parametrized Tests**:

```python
@pytest.mark.parametrize("input,expected", [
    ("low", 1),
    ("medium", 2),
    ("high", 3),
])
def test_risk_level_mapping(input, expected):
    """Test risk level to numeric mapping"""
    assert map_risk_level(input) == expected
```

### Go Testing Package Conventions

**Basic Structure**:

```go
// user_service_test.go
package services

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

func TestGetUserByID(t *testing.T) {
    service := NewUserService()
    user, err := service.GetUser(123)
    
    assert.NoError(t, err)
    assert.Equal(t, 123, user.ID)
    assert.NotEmpty(t, user.Name)
}

func TestGetUserNotFound(t *testing.T) {
    service := NewUserService()
    _, err := service.GetUser(999)
    
    assert.Error(t, err)
    assert.IsType(t, &UserNotFoundError{}, err)
}

func TestCreateUser(t *testing.T) {
    mockDB := new(MockDB)
    mockDB.On("Insert", mock.Anything).Return(&User{ID: 123, Name: "Test User"}, nil)
    
    service := NewUserServiceWithDB(mockDB)
    user, err := service.CreateUser("Test User")
    
    assert.NoError(t, err)
    assert.Equal(t, 123, user.ID)
    mockDB.AssertExpectations(t)
}
```

**Table-Driven Tests**:

```go
func TestRiskLevelMapping(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected int
    }{
        {"low", "low", 1},
        {"medium", "medium", 2},
        {"high", "high", 3},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := MapRiskLevel(tt.input)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

### Rust cargo test Isolation

**Basic Structure**:

```rust
// src/user_service.rs
#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;
    
    #[test]
    fn test_get_user_by_id() {
        let service = UserService::new();
        let user = service.get_user(123).unwrap();
        
        assert_eq!(user.id, 123);
        assert!(!user.name.is_empty());
    }
    
    #[test]
    fn test_get_user_not_found() {
        let service = UserService::new();
        let result = service.get_user(999);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), UserError::NotFound));
    }
}
```

### Mocking Guidelines

**When to Mock**:
- External APIs (third-party services)
- Database for unit tests (use real DB for integration)
- File I/O for unit tests
- Network calls for unit tests

**When NOT to Mock**:
- Internal services (use integration tests)
- Database for integration tests
- Core business logic (test directly)
- Simple data structures (test directly)

**Example: Appropriate Mocking**:

```python
# Good: Mock external API
@patch('services.weather_api.get_weather')
def test_get_weather_data(mock_get_weather):
    mock_get_weather.return_value = {"temp": 72, "condition": "sunny"}
    result = get_weather_data("NYC")
    assert result["temp"] == 72

# Bad: Don't mock internal service
def test_internal_service():
    # Don't mock internal services; use integration tests
    service = InternalService()
    result = service.process_data(data)
    assert result is not None
```

### Snapshot Tests for Configurations

**Pattern**: Test configuration files against snapshots.

```python
# tests/unit/test_nginx_config.py
import pytest
from pathlib import Path

def test_nginx_config_snapshot():
    """Test NGINX config against snapshot"""
    config_path = Path("config/nginx.conf")
    config_content = config_path.read_text()
    
    snapshot_path = Path("tests/snapshots/nginx.conf.snapshot")
    
    if not snapshot_path.exists():
        snapshot_path.write_text(config_content)
        pytest.skip("Snapshot created")
    
    expected = snapshot_path.read_text()
    assert config_content == expected
```

### Unit Testing ONNX/Pydantic/SQL Models

**ONNX Model Testing**:

```python
# tests/unit/test_onnx_model.py
import onnxruntime as ort
import numpy as np

def test_onnx_model_inference():
    """Test ONNX model inference"""
    session = ort.InferenceSession("models/risk_model.onnx")
    
    # Test input
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    # Run inference
    outputs = session.run(None, {"input": input_data})
    
    # Validate output
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 3)  # 3 risk levels
    assert np.allclose(outputs[0].sum(axis=1), 1.0)  # Probabilities sum to 1
```

**Pydantic Model Testing**:

```python
# tests/unit/test_models.py
from pydantic import ValidationError
from models import User, RiskLevel

def test_user_model_validation():
    """Test Pydantic user model validation"""
    # Valid user
    user = User(id=123, name="Test User", risk_level="low")
    assert user.id == 123
    
    # Invalid user
    with pytest.raises(ValidationError):
        User(id="not-an-int", name="Test User")

def test_risk_level_enum():
    """Test risk level enum validation"""
    assert RiskLevel("low") == RiskLevel.low
    assert RiskLevel("medium") == RiskLevel.medium
    assert RiskLevel("high") == RiskLevel.high
    
    with pytest.raises(ValidationError):
        RiskLevel("invalid")
```

**SQL Constraint Testing**:

```python
# tests/unit/test_sql_constraints.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def test_sql_check_constraint():
    """Test SQL CHECK constraint"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Valid data
    road = Road(road_id=1, risk_level="low")
    session.add(road)
    session.commit()
    
    # Invalid data (should fail CHECK constraint)
    invalid_road = Road(road_id=2, risk_level="invalid")
    session.add(invalid_road)
    
    with pytest.raises(Exception):  # SQLAlchemy raises exception
        session.commit()
```

## Integration Testing Best Practices

### Postgres + PGO Integration Testing

**Dockerized Ephemeral Postgres**:

```python
# tests/integration/conftest.py
import pytest
import psycopg2
from docker import DockerClient

@pytest.fixture(scope="session")
def postgres_container():
    """Create ephemeral Postgres container"""
    client = DockerClient()
    container = client.containers.run(
        "postgres:15",
        environment={
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_DB": "testdb"
        },
        ports={"5432/tcp": None},
        detach=True
    )
    
    # Wait for Postgres to be ready
    import time
    for _ in range(30):
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"],
                user="test",
                password="test",
                database="testdb"
            )
            conn.close()
            break
        except:
            time.sleep(1)
    
    yield container
    
    container.stop()
    container.remove()

@pytest.fixture
def db_connection(postgres_container):
    """Get database connection"""
    port = postgres_container.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]
    conn = psycopg2.connect(
        host="localhost",
        port=port,
        user="test",
        password="test",
        database="testdb"
    )
    yield conn
    conn.close()
```

**Schema Migration Testing**:

```python
# tests/integration/test_migrations.py
def test_schema_migration(db_connection):
    """Test schema migration"""
    cur = db_connection.cursor()
    
    # Run migration
    with open("migrations/001_create_roads.sql") as f:
        cur.execute(f.read())
    
    db_connection.commit()
    
    # Verify schema
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'roads_v2'
        ORDER BY ordinal_position
    """)
    
    columns = cur.fetchall()
    assert len(columns) == 5
    assert columns[0][0] == "road_id"
```

**FDW Behavior Testing**:

```python
# tests/integration/test_fdw.py
def test_parquet_fdw_query(db_connection):
    """Test Parquet FDW query"""
    cur = db_connection.cursor()
    
    # Create FDW server
    cur.execute("""
        CREATE SERVER parquet_s3
        FOREIGN DATA WRAPPER parquet_s3_fdw
        OPTIONS (
            aws_access_key 'test',
            aws_secret_key 'test',
            endpoint 'http://localhost:9000'
        )
    """)
    
    # Create foreign table
    cur.execute("""
        CREATE FOREIGN TABLE roads_parquet (
            road_id BIGINT,
            road_name TEXT,
            risk_level VARCHAR(20),
            geometry BYTEA
        )
        SERVER parquet_s3
        OPTIONS (
            filename 's3://test-bucket/roads.parquet'
        )
    """)
    
    # Query foreign table
    cur.execute("SELECT COUNT(*) FROM roads_parquet")
    count = cur.fetchone()[0]
    assert count > 0
```

**Geo Query Testing**:

```python
# tests/integration/test_geo_queries.py
def test_st_intersects_query(db_connection):
    """Test ST_Intersects spatial query"""
    cur = db_connection.cursor()
    
    # Create test geometries
    cur.execute("""
        CREATE TABLE test_roads (
            road_id BIGINT PRIMARY KEY,
            geometry GEOMETRY(LINESTRING, 4326)
        )
    """)
    
    cur.execute("""
        CREATE TABLE test_zones (
            zone_id BIGINT PRIMARY KEY,
            geometry GEOMETRY(POLYGON, 4326)
        )
    """)
    
    # Insert test data
    cur.execute("""
        INSERT INTO test_roads (road_id, geometry)
        VALUES (1, ST_GeomFromText('LINESTRING(0 0, 1 1)', 4326))
    """)
    
    cur.execute("""
        INSERT INTO test_zones (zone_id, geometry)
        VALUES (1, ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 4326))
    """)
    
    # Test ST_Intersects
    cur.execute("""
        SELECT r.road_id
        FROM test_roads r
        JOIN test_zones z ON ST_Intersects(r.geometry, z.geometry)
    """)
    
    results = cur.fetchall()
    assert len(results) == 1
    assert results[0][0] == 1
```

### Redis Integration Testing

**Testing TTL Behavior**:

```python
# tests/integration/test_redis_ttl.py
import redis
import time

@pytest.fixture
def redis_client():
    """Create Redis client for testing"""
    return redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)

def test_ttl_expiration(redis_client):
    """Test TTL expiration"""
    key = "test:ttl"
    redis_client.setex(key, 1, "value")
    
    # Should exist
    assert redis_client.get(key) == "value"
    
    # Wait for expiration
    time.sleep(2)
    
    # Should be expired
    assert redis_client.get(key) is None
```

**Stream Consumer Testing**:

```python
# tests/integration/test_redis_streams.py
def test_redis_stream_consumer(redis_client):
    """Test Redis stream consumer"""
    stream_name = "test:stream"
    
    # Add message to stream
    message_id = redis_client.xadd(stream_name, {"type": "test", "data": "value"})
    
    # Read from stream
    messages = redis_client.xread({stream_name: "0"}, count=1)
    
    assert len(messages) == 1
    assert messages[0][1][0][1][b"type"] == b"test"
```

**Cache Invalidation Flow Testing**:

```python
# tests/integration/test_cache_invalidation.py
def test_cache_invalidation_flow(redis_client):
    """Test cache invalidation flow"""
    # Set cache
    redis_client.setex("user:123:profile", 3600, '{"name": "Test User"}')
    
    # Verify cache exists
    assert redis_client.get("user:123:profile") is not None
    
    # Invalidate cache
    redis_client.delete("user:123:profile")
    
    # Verify cache is gone
    assert redis_client.get("user:123:profile") is None
```

### NGINX Integration Testing

**NGINX Config Validation**:

```python
# tests/integration/test_nginx_config.py
import subprocess

def test_nginx_config_valid():
    """Test NGINX configuration is valid"""
    result = subprocess.run(
        ["nginx", "-t", "-c", "config/nginx.conf"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "syntax is ok" in result.stdout
```

**Reverse-Proxy Routing Tests**:

```python
# tests/integration/test_nginx_routing.py
import requests

def test_nginx_api_routing():
    """Test NGINX API routing"""
    response = requests.get("http://localhost/api/v1/users/123")
    
    assert response.status_code == 200
    assert "X-Cache-Status" in response.headers
```

**TLS Handshake Tests**:

```python
# tests/integration/test_nginx_tls.py
import ssl
import socket

def test_nginx_tls_handshake():
    """Test NGINX TLS handshake"""
    context = ssl.create_default_context()
    
    with socket.create_connection(("localhost", 443)) as sock:
        with context.wrap_socket(sock, server_hostname="api.example.com") as ssock:
            assert ssock.version() in ["TLSv1.2", "TLSv1.3"]
```

### Data Pipeline Integration Testing

**Prefect ETL Integration Tests**:

```python
# tests/integration/test_prefect_etl.py
from prefect import flow, task
from prefect.testing import prefect_test_harness

def test_etl_flow():
    """Test Prefect ETL flow"""
    with prefect_test_harness():
        @flow
        def test_flow():
            @task
            def load_data():
                return [1, 2, 3]
            
            @task
            def transform_data(data):
                return [x * 2 for x in data]
            
            data = load_data()
            result = transform_data(data)
            return result
        
        result = test_flow()
        assert result == [2, 4, 6]
```

**End-to-End Parquet Validation**:

```python
# tests/integration/test_parquet_pipeline.py
import polars as pl
import pyarrow.parquet as pq

def test_parquet_pipeline():
    """Test end-to-end Parquet pipeline"""
    # Load Parquet file
    df = pl.read_parquet("test_data/roads.parquet")
    
    # Validate schema
    assert "road_id" in df.columns
    assert "risk_level" in df.columns
    assert df["road_id"].dtype == pl.Int64
    
    # Validate data
    assert df.height > 0
    assert df["risk_level"].unique().to_list() == ["low", "medium", "high"]
    
    # Write and read back
    df.write_parquet("test_output/roads.parquet")
    df_read = pl.read_parquet("test_output/roads.parquet")
    
    assert df_read.equals(df)
```

**GeoParquet Schema Compatibility**:

```python
# tests/integration/test_geoparquet_schema.py
import geopandas as gpd

def test_geoparquet_schema_compatibility():
    """Test GeoParquet schema compatibility"""
    gdf = gpd.read_parquet("test_data/roads.geoparquet")
    
    # Validate geometry column
    assert "geometry" in gdf.columns
    assert gdf.geometry.crs == "EPSG:4326"
    assert gdf.geometry.geom_type.unique()[0] == "LineString"
    
    # Validate metadata
    assert gdf.attrs.get("geo") is not None
```

### ML/ONNX Integration Testing

**Feature Drift Detection**:

```python
# tests/integration/test_feature_drift.py
import numpy as np
from scipy import stats

def test_feature_drift_detection():
    """Test ML feature drift detection"""
    # Training data statistics
    train_mean = 10.0
    train_std = 2.0
    
    # Production data
    production_data = np.array([12.0, 11.0, 13.0, 10.0, 12.0])
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(
        production_data,
        lambda x: stats.norm.cdf(x, train_mean, train_std)
    )
    
    # Alert if p-value < 0.05 (significant drift)
    assert p_value > 0.05, "Feature drift detected"
```

**ONNX Inference Validation**:

```python
# tests/integration/test_onnx_inference.py
import onnxruntime as ort
import numpy as np

def test_onnx_inference_cpu():
    """Test ONNX inference on CPU"""
    session = ort.InferenceSession(
        "models/risk_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    outputs = session.run(None, {"input": input_data})
    
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 3)

def test_onnx_inference_gpu():
    """Test ONNX inference on GPU"""
    session = ort.InferenceSession(
        "models/risk_model.onnx",
        providers=["CUDAExecutionProvider"]
    )
    
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    outputs = session.run(None, {"input": input_data})
    
    assert len(outputs) == 1
```

**Model Reproducibility Tests**:

```python
# tests/integration/test_model_reproducibility.py
def test_model_reproducibility():
    """Test ML model reproducibility"""
    model = load_model("models/risk_model.pkl")
    
    input_data = np.array([[1.0, 2.0, 3.0]])
    
    # Run inference multiple times
    results = []
    for _ in range(10):
        result = model.predict(input_data)
        results.append(result)
    
    # All results should be identical
    assert all(np.array_equal(results[0], r) for r in results)
```

## End-to-End (E2E) Testing

### E2E Testing Framework

**Folder Structure**:

```
tests/e2e/
├── conftest.py
├── fixtures/
│   ├── test_data/
│   ├── test_models/
│   └── test_configs/
├── workflows/
│   ├── test_api_to_db_to_cache.py
│   ├── test_mlflow_to_inference.py
│   ├── test_nicegui_to_fastapi.py
│   ├── test_etl_dag.py
│   └── test_gis_tiling.py
└── helpers/
    ├── api_client.py
    ├── db_client.py
    └── ml_client.py
```

**Full API → DB → Cache → ML → UI Workflow**:

```python
# tests/e2e/workflows/test_api_to_ml_workflow.py
import pytest
import requests
import time

@pytest.mark.e2e
def test_api_to_ml_workflow():
    """Test complete API → DB → Cache → ML → UI workflow"""
    # 1. API request
    response = requests.post(
        "http://localhost/api/v1/analyze",
        json={"text": "Test input"}
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    # 2. Wait for processing
    for _ in range(30):
        status_response = requests.get(f"http://localhost/api/v1/jobs/{job_id}")
        status = status_response.json()["status"]
        
        if status == "completed":
            break
        time.sleep(1)
    
    assert status == "completed"
    
    # 3. Verify result
    result_response = requests.get(f"http://localhost/api/v1/jobs/{job_id}/result")
    result = result_response.json()
    
    assert "risk_level" in result
    assert result["risk_level"] in ["low", "medium", "high"]
    
    # 4. Verify cache
    cache_key = f"ml:inference:{job_id}"
    cached = redis_client.get(cache_key)
    assert cached is not None
    
    # 5. Verify database
    db_result = db_client.query("SELECT * FROM analysis_results WHERE job_id = %s", job_id)
    assert db_result is not None
```

**MLflow → Model Registry → Inference Pipeline**:

```python
# tests/e2e/workflows/test_mlflow_pipeline.py
import mlflow
import requests

@pytest.mark.e2e
def test_mlflow_to_inference_pipeline():
    """Test MLflow → model registry → inference pipeline"""
    # 1. Load model from MLflow
    model_uri = "models:/risk-model/Production"
    model = mlflow.sklearn.load_model(model_uri)
    
    # 2. Run inference
    input_data = {"features": [1.0, 2.0, 3.0]}
    prediction = model.predict([input_data["features"]])
    
    # 3. Verify prediction format
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]  # Risk levels
    
    # 4. Verify inference API
    api_response = requests.post(
        "http://localhost/api/v1/ml/infer",
        json=input_data
    )
    assert api_response.status_code == 200
    api_prediction = api_response.json()["prediction"]
    assert api_prediction == int(prediction[0])
```

**NiceGUI → FastAPI → Postgres → Redis → ONNX**:

```python
# tests/e2e/workflows/test_nicegui_workflow.py
from nicegui import ui
import asyncio

@pytest.mark.e2e
async def test_nicegui_to_onnx_workflow():
    """Test NiceGUI → FastAPI → Postgres → Redis → ONNX workflow"""
    # 1. Start NiceGUI app
    app = create_nicegui_app()
    
    # 2. Simulate user interaction
    await app.page("/dashboard").open()
    await app.page("/dashboard").fill_input("text_input", "Test input")
    await app.page("/dashboard").click_button("analyze")
    
    # 3. Wait for result
    await asyncio.sleep(5)
    
    # 4. Verify result displayed
    result = await app.page("/dashboard").get_text("result")
    assert "risk_level" in result
    
    # 5. Verify backend processing
    db_result = db_client.query("SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT 1")
    assert db_result is not None
    
    # 6. Verify cache
    cache_key = f"ml:inference:{db_result['job_id']}"
    cached = redis_client.get(cache_key)
    assert cached is not None
```

**ETL Pipeline DAG Correctness**:

```python
# tests/e2e/workflows/test_etl_dag.py
from prefect import flow

@pytest.mark.e2e
def test_etl_dag_correctness():
    """Test complete ETL DAG correctness"""
    # Run ETL flow
    result = run_etl_flow(
        input_path="s3://data-lake/raw/roads.parquet",
        output_path="s3://data-lake/processed/roads-v2.parquet"
    )
    
    # Verify output exists
    assert s3_object_exists(result["output_path"])
    
    # Verify output schema
    df = pl.read_parquet(result["output_path"])
    assert "road_id" in df.columns
    assert "risk_level" in df.columns
    
    # Verify data quality
    assert df.height > 0
    assert df["risk_level"].unique().to_list() == ["low", "medium", "high"]
    
    # Verify lineage recorded
    lineage = get_lineage(result["output_path"])
    assert len(lineage["parents"]) > 0
```

**GIS Tiling Pipeline**:

```python
# tests/e2e/workflows/test_gis_tiling.py
def test_gis_tiling_pipeline():
    """Test GIS tiling pipeline from raw data to tile server"""
    # 1. Load raw data
    gdf = gpd.read_file("test_data/roads.geojson")
    
    # 2. Generate tiles
    tiles = generate_tiles(gdf, zoom_levels=[10, 11, 12])
    
    # 3. Verify tiles generated
    assert len(tiles) > 0
    
    # 4. Verify tile format
    for tile in tiles:
        assert "z" in tile
        assert "x" in tile
        assert "y" in tile
        assert "data" in tile
    
    # 5. Verify tile server can serve tiles
    tile_response = requests.get(
        f"http://localhost/tiles/{tiles[0]['z']}/{tiles[0]['x']}/{tiles[0]['y']}.pbf"
    )
    assert tile_response.status_code == 200
    assert tile_response.headers["Content-Type"] == "application/x-protobuf"
```

## Contract Testing

### API Contract Testing

**OpenAPI + Schemathesis**:

```python
# tests/contract/test_api_contract.py
import schemathesis

schema = schemathesis.from_file("openapi.yaml")

@schema.parametrize()
def test_api_contract(case):
    """Test API contract compliance"""
    response = case.call()
    case.validate_response(response)
```

**API Contract YAML**:

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: Roads API
  version: 1.0.0
paths:
  /api/v1/roads/{road_id}:
    get:
      parameters:
        - name: road_id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Road data
          content:
            application/json:
              schema:
                type: object
                required:
                  - road_id
                  - risk_level
                properties:
                  road_id:
                    type: integer
                  risk_level:
                    type: string
                    enum: [low, medium, high]
```

### Data Contract Testing

**Schema + Metadata Validation**:

```python
# tests/contract/test_data_contract.py
import yaml
import polars as pl

def test_data_contract_compliance():
    """Test data contract compliance"""
    # Load contract
    with open("contracts/etl-roads-v2.yaml") as f:
        contract = yaml.safe_load(f)
    
    # Load data
    df = pl.read_parquet("test_data/roads-v2.parquet")
    
    # Validate schema
    required_fields = contract["schema"]["requiredFields"]
    for field in required_fields:
        assert field in df.columns
    
    # Validate constraints
    for field_spec in contract["schema"]["fields"]:
        field_name = field_spec["name"]
        if "constraints" in field_spec:
            constraints = field_spec["constraints"]
            
            if "allowedValues" in constraints:
                allowed = set(constraints["allowedValues"])
                actual = set(df[field_name].unique().to_list())
                assert actual.issubset(allowed)
```

**GeoParquet Contract**:

```python
# tests/contract/test_geoparquet_contract.py
import geopandas as gpd

def test_geoparquet_contract():
    """Test GeoParquet contract compliance"""
    gdf = gpd.read_parquet("test_data/roads.geoparquet")
    
    # Validate CRS
    assert gdf.crs == "EPSG:4326"
    
    # Validate geometry type
    assert gdf.geometry.geom_type.unique()[0] == "LineString"
    
    # Validate spatial extent
    bbox = gdf.total_bounds
    assert bbox[0] >= -180 and bbox[2] <= 180  # Longitude
    assert bbox[1] >= -90 and bbox[3] <= 90    # Latitude
```

### ML Contract Testing

**Feature Schema Validation**:

```python
# tests/contract/test_ml_contract.py
def test_ml_feature_schema():
    """Test ML feature schema contract"""
    contract = load_contract("contracts/ml-risk-model-input-v1.yaml")
    
    # Test valid input
    valid_input = {
        "road_id": 123,
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
        "road_type": "highway",
        "traffic_volume": 5000.0
    }
    
    assert validate_against_contract(valid_input, contract)
    
    # Test invalid input
    invalid_input = {
        "road_id": 123,
        "road_type": "invalid_type"  # Not in allowed values
    }
    
    assert not validate_against_contract(invalid_input, contract)
```

**Output Shape Validation**:

```python
def test_ml_output_shape():
    """Test ML model output shape contract"""
    model = load_model("models/risk_model.onnx")
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    output = model.predict(input_data)
    
    # Validate output shape
    assert output.shape == (1, 3)  # Batch size 1, 3 risk levels
    
    # Validate probabilities sum to 1
    assert np.allclose(output.sum(axis=1), 1.0)
```

### GIS Contract Testing

**SRS Validation**:

```python
# tests/contract/test_gis_contract.py
def test_gis_srs_contract():
    """Test GIS SRS contract compliance"""
    gdf = gpd.read_file("test_data/roads.geojson")
    
    # Validate CRS
    assert gdf.crs == "EPSG:4326"
    
    # Validate all geometries use same CRS
    assert all(gdf.geometry.crs == "EPSG:4326" for _ in gdf.geometry)
```

**Bounding Box Validation**:

```python
def test_gis_bbox_contract():
    """Test GIS bounding box contract"""
    contract = load_contract("contracts/geospatial-roads-v2.yaml")
    bbox_requirements = contract["spatialRequirements"]["extent"]
    
    gdf = gpd.read_file("test_data/roads.geojson")
    bbox = gdf.total_bounds
    
    assert bbox[0] >= bbox_requirements["minX"]
    assert bbox[2] <= bbox_requirements["maxX"]
    assert bbox[1] >= bbox_requirements["minY"]
    assert bbox[3] <= bbox_requirements["maxY"]
```

## Test Data Strategy

### Synthetic Test Data

**Deterministic Generation**:

```python
# tests/fixtures/generate_synthetic_data.py
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)  # Deterministic

def generate_synthetic_roads(n: int = 1000) -> pd.DataFrame:
    """Generate deterministic synthetic road data"""
    np.random.seed(42)
    
    roads = []
    for i in range(n):
        roads.append({
            "road_id": i + 1,
            "road_name": fake.street_name(),
            "risk_level": np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
            "geometry": generate_random_linestring()
        })
    
    return pd.DataFrame(roads)
```

**ML Model Mock Datasets**:

```python
# tests/fixtures/generate_ml_data.py
def generate_ml_training_data(n: int = 10000) -> np.ndarray:
    """Generate synthetic ML training data"""
    np.random.seed(42)
    
    # Generate features with known distribution
    features = np.random.normal(0, 1, (n, 10))
    
    # Generate labels based on features
    labels = (features.sum(axis=1) > 0).astype(int)
    
    return features, labels
```

**Synthetic GIS Grids**:

```python
# tests/fixtures/generate_gis_data.py
import geopandas as gpd
from shapely.geometry import LineString

def generate_synthetic_road_network(bbox: tuple, n_roads: int = 100) -> gpd.GeoDataFrame:
    """Generate synthetic road network"""
    np.random.seed(42)
    
    roads = []
    for i in range(n_roads):
        # Generate random line within bbox
        x1 = np.random.uniform(bbox[0], bbox[2])
        y1 = np.random.uniform(bbox[1], bbox[3])
        x2 = np.random.uniform(bbox[0], bbox[2])
        y2 = np.random.uniform(bbox[1], bbox[3])
        
        geometry = LineString([(x1, y1), (x2, y2)])
        roads.append({
            "road_id": i + 1,
            "geometry": geometry
        })
    
    return gpd.GeoDataFrame(roads, crs="EPSG:4326")
```

### Reduced "Golden Dataset"

**Stable Minibatches**:

```python
# tests/fixtures/golden_datasets.py
def create_golden_dataset(source_path: str, output_path: str, n_samples: int = 1000):
    """Create golden dataset from real data"""
    df = pl.read_parquet(source_path)
    
    # Stratified sample to maintain distribution
    golden = df.sample(n=n_samples, seed=42)
    
    # Save with version
    golden.write_parquet(output_path)
    
    # Compute and store hash
    hash_value = compute_file_hash(output_path)
    with open(f"{output_path}.hash", "w") as f:
        f.write(hash_value)
```

**Versioned Test Dataset Directories**:

```
test_data/
├── roads/
│   ├── v1/
│   │   ├── roads-v1.parquet
│   │   └── roads-v1.parquet.hash
│   └── v2/
│       ├── roads-v2.parquet
│       └── roads-v2.parquet.hash
└── metadata.yaml
```

**Hash-Based Validation**:

```python
# tests/fixtures/validate_golden_dataset.py
def validate_golden_dataset(file_path: str) -> bool:
    """Validate golden dataset hasn't changed"""
    expected_hash = Path(f"{file_path}.hash").read_text().strip()
    actual_hash = compute_file_hash(file_path)
    
    return expected_hash == actual_hash
```

### Parquet/GeoParquet Test Cases

**Small Row-Group Subsets**:

```python
# tests/fixtures/create_parquet_test_cases.py
def create_parquet_test_cases(source_path: str, output_dir: str):
    """Create small Parquet test cases"""
    df = pl.read_parquet(source_path)
    
    # Create test cases for different scenarios
    test_cases = {
        "empty": pl.DataFrame(schema=df.schema),
        "single_row": df.head(1),
        "small_batch": df.head(100),
        "edge_cases": df.filter(pl.col("risk_level") == "high").head(10)
    }
    
    for name, test_df in test_cases.items():
        test_df.write_parquet(f"{output_dir}/{name}.parquet")
```

**Schema Coverage Map**:

```python
# tests/fixtures/schema_coverage.py
def test_schema_coverage():
    """Test that test data covers all schema cases"""
    schema = load_schema("schemas/roads-v2.json")
    
    test_cases = [
        "test_data/roads/empty.parquet",
        "test_data/roads/single_row.parquet",
        "test_data/roads/small_batch.parquet"
    ]
    
    for test_case in test_cases:
        df = pl.read_parquet(test_case)
        
        # Verify all required fields present
        required_fields = schema["requiredFields"]
        assert all(field in df.columns for field in required_fields)
```

### Sensitive Data Handling

**Masking**:

```python
# tests/fixtures/mask_sensitive_data.py
def mask_sensitive_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mask sensitive data in test dataset"""
    masked = df.copy()
    
    # Mask email addresses
    if "email" in masked.columns:
        masked["email"] = masked["email"].str.replace(
            r'(.+?)@(.+?)', r'user\1@example.com', regex=True
        )
    
    # Mask phone numbers
    if "phone" in masked.columns:
        masked["phone"] = "555-000-0000"
    
    return masked
```

**Tokenization**:

```python
# tests/fixtures/tokenize_data.py
def tokenize_sensitive_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tokenize sensitive data"""
    tokenized = df.copy()
    
    # Tokenize user IDs
    if "user_id" in tokenized.columns:
        token_map = {uid: f"TOKEN_{i}" for i, uid in enumerate(tokenized["user_id"].unique())}
        tokenized["user_id"] = tokenized["user_id"].map(token_map)
    
    return tokenized
```

**Synthetic Substitution**:

```python
# tests/fixtures/synthetic_substitution.py
from faker import Faker

fake = Faker()

def substitute_with_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Substitute sensitive data with synthetic"""
    synthetic = df.copy()
    
    # Replace names with synthetic
    if "name" in synthetic.columns:
        synthetic["name"] = [fake.name() for _ in range(len(synthetic))]
    
    return synthetic
```

## Performance Testing

### Load Testing NGINX/API

**k6 Load Test**:

```javascript
// tests/performance/load_test_api.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '30s', target: 100 },  // Ramp up to 100 users
        { duration: '1m', target: 100 },  // Stay at 100 users
        { duration: '30s', target: 0 },     // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<200'],  // 95% of requests < 200ms
        http_req_failed: ['rate<0.01'],     // Error rate < 1%
    },
};

export default function () {
    const response = http.get('http://localhost/api/v1/roads/123');
    
    check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 200ms': (r) => r.timings.duration < 200,
    });
    
    sleep(1);
}
```

**Locust Load Test**:

```python
# tests/performance/locust_load_test.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def get_road(self):
        self.client.get("/api/v1/roads/123")
    
    @task(3)
    def get_roads_list(self):
        self.client.get("/api/v1/roads?limit=100")
```

### Postgres Performance Testing

**pgbench + Custom Geospatial Workloads**:

```bash
# tests/performance/pgbench_geo.sh
#!/bin/bash

# Standard pgbench
pgbench -c 10 -j 2 -T 60 -U postgres mydb

# Custom geospatial workload
psql -U postgres -d mydb -f tests/performance/geo_workload.sql
```

**Custom Geospatial Workload**:

```sql
-- tests/performance/geo_workload.sql
EXPLAIN ANALYZE
SELECT 
    r.road_id,
    r.risk_level,
    z.zone_name
FROM roads_v2 r
JOIN risk_zones z ON ST_Intersects(r.geometry, z.geometry)
WHERE r.risk_level = 'high'
LIMIT 1000;
```

### Redis Latency & Eviction Testing

**Redis Latency Test**:

```python
# tests/performance/test_redis_latency.py
import redis
import time

def test_redis_latency():
    """Test Redis operation latency"""
    client = redis.Redis(host="localhost", port=6379)
    
    latencies = []
    for _ in range(1000):
        start = time.time()
        client.get("test:key")
        latencies.append((time.time() - start) * 1000)  # Convert to ms
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    assert p50 < 1.0, f"P50 latency too high: {p50}ms"
    assert p95 < 5.0, f"P95 latency too high: {p95}ms"
    assert p99 < 10.0, f"P99 latency too high: {p99}ms"
```

### ETL Throughput Benchmarks

**ETL Performance Test**:

```python
# tests/performance/test_etl_throughput.py
import time

def test_etl_throughput():
    """Test ETL pipeline throughput"""
    input_size_mb = 100
    start_time = time.time()
    
    # Run ETL
    result = run_etl_pipeline("test_data/large_input.parquet")
    
    duration = time.time() - start_time
    throughput_mbps = input_size_mb / duration
    
    assert throughput_mbps > 10, f"Throughput too low: {throughput_mbps} MB/s"
```

### ML Inference Latency Testing

**ONNX Inference Latency**:

```python
# tests/performance/test_ml_latency.py
import onnxruntime as ort
import numpy as np
import time

def test_onnx_inference_latency():
    """Test ONNX inference latency"""
    session = ort.InferenceSession("models/risk_model.onnx")
    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    latencies = []
    for _ in range(100):
        start = time.time()
        session.run(None, {"input": input_data})
        latencies.append((time.time() - start) * 1000)  # ms
    
    p95 = np.percentile(latencies, 95)
    assert p95 < 50, f"P95 inference latency too high: {p95}ms"
```

## Chaos Engineering / Fault Injection

### Node Failures in RKE2

**LitmusChaos Example**:

```yaml
# chaos/node-failure.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: node-failure-test
spec:
  appinfo:
    appns: production
    applabel: app=myapp
    appkind: deployment
  chaosServiceAccount: litmus
  monitoring: true
  jobCleanUpPolicy: retain
  experiments:
    - name: node-drain
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: "60"
            - name: NODE_LABEL
              value: "node-role=worker"
```

### Postgres Primary Crash

**Chaos Test**:

```python
# tests/chaos/test_postgres_failover.py
def test_postgres_primary_crash():
    """Test Postgres primary crash and failover"""
    # Get primary node
    primary = get_postgres_primary()
    
    # Crash primary
    crash_node(primary)
    
    # Wait for failover
    time.sleep(30)
    
    # Verify new primary elected
    new_primary = get_postgres_primary()
    assert new_primary != primary
    
    # Verify service still works
    result = db_client.query("SELECT 1")
    assert result is not None
```

### Redis Eviction Storms

**Chaos Test**:

```python
# tests/chaos/test_redis_eviction.py
def test_redis_eviction_storm():
    """Test Redis eviction under memory pressure"""
    # Fill Redis to capacity
    for i in range(100000):
        redis_client.setex(f"test:key:{i}", 3600, "x" * 1000)
    
    # Monitor evictions
    info_before = redis_client.info("stats")
    evictions_before = info_before.get("evicted_keys", 0)
    
    # Add more data to trigger evictions
    for i in range(10000):
        redis_client.setex(f"test:key:new:{i}", 3600, "x" * 1000)
    
    info_after = redis_client.info("stats")
    evictions_after = info_after.get("evicted_keys", 0)
    
    assert evictions_after > evictions_before
```

### Network Partitions

**Chaos Test**:

```python
# tests/chaos/test_network_partition.py
def test_network_partition():
    """Test system behavior under network partition"""
    # Partition network between services
    partition_network("api-service", "database")
    
    # Verify graceful degradation
    response = api_client.get("/api/v1/health")
    assert response.status_code in [200, 503]  # Service or unavailable
    
    # Restore network
    restore_network()
    
    # Verify recovery
    time.sleep(10)
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
```

## Test Environments

### Local Ephemeral Environment

**Docker Compose**:

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres-test:
    image: postgres:15
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
  
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
  
  minio-test:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: test
      MINIO_ROOT_PASSWORD: test
    ports:
      - "9000:9000"
      - "9001:9001"
```

### CI Ephemeral Environment

**GitHub Actions**:

```yaml
# .github/workflows/test.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://test:test@localhost/testdb
          REDIS_URL: redis://localhost:6379
        run: pytest tests/
```

### Shared Staging Cluster

**RKE2 Staging Environment**:

```yaml
# k8s/staging/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: staging-test
```

**GitOps Automation**:

```yaml
# argocd/staging-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: staging-test
spec:
  project: default
  source:
    repoURL: https://github.com/org/repo
    path: k8s/staging
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: staging-test
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## CI/CD Integration

### Automated Test Pipelines

**GitHub Actions Full Pipeline**:

```yaml
# .github/workflows/full-test-pipeline.yml
name: Full Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit/
  
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres: # ... as above
      redis: # ... as above
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/
  
  contract-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run contract tests
        run: pytest tests/contract/
  
  e2e-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    steps:
      - uses: actions/checkout@v3
      - name: Deploy test environment
        run: ./scripts/deploy-test-env.sh
      - name: Run E2E tests
        run: pytest tests/e2e/
  
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Trivy scan
        run: |
          trivy fs --exit-code 1 --severity HIGH,CRITICAL .
      - name: Generate SBOM
        run: |
          trivy fs --format cyclonedx -o sbom.json .
```

### Require Test Success Before Deploy

**GitLab CI**:

```yaml
# .gitlab-ci.yml
stages:
  - test
  - security
  - deploy

unit-tests:
  stage: test
  script:
    - pytest tests/unit/
  only:
    - merge_requests
    - main

integration-tests:
  stage: test
  services:
    - postgres:15
    - redis:7-alpine
  script:
    - pytest tests/integration/
  only:
    - merge_requests
    - main

deploy-staging:
  stage: deploy
  script:
    - ./scripts/deploy.sh staging
  only:
    - main
  when: on_success
```

## Testing in Air-Gapped Environments

### Offline Test Harnesses

**Packaged Test Environment**:

```bash
# package_test_environment.sh
#!/bin/bash

# Package test data
tar czf test_data.tar.gz test_data/

# Package test containers
docker save postgres:15 redis:7-alpine > test_containers.tar

# Package test scripts
tar czf test_scripts.tar.gz tests/ scripts/
```

### Test Dataset Packaging

**Test Data Bundle**:

```
test_data_bundle/
├── datasets/
│   ├── roads-v2.parquet
│   └── metadata.yaml
├── models/
│   └── risk_model.onnx
└── configs/
    └── test_config.yaml
```

## Observability for Testing

### Using Logs, Metrics, Traces

**PromQL Queries**:

```promql
# API latency during tests
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket{test_run="integration"}[5m])
)

# Error rate during tests
rate(http_requests_total{status=~"5..", test_run="integration"}[5m])
```

**Loki Qus**:

```logql
# Test execution logs
{test_suite="integration"} |= "ERROR" | json | test_name="test_api_workflow"

# ETL pipeline logs during tests
{job="etl-pipeline", test_run="e2e"} | json | level="error"
```

## Anti-Patterns

### "All Integration Tests, No Unit Tests"

**Symptom**: Test suite is slow, tests are flaky, hard to debug.

**Fix**: Follow testing pyramid: 50-60% unit tests, 30-40% integration, 10-20% E2E.

### "Only Testing Happy Path"

**Symptom**: Tests pass but production fails on edge cases.

**Fix**: Test error cases, edge cases, boundary conditions.

### "End-to-End Tests That Don't Test Anything End-to-End"

**Symptom**: E2E tests mock too much, don't test real workflows.

**Fix**: E2E tests should use real services, test complete workflows.

### "Fake Mocks That Don't Match Real Services"

**Symptom**: Tests pass but integration fails.

**Fix**: Use contract tests, verify mocks match real service behavior.

### "ML Model Tests That Ignore Drift"

**Symptom**: Model works in tests but fails in production.

**Fix**: Test feature drift, monitor model performance, test with production-like data.

### "GIS Pipeline Tests Without Spatial Assertions"

**Symptom**: Tests pass but spatial results are wrong.

**Fix**: Validate CRS, geometry types, spatial relationships, bounding boxes.

### "Test Data That Is Too Large / Too Small"

**Symptom**: Tests are slow or don't catch real issues.

**Fix**: Use representative test data, golden datasets, synthetic data generators.

### "Testing in Production Without Guardrails"

**Symptom**: Production incidents from test data/load.

**Fix**: Use separate test environments, read-only production tests, feature flags.

### "CI Tests That Don't Run Locally"

**Symptom**: Developers can't reproduce CI failures.

**Fix**: Use same test environment locally, Docker Compose, documented setup.

### "Flaky Tests Tolerated for Months"

**Symptom**: Tests randomly fail, team ignores failures.

**Fix**: Fix flaky tests immediately, use retries sparingly, investigate root causes.

### "Ignoring Timezone/Data-Type Edge Cases"

**Symptom**: Tests pass but production fails on edge cases.

**Fix**: Test timezone handling, data type boundaries, null handling.

### "Skipping DR/Chaos Tests Due to Fear"

**Symptom**: System untested under failure conditions.

**Fix**: Regular chaos tests, DR drills, failure scenario testing.

## Final Checklists

### Unit Testing Checklist

- [ ] All business logic has unit tests
- [ ] Test coverage > 80%
- [ ] Tests run in < 1s each
- [ ] No external dependencies
- [ ] Tests are deterministic
- [ ] Edge cases tested
- [ ] Error cases tested

### Integration Testing Checklist

- [ ] Critical service interactions tested
- [ ] Database integration tested
- [ ] Cache integration tested
- [ ] API contracts validated
- [ ] ETL transformations tested
- [ ] ML pipelines tested
- [ ] GIS computations tested

### E2E Testing Checklist

- [ ] Critical user journeys tested
- [ ] Full workflows tested
- [ ] Cross-service interactions tested
- [ ] Error recovery tested
- [ ] Performance validated
- [ ] Security validated

### GIS Testing Checklist

- [ ] CRS validation
- [ ] Geometry type validation
- [ ] Spatial relationship tests
- [ ] Bounding box validation
- [ ] Tile generation tests
- [ ] Spatial query correctness

### ML Testing Checklist

- [ ] Feature schema validation
- [ ] Model input/output validation
- [ ] Inference latency tests
- [ ] Feature drift detection
- [ ] Model reproducibility
- [ ] Confidence calibration

### Data Contract Testing Checklist

- [ ] Schema validation
- [ ] Constraint validation
- [ ] Type validation
- [ ] Version compatibility
- [ ] Breaking change detection

### Performance Testing Checklist

- [ ] API latency tests
- [ ] Database query performance
- [ ] Cache hit rate validation
- [ ] ETL throughput tests
- [ ] ML inference latency
- [ ] Load testing

### Chaos Testing Checklist

- [ ] Node failure tests
- [ ] Database failover tests
- [ ] Cache eviction tests
- [ ] Network partition tests
- [ ] Service crash tests
- [ ] Recovery validation

### CI/CD Test Pipeline Checklist

- [ ] Unit tests in CI
- [ ] Integration tests in CI
- [ ] Contract tests in CI
- [ ] E2E tests in CI
- [ ] Security scans in CI
- [ ] Performance tests in CI
- [ ] Test artifacts stored
- [ ] Test reports generated

## See Also

- **[Testing Best Practices](../operations-monitoring/testing-best-practices.md)** - General testing patterns
- **[CI/CD Pipeline Best Practices](../architecture-design/ci-cd-pipelines.md)** - Pipeline patterns
- **[Data Contract Testing](../data-governance/metadata-provenance-contracts.md)** - Contract validation

---

*This guide provides a complete framework for end-to-end testing in distributed systems. Start with unit tests, add integration tests for critical paths, then implement E2E tests for key workflows. The goal is confidence without complexity.*

