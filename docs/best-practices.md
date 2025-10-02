# Technical Best Practices

## Overview

This section demonstrates comprehensive technical best practices across geospatial systems, data engineering, cloud architecture, and modern web development. These practices represent industry standards and proven methodologies used in production environments.

## Geospatial Data Engineering Best Practices

### Spatial Data Management

#### GeoParquet Implementation
```python
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import Point, Polygon
import pyproj

def create_optimized_geoparquet(data_path, output_path, target_crs='EPSG:3857'):
    """
    Create optimized GeoParquet with spatial indexing and compression
    """
    # Load spatial data
    gdf = gpd.read_file(data_path)
    
    # Ensure consistent CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    # Add spatial index
    gdf = gdf.set_index('id') if 'id' not in gdf.columns else gdf
    
    # Optimize for analytics
    gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt)
    
    # Write with compression and spatial metadata
    gdf.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=True,
        schema={
            'geometry': 'string',
            'properties': 'struct<...>'  # Define your schema
        }
    )
    
    return gdf

# Usage example
optimized_data = create_optimized_geoparquet(
    'input.geojson',
    'output.parquet',
    'EPSG:3857'
)
```

#### Spatial Indexing Strategies
```python
import geopandas as gpd
from shapely.geometry import Point
import rtree

def create_spatial_index(gdf, index_name='spatial_idx'):
    """
    Create efficient spatial index for fast spatial queries
    """
    # Create R-tree index
    spatial_index = rtree.index.Index()
    
    for idx, row in gdf.iterrows():
        bounds = row.geometry.bounds
        spatial_index.insert(idx, bounds)
    
    return spatial_index

def spatial_join_optimized(left_gdf, right_gdf, predicate='intersects'):
    """
    Optimized spatial join using spatial indexing
    """
    # Create spatial index for right dataframe
    right_index = create_spatial_index(right_gdf)
    
    results = []
    for idx, left_row in left_gdf.iterrows():
        # Query spatial index
        candidates = list(right_index.intersection(left_row.geometry.bounds))
        
        # Perform precise geometric operations only on candidates
        for candidate_idx in candidates:
            if getattr(left_row.geometry, predicate)(right_gdf.iloc[candidate_idx].geometry):
                results.append({
                    'left_idx': idx,
                    'right_idx': candidate_idx
                })
    
    return results
```

### Raster Processing Pipelines

#### Scalable Raster Operations
```python
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import dask.array as da
from dask.distributed import Client

def process_large_raster(input_path, output_path, target_crs='EPSG:3857'):
    """
    Process large raster files using Dask for parallel processing
    """
    with rasterio.open(input_path) as src:
        # Calculate transform for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        # Create output profile
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Process in chunks
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(0, src.height, 1000):  # Process in 1000-pixel chunks
                window = rasterio.windows.Window(0, i, src.width, min(1000, src.height - i))
                
                # Read chunk
                data = src.read(window=window)
                
                # Process chunk (example: apply some transformation)
                processed_data = process_raster_chunk(data)
                
                # Write chunk
                dst.write(processed_data, window=window)

def process_raster_chunk(data):
    """
    Process a chunk of raster data
    """
    # Example: Apply NDVI calculation
    if data.shape[0] >= 2:  # Ensure we have at least 2 bands
        red = data[0].astype(np.float32)
        nir = data[1].astype(np.float32)
        
        # Avoid division by zero
        ndvi = np.where(
            (nir + red) != 0,
            (nir - red) / (nir + red),
            0
        )
        
        return np.clip(ndvi, -1, 1)
    
    return data
```

## Cloud Architecture Best Practices

### AWS Infrastructure Patterns

#### Serverless Geospatial Processing
```python
import boto3
import json
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger()
tracer = Tracer()

@tracer.capture_lambda_handler
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Serverless geospatial processing with AWS Lambda
    """
    try:
        # Extract parameters from event
        s3_bucket = event['s3_bucket']
        s3_key = event['s3_key']
        operation = event['operation']
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Process based on operation type
        if operation == 'spatial_analysis':
            result = perform_spatial_analysis(s3_bucket, s3_key)
        elif operation == 'raster_processing':
            result = perform_raster_processing(s3_bucket, s3_key)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'result': result
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing geospatial data: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def perform_spatial_analysis(bucket: str, key: str) -> dict:
    """
    Perform spatial analysis on S3-stored data
    """
    # Implementation for spatial analysis
    pass

def perform_raster_processing(bucket: str, key: str) -> dict:
    """
    Perform raster processing on S3-stored data
    """
    # Implementation for raster processing
    pass
```

#### Containerized Microservices
```dockerfile
# Multi-stage build for geospatial processing
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialite-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "app.py"]
```

### Kubernetes Deployment Patterns

#### Geospatial Service Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: geospatial-api
  labels:
    app: geospatial-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: geospatial-api
  template:
    metadata:
      labels:
        app: geospatial-api
    spec:
      containers:
      - name: geospatial-api
        image: sempervent/geospatial-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: geospatial-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: geospatial-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: geospatial-api-service
spec:
  selector:
    app: geospatial-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Data Engineering Best Practices

### ETL Pipeline Design

#### Modern Airflow DAGs
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'geospatial_etl_pipeline',
    default_args=default_args,
    description='Geospatial ETL pipeline with data quality checks',
    schedule_interval=timedelta(hours=6),
    max_active_runs=1,
    tags=['geospatial', 'etl', 'data-quality']
)

def extract_geospatial_data(**context):
    """
    Extract geospatial data from multiple sources
    """
    # Extract from various sources
    sources = [
        's3://data-lake/raw/geospatial/',
        'postgresql://source-db/geospatial_data',
        'https://api.example.com/geospatial'
    ]
    
    extracted_data = []
    for source in sources:
        # Implementation for each source
        data = extract_from_source(source)
        extracted_data.append(data)
    
    return extracted_data

def transform_geospatial_data(**context):
    """
    Transform and clean geospatial data
    """
    extracted_data = context['task_instance'].xcom_pull(task_ids='extract_data')
    
    transformed_data = []
    for data in extracted_data:
        # Apply transformations
        cleaned_data = clean_geospatial_data(data)
        standardized_data = standardize_crs(cleaned_data)
        transformed_data.append(standardized_data)
    
    return transformed_data

def load_to_warehouse(**context):
    """
    Load transformed data to data warehouse
    """
    transformed_data = context['task_instance'].xcom_pull(task_ids='transform_data')
    
    # Load to data warehouse
    for data in transformed_data:
        load_to_postgres(data)
        load_to_s3(data)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_geospatial_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_geospatial_data,
    dag=dag
)

quality_check = DockerOperator(
    task_id='data_quality_check',
    image='geospatial-quality-checker:latest',
    command=['python', 'quality_check.py'],
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_to_warehouse,
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> quality_check >> load_task
```

### Real-time Data Processing

#### Kafka Stream Processing
```python
from kafka import KafkaConsumer, KafkaProducer
import json
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

class GeospatialStreamProcessor:
    def __init__(self, kafka_config, postgres_config):
        self.consumer = KafkaConsumer(
            'geospatial-events',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='geospatial-processor'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.postgres_config = postgres_config
    
    def process_stream(self):
        """
        Process geospatial data stream in real-time
        """
        for message in self.consumer:
            try:
                # Parse incoming data
                event_data = message.value
                
                # Process geospatial event
                processed_data = self.process_geospatial_event(event_data)
                
                # Store in database
                self.store_processed_data(processed_data)
                
                # Send to output topic
                self.producer.send('processed-geospatial', processed_data)
                
            except Exception as e:
                print(f"Error processing message: {e}")
                # Send to error topic
                self.producer.send('geospatial-errors', {
                    'original_message': event_data,
                    'error': str(e)
                })
    
    def process_geospatial_event(self, event_data):
        """
        Process individual geospatial event
        """
        # Extract coordinates
        lat = event_data.get('latitude')
        lon = event_data.get('longitude')
        
        if lat is None or lon is None:
            raise ValueError("Missing coordinates")
        
        # Create point geometry
        point = Point(lon, lat)
        
        # Perform spatial analysis
        spatial_analysis = self.perform_spatial_analysis(point, event_data)
        
        # Combine with original data
        processed_data = {
            **event_data,
            'geometry': point.wkt,
            'spatial_analysis': spatial_analysis,
            'processed_at': pd.Timestamp.now().isoformat()
        }
        
        return processed_data
    
    def perform_spatial_analysis(self, point, event_data):
        """
        Perform spatial analysis on point
        """
        # Implementation for spatial analysis
        return {
            'buffer_analysis': self.buffer_analysis(point),
            'proximity_analysis': self.proximity_analysis(point),
            'spatial_join': self.spatial_join_analysis(point)
        }
    
    def store_processed_data(self, data):
        """
        Store processed data in database
        """
        # Implementation for database storage
        pass
```

## API Development Best Practices

### FastAPI Geospatial API

#### Production-Ready API
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import geopandas as gpd
from shapely.geometry import Point, Polygon
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class SpatialQuery(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON geometry")
    buffer_distance: Optional[float] = Field(None, ge=0, description="Buffer distance in meters")
    attributes: Optional[List[str]] = Field(None, description="Attributes to return")
    spatial_operation: str = Field("intersects", description="Spatial operation to perform")

class SpatialResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    processing_time: float
    spatial_bounds: Dict[str, float]

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    database_status: str

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Geospatial API")
    # Initialize database connections, load spatial data, etc.
    yield
    # Shutdown
    logger.info("Shutting down Geospatial API")

# Create FastAPI app
app = FastAPI(
    title="Geospatial Systems API",
    description="Production-ready geospatial API for spatial analysis and data processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sempervent.github.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["sempervent.github.io", "localhost"]
)

# Dependency injection
async def get_database():
    """Database connection dependency"""
    # Implementation for database connection
    pass

# API endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=pd.Timestamp.now().isoformat(),
        version="1.0.0",
        database_status="connected"
    )

@app.post("/spatial/query", response_model=SpatialResponse)
async def spatial_query(
    query: SpatialQuery,
    background_tasks: BackgroundTasks,
    db=Depends(get_database)
):
    """
    Perform spatial query with comprehensive error handling
    """
    try:
        start_time = time.time()
        
        # Validate geometry
        if not validate_geometry(query.geometry):
            raise HTTPException(status_code=400, detail="Invalid geometry")
        
        # Perform spatial query
        results = await perform_spatial_query(query, db)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log query for analytics
        background_tasks.add_task(log_spatial_query, query, processing_time)
        
        return SpatialResponse(
            results=results,
            total_count=len(results),
            processing_time=processing_time,
            spatial_bounds=calculate_bounds(results)
        )
        
    except Exception as e:
        logger.error(f"Spatial query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/spatial/bounds")
async def get_spatial_bounds(db=Depends(get_database)):
    """Get spatial bounds of all data"""
    try:
        bounds = await get_data_bounds(db)
        return {"bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def validate_geometry(geometry: Dict[str, Any]) -> bool:
    """Validate GeoJSON geometry"""
    try:
        from shapely.geometry import shape
        shape(geometry)
        return True
    except:
        return False

async def perform_spatial_query(query: SpatialQuery, db) -> List[Dict[str, Any]]:
    """Perform the actual spatial query"""
    # Implementation for spatial query
    pass

def calculate_bounds(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate spatial bounds of results"""
    # Implementation for bounds calculation
    pass

async def log_spatial_query(query: SpatialQuery, processing_time: float):
    """Log query for analytics"""
    logger.info(f"Query processed in {processing_time:.3f}s: {query.spatial_operation}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## Database Optimization Best Practices

### PostgreSQL/PostGIS Tuning

#### Spatial Database Configuration
```sql
-- PostGIS spatial database optimization
-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Optimize PostgreSQL for spatial workloads
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Spatial indexing best practices
CREATE INDEX CONCURRENTLY idx_spatial_data_geom 
ON spatial_data USING GIST (geometry);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_spatial_data_active 
ON spatial_data USING GIST (geometry) 
WHERE status = 'active';

-- Covering indexes for performance
CREATE INDEX CONCURRENTLY idx_spatial_data_covering 
ON spatial_data USING GIST (geometry) 
INCLUDE (id, name, created_at);

-- Analyze tables for query optimization
ANALYZE spatial_data;
```

#### Query Optimization Patterns
```sql
-- Optimized spatial queries
-- Use spatial indexes effectively
EXPLAIN (ANALYZE, BUFFERS) 
SELECT s1.id, s1.name, ST_Distance(s1.geometry, s2.geometry) as distance
FROM spatial_data s1
CROSS JOIN LATERAL (
    SELECT s2.geometry
    FROM spatial_data s2
    WHERE s2.id != s1.id
    ORDER BY s1.geometry <-> s2.geometry
    LIMIT 1
) s2;

-- Spatial joins with proper indexing
SELECT a.id, b.id, ST_Area(ST_Intersection(a.geometry, b.geometry)) as intersection_area
FROM polygons_a a
JOIN polygons_b b ON ST_Intersects(a.geometry, b.geometry)
WHERE ST_Area(ST_Intersection(a.geometry, b.geometry)) > 1000;

-- Clustering for spatial locality
CLUSTER spatial_data USING idx_spatial_data_geom;
```

## Testing Best Practices

### Comprehensive Test Suite

#### Unit Testing
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

## Performance Monitoring Best Practices

### Application Performance Monitoring

#### Custom Metrics Collection
```python
import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any
import json

class PerformanceMonitor:
    """Performance monitoring for geospatial applications"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def track_performance(self, operation_name: str):
        """Decorator to track function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success metrics
                    self.record_metric(operation_name, {
                        'status': 'success',
                        'execution_time': time.time() - start_time,
                        'memory_usage': psutil.Process().memory_info().rss - start_memory,
                        'timestamp': time.time()
                    })
                    
                    return result
                    
                except Exception as e:
                    # Record error metrics
                    self.record_metric(operation_name, {
                        'status': 'error',
                        'execution_time': time.time() - start_time,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    raise
            
            return wrapper
        return decorator
    
    def record_metric(self, operation: str, metrics: Dict[str, Any]):
        """Record performance metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(metrics)
        
        # Log metrics
        self.logger.info(f"Performance metric: {operation} - {json.dumps(metrics)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, metrics in self.metrics.items():
            if metrics:
                execution_times = [m['execution_time'] for m in metrics if 'execution_time' in m]
                success_count = len([m for m in metrics if m.get('status') == 'success'])
                error_count = len([m for m in metrics if m.get('status') == 'error'])
                
                summary[operation] = {
                    'total_executions': len(metrics),
                    'success_count': success_count,
                    'error_count': error_count,
                    'success_rate': success_count / len(metrics) if metrics else 0,
                    'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                    'min_execution_time': min(execution_times) if execution_times else 0,
                    'max_execution_time': max(execution_times) if execution_times else 0
                }
        
        return summary

# Usage example
monitor = PerformanceMonitor()

@monitor.track_performance('spatial_analysis')
def perform_spatial_analysis(data):
    """Example function with performance monitoring"""
    # Implementation
    time.sleep(0.1)  # Simulate processing
    return {"result": "analysis_complete"}

# Get performance summary
summary = monitor.get_performance_summary()
print(json.dumps(summary, indent=2))
```

## Additional Resources

- [Professional Profile](about.md) - Background and experience
- [Projects Portfolio](projects.md) - Technical implementations
- [Technical Documentation](documentation.md) - Methodologies and best practices
- [Contact & Collaboration](getting-started.md) - Get in touch
