# FastAPI Geospatial API Development

**Objective**: Build high-performance, scalable geospatial APIs using FastAPI with proper spatial data handling and optimization.

FastAPI provides the perfect foundation for geospatial APIs—async support, automatic OpenAPI documentation, and type safety. This guide covers building production-ready geospatial APIs that scale.

## 1) Core API Architecture

### FastAPI Application Setup

```python
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import geopandas as gpd
from shapely.geometry import Point, Polygon
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Geospatial API",
    description="High-performance geospatial data API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SpatialQuery(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON geometry")
    buffer_distance: Optional[float] = Field(None, description="Buffer distance in meters")
    attributes: Optional[List[str]] = Field(None, description="Attributes to return")
    limit: Optional[int] = Field(100, description="Maximum number of results")

class SpatialResponse(BaseModel):
    features: List[Dict[str, Any]]
    total_count: int
    query_time_ms: float
```

**Why**: FastAPI's automatic OpenAPI generation, type safety, and async support make it ideal for geospatial APIs that need to handle complex spatial operations efficiently.

### Spatial Data Models

```python
from sqlalchemy import Column, Integer, String, Float, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON, ST_Transform
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class GeospatialFeature(Base):
    __tablename__ = "geospatial_features"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    geometry = Column(Geometry('POLYGON', srid=4326), nullable=False)
    properties = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Spatial index
    __table_args__ = (
        Index('idx_geometry', geometry, postgresql_using='gist'),
    )

# Spatial query functions
def spatial_intersects_query(geom, buffer_distance=None):
    """Query features that intersect with geometry"""
    query = session.query(GeospatialFeature)
    
    if buffer_distance:
        # Add buffer to geometry
        buffered_geom = session.query(
            ST_Buffer(ST_Transform(geom, 3857), buffer_distance)
        ).scalar()
        query = query.filter(GeospatialFeature.geometry.ST_Intersects(buffered_geom))
    else:
        query = query.filter(GeospatialFeature.geometry.ST_Intersects(geom))
    
    return query
```

**Why**: Proper spatial data models with indexes enable efficient spatial queries and ensure data integrity.

## 2) Spatial Query Endpoints

### Intersection Queries

```python
@app.post("/spatial/intersects", response_model=SpatialResponse)
async def spatial_intersects(
    query: SpatialQuery,
    db: Session = Depends(get_db)
):
    """Find features that intersect with query geometry"""
    start_time = time.time()
    
    try:
        # Parse geometry from GeoJSON
        geom = shape(query.geometry)
        
        # Build spatial query
        spatial_query = spatial_intersects_query(geom, query.buffer_distance)
        
        # Apply limit
        if query.limit:
            spatial_query = spatial_query.limit(query.limit)
        
        # Execute query
        features = spatial_query.all()
        
        # Format response
        response_features = []
        for feature in features:
            feature_dict = {
                "id": feature.id,
                "name": feature.name,
                "geometry": json.loads(
                    session.query(ST_AsGeoJSON(feature.geometry)).scalar()
                ),
                "properties": feature.properties
            }
            response_features.append(feature_dict)
        
        query_time = (time.time() - start_time) * 1000
        
        return SpatialResponse(
            features=response_features,
            total_count=len(features),
            query_time_ms=query_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Why**: Proper spatial query implementation with error handling ensures reliable geospatial operations and good user experience.

### Distance and Buffer Operations

```python
@app.post("/spatial/within-distance")
async def spatial_within_distance(
    center: Dict[str, float] = Field(..., description="Center point {lat, lon}"),
    distance: float = Field(..., description="Distance in meters"),
    limit: int = Query(100, description="Maximum results")
):
    """Find features within distance of center point"""
    try:
        # Create center point
        center_point = Point(center['lon'], center['lat'])
        
        # Transform to projected coordinate system for distance calculations
        center_projected = session.query(
            ST_Transform(center_point, 3857)
        ).scalar()
        
        # Query features within distance
        features = session.query(GeospatialFeature).filter(
            GeospatialFeature.geometry.ST_DWithin(
                center_projected, distance
            )
        ).limit(limit).all()
        
        # Format response with actual distances
        response_features = []
        for feature in features:
            distance_m = session.query(
                ST_Distance(
                    ST_Transform(feature.geometry, 3857),
                    center_projected
                )
            ).scalar()
            
            response_features.append({
                "id": feature.id,
                "name": feature.name,
                "distance_meters": distance_m,
                "geometry": json.loads(
                    session.query(ST_AsGeoJSON(feature.geometry)).scalar()
                )
            })
        
        return {"features": response_features}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Why**: Distance-based queries are common in geospatial applications and require proper coordinate system handling for accurate results.

## 3) Performance Optimization

### Spatial Indexing

```python
# Ensure spatial indexes are created
def create_spatial_indexes():
    """Create spatial indexes for optimal performance"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_geometry_gist ON geospatial_features USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_geometry_brin ON geospatial_features USING BRIN (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_name_btree ON geospatial_features (name);"
    ]
    
    for index_sql in indexes:
        session.execute(index_sql)
    
    session.commit()

# Query optimization
def optimize_spatial_query(query, bbox=None):
    """Optimize spatial query with bounding box filter"""
    if bbox:
        # Add bounding box filter for initial filtering
        bbox_filter = f"""
        ST_Intersects(geometry, ST_MakeEnvelope(
            {bbox['minx']}, {bbox['miny']}, 
            {bbox['maxx']}, {bbox['maxy']}, 4326
        ))
        """
        query = query.filter(text(bbox_filter))
    
    return query
```

**Why**: Spatial indexes dramatically improve query performance, especially for large datasets with complex geometries.

### Caching Strategies

```python
from functools import lru_cache
import redis
import json

# Redis cache for spatial queries
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_spatial_query(cache_key, result, ttl=3600):
    """Cache spatial query results"""
    redis_client.setex(cache_key, ttl, json.dumps(result))

def get_cached_spatial_query(cache_key):
    """Retrieve cached spatial query results"""
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

@app.post("/spatial/cached-intersects")
async def cached_spatial_intersects(query: SpatialQuery):
    """Cached spatial intersection query"""
    # Generate cache key
    cache_key = f"spatial_intersects_{hash(str(query.dict()))}"
    
    # Check cache first
    cached_result = get_cached_spatial_query(cache_key)
    if cached_result:
        return cached_result
    
    # Execute query
    result = await spatial_intersects(query)
    
    # Cache result
    cache_spatial_query(cache_key, result.dict())
    
    return result
```

**Why**: Caching reduces database load and improves response times for frequently accessed spatial data.

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-intensive spatial operations
executor = ThreadPoolExecutor(max_workers=4)

async def async_spatial_processing(geometries):
    """Process multiple geometries asynchronously"""
    loop = asyncio.get_event_loop()
    
    # Process geometries in parallel
    tasks = []
    for geom in geometries:
        task = loop.run_in_executor(
            executor, 
            process_single_geometry, 
            geom
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    return results

def process_single_geometry(geom):
    """Process a single geometry (CPU-intensive)"""
    # Perform spatial operations
    area = geom.area
    centroid = geom.centroid
    bounds = geom.bounds
    
    return {
        'area': area,
        'centroid': [centroid.x, centroid.y],
        'bounds': bounds
    }
```

**Why**: Async processing enables handling multiple spatial operations concurrently, improving API throughput.

## 4) Data Validation and Security

### Input Validation

```python
from pydantic import validator
import shapely.geometry
import shapely.validation

class SpatialQuery(BaseModel):
    geometry: Dict[str, Any]
    buffer_distance: Optional[float] = None
    
    @validator('geometry')
    def validate_geometry(cls, v):
        """Validate geometry input"""
        try:
            geom = shape(v)
            
            # Check if geometry is valid
            if not geom.is_valid:
                raise ValueError("Invalid geometry provided")
            
            # Check geometry type
            if not isinstance(geom, (Point, Polygon, LineString)):
                raise ValueError("Unsupported geometry type")
            
            return v
            
        except Exception as e:
            raise ValueError(f"Geometry validation failed: {str(e)}")
    
    @validator('buffer_distance')
    def validate_buffer_distance(cls, v):
        """Validate buffer distance"""
        if v is not None:
            if v < 0:
                raise ValueError("Buffer distance must be positive")
            if v > 10000:  # 10km limit
                raise ValueError("Buffer distance too large")
        return v
```

**Why**: Input validation prevents invalid spatial data from causing API errors and ensures data quality.

### Rate Limiting and Security

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/spatial/intersects")
@limiter.limit("100/minute")
async def rate_limited_spatial_intersects(
    request: Request,
    query: SpatialQuery
):
    """Rate-limited spatial intersection query"""
    return await spatial_intersects(query)

# API key authentication
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Validate API key
    if not is_valid_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

@app.post("/spatial/intersects")
async def authenticated_spatial_intersects(
    query: SpatialQuery,
    api_key: str = Depends(verify_api_key)
):
    """Authenticated spatial intersection query"""
    return await spatial_intersects(query)
```

**Why**: Rate limiting and authentication protect the API from abuse and ensure proper access control.

## 5) Monitoring and Observability

### Custom Metrics

```python
import time
from prometheus_client import Counter, Histogram, generate_latest

# Prometheus metrics
spatial_queries_total = Counter('spatial_queries_total', 'Total spatial queries', ['endpoint'])
spatial_query_duration = Histogram('spatial_query_duration_seconds', 'Spatial query duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting metrics"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    spatial_query_duration.observe(duration)
    
    if request.url.path.startswith('/spatial/'):
        spatial_queries_total.labels(endpoint=request.url.path).inc()
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

**Why**: Comprehensive metrics enable monitoring API performance and identifying bottlenecks.

### Logging and Error Tracking

```python
import logging
import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

# Structured logging
logger = structlog.get_logger()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging"""
    logger.error(
        "API error",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Custom logging for spatial operations
def log_spatial_operation(operation, geometry_type, record_count, duration):
    """Log spatial operation details"""
    logger.info(
        "spatial_operation",
        operation=operation,
        geometry_type=geometry_type,
        record_count=record_count,
        duration_ms=duration * 1000
    )
```

**Why**: Comprehensive logging enables debugging, performance monitoring, and audit trails for geospatial operations.

## 6) API Documentation

### OpenAPI Schema

```python
# Enhanced OpenAPI schema
app = FastAPI(
    title="Geospatial API",
    description="""
    High-performance geospatial data API with spatial query capabilities.
    
    ## Features
    - Spatial intersection queries
    - Distance-based searches
    - Buffer operations
    - Real-time spatial analytics
    
    ## Authentication
    Include your API key in the `X-API-Key` header.
    """,
    version="1.0.0",
    contact={
        "name": "Geospatial API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Custom response models
class SpatialFeature(BaseModel):
    id: int
    name: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Sample Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"area": 1.0, "type": "building"}
            }
        }
```

**Why**: Comprehensive API documentation improves developer experience and reduces integration time.

## 7) TL;DR Quickstart

```python
# 1. Install dependencies
# pip install fastapi uvicorn geoalchemy2 shapely

# 2. Create FastAPI app
app = FastAPI(title="Geospatial API")

# 3. Define spatial endpoints
@app.post("/spatial/intersects")
async def spatial_intersects(query: SpatialQuery):
    # Implement spatial query logic
    pass

# 4. Add authentication and rate limiting
@app.post("/spatial/intersects")
@limiter.limit("100/minute")
async def rate_limited_intersects(request: Request, query: SpatialQuery):
    pass

# 5. Run with uvicorn
# uvicorn main:app --reload
```

## 8) Anti-Patterns to Avoid

- **Don't ignore spatial indexing**—spatial queries without indexes are slow
- **Don't skip input validation**—invalid geometries cause API errors
- **Don't ignore rate limiting**—spatial operations are resource-intensive
- **Don't skip error handling**—spatial operations can fail in unexpected ways
- **Don't ignore monitoring**—spatial APIs need comprehensive observability

**Why**: These anti-patterns lead to poor performance, security issues, and unreliable geospatial APIs.

---

*This guide provides the foundation for building production-ready geospatial APIs that scale with FastAPI.*
