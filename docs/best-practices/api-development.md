# API Development Best Practices

This document establishes production-ready API development patterns for geospatial systems, covering FastAPI implementation, authentication, rate limiting, and comprehensive error handling.

## FastAPI Geospatial API

### Production-Ready API

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

**Why:** FastAPI provides automatic OpenAPI documentation, type validation, and async support. Middleware ensures security and CORS handling for production deployments.

## Authentication and Authorization

### JWT Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

# Security configuration
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class User:
    def __init__(self, username: str, email: str, hashed_password: str):
        self.username = username
        self.email = email
        self.hashed_password = hashed_password

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # In production, fetch user from database
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

# Protected endpoint example
@app.get("/spatial/protected")
async def protected_spatial_query(
    current_user: User = Depends(get_current_user)
):
    """Protected spatial query endpoint"""
    return {"message": f"Hello {current_user.username}, this is a protected endpoint"}
```

**Why:** JWT authentication provides stateless, scalable authentication. Password hashing with bcrypt ensures secure credential storage.

### Role-Based Access Control

```python
from enum import Enum
from typing import List

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    READ_SPATIAL = "read:spatial"
    WRITE_SPATIAL = "write:spatial"
    DELETE_SPATIAL = "delete:spatial"
    ADMIN_ACCESS = "admin:access"

# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [Permission.READ_SPATIAL, Permission.WRITE_SPATIAL, Permission.DELETE_SPATIAL, Permission.ADMIN_ACCESS],
    UserRole.ANALYST: [Permission.READ_SPATIAL, Permission.WRITE_SPATIAL],
    UserRole.VIEWER: [Permission.READ_SPATIAL]
}

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
            if permission not in user_permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage example
@app.post("/spatial/data")
@require_permission(Permission.WRITE_SPATIAL)
async def create_spatial_data(
    data: SpatialQuery,
    current_user: User = Depends(get_current_user)
):
    """Create spatial data - requires write permission"""
    # Implementation
    pass
```

**Why:** Role-based access control provides fine-grained permission management. Decorators enable clean, reusable authorization logic.

## Rate Limiting and Throttling

### Redis-based Rate Limiting

```python
import redis
import time
from fastapi import Request, HTTPException
from typing import Optional

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """
        Check if request is allowed based on rate limit
        """
        current_time = int(time.time())
        window_start = current_time - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = self.redis.zcard(key)
        
        if current_requests >= limit:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, window)
        
        return True

# Rate limiting dependency
async def rate_limit_dependency(request: Request):
    """Rate limiting dependency"""
    client_ip = request.client.host
    rate_limiter = RateLimiter(redis_client)
    
    # Different limits for different endpoints
    if request.url.path.startswith("/spatial/query"):
        limit = 100  # 100 requests
        window = 3600  # per hour
    else:
        limit = 1000  # 1000 requests
        window = 3600  # per hour
    
    key = f"rate_limit:{client_ip}:{request.url.path}"
    
    if not await rate_limiter.is_allowed(key, limit, window):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

# Apply rate limiting to endpoints
@app.post("/spatial/query")
async def spatial_query_with_rate_limit(
    query: SpatialQuery,
    request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """Spatial query with rate limiting"""
    # Implementation
    pass
```

**Why:** Rate limiting prevents API abuse and ensures fair resource usage. Redis provides distributed rate limiting across multiple API instances.

## API Documentation and Testing

### Comprehensive API Testing

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

@pytest.fixture
def client():
    """Test client fixture"""
    from main import app
    return TestClient(app)

@pytest.fixture
def sample_spatial_query():
    """Sample spatial query fixture"""
    return {
        "geometry": {
            "type": "Point",
            "coordinates": [0, 0]
        },
        "buffer_distance": 1000,
        "spatial_operation": "intersects"
    }

class TestSpatialAPI:
    """Test suite for spatial API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_spatial_query_valid(self, client, sample_spatial_query):
        """Test valid spatial query"""
        with patch('main.perform_spatial_query') as mock_query:
            mock_query.return_value = [
                {"id": 1, "name": "Test Feature", "geometry": "POINT(0 0)"}
            ]
            
            response = client.post("/spatial/query", json=sample_spatial_query)
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "total_count" in data
            assert "processing_time" in data
    
    def test_spatial_query_invalid_geometry(self, client):
        """Test spatial query with invalid geometry"""
        invalid_query = {
            "geometry": {
                "type": "InvalidType",
                "coordinates": [0, 0]
            }
        }
        
        response = client.post("/spatial/query", json=invalid_query)
        
        assert response.status_code == 400
        assert "Invalid geometry" in response.json()["detail"]
    
    def test_spatial_query_rate_limit(self, client, sample_spatial_query):
        """Test rate limiting"""
        # Make multiple requests to trigger rate limit
        for _ in range(101):  # Exceed rate limit
            response = client.post("/spatial/query", json=sample_spatial_query)
            
            if response.status_code == 429:
                assert "Rate limit exceeded" in response.json()["detail"]
                break
    
    def test_authentication_required(self, client):
        """Test that protected endpoints require authentication"""
        response = client.get("/spatial/protected")
        
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]
    
    def test_authorization_permissions(self, client):
        """Test role-based authorization"""
        # Test with different user roles
        admin_token = create_test_token({"sub": "admin", "role": "admin"})
        viewer_token = create_test_token({"sub": "viewer", "role": "viewer"})
        
        # Admin should have access
        response = client.get(
            "/spatial/protected",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        
        # Viewer should be denied for write operations
        response = client.post(
            "/spatial/data",
            json={"test": "data"},
            headers={"Authorization": f"Bearer {viewer_token}"}
        )
        assert response.status_code == 403

def create_test_token(payload: dict) -> str:
    """Create test JWT token"""
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
```

**Why:** Comprehensive testing ensures API reliability and security. Mocking external dependencies enables isolated testing of API logic.

## API Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
import redis
import json
from typing import Optional

class APICache:
    """API caching implementation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[dict]:
        """Get cached data"""
        cached_data = self.redis.get(key)
        if cached_data:
            return json.loads(cached_data)
        return None
    
    async def set(self, key: str, data: dict, ttl: Optional[int] = None) -> None:
        """Set cached data"""
        ttl = ttl or self.default_ttl
        self.redis.setex(key, ttl, json.dumps(data))
    
    def generate_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return f"cache:{endpoint}:{hash(param_str)}"

# Caching decorator
def cache_response(ttl: int = 3600):
    """Decorator to cache API responses"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = APICache(redis_client)
            
            # Generate cache key
            cache_key = cache.generate_cache_key(func.__name__, kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage example
@app.get("/spatial/bounds")
@cache_response(ttl=1800)  # Cache for 30 minutes
async def get_spatial_bounds_cached(db=Depends(get_database)):
    """Get spatial bounds with caching"""
    bounds = await get_data_bounds(db)
    return {"bounds": bounds}
```

**Why:** Caching reduces database load and improves response times. Redis provides distributed caching across multiple API instances.

### Database Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

# Database configuration
DATABASE_URL = "postgresql://user:password@localhost/geospatial"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@asynccontextmanager
async def get_db_session():
    """Database session context manager"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Database dependency
async def get_database():
    """Database connection dependency with pooling"""
    async with get_db_session() as session:
        yield session
```

**Why:** Connection pooling improves database performance and resource utilization. Pre-ping ensures connection health, while pool recycling prevents stale connections.
