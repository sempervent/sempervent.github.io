# Python Web Services Best Practices

**Objective**: Master senior-level Python web service patterns for production systems. When you need to build scalable, reliable web services, when you want to implement modern web frameworks, when you need enterprise-grade web service architectures—these best practices become your weapon of choice.

## Core Principles

- **Framework Selection**: Choose the right framework for your use case
- **Scalability**: Design for horizontal and vertical scaling
- **Reliability**: Implement robust error handling and monitoring
- **Performance**: Optimize for speed and throughput
- **Security**: Implement comprehensive security measures

## FastAPI Web Services

### Modern FastAPI Setup

```python
# python/01-fastapi-setup.py

"""
Modern FastAPI web service setup and configuration
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebServiceConfig:
    """Web service configuration"""
    
    def __init__(self):
        self.title = "Modern Python Web Service"
        self.description = "A production-ready FastAPI web service"
        self.version = "1.0.0"
        self.debug = False
        self.host = "0.0.0.0"
        self.port = 8000

# Pydantic models
class UserCreate(BaseModel):
    """User creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=120)

class UserResponse(BaseModel):
    """User response model"""
    id: int
    name: str
    email: str
    age: Optional[int]
    created_at: str

class UserUpdate(BaseModel):
    """User update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = Field(None, regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=120)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    status_code: int

# Database simulation
class Database:
    """Simple in-memory database simulation"""
    
    def __init__(self):
        self.users = {}
        self.next_id = 1
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        user_id = self.next_id
        self.next_id += 1
        
        user = UserResponse(
            id=user_id,
            name=user_data.name,
            email=user_data.email,
            age=user_data.age,
            created_at="2024-01-01T00:00:00Z"
        )
        
        self.users[user_id] = user
        return user
    
    async def get_user(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def get_users(self, skip: int = 0, limit: int = 100) -> List[UserResponse]:
        """Get all users with pagination"""
        users = list(self.users.values())
        return users[skip:skip + limit]
    
    async def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update user"""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        update_data = user_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        return user
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user"""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False

# Global database instance
db = Database()

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    # In a real application, you would validate the JWT token here
    if credentials.credentials != "valid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": 1, "username": "admin"}

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting web service...")
    yield
    # Shutdown
    logger.info("Shutting down web service...")

# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    config = WebServiceConfig()
    
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        debug=config.debug,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add routes
    setup_routes(app)
    
    return app

def setup_routes(app: FastAPI):
    """Setup application routes"""
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {"message": "Welcome to the Python Web Service API"}
    
    @app.get("/health", response_model=Dict[str, str])
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}
    
    @app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(user: UserCreate, current_user: dict = Depends(get_current_user)):
        """Create a new user"""
        try:
            new_user = await db.create_user(user)
            logger.info(f"Created user: {new_user.id}")
            return new_user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    @app.get("/users", response_model=List[UserResponse])
    async def get_users(
        skip: int = 0,
        limit: int = 100,
        current_user: dict = Depends(get_current_user)
    ):
        """Get all users with pagination"""
        try:
            users = await db.get_users(skip=skip, limit=limit)
            return users
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get users"
            )
    
    @app.get("/users/{user_id}", response_model=UserResponse)
    async def get_user(user_id: int, current_user: dict = Depends(get_current_user)):
        """Get user by ID"""
        try:
            user = await db.get_user(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            return user
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user"
            )
    
    @app.put("/users/{user_id}", response_model=UserResponse)
    async def update_user(
        user_id: int,
        user_data: UserUpdate,
        current_user: dict = Depends(get_current_user)
    ):
        """Update user"""
        try:
            updated_user = await db.update_user(user_id, user_data)
            if not updated_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            return updated_user
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
    
    @app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
        """Delete user"""
        try:
            success = await db.delete_user(user_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )

# Create application instance
app = create_app()

# Run application
if __name__ == "__main__":
    config = WebServiceConfig()
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
```

### Advanced FastAPI Patterns

```python
# python/02-advanced-fastapi.py

"""
Advanced FastAPI patterns and middleware
"""

from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import uuid
from typing import Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(f"Request {request_id}: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response {request_id}: {response.status_code} in {process_time:.3f}s")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > current_time - self.period for t in times)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            self.clients[client_ip] = [t for t in self.clients[client_ip] if t > current_time - self.period]
            if len(self.clients[client_ip]) >= self.calls:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
        
        # Add current request
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        self.clients[client_ip].append(current_time)
        
        return await call_next(request)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )

def create_advanced_app() -> FastAPI:
    """Create FastAPI app with advanced middleware"""
    app = FastAPI(
        title="Advanced Python Web Service",
        description="Production-ready FastAPI with advanced patterns",
        version="1.0.0"
    )
    
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitingMiddleware, calls=100, period=60)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Add exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "details": exc.errors()}
        )
    
    return app

# Dependency injection patterns
class DatabaseService:
    """Database service for dependency injection"""
    
    def __init__(self):
        self.connection_string = "sqlite:///app.db"
    
    async def get_connection(self):
        """Get database connection"""
        # In a real app, you would return an actual database connection
        return {"connection": "active"}

async def get_database_service() -> DatabaseService:
    """Get database service dependency"""
    return DatabaseService()

async def get_current_user_id(request: Request) -> int:
    """Get current user ID from request"""
    # In a real app, you would extract this from JWT or session
    return 1

# Advanced route patterns
def setup_advanced_routes(app: FastAPI):
    """Setup advanced route patterns"""
    
    @app.get("/protected")
    async def protected_route(
        user_id: int = Depends(get_current_user_id),
        db_service: DatabaseService = Depends(get_database_service)
    ):
        """Protected route with dependencies"""
        connection = await db_service.get_connection()
        return {
            "message": "This is a protected route",
            "user_id": user_id,
            "database": connection
        }
    
    @app.get("/async-data")
    async def async_data_route():
        """Async data processing route"""
        import asyncio
        
        async def fetch_data(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Data fetched after {delay}s"
        
        # Fetch data concurrently
        results = await asyncio.gather(
            fetch_data(0.1),
            fetch_data(0.2),
            fetch_data(0.3)
        )
        
        return {"results": results}
```

## Flask Web Services

### Modern Flask Setup

```python
# python/03-flask-setup.py

"""
Modern Flask web service setup and configuration
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlaskWebService:
    """Modern Flask web service"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key'
        self.app.config['JSON_SORT_KEYS'] = False
        
        # Setup extensions
        self.setup_extensions()
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Setup error handlers
        self.setup_error_handlers()
    
    def setup_extensions(self):
        """Setup Flask extensions"""
        # CORS
        CORS(self.app, origins=["*"])
        
        # Rate limiting
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
    
    def setup_middleware(self):
        """Setup middleware"""
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = str(uuid.uuid4())
            logger.info(f"Request {g.request_id}: {request.method} {request.url}")
        
        @self.app.after_request
        def after_request(response):
            if hasattr(g, 'start_time'):
                process_time = time.time() - g.start_time
                response.headers['X-Process-Time'] = str(process_time)
                response.headers['X-Request-ID'] = g.request_id
                logger.info(f"Response {g.request_id}: {response.status_code} in {process_time:.3f}s")
            return response
    
    def setup_routes(self):
        """Setup application routes"""
        
        @self.app.route('/', methods=['GET'])
        def root():
            """Root endpoint"""
            return jsonify({"message": "Welcome to Flask Web Service"})
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({"status": "healthy"})
        
        @self.app.route('/users', methods=['GET'])
        @self.limiter.limit("10 per minute")
        def get_users():
            """Get all users"""
            try:
                # Simulate database query
                users = [
                    {"id": 1, "name": "Alice", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com"}
                ]
                return jsonify({"users": users})
            except Exception as e:
                logger.error(f"Error getting users: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/users/<int:user_id>', methods=['GET'])
        def get_user(user_id):
            """Get user by ID"""
            try:
                # Simulate database query
                if user_id == 1:
                    user = {"id": 1, "name": "Alice", "email": "alice@example.com"}
                    return jsonify(user)
                else:
                    return jsonify({"error": "User not found"}), 404
            except Exception as e:
                logger.error(f"Error getting user {user_id}: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/users', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def create_user():
            """Create new user"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                # Validate required fields
                required_fields = ['name', 'email']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Missing required field: {field}"}), 400
                
                # Simulate user creation
                user = {
                    "id": 3,
                    "name": data['name'],
                    "email": data['email']
                }
                
                return jsonify(user), 201
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                return jsonify({"error": "Internal server error"}), 500
    
    def setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(HTTPException)
        def handle_http_exception(e):
            """Handle HTTP exceptions"""
            return jsonify({"error": e.description}), e.code
        
        @self.app.errorhandler(404)
        def handle_not_found(e):
            """Handle 404 errors"""
            return jsonify({"error": "Not found"}), 404
        
        @self.app.errorhandler(500)
        def handle_internal_error(e):
            """Handle 500 errors"""
            return jsonify({"error": "Internal server error"}), 500

# Create Flask application
flask_app = FlaskWebService().app

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=5000, debug=True)
```

## Django Web Services

### Modern Django Setup

```python
# python/04-django-setup.py

"""
Modern Django web service setup and configuration
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.conf import settings
from django.core.exceptions import ValidationError
import json
import logging

logger = logging.getLogger(__name__)

class DjangoWebService:
    """Modern Django web service"""
    
    def __init__(self):
        self.setup_settings()
        self.setup_middleware()
        self.setup_urls()
    
    def setup_settings(self):
        """Setup Django settings"""
        settings.configure(
            DEBUG=True,
            SECRET_KEY='your-secret-key',
            ROOT_URLCONF=__name__,
            MIDDLEWARE=[
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
            ],
            ALLOWED_HOSTS=['*'],
        )
    
    def setup_middleware(self):
        """Setup custom middleware"""
        pass
    
    def setup_urls(self):
        """Setup URL patterns"""
        from django.urls import path, include
        
        urlpatterns = [
            path('', self.root_view),
            path('health/', self.health_view),
            path('users/', self.users_view),
            path('users/<int:user_id>/', self.user_detail_view),
        ]
        
        return urlpatterns
    
    def root_view(self, request):
        """Root view"""
        return JsonResponse({"message": "Welcome to Django Web Service"})
    
    def health_view(self, request):
        """Health check view"""
        return JsonResponse({"status": "healthy"})
    
    @csrf_exempt
    @require_http_methods(["GET", "POST"])
    def users_view(self, request):
        """Users view"""
        if request.method == 'GET':
            return self.get_users(request)
        elif request.method == 'POST':
            return self.create_user(request)
    
    def get_users(self, request):
        """Get all users"""
        try:
            # Simulate database query
            users = [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ]
            return JsonResponse({"users": users})
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)
    
    def create_user(self, request):
        """Create new user"""
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            required_fields = ['name', 'email']
            for field in required_fields:
                if field not in data:
                    return JsonResponse({"error": f"Missing required field: {field}"}, status=400)
            
            # Simulate user creation
            user = {
                "id": 3,
                "name": data['name'],
                "email": data['email']
            }
            
            return JsonResponse(user, status=201)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)
    
    @require_http_methods(["GET"])
    def user_detail_view(self, request, user_id):
        """User detail view"""
        try:
            # Simulate database query
            if user_id == 1:
                user = {"id": 1, "name": "Alice", "email": "alice@example.com"}
                return JsonResponse(user)
            else:
                return JsonResponse({"error": "User not found"}, status=404)
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)

# Create Django application
django_app = DjangoWebService()

if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line()
```

## TL;DR Runbook

### Quick Start

```python
# 1. FastAPI Web Service
from python.fastapi_setup import create_app
app = create_app()

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)

# 2. Flask Web Service
from python.flask_setup import FlaskWebService
flask_app = FlaskWebService().app
flask_app.run(host='0.0.0.0', port=5000)

# 3. Django Web Service
from python.django_setup import DjangoWebService
django_app = DjangoWebService()
```

### Essential Patterns

```python
# Complete web service setup
def create_production_web_service(framework: str = "fastapi"):
    """Create production-ready web service"""
    
    if framework == "fastapi":
        from python.fastapi_setup import create_advanced_app
        app = create_advanced_app()
    elif framework == "flask":
        from python.flask_setup import FlaskWebService
        app = FlaskWebService().app
    elif framework == "django":
        from python.django_setup import DjangoWebService
        app = DjangoWebService()
    
    return app
```

---

*This guide provides the complete machinery for Python web services. Each pattern includes implementation examples, framework-specific strategies, and real-world usage patterns for enterprise web service development.*
