# Python API Design Best Practices

**Objective**: Master senior-level Python API design patterns for production systems. When you need to build robust, scalable APIs, when you want to implement comprehensive API versioning, when you need enterprise-grade API design strategies—these best practices become your weapon of choice.

## Core Principles

- **RESTful Design**: Follow REST principles for consistent API design
- **Versioning**: Implement proper API versioning strategies
- **Documentation**: Provide comprehensive API documentation
- **Error Handling**: Implement consistent error responses
- **Security**: Build secure APIs with proper authentication and authorization

## RESTful API Design

### Resource-Based Design

```python
# python/01-restful-api-design.py

"""
RESTful API design patterns and resource-based architecture
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
from abc import ABC, abstractmethod

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

class HTTPStatus(Enum):
    """HTTP status codes"""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500

@dataclass
class APIResponse:
    """Standard API response"""
    status_code: int
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        response = {
            "status_code": self.status_code,
            "success": 200 <= self.status_code < 300
        }
        
        if self.data is not None:
            response["data"] = self.data
        
        if self.message:
            response["message"] = self.message
        
        if self.errors:
            response["errors"] = self.errors
        
        if self.metadata:
            response["metadata"] = self.metadata
        
        return response

@dataclass
class PaginationInfo:
    """Pagination information"""
    page: int
    per_page: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "page": self.page,
            "per_page": self.per_page,
            "total": self.total,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev
        }

class Resource(ABC):
    """Base resource class"""
    
    def __init__(self, resource_id: str, created_at: datetime = None, updated_at: datetime = None):
        self.id = resource_id
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Create resource from dictionary"""
        pass

@dataclass
class User(Resource):
    """User resource"""
    name: str
    email: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary"""
        return cls(
            resource_id=data["id"],
            name=data["name"],
            email=data["email"],
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

class APIResource:
    """API resource handler"""
    
    def __init__(self, resource_type: str):
        self.resource_type = resource_type
        self.resources: Dict[str, Resource] = {}
        self.next_id = 1
    
    def create(self, data: Dict[str, Any]) -> APIResponse:
        """Create resource"""
        try:
            # Validate required fields
            if not data.get("name"):
                return APIResponse(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    errors=["Name is required"]
                )
            
            # Create resource
            resource_id = str(self.next_id)
            self.next_id += 1
            
            if self.resource_type == "user":
                resource = User(
                    resource_id=resource_id,
                    name=data["name"],
                    email=data["email"],
                    is_active=data.get("is_active", True)
                )
            else:
                raise ValueError(f"Unknown resource type: {self.resource_type}")
            
            self.resources[resource_id] = resource
            
            return APIResponse(
                status_code=HTTPStatus.CREATED.value,
                data=resource.to_dict(),
                message=f"{self.resource_type.title()} created successfully"
            )
        
        except Exception as e:
            return APIResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                errors=[str(e)]
            )
    
    def get(self, resource_id: str) -> APIResponse:
        """Get resource by ID"""
        if resource_id not in self.resources:
            return APIResponse(
                status_code=HTTPStatus.NOT_FOUND.value,
                errors=[f"{self.resource_type.title()} not found"]
            )
        
        resource = self.resources[resource_id]
        return APIResponse(
            status_code=HTTPStatus.OK.value,
            data=resource.to_dict()
        )
    
    def list(self, page: int = 1, per_page: int = 10) -> APIResponse:
        """List resources with pagination"""
        total = len(self.resources)
        total_pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        
        resources = list(self.resources.values())[start:end]
        data = [resource.to_dict() for resource in resources]
        
        pagination = PaginationInfo(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
        return APIResponse(
            status_code=HTTPStatus.OK.value,
            data=data,
            metadata={"pagination": pagination.to_dict()}
        )
    
    def update(self, resource_id: str, data: Dict[str, Any]) -> APIResponse:
        """Update resource"""
        if resource_id not in self.resources:
            return APIResponse(
                status_code=HTTPStatus.NOT_FOUND.value,
                errors=[f"{self.resource_type.title()} not found"]
            )
        
        try:
            resource = self.resources[resource_id]
            
            # Update fields
            if "name" in data:
                resource.name = data["name"]
            if "email" in data:
                resource.email = data["email"]
            if "is_active" in data:
                resource.is_active = data["is_active"]
            
            resource.updated_at = datetime.utcnow()
            
            return APIResponse(
                status_code=HTTPStatus.OK.value,
                data=resource.to_dict(),
                message=f"{self.resource_type.title()} updated successfully"
            )
        
        except Exception as e:
            return APIResponse(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                errors=[str(e)]
            )
    
    def delete(self, resource_id: str) -> APIResponse:
        """Delete resource"""
        if resource_id not in self.resources:
            return APIResponse(
                status_code=HTTPStatus.NOT_FOUND.value,
                errors=[f"{self.resource_type.title()} not found"]
            )
        
        del self.resources[resource_id]
        
        return APIResponse(
            status_code=HTTPStatus.NO_CONTENT.value,
            message=f"{self.resource_type.title()} deleted successfully"
        )

# Usage examples
def example_restful_api():
    """Example RESTful API usage"""
    # Create API resource
    user_api = APIResource("user")
    
    # Create user
    create_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "is_active": True
    }
    create_response = user_api.create(create_data)
    print(f"Create response: {create_response.to_dict()}")
    
    # Get user
    user_id = "1"
    get_response = user_api.get(user_id)
    print(f"Get response: {get_response.to_dict()}")
    
    # List users
    list_response = user_api.list(page=1, per_page=10)
    print(f"List response: {list_response.to_dict()}")
    
    # Update user
    update_data = {"name": "Jane Doe"}
    update_response = user_api.update(user_id, update_data)
    print(f"Update response: {update_response.to_dict()}")
    
    # Delete user
    delete_response = user_api.delete(user_id)
    print(f"Delete response: {delete_response.to_dict()}")
```

### API Versioning

```python
# python/02-api-versioning.py

"""
API versioning strategies and implementation
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class APIVersion(Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

@dataclass
class VersionInfo:
    """API version information"""
    version: str
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    is_deprecated: bool = False
    is_sunset: bool = False
    sunset_date: Optional[datetime] = None

class APIVersionManager:
    """API version manager"""
    
    def __init__(self):
        self.versions: Dict[str, VersionInfo] = {}
        self.current_version = APIVersion.V1.value
        self.default_version = APIVersion.V1.value
    
    def register_version(self, version: str, release_date: datetime, 
                        deprecation_date: Optional[datetime] = None,
                        sunset_date: Optional[datetime] = None) -> None:
        """Register API version"""
        version_info = VersionInfo(
            version=version,
            release_date=release_date,
            deprecation_date=deprecation_date,
            is_deprecated=deprecation_date is not None,
            sunset_date=sunset_date,
            is_sunset=sunset_date is not None
        )
        self.versions[version] = version_info
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get version information"""
        return self.versions.get(version)
    
    def is_version_supported(self, version: str) -> bool:
        """Check if version is supported"""
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        return not version_info.is_sunset
    
    def get_supported_versions(self) -> List[str]:
        """Get list of supported versions"""
        return [
            version for version, info in self.versions.items()
            if not info.is_sunset
        ]
    
    def get_deprecated_versions(self) -> List[str]:
        """Get list of deprecated versions"""
        return [
            version for version, info in self.versions.items()
            if info.is_deprecated and not info.is_sunset
        ]
    
    def get_sunset_versions(self) -> List[str]:
        """Get list of sunset versions"""
        return [
            version for version, info in self.versions.items()
            if info.is_sunset
        ]

class VersionedAPI:
    """Versioned API implementation"""
    
    def __init__(self):
        self.version_manager = APIVersionManager()
        self.handlers: Dict[str, Dict[str, Any]] = {}
        self.setup_versions()
    
    def setup_versions(self) -> None:
        """Setup API versions"""
        # Register versions
        self.version_manager.register_version(
            "v1", 
            datetime(2023, 1, 1),
            deprecation_date=datetime(2024, 1, 1)
        )
        self.version_manager.register_version(
            "v2", 
            datetime(2023, 6, 1)
        )
        self.version_manager.register_version(
            "v3", 
            datetime(2024, 1, 1)
        )
    
    def register_handler(self, version: str, endpoint: str, handler: Any) -> None:
        """Register versioned handler"""
        if version not in self.handlers:
            self.handlers[version] = {}
        self.handlers[version][endpoint] = handler
    
    def get_handler(self, version: str, endpoint: str) -> Optional[Any]:
        """Get versioned handler"""
        if version not in self.handlers:
            return None
        return self.handlers[version].get(endpoint)
    
    def handle_request(self, version: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle versioned request"""
        # Check if version is supported
        if not self.version_manager.is_version_supported(version):
            return {
                "error": "Unsupported API version",
                "supported_versions": self.version_manager.get_supported_versions()
            }
        
        # Get handler
        handler = self.get_handler(version, endpoint)
        if not handler:
            return {
                "error": "Endpoint not found",
                "version": version,
                "endpoint": endpoint
            }
        
        # Execute handler
        try:
            result = handler(method, data)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "version": version,
                "endpoint": endpoint
            }
    
    def get_version_headers(self, version: str) -> Dict[str, str]:
        """Get version-specific headers"""
        version_info = self.version_manager.get_version_info(version)
        if not version_info:
            return {}
        
        headers = {
            "API-Version": version,
            "API-Release-Date": version_info.release_date.isoformat()
        }
        
        if version_info.is_deprecated:
            headers["API-Deprecation-Date"] = version_info.deprecation_date.isoformat()
            headers["Warning"] = f"API version {version} is deprecated"
        
        if version_info.is_sunset:
            headers["API-Sunset-Date"] = version_info.sunset_date.isoformat()
            headers["Warning"] = f"API version {version} is sunset"
        
        return headers

# Usage examples
def example_api_versioning():
    """Example API versioning usage"""
    # Create versioned API
    api = VersionedAPI()
    
    # Register handlers for different versions
    def v1_user_handler(method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"version": "v1", "method": method, "data": data}
    
    def v2_user_handler(method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"version": "v2", "method": method, "data": data, "enhanced": True}
    
    api.register_handler("v1", "users", v1_user_handler)
    api.register_handler("v2", "users", v2_user_handler)
    
    # Handle requests
    v1_response = api.handle_request("v1", "users", "GET", {})
    print(f"V1 response: {v1_response}")
    
    v2_response = api.handle_request("v2", "users", "GET", {})
    print(f"V2 response: {v2_response}")
    
    # Get version headers
    v1_headers = api.get_version_headers("v1")
    print(f"V1 headers: {v1_headers}")
    
    # Get supported versions
    supported = api.version_manager.get_supported_versions()
    print(f"Supported versions: {supported}")
```

### API Documentation

```python
# python/03-api-documentation.py

"""
API documentation generation and OpenAPI integration
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

@dataclass
class APIParameter:
    """API parameter definition"""
    name: str
    type: str
    required: bool = False
    description: Optional[str] = None
    example: Optional[Any] = None
    default: Optional[Any] = None

@dataclass
class APIResponse:
    """API response definition"""
    status_code: int
    description: str
    schema: Optional[Dict[str, Any]] = None
    example: Optional[Any] = None

@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: HTTPMethod
    summary: str
    description: str
    parameters: List[APIParameter]
    responses: List[APIResponse]
    tags: List[str]
    deprecated: bool = False

class APIDocumentation:
    """API documentation generator"""
    
    def __init__(self, title: str, version: str, description: str):
        self.title = title
        self.version = version
        self.description = description
        self.endpoints: List[APIEndpoint] = []
        self.tags: Dict[str, str] = {}
    
    def add_tag(self, name: str, description: str) -> None:
        """Add API tag"""
        self.tags[name] = description
    
    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add API endpoint"""
        self.endpoints.append(endpoint)
    
    def generate_openapi(self) -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description
            },
            "servers": [
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.example.com",
                    "description": "Staging server"
                }
            ],
            "tags": [
                {"name": name, "description": description}
                for name, description in self.tags.items()
            ],
            "paths": {}
        }
        
        # Group endpoints by path
        paths = {}
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            paths[endpoint.path][endpoint.method.value.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "parameters": [
                    {
                        "name": param.name,
                        "in": "query" if param.name in ["page", "per_page", "sort"] else "path",
                        "required": param.required,
                        "description": param.description,
                        "schema": {"type": param.type},
                        "example": param.example
                    }
                    for param in endpoint.parameters
                ],
                "responses": {
                    str(response.status_code): {
                        "description": response.description,
                        "content": {
                            "application/json": {
                                "schema": response.schema or {"type": "object"},
                                "example": response.example
                            }
                        }
                    }
                    for response in endpoint.responses
                }
            }
        
        openapi_spec["paths"] = paths
        return openapi_spec
    
    def generate_markdown(self) -> str:
        """Generate Markdown documentation"""
        md_lines = [
            f"# {self.title}",
            f"**Version:** {self.version}",
            f"**Description:** {self.description}",
            "",
            "## Endpoints",
            ""
        ]
        
        # Group endpoints by tag
        endpoints_by_tag = {}
        for endpoint in self.endpoints:
            for tag in endpoint.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        for tag, endpoints in endpoints_by_tag.items():
            md_lines.extend([
                f"### {tag.title()}",
                ""
            ])
            
            for endpoint in endpoints:
                md_lines.extend([
                    f"#### {endpoint.method.value} {endpoint.path}",
                    f"**Summary:** {endpoint.summary}",
                    f"**Description:** {endpoint.description}",
                    ""
                ])
                
                if endpoint.parameters:
                    md_lines.extend([
                        "**Parameters:**",
                        ""
                    ])
                    for param in endpoint.parameters:
                        md_lines.extend([
                            f"- **{param.name}** ({param.type}){' - Required' if param.required else ''}",
                            f"  - {param.description}",
                            ""
                        ])
                
                if endpoint.responses:
                    md_lines.extend([
                        "**Responses:**",
                        ""
                    ])
                    for response in endpoint.responses:
                        md_lines.extend([
                            f"- **{response.status_code}** - {response.description}",
                            ""
                        ])
                
                md_lines.append("---")
                md_lines.append("")
        
        return "\n".join(md_lines)

# Usage examples
def example_api_documentation():
    """Example API documentation usage"""
    # Create API documentation
    doc = APIDocumentation(
        title="User Management API",
        version="1.0.0",
        description="API for managing users"
    )
    
    # Add tags
    doc.add_tag("users", "User management operations")
    doc.add_tag("authentication", "Authentication operations")
    
    # Add endpoints
    get_users_endpoint = APIEndpoint(
        path="/users",
        method=HTTPMethod.GET,
        summary="List users",
        description="Get a list of all users",
        parameters=[
            APIParameter("page", "integer", False, "Page number", 1),
            APIParameter("per_page", "integer", False, "Items per page", 10)
        ],
        responses=[
            APIResponse(200, "Success", {"type": "array", "items": {"type": "object"}}),
            APIResponse(400, "Bad Request"),
            APIResponse(500, "Internal Server Error")
        ],
        tags=["users"]
    )
    
    create_user_endpoint = APIEndpoint(
        path="/users",
        method=HTTPMethod.POST,
        summary="Create user",
        description="Create a new user",
        parameters=[
            APIParameter("name", "string", True, "User name"),
            APIParameter("email", "string", True, "User email")
        ],
        responses=[
            APIResponse(201, "Created", {"type": "object"}),
            APIResponse(400, "Bad Request"),
            APIResponse(422, "Validation Error")
        ],
        tags=["users"]
    )
    
    doc.add_endpoint(get_users_endpoint)
    doc.add_endpoint(create_user_endpoint)
    
    # Generate OpenAPI spec
    openapi_spec = doc.generate_openapi()
    print("OpenAPI spec:")
    print(json.dumps(openapi_spec, indent=2))
    
    # Generate Markdown documentation
    markdown_doc = doc.generate_markdown()
    print("\nMarkdown documentation:")
    print(markdown_doc)
```

## Error Handling and Validation

### API Error Handling

```python
# python/04-api-error-handling.py

"""
API error handling and validation patterns
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class APIErrorCode(Enum):
    """API error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

@dataclass
class APIError:
    """API error definition"""
    code: APIErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    field: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        error_dict = {
            "code": self.code.value,
            "message": self.message
        }
        
        if self.details:
            error_dict["details"] = self.details
        
        if self.field:
            error_dict["field"] = self.field
        
        return error_dict

@dataclass
class APIErrorResponse:
    """API error response"""
    status_code: int
    errors: List[APIError]
    request_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        response = {
            "status_code": self.status_code,
            "success": False,
            "errors": [error.to_dict() for error in self.errors],
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.request_id:
            response["request_id"] = self.request_id
        
        return response

class APIValidator:
    """API request validator"""
    
    def __init__(self):
        self.validators: Dict[str, callable] = {}
        self.setup_validators()
    
    def setup_validators(self) -> None:
        """Setup validators"""
        self.validators = {
            "required": self._validate_required,
            "email": self._validate_email,
            "min_length": self._validate_min_length,
            "max_length": self._validate_max_length,
            "min_value": self._validate_min_value,
            "max_value": self._validate_max_value,
            "pattern": self._validate_pattern
        }
    
    def validate(self, data: Dict[str, Any], rules: Dict[str, Dict[str, Any]]) -> List[APIError]:
        """Validate data against rules"""
        errors = []
        
        for field, field_rules in rules.items():
            value = data.get(field)
            
            for rule_name, rule_value in field_rules.items():
                if rule_name in self.validators:
                    error = self.validators[rule_name](field, value, rule_value)
                    if error:
                        errors.append(error)
        
        return errors
    
    def _validate_required(self, field: str, value: Any, required: bool) -> Optional[APIError]:
        """Validate required field"""
        if required and (value is None or value == ""):
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' is required",
                field=field
            )
        return None
    
    def _validate_email(self, field: str, value: Any, is_email: bool) -> Optional[APIError]:
        """Validate email format"""
        if is_email and value and "@" not in str(value):
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' must be a valid email",
                field=field
            )
        return None
    
    def _validate_min_length(self, field: str, value: Any, min_length: int) -> Optional[APIError]:
        """Validate minimum length"""
        if value and len(str(value)) < min_length:
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' must be at least {min_length} characters",
                field=field
            )
        return None
    
    def _validate_max_length(self, field: str, value: Any, max_length: int) -> Optional[APIError]:
        """Validate maximum length"""
        if value and len(str(value)) > max_length:
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' must be at most {max_length} characters",
                field=field
            )
        return None
    
    def _validate_min_value(self, field: str, value: Any, min_value: Union[int, float]) -> Optional[APIError]:
        """Validate minimum value"""
        if value is not None and value < min_value:
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' must be at least {min_value}",
                field=field
            )
        return None
    
    def _validate_max_value(self, field: str, value: Any, max_value: Union[int, float]) -> Optional[APIError]:
        """Validate maximum value"""
        if value is not None and value > max_value:
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' must be at most {max_value}",
                field=field
            )
        return None
    
    def _validate_pattern(self, field: str, value: Any, pattern: str) -> Optional[APIError]:
        """Validate pattern"""
        if value and not re.match(pattern, str(value)):
            return APIError(
                code=APIErrorCode.VALIDATION_ERROR,
                message=f"Field '{field}' does not match required pattern",
                field=field
            )
        return None

class APIErrorHandler:
    """API error handler"""
    
    def __init__(self):
        self.validator = APIValidator()
    
    def handle_validation_error(self, errors: List[APIError]) -> APIErrorResponse:
        """Handle validation errors"""
        return APIErrorResponse(
            status_code=400,
            errors=errors
        )
    
    def handle_authentication_error(self, message: str = "Authentication required") -> APIErrorResponse:
        """Handle authentication errors"""
        return APIErrorResponse(
            status_code=401,
            errors=[APIError(
                code=APIErrorCode.AUTHENTICATION_ERROR,
                message=message
            )]
        )
    
    def handle_authorization_error(self, message: str = "Insufficient permissions") -> APIErrorResponse:
        """Handle authorization errors"""
        return APIErrorResponse(
            status_code=403,
            errors=[APIError(
                code=APIErrorCode.AUTHORIZATION_ERROR,
                message=message
            )]
        )
    
    def handle_not_found_error(self, resource: str = "Resource") -> APIErrorResponse:
        """Handle not found errors"""
        return APIErrorResponse(
            status_code=404,
            errors=[APIError(
                code=APIErrorCode.NOT_FOUND,
                message=f"{resource} not found"
            )]
        )
    
    def handle_conflict_error(self, message: str = "Resource conflict") -> APIErrorResponse:
        """Handle conflict errors"""
        return APIErrorResponse(
            status_code=409,
            errors=[APIError(
                code=APIErrorCode.CONFLICT,
                message=message
            )]
        )
    
    def handle_internal_error(self, message: str = "Internal server error") -> APIErrorResponse:
        """Handle internal errors"""
        return APIErrorResponse(
            status_code=500,
            errors=[APIError(
                code=APIErrorCode.INTERNAL_SERVER_ERROR,
                message=message
            )]
        )

# Usage examples
def example_api_error_handling():
    """Example API error handling usage"""
    # Create error handler
    error_handler = APIErrorHandler()
    
    # Create validator
    validator = APIValidator()
    
    # Validate data
    data = {"name": "", "email": "invalid-email"}
    rules = {
        "name": {"required": True, "min_length": 2},
        "email": {"required": True, "email": True}
    }
    
    errors = validator.validate(data, rules)
    if errors:
        error_response = error_handler.handle_validation_error(errors)
        print(f"Validation error: {error_response.to_dict()}")
    
    # Handle different error types
    auth_error = error_handler.handle_authentication_error()
    print(f"Auth error: {auth_error.to_dict()}")
    
    not_found_error = error_handler.handle_not_found_error("User")
    print(f"Not found error: {not_found_error.to_dict()}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Basic API response
response = APIResponse(
    status_code=200,
    data={"id": 1, "name": "John"},
    message="Success"
)

# 2. API versioning
version_manager = APIVersionManager()
version_manager.register_version("v1", datetime(2023, 1, 1))

# 3. API documentation
doc = APIDocumentation("My API", "1.0.0", "API description")
doc.add_endpoint(endpoint)

# 4. Error handling
error_handler = APIErrorHandler()
validation_error = error_handler.handle_validation_error(errors)

# 5. Request validation
validator = APIValidator()
errors = validator.validate(data, rules)
```

### Essential Patterns

```python
# Complete API design setup
def setup_api_design():
    """Setup complete API design environment"""
    
    # API resource
    api_resource = APIResource("users")
    
    # Version manager
    version_manager = APIVersionManager()
    
    # Documentation
    doc = APIDocumentation("User API", "1.0.0", "User management API")
    
    # Error handler
    error_handler = APIErrorHandler()
    
    # Validator
    validator = APIValidator()
    
    print("API design setup complete!")
```

---

*This guide provides the complete machinery for Python API design. Each pattern includes implementation examples, design strategies, and real-world usage patterns for enterprise API development.*
