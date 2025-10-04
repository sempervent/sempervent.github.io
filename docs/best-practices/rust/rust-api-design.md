# Rust API Design Best Practices

**Objective**: Master senior-level Rust API design patterns for production systems. When you need to build robust, scalable APIs, when you want to create maintainable interfaces, when you need enterprise-grade API design—these best practices become your weapon of choice.

## Core Principles

- **RESTful Design**: Follow REST principles for API design
- **Type Safety**: Leverage Rust's type system for API safety
- **Versioning**: Implement proper API versioning strategies
- **Documentation**: Comprehensive API documentation
- **Error Handling**: Consistent error responses across APIs

## RESTful API Patterns

### Resource Design

```rust
// rust/01-resource-design.rs

/*
RESTful resource design patterns and best practices
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// User resource representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// User creation request.
#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
}

/// User update request.
#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

/// User response wrapper.
#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub data: User,
}

/// Users collection response.
#[derive(Debug, Serialize)]
pub struct UsersResponse {
    pub data: Vec<User>,
    pub meta: PaginationMeta,
}

/// Pagination metadata.
#[derive(Debug, Serialize)]
pub struct PaginationMeta {
    pub total: u64,
    pub page: u64,
    pub per_page: u64,
    pub total_pages: u64,
}

/// Query parameters for listing users.
#[derive(Debug, Deserialize)]
pub struct ListUsersQuery {
    pub page: Option<u64>,
    pub per_page: Option<u64>,
    pub search: Option<String>,
    pub sort: Option<String>,
    pub order: Option<String>,
}

/// Error response structure.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail structure.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// API response wrapper.
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub data: T,
    pub meta: Option<ResponseMeta>,
}

/// Response metadata.
#[derive(Debug, Serialize)]
pub struct ResponseMeta {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub request_id: String,
}

/// User service for managing users.
pub struct UserService {
    users: HashMap<Uuid, User>,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }
    
    pub fn create_user(&mut self, req: CreateUserRequest) -> Result<User, String> {
        // Validate email format
        if !req.email.contains('@') {
            return Err("Invalid email format".to_string());
        }
        
        // Check for duplicate email
        if self.users.values().any(|u| u.email == req.email) {
            return Err("Email already exists".to_string());
        }
        
        let now = chrono::Utc::now();
        let user = User {
            id: Uuid::new_v4(),
            name: req.name,
            email: req.email,
            created_at: now,
            updated_at: now,
        };
        
        self.users.insert(user.id, user.clone());
        Ok(user)
    }
    
    pub fn get_user(&self, id: Uuid) -> Result<Option<User>, String> {
        Ok(self.users.get(&id).cloned())
    }
    
    pub fn list_users(&self, query: ListUsersQuery) -> Result<Vec<User>, String> {
        let mut users: Vec<User> = self.users.values().cloned().collect();
        
        // Apply search filter
        if let Some(search) = query.search {
            users.retain(|u| u.name.contains(&search) || u.email.contains(&search));
        }
        
        // Apply sorting
        if let Some(sort) = query.sort {
            match sort.as_str() {
                "name" => users.sort_by(|a, b| a.name.cmp(&b.name)),
                "email" => users.sort_by(|a, b| a.email.cmp(&b.email)),
                "created_at" => users.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
                _ => {}
            }
            
            if let Some(order) = query.order {
                if order == "desc" {
                    users.reverse();
                }
            }
        }
        
        // Apply pagination
        let page = query.page.unwrap_or(1);
        let per_page = query.per_page.unwrap_or(10);
        let start = (page - 1) * per_page;
        let end = start + per_page;
        
        if start < users.len() as u64 {
            users = users.into_iter().skip(start as usize).take(per_page as usize).collect();
        } else {
            users.clear();
        }
        
        Ok(users)
    }
    
    pub fn update_user(&mut self, id: Uuid, req: UpdateUserRequest) -> Result<Option<User>, String> {
        if let Some(user) = self.users.get_mut(&id) {
            if let Some(name) = req.name {
                user.name = name;
            }
            if let Some(email) = req.email {
                // Validate email format
                if !email.contains('@') {
                    return Err("Invalid email format".to_string());
                }
                
                // Check for duplicate email
                if self.users.values().any(|u| u.email == email && u.id != id) {
                    return Err("Email already exists".to_string());
                }
                
                user.email = email;
            }
            user.updated_at = chrono::Utc::now();
            Ok(Some(user.clone()))
        } else {
            Ok(None)
        }
    }
    
    pub fn delete_user(&mut self, id: Uuid) -> Result<bool, String> {
        Ok(self.users.remove(&id).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_user() {
        let mut service = UserService::new();
        let req = CreateUserRequest {
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
        };
        
        let user = service.create_user(req).unwrap();
        assert_eq!(user.name, "John Doe");
        assert_eq!(user.email, "john@example.com");
    }
    
    #[test]
    fn test_get_user() {
        let mut service = UserService::new();
        let req = CreateUserRequest {
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
        };
        
        let user = service.create_user(req).unwrap();
        let retrieved = service.get_user(user.id).unwrap().unwrap();
        assert_eq!(retrieved.name, "John Doe");
    }
    
    #[test]
    fn test_update_user() {
        let mut service = UserService::new();
        let req = CreateUserRequest {
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
        };
        
        let user = service.create_user(req).unwrap();
        let update_req = UpdateUserRequest {
            name: Some("Jane Doe".to_string()),
            email: None,
        };
        
        let updated = service.update_user(user.id, update_req).unwrap().unwrap();
        assert_eq!(updated.name, "Jane Doe");
    }
}
```

### API Versioning

```rust
// rust/02-api-versioning.rs

/*
API versioning patterns and best practices
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// API version enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApiVersion {
    V1,
    V2,
    V3,
}

impl ApiVersion {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "v1" | "1" => Some(ApiVersion::V1),
            "v2" | "2" => Some(ApiVersion::V2),
            "v3" | "3" => Some(ApiVersion::V3),
            _ => None,
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            ApiVersion::V1 => "v1".to_string(),
            ApiVersion::V2 => "v2".to_string(),
            ApiVersion::V3 => "v3".to_string(),
        }
    }
}

/// Versioned user structure for V1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserV1 {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Versioned user structure for V2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserV2 {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Versioned user structure for V3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserV3 {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Versioned API response.
#[derive(Debug, Serialize)]
pub struct VersionedResponse<T> {
    pub data: T,
    pub version: String,
    pub api_version: String,
}

/// Versioned API service.
pub struct VersionedApiService {
    users: HashMap<u64, UserV3>,
    next_id: u64,
}

impl VersionedApiService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn create_user(&mut self, name: String, email: String) -> Result<UserV3, String> {
        let now = chrono::Utc::now();
        let user = UserV3 {
            id: self.next_id,
            name,
            email,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        };
        
        self.next_id += 1;
        self.users.insert(user.id, user.clone());
        Ok(user)
    }
    
    pub fn get_user(&self, id: u64) -> Option<&UserV3> {
        self.users.get(&id)
    }
    
    pub fn get_user_v1(&self, id: u64) -> Option<UserV1> {
        self.users.get(&id).map(|u| UserV1 {
            id: u.id,
            name: u.name.clone(),
            email: u.email.clone(),
        })
    }
    
    pub fn get_user_v2(&self, id: u64) -> Option<UserV2> {
        self.users.get(&id).map(|u| UserV2 {
            id: u.id,
            name: u.name.clone(),
            email: u.email.clone(),
            created_at: u.created_at,
        })
    }
    
    pub fn get_user_v3(&self, id: u64) -> Option<&UserV3> {
        self.users.get(&id)
    }
    
    pub fn list_users(&self, version: ApiVersion) -> Vec<Box<dyn serde::Serialize>> {
        match version {
            ApiVersion::V1 => {
                let users: Vec<UserV1> = self.users.values()
                    .map(|u| UserV1 {
                        id: u.id,
                        name: u.name.clone(),
                        email: u.email.clone(),
                    })
                    .collect();
                users.into_iter().map(|u| Box::new(u) as Box<dyn serde::Serialize>).collect()
            }
            ApiVersion::V2 => {
                let users: Vec<UserV2> = self.users.values()
                    .map(|u| UserV2 {
                        id: u.id,
                        name: u.name.clone(),
                        email: u.email.clone(),
                        created_at: u.created_at,
                    })
                    .collect();
                users.into_iter().map(|u| Box::new(u) as Box<dyn serde::Serialize>).collect()
            }
            ApiVersion::V3 => {
                let users: Vec<&UserV3> = self.users.values().collect();
                users.into_iter().map(|u| Box::new(*u) as Box<dyn serde::Serialize>).collect()
            }
        }
    }
}

/// API version middleware.
pub struct ApiVersionMiddleware {
    default_version: ApiVersion,
    supported_versions: Vec<ApiVersion>,
}

impl ApiVersionMiddleware {
    pub fn new(default_version: ApiVersion) -> Self {
        Self {
            default_version,
            supported_versions: vec![ApiVersion::V1, ApiVersion::V2, ApiVersion::V3],
        }
    }
    
    pub fn extract_version(&self, path: &str, header: Option<&str>) -> ApiVersion {
        // Try to extract version from path
        if let Some(version_str) = path.split('/').nth(1) {
            if let Some(version) = ApiVersion::from_str(version_str) {
                if self.supported_versions.contains(&version) {
                    return version;
                }
            }
        }
        
        // Try to extract version from header
        if let Some(header) = header {
            if let Some(version_str) = header.split('=').nth(1) {
                if let Some(version) = ApiVersion::from_str(version_str) {
                    if self.supported_versions.contains(&version) {
                        return version;
                    }
                }
            }
        }
        
        // Return default version
        self.default_version.clone()
    }
    
    pub fn is_version_supported(&self, version: &ApiVersion) -> bool {
        self.supported_versions.contains(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_api_version_parsing() {
        assert_eq!(ApiVersion::from_str("v1"), Some(ApiVersion::V1));
        assert_eq!(ApiVersion::from_str("v2"), Some(ApiVersion::V2));
        assert_eq!(ApiVersion::from_str("v3"), Some(ApiVersion::V3));
        assert_eq!(ApiVersion::from_str("invalid"), None);
    }
    
    #[test]
    fn test_versioned_api_service() {
        let mut service = VersionedApiService::new();
        let user = service.create_user("John".to_string(), "john@example.com".to_string()).unwrap();
        
        assert_eq!(user.name, "John");
        assert_eq!(user.email, "john@example.com");
        
        let user_v1 = service.get_user_v1(user.id).unwrap();
        assert_eq!(user_v1.name, "John");
        
        let user_v2 = service.get_user_v2(user.id).unwrap();
        assert_eq!(user_v2.name, "John");
        
        let user_v3 = service.get_user_v3(user.id).unwrap();
        assert_eq!(user_v3.name, "John");
    }
    
    #[test]
    fn test_api_version_middleware() {
        let middleware = ApiVersionMiddleware::new(ApiVersion::V1);
        
        let version = middleware.extract_version("/api/v2/users", None);
        assert_eq!(version, ApiVersion::V2);
        
        let version = middleware.extract_version("/api/invalid/users", None);
        assert_eq!(version, ApiVersion::V1);
        
        assert!(middleware.is_version_supported(&ApiVersion::V1));
        assert!(middleware.is_version_supported(&ApiVersion::V2));
        assert!(middleware.is_version_supported(&ApiVersion::V3));
    }
}
```

### Error Handling

```rust
// rust/03-error-handling.rs

/*
API error handling patterns and best practices
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// API error enumeration.
#[derive(Error, Debug, Serialize)]
pub enum ApiError {
    #[error("Bad Request: {0}")]
    BadRequest(String),
    
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    
    #[error("Forbidden: {0}")]
    Forbidden(String),
    
    #[error("Not Found: {0}")]
    NotFound(String),
    
    #[error("Conflict: {0}")]
    Conflict(String),
    
    #[error("Unprocessable Entity: {0}")]
    UnprocessableEntity(String),
    
    #[error("Internal Server Error: {0}")]
    InternalServerError(String),
    
    #[error("Service Unavailable: {0}")]
    ServiceUnavailable(String),
}

impl ApiError {
    pub fn status_code(&self) -> u16 {
        match self {
            ApiError::BadRequest(_) => 400,
            ApiError::Unauthorized(_) => 401,
            ApiError::Forbidden(_) => 403,
            ApiError::NotFound(_) => 404,
            ApiError::Conflict(_) => 409,
            ApiError::UnprocessableEntity(_) => 422,
            ApiError::InternalServerError(_) => 500,
            ApiError::ServiceUnavailable(_) => 503,
        }
    }
    
    pub fn error_code(&self) -> String {
        match self {
            ApiError::BadRequest(_) => "BAD_REQUEST".to_string(),
            ApiError::Unauthorized(_) => "UNAUTHORIZED".to_string(),
            ApiError::Forbidden(_) => "FORBIDDEN".to_string(),
            ApiError::NotFound(_) => "NOT_FOUND".to_string(),
            ApiError::Conflict(_) => "CONFLICT".to_string(),
            ApiError::UnprocessableEntity(_) => "UNPROCESSABLE_ENTITY".to_string(),
            ApiError::InternalServerError(_) => "INTERNAL_SERVER_ERROR".to_string(),
            ApiError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE".to_string(),
        }
    }
}

/// Error response structure.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
    pub meta: ErrorMeta,
}

/// Error detail structure.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// Error metadata.
#[derive(Debug, Serialize)]
pub struct ErrorMeta {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub request_id: String,
    pub version: String,
}

impl From<ApiError> for ErrorResponse {
    fn from(error: ApiError) -> Self {
        Self {
            error: ErrorDetail {
                code: error.error_code(),
                message: error.to_string(),
                details: None,
            },
            meta: ErrorMeta {
                timestamp: chrono::Utc::now(),
                request_id: uuid::Uuid::new_v4().to_string(),
                version: "1.0.0".to_string(),
            },
        }
    }
}

/// Validation error structure.
#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub value: Option<serde_json::Value>,
}

/// Validation error response.
#[derive(Debug, Serialize)]
pub struct ValidationErrorResponse {
    pub error: ErrorDetail,
    pub validation_errors: Vec<ValidationError>,
    pub meta: ErrorMeta,
}

impl From<Vec<ValidationError>> for ValidationErrorResponse {
    fn from(errors: Vec<ValidationError>) -> Self {
        Self {
            error: ErrorDetail {
                code: "VALIDATION_ERROR".to_string(),
                message: "Validation failed".to_string(),
                details: None,
            },
            validation_errors: errors,
            meta: ErrorMeta {
                timestamp: chrono::Utc::now(),
                request_id: uuid::Uuid::new_v4().to_string(),
                version: "1.0.0".to_string(),
            },
        }
    }
}

/// API result type.
pub type ApiResult<T> = Result<T, ApiError>;

/// Error handler trait.
pub trait ErrorHandler {
    fn handle_error(&self, error: &ApiError) -> ErrorResponse;
    fn handle_validation_error(&self, errors: Vec<ValidationError>) -> ValidationErrorResponse;
}

/// Default error handler.
pub struct DefaultErrorHandler;

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, error: &ApiError) -> ErrorResponse {
        ErrorResponse::from(error.clone())
    }
    
    fn handle_validation_error(&self, errors: Vec<ValidationError>) -> ValidationErrorResponse {
        ValidationErrorResponse::from(errors)
    }
}

/// API error middleware.
pub struct ErrorMiddleware {
    handler: Box<dyn ErrorHandler>,
}

impl ErrorMiddleware {
    pub fn new(handler: Box<dyn ErrorHandler>) -> Self {
        Self { handler }
    }
    
    pub fn handle_error(&self, error: &ApiError) -> ErrorResponse {
        self.handler.handle_error(error)
    }
    
    pub fn handle_validation_error(&self, errors: Vec<ValidationError>) -> ValidationErrorResponse {
        self.handler.handle_validation_error(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_api_error_status_codes() {
        assert_eq!(ApiError::BadRequest("test".to_string()).status_code(), 400);
        assert_eq!(ApiError::Unauthorized("test".to_string()).status_code(), 401);
        assert_eq!(ApiError::Forbidden("test".to_string()).status_code(), 403);
        assert_eq!(ApiError::NotFound("test".to_string()).status_code(), 404);
        assert_eq!(ApiError::Conflict("test".to_string()).status_code(), 409);
        assert_eq!(ApiError::UnprocessableEntity("test".to_string()).status_code(), 422);
        assert_eq!(ApiError::InternalServerError("test".to_string()).status_code(), 500);
        assert_eq!(ApiError::ServiceUnavailable("test".to_string()).status_code(), 503);
    }
    
    #[test]
    fn test_error_response_conversion() {
        let error = ApiError::BadRequest("Invalid input".to_string());
        let response = ErrorResponse::from(error);
        
        assert_eq!(response.error.code, "BAD_REQUEST");
        assert_eq!(response.error.message, "Bad Request: Invalid input");
    }
    
    #[test]
    fn test_validation_error_response() {
        let errors = vec![
            ValidationError {
                field: "email".to_string(),
                message: "Invalid email format".to_string(),
                value: Some(serde_json::Value::String("invalid".to_string())),
            },
            ValidationError {
                field: "name".to_string(),
                message: "Name is required".to_string(),
                value: None,
            },
        ];
        
        let response = ValidationErrorResponse::from(errors);
        assert_eq!(response.error.code, "VALIDATION_ERROR");
        assert_eq!(response.validation_errors.len(), 2);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. RESTful resource design
#[derive(Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

// 2. API versioning
enum ApiVersion {
    V1,
    V2,
    V3,
}

// 3. Error handling
#[derive(Error, Debug)]
enum ApiError {
    #[error("Bad Request: {0}")]
    BadRequest(String),
    #[error("Not Found: {0}")]
    NotFound(String),
}

// 4. Response wrappers
#[derive(Serialize)]
struct ApiResponse<T> {
    data: T,
    meta: ResponseMeta,
}
```

### Essential Patterns

```rust
// Complete API design setup
pub fn setup_rust_api_design() {
    // 1. RESTful design
    // 2. API versioning
    // 3. Error handling
    // 4. Response wrappers
    // 5. Validation
    // 6. Documentation
    // 7. Middleware
    // 8. Testing
    
    println!("Rust API design setup complete!");
}
```

---

*This guide provides the complete machinery for Rust API design. Each pattern includes implementation examples, API strategies, and real-world usage patterns for enterprise API development.*
