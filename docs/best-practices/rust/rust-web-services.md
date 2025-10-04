# Rust Web Services Best Practices

**Objective**: Master senior-level Rust web service patterns for production systems. When you need to build scalable web services, when you want to leverage Rust's performance for web applications, when you need enterprise-grade web service patterns—these best practices become your weapon of choice.

## Core Principles

- **Performance**: Leverage Rust's zero-cost abstractions for web services
- **Type Safety**: Use Rust's type system for API safety
- **Async Programming**: Build high-performance async web services
- **Middleware**: Implement reusable middleware patterns
- **Error Handling**: Comprehensive error handling for web services

## Web Framework Patterns

### Actix Web Patterns

```rust
// rust/01-actix-patterns.rs

/*
Actix Web patterns and best practices for Rust
*/

use actix_web::{web, App, HttpServer, HttpResponse, Result, middleware};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// User data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// User service for managing users.
pub struct UserService {
    users: Mutex<Vec<User>>,
    next_id: Mutex<u64>,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: Mutex::new(Vec::new()),
            next_id: Mutex::new(1),
        }
    }
    
    pub fn create_user(&self, name: String, email: String) -> Result<User, String> {
        let mut users = self.users.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();
        
        let user = User {
            id: *next_id,
            name,
            email,
        };
        
        *next_id += 1;
        users.push(user.clone());
        Ok(user)
    }
    
    pub fn get_user(&self, id: u64) -> Result<Option<User>, String> {
        let users = self.users.lock().unwrap();
        Ok(users.iter().find(|u| u.id == id).cloned())
    }
    
    pub fn get_all_users(&self) -> Result<Vec<User>, String> {
        let users = self.users.lock().unwrap();
        Ok(users.clone())
    }
    
    pub fn update_user(&self, id: u64, name: Option<String>, email: Option<String>) -> Result<Option<User>, String> {
        let mut users = self.users.lock().unwrap();
        
        if let Some(user) = users.iter_mut().find(|u| u.id == id) {
            if let Some(name) = name {
                user.name = name;
            }
            if let Some(email) = email {
                user.email = email;
            }
            Ok(Some(user.clone()))
        } else {
            Ok(None)
        }
    }
    
    pub fn delete_user(&self, id: u64) -> Result<bool, String> {
        let mut users = self.users.lock().unwrap();
        let initial_len = users.len();
        users.retain(|u| u.id != id);
        Ok(users.len() < initial_len)
    }
}

/// Request/Response structures.
#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

/// Handler functions.
pub async fn create_user(
    data: web::Data<UserService>,
    req: web::Json<CreateUserRequest>,
) -> Result<HttpResponse, actix_web::Error> {
    let user = data.create_user(req.name.clone(), req.email.clone())
        .map_err(|e| actix_web::error::ErrorBadRequest(e))?;
    
    Ok(HttpResponse::Created().json(user))
}

pub async fn get_user(
    data: web::Data<UserService>,
    path: web::Path<u64>,
) -> Result<HttpResponse, actix_web::Error> {
    let user = data.get_user(path.into_inner())
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    
    match user {
        Some(user) => Ok(HttpResponse::Ok().json(user)),
        None => Ok(HttpResponse::NotFound().json("User not found")),
    }
}

pub async fn get_all_users(
    data: web::Data<UserService>,
) -> Result<HttpResponse, actix_web::Error> {
    let users = data.get_all_users()
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    
    Ok(HttpResponse::Ok().json(users))
}

pub async fn update_user(
    data: web::Data<UserService>,
    path: web::Path<u64>,
    req: web::Json<UpdateUserRequest>,
) -> Result<HttpResponse, actix_web::Error> {
    let user = data.update_user(path.into_inner(), req.name.clone(), req.email.clone())
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    
    match user {
        Some(user) => Ok(HttpResponse::Ok().json(user)),
        None => Ok(HttpResponse::NotFound().json("User not found")),
    }
}

pub async fn delete_user(
    data: web::Data<UserService>,
    path: web::Path<u64>,
) -> Result<HttpResponse, actix_web::Error> {
    let deleted = data.delete_user(path.into_inner())
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    
    if deleted {
        Ok(HttpResponse::NoContent().finish())
    } else {
        Ok(HttpResponse::NotFound().json("User not found"))
    }
}

/// Application configuration.
pub fn configure_app(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .service(
                web::resource("/users")
                    .route(web::get().to(get_all_users))
                    .route(web::post().to(create_user))
            )
            .service(
                web::resource("/users/{id}")
                    .route(web::get().to(get_user))
                    .route(web::put().to(update_user))
                    .route(web::delete().to(delete_user))
            )
    );
}

/// Main application setup.
pub async fn create_app() -> std::io::Result<()> {
    let user_service = web::Data::new(UserService::new());
    
    HttpServer::new(move || {
        App::new()
            .app_data(user_service.clone())
            .wrap(middleware::Logger::default())
            .wrap(middleware::DefaultHeaders::new().header("X-Version", "1.0"))
            .configure(configure_app)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_service() {
        let service = UserService::new();
        
        let user = service.create_user("John".to_string(), "john@example.com".to_string()).unwrap();
        assert_eq!(user.name, "John");
        assert_eq!(user.email, "john@example.com");
        
        let retrieved = service.get_user(user.id).unwrap().unwrap();
        assert_eq!(retrieved.name, "John");
    }
}
```

### Axum Patterns

```rust
// rust/02-axum-patterns.rs

/*
Axum patterns and best practices for Rust
*/

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// User data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Application state.
pub struct AppState {
    pub users: Mutex<Vec<User>>,
    pub next_id: Mutex<u64>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            users: Mutex::new(Vec::new()),
            next_id: Mutex::new(1),
        }
    }
}

/// Request/Response structures.
#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

#[derive(Deserialize)]
pub struct UserQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Handler functions.
pub async fn create_user(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateUserRequest>,
) -> Result<Json<User>, StatusCode> {
    let mut users = state.users.lock().await;
    let mut next_id = state.next_id.lock().await;
    
    let user = User {
        id: *next_id,
        name: req.name,
        email: req.email,
    };
    
    *next_id += 1;
    users.push(user.clone());
    
    Ok(Json(user))
}

pub async fn get_user(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<User>, StatusCode> {
    let users = state.users.lock().await;
    
    match users.iter().find(|u| u.id == id) {
        Some(user) => Ok(Json(user.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn get_all_users(
    State(state): State<Arc<AppState>>,
    Query(query): Query<UserQuery>,
) -> Result<Json<Vec<User>>, StatusCode> {
    let users = state.users.lock().await;
    let mut result = users.clone();
    
    if let Some(offset) = query.offset {
        result = result.into_iter().skip(offset).collect();
    }
    
    if let Some(limit) = query.limit {
        result = result.into_iter().take(limit).collect();
    }
    
    Ok(Json(result))
}

pub async fn update_user(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<UpdateUserRequest>,
) -> Result<Json<User>, StatusCode> {
    let mut users = state.users.lock().await;
    
    match users.iter_mut().find(|u| u.id == id) {
        Some(user) => {
            if let Some(name) = req.name {
                user.name = name;
            }
            if let Some(email) = req.email {
                user.email = email;
            }
            Ok(Json(user.clone()))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn delete_user(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<StatusCode, StatusCode> {
    let mut users = state.users.lock().await;
    let initial_len = users.len();
    users.retain(|u| u.id != id);
    
    if users.len() < initial_len {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Application setup.
pub fn create_app() -> Router {
    let state = Arc::new(AppState::new());
    
    Router::new()
        .route("/api/v1/users", get(get_all_users).post(create_user))
        .route("/api/v1/users/:id", get(get_user).put(update_user).delete(delete_user))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_app_state() {
        let state = Arc::new(AppState::new());
        
        let req = CreateUserRequest {
            name: "John".to_string(),
            email: "john@example.com".to_string(),
        };
        
        let result = create_user(State(state), Json(req)).await;
        assert!(result.is_ok());
    }
}
```

### Warp Patterns

```rust
// rust/03-warp-patterns.rs

/*
Warp patterns and best practices for Rust
*/

use warp::{
    Filter, Rejection, Reply,
    filters::cors::CorsForbidden,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// User data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Application state.
pub struct AppState {
    pub users: Mutex<Vec<User>>,
    pub next_id: Mutex<u64>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            users: Mutex::new(Vec::new()),
            next_id: Mutex::new(1),
        }
    }
}

/// Request/Response structures.
#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
    pub email: String,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub name: Option<String>,
    pub email: Option<String>,
}

/// Custom error type.
#[derive(Debug)]
pub enum AppError {
    UserNotFound,
    InvalidRequest,
    InternalError,
}

impl warp::reject::Reject for AppError {}

/// Handler functions.
pub async fn create_user(
    req: CreateUserRequest,
    state: Arc<AppState>,
) -> Result<impl Reply, Rejection> {
    let mut users = state.users.lock().await;
    let mut next_id = state.next_id.lock().await;
    
    let user = User {
        id: *next_id,
        name: req.name,
        email: req.email,
    };
    
    *next_id += 1;
    users.push(user.clone());
    
    Ok(warp::reply::json(&user))
}

pub async fn get_user(
    id: u64,
    state: Arc<AppState>,
) -> Result<impl Reply, Rejection> {
    let users = state.users.lock().await;
    
    match users.iter().find(|u| u.id == id) {
        Some(user) => Ok(warp::reply::json(user)),
        None => Err(warp::reject::custom(AppError::UserNotFound)),
    }
}

pub async fn get_all_users(
    state: Arc<AppState>,
) -> Result<impl Reply, Rejection> {
    let users = state.users.lock().await;
    Ok(warp::reply::json(&*users))
}

pub async fn update_user(
    id: u64,
    req: UpdateUserRequest,
    state: Arc<AppState>,
) -> Result<impl Reply, Rejection> {
    let mut users = state.users.lock().await;
    
    match users.iter_mut().find(|u| u.id == id) {
        Some(user) => {
            if let Some(name) = req.name {
                user.name = name;
            }
            if let Some(email) = req.email {
                user.email = email;
            }
            Ok(warp::reply::json(user))
        }
        None => Err(warp::reject::custom(AppError::UserNotFound)),
    }
}

pub async fn delete_user(
    id: u64,
    state: Arc<AppState>,
) -> Result<impl Reply, Rejection> {
    let mut users = state.users.lock().await;
    let initial_len = users.len();
    users.retain(|u| u.id != id);
    
    if users.len() < initial_len {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(warp::reject::custom(AppError::UserNotFound))
    }
}

/// Filter functions.
pub fn with_state(
    state: Arc<AppState>,
) -> impl Filter<Extract = (Arc<AppState>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

pub fn json_body<T>() -> impl Filter<Extract = (T,), Error = warp::Rejection> + Clone
where
    T: for<'de> serde::Deserialize<'de> + Send,
{
    warp::body::content_length_limit(1024 * 16).and(warp::body::json())
}

/// Application setup.
pub fn create_app() -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
    let state = Arc::new(AppState::new());
    
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type"])
        .allow_methods(vec!["GET", "POST", "PUT", "DELETE"]);
    
    let create_user = warp::path("api")
        .and(warp::path("v1"))
        .and(warp::path("users"))
        .and(warp::post())
        .and(json_body::<CreateUserRequest>())
        .and(with_state(state.clone()))
        .and_then(create_user);
    
    let get_user = warp::path("api")
        .and(warp::path("v1"))
        .and(warp::path("users"))
        .and(warp::path::param::<u64>())
        .and(warp::get())
        .and(with_state(state.clone()))
        .and_then(get_user);
    
    let get_all_users = warp::path("api")
        .and(warp::path("v1"))
        .and(warp::path("users"))
        .and(warp::get())
        .and(with_state(state.clone()))
        .and_then(get_all_users);
    
    let update_user = warp::path("api")
        .and(warp::path("v1"))
        .and(warp::path("users"))
        .and(warp::path::param::<u64>())
        .and(warp::put())
        .and(json_body::<UpdateUserRequest>())
        .and(with_state(state.clone()))
        .and_then(update_user);
    
    let delete_user = warp::path("api")
        .and(warp::path("v1"))
        .and(warp::path("users"))
        .and(warp::path::param::<u64>())
        .and(warp::delete())
        .and(with_state(state.clone()))
        .and_then(delete_user);
    
    create_user
        .or(get_user)
        .or(get_all_users)
        .or(update_user)
        .or(delete_user)
        .with(cors)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_app_state() {
        let state = Arc::new(AppState::new());
        
        let req = CreateUserRequest {
            name: "John".to_string(),
            email: "john@example.com".to_string(),
        };
        
        let result = create_user(req, state).await;
        assert!(result.is_ok());
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Actix Web
use actix_web::{web, App, HttpServer, HttpResponse};

async fn handler() -> HttpResponse {
    HttpResponse::Ok().json("Hello, World!")
}

// 2. Axum
use axum::{Router, routing::get, Json};

async fn handler() -> Json<&'static str> {
    Json("Hello, World!")
}

// 3. Warp
use warp::Filter;

let routes = warp::path("hello")
    .map(|| "Hello, World!");
```

### Essential Patterns

```rust
// Complete web services setup
pub fn setup_rust_web_services() {
    // 1. Web frameworks
    // 2. Async programming
    // 3. Middleware patterns
    // 4. Error handling
    // 5. State management
    // 6. Routing patterns
    // 7. Serialization
    // 8. Performance optimization
    
    println!("Rust web services setup complete!");
}
```

---

*This guide provides the complete machinery for Rust web services. Each pattern includes implementation examples, web service strategies, and real-world usage patterns for enterprise web development.*
