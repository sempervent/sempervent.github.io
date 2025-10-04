# Rust Code Quality Best Practices

**Objective**: Master senior-level Rust code quality patterns for production systems. When you need to maintain high code quality, when you want to enforce coding standards, when you need enterprise-grade quality assurance—these best practices become your weapon of choice.

## Core Principles

- **Consistency**: Follow consistent coding style and patterns
- **Readability**: Write code that is easy to understand and maintain
- **Performance**: Optimize for speed and memory efficiency
- **Safety**: Leverage Rust's safety guarantees
- **Documentation**: Provide comprehensive documentation

## Linting & Formatting

### Clippy Configuration

```rust
// rust/01-clippy-configuration.rs

/*
Clippy configuration and linting patterns for Rust code quality
*/

// Clippy configuration in Cargo.toml
/*
[lints.clippy]
# Performance lints
all = "warn"
pedantic = "warn"
nursery = "warn"
cargo = "warn"

# Specific lint configurations
clippy::all = "warn"
clippy::pedantic = "warn"
clippy::nursery = "warn"
clippy::cargo = "warn"

# Allow some pedantic lints that might be too strict
clippy::module_name_repetitions = "allow"
clippy::must_use_candidate = "allow"
clippy::missing_errors_doc = "allow"
*/

use std::collections::HashMap;
use std::fmt;
use anyhow::Result;

/// A high-quality data structure for managing key-value pairs.
///
/// This structure provides efficient storage and retrieval of string key-value pairs
/// with built-in validation and error handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValueStore {
    data: HashMap<String, String>,
    max_size: usize,
}

impl KeyValueStore {
    /// Creates a new `KeyValueStore` with the specified maximum size.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum number of key-value pairs that can be stored
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let store = KeyValueStore::new(1000);
    /// ```
    pub fn new(max_size: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_size,
        }
    }
    
    /// Inserts a key-value pair into the store.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert
    /// * `value` - The value to associate with the key
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the insertion was successful, or an error if:
    /// - The key is empty
    /// - The store is at maximum capacity
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("user_id", "12345").unwrap();
    /// ```
    pub fn insert(&mut self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        if self.data.len() >= self.max_size && !self.data.contains_key(key) {
            return Err(anyhow::anyhow!("Store is at maximum capacity"));
        }
        
        self.data.insert(key.to_string(), value.to_string());
        Ok(())
    }
    
    /// Retrieves a value by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// Returns `Some(&str)` if the key exists, or `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("user_id", "12345").unwrap();
    /// assert_eq!(store.get("user_id"), Some("12345"));
    /// ```
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
    
    /// Removes a key-value pair from the store.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to remove
    ///
    /// # Returns
    ///
    /// Returns `Some(String)` if the key existed and was removed, or `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("user_id", "12345").unwrap();
    /// assert_eq!(store.remove("user_id"), Some("12345".to_string()));
    /// ```
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }
    
    /// Returns the number of key-value pairs in the store.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("key1", "value1").unwrap();
    /// store.insert("key2", "value2").unwrap();
    /// assert_eq!(store.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns `true` if the store is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let store = KeyValueStore::new(100);
    /// assert!(store.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Returns `true` if the store contains the specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("user_id", "12345").unwrap();
    /// assert!(store.contains_key("user_id"));
    /// ```
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    /// Clears all key-value pairs from the store.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::KeyValueStore;
    ///
    /// let mut store = KeyValueStore::new(100);
    /// store.insert("key", "value").unwrap();
    /// store.clear();
    /// assert!(store.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for KeyValueStore {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl fmt::Display for KeyValueStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyValueStore({}/{})", self.data.len(), self.max_size)
    }
}

// Iterator implementation for better ergonomics
impl<'a> IntoIterator for &'a KeyValueStore {
    type Item = (&'a String, &'a String);
    type IntoIter = std::collections::hash_map::Iter<'a, String, String>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_store() {
        let store = KeyValueStore::new(100);
        assert_eq!(store.max_size, 100);
        assert!(store.is_empty());
    }
    
    #[test]
    fn test_insert_and_get() {
        let mut store = KeyValueStore::new(100);
        store.insert("key", "value").unwrap();
        assert_eq!(store.get("key"), Some("value"));
    }
    
    #[test]
    fn test_empty_key_error() {
        let mut store = KeyValueStore::new(100);
        let result = store.insert("", "value");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_capacity_limit() {
        let mut store = KeyValueStore::new(1);
        store.insert("key1", "value1").unwrap();
        let result = store.insert("key2", "value2");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_remove() {
        let mut store = KeyValueStore::new(100);
        store.insert("key", "value").unwrap();
        assert_eq!(store.remove("key"), Some("value".to_string()));
        assert!(store.get("key").is_none());
    }
    
    #[test]
    fn test_contains_key() {
        let mut store = KeyValueStore::new(100);
        store.insert("key", "value").unwrap();
        assert!(store.contains_key("key"));
        assert!(!store.contains_key("nonexistent"));
    }
    
    #[test]
    fn test_clear() {
        let mut store = KeyValueStore::new(100);
        store.insert("key", "value").unwrap();
        store.clear();
        assert!(store.is_empty());
    }
    
    #[test]
    fn test_display() {
        let store = KeyValueStore::new(100);
        let display = format!("{}", store);
        assert!(display.contains("KeyValueStore"));
    }
    
    #[test]
    fn test_iterator() {
        let mut store = KeyValueStore::new(100);
        store.insert("key1", "value1").unwrap();
        store.insert("key2", "value2").unwrap();
        
        let mut items: Vec<_> = store.into_iter().collect();
        items.sort_by_key(|(k, _)| *k);
        
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (&"key1".to_string(), &"value1".to_string()));
        assert_eq!(items[1], (&"key2".to_string(), &"value2".to_string()));
    }
}
```

### Rustfmt Configuration

```rust
// rust/02-rustfmt-configuration.rs

/*
Rustfmt configuration and formatting patterns
*/

// rustfmt.toml configuration
/*
# rustfmt.toml
edition = "2021"
max_width = 100
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
indent_style = "Block"
wrap_comments = true
comment_width = 80
normalize_comments = true
normalize_doc_attributes = true
format_code_in_doc_comments = true
format_strings = true
format_macro_matchers = true
format_generated_files = false
required_version = "1.70.0"
*/

use std::collections::HashMap;
use std::fmt;
use anyhow::Result;

/// A well-formatted data structure for managing user sessions.
///
/// This structure provides efficient storage and retrieval of user session data
/// with built-in validation and error handling. It follows Rust formatting
/// best practices for readability and maintainability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserSession {
    user_id: u64,
    session_id: String,
    data: HashMap<String, String>,
    created_at: std::time::SystemTime,
    expires_at: std::time::SystemTime,
}

impl UserSession {
    /// Creates a new `UserSession` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `user_id` - The ID of the user
    /// * `session_id` - The unique session identifier
    /// * `duration_seconds` - Session duration in seconds
    ///
    /// # Returns
    ///
    /// Returns a new `UserSession` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// ```
    pub fn new(
        user_id: u64,
        session_id: String,
        duration_seconds: u64,
    ) -> Self {
        let now = std::time::SystemTime::now();
        let expires_at = now
            + std::time::Duration::from_secs(duration_seconds);
        
        Self {
            user_id,
            session_id,
            data: HashMap::new(),
            created_at: now,
            expires_at,
        }
    }
    
    /// Adds a key-value pair to the session data.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to add
    /// * `value` - The value to associate with the key
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the addition was successful, or an error if:
    /// - The key is empty
    /// - The session has expired
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// session.set_data("theme", "dark").unwrap();
    /// ```
    pub fn set_data(&mut self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        if self.is_expired() {
            return Err(anyhow::anyhow!("Session has expired"));
        }
        
        self.data.insert(key.to_string(), value.to_string());
        Ok(())
    }
    
    /// Retrieves a value by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// Returns `Some(&str)` if the key exists and the session is valid,
    /// or `None` if not found or expired.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// session.set_data("theme", "dark").unwrap();
    /// assert_eq!(session.get_data("theme"), Some("dark"));
    /// ```
    pub fn get_data(&self, key: &str) -> Option<&str> {
        if self.is_expired() {
            return None;
        }
        
        self.data.get(key).map(|s| s.as_str())
    }
    
    /// Checks if the session has expired.
    ///
    /// # Returns
    ///
    /// Returns `true` if the session has expired, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// assert!(!session.is_expired());
    /// ```
    pub fn is_expired(&self) -> bool {
        std::time::SystemTime::now() > self.expires_at
    }
    
    /// Returns the user ID associated with this session.
    ///
    /// # Returns
    ///
    /// Returns the user ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// assert_eq!(session.user_id(), 12345);
    /// ```
    pub fn user_id(&self) -> u64 {
        self.user_id
    }
    
    /// Returns the session ID.
    ///
    /// # Returns
    ///
    /// Returns the session ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// assert_eq!(session.session_id(), "sess_abc123");
    /// ```
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    
    /// Returns the number of data items in the session.
    ///
    /// # Returns
    ///
    /// Returns the number of data items.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// session.set_data("key1", "value1").unwrap();
    /// session.set_data("key2", "value2").unwrap();
    /// assert_eq!(session.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns `true` if the session has no data.
    ///
    /// # Returns
    ///
    /// Returns `true` if the session is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::UserSession;
    ///
    /// let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
    /// assert!(session.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl fmt::Display for UserSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UserSession(user_id={}, session_id={}, data_count={})",
            self.user_id,
            self.session_id,
            self.data.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_session() {
        let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
        assert_eq!(session.user_id(), 12345);
        assert_eq!(session.session_id(), "sess_abc123");
        assert!(!session.is_expired());
        assert!(session.is_empty());
    }
    
    #[test]
    fn test_set_and_get_data() {
        let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
        session.set_data("theme", "dark").unwrap();
        assert_eq!(session.get_data("theme"), Some("dark"));
    }
    
    #[test]
    fn test_empty_key_error() {
        let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
        let result = session.set_data("", "value");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_session_length() {
        let mut session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
        session.set_data("key1", "value1").unwrap();
        session.set_data("key2", "value2").unwrap();
        assert_eq!(session.len(), 2);
    }
    
    #[test]
    fn test_display() {
        let session = UserSession::new(12345, "sess_abc123".to_string(), 3600);
        let display = format!("{}", session);
        assert!(display.contains("UserSession"));
        assert!(display.contains("12345"));
        assert!(display.contains("sess_abc123"));
    }
}
```

## Code Organization

### Module Structure

```rust
// rust/03-module-structure.rs

/*
Module organization and code structure patterns
*/

// lib.rs - Main library entry point
pub mod auth;
pub mod data;
pub mod errors;
pub mod utils;

// Re-export commonly used types
pub use auth::{User, UserService};
pub use data::{DataStore, DataService};
pub use errors::{Error, Result};

// Feature-gated modules
#[cfg(feature = "async")]
pub mod async_utils;

#[cfg(feature = "serde")]
pub mod serialization;

// Example module structure
pub mod auth {
    use std::collections::HashMap;
    use anyhow::Result;
    
    /// Represents a user in the system.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct User {
        pub id: u64,
        pub username: String,
        pub email: String,
        pub is_active: bool,
    }
    
    /// Service for managing user authentication and authorization.
    pub struct UserService {
        users: HashMap<u64, User>,
        sessions: HashMap<String, u64>,
    }
    
    impl UserService {
        pub fn new() -> Self {
            Self {
                users: HashMap::new(),
                sessions: HashMap::new(),
            }
        }
        
        pub fn create_user(
            &mut self,
            username: String,
            email: String,
        ) -> Result<User> {
            if username.is_empty() {
                return Err(anyhow::anyhow!("Username cannot be empty"));
            }
            if email.is_empty() {
                return Err(anyhow::anyhow!("Email cannot be empty"));
            }
            
            let user = User {
                id: self.users.len() as u64 + 1,
                username,
                email,
                is_active: true,
            };
            
            self.users.insert(user.id, user.clone());
            Ok(user)
        }
        
        pub fn get_user(&self, id: u64) -> Option<&User> {
            self.users.get(&id)
        }
        
        pub fn authenticate(&mut self, username: &str) -> Result<String> {
            let user = self.users.values()
                .find(|u| u.username == username)
                .ok_or_else(|| anyhow::anyhow!("User not found"))?;
            
            if !user.is_active {
                return Err(anyhow::anyhow!("User is not active"));
            }
            
            let session_id = format!("sess_{}", uuid::Uuid::new_v4());
            self.sessions.insert(session_id.clone(), user.id);
            Ok(session_id)
        }
        
        pub fn get_user_by_session(&self, session_id: &str) -> Option<&User> {
            let user_id = self.sessions.get(session_id)?;
            self.users.get(user_id)
        }
    }
    
    impl Default for UserService {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub mod data {
    use std::collections::HashMap;
    use anyhow::Result;
    
    /// A data store for managing key-value pairs.
    pub struct DataStore {
        data: HashMap<String, String>,
    }
    
    impl DataStore {
        pub fn new() -> Self {
            Self {
                data: HashMap::new(),
            }
        }
        
        pub fn set(&mut self, key: String, value: String) -> Result<()> {
            if key.is_empty() {
                return Err(anyhow::anyhow!("Key cannot be empty"));
            }
            
            self.data.insert(key, value);
            Ok(())
        }
        
        pub fn get(&self, key: &str) -> Option<&str> {
            self.data.get(key).map(|s| s.as_str())
        }
        
        pub fn remove(&mut self, key: &str) -> Option<String> {
            self.data.remove(key)
        }
    }
    
    impl Default for DataStore {
        fn default() -> Self {
            Self::new()
        }
    }
    
    /// Service for managing data operations.
    pub struct DataService {
        store: DataStore,
    }
    
    impl DataService {
        pub fn new() -> Self {
            Self {
                store: DataStore::new(),
            }
        }
        
        pub fn store_data(&mut self, key: String, value: String) -> Result<()> {
            self.store.set(key, value)
        }
        
        pub fn retrieve_data(&self, key: &str) -> Option<&str> {
            self.store.get(key)
        }
        
        pub fn delete_data(&mut self, key: &str) -> Option<String> {
            self.store.remove(key)
        }
    }
    
    impl Default for DataService {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub mod errors {
    use thiserror::Error;
    
    #[derive(Error, Debug)]
    pub enum Error {
        #[error("Authentication error: {0}")]
        Authentication(String),
        
        #[error("Data error: {0}")]
        Data(String),
        
        #[error("Validation error: {0}")]
        Validation(String),
        
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
    }
    
    pub type Result<T> = std::result::Result<T, Error>;
}

pub mod utils {
    use std::time::{Duration, Instant};
    
    /// Measures the execution time of a function.
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Formats a duration as a human-readable string.
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_secs() > 0 {
            format!("{:.2}s", duration.as_secs_f64())
        } else {
            format!("{:.2}ms", duration.as_millis())
        }
    }
}

#[cfg(feature = "async")]
pub mod async_utils {
    use tokio::time::{sleep, Duration};
    
    /// Performs an async operation with a delay.
    pub async fn async_operation() -> Result<(), Box<dyn std::error::Error>> {
        sleep(Duration::from_millis(100)).await;
        Ok(())
    }
}

#[cfg(feature = "serde")]
pub mod serialization {
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Serialize, Deserialize)]
    pub struct SerializableData {
        pub id: u64,
        pub name: String,
        pub value: f64,
    }
    
    impl SerializableData {
        pub fn to_json(&self) -> Result<String, serde_json::Error> {
            serde_json::to_string(self)
        }
        
        pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(json)
        }
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Clippy configuration
// Cargo.toml
/*
[lints.clippy]
all = "warn"
pedantic = "warn"
*/

// 2. Rustfmt configuration
// rustfmt.toml
/*
edition = "2021"
max_width = 100
tab_spaces = 4
*/

// 3. Code organization
pub mod auth;
pub mod data;
pub mod errors;

// 4. Documentation
/// A high-quality data structure.
/// 
/// # Examples
/// ```
/// use my_crate::MyStruct;
/// let instance = MyStruct::new();
/// ```

// 5. Error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Something went wrong: {0}")]
    Something(String),
}
```

### Essential Patterns

```rust
// Complete code quality setup
pub fn setup_rust_code_quality() {
    // 1. Clippy configuration
    // 2. Rustfmt configuration
    // 3. Module organization
    // 4. Documentation standards
    // 5. Error handling patterns
    // 6. Testing standards
    // 7. Performance considerations
    // 8. Security best practices
    
    println!("Rust code quality setup complete!");
}
```

---

*This guide provides the complete machinery for Rust code quality best practices. Each pattern includes implementation examples, quality standards, and real-world usage patterns for enterprise code development.*
