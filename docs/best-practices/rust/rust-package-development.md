# Rust Package Development Best Practices

**Objective**: Master senior-level Rust package development patterns for production systems. When you need to build and ship Rust crates, when you want to follow proven methodologies, when you need enterprise-grade packaging strategies—these best practices become your weapon of choice.

## Core Principles

- **Semantic Versioning**: Follow semver for version management
- **Documentation**: Comprehensive docs and examples
- **Testing**: Unit, integration, and property-based testing
- **Performance**: Optimize for speed and memory usage
- **Compatibility**: Maintain API stability and backward compatibility

## Crate Structure & Configuration

### Cargo.toml Best Practices

```rust
// rust/01-cargo-configuration.rs

/*
Cargo.toml configuration patterns and crate metadata
*/

// Example Cargo.toml for a production Rust crate
/*
[package]
name = "my-awesome-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "A high-performance Rust library for data processing"
license = "MIT OR Apache-2.0"
repository = "https://github.com/your-org/my-awesome-crate"
homepage = "https://docs.rs/my-awesome-crate"
documentation = "https://docs.rs/my-awesome-crate"
readme = "README.md"
keywords = ["data", "processing", "async", "performance"]
categories = ["data-structures", "web-programming::http-client"]
rust-version = "1.70"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"

[dependencies.reqwest]
version = "0.11"
features = ["json"]
optional = true

[dev-dependencies]
criterion = "0.5"
proptest = "1.0"
mockall = "0.11"

[features]
default = ["reqwest"]
full = ["reqwest", "tracing"]
no-default-features = []

[lib]
name = "my_awesome_crate"
path = "src/lib.rs"
crate-type = ["lib", "cdylib"]

[[bin]]
name = "my-tool"
path = "src/bin/my_tool.rs"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Example of a well-structured Rust crate
pub struct MyAwesomeCrate {
    config: Config,
    data: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub timeout: u64,
    pub max_retries: u32,
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            timeout: 30,
            max_retries: 3,
            debug: false,
        }
    }
}

impl MyAwesomeCrate {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }
    
    /// Create a new instance with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            data: HashMap::new(),
        }
    }
    
    /// Process data with error handling
    pub fn process_data(&mut self, key: String, value: String) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        self.data.insert(key, value);
        Ok(())
    }
    
    /// Get data by key
    pub fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_new_crate() {
        let crate_instance = MyAwesomeCrate::new();
        assert_eq!(crate_instance.config.timeout, 30);
        assert_eq!(crate_instance.config.max_retries, 3);
        assert!(!crate_instance.config.debug);
    }
    
    #[test]
    fn test_process_data() {
        let mut crate_instance = MyAwesomeCrate::new();
        
        let result = crate_instance.process_data("test_key".to_string(), "test_value".to_string());
        assert!(result.is_ok());
        
        assert_eq!(crate_instance.get_data("test_key"), Some(&"test_value".to_string()));
    }
    
    #[test]
    fn test_empty_key_error() {
        let mut crate_instance = MyAwesomeCrate::new();
        
        let result = crate_instance.process_data("".to_string(), "test_value".to_string());
        assert!(result.is_err());
    }
    
    proptest! {
        #[test]
        fn test_process_data_property(key in ".*", value in ".*") {
            let mut crate_instance = MyAwesomeCrate::new();
            let result = crate_instance.process_data(key.clone(), value.clone());
            
            if key.is_empty() {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok());
                assert_eq!(crate_instance.get_data(&key), Some(&value));
            }
        }
    }
}

// Benchmarking example
#[cfg(feature = "bench")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "bench")]
fn benchmark_process_data(c: &mut Criterion) {
    c.bench_function("process_data", |b| {
        b.iter(|| {
            let mut crate_instance = MyAwesomeCrate::new();
            crate_instance.process_data(
                black_box("benchmark_key".to_string()),
                black_box("benchmark_value".to_string())
            )
        })
    });
}

#[cfg(feature = "bench")]
criterion_group!(benches, benchmark_process_data);
#[cfg(feature = "bench")]
criterion_main!(benches);
```

### Module Organization

```rust
// rust/02-module-organization.rs

/*
Module organization patterns and crate structure
*/

// lib.rs - Main library entry point
pub mod config;
pub mod data;
pub mod errors;
pub mod utils;

// Re-export commonly used types
pub use config::Config;
pub use data::DataProcessor;
pub use errors::{Error, Result};

// Feature-gated modules
#[cfg(feature = "async")]
pub mod async_utils;

#[cfg(feature = "serde")]
pub mod serialization;

// Example module structure
pub mod config {
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Config {
        pub database_url: String,
        pub max_connections: u32,
        pub timeout_seconds: u64,
    }
    
    impl Default for Config {
        fn default() -> Self {
            Self {
                database_url: "sqlite://:memory:".to_string(),
                max_connections: 10,
                timeout_seconds: 30,
            }
        }
    }
}

pub mod data {
    use super::config::Config;
    use std::collections::HashMap;
    
    pub struct DataProcessor {
        config: Config,
        cache: HashMap<String, String>,
    }
    
    impl DataProcessor {
        pub fn new(config: Config) -> Self {
            Self {
                config,
                cache: HashMap::new(),
            }
        }
        
        pub fn process(&mut self, key: &str, value: &str) -> Result<(), String> {
            if key.is_empty() {
                return Err("Key cannot be empty".to_string());
            }
            
            self.cache.insert(key.to_string(), value.to_string());
            Ok(())
        }
    }
}

pub mod errors {
    use thiserror::Error;
    
    #[derive(Error, Debug)]
    pub enum Error {
        #[error("Configuration error: {0}")]
        Config(String),
        
        #[error("Data processing error: {0}")]
        DataProcessing(String),
        
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
    }
    
    pub type Result<T> = std::result::Result<T, Error>;
}

pub mod utils {
    use std::time::{Duration, Instant};
    
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    pub fn format_duration(duration: Duration) -> String {
        format!("{:.2}ms", duration.as_secs_f64() * 1000.0)
    }
}

#[cfg(feature = "async")]
pub mod async_utils {
    use tokio::time::{sleep, Duration};
    
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

## Testing Strategies

### Comprehensive Testing

```rust
// rust/03-testing-strategies.rs

/*
Comprehensive testing strategies for Rust crates
*/

use std::collections::HashMap;
use anyhow::Result;
use proptest::prelude::*;
use mockall::mock;

// Unit tests
#[cfg(test)]
mod unit_tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        let mut processor = DataProcessor::new();
        assert!(processor.add_item("key", "value").is_ok());
        assert_eq!(processor.get_item("key"), Some("value"));
    }
    
    #[test]
    fn test_error_handling() {
        let mut processor = DataProcessor::new();
        assert!(processor.add_item("", "value").is_err());
    }
    
    #[test]
    fn test_edge_cases() {
        let mut processor = DataProcessor::new();
        
        // Test with empty value
        assert!(processor.add_item("key", "").is_ok());
        
        // Test with special characters
        assert!(processor.add_item("key@#$", "value!@#").is_ok());
        
        // Test with unicode
        assert!(processor.add_item("ключ", "значение").is_ok());
    }
}

// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_workflow() {
        let mut processor = DataProcessor::new();
        
        // Add multiple items
        processor.add_item("item1", "value1").unwrap();
        processor.add_item("item2", "value2").unwrap();
        processor.add_item("item3", "value3").unwrap();
        
        // Verify all items
        assert_eq!(processor.get_item("item1"), Some("value1"));
        assert_eq!(processor.get_item("item2"), Some("value2"));
        assert_eq!(processor.get_item("item3"), Some("value3"));
        
        // Test removal
        processor.remove_item("item2").unwrap();
        assert_eq!(processor.get_item("item2"), None);
    }
}

// Property-based testing
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_add_get_property(
            key in "[a-zA-Z0-9_]+",
            value in ".*"
        ) {
            let mut processor = DataProcessor::new();
            processor.add_item(&key, &value).unwrap();
            assert_eq!(processor.get_item(&key), Some(&value));
        }
        
        #[test]
        fn test_remove_property(
            key in "[a-zA-Z0-9_]+",
            value in ".*"
        ) {
            let mut processor = DataProcessor::new();
            processor.add_item(&key, &value).unwrap();
            processor.remove_item(&key).unwrap();
            assert_eq!(processor.get_item(&key), None);
        }
    }
}

// Mock testing
mock! {
    pub DataProcessor {}
    
    impl DataProcessor for DataProcessor {
        fn add_item(&mut self, key: &str, value: &str) -> Result<()>;
        fn get_item(&self, key: &str) -> Option<&str>;
        fn remove_item(&mut self, key: &str) -> Result<()>;
    }
}

#[cfg(test)]
mod mock_tests {
    use super::*;
    
    #[test]
    fn test_with_mock() {
        let mut mock = MockDataProcessor::new();
        
        mock.expect_add_item()
            .with(eq("test_key"), eq("test_value"))
            .times(1)
            .returning(|_, _| Ok(()));
        
        mock.expect_get_item()
            .with(eq("test_key"))
            .times(1)
            .returning(|_| Some("test_value"));
        
        // Use the mock
        mock.add_item("test_key", "test_value").unwrap();
        assert_eq!(mock.get_item("test_key"), Some("test_value"));
    }
}

// Performance tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_performance() {
        let mut processor = DataProcessor::new();
        let start = Instant::now();
        
        // Add 1000 items
        for i in 0..1000 {
            processor.add_item(&format!("key_{}", i), &format!("value_{}", i)).unwrap();
        }
        
        let duration = start.elapsed();
        println!("Added 1000 items in {:?}", duration);
        
        // Performance should be under 10ms
        assert!(duration.as_millis() < 10);
    }
}

// Example implementation
pub struct DataProcessor {
    data: HashMap<String, String>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    pub fn add_item(&mut self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        self.data.insert(key.to_string(), value.to_string());
        Ok(())
    }
    
    pub fn get_item(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
    
    pub fn remove_item(&mut self, key: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        self.data.remove(key);
        Ok(())
    }
}
```

## Documentation & Examples

### Comprehensive Documentation

```rust
// rust/04-documentation.rs

/*
Documentation patterns and examples for Rust crates
*/

use std::collections::HashMap;
use anyhow::Result;

/// A high-performance data processor for handling key-value operations.
///
/// # Examples
///
/// Basic usage:
/// ```
/// use my_crate::DataProcessor;
///
/// let mut processor = DataProcessor::new();
/// processor.add_item("key", "value").unwrap();
/// assert_eq!(processor.get_item("key"), Some("value"));
/// ```
///
/// With configuration:
/// ```
/// use my_crate::{DataProcessor, Config};
///
/// let config = Config {
///     max_items: 1000,
///     enable_caching: true,
/// };
/// let processor = DataProcessor::with_config(config);
/// ```
///
/// # Performance
///
/// This processor is optimized for high-throughput scenarios:
/// - O(1) average case for insertions and lookups
/// - Memory-efficient storage
/// - Thread-safe operations
///
/// # Thread Safety
///
/// All operations are thread-safe and can be used in concurrent environments.
pub struct DataProcessor {
    data: HashMap<String, String>,
    config: Config,
}

/// Configuration options for the data processor.
///
/// # Examples
///
/// ```
/// use my_crate::Config;
///
/// let config = Config {
///     max_items: 1000,
///     enable_caching: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum number of items that can be stored.
    pub max_items: usize,
    /// Whether to enable internal caching for better performance.
    pub enable_caching: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_items: 10000,
            enable_caching: true,
        }
    }
}

impl DataProcessor {
    /// Creates a new data processor with default configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let processor = DataProcessor::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }
    
    /// Creates a new data processor with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options for the processor
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::{DataProcessor, Config};
    ///
    /// let config = Config {
    ///     max_items: 1000,
    ///     enable_caching: true,
    /// };
    /// let processor = DataProcessor::with_config(config);
    /// ```
    pub fn with_config(config: Config) -> Self {
        Self {
            data: HashMap::new(),
            config,
        }
    }
    
    /// Adds a new item to the processor.
    ///
    /// # Arguments
    ///
    /// * `key` - The key for the item (cannot be empty)
    /// * `value` - The value to store
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the item was added successfully, or an error if:
    /// - The key is empty
    /// - The maximum number of items has been reached
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let mut processor = DataProcessor::new();
    /// processor.add_item("user_id", "12345").unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// This function will return an error if the key is empty or if the
    /// maximum number of items has been reached.
    pub fn add_item(&mut self, key: &str, value: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        if self.data.len() >= self.config.max_items {
            return Err(anyhow::anyhow!("Maximum number of items reached"));
        }
        
        self.data.insert(key.to_string(), value.to_string());
        Ok(())
    }
    
    /// Retrieves an item by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// Returns `Some(&str)` if the item exists, or `None` if not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let mut processor = DataProcessor::new();
    /// processor.add_item("user_id", "12345").unwrap();
    /// assert_eq!(processor.get_item("user_id"), Some("12345"));
    /// ```
    pub fn get_item(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
    
    /// Removes an item by its key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the item to remove
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the item was removed successfully, or an error if the key is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let mut processor = DataProcessor::new();
    /// processor.add_item("user_id", "12345").unwrap();
    /// processor.remove_item("user_id").unwrap();
    /// assert_eq!(processor.get_item("user_id"), None);
    /// ```
    pub fn remove_item(&mut self, key: &str) -> Result<()> {
        if key.is_empty() {
            return Err(anyhow::anyhow!("Key cannot be empty"));
        }
        
        self.data.remove(key);
        Ok(())
    }
    
    /// Returns the number of items currently stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let mut processor = DataProcessor::new();
    /// processor.add_item("key1", "value1").unwrap();
    /// processor.add_item("key2", "value2").unwrap();
    /// assert_eq!(processor.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns `true` if the processor is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use my_crate::DataProcessor;
    ///
    /// let processor = DataProcessor::new();
    /// assert!(processor.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_documentation_examples() {
        let mut processor = DataProcessor::new();
        processor.add_item("key", "value").unwrap();
        assert_eq!(processor.get_item("key"), Some("value"));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic crate structure
// Cargo.toml with proper metadata and dependencies

// 2. Module organization
pub mod config;
pub mod data;
pub mod errors;

// 3. Testing
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }
}

// 4. Documentation
/// A high-performance data processor.
/// 
/// # Examples
/// ```
/// use my_crate::DataProcessor;
/// let processor = DataProcessor::new();
/// ```

// 5. Benchmarking
#[cfg(feature = "bench")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
```

### Essential Patterns

```rust
// Complete package development setup
pub fn setup_rust_package() {
    // 1. Cargo.toml configuration
    // 2. Module organization
    // 3. Testing strategies
    // 4. Documentation
    // 5. Performance optimization
    // 6. Error handling
    // 7. Feature flags
    // 8. CI/CD integration
    
    println!("Rust package development setup complete!");
}
```

---

*This guide provides the complete machinery for Rust package development. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise crate development.*
