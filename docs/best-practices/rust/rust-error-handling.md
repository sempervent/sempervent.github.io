# Rust Error Handling Best Practices

**Objective**: Master senior-level Rust error handling patterns for production systems. When you need to build robust error handling, when you want to create maintainable error propagation, when you need enterprise-grade error management—these best practices become your weapon of choice.

## Core Principles

- **Explicit Error Handling**: Use Result and Option types
- **Error Propagation**: Chain errors with ? operator
- **Custom Error Types**: Create domain-specific error types
- **Error Context**: Provide meaningful error information
- **Recovery Strategies**: Implement error recovery patterns

## Result and Option Patterns

### Basic Error Handling

```rust
// rust/01-basic-error-handling.rs

/*
Basic error handling patterns with Result and Option
*/

use std::fs::File;
use std::io::{self, Read};
use std::num::ParseIntError;
use std::str::FromStr;

/// Demonstrates basic Result patterns.
pub struct DataProcessor {
    data: Vec<String>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes data with error handling.
    pub fn process_data(&mut self, input: &str) -> Result<usize, String> {
        if input.is_empty() {
            return Err("Input cannot be empty".to_string());
        }
        
        let processed = input.to_uppercase();
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Processes data with Option return.
    pub fn get_data(&self, index: usize) -> Option<&String> {
        self.data.get(index)
    }
    
    /// Processes data with Result and error propagation.
    pub fn process_file(&self, filename: &str) -> Result<String, io::Error> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }
    
    /// Processes data with custom error handling.
    pub fn process_number(&self, input: &str) -> Result<i32, ParseIntError> {
        input.parse::<i32>()
    }
    
    /// Processes data with error mapping.
    pub fn process_number_with_context(&self, input: &str) -> Result<i32, String> {
        input.parse::<i32>()
            .map_err(|e| format!("Failed to parse '{}' as integer: {}", input, e))
    }
}

/// Demonstrates Option patterns.
pub struct OptionDemo {
    data: Vec<Option<String>>,
}

impl OptionDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Adds data with Option handling.
    pub fn add_data(&mut self, data: Option<String>) {
        self.data.push(data);
    }
    
    /// Gets data with Option handling.
    pub fn get_data(&self, index: usize) -> Option<&String> {
        self.data.get(index)?.as_ref()
    }
    
    /// Processes data with Option chaining.
    pub fn process_data(&self, index: usize) -> Option<String> {
        self.data.get(index)?
            .as_ref()
            .map(|s| s.to_uppercase())
    }
    
    /// Processes data with Option and default values.
    pub fn get_data_or_default(&self, index: usize) -> String {
        self.data.get(index)
            .and_then(|opt| opt.as_ref())
            .map(|s| s.clone())
            .unwrap_or_else(|| "default".to_string())
    }
}

/// Demonstrates Result patterns with error chaining.
pub struct ResultDemo {
    data: Vec<Result<String, String>>,
}

impl ResultDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Adds data with Result handling.
    pub fn add_data(&mut self, data: Result<String, String>) {
        self.data.push(data);
    }
    
    /// Gets data with Result handling.
    pub fn get_data(&self, index: usize) -> Result<&String, &String> {
        match self.data.get(index) {
            Some(Ok(data)) => Ok(data),
            Some(Err(error)) => Err(error),
            None => Err(&"Index out of bounds".to_string()),
        }
    }
    
    /// Processes data with Result chaining.
    pub fn process_data(&self, index: usize) -> Result<String, String> {
        let data = self.data.get(index)
            .ok_or("Index out of bounds")?;
        
        let value = data.as_ref()
            .map_err(|e| format!("Error in data: {}", e))?;
        
        Ok(value.to_uppercase())
    }
    
    /// Processes data with Result and error mapping.
    pub fn process_data_with_context(&self, index: usize) -> Result<String, String> {
        let data = self.data.get(index)
            .ok_or_else(|| format!("Index {} out of bounds", index))?;
        
        let value = data.as_ref()
            .map_err(|e| format!("Error in data at index {}: {}", index, e))?;
        
        Ok(value.to_uppercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_processor() {
        let mut processor = DataProcessor::new();
        
        let result = processor.process_data("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = processor.process_data("");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Input cannot be empty");
    }
    
    #[test]
    fn test_option_demo() {
        let mut demo = OptionDemo::new();
        demo.add_data(Some("hello".to_string()));
        demo.add_data(None);
        
        assert_eq!(demo.get_data(0), Some(&"hello".to_string()));
        assert_eq!(demo.get_data(1), None);
        
        assert_eq!(demo.process_data(0), Some("HELLO".to_string()));
        assert_eq!(demo.process_data(1), None);
        
        assert_eq!(demo.get_data_or_default(0), "hello");
        assert_eq!(demo.get_data_or_default(1), "default");
    }
    
    #[test]
    fn test_result_demo() {
        let mut demo = ResultDemo::new();
        demo.add_data(Ok("hello".to_string()));
        demo.add_data(Err("error".to_string()));
        
        assert_eq!(demo.get_data(0), Ok(&"hello".to_string()));
        assert_eq!(demo.get_data(1), Err(&"error".to_string()));
        
        assert_eq!(demo.process_data(0), Ok("HELLO".to_string()));
        assert_eq!(demo.process_data(1), Err("Error in data: error".to_string()));
    }
}
```

### Custom Error Types

```rust
// rust/02-custom-error-types.rs

/*
Custom error types and error handling patterns
*/

use std::fmt;
use std::io;
use std::num::ParseIntError;
use thiserror::Error;

/// Custom error type for data processing.
#[derive(Error, Debug)]
pub enum DataProcessingError {
    #[error("Input validation failed: {0}")]
    Validation(String),
    
    #[error("Processing failed: {0}")]
    Processing(String),
    
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Parse error: {0}")]
    Parse(#[from] ParseIntError),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Custom error type for business logic.
#[derive(Error, Debug)]
pub enum BusinessError {
    #[error("User not found: {user_id}")]
    UserNotFound { user_id: u64 },
    
    #[error("Insufficient permissions for user: {user_id}")]
    InsufficientPermissions { user_id: u64 },
    
    #[error("Resource already exists: {resource_name}")]
    ResourceExists { resource_name: String },
    
    #[error("Business rule violation: {rule}")]
    BusinessRuleViolation { rule: String },
    
    #[error("External service error: {service} - {message}")]
    ExternalService { service: String, message: String },
}

/// Custom error type for API operations.
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),
    
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    
    #[error("Forbidden: {0}")]
    Forbidden(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal server error: {0}")]
    InternalServerError(String),
    
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl ApiError {
    /// Get HTTP status code for the error.
    pub fn status_code(&self) -> u16 {
        match self {
            ApiError::BadRequest(_) => 400,
            ApiError::Unauthorized(_) => 401,
            ApiError::Forbidden(_) => 403,
            ApiError::NotFound(_) => 404,
            ApiError::InternalServerError(_) => 500,
            ApiError::ServiceUnavailable(_) => 503,
        }
    }
}

/// Demonstrates custom error handling.
pub struct CustomErrorDemo {
    data: Vec<String>,
}

impl CustomErrorDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes data with custom error handling.
    pub fn process_data(&mut self, input: &str) -> Result<usize, DataProcessingError> {
        if input.is_empty() {
            return Err(DataProcessingError::Validation("Input cannot be empty".to_string()));
        }
        
        if input.len() > 100 {
            return Err(DataProcessingError::Validation("Input too long".to_string()));
        }
        
        let processed = input.to_uppercase();
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Processes data with business error handling.
    pub fn process_business_data(&self, user_id: u64, data: &str) -> Result<String, BusinessError> {
        if user_id == 0 {
            return Err(BusinessError::UserNotFound { user_id });
        }
        
        if user_id < 1000 {
            return Err(BusinessError::InsufficientPermissions { user_id });
        }
        
        if data.is_empty() {
            return Err(BusinessError::BusinessRuleViolation {
                rule: "Data cannot be empty".to_string(),
            });
        }
        
        Ok(data.to_uppercase())
    }
    
    /// Processes data with API error handling.
    pub fn process_api_data(&self, data: &str) -> Result<String, ApiError> {
        if data.is_empty() {
            return Err(ApiError::BadRequest("Data cannot be empty".to_string()));
        }
        
        if data.len() > 1000 {
            return Err(ApiError::BadRequest("Data too long".to_string()));
        }
        
        if data.contains("error") {
            return Err(ApiError::InternalServerError("Data contains error".to_string()));
        }
        
        Ok(data.to_uppercase())
    }
    
    /// Processes data with error chaining.
    pub fn process_data_with_chain(&mut self, input: &str) -> Result<usize, DataProcessingError> {
        let processed = self.process_data(input)?;
        Ok(processed)
    }
    
    /// Processes data with error mapping.
    pub fn process_data_with_mapping(&mut self, input: &str) -> Result<usize, String> {
        self.process_data(input)
            .map_err(|e| format!("Failed to process data: {}", e))
    }
}

/// Demonstrates error recovery patterns.
pub struct ErrorRecoveryDemo {
    data: Vec<String>,
    retry_count: u32,
    max_retries: u32,
}

impl ErrorRecoveryDemo {
    pub fn new(max_retries: u32) -> Self {
        Self {
            data: Vec::new(),
            retry_count: 0,
            max_retries,
        }
    }
    
    /// Processes data with retry logic.
    pub fn process_data_with_retry(&mut self, input: &str) -> Result<usize, String> {
        self.retry_count = 0;
        
        loop {
            match self.process_data_internal(input) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    self.retry_count += 1;
                    if self.retry_count >= self.max_retries {
                        return Err(format!("Max retries exceeded: {}", error));
                    }
                }
            }
        }
    }
    
    /// Internal processing method.
    fn process_data_internal(&mut self, input: &str) -> Result<usize, String> {
        if input.is_empty() {
            return Err("Input cannot be empty".to_string());
        }
        
        if input.len() > 100 {
            return Err("Input too long".to_string());
        }
        
        let processed = input.to_uppercase();
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Processes data with fallback.
    pub fn process_data_with_fallback(&mut self, input: &str) -> Result<usize, String> {
        match self.process_data_internal(input) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fallback to default processing
                self.data.push("default".to_string());
                Ok(self.data.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_custom_error_demo() {
        let mut demo = CustomErrorDemo::new();
        
        let result = demo.process_data("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = demo.process_data("");
        assert!(result.is_err());
        match result.unwrap_err() {
            DataProcessingError::Validation(msg) => assert_eq!(msg, "Input cannot be empty"),
            _ => panic!("Expected Validation error"),
        }
    }
    
    #[test]
    fn test_business_error_handling() {
        let demo = CustomErrorDemo::new();
        
        let result = demo.process_business_data(1001, "hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "HELLO");
        
        let result = demo.process_business_data(0, "hello");
        assert!(result.is_err());
        match result.unwrap_err() {
            BusinessError::UserNotFound { user_id } => assert_eq!(user_id, 0),
            _ => panic!("Expected UserNotFound error"),
        }
    }
    
    #[test]
    fn test_api_error_handling() {
        let demo = CustomErrorDemo::new();
        
        let result = demo.process_api_data("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "HELLO");
        
        let result = demo.process_api_data("");
        assert!(result.is_err());
        match result.unwrap_err() {
            ApiError::BadRequest(msg) => assert_eq!(msg, "Data cannot be empty"),
            _ => panic!("Expected BadRequest error"),
        }
    }
    
    #[test]
    fn test_error_recovery() {
        let mut demo = ErrorRecoveryDemo::new(3);
        
        let result = demo.process_data_with_retry("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = demo.process_data_with_fallback("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }
}
```

### Error Propagation Patterns

```rust
// rust/03-error-propagation.rs

/*
Error propagation patterns and best practices
*/

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use anyhow::{Context, Result as AnyhowResult};
use thiserror::Error;

/// Custom error type for file operations.
#[derive(Error, Debug)]
pub enum FileError {
    #[error("File not found: {path}")]
    NotFound { path: String },
    
    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },
    
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
}

/// Demonstrates error propagation with ? operator.
pub struct ErrorPropagationDemo {
    data: Vec<String>,
}

impl ErrorPropagationDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes file with error propagation.
    pub fn process_file(&mut self, filename: &str) -> Result<usize, FileError> {
        let content = self.read_file(filename)?;
        let processed = self.process_content(&content)?;
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Reads file with error propagation.
    fn read_file(&self, filename: &str) -> Result<String, FileError> {
        let path = Path::new(filename);
        
        if !path.exists() {
            return Err(FileError::NotFound {
                path: filename.to_string(),
            });
        }
        
        let mut file = File::open(filename)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }
    
    /// Processes content with error propagation.
    fn process_content(&self, content: &str) -> Result<String, FileError> {
        if content.is_empty() {
            return Err(FileError::InvalidFormat("Content is empty".to_string()));
        }
        
        if content.len() > 10000 {
            return Err(FileError::InvalidFormat("Content too long".to_string()));
        }
        
        Ok(content.to_uppercase())
    }
    
    /// Processes multiple files with error propagation.
    pub fn process_multiple_files(&mut self, filenames: &[&str]) -> Result<Vec<usize>, FileError> {
        let mut results = Vec::new();
        
        for filename in filenames {
            let result = self.process_file(filename)?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Demonstrates error propagation with anyhow.
pub struct AnyhowErrorDemo {
    data: Vec<String>,
}

impl AnyhowErrorDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes data with anyhow error handling.
    pub fn process_data(&mut self, input: &str) -> AnyhowResult<usize> {
        let processed = self.validate_input(input)
            .context("Failed to validate input")?;
        
        let result = self.process_content(&processed)
            .context("Failed to process content")?;
        
        self.data.push(result);
        Ok(self.data.len())
    }
    
    /// Validates input with context.
    fn validate_input(&self, input: &str) -> AnyhowResult<String> {
        if input.is_empty() {
            anyhow::bail!("Input cannot be empty");
        }
        
        if input.len() > 100 {
            anyhow::bail!("Input too long: {} characters", input.len());
        }
        
        Ok(input.to_string())
    }
    
    /// Processes content with context.
    fn process_content(&self, content: &str) -> AnyhowResult<String> {
        if content.contains("error") {
            anyhow::bail!("Content contains error keyword");
        }
        
        Ok(content.to_uppercase())
    }
    
    /// Processes data with error chaining.
    pub fn process_data_with_chain(&mut self, input: &str) -> AnyhowResult<usize> {
        let processed = self.process_data(input)?;
        Ok(processed)
    }
}

/// Demonstrates error propagation with custom error types.
pub struct CustomErrorPropagationDemo {
    data: Vec<String>,
}

impl CustomErrorPropagationDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes data with custom error propagation.
    pub fn process_data(&mut self, input: &str) -> Result<usize, FileError> {
        let validated = self.validate_input(input)?;
        let processed = self.process_content(&validated)?;
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Validates input with custom error.
    fn validate_input(&self, input: &str) -> Result<String, FileError> {
        if input.is_empty() {
            return Err(FileError::InvalidFormat("Input cannot be empty".to_string()));
        }
        
        if input.len() > 100 {
            return Err(FileError::InvalidFormat("Input too long".to_string()));
        }
        
        Ok(input.to_string())
    }
    
    /// Processes content with custom error.
    fn process_content(&self, content: &str) -> Result<String, FileError> {
        if content.contains("error") {
            return Err(FileError::InvalidFormat("Content contains error keyword".to_string()));
        }
        
        Ok(content.to_uppercase())
    }
    
    /// Processes data with error mapping.
    pub fn process_data_with_mapping(&mut self, input: &str) -> Result<usize, String> {
        self.process_data(input)
            .map_err(|e| format!("Failed to process data: {}", e))
    }
}

/// Demonstrates error propagation with async code.
pub struct AsyncErrorPropagationDemo {
    data: Vec<String>,
}

impl AsyncErrorPropagationDemo {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Processes data asynchronously with error propagation.
    pub async fn process_data_async(&mut self, input: &str) -> Result<usize, FileError> {
        let validated = self.validate_input_async(input).await?;
        let processed = self.process_content_async(&validated).await?;
        self.data.push(processed);
        Ok(self.data.len())
    }
    
    /// Validates input asynchronously.
    async fn validate_input_async(&self, input: &str) -> Result<String, FileError> {
        // Simulate async work
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        if input.is_empty() {
            return Err(FileError::InvalidFormat("Input cannot be empty".to_string()));
        }
        
        if input.len() > 100 {
            return Err(FileError::InvalidFormat("Input too long".to_string()));
        }
        
        Ok(input.to_string())
    }
    
    /// Processes content asynchronously.
    async fn process_content_async(&self, content: &str) -> Result<String, FileError> {
        // Simulate async work
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        if content.contains("error") {
            return Err(FileError::InvalidFormat("Content contains error keyword".to_string()));
        }
        
        Ok(content.to_uppercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_propagation_demo() {
        let mut demo = ErrorPropagationDemo::new();
        
        // This will fail because the file doesn't exist
        let result = demo.process_file("nonexistent.txt");
        assert!(result.is_err());
        match result.unwrap_err() {
            FileError::NotFound { path } => assert_eq!(path, "nonexistent.txt"),
            _ => panic!("Expected NotFound error"),
        }
    }
    
    #[test]
    fn test_anyhow_error_demo() {
        let mut demo = AnyhowErrorDemo::new();
        
        let result = demo.process_data("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = demo.process_data("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Input cannot be empty"));
    }
    
    #[test]
    fn test_custom_error_propagation_demo() {
        let mut demo = CustomErrorPropagationDemo::new();
        
        let result = demo.process_data("hello");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = demo.process_data("");
        assert!(result.is_err());
        match result.unwrap_err() {
            FileError::InvalidFormat(msg) => assert_eq!(msg, "Input cannot be empty"),
            _ => panic!("Expected InvalidFormat error"),
        }
    }
    
    #[tokio::test]
    async fn test_async_error_propagation_demo() {
        let mut demo = AsyncErrorPropagationDemo::new();
        
        let result = demo.process_data_async("hello").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let result = demo.process_data_async("").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            FileError::InvalidFormat(msg) => assert_eq!(msg, "Input cannot be empty"),
            _ => panic!("Expected InvalidFormat error"),
        }
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic error handling
use std::fs::File;
use std::io::{self, Read};

fn read_file(filename: &str) -> Result<String, io::Error> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// 2. Custom error types
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("Validation failed: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

// 3. Error propagation
fn process_data(input: &str) -> Result<String, MyError> {
    let validated = validate_input(input)?;
    let processed = process_content(&validated)?;
    Ok(processed)
}

// 4. Error context
use anyhow::{Context, Result};

fn process_file(filename: &str) -> Result<String> {
    let content = std::fs::read_to_string(filename)
        .context("Failed to read file")?;
    Ok(content)
}
```

### Essential Patterns

```rust
// Complete error handling setup
pub fn setup_rust_error_handling() {
    // 1. Result and Option patterns
    // 2. Custom error types
    // 3. Error propagation
    // 4. Error context
    // 5. Error recovery
    // 6. Error mapping
    // 7. Error chaining
    // 8. Error testing
    
    println!("Rust error handling setup complete!");
}
```

---

*This guide provides the complete machinery for Rust error handling. Each pattern includes implementation examples, error strategies, and real-world usage patterns for enterprise error management.*
