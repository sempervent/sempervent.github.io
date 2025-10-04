# Rust Testing Best Practices

**Objective**: Master senior-level Rust testing patterns for production systems. When you need to build comprehensive test suites, when you want to ensure code reliability, when you need enterprise-grade testing strategies—these best practices become your weapon of choice.

## Core Principles

- **Test Coverage**: Aim for high test coverage with meaningful tests
- **Test Isolation**: Each test should be independent and isolated
- **Property-Based Testing**: Use property-based testing for complex logic
- **Performance Testing**: Include benchmarks and performance tests
- **Integration Testing**: Test complete workflows and interactions

## Unit Testing Patterns

### Basic Unit Testing

```rust
// rust/01-unit-testing.rs

/*
Unit testing patterns and best practices for Rust
*/

use std::collections::HashMap;
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CalculatorError {
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

pub struct Calculator {
    history: Vec<String>,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }
    
    pub fn add(&mut self, a: f64, b: f64) -> f64 {
        let result = a + b;
        self.history.push(format!("{} + {} = {}", a, b, result));
        result
    }
    
    pub fn subtract(&mut self, a: f64, b: f64) -> f64 {
        let result = a - b;
        self.history.push(format!("{} - {} = {}", a, b, result));
        result
    }
    
    pub fn multiply(&mut self, a: f64, b: f64) -> f64 {
        let result = a * b;
        self.history.push(format!("{} * {} = {}", a, b, result));
        result
    }
    
    pub fn divide(&mut self, a: f64, b: f64) -> Result<f64, CalculatorError> {
        if b == 0.0 {
            return Err(CalculatorError::DivisionByZero);
        }
        let result = a / b;
        self.history.push(format!("{} / {} = {}", a, b, result));
        Ok(result)
    }
    
    pub fn get_history(&self) -> &[String] {
        &self.history
    }
    
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_addition() {
        let mut calc = Calculator::new();
        assert_eq!(calc.add(2.0, 3.0), 5.0);
        assert_eq!(calc.add(-1.0, 1.0), 0.0);
        assert_eq!(calc.add(0.0, 0.0), 0.0);
    }
    
    #[test]
    fn test_subtraction() {
        let mut calc = Calculator::new();
        assert_eq!(calc.subtract(5.0, 3.0), 2.0);
        assert_eq!(calc.subtract(1.0, 1.0), 0.0);
        assert_eq!(calc.subtract(0.0, 5.0), -5.0);
    }
    
    #[test]
    fn test_multiplication() {
        let mut calc = Calculator::new();
        assert_eq!(calc.multiply(2.0, 3.0), 6.0);
        assert_eq!(calc.multiply(-2.0, 3.0), -6.0);
        assert_eq!(calc.multiply(0.0, 5.0), 0.0);
    }
    
    #[test]
    fn test_division_success() {
        let mut calc = Calculator::new();
        assert_eq!(calc.divide(6.0, 2.0).unwrap(), 3.0);
        assert_eq!(calc.divide(5.0, 2.0).unwrap(), 2.5);
        assert_eq!(calc.divide(0.0, 5.0).unwrap(), 0.0);
    }
    
    #[test]
    fn test_division_by_zero() {
        let mut calc = Calculator::new();
        let result = calc.divide(5.0, 0.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            CalculatorError::DivisionByZero => {},
            _ => panic!("Expected DivisionByZero error"),
        }
    }
    
    #[test]
    fn test_history() {
        let mut calc = Calculator::new();
        calc.add(1.0, 2.0);
        calc.subtract(5.0, 3.0);
        
        let history = calc.get_history();
        assert_eq!(history.len(), 2);
        assert!(history[0].contains("1 + 2 = 3"));
        assert!(history[1].contains("5 - 3 = 2"));
    }
    
    #[test]
    fn test_clear_history() {
        let mut calc = Calculator::new();
        calc.add(1.0, 2.0);
        assert_eq!(calc.get_history().len(), 1);
        
        calc.clear_history();
        assert_eq!(calc.get_history().len(), 0);
    }
    
    #[test]
    fn test_floating_point_precision() {
        let mut calc = Calculator::new();
        let result = calc.add(0.1, 0.2);
        assert!((result - 0.3).abs() < 1e-10);
    }
}

// Test with custom test harness
#[cfg(test)]
mod custom_tests {
    use super::*;
    
    // Test with setup and teardown
    fn setup_calculator() -> Calculator {
        let mut calc = Calculator::new();
        calc.add(1.0, 2.0); // Pre-populate with some data
        calc
    }
    
    fn teardown_calculator(calc: &mut Calculator) {
        calc.clear_history();
    }
    
    #[test]
    fn test_with_setup_teardown() {
        let mut calc = setup_calculator();
        assert_eq!(calc.get_history().len(), 1);
        
        calc.add(3.0, 4.0);
        assert_eq!(calc.get_history().len(), 2);
        
        teardown_calculator(&mut calc);
        assert_eq!(calc.get_history().len(), 0);
    }
}
```

### Property-Based Testing

```rust
// rust/02-property-based-testing.rs

/*
Property-based testing patterns with proptest
*/

use proptest::prelude::*;
use std::collections::HashMap;

pub struct StringProcessor {
    data: HashMap<String, String>,
}

impl StringProcessor {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    pub fn add(&mut self, key: String, value: String) -> Result<(), String> {
        if key.is_empty() {
            return Err("Key cannot be empty".to_string());
        }
        if key.len() > 100 {
            return Err("Key too long".to_string());
        }
        if value.len() > 1000 {
            return Err("Value too long".to_string());
        }
        
        self.data.insert(key, value);
        Ok(())
    }
    
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
    
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    
    proptest! {
        #[test]
        fn test_add_get_property(
            key in "[a-zA-Z0-9_]{1,50}",
            value in ".*"
        ) {
            let mut processor = StringProcessor::new();
            processor.add(key.clone(), value.clone()).unwrap();
            assert_eq!(processor.get(&key), Some(&value));
        }
        
        #[test]
        fn test_add_remove_property(
            key in "[a-zA-Z0-9_]{1,50}",
            value in ".*"
        ) {
            let mut processor = StringProcessor::new();
            processor.add(key.clone(), value.clone()).unwrap();
            assert_eq!(processor.remove(&key), Some(value));
            assert_eq!(processor.get(&key), None);
        }
        
        #[test]
        fn test_length_property(
            operations in prop::collection::vec(
                (any::<String>(), any::<String>()),
                0..100
            )
        ) {
            let mut processor = StringProcessor::new();
            let mut expected_len = 0;
            
            for (key, value) in operations {
                if !key.is_empty() && key.len() <= 100 && value.len() <= 1000 {
                    if processor.add(key.clone(), value).is_ok() {
                        expected_len += 1;
                    }
                }
            }
            
            assert_eq!(processor.len(), expected_len);
        }
        
        #[test]
        fn test_empty_key_error(
            value in ".*"
        ) {
            let mut processor = StringProcessor::new();
            let result = processor.add("".to_string(), value);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("empty"));
        }
        
        #[test]
        fn test_key_too_long(
            key in "[a-zA-Z0-9_]{101,200}",
            value in ".*"
        ) {
            let mut processor = StringProcessor::new();
            let result = processor.add(key, value);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("too long"));
        }
        
        #[test]
        fn test_value_too_long(
            key in "[a-zA-Z0-9_]{1,50}",
            value in "[a-zA-Z0-9_]{1001,2000}"
        ) {
            let mut processor = StringProcessor::new();
            let result = processor.add(key, value);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("too long"));
        }
    }
}

// Custom property-based test strategies
#[cfg(test)]
mod custom_property_tests {
    use super::*;
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::TestRunner;
    
    // Custom strategy for generating valid keys
    fn valid_key_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9_]{1,50}".prop_map(|s| s)
    }
    
    // Custom strategy for generating valid values
    fn valid_value_strategy() -> impl Strategy<Value = String> {
        ".*".prop_filter("value length", |s| s.len() <= 1000)
    }
    
    proptest! {
        #[test]
        fn test_custom_strategy(
            key in valid_key_strategy(),
            value in valid_value_strategy()
        ) {
            let mut processor = StringProcessor::new();
            assert!(processor.add(key.clone(), value.clone()).is_ok());
            assert_eq!(processor.get(&key), Some(&value));
        }
    }
}
```

### Mock Testing

```rust
// rust/03-mock-testing.rs

/*
Mock testing patterns with mockall
*/

use mockall::mock;
use std::collections::HashMap;
use anyhow::Result;

// Trait to mock
pub trait DataRepository {
    fn get(&self, key: &str) -> Option<String>;
    fn set(&mut self, key: String, value: String) -> Result<()>;
    fn delete(&mut self, key: &str) -> Result<()>;
    fn exists(&self, key: &str) -> bool;
}

// Implementation
pub struct HashMapRepository {
    data: HashMap<String, String>,
}

impl HashMapRepository {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl DataRepository for HashMapRepository {
    fn get(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    
    fn set(&mut self, key: String, value: String) -> Result<()> {
        self.data.insert(key, value);
        Ok(())
    }
    
    fn delete(&mut self, key: &str) -> Result<()> {
        self.data.remove(key);
        Ok(())
    }
    
    fn exists(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

// Service that uses the repository
pub struct DataService {
    repository: Box<dyn DataRepository>,
}

impl DataService {
    pub fn new(repository: Box<dyn DataRepository>) -> Self {
        Self { repository }
    }
    
    pub fn get_data(&self, key: &str) -> Option<String> {
        self.repository.get(key)
    }
    
    pub fn set_data(&mut self, key: String, value: String) -> Result<()> {
        self.repository.set(key, value)
    }
    
    pub fn delete_data(&mut self, key: &str) -> Result<()> {
        self.repository.delete(key)
    }
    
    pub fn data_exists(&self, key: &str) -> bool {
        self.repository.exists(key)
    }
    
    pub fn get_or_set_default(&mut self, key: String, default_value: String) -> Result<String> {
        if let Some(value) = self.repository.get(&key) {
            Ok(value)
        } else {
            self.repository.set(key.clone(), default_value.clone())?;
            Ok(default_value)
        }
    }
}

// Mock the trait
mock! {
    pub DataRepository {}
    
    impl DataRepository for DataRepository {
        fn get(&self, key: &str) -> Option<String>;
        fn set(&mut self, key: String, value: String) -> Result<()>;
        fn delete(&mut self, key: &str) -> Result<()>;
        fn exists(&self, key: &str) -> bool;
    }
}

#[cfg(test)]
mod mock_tests {
    use super::*;
    use mockall::predicate::*;
    
    #[test]
    fn test_get_data_success() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_get()
            .with(eq("test_key"))
            .times(1)
            .returning(|_| Some("test_value".to_string()));
        
        let service = DataService::new(Box::new(mock_repo));
        let result = service.get_data("test_key");
        
        assert_eq!(result, Some("test_value".to_string()));
    }
    
    #[test]
    fn test_get_data_not_found() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_get()
            .with(eq("nonexistent_key"))
            .times(1)
            .returning(|_| None);
        
        let service = DataService::new(Box::new(mock_repo));
        let result = service.get_data("nonexistent_key");
        
        assert_eq!(result, None);
    }
    
    #[test]
    fn test_set_data_success() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_set()
            .with(eq("test_key".to_string()), eq("test_value".to_string()))
            .times(1)
            .returning(|_, _| Ok(()));
        
        let mut service = DataService::new(Box::new(mock_repo));
        let result = service.set_data("test_key".to_string(), "test_value".to_string());
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_delete_data_success() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_delete()
            .with(eq("test_key"))
            .times(1)
            .returning(|_| Ok(()));
        
        let mut service = DataService::new(Box::new(mock_repo));
        let result = service.delete_data("test_key");
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_data_exists_true() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_exists()
            .with(eq("test_key"))
            .times(1)
            .returning(|_| true);
        
        let service = DataService::new(Box::new(mock_repo));
        let result = service.data_exists("test_key");
        
        assert!(result);
    }
    
    #[test]
    fn test_get_or_set_default_existing() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_get()
            .with(eq("existing_key"))
            .times(1)
            .returning(|_| Some("existing_value".to_string()));
        
        let mut service = DataService::new(Box::new(mock_repo));
        let result = service.get_or_set_default("existing_key".to_string(), "default_value".to_string()).unwrap();
        
        assert_eq!(result, "existing_value");
    }
    
    #[test]
    fn test_get_or_set_default_missing() {
        let mut mock_repo = MockDataRepository::new();
        mock_repo
            .expect_get()
            .with(eq("missing_key"))
            .times(1)
            .returning(|_| None);
        mock_repo
            .expect_set()
            .with(eq("missing_key".to_string()), eq("default_value".to_string()))
            .times(1)
            .returning(|_, _| Ok(()));
        
        let mut service = DataService::new(Box::new(mock_repo));
        let result = service.get_or_set_default("missing_key".to_string(), "default_value".to_string()).unwrap();
        
        assert_eq!(result, "default_value");
    }
}
```

### Integration Testing

```rust
// rust/04-integration-testing.rs

/*
Integration testing patterns for Rust applications
*/

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub age: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserUpdate {
    pub name: Option<String>,
    pub email: Option<String>,
    pub age: Option<u32>,
}

pub struct UserService {
    users: HashMap<u64, User>,
    next_id: u64,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn create_user(&mut self, name: String, email: String, age: u32) -> Result<User> {
        if name.is_empty() {
            return Err(anyhow::anyhow!("Name cannot be empty"));
        }
        if email.is_empty() {
            return Err(anyhow::anyhow!("Email cannot be empty"));
        }
        if age > 150 {
            return Err(anyhow::anyhow!("Age must be less than 150"));
        }
        
        let user = User {
            id: self.next_id,
            name,
            email,
            age,
        };
        
        self.users.insert(user.id, user.clone());
        self.next_id += 1;
        
        Ok(user)
    }
    
    pub fn get_user(&self, id: u64) -> Option<&User> {
        self.users.get(&id)
    }
    
    pub fn update_user(&mut self, id: u64, update: UserUpdate) -> Result<User> {
        let user = self.users.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;
        
        if let Some(name) = update.name {
            if name.is_empty() {
                return Err(anyhow::anyhow!("Name cannot be empty"));
            }
            user.name = name;
        }
        
        if let Some(email) = update.email {
            if email.is_empty() {
                return Err(anyhow::anyhow!("Email cannot be empty"));
            }
            user.email = email;
        }
        
        if let Some(age) = update.age {
            if age > 150 {
                return Err(anyhow::anyhow!("Age must be less than 150"));
            }
            user.age = age;
        }
        
        Ok(user.clone())
    }
    
    pub fn delete_user(&mut self, id: u64) -> Result<User> {
        self.users.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("User not found"))
    }
    
    pub fn list_users(&self) -> Vec<&User> {
        self.users.values().collect()
    }
    
    pub fn search_users(&self, query: &str) -> Vec<&User> {
        self.users.values()
            .filter(|user| user.name.contains(query) || user.email.contains(query))
            .collect()
    }
}

// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_user_lifecycle() {
        let mut service = UserService::new();
        
        // Create user
        let user = service.create_user(
            "John Doe".to_string(),
            "john@example.com".to_string(),
            30
        ).unwrap();
        
        assert_eq!(user.id, 1);
        assert_eq!(user.name, "John Doe");
        assert_eq!(user.email, "john@example.com");
        assert_eq!(user.age, 30);
        
        // Get user
        let retrieved_user = service.get_user(user.id).unwrap();
        assert_eq!(retrieved_user.name, "John Doe");
        
        // Update user
        let update = UserUpdate {
            name: Some("John Smith".to_string()),
            email: None,
            age: Some(31),
        };
        
        let updated_user = service.update_user(user.id, update).unwrap();
        assert_eq!(updated_user.name, "John Smith");
        assert_eq!(updated_user.age, 31);
        assert_eq!(updated_user.email, "john@example.com"); // Unchanged
        
        // Search users
        let search_results = service.search_users("John");
        assert_eq!(search_results.len(), 1);
        assert_eq!(search_results[0].name, "John Smith");
        
        // Delete user
        let deleted_user = service.delete_user(user.id).unwrap();
        assert_eq!(deleted_user.name, "John Smith");
        
        // Verify deletion
        assert!(service.get_user(user.id).is_none());
    }
    
    #[test]
    fn test_multiple_users() {
        let mut service = UserService::new();
        
        // Create multiple users
        let user1 = service.create_user("Alice".to_string(), "alice@example.com".to_string(), 25).unwrap();
        let user2 = service.create_user("Bob".to_string(), "bob@example.com".to_string(), 30).unwrap();
        let user3 = service.create_user("Charlie".to_string(), "charlie@example.com".to_string(), 35).unwrap();
        
        // List all users
        let all_users = service.list_users();
        assert_eq!(all_users.len(), 3);
        
        // Search users
        let search_results = service.search_users("example.com");
        assert_eq!(search_results.len(), 3);
        
        // Update one user
        let update = UserUpdate {
            name: Some("Alice Smith".to_string()),
            email: None,
            age: None,
        };
        
        service.update_user(user1.id, update).unwrap();
        
        // Verify update
        let updated_user = service.get_user(user1.id).unwrap();
        assert_eq!(updated_user.name, "Alice Smith");
        
        // Delete one user
        service.delete_user(user2.id).unwrap();
        
        // Verify deletion
        let remaining_users = service.list_users();
        assert_eq!(remaining_users.len(), 2);
        assert!(service.get_user(user2.id).is_none());
    }
    
    #[test]
    fn test_error_handling() {
        let mut service = UserService::new();
        
        // Test empty name
        let result = service.create_user("".to_string(), "test@example.com".to_string(), 25);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Name cannot be empty"));
        
        // Test empty email
        let result = service.create_user("Test".to_string(), "".to_string(), 25);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Email cannot be empty"));
        
        // Test invalid age
        let result = service.create_user("Test".to_string(), "test@example.com".to_string(), 200);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Age must be less than 150"));
        
        // Test updating non-existent user
        let update = UserUpdate {
            name: Some("New Name".to_string()),
            email: None,
            age: None,
        };
        
        let result = service.update_user(999, update);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("User not found"));
        
        // Test deleting non-existent user
        let result = service.delete_user(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("User not found"));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic unit testing
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }
}

// 2. Property-based testing
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_property(
        input in any::<String>()
    ) {
        // Property test implementation
    }
}

// 3. Mock testing
use mockall::mock;

mock! {
    pub MyTrait {}
    
    impl MyTrait for MyTrait {
        fn method(&self) -> String;
    }
}

// 4. Integration testing
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_workflow() {
        // Integration test implementation
    }
}
```

### Essential Patterns

```rust
// Complete testing setup
pub fn setup_rust_testing() {
    // 1. Unit testing
    // 2. Property-based testing
    // 3. Mock testing
    // 4. Integration testing
    // 5. Performance testing
    // 6. Test organization
    // 7. Test data management
    // 8. Test utilities
    
    println!("Rust testing setup complete!");
}
```

---

*This guide provides the complete machinery for Rust testing best practices. Each pattern includes implementation examples, testing strategies, and real-world usage patterns for enterprise test development.*
