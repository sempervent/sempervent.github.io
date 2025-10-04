# Rust Ownership & Borrowing Best Practices

**Objective**: Master senior-level Rust ownership and borrowing patterns for production systems. When you need to understand Rust's memory safety guarantees, when you want to write efficient and safe code, when you need enterprise-grade ownership patterns—these best practices become your weapon of choice.

## Core Principles

- **Ownership Rules**: Each value has exactly one owner
- **Borrowing**: References allow temporary access without ownership transfer
- **Lifetimes**: Ensure references remain valid for their entire scope
- **Move Semantics**: Ownership transfer is explicit and efficient
- **Zero-Cost Abstractions**: Ownership system has no runtime overhead

## Ownership Patterns

### Basic Ownership

```rust
// rust/01-basic-ownership.rs

/*
Basic ownership patterns and best practices for Rust
*/

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// A data structure that demonstrates ownership patterns.
pub struct DataContainer {
    data: Vec<String>,
    metadata: HashMap<String, String>,
}

impl DataContainer {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Takes ownership of a string and stores it.
    pub fn add_data(&mut self, item: String) {
        self.data.push(item);
    }
    
    /// Returns ownership of data at the specified index.
    pub fn take_data(&mut self, index: usize) -> Option<String> {
        if index < self.data.len() {
            Some(self.data.remove(index))
        } else {
            None
        }
    }
    
    /// Borrows data without taking ownership.
    pub fn get_data(&self, index: usize) -> Option<&String> {
        self.data.get(index)
    }
    
    /// Borrows data mutably without taking ownership.
    pub fn get_data_mut(&mut self, index: usize) -> Option<&mut String> {
        self.data.get_mut(index)
    }
    
    /// Returns the number of items.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns true if the container is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Clears all data, taking ownership of the cleared items.
    pub fn clear(&mut self) -> Vec<String> {
        std::mem::take(&mut self.data)
    }
    
    /// Adds metadata with ownership transfer.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Gets metadata by borrowing.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

impl Default for DataContainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Demonstrates ownership transfer patterns.
pub struct OwnershipDemo {
    data: Vec<i32>,
}

impl OwnershipDemo {
    pub fn new() -> Self {
        Self {
            data: vec![1, 2, 3, 4, 5],
        }
    }
    
    /// Moves ownership of the entire structure.
    pub fn move_ownership(self) -> Vec<i32> {
        self.data
    }
    
    /// Clones data instead of moving.
    pub fn clone_data(&self) -> Vec<i32> {
        self.data.clone()
    }
    
    /// Borrows data immutably.
    pub fn borrow_data(&self) -> &Vec<i32> {
        &self.data
    }
    
    /// Borrows data mutably.
    pub fn borrow_data_mut(&mut self) -> &mut Vec<i32> {
        &mut self.data
    }
    
    /// Demonstrates partial move.
    pub fn partial_move(self) -> (Vec<i32>, usize) {
        let len = self.data.len();
        (self.data, len)
    }
}

/// Demonstrates reference counting patterns.
pub struct ReferenceCountedData {
    data: Rc<String>,
    count: usize,
}

impl ReferenceCountedData {
    pub fn new(data: String) -> Self {
        Self {
            data: Rc::new(data),
            count: 1,
        }
    }
    
    pub fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            count: self.count + 1,
        }
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
    
    pub fn get_count(&self) -> usize {
        Rc::strong_count(&self.data)
    }
}

/// Demonstrates atomic reference counting for thread safety.
pub struct AtomicReferenceCountedData {
    data: Arc<String>,
    count: usize,
}

impl AtomicReferenceCountedData {
    pub fn new(data: String) -> Self {
        Self {
            data: Arc::new(data),
            count: 1,
        }
    }
    
    pub fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            count: self.count + 1,
        }
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
    
    pub fn get_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_container() {
        let mut container = DataContainer::new();
        assert!(container.is_empty());
        
        container.add_data("Hello".to_string());
        container.add_data("World".to_string());
        assert_eq!(container.len(), 2);
        
        assert_eq!(container.get_data(0), Some(&"Hello".to_string()));
        assert_eq!(container.get_data(1), Some(&"World".to_string()));
        
        let taken = container.take_data(0);
        assert_eq!(taken, Some("Hello".to_string()));
        assert_eq!(container.len(), 1);
    }
    
    #[test]
    fn test_ownership_demo() {
        let demo = OwnershipDemo::new();
        let data = demo.move_ownership();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_reference_counted_data() {
        let data = ReferenceCountedData::new("Hello".to_string());
        assert_eq!(data.get_count(), 1);
        
        let cloned = data.clone();
        assert_eq!(data.get_count(), 2);
        assert_eq!(cloned.get_count(), 2);
        
        assert_eq!(data.get_data(), "Hello");
        assert_eq!(cloned.get_data(), "Hello");
    }
    
    #[test]
    fn test_atomic_reference_counted_data() {
        let data = AtomicReferenceCountedData::new("Hello".to_string());
        assert_eq!(data.get_count(), 1);
        
        let cloned = data.clone();
        assert_eq!(data.get_count(), 2);
        assert_eq!(cloned.get_count(), 2);
        
        assert_eq!(data.get_data(), "Hello");
        assert_eq!(cloned.get_data(), "Hello");
    }
}
```

### Borrowing Patterns

```rust
// rust/02-borrowing-patterns.rs

/*
Borrowing patterns and best practices for Rust
*/

use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;

/// Demonstrates different borrowing patterns.
pub struct BorrowingDemo {
    data: Vec<String>,
    cache: HashMap<String, usize>,
}

impl BorrowingDemo {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            cache: HashMap::new(),
        }
    }
    
    /// Immutable borrowing for reading.
    pub fn get_data(&self, index: usize) -> Option<&String> {
        self.data.get(index)
    }
    
    /// Mutable borrowing for writing.
    pub fn add_data(&mut self, item: String) {
        self.data.push(item);
    }
    
    /// Multiple immutable borrows.
    pub fn get_multiple_data(&self, indices: &[usize]) -> Vec<Option<&String>> {
        indices.iter().map(|&i| self.data.get(i)).collect()
    }
    
    /// Mutable borrowing with lifetime.
    pub fn update_data<F>(&mut self, index: usize, f: F) -> bool
    where
        F: FnOnce(&mut String),
    {
        if let Some(item) = self.data.get_mut(index) {
            f(item);
            true
        } else {
            false
        }
    }
    
    /// Borrowing with caching.
    pub fn get_cached_data(&mut self, key: &str) -> Option<&String> {
        if let Some(&index) = self.cache.get(key) {
            self.data.get(index)
        } else {
            None
        }
    }
    
    /// Borrowing with cache update.
    pub fn add_cached_data(&mut self, key: String, value: String) {
        let index = self.data.len();
        self.data.push(value);
        self.cache.insert(key, index);
    }
}

/// Demonstrates interior mutability patterns.
pub struct InteriorMutabilityDemo {
    data: RefCell<Vec<String>>,
    count: Rc<RefCell<usize>>,
}

impl InteriorMutabilityDemo {
    pub fn new() -> Self {
        Self {
            data: RefCell::new(Vec::new()),
            count: Rc::new(RefCell::new(0)),
        }
    }
    
    /// Interior mutability for reading.
    pub fn get_data(&self, index: usize) -> Option<String> {
        self.data.borrow().get(index).cloned()
    }
    
    /// Interior mutability for writing.
    pub fn add_data(&self, item: String) {
        self.data.borrow_mut().push(item);
        *self.count.borrow_mut() += 1;
    }
    
    /// Shared reference counting with interior mutability.
    pub fn clone_count(&self) -> Rc<RefCell<usize>> {
        Rc::clone(&self.count)
    }
    
    /// Get current count.
    pub fn get_count(&self) -> usize {
        *self.count.borrow()
    }
}

/// Demonstrates borrowing with lifetime parameters.
pub struct LifetimeDemo<'a> {
    data: &'a [String],
    index: usize,
}

impl<'a> LifetimeDemo<'a> {
    pub fn new(data: &'a [String]) -> Self {
        Self { data, index: 0 }
    }
    
    /// Borrowing with explicit lifetime.
    pub fn get_current(&self) -> Option<&'a String> {
        self.data.get(self.index)
    }
    
    /// Mutable borrowing with lifetime.
    pub fn next(&mut self) -> Option<&'a String> {
        if self.index < self.data.len() {
            let result = &self.data[self.index];
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
    
    /// Borrowing with lifetime and bounds checking.
    pub fn get_safe(&self, index: usize) -> Option<&'a String> {
        if index < self.data.len() {
            Some(&self.data[index])
        } else {
            None
        }
    }
}

/// Demonstrates borrowing with trait objects.
pub trait DataProcessor {
    fn process(&self, data: &str) -> String;
    fn get_name(&self) -> &str;
}

pub struct UpperCaseProcessor;

impl DataProcessor for UpperCaseProcessor {
    fn process(&self, data: &str) -> String {
        data.to_uppercase()
    }
    
    fn get_name(&self) -> &str {
        "UpperCase"
    }
}

pub struct LowerCaseProcessor;

impl DataProcessor for LowerCaseProcessor {
    fn process(&self, data: &str) -> String {
        data.to_lowercase()
    }
    
    fn get_name(&self) -> &str {
        "LowerCase"
    }
}

/// Demonstrates borrowing with trait objects.
pub struct ProcessorManager {
    processors: Vec<Box<dyn DataProcessor>>,
}

impl ProcessorManager {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }
    
    pub fn add_processor(&mut self, processor: Box<dyn DataProcessor>) {
        self.processors.push(processor);
    }
    
    pub fn process_all(&self, data: &str) -> Vec<String> {
        self.processors
            .iter()
            .map(|p| p.process(data))
            .collect()
    }
    
    pub fn get_processor_names(&self) -> Vec<&str> {
        self.processors
            .iter()
            .map(|p| p.get_name())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_borrowing_demo() {
        let mut demo = BorrowingDemo::new();
        demo.add_data("Hello".to_string());
        demo.add_data("World".to_string());
        
        assert_eq!(demo.get_data(0), Some(&"Hello".to_string()));
        assert_eq!(demo.get_data(1), Some(&"World".to_string()));
        
        let indices = vec![0, 1];
        let results = demo.get_multiple_data(&indices);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Some(&"Hello".to_string()));
        assert_eq!(results[1], Some(&"World".to_string()));
    }
    
    #[test]
    fn test_interior_mutability() {
        let demo = InteriorMutabilityDemo::new();
        demo.add_data("Hello".to_string());
        demo.add_data("World".to_string());
        
        assert_eq!(demo.get_data(0), Some("Hello".to_string()));
        assert_eq!(demo.get_data(1), Some("World".to_string()));
        assert_eq!(demo.get_count(), 2);
    }
    
    #[test]
    fn test_lifetime_demo() {
        let data = vec!["Hello".to_string(), "World".to_string()];
        let mut demo = LifetimeDemo::new(&data);
        
        assert_eq!(demo.get_current(), Some(&"Hello".to_string()));
        assert_eq!(demo.next(), Some(&"Hello".to_string()));
        assert_eq!(demo.next(), Some(&"World".to_string()));
        assert_eq!(demo.next(), None);
    }
    
    #[test]
    fn test_processor_manager() {
        let mut manager = ProcessorManager::new();
        manager.add_processor(Box::new(UpperCaseProcessor));
        manager.add_processor(Box::new(LowerCaseProcessor));
        
        let results = manager.process_all("Hello");
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"HELLO".to_string()));
        assert!(results.contains(&"hello".to_string()));
        
        let names = manager.get_processor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"UpperCase"));
        assert!(names.contains(&"LowerCase"));
    }
}
```

### Lifetime Management

```rust
// rust/03-lifetime-management.rs

/*
Lifetime management patterns and best practices for Rust
*/

use std::collections::HashMap;
use std::marker::PhantomData;

/// Demonstrates lifetime management with explicit lifetimes.
pub struct LifetimeManager<'a> {
    data: &'a [String],
    cache: HashMap<String, &'a String>,
}

impl<'a> LifetimeManager<'a> {
    pub fn new(data: &'a [String]) -> Self {
        Self {
            data,
            cache: HashMap::new(),
        }
    }
    
    /// Borrowing with explicit lifetime.
    pub fn get_data(&self, index: usize) -> Option<&'a String> {
        self.data.get(index)
    }
    
    /// Caching with lifetime management.
    pub fn get_cached_data(&mut self, key: &str) -> Option<&'a String> {
        if let Some(&value) = self.cache.get(key) {
            Some(value)
        } else {
            let value = self.data.iter().find(|s| s == key)?;
            self.cache.insert(key.to_string(), value);
            Some(value)
        }
    }
    
    /// Borrowing with lifetime and bounds checking.
    pub fn get_safe_data(&self, index: usize) -> Option<&'a String> {
        if index < self.data.len() {
            Some(&self.data[index])
        } else {
            None
        }
    }
}

/// Demonstrates lifetime management with generic parameters.
pub struct GenericLifetimeManager<'a, T> {
    data: &'a [T],
    _phantom: PhantomData<T>,
}

impl<'a, T> GenericLifetimeManager<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }
    
    pub fn get_data(&self, index: usize) -> Option<&'a T> {
        self.data.get(index)
    }
    
    pub fn get_safe_data(&self, index: usize) -> Option<&'a T> {
        if index < self.data.len() {
            Some(&self.data[index])
        } else {
            None
        }
    }
}

/// Demonstrates lifetime management with trait bounds.
pub trait LifetimeTrait<'a> {
    fn get_data(&self) -> &'a str;
    fn get_mutable_data(&mut self) -> &'a mut str;
}

pub struct LifetimeTraitImpl<'a> {
    data: &'a mut String,
}

impl<'a> LifetimeTraitImpl<'a> {
    pub fn new(data: &'a mut String) -> Self {
        Self { data }
    }
}

impl<'a> LifetimeTrait<'a> for LifetimeTraitImpl<'a> {
    fn get_data(&self) -> &'a str {
        self.data
    }
    
    fn get_mutable_data(&mut self) -> &'a mut str {
        self.data
    }
}

/// Demonstrates lifetime management with higher-ranked trait bounds.
pub struct HigherRankedLifetimeManager {
    data: Vec<String>,
}

impl HigherRankedLifetimeManager {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    pub fn add_data(&mut self, item: String) {
        self.data.push(item);
    }
    
    /// Higher-ranked trait bound for lifetime management.
    pub fn process_data<F>(&self, processor: F) -> Vec<String>
    where
        F: for<'a> Fn(&'a str) -> String,
    {
        self.data
            .iter()
            .map(|s| processor(s.as_str()))
            .collect()
    }
    
    /// Borrowing with lifetime management.
    pub fn get_data(&self, index: usize) -> Option<&str> {
        self.data.get(index).map(|s| s.as_str())
    }
}

/// Demonstrates lifetime management with async code.
pub struct AsyncLifetimeManager {
    data: Vec<String>,
}

impl AsyncLifetimeManager {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    pub fn add_data(&mut self, item: String) {
        self.data.push(item);
    }
    
    /// Async function with lifetime management.
    pub async fn process_data_async<F, Fut>(&self, processor: F) -> Vec<String>
    where
        F: Fn(&str) -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        let mut results = Vec::new();
        for item in &self.data {
            let result = processor(item.as_str()).await;
            results.push(result);
        }
        results
    }
    
    /// Borrowing with lifetime management in async context.
    pub async fn get_data_async(&self, index: usize) -> Option<&str> {
        self.data.get(index).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lifetime_manager() {
        let data = vec!["Hello".to_string(), "World".to_string()];
        let mut manager = LifetimeManager::new(&data);
        
        assert_eq!(manager.get_data(0), Some(&"Hello".to_string()));
        assert_eq!(manager.get_data(1), Some(&"World".to_string()));
        
        let cached = manager.get_cached_data("Hello");
        assert_eq!(cached, Some(&"Hello".to_string()));
    }
    
    #[test]
    fn test_generic_lifetime_manager() {
        let data = vec![1, 2, 3, 4, 5];
        let manager = GenericLifetimeManager::new(&data);
        
        assert_eq!(manager.get_data(0), Some(&1));
        assert_eq!(manager.get_data(1), Some(&2));
        assert_eq!(manager.get_safe_data(10), None);
    }
    
    #[test]
    fn test_higher_ranked_lifetime_manager() {
        let mut manager = HigherRankedLifetimeManager::new();
        manager.add_data("Hello".to_string());
        manager.add_data("World".to_string());
        
        let results = manager.process_data(|s| s.to_uppercase());
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"HELLO".to_string()));
        assert!(results.contains(&"WORLD".to_string()));
    }
    
    #[test]
    fn test_async_lifetime_manager() {
        let mut manager = AsyncLifetimeManager::new();
        manager.add_data("Hello".to_string());
        manager.add_data("World".to_string());
        
        let results = futures::executor::block_on(
            manager.process_data_async(|s| async { s.to_uppercase() })
        );
        
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"HELLO".to_string()));
        assert!(results.contains(&"WORLD".to_string()));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic ownership
let data = String::from("Hello");
let moved_data = data; // Ownership transferred
// println!("{}", data); // Error: data moved

// 2. Borrowing
let data = String::from("Hello");
let borrowed = &data; // Immutable borrow
let mutable_borrowed = &mut data; // Mutable borrow

// 3. Lifetimes
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 4. Reference counting
use std::rc::Rc;
let data = Rc::new(String::from("Hello"));
let cloned = Rc::clone(&data);

// 5. Interior mutability
use std::cell::RefCell;
let data = RefCell::new(String::from("Hello"));
*data.borrow_mut() = String::from("World");
```

### Essential Patterns

```rust
// Complete ownership and borrowing setup
pub fn setup_rust_ownership_borrowing() {
    // 1. Ownership rules
    // 2. Borrowing patterns
    // 3. Lifetime management
    // 4. Move semantics
    // 5. Reference counting
    // 6. Interior mutability
    // 7. Zero-cost abstractions
    // 8. Memory safety
    
    println!("Rust ownership and borrowing setup complete!");
}
```

---

*This guide provides the complete machinery for Rust ownership and borrowing. Each pattern includes implementation examples, ownership strategies, and real-world usage patterns for enterprise memory safety.*
