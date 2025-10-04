# Rust Memory Management Best Practices

**Objective**: Master senior-level Rust memory management patterns for production systems. When you need to optimize memory usage, when you want to understand Rust's memory model, when you need enterprise-grade memory strategies—these best practices become your weapon of choice.

## Core Principles

- **Zero-Cost Abstractions**: Memory management with no runtime overhead
- **Ownership System**: Leverage Rust's ownership for memory safety
- **Smart Pointers**: Use appropriate smart pointers for different scenarios
- **Memory Layout**: Optimize data structures for cache performance
- **Resource Management**: Proper cleanup and resource management

## Memory Allocation Patterns

### Smart Pointers

```rust
// rust/01-smart-pointers.rs

/*
Smart pointer patterns and best practices for Rust
*/

use std::rc::Rc;
use std::sync::Arc;
use std::cell::RefCell;
use std::sync::Mutex;

/// Demonstrates Rc (Reference Counted) patterns.
pub struct RcDemo {
    data: Rc<String>,
    count: usize,
}

impl RcDemo {
    pub fn new(data: String) -> Self {
        Self {
            data: Rc::new(data),
            count: 1,
        }
    }
    
    pub fn clone_data(&self) -> Rc<String> {
        Rc::clone(&self.data)
    }
    
    pub fn get_strong_count(&self) -> usize {
        Rc::strong_count(&self.data)
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
}

/// Demonstrates Arc (Atomically Reference Counted) patterns.
pub struct ArcDemo {
    data: Arc<String>,
    count: usize,
}

impl ArcDemo {
    pub fn new(data: String) -> Self {
        Self {
            data: Arc::new(data),
            count: 1,
        }
    }
    
    pub fn clone_data(&self) -> Arc<String> {
        Arc::clone(&self.data)
    }
    
    pub fn get_strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
}

/// Demonstrates RefCell patterns for interior mutability.
pub struct RefCellDemo {
    data: RefCell<String>,
    count: usize,
}

impl RefCellDemo {
    pub fn new(data: String) -> Self {
        Self {
            data: RefCell::new(data),
            count: 1,
        }
    }
    
    pub fn get_data(&self) -> String {
        self.data.borrow().clone()
    }
    
    pub fn set_data(&self, new_data: String) {
        *self.data.borrow_mut() = new_data;
    }
    
    pub fn update_data<F>(&self, f: F)
    where
        F: FnOnce(&mut String),
    {
        f(&mut self.data.borrow_mut());
    }
}

/// Demonstrates Mutex patterns for thread-safe interior mutability.
pub struct MutexDemo {
    data: Arc<Mutex<String>>,
    count: usize,
}

impl MutexDemo {
    pub fn new(data: String) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            count: 1,
        }
    }
    
    pub fn get_data(&self) -> String {
        self.data.lock().unwrap().clone()
    }
    
    pub fn set_data(&self, new_data: String) {
        *self.data.lock().unwrap() = new_data;
    }
    
    pub fn update_data<F>(&self, f: F)
    where
        F: FnOnce(&mut String),
    {
        f(&mut self.data.lock().unwrap());
    }
}

/// Demonstrates Box patterns for heap allocation.
pub struct BoxDemo {
    data: Box<String>,
    count: usize,
}

impl BoxDemo {
    pub fn new(data: String) -> Self {
        Self {
            data: Box::new(data),
            count: 1,
        }
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
    
    pub fn get_data_mut(&mut self) -> &mut String {
        &mut self.data
    }
    
    pub fn into_data(self) -> String {
        *self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rc_demo() {
        let demo = RcDemo::new("Hello".to_string());
        assert_eq!(demo.get_strong_count(), 1);
        
        let cloned = demo.clone_data();
        assert_eq!(demo.get_strong_count(), 2);
        assert_eq!(cloned, "Hello");
    }
    
    #[test]
    fn test_arc_demo() {
        let demo = ArcDemo::new("Hello".to_string());
        assert_eq!(demo.get_strong_count(), 1);
        
        let cloned = demo.clone_data();
        assert_eq!(demo.get_strong_count(), 2);
        assert_eq!(cloned, "Hello");
    }
    
    #[test]
    fn test_refcell_demo() {
        let demo = RefCellDemo::new("Hello".to_string());
        assert_eq!(demo.get_data(), "Hello");
        
        demo.set_data("World".to_string());
        assert_eq!(demo.get_data(), "World");
        
        demo.update_data(|s| s.push_str("!"));
        assert_eq!(demo.get_data(), "World!");
    }
    
    #[test]
    fn test_mutex_demo() {
        let demo = MutexDemo::new("Hello".to_string());
        assert_eq!(demo.get_data(), "Hello");
        
        demo.set_data("World".to_string());
        assert_eq!(demo.get_data(), "World");
        
        demo.update_data(|s| s.push_str("!"));
        assert_eq!(demo.get_data(), "World!");
    }
    
    #[test]
    fn test_box_demo() {
        let mut demo = BoxDemo::new("Hello".to_string());
        assert_eq!(demo.get_data(), "Hello");
        
        *demo.get_data_mut() = "World".to_string();
        assert_eq!(demo.get_data(), "World");
        
        let data = demo.into_data();
        assert_eq!(data, "World");
    }
}
```

### Memory Layout Optimization

```rust
// rust/02-memory-layout.rs

/*
Memory layout optimization patterns and best practices
*/

use std::mem;

/// Demonstrates SoA (Structure of Arrays) pattern.
pub struct SoAData {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    mass: Vec<f32>,
}

impl SoAData {
    pub fn new(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            mass: Vec::with_capacity(capacity),
        }
    }
    
    pub fn add_particle(&mut self, x: f32, y: f32, z: f32, mass: f32) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.mass.push(mass);
    }
    
    pub fn compute_center_of_mass(&self) -> (f32, f32, f32) {
        let mut total_mass = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut weighted_z = 0.0;
        
        for i in 0..self.x.len() {
            let mass = self.mass[i];
            total_mass += mass;
            weighted_x += self.x[i] * mass;
            weighted_y += self.y[i] * mass;
            weighted_z += self.z[i] * mass;
        }
        
        if total_mass > 0.0 {
            (
                weighted_x / total_mass,
                weighted_y / total_mass,
                weighted_z / total_mass,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

/// Demonstrates AoS (Array of Structures) pattern.
pub struct AoSData {
    particles: Vec<Particle>,
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub mass: f32,
}

impl AoSData {
    pub fn new(capacity: usize) -> Self {
        Self {
            particles: Vec::with_capacity(capacity),
        }
    }
    
    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
    }
    
    pub fn compute_center_of_mass(&self) -> (f32, f32, f32) {
        let mut total_mass = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut weighted_z = 0.0;
        
        for particle in &self.particles {
            total_mass += particle.mass;
            weighted_x += particle.x * particle.mass;
            weighted_y += particle.y * particle.mass;
            weighted_z += particle.z * particle.mass;
        }
        
        if total_mass > 0.0 {
            (
                weighted_x / total_mass,
                weighted_y / total_mass,
                weighted_z / total_mass,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

/// Demonstrates memory-efficient data structures.
pub struct MemoryEfficientData {
    data: Vec<u8>,
    element_size: usize,
}

impl MemoryEfficientData {
    pub fn new(element_size: usize, capacity: usize) -> Self {
        Self {
            data: vec![0; element_size * capacity],
            element_size,
        }
    }
    
    pub fn set_element(&mut self, index: usize, element: &[u8]) -> Result<(), &'static str> {
        if element.len() != self.element_size {
            return Err("Element size mismatch");
        }
        
        if index * self.element_size + self.element_size > self.data.len() {
            return Err("Index out of bounds");
        }
        
        let start = index * self.element_size;
        self.data[start..start + self.element_size].copy_from_slice(element);
        Ok(())
    }
    
    pub fn get_element(&self, index: usize) -> Result<&[u8], &'static str> {
        if index * self.element_size + self.element_size > self.data.len() {
            return Err("Index out of bounds");
        }
        
        let start = index * self.element_size;
        Ok(&self.data[start..start + self.element_size])
    }
}

/// Demonstrates memory pool pattern.
pub struct MemoryPool<T> {
    pool: Vec<T>,
    available: Vec<usize>,
    next_id: usize,
}

impl<T> MemoryPool<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: Vec::with_capacity(capacity),
            available: Vec::new(),
            next_id: 0,
        }
    }
    
    pub fn allocate(&mut self, item: T) -> Result<usize, T> {
        if let Some(index) = self.available.pop() {
            self.pool[index] = item;
            Ok(index)
        } else if self.pool.len() < self.pool.capacity() {
            let index = self.pool.len();
            self.pool.push(item);
            Ok(index)
        } else {
            Err(item)
        }
    }
    
    pub fn deallocate(&mut self, index: usize) -> Result<T, &'static str> {
        if index >= self.pool.len() {
            return Err("Index out of bounds");
        }
        
        let item = std::mem::replace(&mut self.pool[index], unsafe { std::mem::uninitialized() });
        self.available.push(index);
        Ok(item)
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        self.pool.get(index)
    }
    
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.pool.get_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soa_data() {
        let mut soa = SoAData::new(100);
        soa.add_particle(1.0, 2.0, 3.0, 1.0);
        soa.add_particle(4.0, 5.0, 6.0, 2.0);
        
        let (x, y, z) = soa.compute_center_of_mass();
        assert!((x - 3.0).abs() < 1e-6);
        assert!((y - 4.0).abs() < 1e-6);
        assert!((z - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_aos_data() {
        let mut aos = AoSData::new(100);
        aos.add_particle(Particle { x: 1.0, y: 2.0, z: 3.0, mass: 1.0 });
        aos.add_particle(Particle { x: 4.0, y: 5.0, z: 6.0, mass: 2.0 });
        
        let (x, y, z) = aos.compute_center_of_mass();
        assert!((x - 3.0).abs() < 1e-6);
        assert!((y - 4.0).abs() < 1e-6);
        assert!((z - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_memory_efficient_data() {
        let mut data = MemoryEfficientData::new(4, 10);
        let element = [1, 2, 3, 4];
        data.set_element(0, &element).unwrap();
        
        let retrieved = data.get_element(0).unwrap();
        assert_eq!(retrieved, &element);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(10);
        
        let index1 = pool.allocate("Hello".to_string()).unwrap();
        let index2 = pool.allocate("World".to_string()).unwrap();
        
        assert_eq!(pool.get(index1), Some(&"Hello".to_string()));
        assert_eq!(pool.get(index2), Some(&"World".to_string()));
        
        let item = pool.deallocate(index1).unwrap();
        assert_eq!(item, "Hello");
    }
}
```

### Garbage Collection Patterns

```rust
// rust/03-gc-patterns.rs

/*
Garbage collection patterns and best practices
*/

use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::collections::HashMap;

/// Demonstrates weak reference patterns.
pub struct WeakRefDemo {
    data: Rc<String>,
    weak_ref: Weak<String>,
}

impl WeakRefDemo {
    pub fn new(data: String) -> Self {
        let rc = Rc::new(data);
        let weak = Rc::downgrade(&rc);
        Self {
            data: rc,
            weak_ref: weak,
        }
    }
    
    pub fn get_strong_count(&self) -> usize {
        Rc::strong_count(&self.data)
    }
    
    pub fn get_weak_count(&self) -> usize {
        Rc::weak_count(&self.data)
    }
    
    pub fn try_get_weak(&self) -> Option<Rc<String>> {
        self.weak_ref.upgrade()
    }
}

/// Demonstrates circular reference patterns.
pub struct Node {
    pub data: String,
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    pub fn new(data: String) -> Self {
        Self {
            data,
            parent: None,
            children: Vec::new(),
        }
    }
    
    pub fn add_child(&mut self, child: Rc<RefCell<Node>>) {
        child.borrow_mut().parent = Some(Rc::downgrade(&Rc::new(RefCell::new(self.clone()))));
        self.children.push(child);
    }
    
    pub fn get_parent(&self) -> Option<Rc<RefCell<Node>>> {
        self.parent.as_ref().and_then(|p| p.upgrade())
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            parent: self.parent.clone(),
            children: self.children.clone(),
        }
    }
}

/// Demonstrates reference counting with cleanup.
pub struct RefCountedResource {
    data: String,
    cleanup_callback: Option<Box<dyn Fn()>>,
}

impl RefCountedResource {
    pub fn new(data: String, cleanup_callback: Box<dyn Fn()>) -> Self {
        Self {
            data,
            cleanup_callback: Some(cleanup_callback),
        }
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
}

impl Drop for RefCountedResource {
    fn drop(&mut self) {
        if let Some(callback) = self.cleanup_callback.take() {
            callback();
        }
    }
}

/// Demonstrates memory-efficient string interning.
pub struct StringInterner {
    strings: Vec<String>,
    indices: HashMap<String, usize>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            indices: HashMap::new(),
        }
    }
    
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&index) = self.indices.get(s) {
            index
        } else {
            let index = self.strings.len();
            self.strings.push(s.to_string());
            self.indices.insert(s.to_string(), index);
            index
        }
    }
    
    pub fn get(&self, index: usize) -> Option<&str> {
        self.strings.get(index).map(|s| s.as_str())
    }
    
    pub fn len(&self) -> usize {
        self.strings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weak_ref_demo() {
        let demo = WeakRefDemo::new("Hello".to_string());
        assert_eq!(demo.get_strong_count(), 1);
        assert_eq!(demo.get_weak_count(), 1);
        
        let strong = demo.try_get_weak().unwrap();
        assert_eq!(*strong, "Hello");
    }
    
    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();
        
        let id1 = interner.intern("Hello");
        let id2 = interner.intern("World");
        let id3 = interner.intern("Hello"); // Should return same ID
        
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.get(id1), Some("Hello"));
        assert_eq!(interner.get(id2), Some("World"));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Smart pointers
use std::rc::Rc;
use std::sync::Arc;
use std::cell::RefCell;

let rc = Rc::new("Hello".to_string());
let arc = Arc::new("World".to_string());
let refcell = RefCell::new(42);

// 2. Memory layout optimization
struct SoAData {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
}

// 3. Memory pools
struct MemoryPool<T> {
    pool: Vec<T>,
    available: Vec<usize>,
}

// 4. Weak references
use std::rc::{Rc, Weak};

let rc = Rc::new("Hello".to_string());
let weak = Rc::downgrade(&rc);
```

### Essential Patterns

```rust
// Complete memory management setup
pub fn setup_rust_memory_management() {
    // 1. Smart pointers
    // 2. Memory layout optimization
    // 3. Garbage collection patterns
    // 4. Memory pools
    // 5. String interning
    // 6. Weak references
    // 7. Circular references
    // 8. Resource cleanup
    
    println!("Rust memory management setup complete!");
}
```

---

*This guide provides the complete machinery for Rust memory management. Each pattern includes implementation examples, memory strategies, and real-world usage patterns for enterprise memory optimization.*
