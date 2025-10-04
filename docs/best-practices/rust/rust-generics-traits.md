# Rust Generics & Traits Best Practices

**Objective**: Master senior-level Rust generics and traits patterns for production systems. When you need to build reusable, type-safe code, when you want to leverage Rust's type system, when you need enterprise-grade generic programming—these best practices become your weapon of choice.

## Core Principles

- **Type Safety**: Leverage generics for compile-time type checking
- **Code Reuse**: Write generic code that works with multiple types
- **Trait Bounds**: Use trait bounds to constrain generic types
- **Zero-Cost Abstractions**: Generics have no runtime overhead
- **Composition**: Build complex behavior from simple traits

## Generic Patterns

### Basic Generics

```rust
// rust/01-basic-generics.rs

/*
Basic generic patterns and best practices for Rust
*/

use std::fmt::Display;
use std::cmp::PartialOrd;

/// Generic data structure.
pub struct Container<T> {
    data: T,
}

impl<T> Container<T> {
    pub fn new(data: T) -> Self {
        Self { data }
    }
    
    pub fn get(&self) -> &T {
        &self.data
    }
    
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
    
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Generic function with trait bounds.
pub fn find_max<T>(items: &[T]) -> Option<&T>
where
    T: PartialOrd,
{
    items.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
}

/// Generic function with multiple trait bounds.
pub fn print_and_return<T>(item: T) -> T
where
    T: Display + Clone,
{
    println!("Item: {}", item);
    item
}

/// Generic struct with trait bounds.
pub struct SortedContainer<T>
where
    T: PartialOrd + Clone,
{
    items: Vec<T>,
}

impl<T> SortedContainer<T>
where
    T: PartialOrd + Clone,
{
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
    
    pub fn insert(&mut self, item: T) {
        self.items.push(item);
        self.items.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    
    pub fn get_items(&self) -> &[T] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_container() {
        let container = Container::new(42);
        assert_eq!(*container.get(), 42);
    }
    
    #[test]
    fn test_find_max() {
        let numbers = vec![1, 5, 3, 9, 2];
        assert_eq!(find_max(&numbers), Some(&9));
    }
    
    #[test]
    fn test_sorted_container() {
        let mut container = SortedContainer::new();
        container.insert(3);
        container.insert(1);
        container.insert(2);
        
        let items = container.get_items();
        assert_eq!(items, &[1, 2, 3]);
    }
}
```

### Trait Patterns

```rust
// rust/02-trait-patterns.rs

/*
Trait patterns and best practices for Rust
*/

use std::fmt::Display;

/// Basic trait definition.
pub trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}

/// Trait with default implementation.
pub trait DrawableWithDefault {
    fn draw(&self) {
        println!("Drawing default shape");
    }
    
    fn area(&self) -> f64;
    
    fn perimeter(&self) -> f64 {
        0.0
    }
}

/// Trait with associated types.
pub trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
}

/// Trait with generic parameters.
pub trait Processor<T> {
    fn process(&self, item: T) -> T;
}

/// Trait with lifetime parameters.
pub trait Parser<'a> {
    fn parse(&self, input: &'a str) -> Result<&'a str, &'a str>;
}

/// Trait with multiple bounds.
pub trait ComplexTrait: Display + Clone + PartialEq {
    fn complex_method(&self) -> String;
}

/// Trait with generic associated types.
pub trait Container {
    type Item;
    type Iterator<'a>: Iterator<Item = &'a Self::Item>
    where
        Self: 'a;
    
    fn iter(&self) -> Self::Iterator<'_>;
}

/// Demonstrates trait implementation.
pub struct Circle {
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle with radius {}", self.radius);
    }
    
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}

impl DrawableWithDefault for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    
    fn perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
}

impl Display for Circle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Circle(radius: {})", self.radius)
    }
}

impl Clone for Circle {
    fn clone(&self) -> Self {
        Self { radius: self.radius }
    }
}

impl PartialEq for Circle {
    fn eq(&self, other: &Self) -> bool {
        (self.radius - other.radius).abs() < 1e-10
    }
}

impl ComplexTrait for Circle {
    fn complex_method(&self) -> String {
        format!("Complex circle: {}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circle_drawable() {
        let circle = Circle { radius: 5.0 };
        circle.draw();
        assert!((circle.area() - 78.54).abs() < 1.0);
    }
    
    #[test]
    fn test_circle_display() {
        let circle = Circle { radius: 5.0 };
        let display = format!("{}", circle);
        assert!(display.contains("Circle"));
    }
}
```

### Advanced Generic Patterns

```rust
// rust/03-advanced-generics.rs

/*
Advanced generic patterns and best practices for Rust
*/

use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// Generic function with complex bounds.
pub fn process_items<T, U, F>(items: &[T], processor: F) -> Vec<U>
where
    T: Clone,
    F: Fn(T) -> U,
{
    items.iter().map(|item| processor(item.clone())).collect()
}

/// Generic struct with phantom data.
pub struct PhantomContainer<T> {
    data: Vec<u8>,
    _phantom: PhantomData<T>,
}

impl<T> PhantomContainer<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            _phantom: PhantomData,
        }
    }
    
    pub fn add_data(&mut self, data: u8) {
        self.data.push(data);
    }
    
    pub fn get_data(&self) -> &[u8] {
        &self.data
    }
}

/// Generic trait with associated types.
pub trait Converter {
    type Input;
    type Output;
    
    fn convert(&self, input: Self::Input) -> Self::Output;
}

/// Generic struct implementing Converter.
pub struct StringToIntConverter;

impl Converter for StringToIntConverter {
    type Input = String;
    type Output = i32;
    
    fn convert(&self, input: String) -> i32 {
        input.parse().unwrap_or(0)
    }
}

/// Generic function with trait bounds.
pub fn calculate<T>(a: T, b: T) -> T
where
    T: Add<Output = T> + Mul<Output = T> + Copy,
{
    a * b + a + b
}

/// Generic struct with multiple type parameters.
pub struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    pub fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
    
    pub fn get_first(&self) -> &T {
        &self.first
    }
    
    pub fn get_second(&self) -> &U {
        &self.second
    }
}

/// Generic enum.
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    pub fn is_ok(&self) -> bool {
        matches!(self, Result::Ok(_))
    }
    
    pub fn is_err(&self) -> bool {
        matches!(self, Result::Err(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_items() {
        let numbers = vec![1, 2, 3, 4, 5];
        let doubled: Vec<i32> = process_items(&numbers, |x| x * 2);
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
    }
    
    #[test]
    fn test_phantom_container() {
        let mut container: PhantomContainer<String> = PhantomContainer::new();
        container.add_data(42);
        assert_eq!(container.get_data(), &[42]);
    }
    
    #[test]
    fn test_converter() {
        let converter = StringToIntConverter;
        let result = converter.convert("123".to_string());
        assert_eq!(result, 123);
    }
    
    #[test]
    fn test_calculate() {
        let result = calculate(2, 3);
        assert_eq!(result, 11); // 2*3 + 2 + 3 = 11
    }
    
    #[test]
    fn test_pair() {
        let pair = Pair::new("hello", 42);
        assert_eq!(*pair.get_first(), "hello");
        assert_eq!(*pair.get_second(), 42);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic generics
struct Container<T> {
    data: T,
}

impl<T> Container<T> {
    fn new(data: T) -> Self {
        Self { data }
    }
}

// 2. Trait bounds
fn find_max<T>(items: &[T]) -> Option<&T>
where
    T: PartialOrd,
{
    items.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
}

// 3. Trait definition
trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}

// 4. Trait implementation
impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle");
    }
    
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}
```

### Essential Patterns

```rust
// Complete generics and traits setup
pub fn setup_rust_generics_traits() {
    // 1. Generic functions
    // 2. Generic structs
    // 3. Trait definitions
    // 4. Trait implementations
    // 5. Trait bounds
    // 6. Associated types
    // 7. Generic associated types
    // 8. Phantom data
    
    println!("Rust generics and traits setup complete!");
}
```

---

*This guide provides the complete machinery for Rust generics and traits. Each pattern includes implementation examples, type system strategies, and real-world usage patterns for enterprise generic programming.*
