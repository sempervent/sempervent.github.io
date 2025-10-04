# Rust Performance Tuning Best Practices

**Objective**: Master senior-level Rust performance optimization patterns for production systems. When you need to optimize for speed and memory efficiency, when you want to build high-performance applications, when you need enterprise-grade performance strategies—these best practices become your weapon of choice.

## Core Principles

- **Measure First**: Profile before optimizing
- **Zero-Cost Abstractions**: Leverage Rust's zero-cost abstractions
- **Memory Efficiency**: Minimize allocations
- **Cache Locality**: Optimize for CPU cache performance
- **Parallel Processing**: Utilize multiple cores effectively

## Profiling & Benchmarking

### Performance Measurement

```rust
// rust/01-performance-measurement.rs

/*
Performance measurement and profiling patterns for Rust
*/

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use rayon::prelude::*;

/// A high-performance data processor for benchmarking.
pub struct DataProcessor {
    data: Vec<u64>,
    cache: HashMap<u64, u64>,
}

impl DataProcessor {
    pub fn new(size: usize) -> Self {
        Self {
            data: (0..size as u64).collect(),
            cache: HashMap::new(),
        }
    }
    
    /// Process data with caching for performance measurement.
    pub fn process_with_cache(&mut self) -> u64 {
        let mut sum = 0u64;
        
        for &value in &self.data {
            if let Some(&cached) = self.cache.get(&value) {
                sum += cached;
            } else {
                let processed = self.expensive_calculation(value);
                self.cache.insert(value, processed);
                sum += processed;
            }
        }
        
        sum
    }
    
    /// Process data without caching for comparison.
    pub fn process_without_cache(&self) -> u64 {
        let mut sum = 0u64;
        
        for &value in &self.data {
            sum += self.expensive_calculation(value);
        }
        
        sum
    }
    
    /// Process data in parallel for performance measurement.
    pub fn process_parallel(&self) -> u64 {
        self.data.par_iter()
            .map(|&value| self.expensive_calculation(value))
            .sum()
    }
    
    /// Expensive calculation for benchmarking.
    fn expensive_calculation(&self, value: u64) -> u64 {
        // Simulate expensive computation
        let mut result = value;
        for _ in 0..1000 {
            result = result.wrapping_mul(3).wrapping_add(1);
        }
        result
    }
}

/// Performance measurement utilities.
pub struct PerformanceProfiler {
    measurements: Vec<(String, Duration)>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
    
    /// Measure the execution time of a function.
    pub fn measure<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        self.measurements.push((name.to_string(), duration));
        result
    }
    
    /// Get all measurements.
    pub fn get_measurements(&self) -> &[(String, Duration)] {
        &self.measurements
    }
    
    /// Get the fastest measurement.
    pub fn get_fastest(&self) -> Option<&(String, Duration)> {
        self.measurements.iter().min_by_key(|(_, duration)| *duration)
    }
    
    /// Get the slowest measurement.
    pub fn get_slowest(&self) -> Option<&(String, Duration)> {
        self.measurements.iter().max_by_key(|(_, duration)| *duration)
    }
    
    /// Calculate performance improvement.
    pub fn calculate_improvement(&self, baseline: &str, optimized: &str) -> Option<f64> {
        let baseline_duration = self.measurements.iter()
            .find(|(name, _)| name == baseline)
            .map(|(_, duration)| duration.as_nanos())?;
        
        let optimized_duration = self.measurements.iter()
            .find(|(name, _)| name == optimized)
            .map(|(_, duration)| duration.as_nanos())?;
        
        Some(baseline_duration as f64 / optimized_duration as f64)
    }
}

/// Benchmark suite for performance testing.
pub struct BenchmarkSuite {
    profiler: PerformanceProfiler,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            profiler: PerformanceProfiler::new(),
        }
    }
    
    /// Run comprehensive benchmarks.
    pub fn run_benchmarks(&mut self, data_size: usize) {
        let mut processor = DataProcessor::new(data_size);
        
        // Benchmark without caching
        self.profiler.measure("without_cache", || {
            processor.process_without_cache()
        });
        
        // Benchmark with caching
        self.profiler.measure("with_cache", || {
            processor.process_with_cache()
        });
        
        // Benchmark parallel processing
        self.profiler.measure("parallel", || {
            processor.process_parallel()
        });
        
        // Benchmark with different data sizes
        for size in [100, 1000, 10000] {
            let mut small_processor = DataProcessor::new(size);
            self.profiler.measure(&format!("cached_{}", size), || {
                small_processor.process_with_cache()
            });
        }
    }
    
    /// Print benchmark results.
    pub fn print_results(&self) {
        println!("Benchmark Results:");
        println!("==================");
        
        for (name, duration) in self.profiler.get_measurements() {
            println!("{}: {:?}", name, duration);
        }
        
        if let Some(fastest) = self.profiler.get_fastest() {
            println!("\nFastest: {} ({:?})", fastest.0, fastest.1);
        }
        
        if let Some(slowest) = self.profiler.get_slowest() {
            println!("Slowest: {} ({:?})", slowest.0, slowest.1);
        }
        
        if let Some(improvement) = self.profiler.calculate_improvement("without_cache", "with_cache") {
            println!("Cache improvement: {:.2}x", improvement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_measurement() {
        let mut profiler = PerformanceProfiler::new();
        
        let result = profiler.measure("test", || {
            thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert_eq!(profiler.get_measurements().len(), 1);
    }
    
    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        suite.run_benchmarks(1000);
        
        let measurements = suite.profiler.get_measurements();
        assert!(measurements.len() >= 3);
    }
}
```

### Memory Optimization

```rust
// rust/02-memory-optimization.rs

/*
Memory optimization patterns for Rust applications
*/

use std::collections::HashMap;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Custom memory allocator for tracking allocations.
pub struct TrackingAllocator {
    inner: System,
    allocated: AtomicUsize,
    peak: AtomicUsize,
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = self.allocated.fetch_add(size, Ordering::Relaxed);
            let peak = self.peak.load(Ordering::Relaxed);
            if current + size > peak {
                self.peak.store(current + size, Ordering::Relaxed);
            }
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

impl TrackingAllocator {
    pub fn new() -> Self {
        Self {
            inner: System,
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
    
    pub fn get_allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
    
    pub fn get_peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

/// Memory-efficient data structure using object pooling.
pub struct ObjectPool<T> {
    objects: Vec<T>,
    available: Vec<usize>,
    next_id: usize,
}

impl<T: Default> ObjectPool<T> {
    pub fn new(capacity: usize) -> Self {
        let mut objects = Vec::with_capacity(capacity);
        let mut available = Vec::with_capacity(capacity);
        
        for i in 0..capacity {
            objects.push(T::default());
            available.push(i);
        }
        
        Self {
            objects,
            available,
            next_id: 0,
        }
    }
    
    pub fn acquire(&mut self) -> Option<usize> {
        self.available.pop()
    }
    
    pub fn release(&mut self, id: usize) {
        if id < self.objects.len() {
            self.available.push(id);
        }
    }
    
    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        self.objects.get_mut(id)
    }
    
    pub fn get(&self, id: usize) -> Option<&T> {
        self.objects.get(id)
    }
}

/// Memory-efficient string interning.
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

/// Memory-efficient data structure using small string optimization.
#[derive(Clone)]
pub struct SmallString {
    data: SmallStringData,
}

#[derive(Clone)]
enum SmallStringData {
    Inline([u8; 23]), // 23 bytes for inline storage
    Heap(String),
}

impl SmallString {
    pub fn new(s: &str) -> Self {
        if s.len() <= 23 {
            let mut bytes = [0u8; 23];
            bytes[..s.len()].copy_from_slice(s.as_bytes());
            Self {
                data: SmallStringData::Inline(bytes),
            }
        } else {
            Self {
                data: SmallStringData::Heap(s.to_string()),
            }
        }
    }
    
    pub fn as_str(&self) -> &str {
        match &self.data {
            SmallStringData::Inline(bytes) => {
                let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                unsafe { std::str::from_utf8_unchecked(&bytes[..len]) }
            }
            SmallStringData::Heap(s) => s.as_str(),
        }
    }
    
    pub fn len(&self) -> usize {
        match &self.data {
            SmallStringData::Inline(bytes) => {
                bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())
            }
            SmallStringData::Heap(s) => s.len(),
        }
    }
}

/// Memory-efficient data structure using bit packing.
pub struct BitPackedData {
    data: Vec<u64>,
    bits_per_item: usize,
    items_per_word: usize,
}

impl BitPackedData {
    pub fn new(bits_per_item: usize, capacity: usize) -> Self {
        let items_per_word = 64 / bits_per_item;
        let words_needed = (capacity + items_per_word - 1) / items_per_word;
        
        Self {
            data: vec![0; words_needed],
            bits_per_item,
            items_per_word,
        }
    }
    
    pub fn set(&mut self, index: usize, value: u64) {
        let word_index = index / self.items_per_word;
        let item_index = index % self.items_per_word;
        let bit_offset = item_index * self.bits_per_item;
        
        let mask = (1u64 << self.bits_per_item) - 1;
        let shifted_value = value << bit_offset;
        
        self.data[word_index] &= !(mask << bit_offset);
        self.data[word_index] |= shifted_value;
    }
    
    pub fn get(&self, index: usize) -> u64 {
        let word_index = index / self.items_per_word;
        let item_index = index % self.items_per_word;
        let bit_offset = item_index * self.bits_per_item;
        
        let mask = (1u64 << self.bits_per_item) - 1;
        (self.data[word_index] >> bit_offset) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_object_pool() {
        let mut pool = ObjectPool::<String>::new(10);
        
        let id1 = pool.acquire().unwrap();
        let id2 = pool.acquire().unwrap();
        
        *pool.get_mut(id1).unwrap() = "Hello".to_string();
        *pool.get_mut(id2).unwrap() = "World".to_string();
        
        assert_eq!(pool.get(id1).unwrap(), "Hello");
        assert_eq!(pool.get(id2).unwrap(), "World");
        
        pool.release(id1);
        pool.release(id2);
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
    
    #[test]
    fn test_small_string() {
        let s1 = SmallString::new("Hello");
        let s2 = SmallString::new("This is a very long string that should be stored on the heap");
        
        assert_eq!(s1.as_str(), "Hello");
        assert_eq!(s1.len(), 5);
        assert_eq!(s2.as_str(), "This is a very long string that should be stored on the heap");
        assert_eq!(s2.len(), 58);
    }
    
    #[test]
    fn test_bit_packed_data() {
        let mut data = BitPackedData::new(4, 100); // 4 bits per item
        
        data.set(0, 5);
        data.set(1, 10);
        data.set(2, 15);
        
        assert_eq!(data.get(0), 5);
        assert_eq!(data.get(1), 10);
        assert_eq!(data.get(2), 15);
    }
}
```

### CPU Optimization

```rust
// rust/03-cpu-optimization.rs

/*
CPU optimization patterns for Rust applications
*/

use std::arch::x86_64::*;
use std::mem;
use std::simd::*;

/// SIMD-optimized vector operations.
pub struct SimdVector {
    data: Vec<f32>,
}

impl SimdVector {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }
    
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }
    
    /// SIMD-optimized dot product.
    pub fn dot_product(&self, other: &SimdVector) -> f32 {
        if self.data.len() != other.data.len() {
            panic!("Vector dimensions must match");
        }
        
        let mut sum = 0.0;
        let chunks = self.data.chunks_exact(8);
        let other_chunks = other.data.chunks_exact(8);
        
        for (chunk, other_chunk) in chunks.zip(other_chunks) {
            if chunk.len() == 8 {
                let a = f32x8::from_slice(chunk);
                let b = f32x8::from_slice(other_chunk);
                let product = a * b;
                sum += product.reduce_sum();
            }
        }
        
        // Handle remaining elements
        let remainder = self.data.len() % 8;
        if remainder > 0 {
            let start = self.data.len() - remainder;
            for i in start..self.data.len() {
                sum += self.data[i] * other.data[i];
            }
        }
        
        sum
    }
    
    /// SIMD-optimized vector addition.
    pub fn add(&self, other: &SimdVector) -> SimdVector {
        if self.data.len() != other.data.len() {
            panic!("Vector dimensions must match");
        }
        
        let mut result = vec![0.0; self.data.len()];
        let chunks = self.data.chunks_exact(8);
        let other_chunks = other.data.chunks_exact(8);
        let result_chunks = result.chunks_exact_mut(8);
        
        for ((chunk, other_chunk), result_chunk) in chunks.zip(other_chunks).zip(result_chunks) {
            if chunk.len() == 8 {
                let a = f32x8::from_slice(chunk);
                let b = f32x8::from_slice(other_chunk);
                let sum = a + b;
                result_chunk.copy_from_slice(&sum.to_array());
            }
        }
        
        // Handle remaining elements
        let remainder = self.data.len() % 8;
        if remainder > 0 {
            let start = self.data.len() - remainder;
            for i in start..self.data.len() {
                result[i] = self.data[i] + other.data[i];
            }
        }
        
        SimdVector { data: result }
    }
    
    /// SIMD-optimized vector scaling.
    pub fn scale(&self, factor: f32) -> SimdVector {
        let mut result = vec![0.0; self.data.len()];
        let chunks = self.data.chunks_exact(8);
        let result_chunks = result.chunks_exact_mut(8);
        
        for (chunk, result_chunk) in chunks.zip(result_chunks) {
            if chunk.len() == 8 {
                let a = f32x8::from_slice(chunk);
                let scaled = a * f32x8::splat(factor);
                result_chunk.copy_from_slice(&scaled.to_array());
            }
        }
        
        // Handle remaining elements
        let remainder = self.data.len() % 8;
        if remainder > 0 {
            let start = self.data.len() - remainder;
            for i in start..self.data.len() {
                result[i] = self.data[i] * factor;
            }
        }
        
        SimdVector { data: result }
    }
}

/// Cache-friendly data structure using SoA (Structure of Arrays).
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
    
    /// Cache-friendly computation on all particles.
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
    
    /// Cache-friendly computation with SIMD.
    pub fn compute_center_of_mass_simd(&self) -> (f32, f32, f32) {
        let mut total_mass = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut weighted_z = 0.0;
        
        let chunks = self.x.len() / 8;
        for i in 0..chunks {
            let start = i * 8;
            let x_chunk = f32x8::from_slice(&self.x[start..start + 8]);
            let y_chunk = f32x8::from_slice(&self.y[start..start + 8]);
            let z_chunk = f32x8::from_slice(&self.z[start..start + 8]);
            let mass_chunk = f32x8::from_slice(&self.mass[start..start + 8]);
            
            total_mass += mass_chunk.reduce_sum();
            weighted_x += (x_chunk * mass_chunk).reduce_sum();
            weighted_y += (y_chunk * mass_chunk).reduce_sum();
            weighted_z += (z_chunk * mass_chunk).reduce_sum();
        }
        
        // Handle remaining elements
        let remainder = self.x.len() % 8;
        if remainder > 0 {
            let start = chunks * 8;
            for i in start..self.x.len() {
                let mass = self.mass[i];
                total_mass += mass;
                weighted_x += self.x[i] * mass;
                weighted_y += self.y[i] * mass;
                weighted_z += self.z[i] * mass;
            }
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

/// Branch prediction optimization.
pub struct BranchOptimizedSorter {
    data: Vec<i32>,
}

impl BranchOptimizedSorter {
    pub fn new(data: Vec<i32>) -> Self {
        Self { data }
    }
    
    /// Branch-optimized sorting using bitonic sort.
    pub fn bitonic_sort(&mut self) {
        let n = self.data.len();
        if n <= 1 {
            return;
        }
        
        for k in 2..=n {
            for j in (k / 2)..k {
                for i in 0..n {
                    let l = i ^ j;
                    if l > i {
                        if ((i & k) == 0) == (self.data[i] > self.data[l]) {
                            self.data.swap(i, l);
                        }
                    }
                }
            }
        }
    }
    
    /// Branch-optimized sorting using radix sort.
    pub fn radix_sort(&mut self) {
        let max = self.data.iter().max().copied().unwrap_or(0);
        let mut exp = 1;
        
        while max / exp > 0 {
            self.counting_sort(exp);
            exp *= 10;
        }
    }
    
    fn counting_sort(&mut self, exp: i32) {
        let n = self.data.len();
        let mut output = vec![0; n];
        let mut count = vec![0; 10];
        
        for &item in &self.data {
            count[((item / exp) % 10) as usize] += 1;
        }
        
        for i in 1..10 {
            count[i] += count[i - 1];
        }
        
        for i in (0..n).rev() {
            let index = ((self.data[i] / exp) % 10) as usize;
            output[count[index] - 1] = self.data[i];
            count[index] -= 1;
        }
        
        self.data.copy_from_slice(&output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_vector() {
        let v1 = SimdVector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v2 = SimdVector::from_slice(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        
        let dot = v1.dot_product(&v2);
        assert!((dot - 204.0).abs() < 1e-6);
        
        let sum = v1.add(&v2);
        assert_eq!(sum.data.len(), 8);
    }
    
    #[test]
    fn test_soa_data() {
        let mut data = SoAData::new(100);
        data.add_particle(1.0, 2.0, 3.0, 1.0);
        data.add_particle(4.0, 5.0, 6.0, 2.0);
        
        let (x, y, z) = data.compute_center_of_mass();
        assert!((x - 3.0).abs() < 1e-6);
        assert!((y - 4.0).abs() < 1e-6);
        assert!((z - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_branch_optimized_sorter() {
        let mut sorter = BranchOptimizedSorter::new(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        sorter.bitonic_sort();
        
        let expected = vec![1, 1, 2, 3, 4, 5, 6, 9];
        assert_eq!(sorter.data, expected);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Performance measurement
use std::time::Instant;

let start = Instant::now();
// Your code here
let duration = start.elapsed();

// 2. Memory optimization
use std::collections::HashMap;

let mut cache = HashMap::new();
// Use object pooling and string interning

// 3. CPU optimization
use std::simd::*;

let a = f32x8::from_slice(&data);
let b = f32x8::from_slice(&other_data);
let result = a * b;

// 4. Cache optimization
// Use SoA (Structure of Arrays) instead of AoS (Array of Structures)

// 5. Branch prediction
// Use branchless programming techniques
```

### Essential Patterns

```rust
// Complete performance optimization setup
pub fn setup_rust_performance() {
    // 1. Profiling and benchmarking
    // 2. Memory optimization
    // 3. CPU optimization
    // 4. Cache optimization
    // 5. SIMD programming
    // 6. Branch prediction
    // 7. Parallel processing
    // 8. Zero-cost abstractions
    
    println!("Rust performance optimization setup complete!");
}
```

---

*This guide provides the complete machinery for Rust performance tuning. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise performance optimization.*
