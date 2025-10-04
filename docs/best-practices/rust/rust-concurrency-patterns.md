# Rust Concurrency Patterns Best Practices

**Objective**: Master senior-level Rust concurrency patterns for production systems. When you need to build concurrent and parallel applications, when you want to leverage multiple cores effectively, when you need enterprise-grade concurrency strategies—these best practices become your weapon of choice.

## Core Principles

- **Thread Safety**: Leverage Rust's ownership system for thread safety
- **Zero-Cost Abstractions**: Concurrency patterns with minimal overhead
- **Data Race Prevention**: Compile-time prevention of data races
- **Performance**: Efficient parallel processing
- **Scalability**: Patterns that scale with system resources

## Threading Patterns

### Basic Threading

```rust
// rust/01-basic-threading.rs

/*
Basic threading patterns and best practices for Rust
*/

use std::thread;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::mpsc;
use std::time::Duration;

/// A thread-safe counter using Arc and Mutex.
pub struct ThreadSafeCounter {
    count: Arc<Mutex<u32>>,
}

impl ThreadSafeCounter {
    pub fn new() -> Self {
        Self {
            count: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn increment(&self) {
        let mut count = self.count.lock().unwrap();
        *count += 1;
    }
    
    pub fn get_count(&self) -> u32 {
        let count = self.count.lock().unwrap();
        *count
    }
    
    pub fn increment_by(&self, amount: u32) {
        let mut count = self.count.lock().unwrap();
        *count += amount;
    }
}

impl Clone for ThreadSafeCounter {
    fn clone(&self) -> Self {
        Self {
            count: Arc::clone(&self.count),
        }
    }
}

/// A thread-safe data structure using RwLock for read-write access.
pub struct ThreadSafeData<T> {
    data: Arc<RwLock<T>>,
}

impl<T> ThreadSafeData<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }
    
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let data = self.data.read().unwrap();
        f(&data)
    }
    
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut data = self.data.write().unwrap();
        f(&mut data)
    }
}

impl<T> Clone for ThreadSafeData<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

/// Demonstrates thread communication using channels.
pub struct ThreadCommunicator {
    sender: mpsc::Sender<String>,
    receiver: mpsc::Receiver<String>,
}

impl ThreadCommunicator {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self { sender, receiver }
    }
    
    pub fn send(&self, message: String) -> Result<(), mpsc::SendError<String>> {
        self.sender.send(message)
    }
    
    pub fn receive(&self) -> Result<String, mpsc::RecvError> {
        self.receiver.recv()
    }
    
    pub fn try_receive(&self) -> Result<String, mpsc::TryRecvError> {
        self.receiver.try_recv()
    }
}

/// Demonstrates thread pool pattern.
pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            let receiver = Arc::clone(&receiver);
            
            let worker = thread::spawn(move || {
                loop {
                    let job = receiver.lock().unwrap().recv();
                    
                    match job {
                        Ok(job) => {
                            println!("Worker {} got a job; executing.", id);
                            job();
                        }
                        Err(_) => {
                            println!("Worker {} disconnected; shutting down.", id);
                            break;
                        }
                    }
                }
            });
            
            workers.push(worker);
        }
        
        Self { workers, sender }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(&self.sender);
        
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

/// Demonstrates thread-safe queue pattern.
pub struct ThreadSafeQueue<T> {
    queue: Arc<Mutex<Vec<T>>>,
    not_empty: Arc<Mutex<()>>,
}

impl<T> ThreadSafeQueue<T> {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            not_empty: Arc::new(Mutex::new(())),
        }
    }
    
    pub fn push(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push(item);
    }
    
    pub fn pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop()
    }
    
    pub fn len(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.len()
    }
    
    pub fn is_empty(&self) -> bool {
        let queue = self.queue.lock().unwrap();
        queue.is_empty()
    }
}

impl<T> Clone for ThreadSafeQueue<T> {
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
            not_empty: Arc::clone(&self.not_empty),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thread_safe_counter() {
        let counter = ThreadSafeCounter::new();
        counter.increment();
        counter.increment_by(5);
        assert_eq!(counter.get_count(), 6);
    }
    
    #[test]
    fn test_thread_safe_data() {
        let data = ThreadSafeData::new(42);
        let result = data.read(|x| *x);
        assert_eq!(result, 42);
        
        data.write(|x| *x = 100);
        let result = data.read(|x| *x);
        assert_eq!(result, 100);
    }
    
    #[test]
    fn test_thread_communicator() {
        let communicator = ThreadCommunicator::new();
        communicator.send("Hello".to_string()).unwrap();
        let message = communicator.receive().unwrap();
        assert_eq!(message, "Hello");
    }
    
    #[test]
    fn test_thread_safe_queue() {
        let queue = ThreadSafeQueue::new();
        queue.push(1);
        queue.push(2);
        queue.push(3);
        
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(1));
        assert!(queue.is_empty());
    }
}
```

### Async Programming

```rust
// rust/02-async-programming.rs

/*
Async programming patterns and best practices for Rust
*/

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::sleep;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task;

/// Demonstrates async function patterns.
pub struct AsyncProcessor {
    data: Arc<Mutex<Vec<String>>>,
    counter: Arc<Mutex<u32>>,
}

impl AsyncProcessor {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(Vec::new())),
            counter: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Async function that processes data.
    pub async fn process_data(&self, item: String) -> String {
        // Simulate async work
        sleep(Duration::from_millis(100)).await;
        
        let mut data = self.data.lock().await;
        data.push(item.clone());
        
        let mut counter = self.counter.lock().await;
        *counter += 1;
        
        format!("Processed: {} (count: {})", item, *counter)
    }
    
    /// Async function that processes multiple items.
    pub async fn process_multiple(&self, items: Vec<String>) -> Vec<String> {
        let mut handles = Vec::new();
        
        for item in items {
            let processor = Arc::new(self.clone());
            let handle = task::spawn(async move {
                processor.process_data(item).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }
        
        results
    }
    
    /// Async function with error handling.
    pub async fn process_with_error(&self, item: String) -> Result<String, String> {
        if item.is_empty() {
            return Err("Item cannot be empty".to_string());
        }
        
        sleep(Duration::from_millis(50)).await;
        Ok(format!("Processed: {}", item))
    }
}

impl Clone for AsyncProcessor {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            counter: Arc::clone(&self.counter),
        }
    }
}

/// Demonstrates async stream patterns.
pub struct AsyncStream {
    data: Vec<String>,
    index: usize,
}

impl AsyncStream {
    pub fn new(data: Vec<String>) -> Self {
        Self { data, index: 0 }
    }
    
    /// Async iterator pattern.
    pub async fn next(&mut self) -> Option<String> {
        if self.index < self.data.len() {
            let item = self.data[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
    
    /// Async stream processing.
    pub async fn process_stream<F, Fut>(&mut self, processor: F) -> Vec<String>
    where
        F: Fn(String) -> Fut,
        Fut: Future<Output = String>,
    {
        let mut results = Vec::new();
        
        while let Some(item) = self.next().await {
            let result = processor(item).await;
            results.push(result);
        }
        
        results
    }
}

/// Demonstrates async channel patterns.
pub struct AsyncChannel<T> {
    sender: mpsc::UnboundedSender<T>,
    receiver: mpsc::UnboundedReceiver<T>,
}

impl<T> AsyncChannel<T> {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self { sender, receiver }
    }
    
    pub async fn send(&self, item: T) -> Result<(), mpsc::error::SendError<T>> {
        self.sender.send(item)
    }
    
    pub async fn recv(&mut self) -> Option<T> {
        self.receiver.recv().await
    }
    
    pub async fn try_recv(&mut self) -> Result<T, mpsc::error::TryRecvError> {
        self.receiver.try_recv()
    }
}

/// Demonstrates async task patterns.
pub struct AsyncTaskManager {
    tasks: Vec<task::JoinHandle<String>>,
}

impl AsyncTaskManager {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
        }
    }
    
    pub fn spawn_task<F>(&mut self, task_fn: F)
    where
        F: Future<Output = String> + Send + 'static,
    {
        let handle = task::spawn(task_fn);
        self.tasks.push(handle);
    }
    
    pub async fn wait_all(&mut self) -> Vec<String> {
        let mut results = Vec::new();
        
        for task in self.tasks.drain(..) {
            if let Ok(result) = task.await {
                results.push(result);
            }
        }
        
        results
    }
}

/// Demonstrates async timeout patterns.
pub struct AsyncTimeoutManager {
    timeout: Duration,
}

impl AsyncTimeoutManager {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
    
    pub async fn with_timeout<F, T>(&self, future: F) -> Result<T, String>
    where
        F: Future<Output = T>,
    {
        tokio::time::timeout(self.timeout, future)
            .await
            .map_err(|_| "Timeout occurred".to_string())
    }
    
    pub async fn retry_with_timeout<F, T>(&self, mut future: F, max_retries: u32) -> Result<T, String>
    where
        F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, String>> + Send>>,
    {
        for _ in 0..max_retries {
            match self.with_timeout(future()).await {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }
        
        Err("Max retries exceeded".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_processor() {
        let processor = AsyncProcessor::new();
        let result = processor.process_data("Hello".to_string()).await;
        assert!(result.contains("Processed: Hello"));
    }
    
    #[tokio::test]
    async fn test_async_stream() {
        let data = vec!["Hello".to_string(), "World".to_string()];
        let mut stream = AsyncStream::new(data);
        
        let results = stream.process_stream(|item| async {
            format!("Processed: {}", item)
        }).await;
        
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"Processed: Hello".to_string()));
        assert!(results.contains(&"Processed: World".to_string()));
    }
    
    #[tokio::test]
    async fn test_async_channel() {
        let mut channel = AsyncChannel::new();
        channel.send("Hello".to_string()).await.unwrap();
        channel.send("World".to_string()).await.unwrap();
        
        let message1 = channel.recv().await.unwrap();
        let message2 = channel.recv().await.unwrap();
        
        assert_eq!(message1, "Hello");
        assert_eq!(message2, "World");
    }
    
    #[tokio::test]
    async fn test_async_timeout_manager() {
        let manager = AsyncTimeoutManager::new(Duration::from_millis(100));
        
        let result = manager.with_timeout(async {
            sleep(Duration::from_millis(50)).await;
            "Success".to_string()
        }).await;
        
        assert_eq!(result, Ok("Success".to_string()));
    }
}
```

### Parallel Processing

```rust
// rust/03-parallel-processing.rs

/*
Parallel processing patterns and best practices for Rust
*/

use rayon::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Demonstrates parallel processing with Rayon.
pub struct ParallelProcessor {
    data: Vec<u64>,
}

impl ParallelProcessor {
    pub fn new(size: usize) -> Self {
        Self {
            data: (0..size as u64).collect(),
        }
    }
    
    /// Parallel map operation.
    pub fn parallel_map<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64 + Sync + Send,
    {
        self.data.par_iter().map(|&x| f(x)).collect()
    }
    
    /// Parallel filter operation.
    pub fn parallel_filter<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(&u64) -> bool + Sync + Send,
    {
        self.data.par_iter().filter(|&&x| f(&x)).copied().collect()
    }
    
    /// Parallel reduce operation.
    pub fn parallel_reduce<F>(&self, f: F) -> u64
    where
        F: Fn(u64, u64) -> u64 + Sync + Send,
    {
        self.data.par_iter().copied().reduce(|| 0, f)
    }
    
    /// Parallel sum operation.
    pub fn parallel_sum(&self) -> u64 {
        self.data.par_iter().sum()
    }
    
    /// Parallel sort operation.
    pub fn parallel_sort(&mut self) {
        self.data.par_sort();
    }
    
    /// Parallel group by operation.
    pub fn parallel_group_by<F, K>(&self, key_fn: F) -> HashMap<K, Vec<u64>>
    where
        F: Fn(&u64) -> K + Sync + Send,
        K: std::hash::Hash + Eq + Send,
    {
        self.data.par_iter().map(|&x| (key_fn(&x), x)).collect()
    }
}

/// Demonstrates parallel processing with custom thread pool.
pub struct CustomThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    job_sender: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl CustomThreadPool {
    pub fn new(size: usize) -> Self {
        let (job_sender, job_receiver) = mpsc::channel();
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            let job_receiver = Arc::clone(&job_receiver);
            
            let worker = thread::spawn(move || {
                loop {
                    let job = job_receiver.lock().unwrap().recv();
                    
                    match job {
                        Ok(job) => {
                            println!("Worker {} executing job", id);
                            job();
                        }
                        Err(_) => {
                            println!("Worker {} shutting down", id);
                            break;
                        }
                    }
                }
            });
            
            workers.push(worker);
        }
        
        Self { workers, job_sender }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.job_sender.send(job).unwrap();
    }
}

impl Drop for CustomThreadPool {
    fn drop(&mut self) {
        drop(&self.job_sender);
        
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

/// Demonstrates parallel processing with work stealing.
pub struct WorkStealingProcessor {
    data: Vec<u64>,
    chunk_size: usize,
}

impl WorkStealingProcessor {
    pub fn new(data: Vec<u64>, chunk_size: usize) -> Self {
        Self { data, chunk_size }
    }
    
    /// Process data in parallel chunks.
    pub fn process_chunks<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(&[u64]) -> Vec<u64> + Sync + Send,
    {
        self.data
            .par_chunks(self.chunk_size)
            .map(f)
            .flatten()
            .collect()
    }
    
    /// Process data with work stealing.
    pub fn process_with_work_stealing<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64 + Sync + Send,
    {
        self.data
            .par_iter()
            .map(|&x| f(x))
            .collect()
    }
    
    /// Process data with load balancing.
    pub fn process_with_load_balancing<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64 + Sync + Send,
    {
        self.data
            .par_iter()
            .with_min_len(1)
            .map(|&x| f(x))
            .collect()
    }
}

/// Demonstrates parallel processing with synchronization.
pub struct SynchronizedProcessor {
    data: Arc<Mutex<Vec<u64>>>,
    results: Arc<Mutex<Vec<u64>>>,
}

impl SynchronizedProcessor {
    pub fn new(data: Vec<u64>) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Process data with synchronization.
    pub fn process_synchronized<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64 + Sync + Send,
    {
        let data = self.data.lock().unwrap();
        let results: Vec<u64> = data.par_iter().map(|&x| f(x)).collect();
        results
    }
    
    /// Process data with atomic operations.
    pub fn process_atomic<F>(&self, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64 + Sync + Send,
    {
        let data = self.data.lock().unwrap();
        let results: Vec<u64> = data.par_iter().map(|&x| f(x)).collect();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_processor() {
        let processor = ParallelProcessor::new(1000);
        
        let results = processor.parallel_map(|x| x * 2);
        assert_eq!(results.len(), 1000);
        
        let filtered = processor.parallel_filter(|&x| x % 2 == 0);
        assert!(filtered.len() > 0);
        
        let sum = processor.parallel_sum();
        assert!(sum > 0);
    }
    
    #[test]
    fn test_work_stealing_processor() {
        let data = (0..1000).collect();
        let processor = WorkStealingProcessor::new(data, 100);
        
        let results = processor.process_chunks(|chunk| {
            chunk.iter().map(|&x| x * 2).collect()
        });
        
        assert_eq!(results.len(), 1000);
    }
    
    #[test]
    fn test_synchronized_processor() {
        let data = (0..100).collect();
        let processor = SynchronizedProcessor::new(data);
        
        let results = processor.process_synchronized(|x| x * 2);
        assert_eq!(results.len(), 100);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic threading
use std::thread;
use std::sync::{Arc, Mutex};

let data = Arc::new(Mutex::new(0));
let data_clone = Arc::clone(&data);
thread::spawn(move || {
    let mut data = data_clone.lock().unwrap();
    *data += 1;
});

// 2. Async programming
use tokio::time::sleep;
use std::time::Duration;

async fn async_function() -> String {
    sleep(Duration::from_millis(100)).await;
    "Hello".to_string()
}

// 3. Parallel processing
use rayon::prelude::*;

let data = vec![1, 2, 3, 4, 5];
let results: Vec<i32> = data.par_iter().map(|x| x * 2).collect();

// 4. Channels
use std::sync::mpsc;

let (sender, receiver) = mpsc::channel();
sender.send("Hello".to_string()).unwrap();
let message = receiver.recv().unwrap();
```

### Essential Patterns

```rust
// Complete concurrency setup
pub fn setup_rust_concurrency() {
    // 1. Threading patterns
    // 2. Async programming
    // 3. Parallel processing
    // 4. Synchronization primitives
    // 5. Channel communication
    // 6. Work stealing
    // 7. Load balancing
    // 8. Performance optimization
    
    println!("Rust concurrency setup complete!");
}
```

---

*This guide provides the complete machinery for Rust concurrency patterns. Each pattern includes implementation examples, concurrency strategies, and real-world usage patterns for enterprise parallel processing.*
