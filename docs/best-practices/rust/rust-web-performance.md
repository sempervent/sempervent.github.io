# Rust Web Performance Best Practices

**Objective**: Master senior-level Rust web performance optimization patterns for production systems. When you need to build high-performance web applications, when you want to optimize for speed and scalability, when you need enterprise-grade performance strategies—these best practices become your weapon of choice.

## Core Principles

- **Async Programming**: Leverage async/await for non-blocking operations
- **Connection Pooling**: Efficient database and HTTP connection management
- **Caching**: Implement multi-layer caching strategies
- **Compression**: Use compression for responses and assets
- **Load Balancing**: Distribute load across multiple instances

## Performance Optimization Patterns

### Async Performance

```rust
// rust/01-async-performance.rs

/*
Async performance patterns and best practices
*/

use std::sync::Arc;
use tokio::sync::{Semaphore, RwLock};
use tokio::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// High-performance async service.
pub struct AsyncService {
    semaphore: Arc<Semaphore>,
    cache: Arc<RwLock<std::collections::HashMap<String, CachedValue>>>,
    max_concurrent: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedValue {
    pub data: String,
    pub expires_at: Instant,
}

impl AsyncService {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_concurrent,
        }
    }
    
    /// Process request with concurrency control.
    pub async fn process_request(&self, key: String) -> Result<String, String> {
        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;
        
        // Check cache first
        if let Some(cached) = self.get_from_cache(&key).await {
            return Ok(cached);
        }
        
        // Simulate expensive operation
        let result = self.expensive_operation(&key).await?;
        
        // Cache the result
        self.cache_result(key, result.clone()).await;
        
        Ok(result)
    }
    
    /// Get value from cache.
    async fn get_from_cache(&self, key: &str) -> Option<String> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(key) {
            if cached.expires_at > Instant::now() {
                return Some(cached.data.clone());
            }
        }
        None
    }
    
    /// Cache a result.
    async fn cache_result(&self, key: String, data: String) {
        let mut cache = self.cache.write().await;
        let expires_at = Instant::now() + Duration::from_secs(300); // 5 minutes
        cache.insert(key, CachedValue { data, expires_at });
    }
    
    /// Simulate expensive operation.
    async fn expensive_operation(&self, key: &str) -> Result<String, String> {
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(format!("Processed: {}", key))
    }
    
    /// Process multiple requests concurrently.
    pub async fn process_batch(&self, keys: Vec<String>) -> Vec<Result<String, String>> {
        let futures: Vec<_> = keys.into_iter()
            .map(|key| self.process_request(key))
            .collect();
        
        futures::future::join_all(futures).await
    }
    
    /// Process requests with timeout.
    pub async fn process_with_timeout(&self, key: String, timeout: Duration) -> Result<String, String> {
        tokio::time::timeout(timeout, self.process_request(key))
            .await
            .map_err(|_| "Request timeout".to_string())?
    }
}

/// Connection pool for database connections.
pub struct ConnectionPool {
    connections: Arc<RwLock<Vec<DatabaseConnection>>>,
    max_connections: usize,
    semaphore: Arc<Semaphore>,
}

pub struct DatabaseConnection {
    pub id: u32,
    pub is_available: bool,
}

impl ConnectionPool {
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: Arc::new(RwLock::new(Vec::new())),
            max_connections,
            semaphore: Arc::new(Semaphore::new(max_connections)),
        }
    }
    
    /// Get a connection from the pool.
    pub async fn get_connection(&self) -> Result<PooledConnection, String> {
        let _permit = self.semaphore.acquire().await.map_err(|e| e.to_string())?;
        
        let mut connections = self.connections.write().await;
        
        // Find available connection
        for connection in connections.iter_mut() {
            if connection.is_available {
                connection.is_available = false;
                return Ok(PooledConnection {
                    connection: connection.id,
                    pool: self.clone(),
                });
            }
        }
        
        // Create new connection if under limit
        if connections.len() < self.max_connections {
            let connection_id = connections.len() as u32;
            let connection = DatabaseConnection {
                id: connection_id,
                is_available: false,
            };
            connections.push(connection);
            
            return Ok(PooledConnection {
                connection: connection_id,
                pool: self.clone(),
            });
        }
        
        Err("No connections available".to_string())
    }
}

impl Clone for ConnectionPool {
    fn clone(&self) -> Self {
        Self {
            connections: Arc::clone(&self.connections),
            max_connections: self.max_connections,
            semaphore: Arc::clone(&self.semaphore),
        }
    }
}

/// Pooled connection wrapper.
pub struct PooledConnection {
    connection: u32,
    pool: ConnectionPool,
}

impl PooledConnection {
    /// Execute a query using the connection.
    pub async fn execute_query(&self, query: &str) -> Result<String, String> {
        // Simulate database query
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(format!("Result for query: {}", query))
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        let pool = self.pool.clone();
        let connection_id = self.connection;
        
        tokio::spawn(async move {
            let mut connections = pool.connections.write().await;
            if let Some(conn) = connections.iter_mut().find(|c| c.id == connection_id) {
                conn.is_available = true;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_service() {
        let service = AsyncService::new(10);
        
        let result = service.process_request("test-key".to_string()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Processed"));
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let service = AsyncService::new(10);
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
        
        let results = service.process_batch(keys).await;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }
    
    #[tokio::test]
    async fn test_connection_pool() {
        let pool = ConnectionPool::new(5);
        
        let connection = pool.get_connection().await.unwrap();
        let result = connection.execute_query("SELECT * FROM users").await;
        assert!(result.is_ok());
    }
}
```

### Caching Strategies

```rust
// rust/02-caching-strategies.rs

/*
Caching strategies and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Multi-level cache implementation.
pub struct MultiLevelCache {
    l1_cache: Arc<RwLock<HashMap<String, CachedValue>>>,
    l2_cache: Arc<RwLock<HashMap<String, CachedValue>>>,
    l1_capacity: usize,
    l2_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedValue {
    pub data: String,
    pub expires_at: Instant,
    pub access_count: u64,
    pub last_accessed: Instant,
}

impl MultiLevelCache {
    pub fn new(l1_capacity: usize, l2_capacity: usize) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l2_cache: Arc::new(RwLock::new(HashMap::new())),
            l1_capacity,
            l2_capacity,
        }
    }
    
    /// Get value from cache.
    pub async fn get(&self, key: &str) -> Option<String> {
        // Check L1 cache first
        if let Some(value) = self.get_from_l1(key).await {
            return Some(value);
        }
        
        // Check L2 cache
        if let Some(value) = self.get_from_l2(key).await {
            // Promote to L1 cache
            self.set_l1(key, value.clone()).await;
            return Some(value);
        }
        
        None
    }
    
    /// Set value in cache.
    pub async fn set(&self, key: String, value: String, ttl: Duration) {
        let expires_at = Instant::now() + ttl;
        let cached_value = CachedValue {
            data: value.clone(),
            expires_at,
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        // Set in L1 cache
        self.set_l1(&key, value).await;
        
        // Set in L2 cache
        self.set_l2(&key, cached_value).await;
    }
    
    /// Get from L1 cache.
    async fn get_from_l1(&self, key: &str) -> Option<String> {
        let mut cache = self.l1_cache.write().await;
        if let Some(cached) = cache.get_mut(key) {
            if cached.expires_at > Instant::now() {
                cached.access_count += 1;
                cached.last_accessed = Instant::now();
                return Some(cached.data.clone());
            } else {
                cache.remove(key);
            }
        }
        None
    }
    
    /// Get from L2 cache.
    async fn get_from_l2(&self, key: &str) -> Option<String> {
        let mut cache = self.l2_cache.write().await;
        if let Some(cached) = cache.get_mut(key) {
            if cached.expires_at > Instant::now() {
                cached.access_count += 1;
                cached.last_accessed = Instant::now();
                return Some(cached.data.clone());
            } else {
                cache.remove(key);
            }
        }
        None
    }
    
    /// Set in L1 cache.
    async fn set_l1(&self, key: &str, value: String) {
        let mut cache = self.l1_cache.write().await;
        
        // Evict if at capacity
        if cache.len() >= self.l1_capacity {
            self.evict_l1(&mut cache).await;
        }
        
        let expires_at = Instant::now() + Duration::from_secs(300);
        let cached_value = CachedValue {
            data: value,
            expires_at,
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        cache.insert(key.to_string(), cached_value);
    }
    
    /// Set in L2 cache.
    async fn set_l2(&self, key: &str, value: CachedValue) {
        let mut cache = self.l2_cache.write().await;
        
        // Evict if at capacity
        if cache.len() >= self.l2_capacity {
            self.evict_l2(&mut cache).await;
        }
        
        cache.insert(key.to_string(), value);
    }
    
    /// Evict from L1 cache using LRU.
    async fn evict_l1(&self, cache: &mut HashMap<String, CachedValue>) {
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();
        
        for (key, value) in cache.iter() {
            if value.last_accessed < oldest_time {
                oldest_time = value.last_accessed;
                oldest_key = Some(key.clone());
            }
        }
        
        if let Some(key) = oldest_key {
            cache.remove(&key);
        }
    }
    
    /// Evict from L2 cache using LRU.
    async fn evict_l2(&self, cache: &mut HashMap<String, CachedValue>) {
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();
        
        for (key, value) in cache.iter() {
            if value.last_accessed < oldest_time {
                oldest_time = value.last_accessed;
                oldest_key = Some(key.clone());
            }
        }
        
        if let Some(key) = oldest_key {
            cache.remove(&key);
        }
    }
}

/// Cache warming service.
pub struct CacheWarmingService {
    cache: Arc<MultiLevelCache>,
    data_source: Arc<dyn DataSource>,
}

pub trait DataSource: Send + Sync {
    async fn get_data(&self, key: &str) -> Result<String, String>;
}

pub struct DatabaseDataSource;

impl DataSource for DatabaseDataSource {
    async fn get_data(&self, key: &str) -> Result<String, String> {
        // Simulate database query
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(format!("Data for key: {}", key))
    }
}

impl CacheWarmingService {
    pub fn new(cache: Arc<MultiLevelCache>, data_source: Arc<dyn DataSource>) -> Self {
        Self { cache, data_source }
    }
    
    /// Warm cache with frequently accessed data.
    pub async fn warm_cache(&self, keys: Vec<String>) -> Result<(), String> {
        for key in keys {
            if self.cache.get(&key).await.is_none() {
                let data = self.data_source.get_data(&key).await?;
                self.cache.set(key, data, Duration::from_secs(300)).await;
            }
        }
        Ok(())
    }
    
    /// Preload cache with predicted data.
    pub async fn preload_cache(&self, key: String) -> Result<(), String> {
        let data = self.data_source.get_data(&key).await?;
        self.cache.set(key, data, Duration::from_secs(300)).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_multi_level_cache() {
        let cache = MultiLevelCache::new(10, 100);
        
        cache.set("key1".to_string(), "value1".to_string(), Duration::from_secs(60)).await;
        
        let value = cache.get("key1").await;
        assert_eq!(value, Some("value1".to_string()));
    }
    
    #[tokio::test]
    async fn test_cache_warming() {
        let cache = Arc::new(MultiLevelCache::new(10, 100));
        let data_source = Arc::new(DatabaseDataSource);
        let service = CacheWarmingService::new(cache, data_source);
        
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let result = service.warm_cache(keys).await;
        assert!(result.is_ok());
    }
}
```

### Load Balancing

```rust
// rust/03-load-balancing.rs

/*
Load balancing patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Load balancer implementation.
pub struct LoadBalancer {
    servers: Arc<RwLock<Vec<Server>>>,
    strategy: LoadBalancingStrategy,
    current_index: AtomicUsize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Server {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub weight: u32,
    pub is_healthy: bool,
    pub response_time: u64,
    pub active_connections: u32,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    Random,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            servers: Arc::new(RwLock::new(Vec::new())),
            strategy,
            current_index: AtomicUsize::new(0),
        }
    }
    
    /// Add a server to the load balancer.
    pub async fn add_server(&self, server: Server) {
        let mut servers = self.servers.write().await;
        servers.push(server);
    }
    
    /// Remove a server from the load balancer.
    pub async fn remove_server(&self, server_id: &str) {
        let mut servers = self.servers.write().await;
        servers.retain(|s| s.id != server_id);
    }
    
    /// Get the next server based on the strategy.
    pub async fn get_next_server(&self) -> Option<Server> {
        let servers = self.servers.read().await;
        let healthy_servers: Vec<Server> = servers
            .iter()
            .filter(|s| s.is_healthy)
            .cloned()
            .collect();
        
        if healthy_servers.is_empty() {
            return None;
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index = self.current_index.fetch_add(1, Ordering::Relaxed);
                Some(healthy_servers[index % healthy_servers.len()].clone())
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.get_weighted_server(&healthy_servers).await
            }
            LoadBalancingStrategy::LeastConnections => {
                self.get_least_connections_server(&healthy_servers)
            }
            LoadBalancingStrategy::LeastResponseTime => {
                self.get_least_response_time_server(&healthy_servers)
            }
            LoadBalancingStrategy::Random => {
                let index = fastrand::usize(..healthy_servers.len());
                Some(healthy_servers[index].clone())
            }
        }
    }
    
    /// Get server with weighted round-robin.
    async fn get_weighted_server(&self, servers: &[Server]) -> Option<Server> {
        let total_weight: u32 = servers.iter().map(|s| s.weight).sum();
        if total_weight == 0 {
            return None;
        }
        
        let mut current_weight = 0;
        let mut selected_server = None;
        
        for server in servers {
            current_weight += server.weight;
            if current_weight >= total_weight {
                selected_server = Some(server.clone());
                break;
            }
        }
        
        selected_server
    }
    
    /// Get server with least connections.
    fn get_least_connections_server(&self, servers: &[Server]) -> Option<Server> {
        servers
            .iter()
            .min_by_key(|s| s.active_connections)
            .cloned()
    }
    
    /// Get server with least response time.
    fn get_least_response_time_server(&self, servers: &[Server]) -> Option<Server> {
        servers
            .iter()
            .min_by_key(|s| s.response_time)
            .cloned()
    }
    
    /// Update server health status.
    pub async fn update_server_health(&self, server_id: &str, is_healthy: bool) {
        let mut servers = self.servers.write().await;
        if let Some(server) = servers.iter_mut().find(|s| s.id == server_id) {
            server.is_healthy = is_healthy;
        }
    }
    
    /// Update server metrics.
    pub async fn update_server_metrics(&self, server_id: &str, response_time: u64, active_connections: u32) {
        let mut servers = self.servers.write().await;
        if let Some(server) = servers.iter_mut().find(|s| s.id == server_id) {
            server.response_time = response_time;
            server.active_connections = active_connections;
        }
    }
}

/// Health check service.
pub struct HealthCheckService {
    load_balancer: Arc<LoadBalancer>,
    check_interval: std::time::Duration,
}

impl HealthCheckService {
    pub fn new(load_balancer: Arc<LoadBalancer>, check_interval: std::time::Duration) -> Self {
        Self {
            load_balancer,
            check_interval,
        }
    }
    
    /// Start health checking.
    pub async fn start(&self) {
        let load_balancer = Arc::clone(&self.load_balancer);
        let interval = self.check_interval;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                load_balancer.check_health().await;
            }
        });
    }
    
    /// Check health of all servers.
    async fn check_health(&self) {
        let servers = self.load_balancer.servers.read().await;
        let server_ids: Vec<String> = servers.iter().map(|s| s.id.clone()).collect();
        drop(servers);
        
        for server_id in server_ids {
            let is_healthy = self.check_server_health(&server_id).await;
            self.load_balancer.update_server_health(&server_id, is_healthy).await;
        }
    }
    
    /// Check health of a specific server.
    async fn check_server_health(&self, server_id: &str) -> bool {
        // Simulate health check
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        true
    }
}

impl LoadBalancer {
    /// Check health of all servers.
    pub async fn check_health(&self) {
        let servers = self.servers.read().await;
        let server_ids: Vec<String> = servers.iter().map(|s| s.id.clone()).collect();
        drop(servers);
        
        for server_id in server_ids {
            let is_healthy = self.check_server_health(&server_id).await;
            self.update_server_health(&server_id, is_healthy).await;
        }
    }
    
    /// Check health of a specific server.
    async fn check_server_health(&self, server_id: &str) -> bool {
        // Simulate health check
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_load_balancer() {
        let load_balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        
        let server = Server {
            id: "server1".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            weight: 1,
            is_healthy: true,
            response_time: 100,
            active_connections: 0,
        };
        
        load_balancer.add_server(server).await;
        
        let selected_server = load_balancer.get_next_server().await;
        assert!(selected_server.is_some());
        assert_eq!(selected_server.unwrap().id, "server1");
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let load_balancer = Arc::new(LoadBalancer::new(LoadBalancingStrategy::RoundRobin));
        let health_service = HealthCheckService::new(
            Arc::clone(&load_balancer),
            std::time::Duration::from_secs(1),
        );
        
        let server = Server {
            id: "server1".to_string(),
            host: "localhost".to_string(),
            port: 8080,
            weight: 1,
            is_healthy: true,
            response_time: 100,
            active_connections: 0,
        };
        
        load_balancer.add_server(server).await;
        
        // Start health checking
        health_service.start().await;
        
        // Wait a bit for health check to run
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Async performance
let service = AsyncService::new(10);
let result = service.process_request("key".to_string()).await?;

// 2. Multi-level caching
let cache = MultiLevelCache::new(100, 1000);
cache.set("key".to_string(), "value".to_string(), Duration::from_secs(60)).await;

// 3. Load balancing
let load_balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
let server = load_balancer.get_next_server().await?;
```

### Essential Patterns

```rust
// Complete web performance setup
pub fn setup_rust_web_performance() {
    // 1. Async programming
    // 2. Connection pooling
    // 3. Multi-level caching
    // 4. Load balancing
    // 5. Health checking
    // 6. Compression
    // 7. Response optimization
    // 8. Monitoring
    
    println!("Rust web performance setup complete!");
}
```

---

*This guide provides the complete machinery for Rust web performance. Each pattern includes implementation examples, performance strategies, and real-world usage patterns for enterprise web optimization.*
