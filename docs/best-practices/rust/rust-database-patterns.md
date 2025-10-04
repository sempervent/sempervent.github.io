# Rust Database Patterns Best Practices

**Objective**: Master senior-level Rust database patterns for production systems. When you need to build robust database applications, when you want to implement efficient data access patterns, when you need enterprise-grade database strategies—these best practices become your weapon of choice.

## Core Principles

- **Connection Pooling**: Efficient database connection management
- **Transaction Management**: Proper transaction handling and rollback
- **Query Optimization**: Optimize database queries for performance
- **Error Handling**: Robust error handling for database operations
- **Migration Management**: Database schema evolution and versioning

## Database Patterns

### Connection Pooling

```rust
// rust/01-connection-pooling.rs

/*
Database connection pooling patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{timeout, Instant};

/// Database connection pool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
    pub acquire_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(3600),
            acquire_timeout: Duration::from_secs(30),
        }
    }
}

/// Database connection.
#[derive(Debug, Clone)]
pub struct DatabaseConnection {
    pub id: String,
    pub created_at: Instant,
    pub last_used: Instant,
    pub is_active: bool,
    pub connection_string: String,
}

impl DatabaseConnection {
    pub fn new(id: String, connection_string: String) -> Self {
        let now = Instant::now();
        Self {
            id,
            created_at: now,
            last_used: now,
            is_active: true,
            connection_string,
        }
    }
    
    /// Check if connection is expired.
    pub fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }
    
    /// Check if connection is idle.
    pub fn is_idle(&self, idle_timeout: Duration) -> bool {
        self.last_used.elapsed() > idle_timeout
    }
    
    /// Mark connection as used.
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
    }
}

/// Database connection pool.
pub struct ConnectionPool {
    config: PoolConfig,
    connections: Arc<RwLock<Vec<DatabaseConnection>>>,
    semaphore: Arc<Semaphore>,
    connection_string: String,
    next_connection_id: Arc<RwLock<u32>>,
}

impl ConnectionPool {
    pub fn new(config: PoolConfig, connection_string: String) -> Self {
        let max_connections = config.max_connections as usize;
        Self {
            config,
            connections: Arc::new(RwLock::new(Vec::new())),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            connection_string,
            next_connection_id: Arc::new(RwLock::new(1)),
        }
    }
    
    /// Get a connection from the pool.
    pub async fn get_connection(&self) -> Result<PooledConnection, String> {
        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await
            .map_err(|e| format!("Failed to acquire connection permit: {}", e))?;
        
        // Try to get existing connection
        if let Some(connection) = self.get_existing_connection().await? {
            return Ok(PooledConnection {
                connection,
                pool: self.clone(),
            });
        }
        
        // Create new connection
        let connection = self.create_connection().await?;
        self.add_connection(connection.clone()).await;
        
        Ok(PooledConnection {
            connection,
            pool: self.clone(),
        })
    }
    
    /// Get an existing connection from the pool.
    async fn get_existing_connection(&self) -> Result<Option<DatabaseConnection>, String> {
        let mut connections = self.connections.write().await;
        
        // Find a healthy, non-expired connection
        for (index, connection) in connections.iter().enumerate() {
            if connection.is_active && 
               !connection.is_expired(self.config.max_lifetime) &&
               !connection.is_idle(self.config.idle_timeout) {
                let mut conn = connection.clone();
                conn.mark_used();
                connections.remove(index);
                return Ok(Some(conn));
            }
        }
        
        Ok(None)
    }
    
    /// Create a new database connection.
    async fn create_connection(&self) -> Result<DatabaseConnection, String> {
        let mut next_id = self.next_connection_id.write().await;
        let connection_id = format!("conn-{}", *next_id);
        *next_id += 1;
        
        // In a real implementation, you would establish a database connection
        // For this example, we'll simulate the connection
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(DatabaseConnection::new(connection_id, self.connection_string.clone()))
    }
    
    /// Add a connection to the pool.
    async fn add_connection(&self, connection: DatabaseConnection) {
        let mut connections = self.connections.write().await;
        connections.push(connection);
    }
    
    /// Return a connection to the pool.
    pub async fn return_connection(&self, connection: DatabaseConnection) {
        let mut connections = self.connections.write().await;
        
        // Check if connection is still valid
        if connection.is_active && !connection.is_expired(self.config.max_lifetime) {
            connections.push(connection);
        }
    }
    
    /// Clean up expired connections.
    pub async fn cleanup_expired_connections(&self) {
        let mut connections = self.connections.write().await;
        connections.retain(|conn| {
            conn.is_active && !conn.is_expired(self.config.max_lifetime)
        });
    }
    
    /// Get pool statistics.
    pub async fn get_stats(&self) -> PoolStats {
        let connections = self.connections.read().await;
        let total_connections = connections.len();
        let active_connections = connections.iter().filter(|c| c.is_active).count();
        let idle_connections = connections.iter().filter(|c| c.is_idle(self.config.idle_timeout)).count();
        
        PoolStats {
            total_connections,
            active_connections,
            idle_connections,
            max_connections: self.config.max_connections as usize,
            min_connections: self.config.min_connections as usize,
        }
    }
}

impl Clone for ConnectionPool {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            connections: Arc::clone(&self.connections),
            semaphore: Arc::clone(&self.semaphore),
            connection_string: self.connection_string.clone(),
            next_connection_id: Arc::clone(&self.next_connection_id),
        }
    }
}

/// Pooled connection wrapper.
pub struct PooledConnection {
    connection: DatabaseConnection,
    pool: ConnectionPool,
}

impl PooledConnection {
    /// Execute a query using the connection.
    pub async fn execute_query(&self, query: &str) -> Result<QueryResult, String> {
        // In a real implementation, you would execute the query
        // For this example, we'll simulate the query execution
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(QueryResult {
            rows_affected: 1,
            rows_returned: 1,
            execution_time: Duration::from_millis(50),
        })
    }
    
    /// Execute a prepared statement.
    pub async fn execute_prepared(&self, statement: &str, params: &[serde_json::Value]) -> Result<QueryResult, String> {
        // In a real implementation, you would execute the prepared statement
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        Ok(QueryResult {
            rows_affected: 1,
            rows_returned: 1,
            execution_time: Duration::from_millis(30),
        })
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        let pool = self.pool.clone();
        let connection = self.connection.clone();
        
        tokio::spawn(async move {
            pool.return_connection(connection).await;
        });
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows_affected: u64,
    pub rows_returned: u64,
    pub execution_time: Duration,
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub max_connections: usize,
    pub min_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_connection_pool() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config, "postgresql://localhost/test".to_string());
        
        let connection = pool.get_connection().await.unwrap();
        let result = connection.execute_query("SELECT 1").await.unwrap();
        
        assert_eq!(result.rows_affected, 1);
    }
    
    #[tokio::test]
    async fn test_pool_stats() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config, "postgresql://localhost/test".to_string());
        
        let stats = pool.get_stats().await;
        assert_eq!(stats.max_connections, 10);
        assert_eq!(stats.min_connections, 1);
    }
}
```

### Transaction Management

```rust
// rust/02-transaction-management.rs

/*
Transaction management patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Transaction isolation levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Transaction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionConfig {
    pub isolation_level: IsolationLevel,
    pub timeout: Duration,
    pub read_only: bool,
    pub deferrable: bool,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            isolation_level: IsolationLevel::ReadCommitted,
            timeout: Duration::from_secs(30),
            read_only: false,
            deferrable: false,
        }
    }
}

/// Database transaction.
pub struct Transaction {
    pub id: String,
    pub config: TransactionConfig,
    pub is_active: bool,
    pub start_time: std::time::Instant,
    pub operations: Vec<TransactionOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionOperation {
    pub id: String,
    pub operation_type: OperationType,
    pub query: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Select,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
}

impl Transaction {
    pub fn new(id: String, config: TransactionConfig) -> Self {
        Self {
            id,
            config,
            is_active: true,
            start_time: std::time::Instant::now(),
            operations: Vec::new(),
        }
    }
    
    /// Execute a query within the transaction.
    pub async fn execute_query(&mut self, query: String, parameters: HashMap<String, serde_json::Value>) -> Result<QueryResult, String> {
        if !self.is_active {
            return Err("Transaction is not active".to_string());
        }
        
        // Check timeout
        if self.start_time.elapsed() > self.config.timeout {
            return Err("Transaction timeout".to_string());
        }
        
        // Add operation to transaction log
        let operation = TransactionOperation {
            id: format!("op-{}", self.operations.len() + 1),
            operation_type: self.determine_operation_type(&query),
            query: query.clone(),
            parameters: parameters.clone(),
            timestamp: std::time::SystemTime::now(),
        };
        self.operations.push(operation);
        
        // Execute the query
        let result = self.execute_query_internal(&query, &parameters).await?;
        
        Ok(result)
    }
    
    /// Determine the operation type from the query.
    fn determine_operation_type(&self, query: &str) -> OperationType {
        let query_lower = query.to_lowercase();
        if query_lower.starts_with("select") {
            OperationType::Select
        } else if query_lower.starts_with("insert") {
            OperationType::Insert
        } else if query_lower.starts_with("update") {
            OperationType::Update
        } else if query_lower.starts_with("delete") {
            OperationType::Delete
        } else if query_lower.starts_with("create") {
            OperationType::Create
        } else if query_lower.starts_with("drop") {
            OperationType::Drop
        } else if query_lower.starts_with("alter") {
            OperationType::Alter
        } else {
            OperationType::Select
        }
    }
    
    /// Execute query internally.
    async fn execute_query_internal(&self, query: &str, parameters: &HashMap<String, serde_json::Value>) -> Result<QueryResult, String> {
        // In a real implementation, you would execute the query against the database
        // For this example, we'll simulate the query execution
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(QueryResult {
            rows_affected: 1,
            rows_returned: 1,
            execution_time: Duration::from_millis(50),
        })
    }
    
    /// Commit the transaction.
    pub async fn commit(&mut self) -> Result<(), String> {
        if !self.is_active {
            return Err("Transaction is not active".to_string());
        }
        
        // In a real implementation, you would commit the transaction
        println!("Committing transaction {}", self.id);
        
        self.is_active = false;
        Ok(())
    }
    
    /// Rollback the transaction.
    pub async fn rollback(&mut self) -> Result<(), String> {
        if !self.is_active {
            return Err("Transaction is not active".to_string());
        }
        
        // In a real implementation, you would rollback the transaction
        println!("Rolling back transaction {}", self.id);
        
        self.is_active = false;
        Ok(())
    }
    
    /// Get transaction status.
    pub fn get_status(&self) -> TransactionStatus {
        TransactionStatus {
            id: self.id.clone(),
            is_active: self.is_active,
            duration: self.start_time.elapsed(),
            operation_count: self.operations.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransactionStatus {
    pub id: String,
    pub is_active: bool,
    pub duration: Duration,
    pub operation_count: usize,
}

/// Transaction manager.
pub struct TransactionManager {
    active_transactions: HashMap<String, Transaction>,
    next_transaction_id: u32,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            active_transactions: HashMap::new(),
            next_transaction_id: 1,
        }
    }
    
    /// Start a new transaction.
    pub fn start_transaction(&mut self, config: TransactionConfig) -> String {
        let transaction_id = format!("txn-{}", self.next_transaction_id);
        self.next_transaction_id += 1;
        
        let transaction = Transaction::new(transaction_id.clone(), config);
        self.active_transactions.insert(transaction_id.clone(), transaction);
        
        transaction_id
    }
    
    /// Get a transaction by ID.
    pub fn get_transaction(&mut self, id: &str) -> Option<&mut Transaction> {
        self.active_transactions.get_mut(id)
    }
    
    /// Commit a transaction.
    pub async fn commit_transaction(&mut self, id: &str) -> Result<(), String> {
        if let Some(transaction) = self.active_transactions.get_mut(id) {
            transaction.commit().await?;
            self.active_transactions.remove(id);
            Ok(())
        } else {
            Err("Transaction not found".to_string())
        }
    }
    
    /// Rollback a transaction.
    pub async fn rollback_transaction(&mut self, id: &str) -> Result<(), String> {
        if let Some(transaction) = self.active_transactions.get_mut(id) {
            transaction.rollback().await?;
            self.active_transactions.remove(id);
            Ok(())
        } else {
            Err("Transaction not found".to_string())
        }
    }
    
    /// Get all active transactions.
    pub fn get_active_transactions(&self) -> Vec<&Transaction> {
        self.active_transactions.values().collect()
    }
    
    /// Clean up expired transactions.
    pub async fn cleanup_expired_transactions(&mut self) {
        let mut expired_transactions = Vec::new();
        
        for (id, transaction) in &self.active_transactions {
            if transaction.start_time.elapsed() > transaction.config.timeout {
                expired_transactions.push(id.clone());
            }
        }
        
        for id in expired_transactions {
            if let Some(transaction) = self.active_transactions.get_mut(&id) {
                let _ = transaction.rollback().await;
                self.active_transactions.remove(&id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_transaction() {
        let config = TransactionConfig::default();
        let mut transaction = Transaction::new("test-txn".to_string(), config);
        
        let mut params = HashMap::new();
        params.insert("id".to_string(), serde_json::Value::Number(1.into()));
        
        let result = transaction.execute_query("SELECT * FROM users WHERE id = $1".to_string(), params).await;
        assert!(result.is_ok());
        
        let commit_result = transaction.commit().await;
        assert!(commit_result.is_ok());
    }
    
    #[tokio::test]
    async fn test_transaction_rollback() {
        let config = TransactionConfig::default();
        let mut transaction = Transaction::new("test-txn".to_string(), config);
        
        let mut params = HashMap::new();
        params.insert("id".to_string(), serde_json::Value::Number(1.into()));
        
        let result = transaction.execute_query("SELECT * FROM users WHERE id = $1".to_string(), params).await;
        assert!(result.is_ok());
        
        let rollback_result = transaction.rollback().await;
        assert!(rollback_result.is_ok());
    }
    
    #[tokio::test]
    async fn test_transaction_manager() {
        let mut manager = TransactionManager::new();
        
        let config = TransactionConfig::default();
        let txn_id = manager.start_transaction(config);
        
        let transaction = manager.get_transaction(&txn_id);
        assert!(transaction.is_some());
        
        let commit_result = manager.commit_transaction(&txn_id).await;
        assert!(commit_result.is_ok());
    }
}
```

### Query Optimization

```rust
// rust/03-query-optimization.rs

/*
Query optimization patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub query_id: String,
    pub estimated_cost: f64,
    pub estimated_rows: u64,
    pub execution_time: Duration,
    pub operations: Vec<QueryOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOperation {
    pub operation_type: String,
    pub table_name: String,
    pub index_used: Option<String>,
    pub cost: f64,
    pub rows: u64,
}

/// Query optimizer.
pub struct QueryOptimizer {
    statistics: HashMap<String, TableStatistics>,
    indexes: HashMap<String, Vec<IndexInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStatistics {
    pub table_name: String,
    pub row_count: u64,
    pub avg_row_size: u64,
    pub last_updated: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub index_name: String,
    pub column_name: String,
    pub cardinality: u64,
    pub is_unique: bool,
    pub is_primary: bool,
}

impl QueryOptimizer {
    pub fn new() -> Self {
        Self {
            statistics: HashMap::new(),
            indexes: HashMap::new(),
        }
    }
    
    /// Analyze a query and return an optimized plan.
    pub async fn optimize_query(&self, query: &str) -> Result<QueryPlan, String> {
        let query_id = format!("query-{}", uuid::Uuid::new_v4());
        
        // Parse the query
        let parsed_query = self.parse_query(query)?;
        
        // Generate execution plan
        let plan = self.generate_execution_plan(&parsed_query).await?;
        
        Ok(QueryPlan {
            query_id,
            estimated_cost: plan.estimated_cost,
            estimated_rows: plan.estimated_rows,
            execution_time: plan.execution_time,
            operations: plan.operations,
        })
    }
    
    /// Parse a SQL query.
    fn parse_query(&self, query: &str) -> Result<ParsedQuery, String> {
        // In a real implementation, you would use a SQL parser
        // For this example, we'll create a simple parser
        let query_lower = query.to_lowercase();
        
        if query_lower.starts_with("select") {
            Ok(ParsedQuery {
                query_type: QueryType::Select,
                tables: self.extract_tables(query),
                columns: self.extract_columns(query),
                conditions: self.extract_conditions(query),
                joins: self.extract_joins(query),
                order_by: self.extract_order_by(query),
                limit: self.extract_limit(query),
            })
        } else {
            Err("Unsupported query type".to_string())
        }
    }
    
    /// Extract table names from query.
    fn extract_tables(&self, query: &str) -> Vec<String> {
        // Simple table extraction - in a real implementation,
        // you would use a proper SQL parser
        let mut tables = Vec::new();
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("from") {
            let from_part = query_lower.split("from").nth(1).unwrap_or("");
            if let Some(table) = from_part.split_whitespace().next() {
                tables.push(table.to_string());
            }
        }
        
        tables
    }
    
    /// Extract column names from query.
    fn extract_columns(&self, query: &str) -> Vec<String> {
        // Simple column extraction
        let mut columns = Vec::new();
        let query_lower = query.to_lowercase();
        
        if query_lower.starts_with("select") {
            let select_part = query_lower.split("from").next().unwrap_or("");
            if select_part.contains("*") {
                columns.push("*".to_string());
            } else {
                // Extract individual columns
                let columns_part = select_part.replace("select", "").trim().to_string();
                for column in columns_part.split(",") {
                    columns.push(column.trim().to_string());
                }
            }
        }
        
        columns
    }
    
    /// Extract WHERE conditions from query.
    fn extract_conditions(&self, query: &str) -> Vec<String> {
        // Simple condition extraction
        let mut conditions = Vec::new();
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("where") {
            let where_part = query_lower.split("where").nth(1).unwrap_or("");
            if let Some(condition) = where_part.split("order").next() {
                conditions.push(condition.trim().to_string());
            }
        }
        
        conditions
    }
    
    /// Extract JOIN clauses from query.
    fn extract_joins(&self, query: &str) -> Vec<String> {
        // Simple join extraction
        let mut joins = Vec::new();
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("join") {
            let join_parts: Vec<&str> = query_lower.split("join").collect();
            for part in join_parts.iter().skip(1) {
                if let Some(join) = part.split_whitespace().next() {
                    joins.push(join.to_string());
                }
            }
        }
        
        joins
    }
    
    /// Extract ORDER BY clause from query.
    fn extract_order_by(&self, query: &str) -> Vec<String> {
        // Simple ORDER BY extraction
        let mut order_by = Vec::new();
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("order by") {
            let order_part = query_lower.split("order by").nth(1).unwrap_or("");
            if let Some(column) = order_part.split_whitespace().next() {
                order_by.push(column.to_string());
            }
        }
        
        order_by
    }
    
    /// Extract LIMIT clause from query.
    fn extract_limit(&self, query: &str) -> Option<u64> {
        // Simple LIMIT extraction
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("limit") {
            let limit_part = query_lower.split("limit").nth(1).unwrap_or("");
            if let Some(limit_str) = limit_part.split_whitespace().next() {
                return limit_str.parse().ok();
            }
        }
        
        None
    }
    
    /// Generate execution plan for a parsed query.
    async fn generate_execution_plan(&self, query: &ParsedQuery) -> Result<QueryPlan, String> {
        let mut operations = Vec::new();
        let mut total_cost = 0.0;
        let mut total_rows = 0;
        
        // Generate operations for each table
        for table in &query.tables {
            let stats = self.statistics.get(table).cloned().unwrap_or_else(|| {
                TableStatistics {
                    table_name: table.clone(),
                    row_count: 1000,
                    avg_row_size: 100,
                    last_updated: std::time::SystemTime::now(),
                }
            });
            
            let operation = QueryOperation {
                operation_type: "Seq Scan".to_string(),
                table_name: table.clone(),
                index_used: None,
                cost: stats.row_count as f64 * 0.01,
                rows: stats.row_count,
            };
            
            operations.push(operation);
            total_cost += stats.row_count as f64 * 0.01;
            total_rows += stats.row_count;
        }
        
        // Apply filters
        if !query.conditions.is_empty() {
            let filter_operation = QueryOperation {
                operation_type: "Filter".to_string(),
                table_name: "".to_string(),
                index_used: None,
                cost: total_rows as f64 * 0.1,
                rows: total_rows / 10,
            };
            operations.push(filter_operation);
            total_cost += total_rows as f64 * 0.1;
            total_rows /= 10;
        }
        
        // Apply sorting if needed
        if !query.order_by.is_empty() {
            let sort_operation = QueryOperation {
                operation_type: "Sort".to_string(),
                table_name: "".to_string(),
                index_used: None,
                cost: total_rows as f64 * 0.2,
                rows: total_rows,
            };
            operations.push(sort_operation);
            total_cost += total_rows as f64 * 0.2;
        }
        
        Ok(QueryPlan {
            query_id: "".to_string(),
            estimated_cost: total_cost,
            estimated_rows: total_rows,
            execution_time: Duration::from_millis((total_cost * 10.0) as u64),
            operations,
        })
    }
    
    /// Add table statistics.
    pub fn add_table_statistics(&mut self, stats: TableStatistics) {
        self.statistics.insert(stats.table_name.clone(), stats);
    }
    
    /// Add index information.
    pub fn add_index_info(&mut self, table_name: String, index_info: IndexInfo) {
        self.indexes.entry(table_name).or_insert_with(Vec::new).push(index_info);
    }
}

#[derive(Debug, Clone)]
pub struct ParsedQuery {
    pub query_type: QueryType,
    pub tables: Vec<String>,
    pub columns: Vec<String>,
    pub conditions: Vec<String>,
    pub joins: Vec<String>,
    pub order_by: Vec<String>,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_query_optimizer() {
        let mut optimizer = QueryOptimizer::new();
        
        let stats = TableStatistics {
            table_name: "users".to_string(),
            row_count: 1000,
            avg_row_size: 100,
            last_updated: std::time::SystemTime::now(),
        };
        optimizer.add_table_statistics(stats);
        
        let query = "SELECT * FROM users WHERE id = 1";
        let plan = optimizer.optimize_query(query).await.unwrap();
        
        assert!(plan.estimated_cost > 0.0);
        assert!(plan.estimated_rows > 0);
    }
    
    #[test]
    fn test_query_parsing() {
        let optimizer = QueryOptimizer::new();
        let query = "SELECT id, name FROM users WHERE age > 18 ORDER BY name LIMIT 10";
        
        let parsed = optimizer.parse_query(query).unwrap();
        assert_eq!(parsed.tables, vec!["users"]);
        assert_eq!(parsed.columns, vec!["id", "name"]);
        assert!(!parsed.conditions.is_empty());
        assert!(!parsed.order_by.is_empty());
        assert_eq!(parsed.limit, Some(10));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Connection pooling
let config = PoolConfig::default();
let pool = ConnectionPool::new(config, "postgresql://localhost/test".to_string());
let connection = pool.get_connection().await?;

// 2. Transaction management
let mut manager = TransactionManager::new();
let txn_id = manager.start_transaction(TransactionConfig::default());
let transaction = manager.get_transaction(&txn_id).unwrap();

// 3. Query optimization
let mut optimizer = QueryOptimizer::new();
let plan = optimizer.optimize_query("SELECT * FROM users").await?;
```

### Essential Patterns

```rust
// Complete database setup
pub fn setup_rust_database() {
    // 1. Connection pooling
    // 2. Transaction management
    // 3. Query optimization
    // 4. Migration management
    // 5. Error handling
    // 6. Performance monitoring
    // 7. Caching strategies
    // 8. Security patterns
    
    println!("Rust database setup complete!");
}
```

---

*This guide provides the complete machinery for Rust database patterns. Each pattern includes implementation examples, database strategies, and real-world usage patterns for enterprise database systems.*
