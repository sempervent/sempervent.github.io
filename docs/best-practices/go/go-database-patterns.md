# Go Database Patterns Best Practices

**Objective**: Master senior-level Go database patterns for production systems. When you need to build robust, scalable database applications, when you want to optimize database performance, when you need enterprise-grade database patterns—these best practices become your weapon of choice.

## Core Principles

- **Connection Pooling**: Efficiently manage database connections
- **Transaction Management**: Ensure data consistency and integrity
- **Query Optimization**: Write efficient and maintainable queries
- **Error Handling**: Handle database errors gracefully
- **Migration Management**: Version and manage database schema changes

## Connection Pooling

### Database Connection Pool

```go
// internal/database/connection_pool.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "sync"
    "time"
)

// ConnectionPool represents a database connection pool
type ConnectionPool struct {
    db     *sql.DB
    config PoolConfig
    mutex  sync.RWMutex
    stats  PoolStats
}

// PoolConfig represents connection pool configuration
type PoolConfig struct {
    MaxOpenConns    int           `json:"max_open_conns"`
    MaxIdleConns    int           `json:"max_idle_conns"`
    ConnMaxLifetime time.Duration `json:"conn_max_lifetime"`
    ConnMaxIdleTime time.Duration `json:"conn_max_idle_time"`
    PingInterval    time.Duration `json:"ping_interval"`
}

// PoolStats represents connection pool statistics
type PoolStats struct {
    OpenConnections     int           `json:"open_connections"`
    InUse              int           `json:"in_use"`
    Idle               int           `json:"idle"`
    WaitCount          int64         `json:"wait_count"`
    WaitDuration       time.Duration `json:"wait_duration"`
    MaxIdleClosed      int64         `json:"max_idle_closed"`
    MaxIdleTimeClosed  int64         `json:"max_idle_time_closed"`
    MaxLifetimeClosed  int64         `json:"max_lifetime_closed"`
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(driver, dsn string, config PoolConfig) (*ConnectionPool, error) {
    db, err := sql.Open(driver, dsn)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    pool := &ConnectionPool{
        db:     db,
        config: config,
    }
    
    // Start health monitoring
    go pool.monitorHealth()
    
    return pool, nil
}

// GetDB returns the database connection
func (cp *ConnectionPool) GetDB() *sql.DB {
    return cp.db
}

// Ping pings the database
func (cp *ConnectionPool) Ping(ctx context.Context) error {
    return cp.db.PingContext(ctx)
}

// Close closes the connection pool
func (cp *ConnectionPool) Close() error {
    return cp.db.Close()
}

// GetStats returns connection pool statistics
func (cp *ConnectionPool) GetStats() PoolStats {
    cp.mutex.RLock()
    defer cp.mutex.RUnlock()
    
    stats := cp.db.Stats()
    return PoolStats{
        OpenConnections:    stats.OpenConnections,
        InUse:             stats.InUse,
        Idle:              stats.Idle,
        WaitCount:         stats.WaitCount,
        WaitDuration:      stats.WaitDuration,
        MaxIdleClosed:     stats.MaxIdleClosed,
        MaxIdleTimeClosed: stats.MaxIdleTimeClosed,
        MaxLifetimeClosed: stats.MaxLifetimeClosed,
    }
}

// monitorHealth monitors the health of the connection pool
func (cp *ConnectionPool) monitorHealth() {
    ticker := time.NewTicker(cp.config.PingInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        if err := cp.db.PingContext(ctx); err != nil {
            // Handle ping failure
        }
        cancel()
    }
}

// Execute executes a query with retry logic
func (cp *ConnectionPool) Execute(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    var result sql.Result
    var err error
    
    for i := 0; i < 3; i++ {
        result, err = cp.db.ExecContext(ctx, query, args...)
        if err == nil {
            return result, nil
        }
        
        if i < 2 {
            time.Sleep(time.Duration(i+1) * time.Second)
        }
    }
    
    return nil, fmt.Errorf("failed to execute query after 3 attempts: %w", err)
}

// Query executes a query and returns rows
func (cp *ConnectionPool) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    return cp.db.QueryContext(ctx, query, args...)
}

// QueryRow executes a query and returns a single row
func (cp *ConnectionPool) QueryRow(ctx context.Context, query string, args ...interface{}) *sql.Row {
    return cp.db.QueryRowContext(ctx, query, args...)
}
```

### Prepared Statement Cache

```go
// internal/database/prepared_statements.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "sync"
)

// PreparedStatementCache represents a prepared statement cache
type PreparedStatementCache struct {
    db        *sql.DB
    statements map[string]*sql.Stmt
    mutex     sync.RWMutex
}

// NewPreparedStatementCache creates a new prepared statement cache
func NewPreparedStatementCache(db *sql.DB) *PreparedStatementCache {
    return &PreparedStatementCache{
        db:         db,
        statements: make(map[string]*sql.Stmt),
    }
}

// Get gets a prepared statement
func (psc *PreparedStatementCache) Get(ctx context.Context, query string) (*sql.Stmt, error) {
    psc.mutex.RLock()
    stmt, exists := psc.statements[query]
    psc.mutex.RUnlock()
    
    if exists {
        return stmt, nil
    }
    
    psc.mutex.Lock()
    defer psc.mutex.Unlock()
    
    // Double-check after acquiring write lock
    if stmt, exists := psc.statements[query]; exists {
        return stmt, nil
    }
    
    stmt, err := psc.db.PrepareContext(ctx, query)
    if err != nil {
        return nil, fmt.Errorf("failed to prepare statement: %w", err)
    }
    
    psc.statements[query] = stmt
    return stmt, nil
}

// Execute executes a prepared statement
func (psc *PreparedStatementCache) Execute(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    stmt, err := psc.Get(ctx, query)
    if err != nil {
        return nil, err
    }
    
    return stmt.ExecContext(ctx, args...)
}

// Query executes a prepared statement and returns rows
func (psc *PreparedStatementCache) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    stmt, err := psc.Get(ctx, query)
    if err != nil {
        return nil, err
    }
    
    return stmt.QueryContext(ctx, args...)
}

// QueryRow executes a prepared statement and returns a single row
func (psc *PreparedStatementCache) QueryRow(ctx context.Context, query string, args ...interface{}) *sql.Row {
    stmt, err := psc.Get(ctx, query)
    if err != nil {
        return nil, err
    }
    
    return stmt.QueryRowContext(ctx, args...)
}

// Close closes all prepared statements
func (psc *PreparedStatementCache) Close() error {
    psc.mutex.Lock()
    defer psc.mutex.Unlock()
    
    for _, stmt := range psc.statements {
        if err := stmt.Close(); err != nil {
            return err
        }
    }
    
    psc.statements = make(map[string]*sql.Stmt)
    return nil
}
```

## Transaction Management

### Transaction Manager

```go
// internal/database/transaction_manager.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "sync"
)

// TransactionManager represents a transaction manager
type TransactionManager struct {
    db *sql.DB
}

// NewTransactionManager creates a new transaction manager
func NewTransactionManager(db *sql.DB) *TransactionManager {
    return &TransactionManager{db: db}
}

// Transaction represents a database transaction
type Transaction struct {
    tx     *sql.Tx
    ctx    context.Context
    mutex  sync.RWMutex
    closed bool
}

// NewTransaction creates a new transaction
func (tm *TransactionManager) NewTransaction(ctx context.Context) (*Transaction, error) {
    tx, err := tm.db.BeginTx(ctx, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to begin transaction: %w", err)
    }
    
    return &Transaction{
        tx:     tx,
        ctx:    ctx,
        closed: false,
    }, nil
}

// Execute executes a query within the transaction
func (t *Transaction) Execute(query string, args ...interface{}) (sql.Result, error) {
    t.mutex.RLock()
    defer t.mutex.RUnlock()
    
    if t.closed {
        return nil, fmt.Errorf("transaction is closed")
    }
    
    return t.tx.ExecContext(t.ctx, query, args...)
}

// Query executes a query within the transaction
func (t *Transaction) Query(query string, args ...interface{}) (*sql.Rows, error) {
    t.mutex.RLock()
    defer t.mutex.RUnlock()
    
    if t.closed {
        return nil, fmt.Errorf("transaction is closed")
    }
    
    return t.tx.QueryContext(t.ctx, query, args...)
}

// QueryRow executes a query within the transaction
func (t *Transaction) QueryRow(query string, args ...interface{}) *sql.Row {
    t.mutex.RLock()
    defer t.mutex.RUnlock()
    
    if t.closed {
        return nil
    }
    
    return t.tx.QueryRowContext(t.ctx, query, args...)
}

// Commit commits the transaction
func (t *Transaction) Commit() error {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if t.closed {
        return fmt.Errorf("transaction is already closed")
    }
    
    err := t.tx.Commit()
    t.closed = true
    return err
}

// Rollback rolls back the transaction
func (t *Transaction) Rollback() error {
    t.mutex.Lock()
    defer t.mutex.Unlock()
    
    if t.closed {
        return nil
    }
    
    err := t.tx.Rollback()
    t.closed = true
    return err
}

// WithTransaction executes a function within a transaction
func (tm *TransactionManager) WithTransaction(ctx context.Context, fn func(*Transaction) error) error {
    tx, err := tm.NewTransaction(ctx)
    if err != nil {
        return err
    }
    
    defer func() {
        if !tx.closed {
            tx.Rollback()
        }
    }()
    
    if err := fn(tx); err != nil {
        return err
    }
    
    return tx.Commit()
}
```

### Optimistic Locking

```go
// internal/database/optimistic_locking.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

// OptimisticLock represents an optimistic lock
type OptimisticLock struct {
    ID        int64     `json:"id"`
    Version   int64     `json:"version"`
    UpdatedAt time.Time `json:"updated_at"`
}

// OptimisticLocking provides optimistic locking functionality
type OptimisticLocking struct {
    db *sql.DB
}

// NewOptimisticLocking creates a new optimistic locking instance
func NewOptimisticLocking(db *sql.DB) *OptimisticLocking {
    return &OptimisticLocking{db: db}
}

// UpdateWithOptimisticLock updates a record with optimistic locking
func (ol *OptimisticLocking) UpdateWithOptimisticLock(ctx context.Context, table string, id int64, currentVersion int64, updates map[string]interface{}) error {
    // Build update query
    query := fmt.Sprintf("UPDATE %s SET version = version + 1, updated_at = NOW()", table)
    args := []interface{}{}
    argIndex := 1
    
    for column, value := range updates {
        query += fmt.Sprintf(", %s = $%d", column, argIndex)
        args = append(args, value)
        argIndex++
    }
    
    query += fmt.Sprintf(" WHERE id = $%d AND version = $%d", argIndex, argIndex+1)
    args = append(args, id, currentVersion)
    
    result, err := ol.db.ExecContext(ctx, query, args...)
    if err != nil {
        return fmt.Errorf("failed to update record: %w", err)
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to get rows affected: %w", err)
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("record was modified by another process")
    }
    
    return nil
}

// GetWithOptimisticLock gets a record with optimistic locking
func (ol *OptimisticLocking) GetWithOptimisticLock(ctx context.Context, table string, id int64) (*OptimisticLock, error) {
    query := fmt.Sprintf("SELECT id, version, updated_at FROM %s WHERE id = $1", table)
    
    var lock OptimisticLock
    err := ol.db.QueryRowContext(ctx, query, id).Scan(
        &lock.ID,
        &lock.Version,
        &lock.UpdatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("record not found")
        }
        return nil, fmt.Errorf("failed to get record: %w", err)
    }
    
    return &lock, nil
}
```

## Query Optimization

### Query Builder

```go
// internal/database/query_builder.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "strings"
)

// QueryBuilder represents a query builder
type QueryBuilder struct {
    table     string
    columns   []string
    where     []string
    args      []interface{}
    orderBy   []string
    limit     int
    offset    int
    joins     []string
    groupBy   []string
    having    []string
}

// NewQueryBuilder creates a new query builder
func NewQueryBuilder(table string) *QueryBuilder {
    return &QueryBuilder{
        table:   table,
        columns: []string{"*"},
        where:   make([]string, 0),
        args:    make([]interface{}, 0),
        orderBy: make([]string, 0),
        joins:   make([]string, 0),
        groupBy: make([]string, 0),
        having:  make([]string, 0),
    }
}

// Select sets the columns to select
func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.columns = columns
    return qb
}

// Where adds a WHERE clause
func (qb *QueryBuilder) Where(condition string, args ...interface{}) *QueryBuilder {
    qb.where = append(qb.where, condition)
    qb.args = append(qb.args, args...)
    return qb
}

// Join adds a JOIN clause
func (qb *QueryBuilder) Join(table, condition string) *QueryBuilder {
    qb.joins = append(qb.joins, fmt.Sprintf("JOIN %s ON %s", table, condition))
    return qb
}

// LeftJoin adds a LEFT JOIN clause
func (qb *QueryBuilder) LeftJoin(table, condition string) *QueryBuilder {
    qb.joins = append(qb.joins, fmt.Sprintf("LEFT JOIN %s ON %s", table, condition))
    return qb
}

// OrderBy adds an ORDER BY clause
func (qb *QueryBuilder) OrderBy(column string, direction string) *QueryBuilder {
    qb.orderBy = append(qb.orderBy, fmt.Sprintf("%s %s", column, direction))
    return qb
}

// GroupBy adds a GROUP BY clause
func (qb *QueryBuilder) GroupBy(columns ...string) *QueryBuilder {
    qb.groupBy = append(qb.groupBy, columns...)
    return qb
}

// Having adds a HAVING clause
func (qb *QueryBuilder) Having(condition string, args ...interface{}) *QueryBuilder {
    qb.having = append(qb.having, condition)
    qb.args = append(qb.args, args...)
    return qb
}

// Limit sets the LIMIT
func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
    qb.limit = limit
    return qb
}

// Offset sets the OFFSET
func (qb *QueryBuilder) Offset(offset int) *QueryBuilder {
    qb.offset = offset
    return qb
}

// Build builds the SELECT query
func (qb *QueryBuilder) Build() (string, []interface{}) {
    query := fmt.Sprintf("SELECT %s FROM %s", strings.Join(qb.columns, ", "), qb.table)
    
    if len(qb.joins) > 0 {
        query += " " + strings.Join(qb.joins, " ")
    }
    
    if len(qb.where) > 0 {
        query += " WHERE " + strings.Join(qb.where, " AND ")
    }
    
    if len(qb.groupBy) > 0 {
        query += " GROUP BY " + strings.Join(qb.groupBy, ", ")
    }
    
    if len(qb.having) > 0 {
        query += " HAVING " + strings.Join(qb.having, " AND ")
    }
    
    if len(qb.orderBy) > 0 {
        query += " ORDER BY " + strings.Join(qb.orderBy, ", ")
    }
    
    if qb.limit > 0 {
        query += fmt.Sprintf(" LIMIT %d", qb.limit)
    }
    
    if qb.offset > 0 {
        query += fmt.Sprintf(" OFFSET %d", qb.offset)
    }
    
    return query, qb.args
}

// Execute executes the query
func (qb *QueryBuilder) Execute(ctx context.Context, db *sql.DB) (*sql.Rows, error) {
    query, args := qb.Build()
    return db.QueryContext(ctx, query, args...)
}

// ExecuteRow executes the query and returns a single row
func (qb *QueryBuilder) ExecuteRow(ctx context.Context, db *sql.DB) *sql.Row {
    query, args := qb.Build()
    return db.QueryRowContext(ctx, query, args...)
}
```

### Query Cache

```go
// internal/database/query_cache.go
package database

import (
    "context"
    "crypto/md5"
    "database/sql"
    "encoding/hex"
    "fmt"
    "sync"
    "time"
)

// QueryCache represents a query cache
type QueryCache struct {
    cache    map[string]CacheEntry
    mutex    sync.RWMutex
    ttl      time.Duration
    maxSize  int
}

// CacheEntry represents a cache entry
type CacheEntry struct {
    Data      interface{}
    Timestamp time.Time
    TTL       time.Duration
}

// NewQueryCache creates a new query cache
func NewQueryCache(ttl time.Duration, maxSize int) *QueryCache {
    cache := &QueryCache{
        cache:   make(map[string]CacheEntry),
        ttl:     ttl,
        maxSize: maxSize,
    }
    
    // Start cleanup goroutine
    go cache.cleanup()
    
    return cache
}

// Get gets a value from the cache
func (qc *QueryCache) Get(key string) (interface{}, bool) {
    qc.mutex.RLock()
    defer qc.mutex.RUnlock()
    
    entry, exists := qc.cache[key]
    if !exists {
        return nil, false
    }
    
    if time.Since(entry.Timestamp) > entry.TTL {
        return nil, false
    }
    
    return entry.Data, true
}

// Set sets a value in the cache
func (qc *QueryCache) Set(key string, value interface{}, ttl time.Duration) {
    qc.mutex.Lock()
    defer qc.mutex.Unlock()
    
    // Check if cache is full
    if len(qc.cache) >= qc.maxSize {
        qc.evictOldest()
    }
    
    qc.cache[key] = CacheEntry{
        Data:      value,
        Timestamp: time.Now(),
        TTL:       ttl,
    }
}

// evictOldest evicts the oldest entry
func (qc *QueryCache) evictOldest() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range qc.cache {
        if oldestKey == "" || entry.Timestamp.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.Timestamp
        }
    }
    
    if oldestKey != "" {
        delete(qc.cache, oldestKey)
    }
}

// cleanup removes expired entries
func (qc *QueryCache) cleanup() {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        qc.mutex.Lock()
        now := time.Now()
        for key, entry := range qc.cache {
            if now.Sub(entry.Timestamp) > entry.TTL {
                delete(qc.cache, key)
            }
        }
        qc.mutex.Unlock()
    }
}

// CachedQuery executes a cached query
func (qc *QueryCache) CachedQuery(ctx context.Context, db *sql.DB, query string, args ...interface{}) ([]map[string]interface{}, error) {
    // Generate cache key
    key := qc.generateKey(query, args...)
    
    // Check cache
    if cached, exists := qc.Get(key); exists {
        return cached.([]map[string]interface{}), nil
    }
    
    // Execute query
    rows, err := db.QueryContext(ctx, query, args...)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    // Convert to map
    result, err := qc.rowsToMap(rows)
    if err != nil {
        return nil, err
    }
    
    // Cache result
    qc.Set(key, result, qc.ttl)
    
    return result, nil
}

// generateKey generates a cache key
func (qc *QueryCache) generateKey(query string, args ...interface{}) string {
    data := fmt.Sprintf("%s%v", query, args)
    hash := md5.Sum([]byte(data))
    return hex.EncodeToString(hash[:])
}

// rowsToMap converts rows to map
func (qc *QueryCache) rowsToMap(rows *sql.Rows) ([]map[string]interface{}, error) {
    columns, err := rows.Columns()
    if err != nil {
        return nil, err
    }
    
    var result []map[string]interface{}
    
    for rows.Next() {
        values := make([]interface{}, len(columns))
        valuePtrs := make([]interface{}, len(columns))
        
        for i := range values {
            valuePtrs[i] = &values[i]
        }
        
        if err := rows.Scan(valuePtrs...); err != nil {
            return nil, err
        }
        
        row := make(map[string]interface{})
        for i, col := range columns {
            row[col] = values[i]
        }
        
        result = append(result, row)
    }
    
    return result, nil
}
```

## Migration Management

### Migration Manager

```go
// internal/database/migration_manager.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "sort"
    "time"
)

// Migration represents a database migration
type Migration struct {
    Version     int       `json:"version"`
    Name        string    `json:"name"`
    UpSQL       string    `json:"up_sql"`
    DownSQL     string    `json:"down_sql"`
    CreatedAt   time.Time `json:"created_at"`
    AppliedAt   *time.Time `json:"applied_at,omitempty"`
}

// MigrationManager represents a migration manager
type MigrationManager struct {
    db         *sql.DB
    migrations []Migration
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager(db *sql.DB) *MigrationManager {
    return &MigrationManager{
        db:         db,
        migrations: make([]Migration, 0),
    }
}

// AddMigration adds a migration
func (mm *MigrationManager) AddMigration(migration Migration) {
    mm.migrations = append(mm.migrations, migration)
    sort.Slice(mm.migrations, func(i, j int) bool {
        return mm.migrations[i].Version < mm.migrations[j].Version
    })
}

// Migrate runs all pending migrations
func (mm *MigrationManager) Migrate(ctx context.Context) error {
    // Create migrations table if it doesn't exist
    if err := mm.createMigrationsTable(ctx); err != nil {
        return fmt.Errorf("failed to create migrations table: %w", err)
    }
    
    // Get applied migrations
    applied, err := mm.getAppliedMigrations(ctx)
    if err != nil {
        return fmt.Errorf("failed to get applied migrations: %w", err)
    }
    
    // Run pending migrations
    for _, migration := range mm.migrations {
        if _, exists := applied[migration.Version]; !exists {
            if err := mm.runMigration(ctx, migration); err != nil {
                return fmt.Errorf("failed to run migration %d: %w", migration.Version, err)
            }
        }
    }
    
    return nil
}

// Rollback rolls back the last migration
func (mm *MigrationManager) Rollback(ctx context.Context) error {
    // Get applied migrations
    applied, err := mm.getAppliedMigrations(ctx)
    if err != nil {
        return fmt.Errorf("failed to get applied migrations: %w", err)
    }
    
    // Find the last applied migration
    var lastMigration *Migration
    for _, migration := range mm.migrations {
        if _, exists := applied[migration.Version]; exists {
            lastMigration = &migration
        }
    }
    
    if lastMigration == nil {
        return fmt.Errorf("no migrations to rollback")
    }
    
    // Rollback the migration
    if err := mm.rollbackMigration(ctx, *lastMigration); err != nil {
        return fmt.Errorf("failed to rollback migration %d: %w", lastMigration.Version, err)
    }
    
    return nil
}

// createMigrationsTable creates the migrations table
func (mm *MigrationManager) createMigrationsTable(ctx context.Context) error {
    query := `
        CREATE TABLE IF NOT EXISTS migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    `
    
    _, err := mm.db.ExecContext(ctx, query)
    return err
}

// getAppliedMigrations gets applied migrations
func (mm *MigrationManager) getAppliedMigrations(ctx context.Context) (map[int]bool, error) {
    query := "SELECT version FROM migrations ORDER BY version"
    rows, err := mm.db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    applied := make(map[int]bool)
    for rows.Next() {
        var version int
        if err := rows.Scan(&version); err != nil {
            return nil, err
        }
        applied[version] = true
    }
    
    return applied, nil
}

// runMigration runs a migration
func (mm *MigrationManager) runMigration(ctx context.Context, migration Migration) error {
    // Start transaction
    tx, err := mm.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    // Run migration SQL
    if _, err := tx.ExecContext(ctx, migration.UpSQL); err != nil {
        return err
    }
    
    // Record migration
    query := "INSERT INTO migrations (version, name) VALUES ($1, $2)"
    if _, err := tx.ExecContext(ctx, query, migration.Version, migration.Name); err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}

// rollbackMigration rolls back a migration
func (mm *MigrationManager) rollbackMigration(ctx context.Context, migration Migration) error {
    // Start transaction
    tx, err := mm.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    // Run rollback SQL
    if _, err := tx.ExecContext(ctx, migration.DownSQL); err != nil {
        return err
    }
    
    // Remove migration record
    query := "DELETE FROM migrations WHERE version = $1"
    if _, err := tx.ExecContext(ctx, query, migration.Version); err != nil {
        return err
    }
    
    // Commit transaction
    return tx.Commit()
}
```

## ORM Patterns

### Simple ORM

```go
// internal/database/orm.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "reflect"
    "strings"
    "time"
)

// ORM represents a simple ORM
type ORM struct {
    db *sql.DB
}

// NewORM creates a new ORM
func NewORM(db *sql.DB) *ORM {
    return &ORM{db: db}
}

// Model represents a database model
type Model interface {
    TableName() string
    PrimaryKey() string
    Fields() map[string]interface{}
    SetFields(fields map[string]interface{})
}

// User represents a user model
type User struct {
    ID        int64     `json:"id" db:"id"`
    Name      string    `json:"name" db:"name"`
    Email     string    `json:"email" db:"email"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
    UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// TableName returns the table name
func (u *User) TableName() string {
    return "users"
}

// PrimaryKey returns the primary key field
func (u *User) PrimaryKey() string {
    return "id"
}

// Fields returns the model fields
func (u *User) Fields() map[string]interface{} {
    return map[string]interface{}{
        "id":         u.ID,
        "name":       u.Name,
        "email":      u.Email,
        "created_at": u.CreatedAt,
        "updated_at": u.UpdatedAt,
    }
}

// SetFields sets the model fields
func (u *User) SetFields(fields map[string]interface{}) {
    if id, ok := fields["id"].(int64); ok {
        u.ID = id
    }
    if name, ok := fields["name"].(string); ok {
        u.Name = name
    }
    if email, ok := fields["email"].(string); ok {
        u.Email = email
    }
    if createdAt, ok := fields["created_at"].(time.Time); ok {
        u.CreatedAt = createdAt
    }
    if updatedAt, ok := fields["updated_at"].(time.Time); ok {
        u.UpdatedAt = updatedAt
    }
}

// Create creates a new record
func (orm *ORM) Create(ctx context.Context, model Model) error {
    fields := model.Fields()
    delete(fields, model.PrimaryKey()) // Remove primary key for insert
    
    columns := make([]string, 0, len(fields))
    values := make([]interface{}, 0, len(fields))
    placeholders := make([]string, 0, len(fields))
    
    for column, value := range fields {
        columns = append(columns, column)
        values = append(values, value)
        placeholders = append(placeholders, "?")
    }
    
    query := fmt.Sprintf(
        "INSERT INTO %s (%s) VALUES (%s)",
        model.TableName(),
        strings.Join(columns, ", "),
        strings.Join(placeholders, ", "),
    )
    
    result, err := orm.db.ExecContext(ctx, query, values...)
    if err != nil {
        return err
    }
    
    id, err := result.LastInsertId()
    if err != nil {
        return err
    }
    
    // Set the ID on the model
    fields[model.PrimaryKey()] = id
    model.SetFields(fields)
    
    return nil
}

// Find finds a record by ID
func (orm *ORM) Find(ctx context.Context, model Model, id interface{}) error {
    query := fmt.Sprintf("SELECT * FROM %s WHERE %s = ?", model.TableName(), model.PrimaryKey())
    
    row := orm.db.QueryRowContext(ctx, query, id)
    
    // Get column names
    columns, err := row.Columns()
    if err != nil {
        return err
    }
    
    // Create slice of pointers
    values := make([]interface{}, len(columns))
    valuePtrs := make([]interface{}, len(columns))
    for i := range values {
        valuePtrs[i] = &values[i]
    }
    
    // Scan the row
    if err := row.Scan(valuePtrs...); err != nil {
        return err
    }
    
    // Create fields map
    fields := make(map[string]interface{})
    for i, column := range columns {
        fields[column] = values[i]
    }
    
    model.SetFields(fields)
    return nil
}

// Update updates a record
func (orm *ORM) Update(ctx context.Context, model Model) error {
    fields := model.Fields()
    id := fields[model.PrimaryKey()]
    delete(fields, model.PrimaryKey()) // Remove primary key from update
    
    columns := make([]string, 0, len(fields))
    values := make([]interface{}, 0, len(fields))
    
    for column, value := range fields {
        columns = append(columns, fmt.Sprintf("%s = ?", column))
        values = append(values, value)
    }
    
    query := fmt.Sprintf(
        "UPDATE %s SET %s WHERE %s = ?",
        model.TableName(),
        strings.Join(columns, ", "),
        model.PrimaryKey(),
    )
    
    values = append(values, id)
    
    _, err := orm.db.ExecContext(ctx, query, values...)
    return err
}

// Delete deletes a record
func (orm *ORM) Delete(ctx context.Context, model Model) error {
    fields := model.Fields()
    id := fields[model.PrimaryKey()]
    
    query := fmt.Sprintf("DELETE FROM %s WHERE %s = ?", model.TableName(), model.PrimaryKey())
    
    _, err := orm.db.ExecContext(ctx, query, id)
    return err
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Connection pooling
config := PoolConfig{
    MaxOpenConns:    100,
    MaxIdleConns:    10,
    ConnMaxLifetime: time.Hour,
    ConnMaxIdleTime: time.Minute * 10,
}
pool, err := NewConnectionPool("postgres", dsn, config)

// 2. Transaction management
txManager := NewTransactionManager(pool.GetDB())
err := txManager.WithTransaction(ctx, func(tx *Transaction) error {
    // Perform database operations
    return nil
})

// 3. Query optimization
qb := NewQueryBuilder("users").
    Select("id", "name", "email").
    Where("active = ?", true).
    OrderBy("created_at", "DESC").
    Limit(10)

// 4. Migration management
migrationManager := NewMigrationManager(pool.GetDB())
migrationManager.AddMigration(Migration{
    Version: 1,
    Name:    "create_users_table",
    UpSQL:   "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255))",
    DownSQL: "DROP TABLE users",
})
err := migrationManager.Migrate(ctx)
```

### Essential Patterns

```go
// Prepared statements
cache := NewPreparedStatementCache(pool.GetDB())
result, err := cache.Execute(ctx, "INSERT INTO users (name, email) VALUES (?, ?)", name, email)

// Optimistic locking
locking := NewOptimisticLocking(pool.GetDB())
err := locking.UpdateWithOptimisticLock(ctx, "users", userID, currentVersion, updates)

// Query caching
queryCache := NewQueryCache(5*time.Minute, 1000)
results, err := queryCache.CachedQuery(ctx, pool.GetDB(), "SELECT * FROM users WHERE active = ?", true)

// ORM usage
orm := NewORM(pool.GetDB())
user := &User{Name: "John", Email: "john@example.com"}
err := orm.Create(ctx, user)
```

---

*This guide provides the complete machinery for building robust database applications in Go. Each pattern includes implementation examples, performance considerations, and real-world usage patterns for enterprise deployment.*
