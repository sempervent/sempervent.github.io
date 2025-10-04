# Python Database Patterns Best Practices

**Objective**: Master senior-level Python database patterns for production systems. When you need to build robust database applications, when you want to implement efficient connection pooling, when you need enterprise-grade database strategies—these best practices become your weapon of choice.

## Core Principles

- **Connection Management**: Efficient connection pooling and lifecycle management
- **Transaction Safety**: ACID compliance and transaction isolation
- **Performance**: Query optimization and database tuning
- **Reliability**: Error handling and recovery patterns
- **Security**: SQL injection prevention and data protection

## Database Connection Management

### Connection Pooling

```python
# python/01-connection-pooling.py

"""
Database connection pooling patterns and management
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
import logging
from queue import Queue, Empty
import weakref
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncpg
import aioredis
from redis import ConnectionPool as RedisConnectionPool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    BUSY = "busy"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class ConnectionMetrics:
    """Connection metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_requests: int = 0
    connection_wait_time: float = 0.0
    last_activity: datetime = None

class DatabaseConnectionPool:
    """Advanced database connection pool"""
    
    def __init__(self, connection_string: str, min_connections: int = 5, 
                 max_connections: int = 20, connection_timeout: int = 30):
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pool = None
        self.metrics = ConnectionMetrics()
        self.connection_lock = threading.Lock()
        self.connections = {}
        self.initialize_pool()
    
    def initialize_pool(self) -> None:
        """Initialize connection pool"""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                dsn=self.connection_string,
                connect_timeout=self.connection_timeout
            )
            logger.info(f"Connection pool initialized with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self) -> psycopg2.extensions.connection:
        """Get connection from pool"""
        start_time = time.time()
        
        try:
            connection = self.pool.getconn()
            if connection:
                self.metrics.active_connections += 1
                self.metrics.connection_requests += 1
                self.metrics.connection_wait_time += time.time() - start_time
                self.metrics.last_activity = datetime.utcnow()
                
                # Store connection reference
                with self.connection_lock:
                    self.connections[id(connection)] = {
                        'connection': connection,
                        'state': ConnectionState.BUSY,
                        'created_at': datetime.utcnow()
                    }
                
                logger.debug(f"Connection acquired: {id(connection)}")
                return connection
            else:
                raise Exception("Failed to get connection from pool")
        
        except Exception as e:
            self.metrics.failed_connections += 1
            logger.error(f"Failed to get connection: {e}")
            raise
    
    def return_connection(self, connection: psycopg2.extensions.connection) -> None:
        """Return connection to pool"""
        try:
            connection_id = id(connection)
            
            with self.connection_lock:
                if connection_id in self.connections:
                    self.connections[connection_id]['state'] = ConnectionState.IDLE
                    del self.connections[connection_id]
            
            self.pool.putconn(connection)
            self.metrics.active_connections -= 1
            self.metrics.idle_connections += 1
            
            logger.debug(f"Connection returned: {connection_id}")
        
        except Exception as e:
            logger.error(f"Failed to return connection: {e}")
    
    def close_connection(self, connection: psycopg2.extensions.connection) -> None:
        """Close connection"""
        try:
            connection_id = id(connection)
            
            with self.connection_lock:
                if connection_id in self.connections:
                    self.connections[connection_id]['state'] = ConnectionState.CLOSED
                    del self.connections[connection_id]
            
            connection.close()
            self.metrics.active_connections -= 1
            
            logger.debug(f"Connection closed: {connection_id}")
        
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status and metrics"""
        with self.connection_lock:
            active_connections = len([conn for conn in self.connections.values() 
                                       if conn['state'] == ConnectionState.BUSY])
            idle_connections = len([conn for conn in self.connections.values() 
                                  if conn['state'] == ConnectionState.IDLE])
        
        return {
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "active_connections": active_connections,
            "idle_connections": idle_connections,
            "total_connections": len(self.connections),
            "failed_connections": self.metrics.failed_connections,
            "connection_requests": self.metrics.connection_requests,
            "avg_wait_time": self.metrics.connection_wait_time / max(self.metrics.connection_requests, 1),
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
        }
    
    def close_pool(self) -> None:
        """Close all connections in pool"""
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("Connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close connection pool: {e}")

class AsyncDatabasePool:
    """Async database connection pool"""
    
    def __init__(self, connection_string: str, min_connections: int = 5,
                 max_connections: int = 20):
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pool = None
        self.metrics = ConnectionMetrics()
        self.connection_lock = asyncio.Lock()
        self.connections = {}
    
    async def initialize_pool(self) -> None:
        """Initialize async connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60
            )
            logger.info(f"Async connection pool initialized with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize async connection pool: {e}")
            raise
    
    async def get_connection(self) -> asyncpg.Connection:
        """Get async connection from pool"""
        start_time = time.time()
        
        try:
            connection = await self.pool.acquire()
            self.metrics.active_connections += 1
            self.metrics.connection_requests += 1
            self.metrics.connection_wait_time += time.time() - start_time
            self.metrics.last_activity = datetime.utcnow()
            
            logger.debug(f"Async connection acquired: {id(connection)}")
            return connection
        
        except Exception as e:
            self.metrics.failed_connections += 1
            logger.error(f"Failed to get async connection: {e}")
            raise
    
    async def return_connection(self, connection: asyncpg.Connection) -> None:
        """Return async connection to pool"""
        try:
            await self.pool.release(connection)
            self.metrics.active_connections -= 1
            self.metrics.idle_connections += 1
            
            logger.debug(f"Async connection returned: {id(connection)}")
        
        except Exception as e:
            logger.error(f"Failed to return async connection: {e}")
    
    async def close_pool(self) -> None:
        """Close async connection pool"""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Async connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close async connection pool: {e}")

class SQLAlchemyConnectionManager:
    """SQLAlchemy connection management"""
    
    def __init__(self, connection_string: str, pool_size: int = 10,
                 max_overflow: int = 20, pool_timeout: int = 30):
        self.connection_string = connection_string
        self.engine = None
        self.session_factory = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.initialize_engine()
    
    def initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False
            )
            
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info(f"SQLAlchemy engine initialized with pool size {self.pool_size}")
        
        except Exception as e:
            logger.error(f"Failed to initialize SQLAlchemy engine: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine and pool status"""
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }

class AsyncSQLAlchemyManager:
    """Async SQLAlchemy connection management"""
    
    def __init__(self, connection_string: str, pool_size: int = 10,
                 max_overflow: int = 20):
        self.connection_string = connection_string
        self.engine = None
        self.session_factory = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.initialize_engine()
    
    def initialize_engine(self) -> None:
        """Initialize async SQLAlchemy engine"""
        try:
            self.engine = create_async_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info(f"Async SQLAlchemy engine initialized with pool size {self.pool_size}")
        
        except Exception as e:
            logger.error(f"Failed to initialize async SQLAlchemy engine: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get async database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()

class RedisConnectionManager:
    """Redis connection management"""
    
    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, max_connections: int = 20):
        self.host = host
        self.port = port
        self.db = db
        self.max_connections = max_connections
        self.pool = None
        self.initialize_pool()
    
    def initialize_pool(self) -> None:
        """Initialize Redis connection pool"""
        try:
            self.pool = RedisConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                max_connections=self.max_connections,
                retry_on_timeout=True
            )
            logger.info(f"Redis connection pool initialized with {self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get Redis connection from pool"""
        try:
            from redis import Redis
            return Redis(connection_pool=self.pool)
        except Exception as e:
            logger.error(f"Failed to get Redis connection: {e}")
            raise
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get Redis pool status"""
        return {
            "max_connections": self.max_connections,
            "created_connections": self.pool.created_connections,
            "available_connections": self.pool.available_connections,
            "in_use_connections": self.pool.in_use_connections
        }

# Usage examples
def example_connection_pooling():
    """Example connection pooling usage"""
    # PostgreSQL connection pool
    pg_pool = DatabaseConnectionPool(
        connection_string="postgresql://user:pass@localhost/db",
        min_connections=5,
        max_connections=20
    )
    
    # Get connection
    connection = pg_pool.get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"Query result: {result}")
    finally:
        pg_pool.return_connection(connection)
    
    # Get pool status
    status = pg_pool.get_pool_status()
    print(f"Pool status: {status}")
    
    # SQLAlchemy connection manager
    sqlalchemy_manager = SQLAlchemyConnectionManager(
        connection_string="postgresql://user:pass@localhost/db",
        pool_size=10,
        max_overflow=20
    )
    
    # Use session
    with sqlalchemy_manager.get_session() as session:
        result = session.execute(text("SELECT 1")).fetchone()
        print(f"SQLAlchemy result: {result}")
    
    # Redis connection manager
    redis_manager = RedisConnectionManager(
        host="localhost",
        port=6379,
        max_connections=20
    )
    
    # Use Redis connection
    redis_client = redis_manager.get_connection()
    redis_client.set("test_key", "test_value")
    value = redis_client.get("test_key")
    print(f"Redis value: {value}")
    
    # Close pools
    pg_pool.close_pool()
```

### Transaction Management

```python
# python/02-transaction-management.py

"""
Database transaction management patterns and ACID compliance
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
import logging
from functools import wraps
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED, ISOLATION_LEVEL_REPEATABLE_READ
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import asyncpg
from asyncpg import Connection

logger = logging.getLogger(__name__)

class TransactionIsolation(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"

class TransactionManager:
    """Database transaction manager"""
    
    def __init__(self, connection_pool: DatabaseConnectionPool):
        self.connection_pool = connection_pool
        self.transaction_metrics = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "rollback_count": 0,
            "avg_transaction_time": 0.0
        }
    
    @contextmanager
    def transaction(self, isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED):
        """Context manager for database transactions"""
        connection = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = self.connection_pool.get_connection()
            
            # Set isolation level
            if isolation_level == TransactionIsolation.READ_COMMITTED:
                connection.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
            elif isolation_level == TransactionIsolation.REPEATABLE_READ:
                connection.set_isolation_level(ISOLATION_LEVEL_REPEATABLE_READ)
            
            # Start transaction
            connection.autocommit = False
            
            self.transaction_metrics["total_transactions"] += 1
            
            yield connection
            
            # Commit transaction
            connection.commit()
            self.transaction_metrics["successful_transactions"] += 1
            
            transaction_time = time.time() - start_time
            self._update_avg_transaction_time(transaction_time)
            
            logger.debug(f"Transaction committed in {transaction_time:.3f}s")
        
        except Exception as e:
            # Rollback transaction
            if connection:
                connection.rollback()
                self.transaction_metrics["rollback_count"] += 1
                logger.error(f"Transaction rolled back: {e}")
            
            self.transaction_metrics["failed_transactions"] += 1
            raise
        
        finally:
            # Return connection to pool
            if connection:
                self.connection_pool.return_connection(connection)
    
    def _update_avg_transaction_time(self, transaction_time: float) -> None:
        """Update average transaction time"""
        total_transactions = self.transaction_metrics["total_transactions"]
        current_avg = self.transaction_metrics["avg_transaction_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_transactions - 1)) + transaction_time) / total_transactions
        self.transaction_metrics["avg_transaction_time"] = new_avg
    
    def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get transaction metrics"""
        total = self.transaction_metrics["total_transactions"]
        success_rate = (self.transaction_metrics["successful_transactions"] / total * 100) if total > 0 else 0
        
        return {
            **self.transaction_metrics,
            "success_rate": success_rate
        }

class AsyncTransactionManager:
    """Async database transaction manager"""
    
    def __init__(self, connection_pool: AsyncDatabasePool):
        self.connection_pool = connection_pool
        self.transaction_metrics = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "rollback_count": 0,
            "avg_transaction_time": 0.0
        }
    
    @asynccontextmanager
    async def transaction(self, isolation_level: str = "READ COMMITTED"):
        """Async context manager for database transactions"""
        connection = None
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = await self.connection_pool.get_connection()
            
            # Start transaction
            async with connection.transaction(isolation=isolation_level):
                self.transaction_metrics["total_transactions"] += 1
                
                yield connection
                
                # Transaction will be committed automatically
                self.transaction_metrics["successful_transactions"] += 1
                
                transaction_time = time.time() - start_time
                self._update_avg_transaction_time(transaction_time)
                
                logger.debug(f"Async transaction committed in {transaction_time:.3f}s")
        
        except Exception as e:
            # Transaction will be rolled back automatically
            self.transaction_metrics["failed_transactions"] += 1
            self.transaction_metrics["rollback_count"] += 1
            logger.error(f"Async transaction rolled back: {e}")
            raise
        
        finally:
            # Return connection to pool
            if connection:
                await self.connection_pool.return_connection(connection)
    
    def _update_avg_transaction_time(self, transaction_time: float) -> None:
        """Update average transaction time"""
        total_transactions = self.transaction_metrics["total_transactions"]
        current_avg = self.transaction_metrics["avg_transaction_time"]
        
        new_avg = ((current_avg * (total_transactions - 1)) + transaction_time) / total_transactions
        self.transaction_metrics["avg_transaction_time"] = new_avg
    
    def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get transaction metrics"""
        total = self.transaction_metrics["total_transactions"]
        success_rate = (self.transaction_metrics["successful_transactions"] / total * 100) if total > 0 else 0
        
        return {
            **self.transaction_metrics,
            "success_rate": success_rate
        }

class SQLAlchemyTransactionManager:
    """SQLAlchemy transaction manager"""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.transaction_metrics = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "rollback_count": 0,
            "avg_transaction_time": 0.0
        }
    
    @contextmanager
    def transaction(self):
        """SQLAlchemy transaction context manager"""
        session = self.session_factory()
        start_time = time.time()
        
        try:
            self.transaction_metrics["total_transactions"] += 1
            
            yield session
            
            # Commit transaction
            session.commit()
            self.transaction_metrics["successful_transactions"] += 1
            
            transaction_time = time.time() - start_time
            self._update_avg_transaction_time(transaction_time)
            
            logger.debug(f"SQLAlchemy transaction committed in {transaction_time:.3f}s")
        
        except SQLAlchemyError as e:
            # Rollback transaction
            session.rollback()
            self.transaction_metrics["rollback_count"] += 1
            self.transaction_metrics["failed_transactions"] += 1
            logger.error(f"SQLAlchemy transaction rolled back: {e}")
            raise
        
        finally:
            session.close()
    
    def _update_avg_transaction_time(self, transaction_time: float) -> None:
        """Update average transaction time"""
        total_transactions = self.transaction_metrics["total_transactions"]
        current_avg = self.transaction_metrics["avg_transaction_time"]
        
        new_avg = ((current_avg * (total_transactions - 1)) + transaction_time) / total_transactions
        self.transaction_metrics["avg_transaction_time"] = new_avg
    
    def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get transaction metrics"""
        total = self.transaction_metrics["total_transactions"]
        success_rate = (self.transaction_metrics["successful_transactions"] / total * 100) if total > 0 else 0
        
        return {
            **self.transaction_metrics,
            "success_rate": success_rate
        }

def transactional(isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED):
    """Decorator for transactional methods"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be implemented with actual transaction manager
            # For now, just execute the function
            return func(*args, **kwargs)
        return wrapper
    return decorator

class DatabaseRepository:
    """Repository pattern for database operations"""
    
    def __init__(self, transaction_manager: TransactionManager):
        self.transaction_manager = transaction_manager
    
    def create_user(self, user_data: Dict[str, Any]) -> int:
        """Create user with transaction"""
        with self.transaction_manager.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert user
            cursor.execute("""
                INSERT INTO users (name, email, created_at) 
                VALUES (%s, %s, %s) RETURNING id
            """, (user_data['name'], user_data['email'], datetime.utcnow()))
            
            user_id = cursor.fetchone()[0]
            
            # Insert user profile
            cursor.execute("""
                INSERT INTO user_profiles (user_id, bio, avatar_url) 
                VALUES (%s, %s, %s)
            """, (user_id, user_data.get('bio', ''), user_data.get('avatar_url', '')))
            
            return user_id
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with self.transaction_manager.transaction() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.id, u.name, u.email, u.created_at, 
                       p.bio, p.avatar_url
                FROM users u
                LEFT JOIN user_profiles p ON u.id = p.user_id
                WHERE u.id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'name': result[1],
                    'email': result[2],
                    'created_at': result[3],
                    'bio': result[4],
                    'avatar_url': result[5]
                }
            return None
    
    def update_user(self, user_id: int, user_data: Dict[str, Any]) -> bool:
        """Update user with transaction"""
        with self.transaction_manager.transaction() as conn:
            cursor = conn.cursor()
            
            # Update user
            cursor.execute("""
                UPDATE users 
                SET name = %s, email = %s, updated_at = %s
                WHERE id = %s
            """, (user_data['name'], user_data['email'], datetime.utcnow(), user_id))
            
            # Update user profile
            cursor.execute("""
                UPDATE user_profiles 
                SET bio = %s, avatar_url = %s
                WHERE user_id = %s
            """, (user_data.get('bio', ''), user_data.get('avatar_url', ''), user_id))
            
            return cursor.rowcount > 0
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user with transaction"""
        with self.transaction_manager.transaction() as conn:
            cursor = conn.cursor()
            
            # Delete user profile first (foreign key constraint)
            cursor.execute("DELETE FROM user_profiles WHERE user_id = %s", (user_id,))
            
            # Delete user
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            
            return cursor.rowcount > 0

class DatabaseMigrationManager:
    """Database migration management"""
    
    def __init__(self, connection_pool: DatabaseConnectionPool):
        self.connection_pool = connection_pool
        self.migrations = []
        self.initialize_migrations_table()
    
    def initialize_migrations_table(self) -> None:
        """Initialize migrations tracking table"""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def add_migration(self, version: str, sql: str) -> None:
        """Add migration"""
        self.migrations.append({
            'version': version,
            'sql': sql,
            'applied': False
        })
    
    def run_migrations(self) -> List[str]:
        """Run pending migrations"""
        applied_migrations = []
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get applied migrations
            cursor.execute("SELECT version FROM schema_migrations")
            applied_versions = {row[0] for row in cursor.fetchall()}
            
            # Run pending migrations
            for migration in self.migrations:
                if migration['version'] not in applied_versions:
                    try:
                        cursor.execute(migration['sql'])
                        cursor.execute(
                            "INSERT INTO schema_migrations (version) VALUES (%s)",
                            (migration['version'],)
                        )
                        conn.commit()
                        applied_migrations.append(migration['version'])
                        logger.info(f"Applied migration: {migration['version']}")
                    
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Failed to apply migration {migration['version']}: {e}")
                        raise
        
        return applied_migrations

# Usage examples
def example_transaction_management():
    """Example transaction management usage"""
    # Create connection pool
    connection_pool = DatabaseConnectionPool("postgresql://user:pass@localhost/db")
    
    # Create transaction manager
    transaction_manager = TransactionManager(connection_pool)
    
    # Use transaction
    with transaction_manager.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", ("John", "john@example.com"))
        cursor.execute("INSERT INTO user_profiles (user_id, bio) VALUES (%s, %s)", (1, "Software Developer"))
    
    # Get transaction metrics
    metrics = transaction_manager.get_transaction_metrics()
    print(f"Transaction metrics: {metrics}")
    
    # Repository pattern
    repository = DatabaseRepository(transaction_manager)
    
    # Create user
    user_data = {
        'name': 'Jane Doe',
        'email': 'jane@example.com',
        'bio': 'Data Scientist',
        'avatar_url': 'https://example.com/avatar.jpg'
    }
    user_id = repository.create_user(user_data)
    print(f"Created user with ID: {user_id}")
    
    # Get user
    user = repository.get_user(user_id)
    print(f"Retrieved user: {user}")
    
    # Database migrations
    migration_manager = DatabaseMigrationManager(connection_pool)
    
    # Add migration
    migration_manager.add_migration("001_create_users_table", """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Run migrations
    applied_migrations = migration_manager.run_migrations()
    print(f"Applied migrations: {applied_migrations}")
    
    # Close connection pool
    connection_pool.close_pool()
```

## TL;DR Runbook

### Quick Start

```python
# 1. Connection pooling
pg_pool = DatabaseConnectionPool("postgresql://user:pass@localhost/db")
connection = pg_pool.get_connection()
pg_pool.return_connection(connection)

# 2. Transaction management
transaction_manager = TransactionManager(pg_pool)
with transaction_manager.transaction() as conn:
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name) VALUES (%s)", ("John",))

# 3. SQLAlchemy integration
sqlalchemy_manager = SQLAlchemyConnectionManager("postgresql://user:pass@localhost/db")
with sqlalchemy_manager.get_session() as session:
    result = session.execute(text("SELECT 1")).fetchone()

# 4. Repository pattern
repository = DatabaseRepository(transaction_manager)
user_id = repository.create_user({"name": "John", "email": "john@example.com"})

# 5. Database migrations
migration_manager = DatabaseMigrationManager(pg_pool)
migration_manager.add_migration("001_create_table", "CREATE TABLE users (id SERIAL PRIMARY KEY)")
applied = migration_manager.run_migrations()
```

### Essential Patterns

```python
# Complete database setup
def setup_database_patterns():
    """Setup complete database patterns environment"""
    
    # Connection pooling
    pg_pool = DatabaseConnectionPool("postgresql://user:pass@localhost/db")
    async_pool = AsyncDatabasePool("postgresql://user:pass@localhost/db")
    
    # Transaction management
    transaction_manager = TransactionManager(pg_pool)
    async_transaction_manager = AsyncTransactionManager(async_pool)
    
    # SQLAlchemy integration
    sqlalchemy_manager = SQLAlchemyConnectionManager("postgresql://user:pass@localhost/db")
    async_sqlalchemy_manager = AsyncSQLAlchemyManager("postgresql://user:pass@localhost/db")
    
    # Redis connection
    redis_manager = RedisConnectionManager("localhost", 6379)
    
    # Repository pattern
    repository = DatabaseRepository(transaction_manager)
    
    # Migration management
    migration_manager = DatabaseMigrationManager(pg_pool)
    
    print("Database patterns setup complete!")
```

---

*This guide provides the complete machinery for Python database patterns. Each pattern includes implementation examples, connection strategies, and real-world usage patterns for enterprise database management.*
