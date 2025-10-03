# Migrating from psycopg2 to psycopg ≥ 3: A Brutally Practical Guide

**Objective**: Migrate your psycopg2 codebase to psycopg 3 with surgical precision. When your codebase is glued together with psycopg2, when you need async, safer adapters, better COPY, and cleaner ergonomics—psycopg 3 is the modern rewrite. This guide walks you through the migration with minimal downtime and no mystery meat.

You've got a codebase glued together with psycopg2. It works—until you need async, safer adapters, better COPY, and cleaner ergonomics. psycopg 3 is the modern rewrite. This tutorial walks you—surgically—through the migration with minimal downtime and no mystery meat.

## 0) TL;DR: What Changes, What Doesn't

### The Essentials

```bash
# Package name: psycopg2 → psycopg (aka Psycopg 3)
# Install fast wheel: pip install "psycopg[binary]" (or pip install psycopg-binary)
# Source/C extension build: pip install "psycopg[c]"
# Imports: import psycopg (not psycopg2)
# Connect: still psycopg.connect(...)
# Placeholders: still %s / %(name)s (not ?, not f-strings)
# Exceptions: base is psycopg.Error; rich tree under psycopg.errors.*
# Row access: opt-in row factories (psycopg.rows.dict_row etc.)
# Async: native via psycopg.AsyncConnection / AsyncCursor
# COPY: a first-class API (cursor.copy(...)) instead of fragile copy_expert
# Pooling: external lib psycopg_pool
# Server-side cursors: conn.cursor(name="…")
# Migrations effort: mostly import/row handling/exception names + a couple of APIs
```

**Why This Matters**: psycopg 3 provides better performance, native async support, safer type adaptation, and cleaner APIs. The migration is mostly mechanical with some strategic improvements.

## 1) Installation Matrix

### Pick Your Poison

```bash
# Most users (wheel with C bits bundled; fast)
pip install "psycopg[binary]"

# Or: build C extension from source (needs build deps)
pip install "psycopg[c]"

# If you need pooling
pip install psycopg_pool

# On Alpine/musl: prefer the [c] build with proper build deps
# The prebuilt binary wheels target glibc
```

### Build Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential libpq-dev

# CentOS/RHEL
sudo yum install gcc python3-devel postgresql-devel

# Alpine
apk add --no-cache gcc musl-dev postgresql-dev

# macOS
brew install postgresql
```

**Why These Options**: Binary wheels are fastest for most users. Source builds provide better compatibility with musl-based systems like Alpine Linux.

## 2) Minimal Migration: Before/After

### 2.1 Connect + Simple Query

**Before (psycopg2):**

```python
import psycopg2

conn = psycopg2.connect("dbname=app user=app password=secret host=127.0.0.1")
cur = conn.cursor()
cur.execute("SELECT id, email FROM users WHERE id = %s", (42,))
row = cur.fetchone()
conn.commit()
cur.close()
conn.close()
```

**After (psycopg 3, same sync style):**

```python
import psycopg
from psycopg.rows import tuple_row  # default; shown for clarity

with psycopg.connect("dbname=app user=app password=secret host=127.0.0.1") as conn:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SELECT id, email FROM users WHERE id = %s", (42,))
        row = cur.fetchone()
# context managers commit on success, roll back on exception
```

**Key Differences:**
- Use `with` blocks: auto-commit on success, auto-rollback on exception
- Row factories are explicit (see §5)

### 2.2 Connection String Formats

```python
# All these work the same
conninfo = "postgresql://user:pass@host:5432/dbname"
conninfo = "host=localhost port=5432 dbname=app user=app password=secret"
conninfo = {
    "host": "localhost",
    "port": 5432,
    "dbname": "app",
    "user": "app",
    "password": "secret"
}

with psycopg.connect(conninfo) as conn:
    # Same connection behavior
    pass
```

**Why This Consistency**: psycopg 3 maintains the same connection interface as psycopg2, making migration straightforward for basic operations.

## 3) Async (New Superpower)

### Native Async Support

**If you've been faking async around psycopg2, stop. Use native async:**

```python
import asyncio
import psycopg
from psycopg.rows import dict_row

async def main():
    async with await psycopg.AsyncConnection.connect("postgresql://app:secret@127.0.0.1/app") as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email FROM users WHERE active = %s", (True,))
            rows = await cur.fetchall()
            return rows

rows = asyncio.run(main())
```

**Key Points:**
- Await `connect`, `execute`, `fetch*`
- Same SQL placeholders (`%s`)
- Works with any asyncio event loop (e.g., FastAPI, Starlette)

### Async with FastAPI

```python
from fastapi import FastAPI
import psycopg
from psycopg.rows import dict_row

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with await psycopg.AsyncConnection.connect("postgresql://app:secret@127.0.0.1/app") as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
            user = await cur.fetchone()
            return user or {"error": "not found"}
```

**Why Native Async**: No more threading hacks or asyncpg complexity. Native async support with the same API patterns you know.

## 4) Transactions: Explicit, Controlled, Safer

### Transaction Context Managers

**You can still use `conn.autocommit = True`, but the new transaction context is cleaner:**

```python
import psycopg

with psycopg.connect(conninfo) as conn:
    with conn.transaction():  # begins a tx; commits/rolls back automatically
        with conn.cursor() as cur:
            cur.execute("INSERT INTO events(kind, payload) VALUES (%s, %s)", ("signup", {"id": 1}))
```

### Nested Transactions (Savepoints)

```python
with psycopg.connect(conninfo) as conn:
    with conn.transaction():  # outer transaction
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users(email) VALUES (%s)", ("user@example.com",))
            
            try:
                with conn.transaction():  # savepoint
                    cur.execute("INSERT INTO profiles(user_id, name) VALUES (%s, %s)", (1, "John"))
            except psycopg.Error:
                # savepoint rolled back, outer transaction continues
                pass
```

**Why Transaction Contexts**: Automatic commit/rollback reduces boilerplate and prevents forgotten transactions. Nested transactions enable partial rollbacks.

## 5) Row Factories (Goodbye Tuple Unpacking Everywhere)

### Pick How Rows Are Returned

```python
from psycopg.rows import tuple_row, dict_row, namedtuple_row, class_row

# Dict rows
with psycopg.connect(conninfo) as conn:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, email FROM users")
        r = cur.fetchone()  # {'id': 1, 'email': '…'}

# Namedtuple rows
with psycopg.connect(conninfo) as conn:
    with conn.cursor(row_factory=namedtuple_row) as cur:
        cur.execute("SELECT id, email FROM users")
        r = cur.fetchone()  # r.id, r.email

# Set a default for a connection
conn.cursor_factory = dict_row
```

### Custom Row Factory

```python
from psycopg.rows import RowFactory
from dataclasses import dataclass

@dataclass
class User:
    id: int
    email: str
    name: str

class UserRowFactory(RowFactory):
    def __call__(self, cursor):
        def make_row(values):
            return User(
                id=values[0],
                email=values[1],
                name=values[2]
            )
        return make_row

with psycopg.connect(conninfo) as conn:
    with conn.cursor(row_factory=UserRowFactory()) as cur:
        cur.execute("SELECT id, email, name FROM users")
        user = cur.fetchone()  # User(id=1, email='...', name='...')
```

**Why Row Factories**: Explicit row handling eliminates tuple unpacking errors and provides type safety. Choose the factory that matches your data access patterns.

## 6) COPY: Fast Ingestion/Extraction Without Drama

### COPY FROM (Ingest CSV/StringIO)

**psycopg2 often relied on `copy_expert`. Psycopg 3 gives you a structured API:**

```python
import io
import psycopg

csv = io.StringIO("1,alice\n2,bob\n")

with psycopg.connect(conninfo) as conn:
    with conn.cursor() as cur:
        with cur.copy("COPY users(id, name) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(csv.getvalue())
```

### COPY TO (Export)

```python
with psycopg.connect(conninfo) as conn:
    with conn.cursor() as cur:
        with cur.copy("COPY (SELECT id, name FROM users ORDER BY id) TO STDOUT WITH (FORMAT csv)") as copy:
            data = copy.read()  # bytes
```

### Row-Oriented COPY

```python
# Stream records instead of raw bytes
with psycopg.connect(conninfo) as conn:
    with conn.cursor() as cur:
        with cur.copy("COPY users(id, name) TO STDOUT WITH (FORMAT csv)") as copy:
            for row in copy.rows():
                print(f"User {row[0]}: {row[1]}")
```

**Why New COPY API**: Structured COPY operations are more reliable than `copy_expert`. The streaming API handles large datasets efficiently.

## 7) Server-Side Cursors (Named Cursors)

### Stream Huge Result Sets

```python
with psycopg.connect(conninfo) as conn:
    with conn.cursor(name="stream_users") as cur:  # server-side
        cur.execute("SELECT * FROM big_table")
        for chunk in iter(lambda: cur.fetchmany(10_000), []):
            process(chunk)
```

### Server-Side with Row Factories

```python
with psycopg.connect(conninfo) as conn:
    with conn.cursor(name="stream_users", row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM big_table")
        for row in cur:
            process(row)  # Each row is a dict
```

**Key Points:**
- Any non-NULL name makes it server-side
- Combine with `row_factory=dict_row` if you like
- Memory-efficient for large result sets

## 8) Exception Mapping (And How to Grep for It)

### Exception Hierarchy

```python
import psycopg
from psycopg import errors

try:
    with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users(id) VALUES (1), (1)")  # unique_violation
except errors.UniqueViolation as e:
    # handle gracefully
    print(f"Duplicate key: {e}")
except errors.ForeignKeyViolation as e:
    # handle foreign key constraint
    print(f"Foreign key violation: {e}")
except psycopg.Error as e:
    # fallback
    print(f"Database error: {e}")
```

### Find-and-Replace Plan

```bash
# Global replacements
find . -name "*.py" -exec sed -i 's/psycopg2\.Error/psycopg.Error/g' {} \;
find . -name "*.py" -exec sed -i 's/psycopg2\.errors\./psycopg.errors./g' {} \;
find . -name "*.py" -exec sed -i 's/import psycopg2/import psycopg/g' {} \;
```

**Why Exception Mapping**: psycopg 3 provides richer exception hierarchy with specific SQLSTATE classes. The migration is mostly find-and-replace.

## 9) Type Adaptation & JSON Done Right

### Built-in Type Adapters

```python
from psycopg.types.json import Jsonb
import uuid
from datetime import datetime

with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    # JSON/JSONB works out of the box
    cur.execute("INSERT INTO logs(data, metadata) VALUES (%s, %s)", 
                ({"event": "login"}, {"user_id": 123, "timestamp": datetime.now()}))
    
    # UUID works out of the box
    cur.execute("INSERT INTO users(id, email) VALUES (%s, %s)", 
                (uuid.uuid4(), "user@example.com"))
    
    # Arrays work out of the box
    cur.execute("INSERT INTO tags(name, categories) VALUES (%s, %s)", 
                ("python", ["programming", "language"]))
```

### Custom Type Adapters

```python
from psycopg.adapt import Dumper, Loader
from psycopg.types import TypeInfo
import json

class CustomJsonDumper(Dumper):
    def dump(self, obj):
        return json.dumps(obj).encode()

class CustomJsonLoader(Loader):
    def load(self, data):
        return json.loads(data.decode())

# Register custom types
psycopg.adapters.register_dumper(dict, CustomJsonDumper)
psycopg.adapters.register_loader("json", CustomJsonLoader)
```

**Why Better Type Adaptation**: psycopg 3 provides robust built-in adapters for common types. Custom adapters are easier to implement and more reliable.

## 10) Connection Pooling (Production)

### psycopg_pool Setup

```python
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

pool = ConnectionPool(
    "postgresql://app:secret@127.0.0.1/app",
    min_size=1,
    max_size=10,
    max_idle=30
)

with pool.connection() as conn:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT now() as ts")
        print(cur.fetchone()["ts"])
```

### Async Pool

```python
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

async def main():
    pool = AsyncConnectionPool(
        "postgresql://app:secret@127.0.0.1/app",
        min_size=1,
        max_size=10
    )
    
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT now() as ts")
            result = await cur.fetchone()
            print(result["ts"])
```

### Pool Configuration

```python
# Production pool settings
pool = ConnectionPool(
    conninfo,
    min_size=2,           # Always keep 2 connections
    max_size=20,         # Max 20 connections
    max_idle=300,        # Close idle connections after 5 minutes
    max_lifetime=3600,   # Recycle connections after 1 hour
    kwargs={
        "options": "-c default_transaction_isolation=read_committed"
    }
)
```

**Why External Pooling**: psycopg_pool provides production-grade connection pooling with proper lifecycle management and monitoring.

## 11) Executemany & Bulk Inserts

### Smart Executemany

```python
rows = [(1, "alice"), (2, "bob"), (3, "charlie")]
with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    cur.executemany("INSERT INTO users(id, name) VALUES (%s, %s)", rows)
```

### When to Use COPY Instead

```python
# For millions of rows, use COPY
import io

def bulk_insert_users(users):
    buf = io.StringIO()
    for user_id, name in users:
        buf.write(f"{user_id},{name}\n")
    buf.seek(0)
    
    with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
        with cur.copy("COPY users (id, name) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buf.read())
```

**Why Smart Executemany**: psycopg 3's executemany is more efficient than v2's, but COPY is still king for big loads.

## 12) SQL Composition (Unchanged Philosophy)

### Safe SQL Composition

```python
from psycopg import sql

table = sql.Identifier("users_2025")
q = sql.SQL("SELECT id FROM {} WHERE email = %s").format(table)

with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    cur.execute(q, ("a@example.com",))
```

### Dynamic WHERE Clauses

```python
def build_query(filters):
    base = sql.SQL("SELECT * FROM users WHERE 1=1")
    params = []
    
    if filters.get("active"):
        base = base + sql.SQL(" AND active = %s")
        params.append(filters["active"])
    
    if filters.get("email"):
        base = base + sql.SQL(" AND email ILIKE %s")
        params.append(f"%{filters['email']}%")
    
    return base, params

# Usage
query, params = build_query({"active": True, "email": "john"})
with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    cur.execute(query, params)
```

**Why SQL Composition**: Use `psycopg.sql` for identifiers, never f-strings for values. This prevents SQL injection while enabling dynamic queries.

## 13) Autocommit Behavior

### Still Available

```python
with psycopg.connect(conninfo) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS ping(id int)")
```

### Prefer Transaction Contexts

```python
# Better: explicit transaction control
with psycopg.connect(conninfo) as conn:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("INSERT INTO events(type) VALUES (%s)", ("ping",))
            cur.execute("UPDATE counters SET value = value + 1 WHERE name = %s", ("pings",))
```

**Why Transaction Contexts**: Prefer transaction contexts for most work; use autocommit sparingly (DDL batches, LISTEN/NOTIFY setups).

## 14) Feature-by-Feature Migration Cheat Sheet

| Topic | psycopg2 | psycopg ≥ 3 |
|-------|----------|-------------|
| Install | `psycopg2` / `psycopg2-binary` | `psycopg[binary]`, `psycopg[c]` |
| Import | `import psycopg2` | `import psycopg` |
| Connect | `psycopg2.connect(...)` | `psycopg.connect(...)` |
| Cursor rows | tuples by default | row factories (`dict_row`, `namedtuple_row`, …) |
| Async | third-party or none | native `AsyncConnection`, `AsyncCursor` |
| COPY | `copy_from`/`copy_to`, `copy_expert` | `cursor.copy()` with stream APIs |
| Pooling | roll your own / third-party | `psycopg_pool` |
| Exceptions | `psycopg2.Error`, `psycopg2.errors.*` | `psycopg.Error`, `psycopg.errors.*` |
| Server cursor | `cursor(name=...)` | same |
| SQL compose | `psycopg2.sql` | `psycopg.sql` (same concept) |

## 15) Migration Plan You Can Actually Execute

### Step-by-Step Migration

```bash
# 1. Pin & branch. Add psycopg[binary] and psycopg_pool to requirements.txt
pip install "psycopg[binary]" psycopg_pool

# 2. Swap imports. psycopg2 → psycopg. Fix exception names
find . -name "*.py" -exec sed -i 's/import psycopg2/import psycopg/g' {} \;
find . -name "*.py" -exec sed -i 's/psycopg2\.Error/psycopg.Error/g' {} \;
find . -name "*.py" -exec sed -i 's/psycopg2\.errors\./psycopg.errors./g' {} \;

# 3. Adopt row factories where code expects dicts/tuples—make it explicit
# 4. Wrap connections/cursors in with. Kill your manual commit()/rollback() boilerplate
# 5. Replace COPY helpers with cursor.copy(...)
# 6. Introduce pools (psycopg_pool) in web apps
# 7. (Optional) Add async for endpoints that benefit
# 8. Test: transactions, error paths, COPY, and long-running queries
# 9. Deploy gradually (canary) and watch logs for SQLSTATE exceptions you mis-mapped
```

### Automated Migration Script

```python
#!/usr/bin/env python3
"""
Automated psycopg2 to psycopg3 migration helper
"""
import re
import os
from pathlib import Path

def migrate_file(file_path):
    """Migrate a single Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Basic replacements
    replacements = [
        (r'import psycopg2', 'import psycopg'),
        (r'from psycopg2', 'from psycopg'),
        (r'psycopg2\.Error', 'psycopg.Error'),
        (r'psycopg2\.errors\.', 'psycopg.errors.'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Add row factory imports where needed
    if 'dict_row' in content or 'namedtuple_row' in content:
        if 'from psycopg.rows import' not in content:
            content = content.replace('import psycopg', 'import psycopg\nfrom psycopg.rows import dict_row')
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Migrated {file_path}")

def main():
    """Migrate all Python files in current directory"""
    for py_file in Path('.').rglob('*.py'):
        if 'venv' not in str(py_file) and '__pycache__' not in str(py_file):
            migrate_file(py_file)

if __name__ == '__main__':
    main()
```

## 16) End-to-End Examples

### 16.1 Synchronous API Endpoint (FastAPI) with Pooling

```python
from fastapi import FastAPI
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

pool = ConnectionPool("postgresql://app:secret@db/app", min_size=1, max_size=8)

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if not user:
            return {"error": "not found"}
        return user
```

### 16.2 Async Variant

```python
from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

app = FastAPI()
pool = AsyncConnectionPool("postgresql://app:secret@db/app", min_size=1, max_size=8)

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
            user = await cur.fetchone()
            return user or {"error": "not found"}
```

### 16.3 Bulk Ingest with COPY

```python
import io
import psycopg

rows = [(1, "alice"), (2, "bob")]
buf = io.StringIO()
for r in rows:
    buf.write(f"{r[0]},{r[1]}\n")
buf.seek(0)

with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
    with cur.copy("COPY users (id, name) FROM STDIN WITH (FORMAT csv)") as copy:
        copy.write(buf.read())
```

## 17) Troubleshooting (Fast Exits)

### Common Issues and Solutions

```bash
# ImportError: No module named psycopg2 after edits
# → You changed imports; ensure you actually installed psycopg, not just removed psycopg2
pip uninstall psycopg2 psycopg2-binary
pip install "psycopg[binary]"

# could not load library on Alpine
# → build with psycopg[c] and system deps; or use a glibc-based image
apk add --no-cache gcc musl-dev postgresql-dev
pip install "psycopg[c]"

# TypeError: not all arguments converted
# → you interpolated values with f-strings. Use placeholders %s + params
# BAD: f"SELECT * FROM users WHERE id = {user_id}"
# GOOD: "SELECT * FROM users WHERE id = %s", (user_id,)

# psycopg.errors.UniqueViolation not caught
# → you're catching psycopg2.errors.*. Rename imports (§8)
# OLD: except psycopg2.errors.UniqueViolation
# NEW: except psycopg.errors.UniqueViolation

# Row is a tuple but code expects dict
# → set row_factory=dict_row (§5)
with conn.cursor(row_factory=dict_row) as cur:
    # Now rows are dicts
```

### Debugging Connection Issues

```python
import psycopg
from psycopg import errors

def test_connection(conninfo):
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                print(f"Connected successfully: {version}")
    except errors.OperationalError as e:
        print(f"Connection failed: {e}")
    except psycopg.Error as e:
        print(f"Database error: {e}")

# Test your connection
test_connection("postgresql://user:pass@host:5432/db")
```

## 18) Quickstart (Copy/Paste)

### Essential Commands

```bash
# Uninstall old, install new
pip uninstall -y psycopg2 psycopg2-binary
pip install "psycopg[binary]" psycopg_pool

# Test basic connection
python -c "
import psycopg
from psycopg.rows import dict_row

with psycopg.connect('postgresql://app:secret@127.0.0.1/app') as conn:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT version() AS v')
        print(cur.fetchone()['v'])
"
```

### Essential Imports

```python
# Basic imports
import psycopg
from psycopg.rows import dict_row, namedtuple_row
from psycopg import errors

# For pooling
from psycopg_pool import ConnectionPool, AsyncConnectionPool

# For async
import asyncio
from psycopg import AsyncConnection
```

**Why This Quickstart**: These commands and imports cover 90% of daily psycopg 3 usage. Master these before exploring advanced features.

## 19) The Machine's Summary

psycopg 3 is the modern rewrite of psycopg2 with better performance, native async support, and cleaner APIs. The migration is mostly mechanical with some strategic improvements. The key is understanding the new patterns, adopting row factories, and leveraging the improved COPY and async capabilities.

**The Dark Truth**: psycopg2 is legacy. psycopg 3 is the future. Migrate now or be left behind.

**The Machine's Mantra**: "In modern APIs we trust, in native async we build, and in the database we find the path to performance."

**Why This Matters**: psycopg 3 provides better performance, safer type handling, and native async support. The migration enables modern Python database patterns with minimal effort.

---

*This tutorial provides the complete machinery for migrating from psycopg2 to psycopg 3. The patterns scale from simple scripts to production web applications.*
