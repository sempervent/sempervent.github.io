# A Customizable find_files Function for parquet_s3_fdw

**Objective**: Build a Python helper that discovers Parquet files across local storage, MinIO, Vast, and AWS S3, then generates clean SQL CREATE FOREIGN TABLE statements for parquet_s3_fdw.

The pain of manually writing FDW table DDLs is real. This tutorial provides a machine that does the heavy lifting—discovering Parquet files and generating the SQL to expose them as foreign tables in Postgres.

## 1) The Problem: Manual Table Creation Hell

### The Manual Way

```sql
-- The repetitive nightmare
CREATE FOREIGN TABLE my_table (
    id bigint,
    name text,
    ts timestamptz
)
SERVER parquet_s3
OPTIONS (filename 's3://mybucket/data/my_table.parquet');

CREATE FOREIGN TABLE another_table (
    id bigint,
    name text,
    ts timestamptz
)
SERVER parquet_s3
OPTIONS (filename 's3://mybucket/data/another_table.parquet');

-- And so on for hundreds of files...
```

**Why This Sucks**: Manual table creation doesn't scale. You're writing the same DDL over and over, making typos, and losing track of what tables exist.

### The Automated Solution

```python
# The machine's approach
files = find_files("s3://mybucket/data/", backend="s3")
sql = generate_fdw_sql(files, schema="analytics", server="parquet_s3")
print(sql)
```

**Why This Works**: One command discovers all Parquet files and generates the SQL. The machine handles the complexity, you get clean DDL.

## 2) Building the find_files Function

### Core Implementation

```python
import os
import pathlib
import boto3
from urllib.parse import urlparse
from typing import List, Dict, Optional
import logging

def find_files(
    uri: str, 
    suffix: str = ".parquet", 
    recursive: bool = True,
    backend: str = "local",
    endpoint_url: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Discover Parquet files for FDW table creation.
    
    Args:
        uri: Path or S3 URI (e.g., "s3://bucket/path" or "/local/path")
        suffix: File extension to match (default: ".parquet")
        recursive: Search subdirectories (default: True)
        backend: Storage backend ("local", "s3", "minio", "vast")
        endpoint_url: Custom S3 endpoint for MinIO/Vast
    
    Returns:
        List of dicts with 'table' and 'path' keys
    """
    results = []
    
    if backend == "local":
        results = _find_local_files(uri, suffix, recursive)
    elif backend in ("s3", "minio", "vast"):
        results = _find_s3_files(uri, suffix, recursive, endpoint_url)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return results

def _find_local_files(uri: str, suffix: str, recursive: bool) -> List[Dict[str, str]]:
    """Find files in local filesystem."""
    results = []
    path = pathlib.Path(uri)
    
    if recursive:
        pattern = f"**/*{suffix}"
    else:
        pattern = f"*{suffix}"
    
    for file_path in path.glob(pattern):
        if file_path.is_file():
            table_name = _sanitize_table_name(file_path.stem)
            results.append({
                "table": table_name,
                "path": str(file_path.absolute())
            })
    
    return results

def _find_s3_files(
    uri: str, 
    suffix: str, 
    recursive: bool, 
    endpoint_url: Optional[str]
) -> List[Dict[str, str]]:
    """Find files in S3-compatible storage."""
    results = []
    parsed = urlparse(uri, allow_fragments=False)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    
    # Configure S3 client
    s3_config = {}
    if endpoint_url:
        s3_config["endpoint_url"] = endpoint_url
    
    s3 = boto3.client("s3", **s3_config)
    
    # List objects
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            
            # Filter by suffix
            if not key.endswith(suffix):
                continue
            
            # Handle recursive vs non-recursive
            if not recursive and "/" in key[len(prefix):]:
                continue
            
            table_name = _sanitize_table_name(pathlib.Path(key).stem)
            s3_uri = f"s3://{bucket}/{key}"
            
            results.append({
                "table": table_name,
                "path": s3_uri
            })
    
    return results

def _sanitize_table_name(name: str) -> str:
    """Convert filename to valid SQL table name."""
    # Replace invalid characters with underscores
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Ensure it starts with letter or underscore
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"table_{sanitized}"
    return sanitized or "unnamed_table"
```

**Why This Works**: The function handles multiple backends through a unified interface, sanitizes table names, and provides flexible search options.

## 3) Generating FDW Table DDLs

### SQL Generation Function

```python
def generate_fdw_sql(
    files: List[Dict[str, str]], 
    schema: str = "public",
    server: str = "parquet_s3",
    table_prefix: str = "",
    table_suffix: str = "",
    use_dirpath: bool = False
) -> str:
    """
    Generate CREATE FOREIGN TABLE statements from discovered files.
    
    Args:
        files: Output from find_files()
        schema: Target schema name
        server: FDW server name
        table_prefix: Prefix for table names
        table_suffix: Suffix for table names
        use_dirpath: Use dirpath instead of filename for partitioned data
    
    Returns:
        SQL statements as string
    """
    statements = []
    
    for file_info in files:
        table_name = f"{table_prefix}{file_info['table']}{table_suffix}"
        full_table_name = f"{schema}.{table_name}"
        
        # Determine options based on use_dirpath
        if use_dirpath:
            # Extract directory path for partitioned data
            dir_path = "/".join(file_info['path'].split("/")[:-1])
            options = f"dirpath '{dir_path}'"
        else:
            options = f"filename '{file_info['path']}'"
        
        # Generate SQL
        sql = f"""CREATE FOREIGN TABLE IF NOT EXISTS {full_table_name} (
    -- Columns will be inferred by parquet_s3_fdw
    -- or define explicitly based on your schema
)
SERVER {server}
OPTIONS ({options});"""
        
        statements.append(sql)
    
    return "\n\n".join(statements)

def generate_fdw_sql_with_schema(
    files: List[Dict[str, str]], 
    schema: str = "public",
    server: str = "parquet_s3",
    column_definitions: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate FDW SQL with explicit column definitions.
    
    Args:
        files: Output from find_files()
        schema: Target schema name
        server: FDW server name
        column_definitions: Dict mapping column names to SQL types
    
    Returns:
        SQL statements with explicit column definitions
    """
    statements = []
    
    for file_info in files:
        table_name = file_info['table']
        full_table_name = f"{schema}.{table_name}"
        
        # Build column definitions
        if column_definitions:
            columns = ",\n    ".join([f"{col} {type_}" for col, type_ in column_definitions.items()])
        else:
            columns = "-- Columns will be inferred by parquet_s3_fdw"
        
        sql = f"""CREATE FOREIGN TABLE IF NOT EXISTS {full_table_name} (
    {columns}
)
SERVER {server}
OPTIONS (filename '{file_info['path']}');"""
        
        statements.append(sql)
    
    return "\n\n".join(statements)
```

**Why This Works**: The generator creates clean, valid SQL that can be executed directly in Postgres. It handles both single files and partitioned directories.

## 4) One-Liner Workflow

### Simple Usage

```python
# Discover files and generate SQL
files = find_files("s3://mybucket/data/2025/", backend="s3")
sql = generate_fdw_sql(files, schema="raw", server="parquet_s3")
print(sql)
```

**Output**:
```sql
CREATE FOREIGN TABLE IF NOT EXISTS raw.data_2025_01 (
    -- Columns will be inferred by parquet_s3_fdw
)
SERVER parquet_s3
OPTIONS (filename 's3://mybucket/data/2025/data_2025_01.parquet');

CREATE FOREIGN TABLE IF NOT EXISTS raw.data_2025_02 (
    -- Columns will be inferred by parquet_s3_fdw
)
SERVER parquet_s3
OPTIONS (filename 's3://mybucket/data/2025/data_2025_02.parquet');
```

### Advanced Usage

```python
# Customized table naming and column definitions
files = find_files("s3://analytics-bucket/events/", backend="s3")
sql = generate_fdw_sql_with_schema(
    files, 
    schema="analytics", 
    server="parquet_s3",
    column_definitions={
        "event_id": "bigint",
        "user_id": "text",
        "timestamp": "timestamptz",
        "event_type": "text",
        "properties": "jsonb"
    }
)
print(sql)
```

**Why This Scales**: One command handles discovery, naming, and SQL generation. The machine does the repetitive work.

## 5) Customization & Extensions

### Table Naming Conventions

```python
def find_files_with_naming(
    uri: str,
    backend: str = "local",
    name_template: str = "{prefix}_{table}_{suffix}",
    prefix: str = "",
    suffix: str = "",
    strip_prefix: str = "",
    strip_suffix: str = ""
) -> List[Dict[str, str]]:
    """Find files with custom naming conventions."""
    files = find_files(uri, backend=backend)
    
    for file_info in files:
        table_name = file_info['table']
        
        # Strip prefixes/suffixes
        if strip_prefix and table_name.startswith(strip_prefix):
            table_name = table_name[len(strip_prefix):]
        if strip_suffix and table_name.endswith(strip_suffix):
            table_name = table_name[:-len(strip_suffix)]
        
        # Apply template
        formatted_name = name_template.format(
            prefix=prefix,
            table=table_name,
            suffix=suffix
        )
        
        file_info['table'] = formatted_name
    
    return files

# Usage examples
files = find_files_with_naming(
    "s3://bucket/data/",
    backend="s3",
    name_template="raw_{table}",
    strip_prefix="data_",
    strip_suffix="_processed"
)
```

### Partitioned Data Handling

```python
def find_partitioned_files(
    uri: str,
    backend: str = "s3",
    partition_columns: List[str] = None
) -> List[Dict[str, str]]:
    """Find files in partitioned directories."""
    files = find_files(uri, backend=backend)
    
    # Group files by directory for dirpath usage
    dirs = {}
    for file_info in files:
        dir_path = "/".join(file_info['path'].split("/")[:-1])
        if dir_path not in dirs:
            dirs[dir_path] = []
        dirs[dir_path].append(file_info)
    
    # Generate SQL for each partition
    statements = []
    for dir_path, dir_files in dirs.items():
        table_name = _sanitize_table_name(pathlib.Path(dir_path).name)
        sql = f"""CREATE FOREIGN TABLE IF NOT EXISTS public.{table_name} (
    -- Columns will be inferred
)
SERVER parquet_s3
OPTIONS (dirpath '{dir_path}');"""
        statements.append(sql)
    
    return statements
```

**Why This Matters**: Customization handles real-world naming conventions and partitioned data structures.

## 6) Best Practices

### Schema Consistency

```python
def validate_schema_consistency(files: List[Dict[str, str]], backend: str = "s3") -> Dict[str, List[str]]:
    """Validate that all files have consistent schemas."""
    import pyarrow.parquet as pq
    
    schemas = {}
    for file_info in files:
        try:
            if backend == "local":
                schema = pq.read_schema(file_info['path'])
            else:
                # For S3, you'd need to download or use boto3
                # This is a simplified example
                schema = pq.read_schema(file_info['path'])
            
            schema_str = str(schema)
            if schema_str not in schemas:
                schemas[schema_str] = []
            schemas[schema_str].append(file_info['table'])
        except Exception as e:
            print(f"Warning: Could not read schema for {file_info['path']}: {e}")
    
    return schemas

# Usage
files = find_files("s3://bucket/data/", backend="s3")
schemas = validate_schema_consistency(files)
if len(schemas) > 1:
    print("Warning: Multiple schemas detected")
    for schema, tables in schemas.items():
        print(f"Schema: {schema[:100]}...")
        print(f"Tables: {tables}")
```

### Caching Results

```python
import json
import hashlib
from pathlib import Path

def find_files_with_cache(
    uri: str,
    backend: str = "local",
    cache_file: str = ".parquet_cache.json",
    cache_ttl: int = 3600  # 1 hour
) -> List[Dict[str, str]]:
    """Find files with caching to avoid repeated S3 calls."""
    cache_path = Path(cache_file)
    
    # Generate cache key
    cache_key = hashlib.md5(f"{uri}_{backend}".encode()).hexdigest()
    
    # Check cache
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        
        if cache_key in cache:
            cached_data = cache[cache_key]
            if time.time() - cached_data['timestamp'] < cache_ttl:
                return cached_data['files']
    
    # Discover files
    files = find_files(uri, backend=backend)
    
    # Update cache
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    
    cache[cache_key] = {
        'files': files,
        'timestamp': time.time()
    }
    
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    
    return files
```

**Why This Matters**: Schema consistency prevents query errors, and caching improves performance for large buckets.

## 7) Full Example: MinIO with parquet_s3_fdw

### Complete Workflow

```python
# 1. Discover files in MinIO
files = find_files(
    "s3://minio-bucket/analytics/",
    backend="minio",
    endpoint_url="http://localhost:9000"
)

# 2. Generate SQL with custom schema
sql = generate_fdw_sql_with_schema(
    files,
    schema="analytics",
    server="parquet_minio",
    column_definitions={
        "id": "bigint",
        "event_type": "text",
        "timestamp": "timestamptz",
        "user_id": "text",
        "properties": "jsonb"
    }
)

# 3. Save to file
with open("create_foreign_tables.sql", "w") as f:
    f.write(sql)

print(f"Generated SQL for {len(files)} tables")
```

### MinIO Configuration

```python
# MinIO-specific configuration
def find_files_minio(
    bucket: str,
    prefix: str = "",
    endpoint_url: str = "http://localhost:9000",
    access_key: str = None,
    secret_key: str = None
) -> List[Dict[str, str]]:
    """Find files in MinIO with proper authentication."""
    import boto3
    
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key or os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=secret_key or os.getenv("MINIO_SECRET_KEY")
    )
    
    uri = f"s3://{bucket}/{prefix}"
    return find_files(uri, backend="s3", endpoint_url=endpoint_url)
```

### Postgres Setup

```sql
-- 1. Create FDW server for MinIO
CREATE SERVER parquet_minio
    FOREIGN DATA WRAPPER parquet_s3_fdw
    OPTIONS (
        use_minio 'true',
        aws_region 'us-east-1',
        aws_access_key_id 'minio',
        aws_secret_access_key 'minio123',
        endpoint 'http://localhost:9000'
    );

-- 2. Create schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- 3. Execute generated SQL
\i create_foreign_tables.sql

-- 4. Verify tables
\dt analytics.*
```

**Why This Works**: The complete workflow handles discovery, SQL generation, and Postgres setup for MinIO.

## 8) Advanced Features

### Multi-Backend Support

```python
def find_files_multi_backend(
    backends: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Find files across multiple backends."""
    all_files = []
    
    for backend_config in backends:
        files = find_files(
            backend_config['uri'],
            backend=backend_config['backend'],
            endpoint_url=backend_config.get('endpoint_url')
        )
        
        # Add backend prefix to table names
        for file_info in files:
            file_info['table'] = f"{backend_config['prefix']}_{file_info['table']}"
        
        all_files.extend(files)
    
    return all_files

# Usage
backends = [
    {
        'uri': 's3://aws-bucket/data/',
        'backend': 's3',
        'prefix': 'aws'
    },
    {
        'uri': 's3://minio-bucket/data/',
        'backend': 'minio',
        'endpoint_url': 'http://localhost:9000',
        'prefix': 'minio'
    }
]

files = find_files_multi_backend(backends)
```

### Schema Inference

```python
def infer_schema_from_files(files: List[Dict[str, str]], backend: str = "s3") -> Dict[str, str]:
    """Infer schema from Parquet files."""
    import pyarrow.parquet as pq
    
    schemas = {}
    for file_info in files:
        try:
            if backend == "local":
                schema = pq.read_schema(file_info['path'])
            else:
                # For S3, you'd need to download or use boto3
                schema = pq.read_schema(file_info['path'])
            
            # Convert Arrow schema to SQL types
            sql_types = {}
            for field in schema:
                arrow_type = str(field.type)
                if 'int' in arrow_type:
                    sql_types[field.name] = 'bigint'
                elif 'string' in arrow_type:
                    sql_types[field.name] = 'text'
                elif 'timestamp' in arrow_type:
                    sql_types[field.name] = 'timestamptz'
                else:
                    sql_types[field.name] = 'text'
            
            schemas[file_info['table']] = sql_types
            
        except Exception as e:
            print(f"Warning: Could not infer schema for {file_info['path']}: {e}")
    
    return schemas
```

## 9) TL;DR Quickstart

```python
# 1. Install dependencies
# pip install boto3 pyarrow

# 2. Discover files
files = find_files("s3://bucket/path", backend="s3")

# 3. Generate SQL
sql = generate_fdw_sql(files, schema="analytics", server="parquet_s3")

# 4. Save and execute
with open("tables.sql", "w") as f:
    f.write(sql)

# 5. In Postgres
# \i tables.sql
```

## 10) The Machine's Summary

This `find_files` function eliminates the manual pain of creating foreign tables. It discovers Parquet files across multiple backends, generates clean SQL, and handles the complexity of naming and schema management.

**The Dark Truth**: Manual table creation doesn't scale. The machine handles discovery, naming, and SQL generation. You get clean DDL and reproducible table creation.

**The Machine's Mantra**: "In discovery we trust, in automation we serve, and in the machine we find the path to scalable data access."

---

*This tutorial provides the complete machinery for automating parquet_s3_fdw table creation. The machine handles the complexity, you get clean SQL.*
