# Using parquet_s3_fdw with Local Files, MinIO, Vast, and AWS

This tutorial establishes the definitive approach to reading Parquet directly from object stores into Postgres/PostGIS queries without tedious ETL. We use the parquet_s3_fdw foreign data wrapper to query Parquet data on local disk, S3-compatible systems (MinIO, Vast), and AWS S3 itself—because your data shouldn't be trapped in silos.

**Goal:** Eliminate ETL pipelines that grind your infrastructure into dust. Query Parquet files directly from object storage as if they were native database tables, with pushdown predicates and selective column projection.

## 0. Pre-Flight: Install parquet_s3_fdw

### Build from source (the only way that works)

```bash
# Debian/Ubuntu - install dependencies
apt update && apt install -y \
  postgresql-server-dev-15 \
  build-essential \
  cmake \
  libarrow-dev \
  libparquet-dev

# Clone and build the extension
git clone https://github.com/pgspider/parquet_s3_fdw.git
cd parquet_s3_fdw
make && make install

# Enable in Postgres
psql -d your_database -c "CREATE EXTENSION parquet_s3_fdw;"
```

**Why:** Package managers lie about compatibility. Building from source ensures you get the latest features and proper Arrow/Parquet library integration. Your production data deserves better than broken dependencies.

## 1. Local Files with parquet_s3_fdw

### Despite the name, parquet_s3_fdw can access local paths

```sql
-- Create server for local files
CREATE SERVER parquet_local
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (use_minio 'false');

-- Define foreign table pointing to local Parquet
CREATE FOREIGN TABLE local_data (
  id bigint,
  name text,
  geom geometry(Point, 4326),
  created_at timestamptz
)
SERVER parquet_local
OPTIONS (filename '/data/parquets/local_data.parquet');

-- Query as if it were a native table
SELECT id, name, ST_AsText(geom) as geometry
FROM local_data
WHERE created_at >= '2025-01-01'
  AND ST_Within(geom, ST_MakeEnvelope(-180, -90, 180, 90, 4326));
```

**Why:** Local files provide the fastest path for development and testing. No S3 credentials needed, no network latency, no excuses for slow queries. Perfect for staging Parquet extracts on the database host.

## 2. MinIO as S3 Backend

### MinIO speaks S3. Use it as your local S3-compatible testbed

```yaml
# docker-compose.yml - Complete MinIO + Postgres setup
version: '3.8'
services:
  minio:
    image: quay.io/minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  postgres:
    image: postgis/postgis:15-3.3
    environment:
      POSTGRES_DB: parquet_test
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    depends_on:
      minio:
        condition: service_healthy

volumes:
  minio_data:
  postgres_data:
```

### Postgres FDW Configuration

```sql
-- Create MinIO server connection
CREATE SERVER parquet_minio
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    use_minio 'true',
    aws_region 'us-east-1',
    aws_access_key_id 'minio',
    aws_secret_access_key 'minio123',
    endpoint 'http://minio:9000'
  );

-- Define foreign table for MinIO data
CREATE FOREIGN TABLE minio_data (
  id bigint,
  category text,
  value double precision,
  timestamp timestamptz
)
SERVER parquet_minio
OPTIONS (
  filename 's3://testbucket/data/minio_data.parquet'
);

-- Query with pushdown predicates
SELECT category, AVG(value) as avg_value
FROM minio_data
WHERE timestamp >= '2025-01-01'
  AND category IN ('A', 'B', 'C')
GROUP BY category;
```

**Why:** MinIO provides S3 compatibility without AWS lock-in. Same interface as AWS S3, but local and under your control. Perfect for air-gapped deployments and development workflows that need S3 semantics.

## 3. Vast (NVMe/TCP Object Store with S3)

### Vast's object interface is S3-compatible but faster than traditional object storage

```sql
-- Create Vast server connection
CREATE SERVER parquet_vast
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    use_minio 'true',
    aws_region 'us-east-1',
    aws_access_key_id 'vastuser',
    aws_secret_access_key 'vastsecret',
    endpoint 'https://vast-gateway.example.com'
  );

-- Define foreign table for Vast data
CREATE FOREIGN TABLE vast_data (
  ts timestamptz,
  metric text,
  value numeric,
  tags jsonb
)
SERVER parquet_vast
OPTIONS (
  filename 's3://vastbucket/metrics/2025/01/metrics.parquet'
);

-- Query with time-based filtering
SELECT metric, AVG(value) as avg_value
FROM vast_data
WHERE ts >= '2025-01-01 00:00:00'
  AND ts < '2025-01-02 00:00:00'
  AND metric LIKE 'cpu.%'
GROUP BY metric
ORDER BY avg_value DESC;
```

**Why:** Vast buckets map to performance tiers—place Parquet files in the NVMe tier for hot queries. The FDW pulls row groups selectively when Parquet metadata matches your filters, making it ideal for time-series analytics.

## 4. AWS S3 Proper

### Point directly at AWS with proper credentials and IAM roles

```sql
-- Create AWS S3 server connection
CREATE SERVER parquet_s3
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    aws_region 'us-east-1',
    aws_access_key_id 'AKIA...',
    aws_secret_access_key 'secret...'
  );

-- Single file access
CREATE FOREIGN TABLE aws_data (
  user_id bigint,
  event_type text,
  ts timestamptz,
  properties jsonb
)
SERVER parquet_s3
OPTIONS (
  filename 's3://mycompany-data/events/2025/10/02/events.parquet'
);

-- Directory access for partitioned data
CREATE FOREIGN TABLE aws_dir_data (
  user_id bigint,
  event_type text,
  ts timestamptz,
  properties jsonb
)
SERVER parquet_s3
OPTIONS (
  dirpath 's3://mycompany-data/events/2025/10/'
);

-- Query partitioned data efficiently
SELECT event_type, COUNT(*) as event_count
FROM aws_dir_data
WHERE ts >= '2025-10-01'
  AND ts < '2025-10-02'
  AND event_type IN ('click', 'view', 'purchase')
GROUP BY event_type;
```

**Why:** AWS S3 provides unlimited scale and durability. Use `filename` for individual objects, `dirpath` to expose entire S3 prefixes as tables. This enables querying partitioned datasets without knowing individual file names.

## 5. Performance & Query Rituals

### Pushdown predicates and selective column projection

```sql
-- Pushdown works: WHERE filters on partitioned columns push to Parquet row groups
SELECT user_id, event_type, ts
FROM aws_dir_data
WHERE ts >= '2025-10-01'  -- Pushes to Parquet metadata
  AND event_type = 'purchase'  -- Pushes to Parquet metadata
  AND user_id > 1000;  -- Pushes to Parquet metadata

-- ANALYZE foreign tables for better cost estimation
ANALYZE aws_data;
ANALYZE aws_dir_data;

-- Limit row width: define only columns you need
CREATE FOREIGN TABLE slim_data (
  id bigint,
  name text
  -- Don't include columns you don't need
)
SERVER parquet_s3
OPTIONS (filename 's3://bucket/data.parquet');

-- VACUUM foreign tables? No—stats only, as data is external
-- But ANALYZE helps the planner estimate costs
```

**Why:** Pushdown predicates eliminate unnecessary data transfer. ANALYZE foreign tables to give the planner better cost estimates. Partial column projection reduces I/O and memory usage for large datasets.

## 6. Security Notes

### Credentials and network security

```sql
-- Don't hardcode secrets in DDL
-- Use ALTER SERVER with postgresql.conf GUCs
ALTER SERVER parquet_s3 OPTIONS (
  SET aws_access_key_id '${AWS_ACCESS_KEY_ID}',
  SET aws_secret_access_key '${AWS_SECRET_ACCESS_KEY}'
);

-- Or use IAM role binding if available
-- AWS IAM roles eliminate the need for explicit credentials
```

### Network security

```bash
# Always use HTTPS for endpoints
# Ensure MinIO/Vast gateways are firewalled
# Use VPC endpoints for AWS S3 when possible
```

**Why:** Hardcoded credentials in DDL are a security nightmare. Use environment variables or IAM roles. TLS everywhere prevents credential interception. Network isolation prevents unauthorized access to your object storage.

## 7. Complete Docker Compose Setup

### Full stack with MinIO, Postgres, and parquet_s3_fdw

```yaml
# docker-compose.yml - Production-ready setup
version: '3.8'

services:
  minio:
    image: quay.io/minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minio}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minio123}
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  postgres:
    build:
      context: .
      dockerfile: Dockerfile.postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-parquet_test}
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    depends_on:
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres}"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Optional: Data loader service
  data-loader:
    image: python:3.11-slim
    volumes:
      - ./scripts:/scripts
      - ./data:/data
    command: python /scripts/load_parquet_data.py
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: ${MINIO_ROOT_USER:-minio}
      MINIO_SECRET_KEY: ${MINIO_ROOT_PASSWORD:-minio123}
      POSTGRES_URL: postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-postgres}@postgres:5432/${POSTGRES_DB:-parquet_test}

volumes:
  minio_data:
  postgres_data:
```

### Custom Postgres Dockerfile

```dockerfile
# Dockerfile.postgres
FROM postgis/postgis:15-3.3

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libarrow-dev \
    libparquet-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install parquet_s3_fdw
RUN git clone https://github.com/pgspider/parquet_s3_fdw.git /tmp/parquet_s3_fdw \
    && cd /tmp/parquet_s3_fdw \
    && make \
    && make install \
    && rm -rf /tmp/parquet_s3_fdw

# Create extension on startup
COPY init.sql /docker-entrypoint-initdb.d/01-parquet-s3-fdw.sql
```

### Initialization SQL

```sql
-- init.sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;

-- Create MinIO server
CREATE SERVER parquet_minio
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    use_minio 'true',
    aws_region 'us-east-1',
    aws_access_key_id 'minio',
    aws_secret_access_key 'minio123',
    endpoint 'http://minio:9000'
  );

-- Create sample foreign table
CREATE FOREIGN TABLE sample_data (
  id bigint,
  name text,
  value double precision,
  created_at timestamptz
)
SERVER parquet_minio
OPTIONS (
  filename 's3://testbucket/sample.parquet'
);

-- Grant permissions
GRANT USAGE ON FOREIGN SERVER parquet_minio TO postgres;
GRANT SELECT ON sample_data TO postgres;
```

**Why:** This complete setup eliminates the "works on my machine" problem. Everything is containerized, reproducible, and ready for production deployment. The data loader service demonstrates end-to-end workflows.

## 8. Advanced Patterns

### Partitioned data access

```sql
-- Access partitioned Parquet files as a single table
CREATE FOREIGN TABLE partitioned_events (
  year integer,
  month integer,
  day integer,
  user_id bigint,
  event_type text,
  timestamp timestamptz
)
SERVER parquet_s3
OPTIONS (
  dirpath 's3://events/partitioned/year=2025/month=01/day=01/'
);

-- Query with partition pruning
SELECT event_type, COUNT(*) as count
FROM partitioned_events
WHERE year = 2025
  AND month = 1
  AND day = 1
  AND event_type = 'purchase'
GROUP BY event_type;
```

### Spatial data with PostGIS

```sql
-- Spatial foreign table
CREATE FOREIGN TABLE spatial_data (
  id bigint,
  name text,
  geom geometry(Point, 4326),
  properties jsonb
)
SERVER parquet_s3
OPTIONS (
  filename 's3://spatial-data/points.parquet'
);

-- Spatial queries with pushdown
SELECT id, name, ST_AsText(geom) as geometry
FROM spatial_data
WHERE ST_Within(geom, ST_MakeEnvelope(-180, -90, 180, 90, 4326))
  AND properties->>'category' = 'restaurant';
```

**Why:** Partitioned access enables efficient querying of large datasets. Spatial integration with PostGIS provides powerful geospatial analytics on Parquet data without ETL overhead.

## 9. TL;DR Quickstart

```sql
-- Local files
CREATE SERVER parquet_local 
  FOREIGN DATA WRAPPER parquet_s3_fdw 
  OPTIONS (use_minio 'false');

-- MinIO
CREATE SERVER parquet_minio 
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    use_minio 'true', 
    aws_region 'us-east-1',
    aws_access_key_id 'minio', 
    aws_secret_access_key 'minio123',
    endpoint 'http://localhost:9000'
  );

-- Vast
CREATE SERVER parquet_vast 
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    use_minio 'true', 
    aws_region 'us-east-1',
    aws_access_key_id 'vastuser', 
    aws_secret_access_key 'vastsecret',
    endpoint 'https://vast-gateway.example.com'
  );

-- AWS S3
CREATE SERVER parquet_s3 
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    aws_region 'us-east-1',
    aws_access_key_id 'AKIA...', 
    aws_secret_access_key 'secret...'
  );

-- Then create foreign tables pointing at filename or dirpath
CREATE FOREIGN TABLE my_data (
  id bigint,
  name text,
  value double precision
)
SERVER parquet_s3
OPTIONS (filename 's3://bucket/data.parquet');
```

**Why:** This sequence establishes parquet_s3_fdw connections to all major object storage backends. Each server configuration enables efficient querying of Parquet data without ETL pipelines.

## 10. Performance Anti-Patterns

### Don't query without pushdown predicates

```sql
-- Bad: No pushdown, scans entire Parquet file
SELECT * FROM large_data;

-- Good: Pushdown predicates reduce I/O
SELECT id, name FROM large_data 
WHERE category = 'A' AND value > 100;
```

### Don't ignore column projection

```sql
-- Bad: Selects all columns
SELECT * FROM wide_data WHERE id = 1;

-- Good: Only needed columns
SELECT id, name FROM wide_data WHERE id = 1;
```

**Why:** Pushdown predicates eliminate unnecessary data transfer. Column projection reduces memory usage and I/O. These optimizations are critical for production performance.

## 11. Troubleshooting

### Common issues and solutions

```sql
-- Check foreign table statistics
SELECT * FROM pg_foreign_table WHERE ftrelid = 'my_data'::regclass;

-- Verify server configuration
SELECT * FROM pg_foreign_server WHERE srvname = 'parquet_s3';

-- Test connectivity
SELECT COUNT(*) FROM my_data LIMIT 1;
```

**Why:** These diagnostics reveal configuration issues and connectivity problems. Always test with small queries before running large analytics workloads.

The foreign data wrapper approach eliminates ETL bottlenecks and enables real-time analytics on object storage. Your data deserves better than batch processing delays.
