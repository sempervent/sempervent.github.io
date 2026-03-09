---
tags:
  - postgresql
  - lakehouse
  - parquet
  - iceberg
  - data-engineering
---

# Building a Postgres Lakehouse Image with pg_lake and parquet_s3_fdw

This tutorial walks through a single Docker image that combines **pg_lake** (Iceberg + DuckDB in Postgres) and **parquet_s3_fdw** so you can run SQL in Postgres over both Iceberg tables and Parquet files in object storage. The result is a “Postgres lakehouse” for unified querying and analytics.

**Related**: [Lakehouse vs Warehouse vs Database](../../deep-dives/lakehouse-vs-warehouse-vs-database.md), [Apache Iceberg Mastery](apache-iceberg-mastery.md), [parquet_s3_fdw with Local, MinIO, Vast, and AWS](parquet-s3-fdw.md).

## 1. What You Get

- **PostgreSQL** as the query interface
- **pg_lake**: Iceberg and DuckDB-backed tables accessible via Postgres FDW or native integration
- **parquet_s3_fdw**: Query Parquet files in S3, MinIO, or compatible object storage from Postgres
- One image for local dev, CI, or deployment

## 2. Prerequisites

- Docker (and optionally Docker Compose)
- Object storage endpoint (S3, MinIO, or local path) and credentials if required
- Basic familiarity with Postgres and FDWs

## 3. Dockerfile Outline

Use an official Postgres image (or a derivative) and install:

- pg_lake (or the appropriate Iceberg/DuckDB integration for Postgres)
- parquet_s3_fdw and its dependencies (e.g. parquet_s3_fdw build from source or a prebuilt package)

Example structure:

```dockerfile
FROM postgres:16-alpine
RUN apk add --no-cache build-deps and runtime deps for parquet_s3_fdw
# Build and install parquet_s3_fdw, then pg_lake if needed
# Copy extension files into /usr/local/share/postgresql/extension or equivalent
```

Adjust for your base image and the actual extension install procedure (many projects provide install instructions or images).

## 4. Configure Postgres for Extensions

In `postgresql.conf` or via environment:

- Shared preload or session settings required by the FDW (e.g. `shared_preload_libraries` if needed)
- Set any required GUCs for S3/object storage (endpoint, region, credentials pattern)

Create the extensions and FDW server in SQL:

```sql
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;
CREATE SERVER s3_server FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (endpoint 'http://minio:9000', region 'us-east-1');
-- Create foreign tables pointing to Parquet locations
```

For pg_lake, follow its docs to attach Iceberg tables or DuckDB-backed objects so they appear as Postgres tables.

## 5. Create Foreign Tables for Parquet

```sql
IMPORT FOREIGN SCHEMA s3_parquet
  FROM SERVER s3_server
  INTO public
  OPTIONS (bucket 'my-bucket', prefix 'analytics/');
-- Or create foreign table manually with OPTIONS (path 's3://bucket/key.parquet')
```

You can then query Parquet data with standard SQL and join with local or pg_lake tables.

## 6. Unified Lakehouse Queries

- Join Iceberg-backed tables (via pg_lake) with Parquet-backed foreign tables (via parquet_s3_fdw) in a single query
- Use Postgres roles and permissions to control access
- Run ETL or analytics entirely in SQL from one client

## 7. Verification

- List foreign tables and servers
- Run a few SELECTs against Parquet and Iceberg-backed data
- Confirm connectivity to your object store (MinIO/S3) and that credentials and endpoints are correct

## 8. Operational Notes

- **Credentials**: Prefer IAM roles or env-based config; avoid hardcoding keys in the image
- **Performance**: Tune FDW and Postgres (work_mem, parallel workers) for large Parquet scans
- **Versioning**: Pin extension and Postgres versions in the Dockerfile for reproducibility

## Next Steps

- [Apache Iceberg Mastery](apache-iceberg-mastery.md) — ACID and time travel in the lake
- [parquet_s3_fdw with Local, MinIO, Vast, and AWS](parquet-s3-fdw.md) — Parquet-only usage and options
