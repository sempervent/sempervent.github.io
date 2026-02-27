---
tags:
  - performance
  - geospatial
---

# Parquet Best Practices: Fast Queries, Cheap Storage, Clean Schemas

## 1. TL;DR Checklist

A tight bullet list users can scan in 10 seconds:

- **Partition by low-cardinality, high-selectivity columns** (e.g., `year=`, `month=`, `region=`). Avoid high-cardinality explode-a-directory keys.
- **Target file sizes 128–1024 MB**, row groups ~128–512 MB. Fewer, bigger files > a million crumbs.
- **Prefer Zstandard for cold/analytic**; Snappy for low-latency interactive reads.
- **Enable dictionary encoding for string/categorical columns**; disable where dictionaries explode.
- **Ensure min/max stats are written**; page/column indexes on; bloom filters where supported & selective.
- **Use TIMESTAMP (ms/us) with timezone discipline** (store UTC; document conventions).
- **Validate predicate pushdown & column pruning** with EXPLAIN/profile tools.
- **On S3: serve with correct Content-Type** (`application/x-parquet`), byte-range enabled, set Cache-Control.
- **In Postgres: use `parquet_s3_fdw`**; push filters down; ANALYZE foreign tables; avoid SELECT *.

## 2. Dataset Layout & Partitioning

- **Prefer Hive-style partitions**: `s3://bucket/ds=2025-11-01/region=ne/…/file-000.parquet`.
- **Partition only on columns frequently used as filters**; limit depth to 1–3 levels.
- **Keep partition cardinality sane** (<10k total dirs per dataset is a healthy smell test).

### Example layout

```
s3://my-data/telemetry/
  year=2025/month=11/day=03/
    part-0000.parquet
    part-0001.parquet
```

## 3. File Sizing & Row Groups

- **Start with `row_group_size = 128–512 MB`**. Tune by reading engines (DuckDB/Polars/Spark) scan profiles.
- **Aim for 128–1024 MB per file** after compression.
- **Avoid small-file sprays**; add compaction jobs.

### PyArrow write template

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

table = ...  # build a Table

pq.write_table(
    table,
    "part-0000.parquet",
    compression="zstd",
    write_statistics=True,
    data_page_size=1024*1024,        # 1MB pages (tune)
    use_dictionary=True,
    row_group_size=512*1024*1024//table.nbytes if table.nbytes else None  # or set explicitly
)
```

## 4. Encoding & Compression

- **Default zstd level ~3–6 for analytics**; snappy for clicky/interactive workloads.
- **Enable dictionary encoding for stringy columns**; disable if dictionary blow-up triggers.
- **Keep NA representations consistent**; avoid sentinel values.

## 5. Statistics, Page/Column Indexes & Bloom Filters

- **Ensure writers produce min/max stats and page/column indexes**. Modern Arrow/DuckDB do this; verify.
- **Bloom filters**: enable on join keys and highly selective filter columns if your writer supports it (Spark, some Arrow builds). Configure FPP (~0.01–0.05).

If support is absent, rely on stats and partitioning.

## 6. Timestamps & Timezones

- **Store timestamps in UTC**; write with `TIMESTAMP_MILLIS` or `MICROS` (avoid legacy INT96 unless you must interop).
- **Record timezone policy in dataset README and schema contract**.
- **In engines, set session TZ or cast explicitly** when presenting to humans.

## 7. Schema Evolution

- **Only make additive changes by default** (append nullable columns).
- **For renames/type changes**, perform a backfill migration + version bump.
- **Keep a schema registry** (even a JSON schema doc) with versioned contracts.

### Schema contract stub

```json
{
  "name": "telemetry.v3",
  "primary_keys": ["device_id","timestamp"],
  "partitioning": ["year","month"],
  "columns": {
    "device_id": "string",
    "timestamp": "timestamp[ms, tz=UTC]",
    "region": "string",
    "value": "float64"
  }
}
```

## 8. Predicate Pushdown & Column Pruning (Client Patterns)

### Polars (lazy)

```python
import polars as pl

df = (
  pl.scan_parquet("s3://my-data/telemetry/*.parquet")
    .filter((pl.col("year")==2025) & (pl.col("month")==11) & (pl.col("region")=="ne"))
    .select(["timestamp","device_id","value"])   # column pruning
    .collect()
)
```

### DuckDB

```sql
-- Requires s3 config; filters push into parquet scan
SELECT timestamp, device_id, value
FROM read_parquet('s3://my-data/telemetry/year=2025/month=11/*.parquet')
WHERE region = 'ne' AND timestamp >= '2025-11-01';
```

### PyArrow Dataset

```python
import pyarrow.dataset as ds

dataset = ds.dataset("s3://my-data/telemetry/", format="parquet", partitioning="hive")
scanner = dataset.scanner(
    columns=["timestamp","device_id","value"],
    filter=(ds.field("year")==2025) & (ds.field("month")==11) & (ds.field("region")=="ne")
)
table = scanner.to_table()
```

## 9. Serving via S3 (Byte-Range Slicing & Caching)

- **Ensure S3/MinIO serves byte-range GET** (default yes). Engines fetch column chunks via range.
- **Set metadata**:
  - `Content-Type: application/x-parquet`
  - `Cache-Control: public,max-age=86400,immutable` (tune for your freshness)
- **Prefer multipart upload for big files**; keep ETags stable for caching.
- **Keep partition directory listings fast** (avoid millions of keys per prefix; add shard prefixes if needed).

### Byte-range sanity check

```bash
# Fetch first kilobyte to verify range works
curl -I -H "Range: bytes=0-1023" https://s3.example.com/my-data/telemetry/year=2025/month=11/part-0000.parquet
```

## 10. Postgres: `parquet_s3_fdw` Integration

**Goal**: push filters and projections into Parquet, not Postgres.

### Install & wire

(Option names vary slightly by version; verify on your target—this is the canonical pattern.)

```sql
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;

-- Server: point to your S3/MinIO
CREATE SERVER IF NOT EXISTS parquet_s3_srv
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (
    endpoint 'http://minio:9000',   -- or leave empty for AWS
    aws_region 'us-east-1',
    use_minio 'true',               -- set 'false' for AWS S3
    use_ssl 'false'                  -- 'true' if HTTPS
  );

-- Map credentials (use IAM/instance roles in production if possible)
CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
  SERVER parquet_s3_srv
  OPTIONS (
    aws_access_key_id 'REDACTED',
    aws_secret_access_key 'REDACTED'
  );

-- Define a Parquet-backed foreign table
CREATE SCHEMA IF NOT EXISTS ext;

CREATE FOREIGN TABLE IF NOT EXISTS ext.telemetry (
  device_id text,
  timestamp timestamptz,
  region text,
  value double precision,
  year int,
  month int
)
SERVER parquet_s3_srv
OPTIONS (
  filename 's3://my-data/telemetry/year=*/month=*/*.parquet',
  -- additional options may exist: parallel 'true', list_objects 'true', etc.
  -- check installed extension's \dx+ and docs
  parquet 'true'
);

-- Gather stats for planner (not too frequent)
ANALYZE ext.telemetry;
```

### Verify pushdown

```sql
EXPLAIN (COSTS OFF)
SELECT device_id, value
FROM ext.telemetry
WHERE year=2025 AND month=11 AND region='ne' AND timestamp >= '2025-11-01';

-- Expect a Foreign Scan with remote/reader-level filters; 
-- if you see Seq Scan patterns, your pushdown failed—fix table options or upgrade FDW.
```

### Do:

- **Always select explicit columns** (projection).
- **Filter on partition columns first** (year, month, etc.), then other predicates.

### Avoid:

- **Casting partition columns in predicates** (kills pruning): `WHERE CAST(year AS text) = '2025'` ❌

## 11. Table Formats (Optional Upgrade Path)

When datasets grow feral:

- **Adopt Apache Iceberg or Delta Lake** for manifests, snapshots, evolution, and vacuum/compaction.
- **Keep Parquet files as the storage layer**; let the table format handle planning & pruning.

## 12. Maintenance: Compaction & Hygiene

- **Schedule small-file compaction** (merge to target sizes).
- **Validate stats & schema drift** with periodic scanners.
- **Keep a `DATASET.md` adjacent to each dataset** describing partitions, encoding, and SLAs.

---

## See Also

- **[GeoParquet Best Practices](geoparquet.md)** - Spatial Parquet with WKB geometry, GeoParquet metadata, and CRS rules
- **[GeoParquet Data Warehouses](geoparquet-data-warehouses.md)** - Modern geospatial data warehouses with GeoParquet
- **[parquet_s3_fdw Tutorial](../../tutorials/database-data-engineering/parquet-s3-fdw.md)** - Step-by-step guide for using parquet_s3_fdw with multiple backends
- **[GeoParquet with Polars Tutorial](../../tutorials/database-data-engineering/geoparquet-with-polars.md)** - High-performance spatial analytics with Polars

