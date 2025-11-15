# GeoParquet Best Practices: Fast Spatial, Clean Metadata, Real Pushdown

## TL;DR

- **Geometry** in GeoParquet is **WKB (BYTE_ARRAY)**; add a `geo` metadata block with `version`, `primary_column`, and per-column metadata. If `crs` omitted ⇒ assume **OGC:CRS84** (lon/lat). CRS, when present, must be **PROJJSON**.  

- Partition by **spatially-coherent** and **selective** keys (e.g., `year/quarter`, `region`, or `h3_resX`)—avoid exploding directories.  

- Size files to **128–1024 MB** compressed; aim **row groups ~128–512 MB**.  

- **ZSTD** for analytics, **Snappy** for interactive; verify stats/page indexes are written.  

- Keep **geometry columns minimal**: one **primary** geometry; place derived geometries in separate columns and declare them in metadata.  

- On object stores: enable **byte-range**, set `Content-Type: application/x-parquet`, add cache headers; avoid millions of keys per prefix.  

- In Postgres via **`parquet_s3_fdw`**: set `endpoint`/`region`, `use_minio 'true'` for MinIO; declare **explicit columns**; filter on partition columns; **EXPLAIN** to confirm row-group filtering and pruning.

## 1) GeoParquet Essentials (Spec-Driven)

- **Encoding:** geometry columns **MUST** be WKB in Parquet `BYTE_ARRAY`.  

- **Metadata:** file-level `geo` JSON includes `version`, `primary_column`, and `columns{...}` for each geometry.  

- **CRS:** optional; if missing, treat as **OGC:CRS84**. If present, supply **PROJJSON**.  

- **Multiple geometries:** mark one as `primary_column`; others listed under `columns`.  

- **Axis order:** treat coordinates as `(x, y)`; don't rely on CRS to flip axes.

> References: GeoParquet v1.0+/v1.1 (WKB, metadata schema, CRS rules).  

## 2) Layout, Partitioning, and Files

- Use **Hive-style** partitions for common filters:

```
s3://datasets/roads/year=2025/month=11/region=ne/part-0000.parquet
```

- Keep partition depth **1–3**; total directories **<~10k** per dataset, or you'll pay the listing tax.

- Target **128–1024 MB** compressed per file; compact small files on a schedule.

- Prefer **row groups 128–512 MB**; profile scans to tune.

## 3) Writing GeoParquet (Arrow/Geo stack)

Example scaffolds (adjust to your writer):

### PyArrow (table already has WKB geometry and `geo` metadata):

```python
import pyarrow.parquet as pq
from pyarrow import Table

# table: includes geometry as WKB (binary) + attrs; schema has key_value_metadata with 'geo' JSON
pq.write_table(
  table,
  "part-0000.parquet",
  compression="zstd",
  write_statistics=True,
  data_page_size=1<<20,  # 1MB
  use_dictionary=True
)
```

### GeoPandas → GeoParquet:

```python
import geopandas as gpd

gdf = gpd.read_file("roads.gpkg").to_crs(4326)  # use lon/lat unless you must project
gdf.to_parquet("roads.parquet", index=False)    # recent GeoPandas/pyarrow writes GeoParquet
```

## 4) Serving from S3-Compatible Stores (Garage/MinIO/AWS)

- **Ensure byte-range GET works**; engines will fetch column chunks via ranges.

- **Metadata headers**:
  - `Content-Type: application/x-parquet`
  - `Cache-Control: public,max-age=86400,immutable` (tune)

- **Avoid pathological prefixes** (millions of keys); add shard prefixes when needed.

### Sanity check:

```bash
curl -I -H "Range: bytes=0-1023" https://s3.example/roads/year=2025/month=11/part-0000.parquet
```

## 5) PostgreSQL Integration via `parquet_s3_fdw`

Use the PGSpider build of `parquet_s3_fdw`. Confirm you're on a compatible Postgres (13–17 as of recent README) and that Arrow/AWS SDK deps match.

Key options you'll use: `endpoint`, `region`, `use_minio` (MinIO), plus table options like `filename`, `dirname`, `sorted`, `use_threads`, etc.

### 5.1 Server & Credentials (AWS S3)

```sql
CREATE EXTENSION IF NOT EXISTS parquet_s3_fdw;

CREATE SERVER IF NOT EXISTS parquet_s3_srv
  FOREIGN DATA WRAPPER parquet_s3_fdw;

CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
  SERVER parquet_s3_srv
  OPTIONS (user 'AWS_ACCESS_KEY_ID', password 'AWS_SECRET_ACCESS_KEY');
```

### 5.2 Server (MinIO)

```sql
CREATE SERVER IF NOT EXISTS parquet_s3_srv_minio
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (use_minio 'true', endpoint 'http://minio:9000', region 'us-east-1');

CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
  SERVER parquet_s3_srv_minio
  OPTIONS (user 'MINIO_ACCESS_KEY', password 'MINIO_SECRET_KEY');
```

### 5.3 Server (Garage)

Garage is S3-compatible; set the custom endpoint and region. (`use_minio` is for MinIO specifically—use only if targeting MinIO.)

```sql
CREATE SERVER IF NOT EXISTS parquet_s3_srv_garage
  FOREIGN DATA WRAPPER parquet_s3_fdw
  OPTIONS (endpoint 'http://garage:3900', region 'us-east-1');

CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
  SERVER parquet_s3_srv_garage
  OPTIONS (user 'S3_KEY', password 'S3_SECRET');
```

### 5.4 GeoParquet Foreign Table

Declare columns explicitly. Geometry is WKB as `bytea` (or text if hex), plus any useful partition columns for pruning.

```sql
CREATE SCHEMA IF NOT EXISTS ext;

CREATE FOREIGN TABLE IF NOT EXISTS ext.roads_geoparquet (
  geom_wkb bytea,          -- GeoParquet geometry column (WKB)
  road_id  text,
  class    text,
  region   text,
  year     int,
  month    int
)
SERVER parquet_s3_srv_minio
OPTIONS (
  -- one of:
  filename 's3://datasets/roads/year=*/month=*/*.parquet',
  -- or: dirname 's3://datasets/roads/',
  use_threads 'true',
  sorted 'region,year,month'
);

-- Gather planner stats (periodically)
ANALYZE ext.roads_geoparquet;
```

### Row-group filtering & pruning: verify with EXPLAIN.

```sql
EXPLAIN (COSTS OFF, VERBOSE)
SELECT road_id
FROM ext.roads_geoparquet
WHERE region = 'ne' AND year = 2025 AND month = 11;

-- Expect a Foreign Scan that skips row groups based on stats; 
-- if not, revisit table options/versions.
```

**Note on compression:** `parquet_s3_fdw` supports `SNAPPY`, `ZSTD`, `UNCOMPRESSED` (not `GZIP`). Write files accordingly.

## 6) Query Patterns (Engines)

### DuckDB:

```sql
SELECT count(*)
FROM read_parquet('s3://datasets/roads/year=2025/month=11/*.parquet')
WHERE region='ne';
```

### Polars (lazy):

```python
import polars as pl

pl.scan_parquet("s3://datasets/roads/year=*/month=*/*.parquet") \
  .filter((pl.col("year")==2025) & (pl.col("month")==11) & (pl.col("region")=="ne")) \
  .select(["road_id","class"]) \
  .collect()
```

### GeoPandas (client-side spatial ops):

```python
import geopandas as gpd

gdf = gpd.read_parquet("s3://datasets/roads/year=2025/month=11/")  # GeoParquet-aware
```

## 7) Metadata Hygiene (Don't Skip This)

- **Always set** `geo.version` (e.g., `"1.1.0"`), `geo.primary_column`, and a `columns` map with the geometry column name:
  - `encoding: "WKB"`
  - `geometry_type` (e.g., `"LineString"`, `"Polygon"`, etc.)
  - `crs: PROJJSON` or `null` (defaults to OGC:CRS84)

- **Document dataset README** with partition scheme, CRS, and geometry semantics.

- **Keep a minimal primary geometry**. Derived geometries (simplified, buffered) go in separate columns/files.

## 8) Maintenance

- **Schedule compaction** to your target file/row-group sizes.

- **Validate GeoParquet metadata and CRS** periodically.

- **Re-ANALYZE foreign tables** after major compactions.

---

## See Also

- **[Parquet Best Practices](parquet.md)** - Fast queries, cheap storage, clean schemas: partitioning, file sizing, predicate pushdown
- **[GeoParquet Data Warehouses](geoparquet-data-warehouses.md)** - Modern geospatial data warehouses with GeoParquet
- **[parquet_s3_fdw Tutorial](../../tutorials/database-data-engineering/parquet-s3-fdw.md)** - Step-by-step guide for using parquet_s3_fdw with multiple backends
- **[GeoParquet with Polars Tutorial](../../tutorials/database-data-engineering/geoparquet-with-polars.md)** - High-performance spatial analytics with Polars

## References

- **pgspider/parquet_s3_fdw README** — server options, table options, row group filtering, MinIO/AWS usage. [GitHub](https://github.com/pgspider/parquet_s3_fdw)

- **GeoParquet spec** — WKB requirement, `geo` metadata, CRS default and PROJJSON. [geoparquet.org](https://geoparquet.org/releases/v1.0.0-beta.1/)

> **Footnotes:** `use_minio`, `endpoint`, `region`, table options (`filename`, `dirname`, `sorted`, `use_threads`, etc.) and row-group filtering behavior come from the PGSpider FDW README; compression support note via project issues/releases. GeoParquet spec details: WKB requirement, `geo` metadata, CRS default and PROJJSON.

