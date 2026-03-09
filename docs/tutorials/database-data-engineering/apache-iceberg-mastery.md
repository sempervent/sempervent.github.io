---
tags:
  - data-lake
  - iceberg
  - acid
  - data-engineering
---

# Apache Iceberg Mastery: Build, Break, and Bend ACID Tables in the Data Lake

This tutorial gives you hands-on Apache Iceberg: create tables, run ACID updates and deletes, time-travel, and understand how Iceberg fits into a lakehouse. By the end you can run Iceberg locally, inspect metadata, and reason about when to use it versus raw Parquet or a warehouse.

**Related**: [Lakehouse vs Warehouse vs Database](../../deep-dives/lakehouse-vs-warehouse-vs-database.md) — where Iceberg sits in the architecture. [parquet_s3_fdw](parquet-s3-fdw.md) — querying Parquet from Postgres.

## 1. Prerequisites

- Docker (for Spark/Iceberg runtime) or local Java 17+
- Spark 3.x with Iceberg runtime JARs, or PySpark/Iceberg-Python
- Object storage (local path, S3, or MinIO) for table data and metadata

## 2. What Iceberg Gives You

- **ACID**: Commits, rollback, concurrent writes without corrupting files
- **Time travel**: Query table state at a given snapshot or between snapshots
- **Schema evolution**: Add, drop, rename columns without full rewrites
- **Partition evolution**: Change partitioning without rewriting the table
- **Hidden partitioning**: Partition by derived values (e.g. day of timestamp) without storing partition columns in data

## 3. Create a Table and Insert Data

Example with Spark SQL (Iceberg catalog pointing to a warehouse path):

```sql
CREATE TABLE local_db.sensors (
  id BIGINT,
  ts TIMESTAMP,
  value DOUBLE
) USING iceberg
PARTITIONED BY (days(ts));

INSERT INTO local_db.sensors VALUES
  (1, timestamp '2024-01-15 10:00:00', 23.5),
  (2, timestamp '2024-01-15 10:01:00', 24.1);
```

Run a few more inserts and commits so you have multiple snapshots.

## 4. Time Travel and Snapshots

```sql
-- List snapshots
SELECT * FROM local_db.sensors.snapshots;

-- Query at an older snapshot (by id or time)
SELECT * FROM local_db.sensors VERSION AS OF 1;
SELECT * FROM local_db.sensors TIMESTAMP AS OF '2024-01-15 10:00:00';

-- Compare between two snapshots (incremental read)
SELECT * FROM local_db.sensors.changes BETWEEN snapshot 1 AND snapshot 2;
```

Use this to debug “what changed?” and to implement incremental pipelines.

## 5. Updates and Deletes

```sql
UPDATE local_db.sensors SET value = 0 WHERE id = 1;
DELETE FROM local_db.sensors WHERE value < 0;
```

Iceberg rewrites only affected data files (copy-on-write or merge-on-read, depending on config). Concurrency is handled via optimistic locking and metadata commits.

## 6. Schema and Partition Evolution

```sql
ALTER TABLE local_db.sensors ADD COLUMN unit STRING;
ALTER TABLE local_db.sensors DROP PARTITION FIELD days(ts);
ALTER TABLE local_db.sensors ADD PARTITION FIELD months(ts);
```

Older data stays valid; new data uses the new partition spec. No full table rewrite required.

## 7. Verification and Operations

- **Inspect metadata**: List `metadata/` and `data/` under the table path; understand manifest lists and manifests
- **Expire snapshots**: Remove old snapshots and orphan files to control retention and cost
- **Rewrite data files**: Compact small files for better read performance

## 8. When to Use Iceberg

Use Iceberg when you need ACID, time travel, or schema/partition evolution on object storage at data-lake scale. Prefer raw Parquet when you only append and never update/delete. Prefer a warehouse when you want a single SQL engine and less operational ownership.

## Next Steps

- [Building a Postgres Lakehouse Image with pg_lake and parquet_s3_fdw](postgres-lakehouse-pglake-parquet-fdw.md) — unified Postgres + Iceberg + Parquet
- [parquet_s3_fdw with Local, MinIO, Vast, and AWS](parquet-s3-fdw.md) — query Parquet from Postgres
