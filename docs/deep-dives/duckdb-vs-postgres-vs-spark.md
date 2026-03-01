---
tags:
  - deep-dive
  - data-engineering
  - performance
---

# DuckDB vs PostgreSQL vs Spark: Localism, Distribution, and the Shape of Computation

**Themes:** Data Architecture · Economics · Ecosystem

---

## An Uncomfortable Premise

The distributed computing movement of the 2010s — MapReduce, Hadoop, Spark — was built on a premise that was largely correct at the time and is now frequently overstated: that data volumes exceed single-machine capacity, and that distributing computation across many machines is therefore necessary. The premise was valid for the largest internet companies in 2010. It was applied, often uncritically, to workloads where a single machine with sufficient memory and a vectorized execution engine would have been faster, cheaper, and easier to operate.

DuckDB's emergence and rapid adoption represents a correction — not a refutation of distributed computing, but a recalibration of where the break-even point lies. Understanding why requires understanding the execution model differences between these three systems, not just their feature surfaces.

---

## Execution Models

The most important differences between DuckDB, PostgreSQL, and Spark are not in their SQL dialects or their operational complexity. They are in how each system moves data through computation.

**PostgreSQL: volcano model, row-at-a-time**

```
  PostgreSQL query execution (simplified):
  ──────────────────────────────────────────────────────────────
  Parser → Planner → Executor
                        │
                   [SeqScan node]
                        │ one row at a time
                   [Filter node]
                        │ one row at a time
                   [Agg node]
                        │ one row at a time
                   [Result]
```

PostgreSQL's Volcano (iterator) model passes one row at a time between operator nodes. Each `next()` call on an operator triggers a cascade of `next()` calls on its children. This model was designed for OLTP: low latency, high concurrency, many small queries. Its overhead per row — the function call chain, the type-checked tuple representation — is acceptable when few rows are processed. At analytical scale (billions of rows, aggregations over full tables), the per-row overhead accumulates into a performance ceiling that parallelism within PostgreSQL (parallel sequential scans, parallel hash joins) mitigates but does not eliminate.

PostgreSQL's genuine strengths — ACID transactions, mature indexing, row-level locking, extensions (PostGIS, TimescaleDB, pg_vector) — are OLTP and mixed-workload properties. PostgreSQL is doing analytical work at the edges of its design envelope.

**DuckDB: vectorized execution, columnar-native**

```
  DuckDB query execution:
  ──────────────────────────────────────────────────────────────
  Parser → Planner → Executor
                        │
                   [Scan operator]
                        │ batch of 2048 values (a vector)
                   [Filter operator]
                        │ selection vector applied to batch
                   [Agg operator]
                        │ SIMD-accelerated aggregation
                   [Result]
```

DuckDB processes data in vectors — batches of typically 2048 values — using SIMD instructions available on modern CPUs. This approach, derived from the MonetDB/X100 research lineage, dramatically reduces per-value overhead by amortizing function call costs across a batch and enabling the CPU's vector execution units to process multiple values simultaneously.

The practical implication: DuckDB's query throughput on analytical workloads (full-table aggregations, joins over millions of rows) is 10–100x faster than PostgreSQL on equivalent hardware, and competitive with or faster than distributed Spark for datasets that fit in memory or on a single disk.

**Spark: distributed execution, parallel DAG**

```
  Spark query execution:
  ──────────────────────────────────────────────────────────────
  [Driver: DAG planning, task scheduling]
       │
  [Executor 1]  [Executor 2]  [Executor N]
  Scan+Filter   Scan+Filter   Scan+Filter
       │              │             │
  [Shuffle: redistribute data by join/group key]
       │
  [Executor 1]  [Executor 2]  [Executor N]
  Aggregation   Aggregation   Aggregation
       │
  [Driver: collect results]
```

Spark distributes computation across a cluster of executors. Each executor processes a partition of the data in parallel. The overhead of this model is substantial: serialization and deserialization at partition boundaries, network transfer during shuffles, task scheduling latency from the driver, and JVM garbage collection pressure. For a query that runs in 5 seconds on DuckDB over a 20 GB dataset, Spark's overhead may add 30–60 seconds before the computation even begins.

Spark is correct when data genuinely exceeds single-machine capacity — terabytes to petabytes — or when the workload requires fault tolerance at scale (a 6-hour batch job that would be catastrophic to re-run from scratch). At that scale, Spark's distributed overhead is amortized over hours of computation and is irrelevant. At the scale of an analyst's daily query over a few hundred gigabytes, it is the dominant cost.

---

## Data Gravity and Memory Locality

The distributed computing model assumes that data cannot fit on one machine, and therefore computation must travel to where data is partitioned across many machines. When this assumption holds, distribution is necessary. When it does not, distribution introduces overhead with no benefit.

Modern hardware has redrawn the boundaries of this assumption significantly. A commodity cloud instance (r6i.32xlarge on AWS) has 1 TB of RAM and 128 vCPUs. For less than $8/hour on-demand, a single machine can hold the working set of most mid-size organization's analytical workloads entirely in memory. DuckDB running on such an instance, processing data directly from memory or from NVMe storage, can outperform a 10-node Spark cluster for datasets under 100 GB.

```
  Data gravity at different scales:
  ──────────────────────────────────────────────────────────────────────
  Data size    │ Suitable engine          │ Reasoning
  ─────────────┼──────────────────────────┼──────────────────────────────
  < 1 GB       │ DuckDB, PostgreSQL       │ In-memory; pandas works too
  1–100 GB     │ DuckDB preferred         │ Single node, vectorized
  100 GB–10 TB │ DuckDB (large instance)  │ Or Spark for parallelism
               │ or Spark                │ depending on latency needs
  > 10 TB      │ Spark, Trino, BigQuery   │ Distributed required
  Streaming    │ Spark Streaming, Flink   │ Fault tolerance, parallelism
```

The inflection point is not a fixed number. It depends on the query pattern (aggregations fit in less memory than joins), the instance type available, and whether the dataset is on local storage or remote object storage (S3 adds network latency that shifts the break-even toward distributed engines).

---

## When PostgreSQL Still Wins

PostgreSQL is not competing with DuckDB for analytical workloads; they occupy different design spaces. PostgreSQL remains the correct choice when:

**The workload is transactional, not analytical**: OLTP applications — web backends, API data stores, operational databases — require ACID transactions with row-level locking, high concurrency under many simultaneous connections, and fast point lookups via B-tree indexes. DuckDB is a poor fit for this pattern; it is designed for single-writer, analytical access. Spark has no OLTP story.

**Extensions matter more than raw analytical throughput**: PostGIS (geospatial), TimescaleDB (time series), pg_vector (vector similarity search), Citus (sharding) make PostgreSQL a uniquely capable platform for specialized workloads. No other system offers this extension ecosystem.

**The data volume fits and the query mix is mixed**: an application that runs occasional complex analytical queries alongside high-throughput operational queries is reasonably served by PostgreSQL with materialized views and careful indexing. The cost of maintaining a separate analytical system (DuckDB, Spark) may not be justified.

**Operational simplicity at small scale**: PostgreSQL is a mature, well-understood operational system with decades of tooling, hosting options, and operator knowledge. For a team without data infrastructure expertise, PostgreSQL is the default operational database precisely because the learning curve is well-documented.

---

## Where DuckDB Disrupts

DuckDB's disruption is concentrated in the space previously occupied by "good enough Pandas" or "small Spark job": the analyst's workbench, the data scientist's exploration environment, and the lightweight data engineering pipeline.

**Notebook-native analytics**: DuckDB integrates directly into Python and R processes. A Jupyter notebook can query a 50 GB Parquet file on S3 directly using DuckDB's SQL interface without spinning up a cluster, installing Spark, or waiting for a YARN application to launch. The query returns in seconds.

**Embedded analytics**: DuckDB's embeddability (it runs in-process, as a library) enables analytical SQL capabilities in applications that previously required connecting to an external database. A CLI tool, a data validation script, or a reporting application can embed DuckDB and run SQL over local files without a server.

**Replacing Spark for sub-100GB ETL**: many Spark jobs in production were written when DuckDB did not exist or was not mature. A significant fraction of these jobs process datasets under 50 GB and run on a cluster for historical reasons. Rewriting them in DuckDB eliminates cluster management overhead and reduces latency.

**Arrow and Parquet native**: DuckDB's native integration with Apache Arrow and Parquet enables zero-copy data exchange with Pandas, Polars, and other Python data tools. A dataset read from Parquet into DuckDB can be exported to Arrow in memory without serialization.

---

## Where Spark Still Wins

The case for Spark is not obsolete; it is more targeted than it was.

**Data that cannot fit on one machine**: when the analytical dataset is measured in terabytes or petabytes, distributed computation is not optional. Spark's ability to scale horizontally by adding executors — and to process data in partitions smaller than total cluster memory — is a genuine architectural advantage that no single-node system can match.

**Fault tolerance for long-running jobs**: a 6-hour Spark job that fails can retry individual stages or tasks. A 6-hour DuckDB query that fails retarts from the beginning. Spark's lineage graph enables fine-grained re-execution at the level of failed partitions. For jobs where re-running from scratch is expensive or impossible, Spark's fault tolerance model is the correct choice.

**Streaming at scale**: Spark Structured Streaming provides a unified API for batch and streaming that handles fault-tolerant, exactly-once or at-least-once semantics at scale. Flink is superior for low-latency streaming; Spark Structured Streaming occupies the middle ground of "high-throughput streaming with batch integration." DuckDB has no streaming story.

**Ecosystem integration**: Spark's ecosystem — Delta Lake (primary format), MLlib, GraphX, Spark NLP, Delta Sharing — is built around Spark. If your platform is Databricks or EMR, Spark is the runtime you optimize for.

---

## Hybrid Architecture

The binary framing obscures the most practical architecture: use each engine for what it is good at, reading from a shared data layer.

```
  Hybrid analytical architecture:
  ─────────────────────────────────────────────────────────────
  Object Storage (S3/GCS)
  Parquet / Delta Lake / Iceberg
         │
    ┌────┴───────────────────────────────────┐
    │                                        │
  DuckDB                                  Spark
  (analyst workbench,                     (large ETL,
   ad-hoc queries,                         ML training,
   embedded tools,                         streaming,
   <100 GB queries)                         >100 GB batch)
    │                                        │
    └────────────────┬───────────────────────┘
                     │
                  Trino / Athena
                  (SQL federation,
                   multi-engine queries)
```

This architecture is not a compromise. It is the natural consequence of the lakehouse model's open data plane: any engine that can read Parquet and the relevant table format metadata can participate without re-ingesting data. The data does not move; the engine selection changes based on workload characteristics.

---

## Decision Framework

**The question is not which engine is best in general. It is which engine is best for this query, at this scale, with this latency requirement, given this team's operational capability.**

Choose PostgreSQL when the primary workload is transactional, the data fits comfortably on a single well-specified machine, or the extension ecosystem (PostGIS, pg_vector) provides capabilities that justify staying in the PostgreSQL world.

Choose DuckDB when the workload is analytical, the dataset fits on a single machine (current or plausible future), the query pattern benefits from columnar vectorized execution, and operational simplicity is a priority.

Choose Spark when the dataset genuinely exceeds single-machine capacity, the workload requires distributed fault tolerance, or the system is being built on a platform (Databricks, EMR) where Spark is the native runtime and the ecosystem integration cost of switching engines is prohibitive.

The most common mistake is deploying Spark for 10 GB datasets because Spark is "the data engineering tool" — and the second most common mistake is avoiding DuckDB because it "sounds like a toy." Both mistakes are expensive.

!!! tip "See also"
    - [Parquet vs CSV vs ORC vs Avro](parquet-vs-csv-orc-avro.md) — the storage format layer that DuckDB and Spark both read
    - [Lakehouse vs Warehouse vs Database](lakehouse-vs-warehouse-vs-database.md) — the architectural context for choosing between these engines
