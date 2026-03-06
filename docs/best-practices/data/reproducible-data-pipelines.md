# Reproducible Data Pipelines

**Objective**: Define practices for building deterministic, auditable data pipelines that produce the same outputs from the same inputs and support governance and debugging.

## Why reproducibility matters

### Scientific reproducibility

Reproducibility is the ability to re-run an analysis or pipeline and obtain the same results. In research and regulated domains, this is non-negotiable: conclusions must be verifiable. Pipelines that depend on implicit state, non-versioned code, or unreproducible randomness break the chain of evidence and undermine trust in outcomes.

### Operational debugging

When a pipeline produces wrong or unexpected results, the first question is: *what changed?* Reproducible pipelines answer that by tying outputs to exact input versions and transformation code. Re-running the same pipeline on the same inputs should yield identical outputs; if not, the failure is isolated to environment or non-determinism, which can be systematically fixed.

### Governance and auditing

Auditors and compliance frameworks require evidence of how data was produced: which sources, which transformations, and when. Reproducible pipelines support audit trails through versioned code, hashed inputs, and provenance metadata. Immutable, versioned datasets and deterministic transformations make it possible to demonstrate lineage and re-create historical outputs when required.

## Pipeline determinism

### Idempotent processing

Idempotency means running the same operation multiple times has the same effect as running it once. In pipelines:

- **Writes**: Overwrite or upsert with deterministic keys so re-runs do not duplicate or corrupt data.
- **Deletes**: Define clear boundaries (e.g. partition or time range) so re-runs do not remove more than intended.
- **Side effects**: Avoid non-idempotent side effects (e.g. sending duplicate notifications, appending to logs without a run identifier).

Design tasks so that "run again" is safe and produces the same logical state.

### Immutable datasets

Treat pipeline stages as immutable once written: do not update or delete existing partitions in place for a given run. New runs write to new partitions or new versioned paths. This preserves history, enables time-travel and rollback, and avoids hidden coupling where a later run changes data that an earlier run already consumed.

### Versioned transformations

Transformation logic must be under version control and referenced by commit or tag when running. Pipeline runs should record the exact code version (and, where applicable, container image digest) so that any run can be reproduced from the same repo and ref.

## Data lineage architecture

### Input hashing

Record content hashes (e.g. SHA-256) or version identifiers for all inputs consumed by a run. When inputs are files, hash the files or their metadata (path + size + mtime, or object version in object storage). When inputs are query results or API responses, record query parameters and a stable identifier for the snapshot. Storing input hashes with each run makes it possible to detect when a "re-run" would actually see different data.

### Dataset versioning

Version datasets explicitly: by partition (e.g. `dt=2025-03-05`), by run ID, or by semantic version when publishing curated layers. Downstream consumers should refer to a specific version or partition so that re-execution and debugging are well-defined.

### Provenance tracking

Persist provenance metadata for each output: input identifiers, transformation version, run ID, timestamps, and (where applicable) operator or system that executed the run. This metadata is the basis for lineage graphs, impact analysis, and audit trails. Integrate with existing lineage and catalog systems where they exist.

## Pipeline artifact strategy

### Raw → normalized → curated → analytical

Structure pipelines in clear stages:

| Stage        | Purpose                                      | Typical characteristics                    |
|-------------|----------------------------------------------|--------------------------------------------|
| **Raw**     | Preserve source data unchanged               | Immutable, append-only, checksummed        |
| **Normalized** | Unify schema, encoding, and basic validation | One schema per source type, partitioned    |
| **Curated** | Business-level cleaning and modeling         | Domain entities, conformed types, quality  |
| **Analytical** | Aggregations, features, API-ready views   | Optimized for query and serving            |

Each stage is consumed by the next; avoid back-editing earlier stages. This keeps responsibilities clear and makes it obvious where to fix issues.

### Parquet and columnar formats

Use columnar formats (e.g. Parquet) for analytical and curated layers: better compression, predicate pushdown, and column pruning. Follow [Parquet best practices](../database-data/parquet.md) for partitioning, file sizing, and encoding. For geospatial data, use [GeoParquet](../database-data/geoparquet.md) with correct metadata and CRS.

### Dataset partitioning

Partition by columns that are frequently used in filters (e.g. date, region, tenant). Keep partition cardinality manageable. Prefer Hive-style layout for compatibility with engines and [FDW-based access](../../tutorials/database-data-engineering/parquet-s3-fdw.md). Partitioning enables incremental processing and limits the scope of re-runs.

## Example pipeline flow

```mermaid
flowchart LR
    A[Raw Sources]
    B[Ingestion]
    C[Normalized Data]
    D[Transform]
    E[Curated Dataset]
    F[Analytics / APIs]

    A --> B --> C --> D --> E --> F
```

Raw sources are ingested into normalized storage; transformations produce curated datasets; analytics and APIs consume curated (and sometimes normalized) data. Each step should be deterministic and versioned.

## Orchestration considerations

### Prefect / workflow tools

Use an orchestrator (e.g. [Prefect](https://www.prefect.io/) or Airflow) to schedule runs, manage dependencies, and handle retries. Compare tradeoffs in the [Prefect vs Airflow](../../deep-dives/prefect-vs-airflow.md) deep dive. Ensure the orchestrator records run metadata (run ID, code version, parameters) and that tasks are implemented to be idempotent and restartable.

### Failure recovery

Design for partial failure: failed tasks should be retryable without re-running entire pipelines. Use checkpointing or partition-level granularity so that only failed work is redone. Avoid global state that makes "resume" impossible.

### Restartability

Pipelines should support restart from a defined point (e.g. from a specific task or partition) without manual intervention. This requires deterministic task boundaries and stable identifiers for work units.

## Common anti-patterns

| Anti-pattern | Problem | Prefer |
|--------------|---------|--------|
| **Mutable pipelines** | In-place updates make history and re-runs undefined. | Immutable stages; new partitions/versions per run. |
| **Hidden side effects** | Non-determinism or external state (e.g. "now", random seeds not fixed). | Explicit inputs, fixed seeds, no implicit time in logic. |
| **Implicit schema evolution** | Schema changes without versioning break reproducibility. | Versioned schemas, compatibility checks, contract tests. |

Avoid pipelines that depend on undocumented inputs, unversioned code, or overwriting data that other runs or consumers rely on.

## See also

- [Metadata as Control Plane](metadata-control-plane.md) — how metadata governs pipeline behavior and lineage
- [ETL Pipeline Design](../database-data/etl-pipeline-design.md) — production ETL patterns and orchestration
- [Parquet](../database-data/parquet.md) and [GeoParquet](../database-data/geoparquet.md) — format and layout best practices
- [Data Lineage, Contracts & Provenance](../database-data/data-lineage-contracts.md) — lineage and provenance enforcement
- [Data Pipeline Quality, Governance & Observability](../../tutorials/best-practices-integration/data-pipeline-quality-governance-observability.md) — quality and observability in pipelines
- [Why Most Data Pipelines Fail](../../deep-dives/why-most-data-pipelines-fail.md) — failure modes and structural fixes
- [Lakehouse vs Warehouse vs Database](../../deep-dives/lakehouse-vs-warehouse-vs-database.md) — where pipelines land data
