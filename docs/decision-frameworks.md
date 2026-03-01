---
tags:
  - doctrine
  - decision-frameworks
  - architecture
---

# Decision Framework Index

Each deep dive on this site ends with a structured decision framework. This page aggregates those frameworks as quick-reference summaries. For the full analytical context behind each framework, follow the linked essay.

---

## When to Run Kubernetes

**From**: [Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)

Use Kubernetes when: the team has more than 20 deployed services with dynamic and independent scaling requirements; the environment is regulated and requires multi-tenant workload isolation; the deployment topology requires canary releases, blue-green deployments, or progressive delivery; or the system requires active-active geographic multi-region resilience. A dedicated platform team and mature CI/CD pipeline are prerequisites, not nice-to-haves.

Use Docker Compose, a managed PaaS, or Nomad instead when: fewer than 20 services are deployed; workload volumes are stable; no dedicated platform team exists; or the primary driver for Kubernetes is organizational convention rather than technical necessity.

---

## When to Decompose into Microservices

**From**: [Why Most Microservices Should Be Monoliths](deep-dives/why-most-microservices-should-be-monoliths.md) and [Appropriate Use of Microservices](deep-dives/appropriate-use-of-microservices.md)

Microservices are appropriate when: domain boundaries are stable and well-understood; independent teams own independent services; the deployment frequency of different components differs significantly; regulatory or security isolation requires separate deployment boundaries; or a component has scaling requirements materially different from the rest of the system.

Remain monolithic when: the team is smaller than two pizza teams per service; domain boundaries are still being discovered; the infrastructure for independent deployment (CI/CD per service, distributed tracing, service mesh) is not in place; or when the primary driver is the perception that microservices are "the modern approach."

---

## When to Use Serverless

**From**: [The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)

Serverless is appropriate when: the workload is intermittent and event-driven with significant idle time; latency requirements tolerate cold start variance (P99 > 200ms acceptable); the use case is event-driven glue (S3 triggers, queue consumers, scheduled tasks); or operational simplicity is the priority for a low-criticality, low-traffic application.

Prefer containers or VMs when: the workload has sustained high throughput; P99 latency requirements are below 50ms; the application requires persistent database connections or in-memory state; compliance requirements demand control over the execution environment; or the team must operate the system across multiple cloud providers.

---

## When to Go Real-Time

**From**: [The Hidden Cost of Real-Time Systems](deep-dives/the-hidden-cost-of-real-time-systems.md)

Real-time streaming is justified when: the use case is safety-critical (fraud detection, industrial control, medical monitoring); business decisions must be made within seconds of an event; or user-facing features require sub-minute latency and the business model depends on it.

Use micro-batching (5–30 minute windows) when: latency requirements are loose enough to tolerate batch periodicity; the team does not have the operational maturity for continuous streaming pipelines; or the cost of streaming infrastructure exceeds the business value of reduced latency.

Stay batch when: consumers query data daily or weekly; analytical workloads operate on historical windows rather than event streams; or the latency of batch processing is imperceptible to end users.

---

## Storage Tier Selection

**From**: [The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md)

Optimize for latency (local NVMe SSD, in-memory cache) when: human interaction is in the response path; P99 latency requirements are below 10ms; or the application requires point-lookup IOPS at high concurrency.

Optimize for throughput (columnar Parquet on S3, parallel range requests) when: workloads are analytical batch scans over large datasets; IO patterns are sequential rather than random; or cost per GB-scanned is more important than latency.

Accept cold storage (Glacier, archive tiers) when: data is accessed less than monthly; retrieval latency of minutes-to-hours is acceptable; regulatory retention requirements mandate preservation but not accessibility.

Introduce caching layers when: the same data is read by multiple queries or users within a session or time window; the cache hit ratio justifies the cache infrastructure cost; or the latency of the underlying storage tier is incompatible with the use case.

---

## Analytical System Architecture

**From**: [The End of the Data Warehouse?](deep-dives/the-end-of-the-data-warehouse.md) and [Lakehouse vs Warehouse vs Database](deep-dives/lakehouse-vs-warehouse-vs-database.md)

Use a managed warehouse (Snowflake, BigQuery) when: the team lacks dedicated data engineering for platform management; regulatory requirements mandate native governance (HIPAA, PCI); concurrent query load exceeds 100 analysts; or the operational simplicity of a fully managed system is more valuable than format portability.

Use a lakehouse (Iceberg, Delta Lake on S3) when: the team has existing Spark or Databricks investment; multi-engine interoperability (Trino, DuckDB, Spark) is required; CDC and incremental update patterns are common; or cost-first priorities justify self-managed open formats.

Use DuckDB + Parquet when: the data volume is below 1TB; the team is small; or analytical queries are run by individual analysts rather than concurrent shared infrastructure.

---

## Metadata Governance Entry Point

**From**: [Metadata as Infrastructure](deep-dives/metadata-as-infrastructure.md) and [The Hidden Cost of Metadata Debt](deep-dives/the-hidden-cost-of-metadata-debt.md)

Start lightweight (dbt descriptions, Git-colocated schema files) when: the data estate has fewer than 20 datasets and a single cohesive team. The goal at this stage is preventing metadata debt accumulation, not implementing a catalog.

Introduce a formal catalog (DataHub, OpenMetadata) when: the estate exceeds 20 datasets with multiple teams; discovery friction is measurable; or regulatory requirements demand demonstrable data lineage and ownership.

Enforce metadata contracts when: datasets cross team boundaries; regulatory frameworks (GDPR, HIPAA, FINRA) require lineage and access control demonstration; or duplicate pipeline proliferation indicates catalog trust collapse.

---

## ML System Deployment Readiness

**From**: [Why Most ML Systems Fail in Production](deep-dives/why-ml-systems-fail-in-production.md)

Deploy an ML model when: training and serving feature pipelines are aligned and validated; a model registry tracks version, training data, and hyperparameter provenance; drift monitoring for input distribution and prediction distribution is in place; ground truth collection is structured and automated; and organizational ownership spans training through serving.

Use heuristics or rules instead when: the task is achievable with interpretable rules that non-ML engineers can maintain; the prediction frequency is low enough that human review is cost-effective; or the regulatory environment requires explainability that black-box models cannot provide.

Keep experimentation offline when: the production feature store does not match the training feature computation; the serving infrastructure cannot accommodate the model's memory or latency requirements; or drift monitoring infrastructure is not yet operational.

---

## GPU Infrastructure

**From**: [The Economics of GPU Infrastructure](deep-dives/the-economics-of-gpu-infrastructure.md)

Buy on-premises when: GPU utilization is predictably high (above 60%) over a 3+ year horizon; data residency requirements preclude cloud; or the total cloud cost over 3 years exceeds on-premises acquisition and operational cost.

Use cloud on-demand when: GPU requirements are bursty and unpredictable; the team lacks GPU hardware expertise; or time-to-first-experiment is more important than cost per experiment.

Use reserved cloud instances when: GPU demand is predictable over 1–3 years; the team can forecast training and inference capacity with confidence.

Use CPU inference when: model size is 7B parameters or fewer; throughput requirements are modest (below 5 tokens/second per user); or eliminating GPU infrastructure complexity is worth the throughput trade-off.

---

## Spatial System Architecture

**From**: [The Operational Geometry of Spatial Systems](deep-dives/the-operational-geometry-of-spatial-systems.md)

Use PostGIS R-tree indexing when: exact topological containment is required; point-in-polygon queries on complex boundaries at transactional volume are the primary workload; or write-heavy spatial operations require transactional consistency.

Use H3 + columnar analytics when: uniform spatial aggregation independent of administrative boundaries is required; integer-key joinability with non-spatial datasets is needed; or the workload is analytical (aggregation, density analysis) rather than transactional.

Use DuckDB + GeoParquet when: the dataset exceeds 10M features and the workload is analytical; multi-engine access to the same spatial data is required; or the team is working in a Parquet-native analytical stack.

Use COG on object storage when: the data is raster (imagery, elevation, environmental grids); range request access enables spatial subsetting without full file download; or the read-to-write ratio is high and the data is relatively static.
