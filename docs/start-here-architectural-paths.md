---
tags:
  - doctrine
  - navigation
  - start-here
---

# Start Here — Architectural Compass

This page is a decision-tree-style entry point for engineers who want to navigate the site by problem domain rather than by section. Pick the description that closest matches what you are currently working on, and follow the sequence of essays and references.

For curated end-to-end reading sequences, see [Reading Tracks](reading-tracks.md).
For structured decision tools extracted from the essays, see [Decision Frameworks](decision-frameworks.md).
For recurring mistakes and why they fail, see [Anti-Patterns](anti-patterns.md).

---

## If you are building or evaluating embedded systems

You are selecting hardware for an IoT or sensor system, choosing a communication protocol, or reasoning about the trade-offs between microcontroller and single-board computer platforms.

- **Start with**: [ESP32 vs Raspberry Pi](deep-dives/esp32-vs-raspberry-pi.md) — microcontroller determinism vs Linux convenience; when each class is correct
- **Then read**: [MQTT vs HTTP in IoT Systems](deep-dives/mqtt-vs-http-iot.md) — the protocol choice for constrained environments
- **Then read**: [LoRaWAN vs Raw LoRa](deep-dives/lorawan-vs-raw-lora.md) — long-range radio protocol selection
- **If your system collects data**: [Why Most Data Pipelines Fail](deep-dives/why-most-data-pipelines-fail.md) — the upstream causes of sensor data pipeline failures
- **Best practice reference**: [ESP32 Best Practices](best-practices/esp32/index.md)

---

## If you are designing data pipelines or analytical systems

You are building ETL/ELT pipelines, selecting a data warehouse or lakehouse, or evaluating orchestration tools for a data platform.

- **Start with**: [Parquet vs CSV, ORC, and Avro](deep-dives/parquet-vs-csv-orc-avro.md) — the file format foundation for analytical systems
- **Then read**: [Lakehouse vs Warehouse vs Database](deep-dives/lakehouse-vs-warehouse-vs-database.md) — the architectural model trade-offs
- **Then read**: [Why Most Data Pipelines Fail](deep-dives/why-most-data-pipelines-fail.md) — the organizational failure modes independent of tooling
- **Then read**: [Metadata as Infrastructure](deep-dives/metadata-as-infrastructure.md) — why metadata governance is a control plane, not documentation
- **If you are choosing orchestration**: [Prefect vs Airflow](deep-dives/prefect-vs-airflow.md)
- **If your lake is degrading**: [Why Most Data Lakes Become Data Swamps](deep-dives/why-data-lakes-become-swamps.md)
- **Decision tools**: [Data Architecture Frameworks](decision-frameworks.md#analytical-system-architecture)

---

## If you are evaluating microservices or service decomposition

You are deciding whether to decompose a monolith, evaluating a microservice architecture you have inherited, or designing the service topology for a new system.

- **Start with**: [Why Most Microservices Should Be Monoliths](deep-dives/why-most-microservices-should-be-monoliths.md) — the organizational pre-conditions that make decomposition worthwhile
- **Then read**: [Appropriate Use of Microservices](deep-dives/appropriate-use-of-microservices.md) — domain-driven design, bounded contexts, and the readiness checklist
- **Then read**: [Distributed Systems and the Myth of Infinite Scale](deep-dives/distributed-systems-myth-of-infinite-scale.md) — CAP theorem and the coordination costs of distribution
- **If you are deploying to Kubernetes**: [Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)
- **Anti-pattern reference**: [Premature Microservices](anti-patterns.md#premature-microservices), [Distributed Systems for Small Teams](anti-patterns.md#distributed-systems-for-small-teams)

---

## If you are scaling ML systems or ML infrastructure

You are deploying ML models to production, evaluating MLOps practices, or reasoning about GPU infrastructure investment.

- **Start with**: [Why Most ML Systems Fail in Production](deep-dives/why-ml-systems-fail-in-production.md) — system integration failures and the observability gap
- **Then read**: [The Economics of GPU Infrastructure](deep-dives/the-economics-of-gpu-infrastructure.md) — GPU scarcity, utilization economics, training vs inference cost structures
- **Then read**: [Observability vs Monitoring](deep-dives/observability-vs-monitoring.md) — the observability foundation that ML systems require
- **If you are training large models**: [Distributed Systems and the Myth of Infinite Scale](deep-dives/distributed-systems-myth-of-infinite-scale.md) — coordination and data movement costs in GPU clusters
- **Decision tools**: [ML System Deployment Readiness](decision-frameworks.md#ml-system-deployment-readiness), [GPU Infrastructure](decision-frameworks.md#gpu-infrastructure)

---

## If you are building geospatial systems

You are designing a spatial data pipeline, selecting a spatial database or file format, building a routing system, or working with H3 or COG data.

- **Start with**: [Geospatial File Format Choices](deep-dives/geospatial-file-format-choices.md) — COG, GeoParquet, and spatial format trade-offs
- **Then read**: [The Operational Geometry of Spatial Systems](deep-dives/the-operational-geometry-of-spatial-systems.md) — spatial indexing, H3, partitioning, and routing system architecture
- **Then read**: [Polars vs Pandas for Geospatial Data](deep-dives/polars-vs-pandas-geospatial.md) — the dataframe computation layer for spatial analytics
- **If your queries are slow**: [The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md) — the physical constraints governing spatial query IO
- **Best practice references**: [Geospatial Data Engineering](best-practices/database-data/geospatial-data-engineering.md), [GeoParquet](best-practices/database-data/geoparquet.md)
- **Decision tools**: [Spatial System Architecture](decision-frameworks.md#spatial-system-architecture)

---

## If you are evaluating infrastructure and automation choices

You are selecting between Kubernetes and simpler deployment models, evaluating serverless vs containerized services, or reasoning about IaC and GitOps.

- **Start with**: [Infrastructure as Code vs GitOps](deep-dives/iac-vs-gitops.md) — the infrastructure automation model trade-offs
- **Then read**: [Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md) — when orchestration overhead exceeds its benefit
- **Then read**: [The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md) — the hidden costs of cloud abstraction
- **Then read**: [The Human Cost of Automation](deep-dives/the-human-cost-of-automation.md) — skill atrophy and complexity transfer as automation matures
- **Anti-pattern reference**: [Overusing Kubernetes](anti-patterns.md#overusing-kubernetes), [Serverless Cargo Cult](anti-patterns.md#serverless-cargo-cult)

---

## Foundational vocabulary

If you encounter terms used across multiple essays without definition, the [Systems Thinking Glossary](systems-glossary.md) defines the shared vocabulary: control plane, data plane, abstraction debt, metadata debt, data gravity, blast radius, eventual consistency, schema contract, and operational entropy.

For the intellectual principles behind the analytical positions taken across this site, read the [Philosophy of the Site](philosophy.md).
