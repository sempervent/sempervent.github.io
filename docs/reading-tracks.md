---
tags:
  - doctrine
  - reading-tracks
  - navigation
---

# Reading Tracks

Each track is a curated reading sequence for engineers entering a specific domain or problem space. Essays within a track build on one another — earlier essays establish the conceptual foundation for later ones. Start at the top of whichever track matches your current problem.

---

## Modern Data Architecture

For engineers designing or inheriting analytical data systems: data lakes, warehouses, lakehouses, and the pipelines that feed them.

1. **[Parquet vs CSV, ORC, and Avro](deep-dives/parquet-vs-csv-orc-avro.md)** — The file format layer; why columnar storage exists and when it is appropriate
2. **[The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md)** — The physical constraints (latency, throughput, IO amplification) that all storage choices operate within
3. **[Lakehouse vs Warehouse vs Database](deep-dives/lakehouse-vs-warehouse-vs-database.md)** — The architectural models for analytical data at scale, and their trade-offs
4. **[The End of the Data Warehouse?](deep-dives/the-end-of-the-data-warehouse.md)** — Whether the warehouse abstraction is mutating or being displaced, and how open table formats reshape the landscape
5. **[Why Most Data Pipelines Fail](deep-dives/why-most-data-pipelines-fail.md)** — The organizational and technical failure modes of data pipelines, independent of tooling
6. **[Metadata as Infrastructure](deep-dives/metadata-as-infrastructure.md)** — Why metadata is a control plane, not documentation, and how enforcement differs from registration
7. **[Why Most Data Lakes Become Data Swamps](deep-dives/why-data-lakes-become-swamps.md)** — How governance failures produce data swamps, and the structural practices that prevent them
8. **[The Hidden Cost of Metadata Debt](deep-dives/the-hidden-cost-of-metadata-debt.md)** — The compounding economic cost of catalog drift, ownership vacuum, and lineage gaps

---

## Distributed Systems & Scale

For engineers evaluating whether to distribute, and how to reason about the costs of distribution.

1. **[Why Most Microservices Should Be Monoliths](deep-dives/why-most-microservices-should-be-monoliths.md)** — The organizational and technical pre-conditions that microservices require
2. **[Appropriate Use of Microservices](deep-dives/appropriate-use-of-microservices.md)** — When service decomposition is architecturally correct, and how to know the difference
3. **[Distributed Systems and the Myth of Infinite Scale](deep-dives/distributed-systems-myth-of-infinite-scale.md)** — CAP theorem, coordination overhead, data gravity, and the limits of distribution
4. **[Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)** — The orchestration overhead that most organizations adopt before they need it
5. **[The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)** — How serverless relocates rather than eliminates infrastructure complexity
6. **[The Hidden Cost of Real-Time Systems](deep-dives/the-hidden-cost-of-real-time-systems.md)** — The cost of streaming at every layer: infrastructure, observability, economics

---

## Embedded Systems Doctrine

For engineers selecting hardware and protocols for embedded, IoT, and low-power systems.

1. **[ESP32 vs Raspberry Pi](deep-dives/esp32-vs-raspberry-pi.md)** — Microcontroller determinism vs Linux convenience; when each class is appropriate
2. **[MQTT vs HTTP in IoT Systems](deep-dives/mqtt-vs-http-iot.md)** — Protocol selection for constrained environments; connection models, power, and security
3. **[LoRaWAN vs Raw LoRa](deep-dives/lorawan-vs-raw-lora.md)** — The protocol stack trade-off for long-range, low-power radio communication

---

## Observability & Operations

For engineers building or improving the operational visibility of production systems.

1. **[Observability vs Monitoring](deep-dives/observability-vs-monitoring.md)** — The conceptual distinction and why it matters for how you instrument systems
2. **[The Economics of Observability](deep-dives/the-economics-of-observability.md)** — The cost structure of observability at scale: ingestion, storage, cardinality, and human interpretation
3. **[Why Most ML Systems Fail in Production](deep-dives/why-ml-systems-fail-in-production.md)** — The observability gap in ML systems and what production monitoring requires beyond accuracy metrics
4. **[The Human Cost of Automation](deep-dives/the-human-cost-of-automation.md)** — How automation accumulates abstraction debt and reduces the operational knowledge that observability depends on

---

## Infrastructure Economics

For engineers and engineering leaders making infrastructure investment decisions.

1. **[Container Base Image Philosophy](deep-dives/container-base-image-philosophy.md)** — The security and operational cost trade-offs in container image selection
2. **[Infrastructure as Code vs GitOps](deep-dives/iac-vs-gitops.md)** — How infrastructure automation models differ and when each is appropriate
3. **[The Economics of GPU Infrastructure](deep-dives/the-economics-of-gpu-infrastructure.md)** — GPU scarcity, utilization economics, training vs inference cost structures
4. **[The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)** — Per-invocation vs reserved capacity economics, and when serverless is genuinely cheaper
5. **[Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)** — The full operational cost of cluster ownership
6. **[The Human Cost of Automation](deep-dives/the-human-cost-of-automation.md)** — Skill atrophy, complexity transfer, and the organizational cost of automation

---

## Spatial Systems

For engineers building geospatial data pipelines, spatial analytics systems, or location-aware applications.

1. **[Geospatial File Format Choices](deep-dives/geospatial-file-format-choices.md)** — COG, GeoParquet, Shapefile, and the format decisions for raster and vector spatial data
2. **[The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md)** — The latency and throughput constraints that spatial queries operate within
3. **[Polars vs Pandas for Geospatial Data](deep-dives/polars-vs-pandas-geospatial.md)** — The dataframe computation layer for spatial analytics
4. **[The Operational Geometry of Spatial Systems](deep-dives/the-operational-geometry-of-spatial-systems.md)** — Spatial indexing models (quadtree, R-tree, H3), partitioning strategies, and routing system architecture
5. **[DuckDB vs PostgreSQL vs Spark](deep-dives/duckdb-vs-postgres-vs-spark.md)** — Query engine selection for spatial analytical workloads
