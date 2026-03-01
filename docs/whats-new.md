# What's New

Recently added and updated content — in reverse chronological order.

---

## February 2026

### Diagrams & Tooling

- **[SVG Workflow Generation Best Practices](best-practices/diagrams/svg-workflow-generation.md)** — Mermaid-first, artifact-driven approach to producing committed SVG diagrams: diagram-as-code principles, repo conventions, style guide, CI-ready rendering workflow, and a full data platform example.

- **[Mermaid → SVG Workflow Pipeline (Tutorial)](tutorials/diagrams/mermaid-to-svg-workflow-pipeline.md)** — Step-by-step guide to setting up `@mermaid-js/mermaid-cli`, rendering `.mmd` sources to `.svg` artifacts, and embedding them in MkDocs pages. Includes troubleshooting for font, viewBox, and Puppeteer issues.

- **[Diagram Style Guide](diagrams/style-guide.md)** — When to use Mermaid vs SVG, diagram type selection by content, formatting conventions (orientation, subgraphs, node text limits), accessibility requirements, and a reusable Mermaid snippet library (control plane/data plane, pipeline stages, monolith vs microservices, MQTT broker, ML training vs serving).

- **[ADR 0015: Standardize Diagrams on Mermaid](adr/0015-diagrams-mermaid.md)** — Mermaid becomes the site standard for architecture, flow, state, and sequence diagrams; SVG is reserved for geospatial illustrations and cases where Mermaid's layout engine is insufficient.

### Site Doctrine & Structure

- **[Start Here — Architectural Compass](start-here-architectural-paths.md)** — Decision-tree navigation organized by problem domain: embedded systems, data pipelines, microservices, ML systems, geospatial, and infrastructure. Each path sequences 4–6 essays with context.

- **[Reading Tracks](reading-tracks.md)** — Six curated reading sequences: Modern Data Architecture, Distributed Systems & Scale, Embedded Systems Doctrine, Observability & Operations, Infrastructure Economics, and Spatial Systems.

- **[Decision Frameworks](decision-frameworks.md)** — Aggregated decision frameworks from nine deep dives, each summarized in 5–10 lines: Kubernetes, Microservices, Serverless, Real-Time, Storage, Analytical Systems, Metadata Governance, ML Deployment, GPU Infrastructure, and Spatial Architecture.

- **[Anti-Patterns Index](anti-patterns.md)** — Six architectural anti-patterns with diagnostic signals and deep dive links: Premature Microservices, Overusing Kubernetes, Real-Time by Default, Data Swamp Formation, Serverless Cargo Cult, and Distributed Systems for Small Teams.

- **[Philosophy of the Site](philosophy.md)** — Five principles: Restraint Over Hype, Economics Over Fashion, Determinism Over Abstraction, Governance Over Chaos, Discipline Over Novelty.

- **[Systems Thinking Glossary](systems-glossary.md)** — Operationally-oriented definitions for: control plane, data plane, abstraction debt, blast radius, data gravity, eventual consistency, metadata debt, operational entropy, and schema contract.

- **[ADR 0014: Elevate Site to Systems Doctrine](adr/0014-elevate-site-to-systems-doctrine.md)** — Structural decision to add reading tracks, decision frameworks, anti-patterns, philosophy, and glossary pages to transform the site into a navigable intellectual framework.

### New Deep Dives

- **[Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)** — Orchestration overhead, etcd fragility, networking complexity, organizational maturity requirements, and the portability illusion. Decision matrix by team size and workload. *(Themes: Infrastructure · Architecture · Economics)*

- **[The End of the Data Warehouse?](deep-dives/the-end-of-the-data-warehouse.md)** — The warehouse is not dying — it is mutating. Lakehouse convergence, open table formats, DuckDB compute fragmentation, governance implications, and a tiered decision framework. *(Themes: Data Architecture · Economics · Ecosystem)*

- **[The Economics of GPU Infrastructure](deep-dives/the-economics-of-gpu-infrastructure.md)** — GPU scarcity, utilization patterns (~21% realistic vs 100% theoretical), CUDA/driver operational complexity, PCIe vs NVLink interconnect economics, training vs inference cost structures, and a buy-vs-rent decision matrix. *(Themes: Infrastructure · Economics · ML Systems)*

- **[The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)** — Serverless relocates complexity rather than eliminating it. Cold starts, IAM explosion, observability fragmentation, vendor lock-in, and the economics of per-invocation billing at scale. *(Themes: Infrastructure · Economics · Architecture)*

- **[Why Most ML Systems Fail in Production](deep-dives/why-ml-systems-fail-in-production.md)** — Training/serving mismatch, data drift, feature skew, silent degradation, organizational misalignment, ML observability beyond accuracy metrics, and when heuristics are the right choice. *(Themes: Data Architecture · Organizational · Economics)*

- **[The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md)** — HDD rotational latency through NVMe through cloud object storage. IO amplification, throughput vs IOPS, range requests, cold tiers, and a storage tier decision framework by workload type. *(Themes: Storage · Infrastructure · Economics)*

- **[The Operational Geometry of Spatial Systems](deep-dives/the-operational-geometry-of-spatial-systems.md)** — Quadtree vs R-tree vs H3 hex grid indexing, COG vs GeoParquet format interaction, H3 partitioning strategies, routing graph vs raster cost surface trade-offs. *(Themes: Spatial · Architecture · Data Formats)*

- **[The Hidden Cost of Metadata Debt](deep-dives/the-hidden-cost-of-metadata-debt.md)** — Catalog drift patterns, control plane collapse, governance vs bureaucracy, the economic cost of duplicate pipelines and compliance failure, and a progressive enforcement framework. *(Themes: Governance · Data Architecture · Economics)*

- **[ADR 0012–0013: Deep Dive Scale Governance and Curation Model](adr/0012-deep-dives-scale-governance.md)** — Formal curation criteria (novelty, substance, durability, and non-tutorial tests), cluster soft maxima, cross-link requirements, and obsolescence handling for the now-29-essay Deep Dives corpus.

### ASCII → Mermaid Diagram Conversions

Converted ASCII diagrams in four deep dives to rendered Mermaid blocks:

- **Prefect vs Airflow** — Airflow control plane / execution layer; Prefect API vs worker data plane separation
- **ESP32 vs Raspberry Pi** — Side-by-side execution model stacks (bare-metal vs Linux kernel layers)
- **Why Most Data Pipelines Fail** — Monolithic orchestration fan-out; DAG spaghetti coupling; shared warehouse coupling; event-driven chaos
- **Observability vs Monitoring** — Sampling trade-off (100% / tail-1% / head-0.1%); data pipeline observability dimensions (freshness, completeness, correctness, lineage)

### Recursive Cathedral Generator (Kotlin + Processing)

- **[Recursive Cathedral Generator](tutorials/just-for-fun/kotlin-recursive-cathedral.md)** — Grow Gothic cathedral silhouettes from recursive L-system rules. Deterministic seeds, bilateral symmetry, exportable PNG frames.

- **[Pi-Based Sample Library Server with Live Audition](tutorials/just-for-fun/pi-sample-server.md)** — Raspberry Pi sample server with SQLite indexing, Nginx, FastAPI, WebAudio API, and USB MIDI live audition. Full 10-section deep dive.

## Late 2025

- **[Vibe → Agentic LLMs](best-practices/ml-ai/vibe-to-agentic.md)** — Moving from prompt experimentation to production-grade agentic LLM architectures.

- **[Fractal Art Explorer (JavaScript)](tutorials/just-for-fun/fractal-art-explorer-js.md)** — Real-time Mandelbrot/Julia explorer with GPU acceleration, custom palettes, and orbit traps.

- **[OSC + MQTT + Prometheus + SuperCollider](tutorials/just-for-fun/osc-mqtt-prometheus-supercollider.md)** — Let your metrics sing. Wire Prometheus exporters into SuperCollider via OSC for live data sonification.

- **[MCP + FastAPI Full Stack](best-practices/ml-ai/mcp-fastapi-stack.md)** — Model Context Protocol integrated with FastAPI for production LLM toolchains.

- **[Cross-Domain Identity Federation](best-practices/security/identity-federation-authz-authn-architecture.md)** — Unified identity federation architecture across all system layers and environments.

- **[Environment Promotion Drift Governance](best-practices/operations-monitoring/environment-promotion-drift-governance.md)** — Controlled environment promotion with drift prevention and release channel management.

- **[Data Quality SLA Validation](best-practices/data-governance/data-quality-sla-validation-observability.md)** — Comprehensive data quality governance with SLAs and multi-layer validation for tabular, geospatial, and ML data.

## Mid 2025

- **[IAM & RBAC Governance](best-practices/security/iam-rbac-abac-governance.md)** — Identity and access management across heterogeneous stacks. One of the most-linked pages on the site.

- **[Release Management & Progressive Delivery](best-practices/operations-monitoring/release-management-and-progressive-delivery.md)** — Safe deployment strategies for coordinating changes across applications, databases, data pipelines, and ML systems.

- **[Holistic Capacity Planning](best-practices/architecture-design/capacity-planning-and-workload-modeling.md)** — Workload modeling, scaling economics, and resource prediction frameworks.

- **[ONNX Browser Inference](tutorials/ml-ai/onnx-browser-inference.md)** — Run ML models directly in the browser with ONNX Runtime Web. No server required.

- **[RKE2 on Raspberry Pi Farm](tutorials/docker-infrastructure/rke2-raspberry-pi.md)** — Full Kubernetes cluster on ARM hardware. Practical, reproducible, and genuinely fun.

- **[PostGIS Geometry Indexing](tutorials/database-data-engineering/postgis-geometry-indexing.md)** — Spatial index strategies, operator classes, and query plans for production PostGIS deployments.

---

!!! tip "Want to contribute or suggest content?"
    Open an issue or PR on [GitHub](https://github.com/sempervent/sempervent.github.io). All content requests considered.
