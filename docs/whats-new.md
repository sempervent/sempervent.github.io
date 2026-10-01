# Recent additions

Edited when I add something worth mentioning — not a git log.

---

## September 2026

- Reworked the [home](index.md) and [projects](projects/index.md) pages.
- [Glitch Observatory](tutorials/just-for-fun/js-glitch-observatory.md) is under Just for Fun (old URL redirects).

---

## February 2026

### Diagrams & tooling

- **[SVG Workflow Generation Best Practices](best-practices/diagrams/svg-workflow-generation.md)** — Mermaid-first, artifact-driven approach to committed SVG diagrams: diagram-as-code principles, rendering workflow, CI checks, and a data platform example.

- **[Mermaid → SVG Workflow Pipeline (Tutorial)](tutorials/diagrams/mermaid-to-svg-workflow-pipeline.md)** — Set up `@mermaid-js/mermaid-cli`, render `.mmd` sources to `.svg`, and embed figures in static documentation. Includes troubleshooting for font, viewBox, and Puppeteer issues.

- **[Layered Systems Diagrams: Mermaid → SVG](tutorials/diagrams/layered-systems-diagrams-mermaid-to-svg.md)** — Context and workflow diagrams for an IoT → MQTT → lakehouse example, from source through rendered SVG.

### Reader guides

- **[Architectural Compass](start-here-architectural-paths.md)** — Problem-oriented paths for embedded systems, data pipelines, microservices, ML, geospatial, and infrastructure.

- **[Reading Tracks](reading-tracks.md)** — Curated sequences: Modern Data Architecture, Distributed Systems & Scale, Embedded Systems, Observability & Operations, Infrastructure Economics, and Spatial Systems.

- **[Decision Frameworks](decision-frameworks.md)** — Summaries from nine deep dives: Kubernetes, Microservices, Serverless, Real-Time, Storage, Analytical Systems, Metadata Governance, ML Deployment, GPU Infrastructure, and Spatial Architecture.

- **[Anti-Patterns Index](anti-patterns.md)** — Premature Microservices, Overusing Kubernetes, Real-Time by Default, Data Swamp Formation, Serverless Cargo Cult, and Distributed Systems for Small Teams.

- **[Systems engineering principles](philosophy.md)** — Restraint, economics, determinism, governance, and discipline as design lenses.

- **[Systems Thinking Glossary](systems-glossary.md)** — Control plane, data plane, abstraction debt, blast radius, data gravity, and related terms.

### New deep dives

- **[Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)** — Orchestration overhead, etcd fragility, networking complexity, organizational maturity requirements, and the portability illusion. *(Themes: Infrastructure · Architecture · Economics)*

- **[The End of the Data Warehouse?](deep-dives/the-end-of-the-data-warehouse.md)** — Lakehouse convergence, open table formats, DuckDB compute fragmentation, governance implications, and a tiered decision framework. *(Themes: Data Architecture · Economics · Ecosystem)*

- **[The Economics of GPU Infrastructure](deep-dives/the-economics-of-gpu-infrastructure.md)** — Utilization patterns, interconnect economics, training vs inference cost structures, and a buy-vs-rent decision matrix. *(Themes: Infrastructure · Economics · ML Systems)*

- **[The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)** — Cold starts, IAM explosion, observability fragmentation, vendor lock-in, and per-invocation economics at scale. *(Themes: Infrastructure · Economics · Architecture)*

- **[Why Most ML Systems Fail in Production](deep-dives/why-ml-systems-fail-in-production.md)** — Training/serving mismatch, data drift, feature skew, silent degradation, and when heuristics are the right choice. *(Themes: Data Architecture · Organizational · Economics)*

- **[The Physics of Storage Systems](deep-dives/the-physics-of-storage-systems.md)** — HDD through NVMe through object storage, IO amplification, throughput vs IOPS, and a storage tier decision framework. *(Themes: Storage · Infrastructure · Economics)*

- **[The Operational Geometry of Spatial Systems](deep-dives/the-operational-geometry-of-spatial-systems.md)** — Quadtree vs R-tree vs H3, COG vs GeoParquet, H3 partitioning, routing graph vs raster cost surfaces. *(Themes: Spatial · Architecture · Data Formats)*

- **[The Hidden Cost of Metadata Debt](deep-dives/the-hidden-cost-of-metadata-debt.md)** — Catalog drift, control plane collapse, duplicate pipelines, compliance failure, and progressive enforcement. *(Themes: Governance · Data Architecture · Economics)*

### Recursive Cathedral Generator (Kotlin + Processing)

- **[Recursive Cathedral Generator](tutorials/just-for-fun/kotlin-recursive-cathedral.md)** — Grow Gothic cathedral silhouettes from recursive L-system rules. Deterministic seeds, bilateral symmetry, exportable PNG frames.

- **[Pi-Based Sample Library Server with Live Audition](tutorials/just-for-fun/pi-sample-server.md)** — Raspberry Pi sample server with SQLite indexing, Nginx, FastAPI, WebAudio API, and USB MIDI live audition.

## Late 2025

- **[Vibe → Agentic LLMs](best-practices/ml-ai/vibe-to-agentic.md)** — Moving from prompt experimentation to production-grade agentic LLM architectures.

- **[Fractal Art Explorer (JavaScript)](tutorials/just-for-fun/fractal-art-explorer-js.md)** — Real-time Mandelbrot/Julia explorer with GPU acceleration, custom palettes, and orbit traps.

- **[OSC + MQTT + Prometheus + SuperCollider](tutorials/just-for-fun/osc-mqtt-prometheus-supercollider.md)** — Wire Prometheus exporters into SuperCollider via OSC for live data sonification.

- **[MCP + FastAPI Full Stack](best-practices/ml-ai/mcp-fastapi-stack.md)** — Model Context Protocol integrated with FastAPI for production LLM toolchains.

- **[Cross-Domain Identity Federation](best-practices/security/identity-federation-authz-authn-architecture.md)** — Unified identity federation architecture across system layers and environments.

- **[Environment Promotion Drift Governance](best-practices/operations-monitoring/environment-promotion-drift-governance.md)** — Controlled environment promotion with drift prevention and release channel management.

- **[Data Quality SLA Validation](best-practices/data-governance/data-quality-sla-validation-observability.md)** — Data quality governance with SLAs and multi-layer validation for tabular, geospatial, and ML data.

## Mid 2025

- **[IAM & RBAC Governance](best-practices/security/iam-rbac-abac-governance.md)** — Identity and access management across heterogeneous stacks.

- **[Release Management & Progressive Delivery](best-practices/operations-monitoring/release-management-and-progressive-delivery.md)** — Safe deployment strategies across applications, databases, data pipelines, and ML systems.

- **[Holistic Capacity Planning](best-practices/architecture-design/capacity-planning-and-workload-modeling.md)** — Workload modeling, scaling economics, and resource prediction frameworks.

- **[ONNX Browser Inference](tutorials/ml-ai/onnx-browser-inference.md)** — Run ML models in the browser with ONNX Runtime Web.

- **[RKE2 on Raspberry Pi Farm](tutorials/docker-infrastructure/rke2-raspberry-pi.md)** — Kubernetes cluster on ARM hardware.

- **[PostGIS Geometry Indexing](tutorials/database-data-engineering/postgis-geometry-indexing.md)** — Spatial index strategies, operator classes, and query plans for production PostGIS.

---

!!! tip "Want to contribute or suggest content?"
    Open an issue or PR on [GitHub](https://github.com/sempervent/sempervent.github.io). All content requests considered.
