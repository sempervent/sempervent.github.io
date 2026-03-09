# Cross-System Data Lineage, Inter-Service Metadata Contracts & Provenance Enforcement

**Objective**: Establish data lineage and contract enforcement across pipelines, services, and storage so that provenance is traceable, schema and semantics are agreed, and violations are detectable. When you need to trace data from source to consumption, enforce contracts between producers and consumers, or prove compliance—this guide provides the patterns and integration points.

## Introduction

**Data lineage** answers: where did this dataset or column come from, what transformations were applied, and who consumes it? **Contracts** define the agreed schema, semantics, and freshness expectations between producers and consumers. **Provenance** is the recorded history of how data was produced and moved. Together they make data trustworthy and debuggable across lakehouse, warehouse, Postgres, microservices, and APIs.

Without lineage and contracts, systems drift: a change in one service breaks another, and no one can trace the cause. This document frames the problem, links to the full governance framework, and gives concrete patterns for lineage and contracts.

**What This Guide Covers**:

- Why lineage and contracts matter in distributed data systems
- Lineage tracking patterns (pipeline, column, and operational lineage)
- Contract types: schema, semantic, and freshness
- Enforcement: validation at boundaries, schema registries, and CI
- Provenance storage and query patterns
- Integration with metadata and observability

**Related Documents**:

- **[Metadata Standards, Schema Governance & Data Provenance Contracts](metadata-provenance-contracts.md)** — Full framework for metadata, schema versioning, contracts, and provenance
- **[Data Freshness, SLA/SLO Governance](data-freshness-sla-governance.md)** — Freshness contracts and SLOs
- **[Data Retention, Archival, Lifecycle Governance](data-retention-archival-lifecycle-governance.md)** — Retention and lineage for lifecycle
- **[Metadata as Infrastructure](../data/metadata-control-plane.md)** — Metadata as control plane

## Why Lineage and Contracts Matter

**Lineage** enables:

- **Debugging**: Trace bad data back to source or transformation
- **Impact analysis**: See which consumers are affected by a schema or pipeline change
- **Compliance**: Demonstrate where data came from and how it was processed

**Contracts** enable:

- **Stability**: Producers and consumers agree on schema and semantics; changes are versioned
- **Early failure**: Breaks are caught at build or deployment time, not in production
- **Documentation**: The contract is the single source of truth for the interface

**Provenance** ties them together: each record or dataset carries or references who produced it, when, and from what inputs.

## Lineage Tracking Patterns

- **Pipeline lineage**: Which jobs or services produced this table or file? (e.g. Prefect run X wrote partition Y.)
- **Column lineage**: Which source columns or expressions produced this column? (e.g. for SQL or DAG-based pipelines.)
- **Operational lineage**: Which API calls or events led to this state? (e.g. for event-driven systems.)

Store lineage in a catalog or dedicated store; expose it via APIs and UI so that impact analysis and debugging are possible. Link to **[Metadata Standards, Schema Governance & Data Provenance Contracts](metadata-provenance-contracts.md)** for schema versioning and provenance storage patterns.

## Contract Types and Enforcement

- **Schema contracts**: Column names, types, nullability. Enforce via schema registry, CI checks, or runtime validation at service boundaries.
- **Semantic contracts**: Meaning of fields (e.g. currency, units, enumerations). Enforce via documentation, code, or contract tests.
- **Freshness contracts**: How stale data may be. Enforce via SLOs and monitoring; see **[Data Freshness, SLA/SLO Governance](data-freshness-sla-governance.md)**.

Define contracts at the boundary between producer and consumer; version them and validate on publish and on consume.

## Anti-Patterns

- **No lineage**: Data appears in a table or API with no record of origin or transformation. Debugging and compliance become impossible.
- **Implicit contracts**: Schemas or semantics change without notice; consumers break or produce wrong results.
- **Provenance only in logs**: Lineage exists only in ephemeral logs; no queryable, durable lineage store.

## Summary

Cross-system lineage, inter-service metadata contracts, and provenance enforcement are the backbone of reliable, auditable data systems. Implement lineage tracking and contract boundaries early; integrate with the broader **[Metadata Standards, Schema Governance & Data Provenance Contracts](metadata-provenance-contracts.md)** framework for full governance.
