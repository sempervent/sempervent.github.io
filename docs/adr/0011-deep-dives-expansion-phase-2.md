---
tags:
  - adr
  - documentation
---

# ADR 0011: Deep Dives Expansion Phase II

**Status**: Accepted  
**Date**: 2026-02-26  
**Author**: Joshua N. Grant

---

## Context

ADR 0010 established a consolidated governance model for the Deep Dives section and noted that future deep dive entries do not require an ADR — they require adherence to the criteria established in ADR 0010. This ADR documents Phase II of the corpus expansion as a record of scope decisions rather than as a governance update.

Phase II adds five entries across three new topic clusters: real-time systems economics, data lake governance failure, and distributed systems theory. A fourth entry enhances the existing metadata-as-infrastructure document with a data plane/control plane framework and a metadata debt section. The Human Cost of Automation addresses the organizational side of the infrastructure automation deep dives already present in the corpus.

The corpus now contains twenty-two entries across six sub-sections.

---

## Entries Added in Phase II

**Data Systems & Architecture**

- **The Hidden Cost of Real-Time Systems** — latency taxonomy (hard/soft/streaming/batch), infrastructure costs of stateful stream processing (checkpointing, backpressure, deduplication), observability explosion in streaming, cost comparison table, and a decision framework from batch through micro-batch to streaming
- **Why Most Data Lakes Become Data Swamps** — structural decay mechanisms (missing metadata, schema drift, orphaned datasets, version confusion), technical anti-patterns, cultural failures, lakehouse as partial solution, and zone-based architecture for prevention
- **Metadata as Infrastructure (enhanced)** — addition of data plane vs metadata plane ASCII diagram, metadata debt taxonomy (stale descriptions, broken lineage, inconsistent tags), and updated title framing metadata as a hidden control plane rather than mere documentation

**Systems Design & Architecture**

- **Distributed Systems and the Myth of Infinite Scale** — CAP theorem and PACELC analysis, N² coordination growth diagram, data gravity and cloud egress cost, partial failure taxonomy, rolling upgrade complexity, and a decision framework distinguishing genuine distribution requirements from distribution-as-prestige

**Infrastructure & Automation**

- **The Human Cost of Automation** — skill atrophy mechanisms, complexity transfer from visible toil to invisible fragility, psychological impact of reduced operator agency, automation maturity model (manual through self-healing), and a decision framework with explicit criteria for when automation creates more organizational risk than it removes

---

## Criteria for Future Inclusion

ADR 0010 remains the authoritative governance document. For reference, the criteria are:

**Topic**: genuine architectural tension, consequential scope (1–3 year horizon), non-duplication of existing entries, and analytical durability (not version-specific).

**Structure**: Opening Thesis → Historical Context → Analytical Core (with comparison and diagram) → Decision Framework → Cross-links.

**Tone**: analytical (not tutorial, not marketing, not blog voice). Positions must be argued with evidence and logical structure.

---

## Avoiding Intellectual Dilution

The risk at twenty-two entries is that the corpus begins to include topics that are adjacent to but not clearly within the analytical scope of the section. The test question from ADR 0010 remains the primary filter: would this content belong in an engineering organization's internal knowledge base, cited in architecture review documents?

Two additional tests apply at scale:

**The novelty test**: does the proposed entry add an analytical perspective not already present in the corpus? An entry on "Why Most Kubernetes Deployments Are Overcomplicated" that makes the same structural argument as "Why Most Microservices Should Be Monoliths" applied to a different technology does not pass the novelty test. The corpus should provide a range of analytical perspectives, not the same perspective applied to many technology choices.

**The cross-link test**: a well-positioned deep dive should naturally link to and from at least two existing entries. A proposed entry with no natural cross-links to existing corpus entries is likely analytically isolated from the corpus's themes — a signal that it is not well-suited to the section.

---

## Cross-Linking Discipline

Cross-links in the Deep Dives corpus serve a function beyond navigation: they reveal the analytical graph structure of the corpus. Entries that form a cluster of mutual cross-links address related analytical questions. The current corpus has two main analytical clusters:

**Data systems cluster**: Parquet, Lakehouse, DuckDB, Pipelines, Data Swamps, Metadata, Real-Time, Prefect/Airflow — all address different dimensions of the same set of trade-offs in analytical data architecture.

**Systems and operations cluster**: Microservices, Distributed Systems, Observability, Economics of Observability, Automation, IaC/GitOps, Containers — all address different dimensions of how software systems are built, operated, and governed.

**Embedded cluster**: LoRaWAN, MQTT, ESP32 — domain-specific analytical content for the IoT/embedded systems audience.

Cross-links should predominantly connect entries within clusters. Cross-cluster links are appropriate when the analytical connection is genuine (e.g., the Economics of Observability links to Data Pipeline observability) rather than merely topical.

---

## Consequences

The corpus is mature enough to serve as a coherent analytical reference rather than a collection of independent essays. Future entries should be evaluated for their contribution to the corpus's analytical structure — do they deepen an existing cluster, or do they begin a new cluster with at least two related entries? Isolated single entries on topics without natural corpus neighbors should be evaluated skeptically.

---

## Related Documents

- [ADR 0010: Governance Model for Deep Dives](0010-deep-dives-governance.md) — authoritative governance (criteria, tone, structure, cross-linking)
- [ADR 0009: Deep Dives Expansion](0009-deep-dives-expansion.md) — Phase I expansion record
