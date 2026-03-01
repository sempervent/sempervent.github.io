---
tags:
  - adr
  - governance
  - deep-dives
---

# ADR 0013: Deep Dive Curation and Thematic Clustering Model

**Status**: Accepted
**Date**: 2026-02-26
**Supersedes**: Curation criteria in ADR 0012 (governance thresholds now updated)
**Context**: Deep Dives section has grown to 29 essays across 6 thematic clusters.

---

## Context

ADR 0012 established a governance model for the Deep Dives section when it contained 24 essays. This ADR refines that model as the corpus has grown to 29 essays — crossing the ADR 0012 soft cluster maximum for two clusters (Data Formats & Storage: 5 entries; Data Systems & Architecture: 10 entries) — and as experience with the cross-linking and curation requirements has revealed patterns that require more explicit guidance.

The primary governance challenge at this scale is **conceptual drift**: essays that individually satisfy the novelty and substance tests but that, in aggregate, produce a corpus whose thematic coherence is harder for readers to perceive. A corpus of 29 essays organized into 6 clusters is legible only if the cluster boundaries are sharp enough that readers can use them to navigate rather than reading the entire index.

The secondary challenge is **cross-link maintenance**: at 29 essays, the potential cross-link surface is large enough that maintaining all cross-links manually as essays are added requires disciplined tracking. The current practice of adding cross-links only at creation time and updating 2–5 existing pages produces a cross-link graph that grows slower than the corpus.

---

## Decision

### Curation: Thematic Fit as Prerequisite

Every proposed deep dive must be assignable to exactly one existing thematic cluster before creation. If an essay spans multiple clusters with equal weight, it either belongs in the cluster where its *primary argument* lives, or it reveals a missing cluster boundary. Essays that cannot be assigned to a cluster without forcing are evidence of scope problem rather than thematic novelty.

The curation question is: **"what cluster would a reader look in to find this essay?"** If the answer requires describing a cluster that does not exist, the question becomes: "is there sufficient topical density to justify a new cluster, or is the essay an edge case?"

### Cluster Definitions (Updated)

The six clusters are now explicitly defined by their organizing analytical question:

| Cluster | Organizing question | Current count |
|---|---|---|
| Data Formats & Storage | What physical and format-level constraints govern how data is stored and accessed? | 5 |
| Data Systems & Architecture | What architectural and organizational forces shape analytical data systems? | 10 |
| Systems Design & Architecture | What structural limits constrain the design of distributed and decomposed software systems? | 4 |
| Operations & Reliability | What operational and economic forces govern system observability, monitoring, and reliability? | 2 |
| Infrastructure & Automation | What organizational and technical trade-offs govern infrastructure acquisition, automation, and compute economics? | 5 |
| Embedded & Radio | What physical and ecosystem constraints govern embedded system and radio protocol selection? | 3 |

**Data Systems & Architecture** has reached the soft maximum of 10. The next addition to this cluster requires either demonstrating that the proposed essay has a sufficiently distinct primary argument from the 10 existing essays, or splitting the cluster into two sub-clusters. Candidate split: "Data Storage & Table Formats" (Lakehouse, Warehouse, Parquet, Metadata, Data Lakes, Data Pipelines) vs "Data Computation & Orchestration" (DuckDB/PG/Spark, Prefect/Airflow, Real-Time, ML Systems).

### Cross-Linking: Minimum Requirements

Every new essay must:

1. Include a header cross-link (`*See also: ...*`) to 1–2 essays in *other* clusters, explaining why the reader should continue there.
2. Include a closing `!!! tip "See also"` admonition with 3–5 cross-links with directionally meaningful descriptions.
3. Trigger reciprocal updates in at least 2 existing essays that are the most thematically adjacent.

The purpose of cross-cluster header links is navigational: a reader who arrived at a Data Systems essay from the index should have a visible path to the Infrastructure cluster if the essay assumes infrastructure concepts. The purpose of closing cross-links is analytical: they tell the reader what questions to pursue after this essay.

### Avoiding Conceptual Drift

The symptom of conceptual drift is when a reader looking at the index cannot determine, from the title and one-sentence description, what distinguishes one essay from three adjacent essays. The prevention mechanism is the **distinction test**: before creating a new essay, identify the 2–3 most topically similar existing essays and articulate, in one sentence, how the new essay's central argument differs from each. If the distinction cannot be articulated clearly, the essay is not ready for creation.

---

## Current Corpus Map (29 essays)

```
Deep Dives (29 essays)
│
├── Data Formats & Storage (5)
│   Parquet · Geospatial Formats · Polars vs Pandas Geospatial ·
│   Physics of Storage Systems · Operational Geometry of Spatial Systems
│
├── Data Systems & Architecture (10) ← approaching split threshold
│   Lakehouse · DuckDB vs PG vs Spark · Data Pipeline Failures ·
│   Data Lakes → Swamps · Prefect vs Airflow · Metadata as Infra ·
│   Hidden Cost of Real-Time · End of the Data Warehouse? ·
│   Hidden Cost of Metadata Debt · Why ML Systems Fail in Production
│
├── Systems Design & Architecture (4)
│   Monolith vs Microservices · Appropriate Microservices ·
│   Distributed Scale Myths · Kubernetes Overhead
│
├── Operations & Reliability (2)
│   Observability vs Monitoring · Economics of Observability
│
├── Infrastructure & Automation (5)
│   Container Base Images · IaC vs GitOps ·
│   Human Cost of Automation · Economics of GPU Infrastructure ·
│   Myth of Serverless Simplicity
│
└── Embedded & Radio (3)
    LoRaWAN vs Raw LoRa · MQTT vs HTTP · ESP32 vs Raspberry Pi
```

---

## Rationale

The governance model deliberately imposes friction on new additions. The cost of this friction is slower corpus growth. The benefit is a corpus whose individual essays are more distinct from one another, whose cluster boundaries are more navigable, and whose cross-link graph is maintained rather than stale. The section is more valuable at 30 high-distinction essays than at 50 essays with significant conceptual overlap.

The cluster split decision for Data Systems & Architecture is deferred to the next addition; the split should be made based on where the 11th essay falls, not in anticipation of a hypothetical essay.

---

## Consequences

- Proposed essays must satisfy the distinction test (articulated difference from 2–3 most similar existing essays) before creation.
- Data Systems & Architecture approaching 10 essays requires monitoring; the 11th addition triggers a cluster split decision.
- Cross-linking minimums (header link + 3–5 closing links + 2 reciprocal updates) are formalized as creation requirements.
- ADR 0012 curation criteria remain valid; this ADR refines them for a larger corpus and adds the distinction test and cluster definition model.
