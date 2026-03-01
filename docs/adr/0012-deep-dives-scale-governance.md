---
tags:
  - adr
  - governance
  - deep-dives
---

# ADR 0012: Governance Model for Large-Scale Deep Dive Section

**Status**: Accepted
**Date**: 2026-02-26
**Supersedes**: Curation criteria in ADR 0010 and ADR 0011
**Context**: Deep Dives section has grown to 24 essays across 6 thematic clusters.

---

## Context

The Deep Dives section was established in ADR 0007 to provide a space for long-form analytical essays distinct from tutorials and best practices. It has since expanded through four phases of addition (ADR 0008, 0009, 0010, 0011) and now contains 24 essays spanning data formats, data systems, systems architecture, observability, infrastructure automation, embedded systems, and GPU economics.

At this scale, the section faces three governance challenges that require explicit policy:

1. **Curation discipline**: not every analytical question warrants a new deep dive. Without a clear novelty test, the section will accumulate essays that repeat one another, dilute intellectual signal, and create maintenance burden without proportional reader value.

2. **Cross-linking requirements**: at 24 essays, the potential cross-link surface is large. Without policy, cross-linking becomes arbitrary (linking everything to everything) or stale (links added at creation time but not maintained as the corpus grows).

3. **Thematic clustering**: the navigation must remain legible as the section grows. Sub-sections that contain too few or too many essays degrade navigation usability. Cluster boundaries must be governed explicitly.

---

## Decision

### Curation Discipline

A new deep dive is warranted only when it satisfies all of the following:

**Novelty test**: the essay addresses a question not adequately covered by any existing deep dive, even by implication. If the answer to "does this already exist?" is "mostly, in another essay," the appropriate action is to add a section to the existing essay, not create a new file.

**Analytical substance test**: the essay must contain genuine comparative analysis — trade-off matrices, historical context, ecosystem maturity commentary, economic implications — not just an informed opinion or a list of considerations. If it cannot sustain 800+ words of structured argument, it is not a deep dive.

**Durability test**: the essay must address a structural question with multi-year relevance. Technology announcements, version-specific comparisons, and tactical "how to choose" guides are not deep dives. A deep dive on Kubernetes overhead remains relevant as Kubernetes itself evolves; an essay on "Kubernetes 1.29 upgrade tips" does not.

**Non-tutorial test**: the essay must not contain step-by-step implementation instructions. If it does, the implementation content belongs in the Tutorials section and the analytical content belongs in Deep Dives or Best Practices.

### Cross-Linking Requirements

Every new deep dive must include at minimum 2 and at maximum 5 internal cross-links to other deep dives, placed at the opening (header note) and/or closing (See also admonition). Cross-links must be directionally meaningful — they should explain *why* the linked essay is relevant, not merely that it exists.

When a new deep dive is added, at minimum 2 existing essays that are conceptually adjacent must be updated to add a reciprocal cross-link. This maintains the web of connections as the corpus grows and prevents new essays from being isolated.

Cross-links to Best Practices and Tutorial pages are appropriate when the deep dive directly extends or contextualizes a practice or tutorial. These links should appear in the See also admonition only.

### Thematic Clustering

The six current thematic clusters are:

| Cluster | Current count | Soft maximum |
|---|---|---|
| Data Formats & Storage | 3 | 6 |
| Data Systems & Architecture | 8 | 12 |
| Systems Design & Architecture | 4 | 7 |
| Operations & Reliability | 2 | 5 |
| Infrastructure & Automation | 4 | 7 |
| Embedded & Radio | 3 | 6 |

When a cluster approaches its soft maximum, the appropriate response is one of:
- Split the cluster into two sub-clusters with distinct thematic identity
- Merge essays that substantially overlap into a revised composite essay
- Enforce the novelty test more strictly for proposed additions to that cluster

Sub-cluster maximum: 12 essays per thematic cluster in the navigation. Beyond 12, navigation usability degrades to the point where sub-cluster splitting is mandatory.

### Obsolescence Handling

Deep dives become partially obsolete when the ecosystem evolves significantly: a technology ceases to be relevant, a comparison changes due to product acquisition or open-source abandonment, or the fundamental trade-off the essay describes resolves in one direction definitively. The response is:

- **Minor obsolescence** (one section is stale): add a dated note at the beginning of the affected section. Do not rewrite the essay.
- **Major obsolescence** (the central thesis is invalidated): mark the essay with a deprecation notice at the top, link to any successor essay, and leave the original content in place for historical context.
- **Supersession** (a new essay covers the same territory with superior analysis): the new essay should explicitly state what it supersedes; the old essay should link to the new one.

---

## Thematic Cluster Map (Current State)

At the time of this ADR:

```
Deep Dives (24 essays)
│
├── Data Formats & Storage (3)
│   Parquet · Geospatial Formats · Polars vs Pandas Geospatial
│
├── Data Systems & Architecture (8)
│   Lakehouse · DuckDB vs PG vs Spark · Data Pipeline Failures ·
│   Data Lakes → Swamps · Prefect vs Airflow · Metadata as Infra ·
│   Hidden Cost of Real-Time · End of the Data Warehouse?
│
├── Systems Design & Architecture (4)
│   Monolith vs Microservices · Appropriate Microservices ·
│   Distributed Scale Myths · Kubernetes Overhead
│
├── Operations & Reliability (2)
│   Observability vs Monitoring · Economics of Observability
│
├── Infrastructure & Automation (4)
│   Container Base Images · IaC vs GitOps ·
│   Human Cost of Automation · Economics of GPU Infrastructure
│
└── Embedded & Radio (3)
    LoRaWAN vs Raw LoRa · MQTT vs HTTP · ESP32 vs Raspberry Pi
```

---

## Rationale

The governance model prioritizes intellectual coherence over volume. A corpus of 24 high-quality essays is more valuable than a corpus of 50 essays of inconsistent depth. The soft cluster maximums create a forcing function: adding a new essay to a near-full cluster requires either demonstrating the novelty test is met or splitting the cluster, both of which impose a productive discipline on scope.

The cross-linking requirement prevents the section from fragmenting into isolated essays. The reciprocal link requirement is the mechanism that keeps the web of connections current as the corpus grows.

---

## Alternatives Considered

**No explicit governance**: allow the section to grow organically, with ADRs created only for major structural changes. Rejected because at 24 essays, without explicit policy, the section will accumulate redundant essays and navigation will degrade.

**Hard essay limits**: set an absolute maximum (e.g., 30 essays). Rejected because the correct limit is topical coverage, not count. A section with 25 essays each addressing a distinct structural question is better than 20 essays with significant overlap.

**Separate site section for each thematic cluster**: treat Data Systems deep dives as a separate section from Systems Design deep dives. Rejected because the cross-cluster linkage is a feature — data system decisions interact with systems design decisions — and the unified Deep Dives section preserves navigational context for that interaction.

---

## Consequences

- Future deep dives must pass the novelty, substance, durability, and non-tutorial tests before creation.
- New essays must include 2–5 internal cross-links and trigger reciprocal cross-link updates in at least 2 existing essays.
- Navigation sub-cluster sizes must be monitored; splitting is mandatory when a cluster exceeds 12 entries.
- Obsolescence handling policy is now explicit and must be applied when significant ecosystem changes render existing essays partially stale.
