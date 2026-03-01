---
tags:
  - adr
  - documentation
---

# ADR 0009: Expand Deep Dives into Core Analytical Layer

**Status**: Accepted  
**Date**: 2026-02-26  
**Author**: Joshua N. Grant

---

## Context

ADR 0008 formalized the expansion of Deep Dives from one entry (LoRaWAN vs Raw LoRa) to eight entries, establishing structural sub-sections and criteria for new entries. This document addresses a further expansion: four additional deep dives covering data pipeline reliability, observability architecture, embedded platform selection, and workflow orchestration philosophy.

The expansion brings the Deep Dives section to twelve entries across five sub-sections (Data Formats & Storage, Data Systems & Architecture, Operations & Reliability, Infrastructure & Automation, Embedded & Radio). This constitutes a material increase in scope — the section is no longer a collection of isolated analytical essays but an interconnected analytical corpus with deliberate cross-linking and a consistent epistemic standard.

---

## Decision

Accept the following four deep dives into the corpus:

1. **Why Most Data Pipelines Fail** — organizational and architectural failure modes, schema drift, ownership ambiguity, the CI/CD gap in data systems, and survivable architectural patterns
2. **Observability vs Monitoring** — monitoring vs observability semantics, three pillars analysis, cardinality economics, data pipeline observability, cultural factors, tooling landscape
3. **ESP32 vs Raspberry Pi** — microcontroller determinism vs Linux convenience, power profiles, security surface, ecosystem maturity, and deployment decision matrix
4. **Prefect vs Airflow** — DAG-first vs Python-native orchestration philosophy, execution model architecture, failure semantics, scaling models, and operational burden

Add a new sub-section to the nav: **Operations & Reliability** (currently containing the Observability vs Monitoring deep dive, with room for additional entries in the reliability and incident management domain).

---

## Why These Are Not Blog Posts

Blog posts are timely, personal, and conversational. They may be opinionated without systematic justification, and they do not typically include formal decision frameworks or comparative analyses.

Deep Dives in this corpus are:

- **Structured**: every entry follows a consistent schema (thesis, context, analysis, comparison, decision framework) that enables comparative reading across entries
- **Durable**: the analysis is written to remain relevant across years, not weeks; where technology changes rapidly, the structural principles rather than specific tools are the primary focus
- **Referenced**: cross-links to related deep dives, best practices, and tutorials make each entry part of a navigable analytical corpus
- **Analytical**: positions are taken and justified with evidence and logical argument, not asserted as opinions
- **Non-promotional**: vendor mentions are comparative and contextual, not endorsements

The test question: would this content belong in an engineering organization's internal knowledge base, cited in architecture review documents? If yes, it is a deep dive. If it reads like a personal take on a recent controversy, it is a blog post.

---

## Criteria for Inclusion

A proposed deep dive must satisfy all of the following:

**1. Genuine and consequential tension**: the topic must involve a real architectural decision with non-obvious trade-offs and meaningful consequences for system longevity, team capability, or operational cost. "Which library should I use for X?" is not a deep dive topic. "What does the choice of orchestration model imply for organizational ownership of data pipelines?" is.

**2. Analytical completeness**: the entry must include at minimum: a historical or contextual arc, a comparative analysis of two or more approaches, at least one structured comparison (table, matrix, or diagram), and a decision framework that provides actionable guidance.

**3. Appropriate scope**: a deep dive should address a single coherent question or tension. If the scope requires more than 3,000 words to be analytically complete, it is likely too broad and should be split. If it can be addressed in fewer than 800 words, it is likely a best practice annotation, not a deep dive.

**4. Durability**: the analysis should remain relevant for at least two to three years without significant revision. Technology-specific entries (e.g., "Airflow 2.6 vs 2.7") fail this criterion. Platform-philosophy entries (e.g., "DAG-first vs code-first orchestration models") satisfy it.

**5. Non-overlap with existing content**: a proposed deep dive must not substantially duplicate an existing deep dive. It may reference, contrast, or extend existing deep dives, but the primary analytical contribution must be distinct.

---

## Guardrails Against Dilution

The Deep Dives section degrades in analytical value if it accumulates:

- Entries that are tutorials in analytical clothing (detailed implementation instructions with light framing)
- Entries that are opinion pieces without analytical structure
- Entries on topics so narrow that they would better serve as best practice annotations
- Entries on topics so broad that they cannot be addressed rigorously in a single document
- Entries duplicating content from existing deep dives without a meaningfully distinct analytical contribution

The ADR process serves as the formal check on section-level structural changes (adding new sub-sections, reorganizing the corpus). Individual deep dive entries do not require an ADR if they meet the criteria above. New sub-sections or reorganizations require an ADR update.

---

## Consequences

The Deep Dives section becomes a twelve-entry analytical corpus with five sub-sections. Cross-links between related entries are maintained as part of authoring discipline. The "Operations & Reliability" sub-section is established as a home for future entries addressing failure analysis, incident management, and reliability engineering at the conceptual level.

---

## Related Documents

- [ADR 0008: Expand Deep Dives as a First-Class Analytical Section](0008-expand-deep-dives-section.md) — previous expansion and criteria formalization
- [ADR 0007: Deep Dives Section](0007-deep-dives-section.md) — original section creation
