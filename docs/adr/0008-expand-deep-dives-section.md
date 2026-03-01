---
tags:
  - adr
  - documentation
---

# ADR 0008: Expand Deep Dives as a First-Class Analytical Section

**Status**: Accepted  
**Date**: 2026-02-26  
**Author**: Joshua N. Grant

---

## Context

The site's Deep Dives section was established in ADR 0007 with a single entry (LoRaWAN vs Raw LoRa) as a proof of concept for long-form analytical content. That entry demonstrated the format's value: structured comparative analysis, trade-off matrices, historical context, and decision frameworks — distinct from both tutorials and best-practice checklists.

Several recurring content needs within this site's scope are poorly served by existing content types:

- **Format and storage decisions** involve trade-offs that span years of ecosystem evolution and cannot be reduced to a checklist.
- **System selection decisions** (databases, engines, protocols) require comparative analysis across dimensions that change with organizational scale.
- **Infrastructure philosophy** (container base images, infrastructure automation) involves trade-offs between correctness, operational complexity, and security that are contextual rather than prescriptive.

These topics warrant a separate content type with its own structural guardrails.

---

## Decision

Expand the Deep Dives section from one entry to eight, adding:

1. **Parquet vs CSV vs ORC vs Avro** — columnar storage theory, compression, predicate pushdown, ecosystem analysis
2. **Lakehouse vs Warehouse vs Database** — historical arc, table format emergence, governance complexity, cost structures
3. **DuckDB vs PostgreSQL vs Spark** — execution models, data gravity, hybrid architecture
4. **Geospatial File Format Choices** — raster vs vector, Cloud-Optimized GeoTIFF, GeoParquet emergence
5. **Container Base Image Philosophy** — glibc vs musl, attack surface analysis, operational debugging cost
6. **Infrastructure as Code vs GitOps** — control plane models, drift detection, hybrid governance
7. **MQTT vs HTTP in IoT Systems** — connection models, power consumption, broker centralization, security

These entries are organized under four sub-sections in the navigation:

- Data Formats & Storage
- Data Systems & Architecture
- Infrastructure & Automation
- Embedded & Radio

---

## Why Analytical Essays Deserve Structural Separation

Best practices pages answer the question: "given that I am doing X, how should I do it well?" Tutorials answer: "how do I accomplish X step by step?"

Deep dives answer a different question: "should I be doing X at all, and under what conditions does X beat Y?" This requires a different structure — comparative analysis, historical context, trade-off matrices — and a different tone: analytical and opinionated rather than prescriptive and procedural.

Mixing these content types in the same section would dilute both. Best practices readers want concrete guidance; deep dive readers want structured argument. Structural separation preserves the value of each.

---

## Guardrails Against Content Sprawl

Deep dives carry an expansive mandate ("analytical exploration") that can attract poorly scoped content. The following criteria apply to all deep dive candidates:

**A deep dive is appropriate when**:
- The topic involves a genuine architectural or technology selection decision with non-obvious trade-offs
- The decision consequences extend across months or years of system operation
- Comparative analysis across at least two options is possible and meaningful
- The topic cannot be adequately addressed in a best-practice page without becoming a comparative essay

**A deep dive is not appropriate when**:
- The correct answer is well-established and non-controversial
- The topic is better served by a tutorial (step-by-step) or best-practice (how-to-well)
- The topic is too narrow to warrant the deep dive structure (decision framework, trade-off matrix, historical context)
- The topic duplicates an existing deep dive without adding a meaningfully different perspective

---

## Criteria for New Deep Dives

New deep dives should meet all of the following before being added:

1. **Genuine tension**: two or more approaches with legitimate use cases — not a comparison where one option is obviously correct
2. **Consequential decision**: a choice that affects system architecture, data portability, operational complexity, or long-term maintainability
3. **Analytical completeness**: the entry must include a historical or contextual arc, at least one comparative matrix, ASCII diagram(s) where clarifying, and a decision framework section
4. **Appropriate length**: 1,000–3,000 words; shorter entries should be best practices; longer entries should be split into multiple deep dives

---

## Consequences

The Deep Dives section becomes a stable, growing analytical corpus alongside tutorials and best practices. Navigation is organized by domain (formats, systems, infrastructure, embedded) rather than by chronological addition. Cross-links between related deep dives and related tutorials/best practices are maintained as part of the authoring checklist.

The risk of content sprawl is managed by the criteria above. The ADR process itself (this document) serves as the governance mechanism for section-level structural changes.

---

## Related Documents

- [ADR 0007: Deep Dives Section](0007-deep-dives-section.md) — original decision to create the section
- [ADR 0003: Best Practices vs Tutorials](0003-why-best-practices-vs-tutorials.md) — the content type distinction that deep dives extend
