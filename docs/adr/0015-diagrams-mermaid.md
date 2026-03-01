---
tags:
  - adr
  - diagrams
  - documentation
---

# ADR 0015: Standardize Site Diagrams on Mermaid (with SVG as Optional Exception)

**Status**: Accepted
**Date**: 2026-02-27
**Deciders**: Joshua N. Grant
**Technical Story**: Replace ad-hoc ASCII diagrams with a consistent, maintainable diagram standard that is version-friendly and renders cleanly across the site.

---

## Context

The site increasingly relies on conceptual diagrams to explain architecture, trade-offs, and system boundaries. Many diagrams are currently written as ASCII. While ASCII is portable and fast to write, it has several drawbacks:

- Limited expressive power for complex relationships
- Inconsistent visual vocabulary across pages
- Reduced readability on mobile and narrow screens
- Difficult to reuse or standardize diagram patterns
- Perceived "sketchy" quality even when content is rigorous

SVG diagrams can look excellent but tend to introduce tooling overhead:

- Requires authoring tools or generation pipelines
- Harder to edit and review as text diffs
- Risk of inconsistent styling unless a strict convention exists

Mermaid offers a middle ground:

- Text-based diagrams (diff-friendly, reviewable)
- Easy to embed in Markdown
- Supports common diagram types (flowcharts, sequence diagrams, state diagrams)
- Low friction for authors
- Rendered by MkDocs Material through `pymdownx.superfences` — already configured

---

## Decision

We will standardize on **Mermaid** for the majority of diagrams across the site.

- Mermaid becomes the default for architecture, flow, state, and sequence diagrams.
- SVG remains an optional exception for:
  - High-resolution posters and figures
  - Geospatial illustrations and maps
  - Cases where Mermaid is insufficient or visually unclear

We will:

1. Confirm Mermaid rendering is enabled in `mkdocs.yml` (it is — via `pymdownx.superfences` custom fence).
2. Add a diagram style guide at `docs/diagrams/style-guide.md`.
3. Convert a starter set of high-value ASCII diagrams to Mermaid (see Follow-ups).
4. Gradually convert remaining ASCII diagrams during normal content evolution.

---

## Alternatives Considered

**Continue using ASCII diagrams**: Rejected. Lacks consistency and visual clarity at scale.

**Adopt SVG as the default**: Rejected. Higher authoring friction and tooling requirements.

**Use both equally without a default**: Rejected. Without a default standard, style fragments and diagrams drift.

---

## Consequences

**Positive**:
- Higher clarity and visual consistency across the site
- Better mobile readability
- Text-based diffs for diagrams in pull requests
- Reduced diagram authoring friction compared to SVG
- Stronger perceived polish without sacrificing maintainability

**Negative / Trade-offs**:
- Mermaid features vary by diagram type and renderer version
- Some complex diagrams may require careful layout constraints
- Small learning curve for contributors

---

## Follow-ups

- [x] Confirm `pymdownx.superfences` mermaid custom fence is in `mkdocs.yml`
- [x] Add `docs/diagrams/style-guide.md` with snippet library
- [x] Convert core doctrine diagrams: Prefect vs Airflow, ESP32 vs Pi, Data Pipelines, Observability vs Monitoring
- [ ] Convert remaining ASCII diagrams gradually during content evolution
