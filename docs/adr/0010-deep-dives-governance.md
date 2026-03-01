---
tags:
  - adr
  - documentation
---

# ADR 0010: Governance Model for Deep Dives

**Status**: Accepted  
**Date**: 2026-02-26  
**Author**: Joshua N. Grant

---

## Context

The Deep Dives section has grown to seventeen entries across six sub-sections through ADRs 0007–0009. Three successive expansion ADRs have each restated criteria for inclusion with minor variations. This ADR consolidates the governance model into a single authoritative reference, superseding the inclusion criteria statements in ADRs 0008 and 0009.

The growth of the section raises a specific risk: as the entry count increases, the average quality and analytical rigor of entries tends to decline. The first entries in any analytical corpus establish a standard that becomes harder to maintain as the pressure to fill in topics increases. Explicit governance makes the standard visible and enforceable.

---

## Why Long-Form Analytical Content Requires Curation

Long-form analytical content is not self-governing. The absence of curation produces a set of tendencies that degrade corpus quality over time:

**Scope inflation**: without defined scope, entries expand to cover everything tangentially related to the topic. An entry that begins as an analysis of distributed tracing trade-offs expands to include step-by-step setup instructions, vendor comparison tables, and configuration recommendations. The analytical character is diluted by operational detail that belongs in best practices or tutorials.

**Tone drift**: without a defined tone standard, analytical entries drift toward tutorial voice (instructional), blog voice (conversational and opinionated without systematic justification), or vendor-advocacy voice (promotional). Each is appropriate in a different content type; none is appropriate for this section.

**Redundancy**: without cross-reference awareness, entries duplicate analysis that already exists in the corpus. The economics of observability overlap with the observability vs monitoring entry; the appropriate use of microservices overlaps with the monolith entry. Redundancy is acceptable if the analytical angle is genuinely distinct; it is wasteful if it is merely repetition.

**Obsolescence**: analytical content on rapidly changing technologies can become misleading without update. An entry on a specific tool's architecture that was accurate in 2024 may be incorrect by 2026. The governance model must include a policy for handling obsolescence.

---

## Criteria for Inclusion

A proposed deep dive must satisfy all criteria in each category.

### Topic Criteria

1. **Genuine architectural tension**: the topic involves a decision with consequential trade-offs that cannot be resolved by a simple rule. Decisions where one option is obviously correct for all contexts do not merit deep dive treatment.

2. **Consequential scope**: the decision affects system longevity, team capability, operational cost, or organizational structure across a time horizon of at least one to two years. Short-term implementation decisions belong in tutorials or best practices.

3. **Non-duplication**: the proposed entry must not substantially duplicate an existing entry. A different analytical angle on the same topic is acceptable if the angle is explicitly identified and the new entry cross-references the existing one. Mere restatement is not acceptable.

4. **Durability**: the core analysis must remain relevant for two to three years without requiring substantive revision. Tool-specific version comparisons and feature checklists fail this criterion. Philosophy-level comparisons of execution models, governance approaches, or architectural trade-offs satisfy it.

### Structural Criteria

Every deep dive must include, in order:

1. **Opening Thesis** — a clear statement of the analytical claim the document supports. Not "this document discusses X"; a substantive claim about X that the document then argues for.

2. **Historical Context** — the conditions that produced the current situation, the predecessor approaches, and why the current alternatives exist.

3. **Analytical Core** — the comparison, trade-off analysis, or architectural examination. This section should include at least one structured comparison (table, matrix, or ASCII diagram) and at least two to three paragraphs of dense analytical prose per major claim.

4. **Decision Framework** — actionable guidance organized by stakeholder context (team size, regulatory environment, operational maturity, workload characteristics). Not "it depends" without criteria; specific criteria for each branch of the decision.

5. **Cross-links** — at minimum two links to related content in the site (deep dives, best practices, tutorials, ADRs). Cross-links are not decorative; they should reflect genuine analytical relationships.

### Tone Criteria

All deep dives must:

- Avoid tutorial voice: no step-by-step instructions, no "to do X, first do Y" sequences
- Avoid marketing voice: no vendor endorsements, no superlative claims without evidence, no promotional language for any tool or platform
- Avoid blog voice: positions must be argued with evidence and logical structure, not asserted as personal opinions
- Avoid hedging to the point of uselessness: the Decision Framework must make actual recommendations, not merely list considerations

---

## Sub-Section Governance

New sub-sections may be created when at least two deep dives share a coherent analytical domain that is not adequately captured by existing sub-sections. A single entry does not justify a new sub-section. Sub-section creation requires an ADR update.

Current sub-sections and their intended scope:

| Sub-section | Scope |
|---|---|
| Data Formats & Storage | Physical storage formats, encoding trade-offs, cloud-native implications |
| Data Systems & Architecture | Databases, data platforms, pipeline architecture, orchestration |
| Systems Design & Architecture | Service decomposition, distributed systems, scalability trade-offs |
| Operations & Reliability | Observability, monitoring, incident management, operational economics |
| Infrastructure & Automation | Container philosophy, IaC, GitOps, deployment models |
| Embedded & Radio | Microcontroller platforms, IoT protocols, radio systems |

---

## Handling Obsolescence

Entries that contain analysis that has become materially incorrect due to technology changes should be updated rather than deleted. The update should be noted at the top of the document with the date and nature of the update. If the core analytical claim has been invalidated (not merely that specific tools have changed), the entry should be marked as superseded and a new entry created.

Entries that are superseded or significantly updated trigger a review of cross-links from other entries to ensure referenced conclusions remain accurate.

---

## Cross-Link Expectations

Cross-links serve two functions: they help readers navigate the analytical corpus, and they expose the analytical relationships between topics. Entries that are analytically related but not cross-linked create isolated analysis that readers cannot discover through navigation.

Minimum cross-link expectations:
- Every deep dive links to at least two other entries in the corpus (deep dives, best practices, or tutorials)
- Every new entry is linked from at least one existing entry or from the index
- The deep-dives/index.md is updated for every new entry

Cross-links are maintained as part of the authoring process, not as a retrospective cleanup task.

---

## Consequences

This ADR supersedes the inclusion criteria statements in ADRs 0008 and 0009. Those ADRs remain as historical records of the expansion decisions they document; their criteria sections are no longer authoritative.

Future deep dive entries do not require an ADR. They require adherence to the criteria above. ADR updates are required only for structural changes to the section: new sub-sections, reorganization, or material changes to the governance model itself.

---

## Related Documents

- [ADR 0009: Deep Dives Expansion](0009-deep-dives-expansion.md) — previous expansion (criteria superseded by this ADR)
- [ADR 0008: Expand Deep Dives Section](0008-expand-deep-dives-section.md) — previous expansion (criteria superseded by this ADR)
- [ADR 0007: Deep Dives Section](0007-deep-dives-section.md) — original section creation decision
