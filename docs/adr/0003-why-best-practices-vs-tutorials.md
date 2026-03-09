# ADR-0003: Separate "Best Practices" from "Tutorials" as Distinct Top-Level Sections

**Status**: Accepted

**Date**: 2024-02-01

**Deciders**: Joshua N. Grant

**Tags**: information-architecture, documentation

---

## Context

As the site grew beyond 50 pages, two distinct types of content emerged with different authoring patterns and reader goals:

1. **Conceptual/opinionated guides** — "How to think about X", pattern references, governance frameworks, architectural principles. Readers consult these when making decisions. They rarely run commands from these pages.

2. **Step-by-step implementations** — "How to do X right now", with runnable commands, copy-paste configs, and clear success criteria. Readers follow these linearly to accomplish a specific task.

Mixing both types under a single nav section forced readers to guess which kind of content they were getting before clicking.

## Decision

Maintain **two top-level sections**:

- **Best Practices** — patterns, architectures, governance, opinionated guides. No assumed prerequisite of "do this exact thing right now."
- **Tutorials** — step-by-step, task-oriented, runnable. Has clear Prerequisites, Steps, Verify, and Troubleshoot structure.

Cross-link between the two wherever a tutorial implements a best practice.

## Options Considered

### Option 1: Single "Documentation" section (flat)

**Pros:**
- Simpler nav structure
- No category ambiguity

**Cons:**
- Readers can't tell if a page is "read to understand" vs "follow to implement"
- Harder to maintain consistent page structure when types are mixed
- Harder to recommend "start here" paths for different reader goals

### Option 2: Two sections: Best Practices + Tutorials (chosen)

**Pros:**
- Mental model matches reader intent: "I want to understand" vs "I want to implement"
- Consistent page templates per section (best practice has Context/Pattern/Trade-offs; tutorial has Prereqs/Steps/Verify)
- Cross-links between sections surface the relationship between concept and implementation

**Cons:**
- Some content genuinely fits both categories (e.g., a "best practices" page with runnable examples)
- Duplication risk if a best-practice and tutorial cover the same topic without linking to each other

### Option 3: Tag-based (single section, discover by tag)

**Pros:**
- No category assignment needed
- Tags can span multiple dimensions

**Cons:**
- Requires robust tagging from day one (which doesn't exist at authoring time)
- Harder to build a coherent "recommended reading order" without section structure

## Rationale

The two-section model matches the dominant reader intents on this site. Engineers who know what they want to build open Tutorials. Engineers who are evaluating approaches open Best Practices. Keeping them separate preserves the ability to enforce different page templates per section and recommend distinct "Start Here" paths per audience.

## Consequences

### Positive
- Clear authoring expectations per section
- Separate "Start Here" recommendations per section (technical decision-makers vs implementors)
- Cross-links between sections create a richer content graph

### Negative
- Some pages are genuinely ambiguous in category (handled by placing in the section that best matches the primary reader intent, then cross-linking)
- Nav depth increases because each section has its own subsections

### Neutral / Trade-offs
- The "Just for Fun" section under Tutorials is an intentional exception: it's neither purely conceptual nor purely task-driven, but it belongs in Tutorials because readers follow it to build something

## Related Documents

- [ADR-0002: Why MkDocs + Material](0002-why-mkdocs-material.md)
- [Technical Documentation](../documentation.md) — site structure and navigation
