# ADR 0007: Introduce "Deep Dives" Section for Analytical Long-Form Content

- **Status**: Accepted
- **Date**: 2026-02-26
- **Deciders**: mate (owner)
- **Related**: [ADR-0003](0003-why-best-practices-vs-tutorials.md) (Best Practices vs Tutorials distinction)

---

## Context

The site has two content modes: prescriptive (Best Practices) and procedural (Tutorials). Both are optimized for practical output — a reader leaves with a configuration, a working build, or a mental model of "what to do."

Some subjects resist this framing. They are not best solved by prescription or procedure. They require comparison, historical context, regulatory analysis, threat modeling, and architectural argumentation. Examples already emerging in the content:

- LoRaWAN vs raw LoRa: a decision shaped by spectrum law, security assumptions, infrastructure cost, and fleet size
- MQTT vs CoAP vs HTTP for IoT: protocol choice determined by transport semantics, not code examples
- Scratch vs distroless: not a tutorial topic; a philosophical and operational stance

Attempting to contain these analyses in a Best Practice page produces either a superficial treatment or an overwhelming document that violates the section's prescriptive purpose. Containing them in a tutorial adds irrelevant instructional scaffolding to what is fundamentally an argument.

The existing section structure does not provide a natural home for analytical, argument-driven content.

---

## Decision

Create `docs/deep-dives/` as a top-level content section with its own nav tab: **Deep Dives**.

Content placed here must satisfy all of the following:

- **Comparative**: examines two or more approaches, protocols, architectures, or tools in genuine tension
- **Analytical**: presents reasoning and evidence, not instructions
- **Argument-driven**: arrives at a defensible position or decision framework
- **Less code-heavy**: code snippets are illustrative, never the primary content
- **Intentional**: a deep dive is written because the topic *demands* depth — not as a way to expand the site

Content that is prescriptive belongs in Best Practices. Content that is procedural belongs in Tutorials. Content that is analytical and comparative belongs in Deep Dives.

**Placement**: top-level nav between Best Practices and Tutorials. Deep Dives is neither a reference nor a guide — it is a section a reader visits to think, not to do.

---

## Alternatives Considered

### Overload Best Practices with long-form comparisons

Rejected. Best Practices are prescriptive by contract (see ADR-0003). A 4000-word analysis of LoRaWAN vs raw LoRa is not a best practice — it is a decision support document. Mixing the two dilutes the purpose of the Best Practices section and makes it harder to navigate.

### Write excessively long tutorials

Rejected. Tutorials are procedural. Adding comparative analysis to a tutorial inflates it and frustrates readers who came to build something. A reader who wants to understand the LoRaWAN security model is not the same reader who wants firmware wiring instructions.

### External blog (separate from the site)

Rejected. A separate blog creates friction: different URL, different deployment, different search index, different navigation context. The goal is to keep high-signal content co-located with the technical reference material it references, enabling cross-linking and unified search.

### One overloaded "analysis" page per topic area

Rejected. A single "Embedded Systems Analysis" page containing every comparative discussion would be incoherent and unsearchable. Individual, focused deep dives are more discoverable and maintainable.

---

## Consequences

### Positive

- Provides a legitimate outlet for analytical depth without corrupting the purpose of existing sections
- Raises the intellectual surface area of the site — a reader can move from "how to configure a LoRa transmitter" (Best Practices) to "why LoRaWAN might be wrong for your deployment" (Deep Dives) within the same site
- Encourages higher-quality writing: the analytical mode demands more careful argumentation than a checklist
- Cross-linking between Deep Dives and Best Practices / Tutorials creates a richer content graph

### Negative / Trade-offs

- Increases nav surface area by one top-level tab
- Requires editorial discipline: Deep Dives must remain intentional and rare — if every topic gets a deep dive, the section loses its signal
- Analytical writing takes longer to produce and review than prescriptive writing

---

## Operational Rules

1. A Deep Dive is created only when the topic genuinely resists prescription or procedure
2. Each Deep Dive must cross-link to at least one Best Practice or Tutorial page
3. Deep Dives do not include step-by-step wiring instructions, setup commands, or "how to install X"
4. Style: analytical, comparative, journalistic — not tutorial, not reference
5. If a Deep Dive eventually produces a clear prescription, extract that prescription into a Best Practice page and link back

---

## Follow-ups

- [ ] Add a style guide for Deep Dives (tone, structure, code usage policy) as a meta-page in the section
- [ ] Candidate topics: MQTT vs CoAP vs HTTP for IoT, Scratch vs distroless container philosophy, ADR-driven architecture decision culture, LoRaWAN vs Zigbee vs Thread for smart home
- [ ] Evaluate whether Deep Dives warrant their own RSS feed or tag filtering
