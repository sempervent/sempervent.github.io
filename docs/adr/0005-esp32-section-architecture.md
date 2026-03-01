# ADR 0005: Establish ESP32 & Embedded Systems as a First-Class Documentation Section

- **Status**: Accepted
- **Date**: 2026-02-26
- **Deciders**: mate (owner)
- **Technical Story**: ESP32-based projects have accumulated across the site without a coherent home. Embedded systems combine hardware, firmware, electrical safety, and security in ways that require a dedicated, structured documentation pillar — not a footnote under "just for fun."

---

## Context

The site already has stable pillars for server-side concerns (PostgreSQL, Docker, Python, Rust, Go) and creative projects (Just for Fun). But embedded systems occupy a gap:

- Hardware and software decisions are deeply coupled — you cannot explain firmware without explaining the circuit.
- Electrical safety is non-negotiable and must be part of the documentation, not an afterthought.
- ESP32 projects span best practices (architecture, power, security) *and* tutorials (capstone builds), requiring both documentation modes.
- Without a dedicated section, embedded content risks scattering across "Just for Fun" (too casual), "Docker" (wrong stack), or nowhere at all.

Specific drivers:
1. Growing number of ESP32 personal projects (sensor nodes, room controllers, art frames).
2. Need to document safety-first patterns that apply across all embedded projects, not just one tutorial.
3. Desire to distinguish *durable practices* (deep sleep design, ISR safety, power budgets) from *playful experiments* (e-ink art frames).
4. Embedded docs must cross-link hardware ↔ firmware ↔ security — a flat structure cannot represent these dependencies well.

---

## Decision

**Create `Best Practices → 🔌 Embedded Systems & ESP32`** as a first-class section at the same level as Python, Docker, and PostgreSQL, with the following structure:

| Content type | Location |
|---|---|
| Durable patterns (architecture, power, safety, security, sensors, display) | `docs/best-practices/esp32/` |
| Applied project tutorials | `docs/tutorials/embedded/` |
| Architectural rationale (this ADR) | `docs/adr/` |

**Safety as mandatory content**: every best practice document in this section must address safety implications — electrical, power, or security — as a primary concern, not a sidebar.

**Cross-linking policy**: every tutorial references the relevant best practice pages via `!!! tip "See also"` admonitions. Every best practice page links back to at least one applied tutorial.

**Naming convention**: `kebab-case.md`, `esp32-` prefix for ESP32-specific docs, `embedded/` for the tutorial subdirectory.

---

## Alternatives Considered

### Keep ESP32 under Just for Fun only

Rejected. "Just for Fun" signals low stakes and experimentation. Electrical safety documentation, power budget analysis, and OTA security do not belong under that framing. A reader looking for grounded hardware guidance would not find it.

### Generic "Hardware" catch-all section

Rejected. "Hardware" is too broad. The site has no Raspberry Pi HAT tutorials, no PCB design content, no STM32 content — a generic section would be an empty promise. ESP32 is the concrete, bounded scope that exists today.

### One giant "ESP32 reference" page

Rejected. A monolithic page cannot represent the relationship between topics (e.g., "power gating depends on both the circuit design page and the firmware architecture page"). Separate, cross-linked pages enable navigation by reader intent.

### Separate repo

Rejected. The value of co-locating embedded content with Docker, Go, and security content is cross-pollination: a reader learning about OTA updates should easily reach the general secrets management guide. Splitting the repo breaks that graph.

---

## Consequences

### Positive

- Clear separation between durable patterns (best practices) and applied projects (tutorials) — consistent with [ADR-0003](0003-why-best-practices-vs-tutorials.md)
- Safety documentation is guaranteed visible and linked from every project page
- Embedded becomes a stable, growing pillar alongside server-side content
- Cross-links between embedded tutorials and Docker/Security/Kubernetes content are natural and short
- New contributors have a clear template to follow for future embedded topics

### Negative / Trade-offs

- Increases nav surface area by ~10 entries; sidebar grows slightly longer
- Structural complexity: readers must understand that "programming architecture" is in best-practices, while "RF controller" is in tutorials
- Requires maintaining cross-links as new content is added (manageable with the See Also convention)

### Follow-ups

- [ ] Add LoRa best practices page (SX1276/SX1278, spreading factor, duty cycle limits)
- [ ] Add power electronics section (MOSFET switches, LiPo charging circuits, solar input)
- [ ] Add printable ESP32 safety checklist (HTML/PDF export from Markdown)
- [ ] Add ESP32-S3 and ESP32-C3 notes (USB native, RISC-V, BLE 5.0 differences)
- [ ] Consider an `embedded/` best-practices subdirectory if non-ESP32 MCUs are added (STM32, RP2040)

---

## Notes

- Follow the [URL preservation policy from ADR-0001](0001-site-architecture.md): if any embedded page is ever moved, add a redirect.
- Nav rule: if the embedded section exceeds 12 items, group into sub-sections (e.g., "Core", "Communication", "Power").
- Tag policy: use `embedded` and `esp32` tags consistently; add `security` or `performance` where appropriate.
- This ADR does not cover non-ESP32 microcontrollers. A future ADR should decide whether to expand the section or create a sibling section.
