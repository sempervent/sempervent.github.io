# ADR 0006: Expand Embedded Section — LoRa, Power Electronics, and Variant Notes

- **Status**: Accepted
- **Date**: 2026-02-26
- **Deciders**: mate (owner)
- **Supersedes**: N/A — extends [ADR-0005](0005-esp32-section-architecture.md)

---

## Context

[ADR-0005](0005-esp32-section-architecture.md) established the `best-practices/esp32/` section and committed to follow-ups: LoRa best practices, power electronics, a printable safety checklist, and ESP32-S3/C3 notes.

Implementation revealed a structural decision: **Power Electronics** does not fit cleanly under `esp32/`. The content (MOSFET switching, LiPo charging circuits, solar, buck vs LDO) applies to any embedded microcontroller — not only ESP32. It belongs at a higher level of abstraction.

This ADR formalizes:
1. The creation of `best-practices/embedded/` as a cross-cutting hardware folder
2. The placement of LoRa, printable checklist, and S3/C3 notes under `esp32/`
3. The nav grouping for `🔋 Power Electronics & Embedded Hardware`

---

## Decision

**Create `docs/best-practices/embedded/`** as a sibling to `best-practices/esp32/`, `best-practices/docker-infrastructure/`, etc.

| Content | Location | Rationale |
|---|---|---|
| Power Electronics (MOSFETs, LiPo, solar, buck/LDO) | `best-practices/embedded/` | Platform-agnostic hardware concerns |
| LoRa best practices (SX127x) | `best-practices/esp32/` | ESP32-centric driver and firmware guidance |
| Printable safety checklist | `best-practices/esp32/` | ESP32-specific checklist; referenceable from hardware doc |
| ESP32-S3/C3 notes | `best-practices/esp32/` | Variant comparison for ESP32 ecosystem |

**Nav grouping**: `🔋 Power Electronics & Embedded Hardware` appears in mkdocs.yml alongside `🔌 Embedded Systems & ESP32` under Best Practices.

**Cross-linking policy**: the `embedded/` folder links back to `esp32/` for firmware context; `esp32/` pages link to `embedded/` for hardware context.

---

## Alternatives Considered

### Put power electronics under `esp32/`

Rejected. A page titled "Power Electronics for ESP32" is still useful to Raspberry Pi users, STM32 users, and Arduino users. Placing it under `esp32/` would mislead the IA and make it harder to find for non-ESP32 readers. The content's value is platform-agnostic.

### Create a `hardware/` folder instead of `embedded/`

Rejected. "Hardware" is a broader signal than intended. "Embedded" is more accurate — it connotes electronics that host a microcontroller, which is precisely the scope.

### Single `embedded/` folder replacing `esp32/`

Rejected. The `esp32/` folder contains firmware-level guidance (FreeRTOS, ISRs, deep sleep API) that is ESP32-specific. Merging would dilute the precision of that content.

### No new folder — just add power electronics to `esp32/`

Rejected — see first alternative.

---

## Consequences

### Positive

- Power electronics guidance is reachable by readers working on any embedded platform
- Clear separation: `esp32/` = ESP32 firmware + peripherals; `embedded/` = universal hardware concerns
- Sets a precedent for future MCU-agnostic content (RP2040, STM32, AVR) to have a home without needing to dilute the ESP32 section
- Printable safety checklist is directly reachable from the nav and from hardware build references

### Negative / Trade-offs

- Nav surface area increases by one section and two entries
- A reader unfamiliar with the split may look in the wrong section first — mitigated by cross-links

### Follow-ups

- [ ] Add RP2040 / Raspberry Pi Pico notes to `embedded/` if Pico projects are documented
- [ ] Add STM32 power notes if STM32 tutorials are added
- [ ] Consider `embedded/antenna-and-rf-layout.md` for shared RF layout guidance across LoRa, WiFi, BLE
- [ ] Evaluate whether the printable checklist would benefit from MkDocs print CSS customization
