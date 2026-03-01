---
tags:
  - embedded
  - esp32
---

# 🔌 Embedded Systems & ESP32

**Objective**: Build firmware that is safe, power-efficient, and maintainable. These guides cover the real decisions you make when designing embedded systems around the ESP32 — not just "how to blink an LED" but how to architect, power, secure, and ship firmware responsibly.

## Guides in this section

- **[Programming Architecture](esp32-programming-architecture.md)** — Event loops, FreeRTOS tasks, state machines, ISR safety, and memory discipline
- **[Power Management & Deep Sleep](power-management-and-deep-sleep.md)** — Sleep modes, RTC memory, wake sources, LiPo safety, and designing for sub-100 µA idle
- **[Hardware & Electrical Safety](esp32-hardware-and-electrical-safety.md)** — 3.3 V logic, GPIO limits, level shifting, grounding, ESD, LiPo dos and don'ts
- **[Embedded Security & OTA](embedded-security-and-ota.md)** — NVS secrets, secure boot, flash encryption, OTA signing, WiFi credential hygiene
- **[Sensor Integration](sensor-integration-best-practices.md)** — I2C vs SPI vs analog, pull-ups, filtering, calibration, debouncing
- **[E-Ink Display Integration](e-ink-display-best-practices.md)** — Partial vs full refresh, ghosting, SPI wiring, frame buffer design, over-refresh protection
- **[MQTT Security](mqtt-security-best-practices.md)** — Broker hardening, ACLs, TLS, per-device credentials, topic design, NVS credential storage

## Applied Tutorial

- **[ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md)** — Capstone project: BME280 + BH1750 → e-ink display → deep sleep loop on battery power
- **[ESP32 + MQTT + Home Assistant Integration](../../tutorials/embedded/esp32-mqtt-home-assistant-integration.md)** — Secure sensor node with TLS MQTT, ACL, LWT, and HA auto-discovery
