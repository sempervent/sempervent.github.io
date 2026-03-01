---
tags:
  - embedded
  - esp32
---

# 🔌 Embedded Systems Tutorials

Hands-on guides for building real hardware projects with microcontrollers. These tutorials pair with the [ESP32 Best Practices](../../best-practices/esp32/index.md) section.

## ESP32

- **[ESP32 E-Ink Environmental Monitor](esp32-eink-sensor-monitor.md)** — BME280 + BH1750 sensors → e-ink display → deep sleep loop on battery power. Clean firmware architecture, power budget calculation, safety notes.
- **[ESP32 RF Room Light Controller](esp32-rf-room-light-controller.md)** — 16×2 LCD + rotary encoder + button + 433 MHz RF transmitter. State machine firmware, ISR-driven encoder, debounced buttons, flicker-free LCD updates. Always-on wall controller design.
- **[ESP32 + MQTT + Home Assistant Integration](esp32-mqtt-home-assistant-integration.md)** — BME280 sensor node publishes to a TLS-secured Mosquitto broker; HA auto-discovers the device via MQTT discovery. Includes ACL config, CA cert pinning in firmware, LWT, and a full security hardening checklist.
