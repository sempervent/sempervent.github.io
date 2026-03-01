---
tags:
  - embedded
  - esp32
---

# LoRa Best Practices (SX1276 / SX1278 + ESP32)

LoRa is a physical-layer radio modulation that enables long-range, low-power wireless links. It is not a network protocol — it is a radio waveform. Understanding this distinction prevents most of the design mistakes that make LoRa deployments unreliable or legally non-compliant.

---

## 1. LoRa vs LoRaWAN

| | LoRa | LoRaWAN |
|---|---|---|
| What it is | Radio modulation (PHY layer) | Network stack on top of LoRa |
| Topology | Point-to-point or star (you define) | Gateway → Network Server → App Server |
| Encryption | None by default | AES-128 built in |
| Suitable for | Two ESP32s talking to each other | Connecting to The Things Network |
| Complexity | Low | Higher |

**This document covers raw LoRa** — direct SX127x chip communication between two or more nodes, without a LoRaWAN network server.

**Duty cycle applies to both**: LoRa and LoRaWAN transmissions occupy shared ISM spectrum. Regulatory duty cycle limits apply regardless of whether you use a network stack.

---

## 2. Radio Fundamentals

### Spreading Factor (SF)

SF7 through SF12. Each step up doubles the airtime and increases sensitivity by ~2.5 dB.

| SF | Approx. range | Airtime (10 byte payload) | Receiver sensitivity |
|---|---|---|---|
| SF7 | Short–medium | ~50 ms | –123 dBm |
| SF9 | Medium | ~200 ms | –129 dBm |
| SF12 | Long | ~1.5 s | –137 dBm |

**Rule**: use the lowest SF that achieves reliable link. High SF monopolizes airtime and burns battery.

### Bandwidth

Standard options: 125 kHz, 250 kHz, 500 kHz.

- Narrower bandwidth → better sensitivity, longer airtime
- Wider bandwidth → shorter airtime, less range
- 125 kHz is the standard choice for most IoT applications

### Coding Rate (CR)

CR4/5 through CR4/8. Higher CR adds more forward error correction, increasing resilience at the cost of more airtime. CR4/5 is the most common default.

### Airtime Budget

Transmit time = (payload size + header) × SF × coding rate / bandwidth

For a 20-byte payload at SF9, 125 kHz, CR4/5: approximately 370 ms.

**At 1% duty cycle (EU 868 MHz): maximum 1 transmission per ~37 seconds** for this configuration. Design your polling interval accordingly.

---

## 3. Duty Cycle Regulations

!!! warning "Regulatory compliance"
    This is engineering guidance, not legal advice. Verify your local regulatory requirements before deploying radio equipment.

**EU 868 MHz band (LoRa's primary frequency in Europe)**:
- Sub-band 868.0–868.6 MHz: **1% duty cycle** (36 s silence per 3.6 s transmission)
- Sub-band 868.7–869.2 MHz: 0.1% duty cycle
- Sub-band 869.4–869.65 MHz: 10% duty cycle (used for downlinks)
- Devices that exceed duty cycle limits are illegal to operate and may interfere with emergency services

**US 915 MHz (FCC Part 15)**:
- Frequency hopping required — no fixed-frequency continuous transmission
- Uses FHSS across 64 uplink channels
- No explicit duty cycle percentage, but effective transmission is limited by the hopping requirement
- Most US LoRa modules implement this automatically if configured correctly

**Design for compliance**:
- Track transmit time in firmware; enforce minimum quiet periods
- For EU: calculate airtime per transmission, multiply by 100 to get minimum silence period
- Do not send heartbeats faster than duty cycle allows

```cpp
// Duty cycle enforcement (EU 1%)
static uint32_t lastTxMs = 0;
static uint32_t lastAirtimeMs = 0;  // measured airtime of last packet

bool canTransmit() {
    uint32_t silenceRequired = lastAirtimeMs * 99;  // 1% = 1 part per 100
    return (millis() - lastTxMs) >= silenceRequired;
}
```

---

## 4. Antenna and RF Layout

**Always use a proper antenna.** Operating a LoRa module without an antenna can damage the PA (power amplifier) in the SX127x. A short circuit or open circuit at the antenna port reflects power back into the chip.

| Antenna type | Gain | Best for |
|---|---|---|
| 1/4-wave whip (wire) | ~2 dBi | Short range, prototyping |
| Spring/helical | ~2 dBi | Compact enclosures |
| PCB trace antenna | ~0–2 dBi | Integrated designs |
| External directional | 5–12 dBi | Point-to-point fixed links |

**RF layout rules**:
- Keep the antenna trace and SMA connector away from digital switching signals (SPI clock, PWM, oscillators)
- Use a continuous ground plane under the RF section; do not route digital signals through it
- Avoid bending the antenna wire at sharp angles — keep it straight or in a gentle arc
- On a breadboard, keep the LoRa module as far from the ESP32 as the wires allow — breadboard is lossy RF substrate

**Common modules** (SX1276-based):
- Semtech SX1276 evaluation board
- TTGO LoRa32 (ESP32 + SX1276 + OLED on one board)
- Heltec WiFi LoRa 32
- RFM95W breakout

---

## 5. Power Strategy

LoRa TX current is high — 120–130 mA peak at +20 dBm output power. This is a significant spike relative to ESP32 idle current.

```
  LiPo 3.7V
      │
  Protection IC
      │
  Buck 3.3V ──[100µF bulk]──[100nF ceramic]── VCC rail
                                                │
                                         ESP32 + SX1276
```

**Capacitor placement**: 100 µF electrolytic + 100 nF ceramic within 5 mm of the SX1276 VCC pin. TX spikes cause voltage droops on resistive battery leads; capacitors absorb them.

**Brownout risk**: ESP32 brownout detector triggers at ~2.45 V. During LoRa TX on a weak battery with high-resistance leads, the rail can momentarily dip enough to reset the ESP32. Bulk capacitance and a low-ESR buck converter prevent this.

**TX power selection**: use the minimum TX power that achieves the required link budget. Each 3 dB reduction halves the current draw.

```cpp
// RADIOLIB example (use your library's equivalent)
radio.setOutputPower(14);  // dBm — not 20 unless range truly demands it
```

**Sleep current**: SX1276 sleep mode draws ~1 µA. Always sleep the radio between transmissions.

---

## 6. Firmware Strategy

### Non-blocking send with RadioLib

```cpp
// Async TX — does not block loop()
int state = radio.startTransmit(payload, payloadLen);
// Poll for completion:
if (radio.getIRQFlags() & RADIOLIB_SX127X_IRQ_TX_DONE_MASK) {
    radio.clearIRQFlags();
    // TX complete — schedule next send
}
```

### State machine TX scheduling

```cpp
enum class RadioState { IDLE, TX_PENDING, TX_ACTIVE, COOLDOWN };

RadioState radioState = RadioState::IDLE;
uint32_t   nextTxMs   = 0;
uint32_t   txStartMs  = 0;

void radioTick() {
    switch (radioState) {
        case RadioState::IDLE:
            if (millis() > nextTxMs && dataReady) {
                radio.startTransmit(buf, len);
                txStartMs  = millis();
                radioState = RadioState::TX_ACTIVE;
            }
            break;
        case RadioState::TX_ACTIVE:
            if (txDone()) {
                uint32_t airtime = millis() - txStartMs;
                nextTxMs   = millis() + airtime * 99;  // 1% duty cycle
                radioState = RadioState::COOLDOWN;
            }
            break;
        case RadioState::COOLDOWN:
            if (millis() > nextTxMs) radioState = RadioState::IDLE;
            break;
    }
}
```

### Retry logic

Do not retry blindly — each retry consumes additional airtime and duty cycle budget. Retry only on explicit NAK or timeout, with exponential backoff:

```cpp
uint8_t retries = 0;
uint32_t retryDelayMs = 5000;
// On failure:
if (retries < 3) {
    retries++;
    nextTxMs = millis() + retryDelayMs * retries;
}
```

---

## 7. Security Notes

Raw LoRa transmissions are plaintext. Anyone with a compatible receiver within radio range can receive and decode your packets.

**Implement application-layer AES encryption** for any sensitive payload:

```cpp
// Example using mbedTLS AES (available on ESP32)
#include <mbedtls/aes.h>

void aesEncrypt(const uint8_t* key, const uint8_t* in, uint8_t* out) {
    mbedtls_aes_context ctx;
    mbedtls_aes_init(&ctx);
    mbedtls_aes_setkey_enc(&ctx, key, 128);
    mbedtls_aes_crypt_ecb(&ctx, MBEDTLS_AES_ENCRYPT, in, out);
    mbedtls_aes_free(&ctx);
}
```

Use CBC mode with a random IV for payloads longer than one block. ECB is shown above only for brevity.

**Key management**: embed the AES key in NVS (see [Embedded Security & OTA](embedded-security-and-ota.md)). Rotate keys periodically — LoRa range means an attacker can be far away and invisible.

**Do not transmit**:
- GPS coordinates of the device if its location is sensitive
- Authentication tokens or passwords
- Device identifiers that enable tracking

---

## 8. See Also

!!! tip "See also"
    - [LoRaWAN vs Raw LoRa (Deep Dive)](../../deep-dives/lorawan-vs-raw-lora.md) — analytical comparison of protocol stacks, security models, regulatory implications, and infrastructure trade-offs
    - [Power Management & Deep Sleep](power-management-and-deep-sleep.md) — deep sleep between LoRa transmissions; battery life strategy
    - [ESP32 Hardware & Electrical Safety](esp32-hardware-and-electrical-safety.md) — power supply decoupling, GPIO current limits
    - [Power Electronics for ESP32](../../best-practices/embedded/power-electronics-for-esp32.md) — MOSFET power gating of LoRa module, LiPo circuits for remote nodes
    - [MQTT Security Best Practices](mqtt-security-best-practices.md) — if bridging LoRa packets to MQTT via a gateway ESP32
    - [Embedded Security & OTA](embedded-security-and-ota.md) — AES key storage in NVS, OTA for LoRa gateway nodes
