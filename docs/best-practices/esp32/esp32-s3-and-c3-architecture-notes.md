---
tags:
  - embedded
  - esp32
---

# ESP32-S3 and ESP32-C3 Notes (USB Native, RISC-V, BLE 5.0)

The original ESP32 (WROOM-32 / WROVER) is still the right choice for most WiFi + Bluetooth 4.2 projects. But the S3 and C3 variants address specific gaps — native USB, RISC-V, BLE 5.0, and improved security hardware. This page covers the decision points.

---

## 1. CPU Architecture Differences

| Variant | CPU | Cores | Clock | Key addition |
|---|---|---|---|---|
| ESP32 | Xtensa LX6 | 2 | 240 MHz | Original; most mature ecosystem |
| ESP32-S3 | Xtensa LX7 | 2 | 240 MHz | Vector instructions; USB OTG; PSRAM up to 8 MB |
| ESP32-C3 | RISC-V RV32IMC | 1 | 160 MHz | Minimal BOM cost; RISC-V; BLE 5.0 |
| ESP32-C6 | RISC-V RV32IMAC | 1 | 160 MHz | Adds 802.15.4 (Zigbee/Thread); Matter support |
| ESP32-S2 | Xtensa LX7 | 1 | 240 MHz | USB OTG; no Bluetooth |

**ESP32-S3 vector extensions** (PIE — Processor Instruction Extensions): accelerate matrix operations, 8-bit integer arithmetic, and DSP kernels. Useful for:
- On-device ML inference (TinyML)
- Audio DSP (FFT, FIR filters)
- Image processing on small cameras (OV2640, OV5640)

**ESP32-C3 RISC-V**: no vendor-specific ISA extensions. Smaller, cheaper, lower power at idle. The open ISA means better toolchain diversity and auditability.

---

## 2. Native USB

### ESP32-S3: USB OTG (full-speed, 12 Mbps)

The S3 includes a USB full-speed OTG controller directly on-chip:
- **Device mode**: appears as HID (keyboard, gamepad, MIDI), CDC serial, MSC (mass storage) without any additional chip
- **Host mode**: can enumerate and communicate with USB devices (keyboards, MIDI controllers, flash drives)
- **No USB-UART chip required for flashing**: reduces BOM; programmed directly over USB CDC

```cpp
// Arduino: use TinyUSB (bundled with ESP32 Arduino core for S3)
#include <USB.h>
#include <USBHID.h>
// Or for MIDI:
#include <USBMIDI.h>
```

**Implications for flashing**:
- Hold BOOT (GPIO 0) LOW while plugging USB to enter download mode
- Some S3 devkits have a "USB" and a "UART" port — flash via the "USB" port for native USB, or the "UART" port if the board has a separate CH340/CP2102

### ESP32-C3: USB Serial/JTAG (CDC + debug)

The C3 includes a simpler USB Serial/JTAG peripheral (not full OTG):
- **Device mode only**: appears as a CDC serial port and JTAG debug interface
- Cannot act as USB host
- Eliminates the need for an external USB-UART chip for development
- JTAG debugging works directly over the USB cable — no separate J-Link or FTDI needed

---

## 3. BLE Differences

| Variant | BLE version | BLE 5 long range | BLE mesh | Bluetooth Classic |
|---|---|---|---|---|
| ESP32 | BLE 4.2 | No | Yes (GATT) | Yes (A2DP, HFP, SPP) |
| ESP32-S3 | BLE 5.0 | Yes | Yes | No |
| ESP32-C3 | BLE 5.0 | Yes | Yes | No |

**BLE 5.0 key features over 4.2**:
- **2 Mbps PHY**: doubles throughput for high-bandwidth sensor data
- **Long range (coded PHY)**: +9 dBm link budget improvement using error correction; trades throughput for range (~4x range at 125 kbps)
- **Extended advertising**: longer advertising packets (up to 255 bytes vs 31 bytes in 4.2); useful for beacon applications
- **Periodic advertising**: scheduled, non-connectable broadcasts for synchronized sensor networks

**BLE 5.0 long range in practice**: useful for outdoor sensors where WiFi doesn't reach and LoRa's duty cycle is too restrictive. A C3 + 1/4-wave antenna can reliably reach 100–200 m in open space using coded PHY.

**No Bluetooth Classic on S3/C3**: if you need A2DP audio streaming or SPP serial profiles, use the original ESP32. The S3 and C3 support BLE only.

---

## 4. When to Choose Each

| Use case | Recommended variant |
|---|---|
| General WiFi IoT (sensors, MQTT, HTTP) | ESP32 (WROOM-32) — most libraries, most forum answers |
| USB HID device (keyboard emulator, MIDI controller) | ESP32-S3 — native USB OTG device mode |
| USB host (reading MIDI controllers, USB sensors) | ESP32-S3 — only variant with USB OTG host mode |
| TinyML / on-device inference (TensorFlow Lite) | ESP32-S3 — vector extensions + PSRAM |
| Cost-sensitive production run | ESP32-C3 — smaller, cheaper, adequate for most tasks |
| BLE 5.0 long-range (outdoor sensors) | ESP32-C3 or S3 — both support coded PHY |
| Zigbee / Thread / Matter | ESP32-C6 — dedicated 802.15.4 radio |
| Bluetooth audio (A2DP streaming) | ESP32 (original) — only variant with Classic Bluetooth |
| JTAG debugging without external probe | ESP32-C3 or S3 — both include USB JTAG |

---

## 5. Security Improvements

All variants (ESP32, S3, C3) support secure boot v2 and flash encryption. There are hardware-level differences:

### Digital Signature (DS) peripheral

The ESP32-S3 and C32 include a hardware **Digital Signature peripheral** — a dedicated hardware block that performs RSA/ECC signing without exposing the private key to the CPU. The key is stored in eFuses encrypted by an HMAC-derived key.

Use cases:
- TLS client certificates signed in hardware without key extraction risk
- Firmware signing where the signing key never appears in software
- Device identity for cloud IoT platforms (AWS IoT, Azure IoT Hub)

### HMAC peripheral

The S3 and C3 include a dedicated HMAC peripheral:
- Generates MAC codes for message authentication
- Key stored in eFuses — not readable by software
- Can revoke secure boot key slots

### eFuse key slots

The S3 and C3 have additional eFuse key blocks compared to the original ESP32, enabling more granular secure boot + flash encryption key management.

---

## 6. Arduino and ESP-IDF Support Maturity

| Variant | ESP-IDF support | Arduino core support | TinyUSB | PlatformIO |
|---|---|---|---|---|
| ESP32 | v4.4+ | Excellent | Partial | Full |
| ESP32-S3 | v4.4+ | Good (v2.0.5+) | Full | Full |
| ESP32-C3 | v4.4+ | Good (v2.0.3+) | Limited | Full |

**Arduino core gotchas**:
- ESP32-S3 native USB requires `Tools → USB Mode → USB-OTG (TinyUSB)` in Arduino IDE
- Some libraries that use `Serial` directly may need updating to use `USBSerial` on S3
- On C3, `analogRead()` uses ADC1 only (no ADC2); check pin mapping before migrating projects

**PlatformIO**:
```ini
; ESP32-S3
[env:esp32s3]
platform = espressif32
board    = esp32-s3-devkitc-1
framework = arduino

; ESP32-C3
[env:esp32c3]
platform = espressif32
board    = esp32-c3-devkitm-1
framework = arduino
```

---

## 7. Migration Notes from ESP32

**Pin mapping is different**: GPIO numbers are the same on some boards but physical positions are not standardized across variants. Always check the specific DevKit pinout.

**Removed peripherals on S3**: no internal Hall sensor; no internal temperature sensor (use external BME280 or similar).

**Removed on C3**: no Ethernet MAC; no CAN; no I2S; no DAC. Verify peripherals before committing to C3 for existing designs.

**SDK version pinning**: if your project uses IDF 5.x features, verify the Arduino core version supports them. Check `esp32/platform.json` in your PlatformIO cache for the bundled IDF version.

---

## 8. See Also

!!! tip "See also"
    - [ESP32 Programming Architecture](esp32-programming-architecture.md) — state machines, FreeRTOS task layout, ISR rules (apply equally to all variants)
    - [Embedded Security & OTA](embedded-security-and-ota.md) — secure boot v2, flash encryption, DS peripheral usage
    - [Power Management & Deep Sleep](power-management-and-deep-sleep.md) — deep sleep and light sleep behavior is consistent across all variants but some API calls differ
