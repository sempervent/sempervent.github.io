---
tags:
  - embedded
  - esp32
---

# ESP32 Power Management & Deep Sleep

Power is the hidden constraint of every battery-operated project. A device that drains a 2000 mAh LiPo in 6 hours instead of 6 months is a design failure. These patterns get you to <100 µA sleep current.

---

## Power Architecture Overview

```
  [LiPo / 18650]
       │
       ├── Charger IC (TP4056 or MCP73831)
       │
       └── Protection IC (DW01A / FS8205A — short, overcurrent, overdischarge)
             │
             └── Buck or LDO regulator → 3.3 V rail
                   │
                   ├── ESP32 (3.3 V)
                   ├── Sensor VCC (3.3 V or 1.8 V — check datasheet)
                   └── E-Ink VCC (3.3 V)
```

**LDO** (e.g., AMS1117-3.3): simple, cheap, dissipates excess voltage as heat. Fine for USB-powered projects or short battery life (<1 week).

**Buck converter** (e.g., MP2307, TPS563201): 85–95% efficient. Use for any project aiming at weeks or months on battery.

---

## ESP32 Sleep Modes

| Mode | CPU | Peripherals | RTC | Typical current |
|---|---|---|---|---|
| Active (WiFi TX) | ON | ON | ON | 240–320 mA |
| Active (no WiFi) | ON | ON | ON | 30–80 mA |
| Light sleep | Paused | Paused | ON | 0.8–1.0 mA |
| Deep sleep | OFF | OFF | ON | 10–150 µA |
| Hibernation | OFF | OFF | Minimal | ~5 µA |

For a read–render–sleep device, **deep sleep** is the right choice. WiFi-free designs can hit 10–30 µA.

---

## Deep Sleep API

```cpp
#include <esp_sleep.h>

// Timer wake: wake after N microseconds
esp_sleep_enable_timer_wakeup(300ULL * 1000000ULL); // 5 minutes

// GPIO wake (level trigger on GPIO 33, active LOW)
esp_sleep_enable_ext0_wakeup(GPIO_NUM_33, 0);

// Multiple GPIO wake (any of the listed pins, active LOW)
esp_sleep_enable_ext1_wakeup(
    (1ULL << GPIO_NUM_33) | (1ULL << GPIO_NUM_34),
    ESP_EXT1_WAKEUP_ANY_LOW
);

// Enter deep sleep — execution does NOT return here
esp_deep_sleep_start();
```

After waking from deep sleep, the ESP32 re-runs `setup()` (Arduino) or `app_main()` (IDF). Treat every boot as cold start unless you check `esp_sleep_get_wakeup_cause()`.

---

## RTC Memory — Surviving Deep Sleep

Normal RAM is wiped on deep sleep. Use RTC-tagged variables to persist state:

```cpp
RTC_DATA_ATTR uint32_t bootCount   = 0;
RTC_DATA_ATTR float    lastTempC   = 0.0f;
RTC_DATA_ATTR uint8_t  errorFlags  = 0;

void setup() {
    bootCount++;
    auto wakeReason = esp_sleep_get_wakeup_cause();
    if (wakeReason == ESP_SLEEP_WAKEUP_TIMER) {
        // Normal scheduled wake — skip re-init
    }
}
```

RTC memory is 8 KB total on ESP32. Don't stuff large buffers there.

---

## Peripheral Power Gating

Power-gate sensors and peripherals that draw current during sleep:

```cpp
#define SENSOR_POWER_PIN 26  // N-channel MOSFET gate or PNP base

void powerSensorsOn() {
    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(20); // stabilization
}

void powerSensorsOff() {
    digitalWrite(SENSOR_POWER_PIN, LOW);
}
```

Use a P-channel MOSFET (e.g., AO3407) on the high side for clean power switching. Check that `pinMode` on unused GPIOs is `INPUT` before deep sleep — floating driven GPIOs waste current.

---

## Battery Safety Checklist

- [ ] **Never charge LiPo above 4.2 V** — use a dedicated charger IC, not a raw voltage source
- [ ] **Never discharge LiPo below 3.0 V** — implement software brownout cutoff at 3.3 V
- [ ] **Over-discharge protection IC is mandatory** — don't rely on software alone
- [ ] **Use a fuse or polyfuse** on the battery positive terminal (250 mA–1 A for typical IoT loads)
- [ ] **Charge in a fire-safe location** during development — LiPo thermal runaway is real
- [ ] **Short-circuit protection** — DW01A + FS8205A combo is standard for single-cell packs
- [ ] **No puncture, no bending, no compression** of Li cells
- [ ] **Temperature monitoring** — halt charging if cell > 45 °C
- [ ] **Labeled polarity** — LiPo connectors are not polarized by default; use JST-PH with correct orientation

---

## Measuring Real Current Draw

Theoretical estimates are usually wrong. Measure:

1. **μCurrent Gold / Nordic PPK2** — purpose-built for embedded current measurement (nA resolution)
2. **Multimeter in series** — works for mA range; too slow for microsecond peaks
3. **Current-sense resistor + oscilloscope** — 0.1 Ω shunt on battery negative; voltage = current × R

Profile all phases:
- Wake → init: peak current duration
- Sensor read: I²C/SPI active current
- Render: e-ink update peak (can be 20–30 mA for 1–2 seconds)
- Sleep: steady-state idle

**Budget example** for 5-minute update cycle on 2000 mAh cell:

| Phase | Duration | Current | Energy (µAh) |
|---|---|---|---|
| Wake + sensors | 500 ms | 40 mA | 5555 |
| E-ink update | 2 s | 25 mA | 13888 |
| Deep sleep | 297 s | 20 µA | 1650 |
| **Total per cycle** | 300 s | — | **~21 mAh** |

2000 mAh ÷ 21 mAh/cycle × (300 s / 3600) ≈ **~8 days** battery life. Real-world: ~6 days with regulator and protection losses.

---

## Brownout Detection

```cpp
// Reduce brownout voltage threshold (default ~2.45V is conservative)
// Useful if LDO dropout is near 3.3V rail
WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // disable (risky, for debugging only)

// Better: implement software low-battery detection
#include <driver/adc.h>
float readBatteryVoltage() {
    // Voltage divider: BAT+ → 100k → GPIO35 → 100k → GND
    // Full scale 3.3 V ADC = 4.2 V battery
    int raw = analogRead(35);
    return (raw / 4095.0f) * 3.3f * 2.0f; // 2.0 = divider ratio
}
```

Halt and enter hibernation when battery < 3.2 V to prevent over-discharge.

---

## See Also

!!! tip "See also"
    - [ESP32 Hardware & Electrical Safety](esp32-hardware-and-electrical-safety.md) — power circuits and GPIO safety
    - [ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md) — deep sleep applied in a full project
    - [ESP32 Programming Architecture](esp32-programming-architecture.md) — how to structure the wake/sleep state machine
