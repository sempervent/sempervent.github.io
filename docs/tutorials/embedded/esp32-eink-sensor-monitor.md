---
tags:
  - embedded
  - esp32
---

# ESP32 E-Ink Environmental Monitor (Low Power + Safe Design)

**Objective**: Build a battery-powered environmental sensor node that reads temperature, humidity, pressure, and light every 5 minutes, renders a clean dashboard on a 1.54" e-ink display, then deep-sleeps. Target: >7 days on a 2000 mAh LiPo.

---

## 1. What You're Building

```
  ┌─────────────────────────────────────────────────────┐
  │ LiPo 3.7V → Charger (TP4056) → Buck 3.3V           │
  │                     │                               │
  │              ┌──────┴──────────────────┐            │
  │              │        ESP32            │            │
  │              │                         │            │
  │  BME280 ──I2C┤ GPIO21/22               │            │
  │  BH1750 ──I2C┤ (shared bus)            │            │
  │              │                         │            │
  │              │ GPIO5/18/23/17/16/4 ────┤── E-Ink    │
  │              │                         │   1.54"    │
  │              └─────────────────────────┘            │
  │                                                     │
  │  Display cycle: read → render → hibernate → sleep   │
  └─────────────────────────────────────────────────────┘
```

**Sensors**:
- BME280 — temperature, humidity, barometric pressure (I2C `0x76`)
- BH1750 — ambient light in lux (I2C `0x23`)

**Display**: Waveshare 1.54" e-ink, 200×200, black/white (or B/W/Red if available)

**Sleep cycle**: 5 minutes (configurable via `config.h`)

---

## 2. Hardware Assembly

### Bill of Materials

| Component | Notes |
|---|---|
| ESP32 DevKit | WROOM-32 or WROVER; use a module with exposed `EN` and `GPIO0` |
| BME280 breakout | 3.3 V version (check the onboard regulator) |
| BH1750 breakout | Standard I2C breakout |
| Waveshare 1.54" e-ink | Model: 1.54inch e-Paper Module |
| TP4056 LiPo charger | USB-C variant recommended |
| Buck converter | MP1584 or similar; set to 3.3 V output |
| 2000 mAh LiPo | Single cell; with protection circuit |
| 2× 4.7 kΩ resistors | I2C pull-ups |
| 1× 100 µF capacitor | Bulk decoupling on 3.3 V rail |
| JST-PH 2mm connectors | Battery and sensor connections |

### Wiring Summary

```
3.3V rail ──┬── BME280 VCC
            ├── BH1750 VCC
            ├── E-Ink VCC
            └── ESP32 3V3

GND ────────┬── BME280 GND
            ├── BH1750 GND
            ├── E-Ink GND
            └── ESP32 GND

I2C bus (4.7kΩ pull-ups to 3.3V):
GPIO21 SDA ──── BME280 SDA + BH1750 SDA
GPIO22 SCL ──── BME280 SCL + BH1750 SCL

SPI E-Ink:
GPIO18 (SCK)  ──► E-Ink CLK
GPIO23 (MOSI) ──► E-Ink DIN
GPIO 5        ──► E-Ink CS
GPIO17        ──► E-Ink DC
GPIO16        ──► E-Ink RST
GPIO 4        ◄── E-Ink BUSY
```

---

## 3. Project Layout

```
esp32-eink-monitor/
├── platformio.ini
├── config.h
├── src/
│   ├── main.cpp
│   ├── sensors.cpp / sensors.h
│   ├── display.cpp / display.h
│   └── power.cpp   / power.h
└── lib/            # vendor libraries if not in PlatformIO registry
```

---

## 4. Configuration

```cpp
// config.h
#pragma once

// Pins — E-Ink
#define EINK_SCK   18
#define EINK_MOSI  23
#define EINK_CS     5
#define EINK_DC    17
#define EINK_RST   16
#define EINK_BUSY   4

// Pins — I2C
#define I2C_SDA    21
#define I2C_SCL    22

// Sleep
#define SLEEP_SECONDS 300UL   // 5 minutes

// Battery ADC (voltage divider: 100k/100k, GPIO35 = ADC1_7)
#define BAT_ADC_PIN 35
#define BAT_LOW_V   3.2f      // enter hibernation below this voltage
```

---

## 5. PlatformIO Config

```ini
; platformio.ini
[env:esp32dev]
platform  = espressif32
board     = esp32dev
framework = arduino
monitor_speed = 115200

lib_deps =
    adafruit/Adafruit BME280 Library
    claws/BH1750
    ZinggJM/GxEPD2
```

---

## 6. Sensor Module

```cpp
// sensors.h
#pragma once
struct Readings {
    float tempC;
    float humidity;
    float pressurePa;
    uint32_t lux;
    bool valid;
};

Readings readSensors();
```

```cpp
// sensors.cpp
#include <Wire.h>
#include <Adafruit_BME280.h>
#include <BH1750.h>
#include "sensors.h"

static Adafruit_BME280 bme;
static BH1750 lightMeter;

// Simple exponential moving average for smoothing
static float ema(float prev, float next, float alpha = 0.3f) {
    return (prev == 0.0f) ? next : alpha * next + (1 - alpha) * prev;
}

// RTC-persisted smoothed values survive deep sleep
RTC_DATA_ATTR float smoothTemp  = 0.0f;
RTC_DATA_ATTR float smoothHum   = 0.0f;
RTC_DATA_ATTR float smoothPa    = 0.0f;

Readings readSensors() {
    Wire.begin(I2C_SDA, I2C_SCL, 400000);

    Readings r{};
    if (!bme.begin(0x76)) {
        Serial.println("[SENSOR] BME280 not found");
        r.valid = false;
        return r;
    }
    if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
        Serial.println("[SENSOR] BH1750 not found");
    }

    r.tempC     = bme.readTemperature();
    r.humidity  = bme.readHumidity();
    r.pressurePa = bme.readPressure();
    r.lux       = lightMeter.readLightLevel();

    // Smooth — persist across deep sleep cycles
    smoothTemp = ema(smoothTemp, r.tempC);
    smoothHum  = ema(smoothHum, r.humidity);
    smoothPa   = ema(smoothPa, r.pressurePa);

    r.tempC      = smoothTemp;
    r.humidity   = smoothHum;
    r.pressurePa = smoothPa;
    r.valid      = true;
    return r;
}
```

---

## 7. Display Module

```cpp
// display.h
#pragma once
#include "sensors.h"
void renderDisplay(const Readings& r, float battV, uint32_t bootCount);
```

```cpp
// display.cpp
#include <GxEPD2_BW.h>
#include <Fonts/FreeSansBold12pt7b.h>
#include <Fonts/FreeSans9pt7b.h>
#include "display.h"
#include "config.h"

// Adjust for your exact panel model
GxEPD2_BW<GxEPD2_154_D67, GxEPD2_154_D67::HEIGHT> display(
    GxEPD2_154_D67(EINK_CS, EINK_DC, EINK_RST, EINK_BUSY));

RTC_DATA_ATTR uint8_t partialCount = 0;

void renderDisplay(const Readings& r, float battV, uint32_t bootCount) {
    display.init(115200, true, 2, false);
    display.setRotation(1);
    display.setTextColor(GxEPD_BLACK);

    // Full refresh every 10 cycles to clear ghosting
    bool fullRefresh = (partialCount == 0);
    if (fullRefresh) {
        partialCount = 10;
        display.setFullWindow();
    } else {
        partialCount--;
        display.setPartialWindow(0, 0, display.width(), display.height());
    }

    display.firstPage();
    do {
        display.fillScreen(GxEPD_WHITE);

        // Title bar
        display.setFont(&FreeSansBold12pt7b);
        display.setCursor(4, 22);
        display.print("Environment");

        // Sensor values
        display.setFont(&FreeSans9pt7b);
        display.setCursor(4, 50);
        display.printf("Temp:  %.1f C", r.tempC);
        display.setCursor(4, 72);
        display.printf("Hum:   %.1f %%", r.humidity);
        display.setCursor(4, 94);
        display.printf("Press: %.0f hPa", r.pressurePa / 100.0f);
        display.setCursor(4, 116);
        display.printf("Light: %u lux", r.lux);

        // Footer: battery + boot count
        display.setFont(nullptr);
        display.setCursor(4, 190);
        display.printf("BAT %.2fV  #%lu", battV, bootCount);
    } while (display.nextPage());

    display.hibernate();  // mandatory — powers down driver
}
```

---

## 8. Power Module

```cpp
// power.cpp
#include <Arduino.h>
#include <esp_sleep.h>
#include "config.h"

RTC_DATA_ATTR uint32_t bootCount = 0;

void enterDeepSleep() {
    // Ensure all SPI peripherals are idle before sleep
    SPI.end();
    // Release GPIO drive to reduce leakage
    gpio_deep_sleep_hold_dis();

    esp_sleep_enable_timer_wakeup(SLEEP_SECONDS * 1000000ULL);
    Serial.printf("[PWR] Entering deep sleep for %lus\n", SLEEP_SECONDS);
    Serial.flush();
    esp_deep_sleep_start();
}

float readBatteryVoltage() {
    // Voltage divider: BAT+ → 100k → GPIO35 → 100k → GND
    // Full-scale: 4.2V battery → 2.1V at GPIO35 → ADC reads ~2610 (12-bit, 3.3V ref)
    int raw = analogRead(BAT_ADC_PIN);
    return (raw / 4095.0f) * 3.3f * 2.0f;
}

bool batteryLow(float v) { return v < BAT_LOW_V; }
```

---

## 9. Main Orchestration

```cpp
// main.cpp
#include <Arduino.h>
#include "config.h"
#include "sensors.h"
#include "display.h"
#include "power.cpp"  // inline for brevity; split in real project

extern RTC_DATA_ATTR uint32_t bootCount;

void setup() {
    Serial.begin(115200);
    bootCount++;

    float battV = readBatteryVoltage();
    Serial.printf("[MAIN] Boot #%lu  Bat=%.2fV\n", bootCount, battV);

    if (batteryLow(battV)) {
        Serial.println("[MAIN] Battery low — hibernating");
        esp_deep_sleep_start();  // no timer — wake only on power restore
    }

    Readings r = readSensors();
    if (!r.valid) {
        // Sensor failure: display error state, sleep, retry next cycle
        Serial.println("[MAIN] Sensor read failed");
    }

    renderDisplay(r, battV, bootCount);
    enterDeepSleep();
}

void loop() {}  // never reached — deep sleep re-runs setup()
```

---

## 10. Power Budget

| Phase | Duration | Current | Energy (µAh/cycle) |
|---|---|---|---|
| Boot + init | 200 ms | 80 mA | 4444 |
| Sensor read (I2C) | 300 ms | 20 mA | 1666 |
| E-ink update (full) | 2000 ms | 30 mA | 16666 |
| Deep sleep | 297.5 s | 25 µA | 2066 |
| **Cycle total** | **300 s** | — | **~25 mAh** |

**2000 mAh ÷ 25 mAh/cycle × (300 s ÷ 3600 s/h) ≈ ~6.7 days**

Real-world: ~5–6 days accounting for regulator efficiency and voltage sag. To extend life:
- Use partial refresh (8 of 10 cycles) → saves ~12 mAh per cycle
- Reduce sleep current: use a module without USB-UART chip (bare ESP32 module)
- Power-gate sensors with a P-channel MOSFET
- Use efficient buck (MP2307) instead of LDO

---

## 11. Battery Safety Note

!!! warning "LiPo Safety"
    - Never charge unattended or on flammable surfaces during development
    - TP4056 board must have a protection cell (boards with 2 chips) — not just the charger
    - Add a software low-voltage cutoff at 3.2 V (implemented in `batteryLow()` above)
    - A swollen or hot battery must be safely disposed of immediately

---

## 12. Future Expansion Ideas

- **CO2 sensor** (SCD40) — add to I2C bus; CO2 levels affect display layout
- **MQTT publish** — wake, read, publish to broker, sleep; one WiFi connection per cycle (~2.5 s wake time)
- **OTA updates** — check update server once per day using RTC counter
- **Soil moisture** — analog channel on GPIO35 with calibrated output
- **Second e-ink page** — partial scroll through multiple data views on each button press
- **LoRa uplink** — replace WiFi with SX1276 LoRa module; useful for remote field deployment
- **GPS** — add NEO-6M; display lat/lon + elevation outdoors

---

## 13. ESP-IDF Translation Notes

The above uses Arduino Core for accessibility. ESP-IDF equivalents:

| Arduino | ESP-IDF |
|---|---|
| `Wire.begin()` | `i2c_driver_install()` + `i2c_master_start()` |
| `analogRead()` | `adc1_get_raw()` + `esp_adc_cal_raw_to_voltage()` |
| `esp_deep_sleep_start()` | Same — this is the IDF API |
| `RTC_DATA_ATTR` | Same — this is the IDF attribute |
| `Serial.printf()` | `ESP_LOGI()` / `printf()` |
| `delay()` | `vTaskDelay(pdMS_TO_TICKS(ms))` |

---

## 14. See Also

!!! tip "See also"
    - [ESP32 Programming Architecture](../../best-practices/esp32/esp32-programming-architecture.md) — state machines, FreeRTOS, ISR rules
    - [Power Management & Deep Sleep](../../best-practices/esp32/power-management-and-deep-sleep.md) — deep sleep API, RTC memory, battery safety
    - [ESP32 Hardware & Electrical Safety](../../best-practices/esp32/esp32-hardware-and-electrical-safety.md) — wiring, GPIO limits, LiPo handling
    - [Sensor Integration Best Practices](../../best-practices/esp32/sensor-integration-best-practices.md) — I2C pull-ups, calibration, filtering
    - [E-Ink Display Best Practices](../../best-practices/esp32/e-ink-display-best-practices.md) — partial vs full refresh, hibernate, ghosting
    - [Embedded Security & OTA](../../best-practices/esp32/embedded-security-and-ota.md) — NVS credentials, OTA updates
