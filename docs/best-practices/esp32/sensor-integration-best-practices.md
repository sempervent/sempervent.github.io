---
tags:
  - embedded
  - esp32
---

# Sensor Integration Best Practices

Sensors lie. They drift, they glitch, they pick up interference, and they have specs that only hold at 25 °C with a stable 3.3 V supply. These practices help you get useful data anyway.

---

## Interface Selection

| Interface | Typical sensors | Speed | Wiring complexity | Notes |
|---|---|---|---|---|
| I2C | BME280, BH1750, SHT31, OLED | 100–400 kHz | 2 wires + power | Multiple sensors share one bus; address conflicts are common |
| SPI | E-ink, SD cards, high-speed ADCs | 1–80 MHz | 4 wires + CS per device | Faster; more wires; no address conflicts |
| UART | GPS, CO2 sensors (MH-Z19) | 9600–115200 baud | 2 wires | Async; needs buffering |
| 1-Wire | DS18B20 temperature | Low | 1 wire + power | Up to ~100 devices per pin with addressing |
| Analog (ADC) | Soil moisture, LDR, potentiometers | — | 1 wire | See ADC caveats below |
| Digital GPIO | PIR, limit switches, encoders | — | 1 wire | Needs debounce |

**Prefer I2C or SPI** for new designs. Analog is messy.

---

## I2C Wiring Best Practices

```
ESP32          Sensor
─────          ──────
GPIO21 (SDA) ──┬── SDA
GPIO22 (SCL) ──┼── SCL
               │
            [4.7 kΩ to 3.3 V]   ← required external pull-ups
               │
             3.3 V rail
```

- **Always use external 4.7 kΩ pull-ups** to 3.3 V (not 5 V). Internal ~45 kΩ pull-ups are too weak for reliable I2C.
- **Bus capacitance limit**: keep total trace/wire length < 1 m for 400 kHz operation. Longer runs → lower speed.
- **Address conflicts**: many cheap sensors ship with fixed I2C addresses. When two sensors share an address, use an I2C multiplexer (TCA9548A) or separate I2C buses.
- **Scan at startup** to verify all sensors are present:

```cpp
void i2cScan() {
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.printf("Found device at 0x%02X\n", addr);
        }
    }
}
```

---

## ESP32 ADC Caveats

The ESP32's onboard ADC is notoriously inaccurate:
- **Non-linear** — especially below 150 mV and above 3.0 V (on a 3.3 V supply)
- **Reference drift** — varies with temperature and supply voltage
- **ADC2 conflicts with WiFi** — ADC2 cannot be used while WiFi is active; use ADC1 pins (GPIO32–GPIO39) only
- **Effective resolution**: ~11 bits usable despite 12-bit register

For precision analog measurements:
- Use an external ADC (ADS1115 over I2C — 16-bit, differential, true reference)
- Or use the internal ADC with calibration compensation:

```cpp
#include <esp_adc_cal.h>
esp_adc_cal_characteristics_t adcCal;
esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,
                         ADC_WIDTH_BIT_12, 1100, &adcCal);
uint32_t voltage_mV = esp_adc_cal_raw_to_voltage(analogRead(A0), &adcCal);
```

---

## Calibration Strategy

Every sensor needs calibration. "Factory calibration" from a $2 module is optimistic.

**Two-point calibration** (most common):

```cpp
// Map raw reading to physical units using two known reference points
float calibrate(int raw, int rawLow, int rawHigh, float physLow, float physHigh) {
    return physLow + (float)(raw - rawLow) / (rawHigh - rawLow) * (physHigh - physLow);
}

// Example: soil moisture sensor
// rawDry=3200, rawWet=1200, physDry=0%, physWet=100%
float moisture = calibrate(analogRead(SOIL_PIN), 3200, 1200, 0.0f, 100.0f);
```

Store calibration constants in NVS so they survive firmware updates.

---

## Debouncing Digital Inputs

Mechanical switches bounce — they register multiple edges for a single press.

**Software debounce (simple)**:
```cpp
const uint32_t DEBOUNCE_MS = 50;
bool lastState = HIGH;
uint32_t lastChangeMs = 0;

void checkButton() {
    bool state = digitalRead(BTN_PIN);
    if (state != lastState && (millis() - lastChangeMs) > DEBOUNCE_MS) {
        lastState = state;
        lastChangeMs = millis();
        if (state == LOW) onButtonPressed();
    }
}
```

For interrupt-driven debounce: set a timer in the ISR; process the stable state when the timer fires.

---

## Filtering Noisy Data

Raw sensor readings fluctuate. Filtering smooths the signal without adding significant latency.

**Simple moving average** (fixed window, low lag):
```cpp
template<typename T, size_t N>
class MovingAverage {
    T buf[N] = {};
    size_t idx = 0;
    T sum = 0;
public:
    T update(T val) {
        sum -= buf[idx];
        buf[idx] = val;
        sum += val;
        idx = (idx + 1) % N;
        return sum / N;
    }
};

MovingAverage<float, 8> tempFilter;
float smoothTemp = tempFilter.update(rawTemp);
```

**Exponential moving average** (lower memory, adjustable decay):
```cpp
class EMA {
    float alpha;  // 0 < alpha < 1; higher = faster response
    float value;
    bool first = true;
public:
    explicit EMA(float alpha) : alpha(alpha), value(0) {}
    float update(float x) {
        if (first) { value = x; first = false; }
        else value = alpha * x + (1 - alpha) * value;
        return value;
    }
};

EMA humidityFilter(0.2f);  // alpha=0.2 → smooth, slow to react
float smoothHum = humidityFilter.update(rawHum);
```

Use EMA for slow-changing values (temperature, humidity, pressure). Use a sliding window for sensors with occasional spikes (light, soil moisture).

---

## Shielded Wiring and Noise

For long sensor cable runs or electrically noisy environments (near motors, switching supplies):

- Use **shielded twisted pair (STP)** cable; connect shield to GND at one end only
- Add **series resistors** (100–300 Ω) on I2C/SPI lines to dampen reflections
- Keep analog signal wires away from power wires and PWM lines
- Add **decoupling capacitors** (100 nF) on sensor VCC pins, close to the sensor
- **Ferrite beads** on power lines if switching noise is an issue

---

## See Also

!!! tip "See also"
    - [ESP32 Hardware & Electrical Safety](esp32-hardware-and-electrical-safety.md) — wiring rules and GPIO limits
    - [E-Ink Display Best Practices](e-ink-display-best-practices.md) — SPI peripheral integration
    - [ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md) — BME280 and BH1750 integration examples
