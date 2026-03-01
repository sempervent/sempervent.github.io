---
tags:
  - embedded
  - esp32
---

# ESP32 Hardware & Electrical Safety

The ESP32 is resilient, but GPIO pins are not forgiving. One wrong voltage, one floating pin, one crossed ground — and you're replacing modules. These rules prevent the most common hardware mistakes.

---

## The Cardinal Rule: 3.3 V Logic Only

**The ESP32 is a 3.3 V device. GPIO pins are NOT 5 V tolerant.**

Applying 5 V to a GPIO input will damage or destroy the pin — often silently. Check every peripheral: if it outputs 5 V logic (many legacy sensors, most Arduino shields), you need a level shifter.

```
5V signal → [Level Shifter TXS0108E / BSS138 + resistors] → 3.3V GPIO
```

Common offenders:
- HC-SR04 ultrasonic sensor (5 V echo line)
- Many OLED/LCD displays with SDA/SCL pulled to 5 V
- DHT22 clones with 5 V data lines
- DS18B20 in parasitic power mode on 5 V

---

## GPIO Current Limits

| Limit | Value |
|---|---|
| Max current per GPIO pin | **12 mA** (source or sink) |
| Recommended GPIO drive | ≤ 8 mA |
| Max total chip current (all GPIOs combined) | **1200 mA** |
| Internal pull-up/pull-down | ~45 kΩ |

Driving LEDs directly from GPIO: always use a current-limiting resistor.

```
GPIO (3.3V) → [220Ω resistor] → LED anode → LED cathode → GND
Current = (3.3V - 2.0V) / 220Ω ≈ 5.9 mA  ✓ safe
```

For multiple LEDs or any inductive load (relay, motor), use a transistor or driver IC. The ESP32 is not a power delivery device.

---

## Pull-Up and Pull-Down Best Practices

| Scenario | Recommendation |
|---|---|
| I2C SDA/SCL | External 4.7 kΩ to 3.3 V (not 5 V) |
| Button input | External 10 kΩ to GND + `INPUT_PULLUP` in firmware |
| UART RX at idle | Pull high with 10 kΩ to prevent noise |
| Floating analog input | Connect to GND through resistor or use `INPUT_PULLDOWN` |

Internal pull-ups (~45 kΩ) are fine for buttons. They are too weak for I2C — use external 4.7 kΩ or 3.3 kΩ resistors.

**Floating pins** — a GPIO with no connection and no pull will float at random voltages. This causes:
- Phantom button presses
- Unpredictable ADC readings
- Interrupt storms

Set all unused GPIOs to a defined state in firmware:
```cpp
pinMode(UNUSED_PIN, INPUT_PULLDOWN);
```

---

## Strapping Pins

Certain GPIO pins control boot mode and must be in a specific state at power-on:

| Pin | Boot behavior |
|---|---|
| GPIO 0 | LOW = download mode; HIGH = normal boot |
| GPIO 2 | Must be LOW or floating during download |
| GPIO 15 | HIGH = SDIO output; LOW = silent boot |
| GPIO 12 | Controls flash voltage — leave floating for 3.3 V flash |

Avoid using GPIO 0, 2, 12, and 15 for peripheral control unless you fully understand the boot constraints. GPIO 34–39 are **input-only** — they have no internal pull-up/down and cannot source current.

---

## Grounding

Ground is not optional. Ground is a design decision.

- **Single ground reference**: all components share the same GND net
- **Star topology**: connect GND from each power domain to a single central point (star), not daisy-chained
- **Analog and digital grounds**: on sensitive ADC designs, keep analog GND and digital GND separate and join at one point near the power entry
- **Short ground traces / wires**: minimize inductance in ground return paths for high-frequency signals
- **Decoupling capacitors**: 100 nF ceramic + 10 µF electrolytic near each power pin of each IC

On breadboards, keep ground rows continuous. Never assume solder-in jumpers are making contact.

---

## ESD (Electrostatic Discharge) Awareness

The ESP32's GPIO pins have minimal ESD protection. In a dry environment, you can destroy a GPIO by touching an antenna wire with a finger.

- Handle the ESP32 module by its edges
- Use an ESD wrist strap when working with production hardware
- On boards exposed to users, add TVS diodes (PRTR5V0U2X or similar) on exposed GPIO lines
- Store modules in anti-static bags when not in use
- Connect wrist or mat ground before plugging in a module

---

## Short Circuit Protection

| Component | Protection provided |
|---|---|
| Polyfuse (PTC) on power rail | Resets after overcurrent — good for USB-powered projects |
| Blade fuse (250 mA–1 A) | Battery-powered projects; must be replaced after trip |
| DW01A + FS8205A | Combined over/under-voltage + short-circuit for LiPo cells |
| Schottky diode on V_IN | Reverse polarity protection (1N5819, very low forward drop) |

Always put a fuse between the battery and the circuit. 500 mA is appropriate for most ESP32 IoT nodes.

---

## LiPo Battery Safety

- **Voltage limits**: 4.2 V charged, 3.0 V minimum discharge
- **Never reverse polarity**: LiPo connectors aren't keyed by default
- **Never puncture, bend, compress, or heat** a LiPo cell
- **Charge only with dedicated charger IC**: TP4056 for single cell, MCP73831 for low-current
- **Do not charge below 0 °C or above 45 °C**
- **Swollen battery = immediate disposal** in a certified collection point
- **Store at ~50% charge** if unused for extended periods
- **Fire-safe charging**: charge in a LiPo-safe bag or on a non-flammable surface

---

## Breadboard vs PCB

| Consideration | Breadboard | PCB |
|---|---|---|
| Speed | Fast for prototyping | Slow to iterate |
| Reliability | Poor (intermittent contacts) | Excellent |
| High-frequency signals | Unusable above ~5 MHz | Fine with good layout |
| Decoupling caps | Often omitted | Can be placed optimally |
| Production | Never | Required |

Use breadboards for concept validation only. Any project you run for more than a few days should be on perfboard or a custom PCB. Intermittent breadboard contacts are the #1 source of "ghost bugs."

---

## Safe Soldering Practices

- Temperature: 320–360 °C for lead-free solder (SAC305); 280–320 °C for leaded
- Use flux: solder wets better, reduces cold joints
- No more than 3–4 seconds of iron contact per joint — prolonged heat damages pads and components
- Inspect joints under magnification: a good joint is shiny, concave, and cone-shaped
- De-solder wick or solder sucker for rework — avoid re-heating the same pad more than 2–3 times
- Wash flux residue with IPA if using no-clean flux on high-impedance analog circuits

---

## See Also

!!! tip "See also"
    - [Power Management & Deep Sleep](power-management-and-deep-sleep.md) — battery circuit design and power gating
    - [Sensor Integration Best Practices](sensor-integration-best-practices.md) — I2C/SPI wiring details
    - [ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md) — hardware assembly in practice
