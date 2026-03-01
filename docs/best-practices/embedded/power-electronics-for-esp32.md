---
tags:
  - embedded
  - esp32
---

# Power Electronics for ESP32 Projects (MOSFETs, LiPo, Solar)

The ESP32's firmware can be perfect and still fail if the power supply is poorly designed. Most mystery crashes, random reboots, and "works on USB but not battery" problems are power problems. This guide covers the hardware fundamentals.

---

## 1. MOSFET Switching

MOSFETs let the ESP32 (a 3.3 V, low-current GPIO device) switch higher-current or higher-voltage loads — sensors, relays, LoRa modules, LED strips.

### Logic-level vs generic MOSFETs

**Logic-level MOSFETs** (e.g., IRLZ44N, 2N7002, AO3400, BSS138) are fully enhanced at gate voltages of 2.5–3.3 V.

**Generic power MOSFETs** (e.g., IRF540, IRF3205) require 10 V at the gate to turn on fully. They will not switch reliably from a 3.3 V GPIO. Do not use generic MOSFETs with ESP32 unless you add a gate driver.

### Low-side N-channel switching (most common)

```
  Load (sensor, LED strip, LoRa module)
     │
    VCC ──────────────────── Load+
     │
    Load─ ─────────────────► Drain (MOSFET)
                              Source ─── GND
     │
  GPIO ──[100Ω gate resistor]─► Gate
```

The N-channel MOSFET connects the load's negative terminal to GND when the gate is driven HIGH.

```cpp
#define LOAD_PIN 26
pinMode(LOAD_PIN, OUTPUT);
digitalWrite(LOAD_PIN, HIGH);  // MOSFET on → load powered
digitalWrite(LOAD_PIN, LOW);   // MOSFET off → load unpowered
```

**Gate resistor** (100–470 Ω): limits gate charge current, suppresses oscillation. Not always required, but good practice — costs one resistor, prevents potential RFI and MOSFET stress.

### High-side P-channel switching

Use a P-channel MOSFET (e.g., AO3407, IRF9540) when you need to switch the high side (VCC) of the load. This is preferable for power gating entire circuits because it keeps the load ground reference clean.

```
  VCC ─────────────────────► Source (P-MOSFET)
                              Drain ──────── Load+
                              Gate ◄── [10kΩ to VCC] + GPIO via [NPN transistor]
                              Load─ ─────── GND
```

Note: a P-channel MOSFET turns ON when the gate is LOW (driven to GND relative to source). Driving a P-MOSFET gate from a 3.3 V GPIO on a 3.3 V rail requires a small NPN transistor (2N3904, BC547) to invert the logic signal.

### Flyback diode for inductive loads

Relays, solenoids, and motors are inductive. When switched off, they generate a voltage spike that can destroy the MOSFET.

```
  Load (relay coil)
    │         │
   +│        -│
    └──[D1]──┘  ← flyback diode (1N4001 or schottky), cathode to VCC side
```

Always add a flyback diode across any inductive load. For fast switching (PWM motors), use a Schottky diode (1N5819) — faster recovery than standard rectifier diodes.

!!! warning "Never switch mains with an ESP32 directly"
    The ESP32 GPIO can drive a MOSFET or relay coil operating at low voltage. The relay's switch contacts can then control mains loads. But the ESP32 and relay coil circuit must be **galvanically isolated** from mains — use a relay with mechanical isolation or an optocoupler-relay module. Never let a mains voltage path come within reach of any ESP32 pin.

---

## 2. LiPo Charging Circuits

### TP4056 breakout boards

The TP4056 is a single-cell lithium-ion/LiPo charger IC. Cheap breakout boards are widely available.

**Two variants**:
- **Charger-only** (one IC): just the TP4056. No protection. Do not use.
- **Charger + protection** (two ICs, DW01A + FS8205A): includes overcharge, overdischarge, and short-circuit protection. Use this version.

Identify by component count: protection boards have a second smaller IC package on the reverse.

### Charging current selection

The TP4056 sets charging current via a programming resistor (PROG pin):

| R_PROG | Charge current |
|---|---|
| 10 kΩ | 130 mA |
| 5 kΩ | 250 mA |
| 2 kΩ | 580 mA |
| 1.2 kΩ | 1000 mA |

**Rule**: charging current should not exceed C/2 (half the battery's mAh rating) for standard LiPo cells. For a 500 mAh cell, max 250 mA. Faster charging generates heat and degrades the cell.

### Thermal awareness

The TP4056 operates in constant current / constant voltage (CC/CV) mode. At high charging current, the IC dissipates heat. At 1 A charge rate, the IC may reach 60–70 °C. Do not enclose without ventilation. Add a small heatsink pad if running near 1 A continuously.

### Charge indicators

- CHG (red): charging in progress
- STDBY (blue/green): charge complete

Monitor these with GPIO if your firmware needs to know battery state.

!!! warning "Never charge a LiPo unattended in a poorly ventilated enclosure"
    LiPo thermal runaway is rare but catastrophic. During development and prototyping, charge in the open, on a non-flammable surface, while present.

---

## 3. Solar Input

### Panel voltage variability

A 5 V / 1 W solar panel produces 5–6 V open-circuit and drops to 4–4.5 V under load — but only in direct sunlight. In shade or indoors, output can fall to 1–2 V. The charging circuit must tolerate this variation.

**The TP4056 minimum input voltage is 4.5 V**. In low-light conditions, a 5 V panel may not charge at all.

**Better option for solar**: use a panel rated 6–7 V with an MPPT charge controller (CN3791 or similar), which tracks the panel's maximum power point regardless of load.

### MPPT vs simple charge controller

| | Simple (TP4056 from panel) | MPPT (CN3791 / BQ25570) |
|---|---|---|
| Efficiency | 60–75% | 90–95% |
| Low-light performance | Poor | Good |
| Component complexity | Low | Medium |
| Cost | Very low | Low–medium |

For an outdoor sensor node expected to run year-round, MPPT is worth the complexity.

### Blocking diode

If you connect a solar panel directly to the battery circuit without a diode, the battery can back-feed current into the panel at night, slowly discharging it.

```
  Solar panel+ ──[Schottky diode]──► Charge controller+ ──► Battery+
```

MPPT controllers typically include an internal blocking function. Simple circuits need an explicit diode (1N5819 — low forward voltage drop).

---

## 4. Buck vs LDO

### Linear regulator (LDO)

A low-dropout linear regulator (e.g., AMS1117-3.3) converts excess voltage to heat. It is simple, cheap, and generates no switching noise.

**Heat dissipation**: P = (V_in − V_out) × I_load

For a 5 V input, 3.3 V output, 200 mA load:
P = (5 − 3.3) × 0.2 = **340 mW** → the LDO runs warm but is within spec.

At 500 mA: 850 mW → requires a heatsink.

**Use LDO when**:
- Input is close to output (< 1.5 V drop at expected current)
- Current is low (< 300 mA)
- Switching noise is unacceptable (precision analog circuits)

### Buck converter (switching regulator)

A buck converter (e.g., MP1584, TPS563201, LM2596) steps voltage down with 85–95% efficiency — excess energy is stored in an inductor, not wasted as heat.

**Use buck converter when**:
- Input voltage is significantly higher than output (battery 4.2 V → 3.3 V at high current)
- Current draw is high (> 300 mA)
- Long battery life is required

**Caveat**: buck converters generate switching noise at their frequency (typically 300 kHz – 1.5 MHz). Place input/output capacitors close to the IC. Keep the inductor away from RF sections and sensitive analog inputs.

---

## 5. Brownout and Transients

**Brownout**: a voltage dip below the ESP32's brownout threshold (~2.45 V) causes a hardware reset. Common causes:
- LoRa, WiFi, or LED TX current spikes on a weak power supply
- Long battery leads with high resistance
- Inadequate bulk capacitance

**Fix**: 100 µF electrolytic + 100 nF ceramic capacitor close to the ESP32 VCC pin. For LoRa modules: add 220 µF close to the SX127x.

**Transient suppression**: switching inductive loads (relays, motors) can generate spikes hundreds of volts above rail. Always use a flyback diode (Section 1). On long cable runs to external loads, add a TVS diode (P6KE6.8A or similar) at the connector.

**Decoupling discipline**:
- Every IC should have a 100 nF ceramic cap within 5 mm of its VCC pin
- Add one 10 µF ceramic or electrolytic per power domain
- Do not rely on bulk capacitance far from the IC — inductance in PCB traces limits effectiveness

---

## 6. Safety Checklist

- [ ] No exposed LiPo wiring — shrink-wrap all connections
- [ ] Protection IC (DW01A) present on all LiPo circuits — not just the TP4056
- [ ] Fuse on battery positive: 500 mA for small nodes, 1 A for nodes with relays or LoRa
- [ ] Strain relief on all power cables — tape or zip-tie at PCB entry point
- [ ] Flyback diode on every relay coil and motor
- [ ] Buck converter output verified with multimeter before connecting ESP32
- [ ] Charging current appropriate for battery capacity (≤ C/2)
- [ ] No charging in enclosed, unventilated enclosure
- [ ] Solar blocking diode present if using simple charge circuit

---

## 7. See Also

!!! tip "See also"
    - [ESP32 Power Management & Deep Sleep](../esp32/power-management-and-deep-sleep.md) — firmware-side power strategy; sleep modes; battery voltage monitoring
    - [ESP32 Hardware & Electrical Safety](../esp32/esp32-hardware-and-electrical-safety.md) — GPIO limits, 3.3 V logic, LiPo safety rules
    - [ESP32 Safety Checklist (Printable)](../esp32/esp32-safety-checklist-printable.md) — printable pre-build checklist covering electrical, power, and firmware safety
    - [LoRa Best Practices](../esp32/lora-best-practices-sx127x.md) — power strategy for LoRa TX spikes and battery-powered remote nodes
