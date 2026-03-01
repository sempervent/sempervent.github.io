# ESP32 Safety Checklist

To print: use browser **Print** (Ctrl+P / Cmd+P) → **Save as PDF**.
Disable "Background graphics" for cleaner output. Enable "Headers and footers" to include the page URL.

This checklist covers a complete ESP32 hardware build before first power-on.

---

## Electrical Safety

**3.3 V Logic**

- [ ] All components connected to GPIO pins are verified as 3.3 V tolerant
- [ ] Any 5 V sensor outputs are routed through a level shifter before reaching ESP32 GPIO
- [ ] No GPIO pin is directly connected to a voltage source above 3.3 V
- [ ] Strapping pins (GPIO 0, 2, 12, 15) are accounted for in schematic and will be in correct state at boot

**GPIO Limits**

- [ ] No single GPIO pin is sourcing or sinking more than 12 mA
- [ ] Total combined GPIO current does not exceed 1200 mA
- [ ] LEDs driven from GPIO have a current-limiting resistor (minimum 220 ohm for 3.3 V rails)
- [ ] All unused GPIO pins set to defined state (INPUT_PULLDOWN or INPUT_PULLUP) in firmware

**Grounding**

- [ ] All components share a common ground reference
- [ ] No floating ground connections between modules
- [ ] Analog and digital grounds joined at a single point (if using ADC)

**No Mains Contact**

- [ ] No mains (120 V / 240 V AC) wiring is accessible from the ESP32 side of any relay or MOSFET
- [ ] Relay used provides galvanic isolation between ESP32 circuit and mains load
- [ ] Enclosure prevents accidental contact with any mains-carrying conductor

---

## Power Safety

**Regulator and Supply**

- [ ] Voltage regulator output verified with multimeter before connecting ESP32 (should read 3.28–3.32 V)
- [ ] Regulator is rated for at least 1.5x the expected maximum current draw
- [ ] Decoupling capacitors placed near ESP32 VCC pin (100 nF ceramic + 10 µF electrolytic)
- [ ] Decoupling capacitors placed near each sensor module VCC pin (100 nF ceramic minimum)

**LiPo Battery**

- [ ] Battery protection IC (DW01A or equivalent) is present — not just the charger IC
- [ ] Battery polarity verified before connection (LiPo connectors are not polarized by default)
- [ ] Charging current set appropriately (not exceeding C/2 for standard LiPo cells)
- [ ] Battery is not swollen, punctured, or damaged
- [ ] A fuse (500 mA to 1 A) is present on the battery positive lead

**MOSFET and Load Switching**

- [ ] Logic-level MOSFET confirmed (gate drive sufficient at 3.3 V)
- [ ] Gate resistor (100 to 470 ohm) present in series with GPIO to MOSFET gate
- [ ] Flyback diode present across any relay coil or inductive load (cathode to VCC side)
- [ ] No inductive load switching without snubber or flyback protection

---

## Firmware Safety

**Watchdog Timer**

- [ ] Hardware or software watchdog enabled for all production code
- [ ] Main loop feeds watchdog regularly (or task watchdog is subscribed)
- [ ] Expected maximum loop time is less than watchdog timeout

**Loop Discipline**

- [ ] No `delay()` calls exceeding 100 ms in the main loop (unless intentional)
- [ ] No infinite blocking loops without a timeout or watchdog
- [ ] ISR functions marked with `IRAM_ATTR`
- [ ] ISR does not call `Serial.print`, `malloc`, or I2C/SPI functions

**Input Handling**

- [ ] Button and switch inputs have debounce logic (software or hardware)
- [ ] Debounce period is at least 20 ms
- [ ] Encoder inputs use interrupt-driven reading or fast polling with Gray code validation
- [ ] All input pins have defined pull-up or pull-down state

**Memory**

- [ ] Stack size for each FreeRTOS task verified with `uxTaskGetStackHighWaterMark` during development
- [ ] `ESP.getFreeHeap()` monitored; minimum free heap does not decrease over time during extended run
- [ ] No unbounded heap allocations in main loop

---

## RF Safety

**LoRa / 433 MHz / 2.4 GHz**

- [ ] Antenna connected before powering module (operating without antenna can damage PA)
- [ ] Antenna type appropriate for frequency and application
- [ ] Duty cycle compliance verified for operating frequency band (EU 868 MHz: 1% typical)
- [ ] RF module decoupling capacitors placed near module VCC (100 nF + 100 µF)
- [ ] RF trace / antenna wire kept away from digital switching signals

**WiFi / Bluetooth**

- [ ] ADC1 used for analog readings (ADC2 conflicts with WiFi)
- [ ] WiFi TX power set appropriately for range needed (higher power not always better)
- [ ] Bluetooth and WiFi coexistence mode configured if using both simultaneously

---

## Physical Safety

**Wiring and Assembly**

- [ ] All solder joints inspected under magnification — shiny, concave, cone-shaped
- [ ] No solder bridges between adjacent pins
- [ ] No cold solder joints (dull, grainy, or uneven appearance)
- [ ] No stray wire strands that could cause shorts
- [ ] All wire connections rated for expected current

**Insulation and Strain Relief**

- [ ] No bare wire terminations exposed where they could contact other conductors
- [ ] Heat shrink or electrical tape applied to all single-conductor bare ends
- [ ] Power cables secured with zip tie or tape at entry point to PCB — no tension on solder joints
- [ ] Battery wires have strain relief at connector and at PCB

**Enclosure**

- [ ] Enclosure material is non-conductive or all conductive surfaces are grounded
- [ ] Ventilation present if enclosure will contain a charging LiPo or high-current regulator
- [ ] Enclosure provides mechanical protection against accidental short circuits
- [ ] Mounting hardware (screws, standoffs) does not contact PCB traces or components

---

## First Power-On Sequence

- [ ] Verify power supply output voltage with multimeter before connecting ESP32
- [ ] Connect ESP32 last (after all other wiring is verified)
- [ ] Monitor serial output during first boot for error messages
- [ ] Verify `ESP.getFreeHeap()` is reasonable (>100 KB for typical sketches)
- [ ] Check for unexpected heat from any component within 30 seconds of power-on
- [ ] Verify expected LED blink or serial message indicating firmware is running
- [ ] Disconnect power immediately if any component becomes hot to the touch

---

## See Also

- [ESP32 Hardware and Electrical Safety](esp32-hardware-and-electrical-safety.md)
- [Power Management and Deep Sleep](power-management-and-deep-sleep.md)
- [Power Electronics for ESP32](../../best-practices/embedded/power-electronics-for-esp32.md)
- [Sensor Integration Best Practices](sensor-integration-best-practices.md)
