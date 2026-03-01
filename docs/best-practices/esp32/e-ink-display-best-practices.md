---
tags:
  - embedded
  - esp32
---

# E-Ink Display Integration Best Practices

E-ink (e-paper) displays are the right choice for battery-powered devices that update infrequently. They retain their image with zero power. But they are slow, fragile, and easy to misuse.

---

## Why E-Ink for Battery Projects

| Property | E-Ink | OLED | TFT LCD |
|---|---|---|---|
| Power at rest | 0 µA | 1–10 mA | 5–50 mA |
| Update power | 20–40 mA peak | ~20 mA | ~50 mA |
| Update time | 1–3 s (full) | <50 ms | <50 ms |
| Sunlight readable | Excellent | Poor | Poor–fair |
| Image retention | Permanent (no power) | Fade on power-off | Off = black |
| Viewing angle | 180° | Varies | Limited |

**Best fit**: displays that update once per minute or slower. Poor fit: fast-changing data, animations, gauges.

---

## Partial vs Full Refresh

| Mode | Speed | Ghosting | Use for |
|---|---|---|---|
| Full refresh | 1–3 s | None | First boot, periodic full updates |
| Partial refresh | 100–500 ms | Moderate | Small area updates (clocks, counters) |
| Fast refresh | ~500 ms | High | If library supports; use with caution |

**Full refresh** clears the display through several flicker cycles before settling — it's slow but leaves no artifacts.

**Partial refresh** updates only the changed pixels. Faster, but ghosting (shadow of previous content) accumulates. Perform a full refresh every 5–10 partial updates to clear ghosts.

**Ghosting mitigation**:
- Track update count in RTC memory across deep sleep cycles
- Trigger full refresh when `updateCount % 10 == 0`

```cpp
RTC_DATA_ATTR uint8_t partialUpdateCount = 0;

void refreshDisplay(GxEPD2_BW& display, bool forceFullUpdate = false) {
    if (forceFullUpdate || partialUpdateCount == 0) {
        display.setFullWindow();
        display.firstPage();
        // draw...
        display.nextPage();
        partialUpdateCount = 10;  // reset counter
    } else {
        display.setPartialWindow(x, y, w, h);
        // draw partial area...
        partialUpdateCount--;
    }
}
```

---

## SPI Wiring

E-ink panels connect via SPI plus 3 control lines:

```
ESP32             E-Ink Panel
─────             ──────────────────
GPIO18 (SCK)  ──► CLK
GPIO23 (MOSI) ──► DIN / SDA
GPIO 5 (CS)   ──► CS
GPIO17 (DC)   ──► DC (Data/Command)
GPIO16 (RST)  ──► RST
GPIO 4 (BUSY) ◄── BUSY
              VCC ◄── 3.3 V only
              GND ◄── GND
```

**BUSY is critical**: the display drives BUSY LOW while processing. Never send commands while BUSY is LOW. Always poll or wait for BUSY HIGH before the next operation. Ignoring BUSY causes garbled display or display damage.

```cpp
void waitBusy() {
    while (digitalRead(EINK_BUSY_PIN) == LOW) {
        delay(1);
    }
}
```

---

## Over-Refresh and Display Longevity

E-ink panels have a rated refresh count — typically **1 million full refresh cycles** for quality panels. At 1 full refresh per 5 minutes, that's ~10 years. That's fine.

**Risks from over-refresh**:
- Driving partial refresh continuously without periodic full refresh → permanent ghosting
- Not waiting for BUSY signal → incomplete waveform sequences → pixel state corruption
- Displaying the same static content without any refresh for **years** → image sticking (though less common than LCD burn-in)

**Safe update loop**:

```cpp
void updateDisplay(GxEPD2_BW<GxEPD2_154, GxEPD2_154::HEIGHT>& display) {
    display.init(115200, true, 2, false);  // reset on init
    display.setRotation(1);
    display.setTextColor(GxEPD_BLACK);

    display.setFullWindow();
    display.firstPage();
    do {
        display.fillScreen(GxEPD_WHITE);
        drawContent(display);
    } while (display.nextPage());

    display.hibernate();  // power down display driver — essential
}
```

**`display.hibernate()` is mandatory** after every update for low-power designs. It powers down the display driver IC; without it, the driver draws 1–3 mA continuously.

---

## Frame Buffer Handling

E-ink libraries (GxEPD2, EPD_Libraries) manage an in-memory frame buffer. The buffer is sent to the panel on `nextPage()`.

- **Buffer size**: a 1.54" 200×200 1-bit display = 200×200÷8 = 5000 bytes in RAM
- **Larger panels** (4.2" 400×300) = 15,000 bytes — significant on ESP32 with 320 KB SRAM
- For very large displays, use **paged drawing** (`setFullWindow` + `firstPage`/`nextPage` loop) instead of full-frame buffer

**Paged drawing pattern** (GxEPD2):

```cpp
display.firstPage();
do {
    // This callback is called once per page (horizontal band)
    // Only draw what's in the current window — use display.getCursorY() to determine page
    display.fillScreen(GxEPD_WHITE);
    display.setCursor(10, 20);
    display.print("Hello");
} while (display.nextPage());
```

---

## Sleep and Wake

E-ink displays draw current during refresh only. After refreshing:

1. Call `display.hibernate()` — powers down the driver IC
2. De-assert CS, DC, RST pins (set LOW or high-impedance) — prevents leakage through GPIO
3. Power-gate the display's VCC rail if sleep current matters (<1 µA target)

On wake:
1. Re-initialize SPI (`SPI.begin(...)`)
2. Re-initialize the display library (`display.init(...)`)
3. Draw and refresh
4. Hibernate again

---

## Common Display Libraries

| Library | Displays supported | Notes |
|---|---|---|
| [GxEPD2](https://github.com/ZinggJM/GxEPD2) | Waveshare, GDEY, Pimoroni | Most actively maintained; partial refresh support |
| [Adafruit EPD](https://github.com/adafruit/Adafruit_EPD) | Adafruit-branded panels | Good Adafruit-ecosystem integration |
| [EPaper ESP32](https://github.com/Xinyuan-LilyGO/LilyGo-EPD47) | LilyGo T5/T5S | Vendor-specific for LilyGo boards |

---

## See Also

!!! tip "See also"
    - [Sensor Integration Best Practices](sensor-integration-best-practices.md) — SPI and I2C wiring alongside e-ink
    - [Power Management & Deep Sleep](power-management-and-deep-sleep.md) — hibernate + deep sleep together
    - [ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md) — complete working example using GxEPD2
