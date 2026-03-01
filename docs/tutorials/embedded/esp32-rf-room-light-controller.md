---
tags:
  - embedded
  - esp32
---

# ESP32 RF Room Light Controller (LCD + Rotary Encoder + Button + 433 MHz RF)

**Objective**: Build a wall-mounted ESP32 controller that selects a room with a rotary encoder, toggles lights over 433 MHz RF, and displays current state on a 16×2 I2C LCD — all from a clean, non-blocking firmware.

No cloud. No app. A physical knob on a wall that works instantly.

---

## 1. What You're Building

- Rotate encoder → cycle through rooms/channels
- Press encoder button → toggle selected room light
- Dedicated toggle button → quick toggle current room
- LCD → shows room name + ON/OFF state
- 433 MHz RF TX → sends signal to RF-controlled wall outlets

```
  ┌────────────────────────────────────────────────┐
  │                                                │
  │   Rotary Encoder ──────────────────────────┐  │
  │   (KY-040)                                 │  │
  │                                         ┌──▼──┴──────┐
  │   Toggle Button ──────────────────────► │   ESP32    │──► 433 MHz RF TX ──► Outlets
  │                                         │            │
  │   16×2 I2C LCD  ◄─────────────────────── │            │
  │   (display state)                        └────────────┘
  │                                                │
  └────────────────────────────────────────────────┘
```

---

## 2. Hardware Components

| Component | Suggested Part | Notes |
|---|---|---|
| ESP32 | DevKitC (WROOM-32) | Any DevKit with exposed GPIOs |
| LCD | 16×2 I2C LCD (PCF8574 backpack) | I2C address `0x27` or `0x3F` |
| Rotary encoder | KY-040 | Built-in pushbutton; outputs A/B + SW |
| Toggle button | Momentary N.O. tactile switch | 6×6 mm or panel-mount |
| 433 MHz RF TX | FS1000A or XY-FST | Pair with an RF remote outlet set |
| Optional | 2N2222 NPN transistor | Boost RF TX data line drive current |
| Decoupling caps | 100 nF ceramic per VCC pin | Reduces noise on RF TX and LCD |
| Resistors | 10 kΩ × 3 | Button/encoder pull-ups if needed |

---

## 3. Electrical Safety Notes

!!! warning "Hardware safety"
    - **3.3 V logic only.** The ESP32 GPIO cannot tolerate 5 V inputs. KY-040 runs fine at 3.3 V; LCD PCF8574 backpacks typically accept 3.3–5 V but confirm your module.
    - **Never connect the ESP32 directly to mains wiring.** The RF outlets handle the 120/240 V side — the ESP32 only drives a 433 MHz transmitter module.
    - **RF TX stability:** The FS1000A is sensitive to supply noise. Decouple VCC with a 100 nF ceramic + 10 µF electrolytic close to the module.
    - **Floating inputs cause phantom events.** Use internal or external pull-ups on all button and encoder pins.
    - **Level shifting:** If your LCD backpack outputs 5 V on SDA/SCL, add a BSS138-based bidirectional level shifter between the backpack and the ESP32 I2C pins.

---

## 4. Wiring

```
Component         ESP32 Pin   Notes
─────────────     ─────────   ──────────────────────────────
LCD SDA           GPIO 21     4.7 kΩ pull-up to 3.3 V
LCD SCL           GPIO 22     4.7 kΩ pull-up to 3.3 V
LCD VCC           3.3 V       or 5 V if backpack has own reg
LCD GND           GND

Encoder A (CLK)   GPIO 32     INPUT_PULLUP
Encoder B (DT)    GPIO 33     INPUT_PULLUP
Encoder SW (btn)  GPIO 25     INPUT_PULLUP, active LOW
Encoder VCC       3.3 V
Encoder GND       GND

Toggle Button     GPIO 26     INPUT_PULLUP, active LOW
Button other leg  GND

RF TX DATA        GPIO 27     (optional: via 2N2222 base)
RF TX VCC         3.3 V       decoupled
RF TX GND         GND
```

**Optional RF drive transistor** (improves RF range and reduces GPIO current load):

```
GPIO27 ──[1 kΩ]──► 2N2222 Base
                    Emitter → GND
                    Collector → RF TX DATA pin
RF TX DATA pulled HIGH to VCC via 10 kΩ
```

---

## 5. Project Layout

```
esp32-rf-controller/
├── platformio.ini
├── config.h
└── src/
    ├── main.cpp      # setup(), loop() — thin orchestration only
    ├── state.h       # AppState enum + RoomCtx struct
    ├── input.cpp/.h  # encoder + button read, debounce, events
    ├── display.cpp/.h# LCD rendering, row diffing
    └── rf.cpp/.h     # RF code table, send with guard
```

---

## 6. Configuration

```cpp
// config.h
#pragma once

// I2C
#define I2C_SDA       21
#define I2C_SCL       22
#define LCD_ADDR      0x27
#define LCD_COLS      16
#define LCD_ROWS       2

// Encoder
#define ENC_A         32
#define ENC_B         33
#define ENC_SW        25   // active LOW

// Button
#define BTN_TOGGLE    26   // active LOW

// RF
#define RF_TX_PIN     27
#define RF_PROTOCOL    1   // RC-Switch protocol (1 = standard)
#define RF_PULSE_LEN 350   // µs — tune per receiver
#define RF_REPEAT      3   // transmit N times per command

// Rooms
#define NUM_ROOMS      4
```

---

## 7. State Machine

```cpp
// state.h
#pragma once
#include <stdint.h>

enum class AppMode {
    ROOM_SELECT,    // encoder changes active room
    TOGGLE_CONFIRM, // brief visual feedback after toggle
    SETTINGS,       // future: protocol/code config
};

struct RoomCtx {
    const char* name;
    uint32_t    rfCodeOn;
    uint32_t    rfCodeOff;
    bool        isOn;
};

struct AppState {
    AppMode     mode       = AppMode::ROOM_SELECT;
    uint8_t     activeRoom = 0;
    uint32_t    modeEnteredMs = 0;
    RoomCtx     rooms[NUM_ROOMS] = {
        {"Living Rm", 0x111111, 0x111110, false},
        {"Bedroom",   0x222221, 0x222220, false},
        {"Office",    0x333331, 0x333330, false},
        {"Hallway",   0x444441, 0x444440, false},
    };
};
```

The RF codes (`rfCodeOn`/`rfCodeOff`) must be sniffed from your actual outlet remotes using an RF receiver and `rc-switch`. Replace the placeholder values before use (see Section 9).

---

## 8. Rotary Encoder Handling

The KY-040 produces a Gray-coded quadrature signal. Read it in a hardware interrupt for zero-miss guarantee.

```cpp
// input.cpp (partial)
#include <Arduino.h>
#include "config.h"
#include "input.h"

static volatile int8_t  encoderDelta = 0;
static volatile uint8_t lastEncState = 0;

void IRAM_ATTR onEncoderISR() {
    uint8_t a = digitalRead(ENC_A);
    uint8_t b = digitalRead(ENC_B);
    uint8_t cur = (a << 1) | b;
    // Gray code transition table
    static const int8_t table[16] = {
         0,-1, 1, 0,
         1, 0, 0,-1,
        -1, 0, 0, 1,
         0, 1,-1, 0
    };
    int8_t delta = table[(lastEncState << 2) | cur];
    encoderDelta += delta;
    lastEncState = cur;
}

void inputInit() {
    pinMode(ENC_A,       INPUT_PULLUP);
    pinMode(ENC_B,       INPUT_PULLUP);
    pinMode(ENC_SW,      INPUT_PULLUP);
    pinMode(BTN_TOGGLE,  INPUT_PULLUP);

    lastEncState = (digitalRead(ENC_A) << 1) | digitalRead(ENC_B);
    attachInterrupt(digitalPinToInterrupt(ENC_A), onEncoderISR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_B), onEncoderISR, CHANGE);
}

// Call from loop(): returns net steps since last call
int8_t inputReadEncoder() {
    noInterrupts();
    int8_t delta = encoderDelta;
    encoderDelta  = 0;
    interrupts();
    return delta;
}
```

---

## 9. Button Handling

Both buttons are active-LOW with internal pull-ups. Debounce in software:

```cpp
// input.cpp (continued)

struct ButtonState {
    uint8_t  pin;
    bool     lastStable;
    bool     reading;
    uint32_t lastChangeMs;
};

static ButtonState encBtn   = {ENC_SW,      true, true, 0};
static ButtonState toggleBtn= {BTN_TOGGLE,  true, true, 0};

static const uint32_t DEBOUNCE_MS   = 50;
static const uint32_t LONG_PRESS_MS = 800;

enum class BtnEvent { NONE, SHORT_PRESS, LONG_PRESS };

BtnEvent inputPollButton(ButtonState& b) {
    bool raw = digitalRead(b.pin);  // HIGH=idle, LOW=pressed
    uint32_t now = millis();

    if (raw != b.reading) {
        b.reading      = raw;
        b.lastChangeMs = now;
    }
    if ((now - b.lastChangeMs) < DEBOUNCE_MS) return BtnEvent::NONE;
    if (raw == b.lastStable)                  return BtnEvent::NONE;

    b.lastStable = raw;
    if (!raw) return BtnEvent::NONE;  // pressed edge — wait for release

    // Release edge: measure how long it was held
    uint32_t held = now - b.lastChangeMs;
    return (held >= LONG_PRESS_MS) ? BtnEvent::LONG_PRESS : BtnEvent::SHORT_PRESS;
}

// Convenience wrappers used from main.cpp
BtnEvent inputEncButton()    { return inputPollButton(encBtn); }
BtnEvent inputToggleButton() { return inputPollButton(toggleBtn); }
```

---

## 10. RF Transmission

Use the [rc-switch](https://github.com/sui77/rc-switch) library. First, sniff your remote's codes:

```ini
; platformio.ini — add to lib_deps
sui77/rc-switch
marcoschwartz/LiquidCrystal_I2C
```

```cpp
// rf.cpp
#include <RCSwitch.h>
#include "config.h"
#include "rf.h"

static RCSwitch rf;
static uint32_t lastSendMs = 0;
static const uint32_t SEND_COOLDOWN_MS = 500;  // prevent RF spam

void rfInit() {
    rf.enableTransmit(RF_TX_PIN);
    rf.setProtocol(RF_PROTOCOL);
    rf.setPulseLength(RF_PULSE_LEN);
    rf.setRepeatTransmit(RF_REPEAT);
}

bool rfSend(uint32_t code, uint8_t bits = 24) {
    uint32_t now = millis();
    if ((now - lastSendMs) < SEND_COOLDOWN_MS) return false;
    rf.send(code, bits);
    lastSendMs = now;
    return true;
}
```

**Sniffing your remote's RF codes** (requires a 433 MHz receiver module):

```cpp
// Temporary sketch — flash to sniff, then remove
#include <RCSwitch.h>
RCSwitch rf;
void setup() { Serial.begin(115200); rf.enableReceive(digitalPinToInterrupt(4)); }
void loop() {
    if (rf.available()) {
        Serial.printf("Code: %lu  Bits: %u  Protocol: %u\n",
            rf.getReceivedValue(), rf.getReceivedBitlength(), rf.getReceivedProtocol());
        rf.resetAvailable();
    }
}
```

Copy the `Code` value into `rfCodeOn` / `rfCodeOff` for each room's RF outlet.

---

## 11. LCD Rendering

Minimize I2C writes by tracking what's currently on each row and only updating changed lines.

```cpp
// display.cpp
#include <LiquidCrystal_I2C.h>
#include "config.h"
#include "display.h"

static LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);
static char prevRow[2][LCD_COLS + 1] = {"", ""};

void displayInit() {
    lcd.init();
    lcd.backlight();
    lcd.clear();
}

static void setRow(uint8_t row, const char* text) {
    if (strncmp(prevRow[row], text, LCD_COLS) == 0) return;  // no change
    strncpy(prevRow[row], text, LCD_COLS);
    lcd.setCursor(0, row);
    lcd.printf("%-*s", LCD_COLS, text);  // left-align, pad to full width
}

void displayUpdate(const char* roomName, bool isOn, bool txActive) {
    char row0[LCD_COLS + 1];
    char row1[LCD_COLS + 1];
    snprintf(row0, sizeof(row0), "Room: %s", roomName);
    snprintf(row1, sizeof(row1), "Light: %s%s", isOn ? "ON " : "OFF", txActive ? " [TX]" : "    ");
    setRow(0, row0);
    setRow(1, row1);
}
```

The `[TX]` indicator appears briefly after RF transmission.

---

## 12. Main Orchestration

```cpp
// main.cpp
#include <Arduino.h>
#include "config.h"
#include "state.h"
#include "input.h"
#include "display.h"
#include "rf.h"

static AppState app;
static bool     txIndicator    = false;
static uint32_t txIndicatorOff = 0;

void setup() {
    Serial.begin(115200);
    inputInit();
    displayInit();
    rfInit();
    displayUpdate(app.rooms[app.activeRoom].name,
                  app.rooms[app.activeRoom].isOn, false);
}

void loop() {
    // Encoder movement → change active room
    int8_t delta = inputReadEncoder();
    if (delta != 0) {
        int next = (int)app.activeRoom + (delta > 0 ? 1 : -1);
        app.activeRoom = (uint8_t)((next + NUM_ROOMS) % NUM_ROOMS);
    }

    // Encoder button OR toggle button → toggle current room
    auto doToggle = [&]() {
        auto& room = app.rooms[app.activeRoom];
        room.isOn = !room.isOn;
        rfSend(room.isOn ? room.rfCodeOn : room.rfCodeOff);
        txIndicator    = true;
        txIndicatorOff = millis() + 600;
    };

    if (inputEncButton()    == BtnEvent::SHORT_PRESS) doToggle();
    if (inputToggleButton() == BtnEvent::SHORT_PRESS) doToggle();

    // Clear TX indicator after timeout
    if (txIndicator && millis() > txIndicatorOff) {
        txIndicator = false;
    }

    // Render — only writes to LCD if content changed
    displayUpdate(app.rooms[app.activeRoom].name,
                  app.rooms[app.activeRoom].isOn,
                  txIndicator);

    delay(10);  // 10 ms poll interval — acceptable for interactive controller
}
```

**Why `delay(10)` here instead of no-delay?** This is an interactive, always-on USB-powered controller — not a battery device. A 10 ms loop keeps CPU load low without impacting responsiveness. For true non-blocking design, replace with `millis()` timestamps per task.

---

## 13. Power Considerations

This controller is designed for always-on USB power:

| Condition | Estimated current |
|---|---|
| ESP32 active (no WiFi) | 30–80 mA |
| LCD + backlight | 20–40 mA |
| RF TX during transmission | 25–40 mA (burst, ~50 ms) |
| Idle total | ~60–90 mA |

**Deep sleep is not appropriate here.** A wall controller must respond immediately to physical input — waking from deep sleep takes 200–500 ms and loses encoder interrupt state. Use light sleep with GPIO wake if power saving is a concern.

**USB power supply recommendations**:
- Minimum 500 mA USB supply (standard phone charger is fine)
- Use a dedicated 3.3 V regulated rail if powering from barrel jack
- Add 100 µF bulk capacitor near ESP32 power pin to absorb RF TX transients

**Optional battery mode**: If battery-powered, power-gate the LCD backlight (most power consumer) during idle periods via a GPIO-controlled transistor. Keep the encoder interrupt live so any rotation wakes the display.

---

## 14. Security Notes

!!! warning "433 MHz RF is not secure"
    - **Replay attacks are trivial.** Any SDR receiver within range can capture and replay your RF codes.
    - **No authentication or encryption.** Anyone with a 433 MHz transmitter and your codes can control your outlets.
    - **Do not use this for locks, alarms, garage doors, or any safety-critical application.**
    - **Suitable for**: low-stakes room lighting, decorative outlets, non-critical convenience switching.
    - **Alternative for higher security**: nRF24L01 with AES encryption, or WiFi-controlled smart outlets with TLS + authentication.

---

## 15. Future Improvements

- **WiFi MQTT fallback** — publish room state to a broker; home automation integration
- **Web UI** — embed a tiny HTTP server (AsyncWebServer); expose `/toggle/{room}` endpoint
- **nRF24L01 upgrade** — bidirectional, 2.4 GHz, AES-capable; replace 433 MHz for two-way confirmation
- **Touch screen** — replace encoder + LCD with a small ILI9341 TFT and capacitive touch
- **State persistence** — save last known room states to NVS; survive power cycles
- **OTA firmware update** — update over WiFi without removing the device from the wall
- **Multi-controller sync** — two wall panels synchronized via MQTT or ESP-NOW

---

## 16. See Also

!!! tip "See also"
    - [ESP32 Programming Architecture](../../best-practices/esp32/esp32-programming-architecture.md) — state machines, ISR safety, non-blocking patterns used throughout this project
    - [ESP32 Hardware & Electrical Safety](../../best-practices/esp32/esp32-hardware-and-electrical-safety.md) — GPIO current limits, pull-ups, 3.3 V logic rules
    - [Power Management & Deep Sleep](../../best-practices/esp32/power-management-and-deep-sleep.md) — why deep sleep is skipped here, and when to use light sleep
    - [Sensor Integration Best Practices](../../best-practices/esp32/sensor-integration-best-practices.md) — I2C wiring and debouncing patterns applied to encoder/button inputs
    - [Embedded Security & OTA](../../best-practices/esp32/embedded-security-and-ota.md) — how to add OTA updates and why 433 MHz is not a secure channel
    - [ESP32 E-Ink Environmental Monitor](esp32-eink-sensor-monitor.md) — companion project: same platform, different application (battery + sensors + e-ink)
