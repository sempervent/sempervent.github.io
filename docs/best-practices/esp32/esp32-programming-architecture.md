---
tags:
  - embedded
  - esp32
---

# ESP32 Programming Architecture

Good firmware architecture is invisible when it works and catastrophic when it doesn't. These patterns apply whether you're using Arduino Core or ESP-IDF.

---

## Firmware Layout

```
project/
├── src/
│   ├── main.cpp         # setup() / app_main() — thin glue only
│   ├── sensors.cpp/.h   # sensor read, calibration, smoothing
│   ├── display.cpp/.h   # render logic, frame buffer
│   ├── power.cpp/.h     # sleep configuration, wake reason
│   ├── config.h         # compile-time constants
│   └── state.h          # shared state struct (read-only from ISR)
├── platformio.ini        # or CMakeLists.txt for ESP-IDF
└── partitions.csv        # custom partition table if OTA/NVS needed
```

**Rule**: `main.cpp` orchestrates; it does not implement. Keep it under 60 lines.

---

## Event Loop vs Polling

| Approach | Use when |
|---|---|
| Polling in `loop()` | Simple, single-concern sketches; rapid prototyping |
| FreeRTOS tasks | Multiple concurrent concerns; blocking I/O; precise timing |
| Interrupt + queue | Edge detection, UART RX, encoder counting |
| Timer callbacks | Periodic sampling with strict cadence |

**Never** block in an ISR. Never call `delay()` in a task with tight timing requirements.

---

## State Machine Pattern

Model complex behavior as explicit states rather than nested `if` chains.

```cpp
// config.h
enum class AppState {
    INIT,
    READING_SENSORS,
    RENDERING,
    SLEEPING,
    ERROR,
};

// state.h — shared, written only from main loop
struct AppCtx {
    AppState    state     = AppState::INIT;
    float       tempC     = 0.0f;
    float       humidity  = 0.0f;
    float       pressure  = 0.0f;
    uint32_t    lux       = 0;
    uint8_t     errorCode = 0;
};
```

```cpp
// main.cpp
void loop() {
    switch (ctx.state) {
        case AppState::INIT:          runInit(ctx);     break;
        case AppState::READING_SENSORS: readSensors(ctx); break;
        case AppState::RENDERING:     renderDisplay(ctx); break;
        case AppState::SLEEPING:      enterSleep(ctx);  break;
        case AppState::ERROR:         handleError(ctx); break;
    }
}
```

Transitions are explicit. Every state has a defined exit path.

---

## FreeRTOS Task Separation

Arduino Core on ESP32 runs on FreeRTOS. Use it.

```cpp
// Two tasks: one for sensors, one for display
xTaskCreatePinnedToCore(sensorTask, "sensors", 4096, &ctx, 2, nullptr, 0); // Core 0
xTaskCreatePinnedToCore(displayTask, "display", 8192, &ctx, 1, nullptr, 1); // Core 1
```

Stack sizing:
- Sensor task: 2–4 KB typical
- Display/graphics: 8–16 KB (frame buffer operations are stack-hungry)
- Too small → `Guru Meditation Error: Core X panic'ed (Unhandled debug exception)`

Use `uxTaskGetStackHighWaterMark(nullptr)` to measure actual stack use during development.

---

## ISR Safety Rules

```cpp
// DO: volatile for ISR-shared variables
volatile bool buttonPressed = false;
// DO: use IRAM_ATTR so the ISR lives in fast RAM, not flash
void IRAM_ATTR onButtonPress() { buttonPressed = true; }

// DO NOT: call Serial.print, malloc, or delay() inside ISR
// DO NOT: access I2C/SPI peripherals from ISR
```

For data larger than a flag, use a FreeRTOS queue:

```cpp
static QueueHandle_t eventQueue;
// In ISR:
BaseType_t woken = pdFALSE;
xQueueSendFromISR(eventQueue, &event, &woken);
portYIELD_FROM_ISR(woken);
```

---

## Watchdog Timer

Enable the task watchdog on long-running tasks:

```cpp
// ESP-IDF
esp_task_wdt_add(nullptr);  // subscribe current task
// ...in loop...
esp_task_wdt_reset();       // feed the dog
```

On Arduino Core, the IDLE task feeds the WDT. If your `loop()` blocks for >5 s (default), the ESP resets. This is **good** — it surfaces hangs.

---

## Logging Strategy

```cpp
// ESP-IDF: structured, level-filtered
ESP_LOGI("SENSOR", "Temp=%.1f°C Hum=%.1f%%", tempC, humidity);
ESP_LOGE("OTA",    "Update failed: %s", esp_err_to_name(err));

// Arduino Core: use Serial.printf, but gate behind a flag
#define LOG_LEVEL 2  // 0=off, 1=error, 2=info, 3=debug
#if LOG_LEVEL >= 2
  Serial.printf("[INFO] temp=%.1f\n", tempC);
#endif
```

**Remove verbose logging before flashing battery-powered devices.** `Serial.printf` keeps the UART peripheral active, burning power.

---

## Compile-Time Configuration

```cpp
// config.h — single source of truth
#pragma once
#define WIFI_SSID      "YOUR_SSID"    // replace with NVS in production
#define SAMPLE_RATE_MS  30000UL       // 30-second sensor poll
#define SLEEP_DURATION_S 300UL        // 5-minute deep sleep
#define EINK_BUSY_PIN   4
#define EINK_RST_PIN    16
#define EINK_DC_PIN     17
#define EINK_CS_PIN     5
#define I2C_SDA_PIN     21
#define I2C_SCL_PIN     22
#define I2C_FREQ_HZ     400000        // 400 kHz fast mode
```

Never scatter magic numbers through source files.

---

## Memory Discipline

**Stack vs heap**:
- Prefer stack allocation for small, short-lived objects
- Use `static` for large buffers (frame buffers, string scratch pads)
- Avoid `new`/`malloc` after init — heap fragmentation kills long-running firmware

**Heap monitoring**:
```cpp
Serial.printf("Free heap: %u bytes\n", ESP.getFreeHeap());
Serial.printf("Min free heap: %u bytes\n", ESP.getMinFreeHeap());
```

Run for 24 hours and verify min free heap doesn't drift downward. If it does, you have a leak.

---

## See Also

!!! tip "See also"
    - [Power Management & Deep Sleep](power-management-and-deep-sleep.md) — complement architecture with a proper sleep strategy
    - [ESP32 E-Ink Environmental Monitor](../../tutorials/embedded/esp32-eink-sensor-monitor.md) — see these patterns applied end-to-end
    - [Embedded Security & OTA](embedded-security-and-ota.md) — secrets management and OTA update discipline
