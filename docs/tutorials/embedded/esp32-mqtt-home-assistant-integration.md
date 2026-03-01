---
tags:
  - embedded
  - esp32
  - security
---

# ESP32 + MQTT + Home Assistant Integration (Secure by Design)

**Objective**: Connect an ESP32 sensor node to Home Assistant via a TLS-secured Mosquitto MQTT broker. The ESP32 publishes BME280 temperature/humidity readings and responds to LED toggle commands. HA auto-discovers the device. Everything is authenticated and encrypted.

---

## 1. What You're Building

```
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ESP32 (BME280 + LED)                                       │
  │    │  publishes: home/office/env/temperature                │
  │    │             home/office/env/humidity                   │
  │    │             home/office/env/availability               │
  │    │  subscribes: home/office/led/cmd                       │
  │    │                                                         │
  │    └──── TLS 8883 ────► Mosquitto Broker                    │
  │                              │  (ACL: device writes its own │
  │                              │   namespace only)            │
  │                              │                              │
  │                              └──── Home Assistant ◄─────── │
  │                                    MQTT integration         │
  │                                    Auto-discovery           │
  └─────────────────────────────────────────────────────────────┘
```

**Stack**:
- ESP32 (Arduino Core) + BME280 + LED
- Mosquitto 2.x with TLS + password auth + ACL
- Home Assistant (any installation method)
- PubSubClient + WiFiClientSecure libraries

---

## 2. Hardware

| Component | Notes |
|---|---|
| ESP32 DevKit (WROOM-32) | Any DevKit with exposed GPIO |
| BME280 breakout (3.3 V) | I2C, address `0x76` |
| LED + 220 Ω resistor | Connected to GPIO 2 (built-in LED works too) |
| Optional: relay module | Low-voltage switching only — never wire ESP32 to mains |

```
I2C wiring (4.7 kΩ pull-ups to 3.3 V required):
GPIO21 (SDA) ──── BME280 SDA
GPIO22 (SCL) ──── BME280 SCL

LED:
GPIO2 ──[220Ω]──► LED+ LED- ──► GND
```

---

## 3. MQTT Broker Setup (Mosquitto)

### Install

```bash
# Debian/Ubuntu (or Home Assistant add-on)
sudo apt install mosquitto mosquitto-clients
```

### TLS certificates

Generate a self-signed CA and broker cert (full commands in [MQTT Security Best Practices](../../best-practices/esp32/mqtt-security-best-practices.md)):

```bash
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/CN=HomeIoT-CA"
openssl genrsa -out broker.key 2048
openssl req -new -key broker.key -out broker.csr -subj "/CN=broker.home.local"
openssl x509 -req -days 3650 -in broker.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out broker.crt
```

### Mosquitto config

```conf
# /etc/mosquitto/mosquitto.conf
allow_anonymous false
password_file   /etc/mosquitto/passwd
acl_file        /etc/mosquitto/acl

listener 8883
cafile   /etc/mosquitto/certs/ca.crt
certfile /etc/mosquitto/certs/broker.crt
keyfile  /etc/mosquitto/certs/broker.key
tls_version tlsv1.2
```

### Create users

```bash
sudo mosquitto_passwd -c /etc/mosquitto/passwd esp32_office
sudo mosquitto_passwd    /etc/mosquitto/passwd homeassistant
sudo systemctl restart mosquitto
```

### ACL file

```conf
# /etc/mosquitto/acl

user homeassistant
topic read  home/#
topic write home/+/+/cmd

user esp32_office
topic read  home/office/+/cmd
topic write home/office/+/state
topic write home/office/+/telemetry
topic write home/office/+/availability
# MQTT discovery (required for HA auto-discovery)
topic write homeassistant/sensor/esp32_office/#
topic write homeassistant/light/esp32_office/#
```

---

## 4. ESP32 Firmware

### PlatformIO config

```ini
; platformio.ini
[env:esp32dev]
platform       = espressif32
board          = esp32dev
framework      = arduino
monitor_speed  = 115200
lib_deps =
    knolleary/PubSubClient
    adafruit/Adafruit BME280 Library
```

### config.h

```cpp
#pragma once

// WiFi — load from NVS in production; hardcoded here for brevity
#define WIFI_SSID    "your_ssid"
#define WIFI_PASS    "your_password"

// MQTT broker
#define MQTT_HOST    "192.168.1.10"   // broker LAN IP
#define MQTT_PORT    8883
#define MQTT_USER    "esp32_office"
#define MQTT_PASS    "your_device_password"  // store in NVS, not here
#define MQTT_CLIENT_ID "esp32_office_01"     // unique per device

// Topic namespace
#define TOPIC_TEMP   "home/office/env/temperature"
#define TOPIC_HUM    "home/office/env/humidity"
#define TOPIC_LED_CMD "home/office/led/cmd"
#define TOPIC_LED_STATE "home/office/led/state"
#define TOPIC_AVAIL  "home/office/env/availability"

// Pins
#define LED_PIN      2
#define I2C_SDA      21
#define I2C_SCL      22
#define PUBLISH_INTERVAL_MS 30000UL
```

### CA Certificate

Paste your `ca.crt` contents into firmware — it is a public trust anchor, not a secret:

```cpp
// certs.h
#pragma once
static const char MQTT_CA_CERT[] = R"EOF(
-----BEGIN CERTIFICATE-----
MIICpDCCAYwCCQD... (your ca.crt contents) ...
-----END CERTIFICATE-----
)EOF";
```

### main.cpp

```cpp
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_BME280.h>
#include "config.h"
#include "certs.h"

static WiFiClientSecure tlsClient;
static PubSubClient     mqtt(tlsClient);
static Adafruit_BME280  bme;
static uint32_t         lastPublishMs = 0;
static bool             ledState = false;

// ── WiFi ─────────────────────────────────────────────────────────────────────

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    Serial.printf("[WiFi] Connecting to %s ", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    uint8_t attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500); Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected, IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] Failed — will retry");
    }
}

// ── MQTT callbacks ────────────────────────────────────────────────────────────

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
    String msg;
    for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
    Serial.printf("[MQTT] Received %s = %s\n", topic, msg.c_str());

    if (String(topic) == TOPIC_LED_CMD) {
        ledState = (msg == "ON");
        digitalWrite(LED_PIN, ledState ? HIGH : LOW);
        // Publish confirmed state back
        mqtt.publish(TOPIC_LED_STATE, ledState ? "ON" : "OFF", false);
    }
}

// ── MQTT connect ──────────────────────────────────────────────────────────────

bool connectMqtt() {
    if (mqtt.connected()) return true;
    Serial.print("[MQTT] Connecting...");

    // LWT: broker publishes "offline" if ESP32 disconnects uncleanly
    bool ok = mqtt.connect(MQTT_CLIENT_ID, MQTT_USER, MQTT_PASS,
                           TOPIC_AVAIL, 1, true, "offline");
    if (ok) {
        Serial.println(" connected");
        mqtt.publish(TOPIC_AVAIL, "online", true);   // retained availability
        mqtt.subscribe(TOPIC_LED_CMD, 1);
        publishHADiscovery();
    } else {
        Serial.printf(" failed, rc=%d\n", mqtt.state());
    }
    return ok;
}

// ── HA MQTT Discovery ─────────────────────────────────────────────────────────

void publishHADiscovery() {
    // Temperature sensor
    const char* tempCfg = R"({
      "name":"Office Temperature",
      "unique_id":"esp32_office_temp",
      "state_topic":"home/office/env/temperature",
      "unit_of_measurement":"°C",
      "device_class":"temperature",
      "availability_topic":"home/office/env/availability",
      "device":{"identifiers":["esp32_office"],"name":"ESP32 Office","model":"DevKit","manufacturer":"Espressif"}
    })";
    mqtt.publish("homeassistant/sensor/esp32_office/temperature/config", tempCfg, true);

    // Humidity sensor
    const char* humCfg = R"({
      "name":"Office Humidity",
      "unique_id":"esp32_office_hum",
      "state_topic":"home/office/env/humidity",
      "unit_of_measurement":"%",
      "device_class":"humidity",
      "availability_topic":"home/office/env/availability",
      "device":{"identifiers":["esp32_office"]}
    })";
    mqtt.publish("homeassistant/sensor/esp32_office/humidity/config", humCfg, true);

    // LED light
    const char* ledCfg = R"({
      "name":"Office LED",
      "unique_id":"esp32_office_led",
      "command_topic":"home/office/led/cmd",
      "state_topic":"home/office/led/state",
      "availability_topic":"home/office/env/availability",
      "payload_on":"ON","payload_off":"OFF",
      "device":{"identifiers":["esp32_office"]}
    })";
    mqtt.publish("homeassistant/light/esp32_office/led/config", ledCfg, true);
}

// ── Sensor publish ────────────────────────────────────────────────────────────

void publishSensors() {
    char buf[16];
    snprintf(buf, sizeof(buf), "%.1f", bme.readTemperature());
    mqtt.publish(TOPIC_TEMP, buf, false);
    snprintf(buf, sizeof(buf), "%.1f", bme.readHumidity());
    mqtt.publish(TOPIC_HUM, buf, false);
    Serial.printf("[MQTT] Published temp=%s hum=%s\n", buf, buf);
}

// ── Setup / Loop ─────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    Wire.begin(I2C_SDA, I2C_SCL, 400000);

    if (!bme.begin(0x76)) {
        Serial.println("[BME280] Not found — check wiring");
    }

    tlsClient.setCACert(MQTT_CA_CERT);  // pin our self-signed CA
    // tlsClient.setInsecure();  // ← NEVER use this in production

    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    mqtt.setCallback(onMqttMessage);
    mqtt.setBufferSize(512);  // increase if discovery JSON is larger
    mqtt.setKeepAlive(30);

    connectWiFi();
    connectMqtt();
}

void loop() {
    // Reconnect if needed — non-blocking (no delay)
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    if (!mqtt.connected()) connectMqtt();

    mqtt.loop();  // process incoming messages + keepalive

    // Publish sensor data on interval
    if (millis() - lastPublishMs > PUBLISH_INTERVAL_MS) {
        publishSensors();
        lastPublishMs = millis();
    }
}
```

---

## 5. Home Assistant Configuration

### Option A — MQTT Discovery (recommended)

No manual YAML needed. After the ESP32 connects and publishes discovery messages:

1. Settings → Devices & Services → MQTT → confirm integration is active
2. Navigate to Settings → Devices — "ESP32 Office" appears automatically
3. The device shows: Office Temperature, Office Humidity, Office LED

If the device doesn't appear within 60 seconds, check the discovery topic path and that the HA MQTT user has `read` ACL on `homeassistant/#`.

### Option B — Manual YAML (fallback)

```yaml
# configuration.yaml
mqtt:
  sensor:
    - name: "Office Temperature"
      unique_id: esp32_office_temp
      state_topic: "home/office/env/temperature"
      unit_of_measurement: "°C"
      device_class: temperature
      availability_topic: "home/office/env/availability"

    - name: "Office Humidity"
      unique_id: esp32_office_hum
      state_topic: "home/office/env/humidity"
      unit_of_measurement: "%"
      device_class: humidity
      availability_topic: "home/office/env/availability"

  light:
    - name: "Office LED"
      unique_id: esp32_office_led
      command_topic: "home/office/led/cmd"
      state_topic: "home/office/led/state"
      availability_topic: "home/office/env/availability"
      payload_on: "ON"
      payload_off: "OFF"
```

Add your MQTT broker credentials in: Settings → Devices & Services → MQTT → Configure.

---

## 6. Security Hardening Checklist

- [ ] Mosquitto `allow_anonymous false` — confirmed
- [ ] TLS on port 8883 — no plaintext 1883 listener active
- [ ] CA cert pinned in firmware via `setCACert()` — `setInsecure()` never used
- [ ] Per-device MQTT credentials in NVS, not in source code
- [ ] Unique `MQTT_CLIENT_ID` per device (include MAC suffix in production)
- [ ] ACL: device writes only to its own topic namespace
- [ ] Command topics NOT retained (`retain=false` on HA publish)
- [ ] LWT configured for availability topic
- [ ] MQTT broker not reachable from internet
- [ ] IoT VLAN isolating ESP32 from main LAN (optional but recommended)

---

## 7. Testing

### Verify broker TLS from host

```bash
# Confirm TLS handshake succeeds (should return broker CONNACK)
mosquitto_pub --cafile ca.crt -h 192.168.1.10 -p 8883 \
  -u homeassistant -P your_ha_password \
  -t "home/office/test" -m "hello" -d
```

### Simulate ESP32 publish

```bash
mosquitto_pub --cafile ca.crt -h 192.168.1.10 -p 8883 \
  -u esp32_office -P your_device_password \
  -t "home/office/env/temperature" -m "22.4"
```

Watch HA: the Office Temperature entity should update immediately.

### Send command to LED

```bash
mosquitto_pub --cafile ca.crt -h 192.168.1.10 -p 8883 \
  -u homeassistant -P your_ha_password \
  -t "home/office/led/cmd" -m "ON"
```

### ESP32 serial log (expected)

```
[WiFi] Connected, IP: 192.168.1.42
[MQTT] Connecting... connected
[MQTT] Published temp=22.4 hum=58.1
[MQTT] Received home/office/led/cmd = ON
```

---

## 8. Failure Modes

**TLS handshake fails** (`mqtt.state()` = -2):
- Verify `ca.crt` content matches the broker's signing CA exactly
- Check that `MQTT_PORT` is `8883` not `1883`
- Ensure ESP32 heap before connect is >80 KB: `Serial.println(ESP.getFreeHeap())`
- Try `tlsClient.setInsecure()` temporarily to isolate whether the issue is the cert or the network — then fix the cert

**ESP32 disconnects repeatedly** (`rc=-3` or `-4`):
- Increase `mqtt.setKeepAlive(60)` — poor WiFi signal can drop connections
- Check broker `log_type all` for disconnect reason
- Verify `MQTT_CLIENT_ID` is unique — duplicate client IDs cause kick-loops

**HA not discovering device**:
- Check ACL: `esp32_office` user must have `write` on `homeassistant/sensor/esp32_office/#`
- Confirm HA MQTT integration is configured with same broker credentials
- Subscribe manually from HA host: `mosquitto_sub ... -t "homeassistant/#" -v` and watch for discovery messages

**Auth failure** (`rc=-5`):
- Verify username/password match `mosquitto_passwd` file exactly (case-sensitive)
- Restart Mosquitto after any `passwd` file changes: `sudo systemctl restart mosquitto`

---

## 9. Scaling Strategy

**Per-room topic segmentation**: already built into the `home/{room}/{device}/{measurement}` namespace. Add more rooms by creating new users and ACL entries.

**Separate broker for untrusted IoT**: if you have unvetted IoT devices (cheap smart plugs, cameras), run a second Mosquitto instance on a separate port/VLAN. HA can bridge topics selectively between brokers.

**Rate limiting**: Mosquitto 2.x supports `max_inflight_messages` and `max_queued_messages` per client. For sensor spammers, add per-user publish rate limiting via a plugin or upstream proxy.

**Device rotation**: when replacing a device, create a new credential, update NVS, then revoke the old credential. Because each device has its own MQTT username, revocation is surgical — no other device is affected.

**OTA updates**: see [Embedded Security & OTA](../../best-practices/esp32/embedded-security-and-ota.md) for signing and delivery strategy. HA can trigger OTA via a command topic.

---

## 10. See Also

!!! tip "See also"
    - [MQTT Security Best Practices](../../best-practices/esp32/mqtt-security-best-practices.md) — broker hardening, ACL design, TLS cert generation, topic naming rules
    - [Home Assistant Security Best Practices](../../best-practices/home-automation/home-assistant-security-best-practices.md) — HA access control, reverse proxy, IoT VLAN, zero-trust mindset
    - [ESP32 Programming Architecture](../../best-practices/esp32/esp32-programming-architecture.md) — non-blocking reconnect patterns, state machines, ISR safety
    - [Embedded Security & OTA](../../best-practices/esp32/embedded-security-and-ota.md) — NVS credential storage, OTA firmware updates
    - [ESP32 Hardware & Electrical Safety](../../best-practices/esp32/esp32-hardware-and-electrical-safety.md) — GPIO limits, 3.3 V logic, relay safety
    - [ESP32 E-Ink Environmental Monitor](esp32-eink-sensor-monitor.md) — companion project: same BME280 sensor, different output (e-ink + deep sleep instead of MQTT)
