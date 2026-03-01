---
tags:
  - embedded
  - esp32
  - security
---

# MQTT Security Best Practices (ESP32 + Home Networks)

MQTT is the backbone of most home IoT systems — lightweight, fast, and almost always misconfigured. A cleartext broker with anonymous access on your home LAN is one compromised laptop away from a full network pivot. These practices are proportionate for home use and scale to small commercial deployments.

---

## 1. Threat Model

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Guest WiFi device sniffs MQTT | Medium | High (credential + data) | TLS, isolate MQTT to trusted VLAN |
| Compromised IoT device publishes to wrong topics | Medium | Medium (spoof sensor data) | Per-device ACLs |
| Default broker credentials | High | Critical | Disable anonymous, per-device passwords |
| Replay attack via retained command | Medium | Medium | No retained commands; QoS 0 for cmds |
| Internet-exposed broker | Low (if misconfigured) | Critical | Firewall; broker internal-only |
| Compromised device pivoting LAN | Medium | High | IoT VLAN isolation |

**Realistic adversary at home**: a compromised device on the same WiFi network. TLS + ACL + per-device credentials defeats most passive and active attacks in this context.

---

## 2. Broker Hardening

### Mosquitto (recommended for home use)

```conf
# /etc/mosquitto/mosquitto.conf

# Disable anonymous access — this is the single most important setting
allow_anonymous false

# Auth via password file
password_file /etc/mosquitto/passwd

# Access control list
acl_file /etc/mosquitto/acl

# Bind to internal interface only — do NOT expose to 0.0.0.0 unless behind reverse proxy
listener 8883
bind_address 192.168.1.10   # your broker's LAN IP

# TLS (see Section 3)
cafile  /etc/mosquitto/certs/ca.crt
certfile /etc/mosquitto/certs/broker.crt
keyfile  /etc/mosquitto/certs/broker.key
tls_version tlsv1.2
```

**Create a user**:
```bash
sudo mosquitto_passwd -c /etc/mosquitto/passwd esp32_living_room
sudo mosquitto_passwd /etc/mosquitto/passwd esp32_bedroom
sudo mosquitto_passwd /etc/mosquitto/passwd homeassistant
```

One user per device. Never share credentials between devices — if one is compromised, you revoke that one user only.

### ACL file structure

```conf
# /etc/mosquitto/acl

# Home Assistant: read everything, write nothing except cmd topics it controls
user homeassistant
topic read home/#
topic write home/+/+/cmd

# ESP32 living room: read its own cmd, write its own state and telemetry
user esp32_living_room
topic read  home/livingroom/+/cmd
topic write home/livingroom/+/state
topic write home/livingroom/+/telemetry
topic write home/livingroom/+/availability

# Deny all else implicitly — ACL is deny-by-default for listed users
```

**Key rules**:
- No `#` wildcard write access for any device
- Devices can only write to their own namespace
- HA reads broadly but writes narrowly
- Presence of `topic read #` for any user is a red flag — review immediately

---

## 3. TLS Configuration

### Why not port 1883 (plaintext)?

Cleartext MQTT on port 1883 means:
- Any device on the LAN can sniff all sensor data and commands
- Credentials are transmitted in plaintext
- Replay attacks are trivial

**Use port 8883 (MQTT over TLS)** always.

### Self-signed CA (home use)

```bash
# Generate CA key and certificate
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
  -subj "/CN=HomeIoT-CA"

# Generate broker key and CSR
openssl genrsa -out broker.key 2048
openssl req -new -key broker.key -out broker.csr \
  -subj "/CN=broker.home.local"

# Sign broker cert with CA
openssl x509 -req -days 3650 -in broker.csr \
  -CA ca.crt -CAkey ca.key -CAcreateserial -out broker.crt
```

Copy `ca.crt` to all ESP32 devices. The CA cert is ~1400 bytes — small enough for NVS or `#define` in firmware.

### Let's Encrypt (if broker has a public domain)

Use Certbot with DNS challenge. Works even for internal IP, provided you own a domain:
```bash
certbot certonly --dns-cloudflare --dns-cloudflare-credentials ~/.secrets/cf.ini \
  -d mqtt.yourdomain.com
```

Add a cron job for renewal. Update Mosquitto `certfile`/`keyfile` paths and `reload`.

### Client certificate auth (advanced)

Generate per-device client certs for mutual TLS. Requires more key management but eliminates password-based auth:

```bash
openssl genrsa -out esp32_living.key 2048
openssl req -new -key esp32_living.key -out esp32_living.csr -subj "/CN=esp32-living-room"
openssl x509 -req -days 1825 -in esp32_living.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out esp32_living.crt
```

Enable in Mosquitto: `require_certificate true`. ESP32 memory constraint: client certs add ~3–4 KB to heap usage. Test carefully on low-memory builds.

### ESP32 TLS memory considerations

TLS on ESP32 uses mbedTLS. Memory costs:
- TLS handshake: ~35–50 KB heap spike
- Sustained TLS session: ~15–20 KB
- Total free heap needed: at least 80 KB before TLS connect

Check before connecting:
```cpp
Serial.printf("Free heap before MQTT: %u\n", ESP.getFreeHeap());
// Should be > 80000 — if not, reduce buffer sizes or disable logging
```

Do not use `WiFiClientSecure` with `setInsecure()` in production — it disables certificate verification, making TLS useless against MITM.

---

## 4. Topic Design

Use a structured, predictable namespace:

```
home/{room}/{device_type}/{measurement_or_cmd}

Examples:
home/livingroom/light/state        ← current state (ON/OFF), retained
home/livingroom/light/cmd          ← command topic, NOT retained
home/livingroom/bme280/temperature ← sensor reading
home/livingroom/bme280/humidity
home/bedroom/motion/state
home/office/esp32_desk/availability ← LWT topic
```

**Rules**:
- `state` topics: retained, published by device
- `cmd` topics: NOT retained, published by controller (HA)
- `availability` topic: Last Will and Testament (LWT) — set on connect
- Never embed credentials, UUIDs, or passwords in topic names
- Avoid personal or location data that could identify occupancy patterns

---

## 5. Credential Storage on ESP32

```cpp
// ✅ DO: store credentials in NVS, loaded at boot
#include <Preferences.h>
Preferences prefs;

void loadMqttCredentials(char* user, char* pass, size_t len) {
    prefs.begin("mqtt", true);
    prefs.getString("user", user, len);
    prefs.getString("pass", pass, len);
    prefs.end();
}

// ✅ DO: provision via captive portal or BLE provisioning on first boot
// ✅ DO: store CA cert as a string constant (embedded, not secret)
static const char MQTT_CA_CERT[] = R"EOF(
-----BEGIN CERTIFICATE-----
... your ca.crt contents here ...
-----END CERTIFICATE-----
)EOF";

// ❌ NEVER:
const char* mqttUser = "esp32_living_room";  // hardcoded in source = in git history
const char* mqttPass = "hunter2";            // visible in binary flash dump
```

CA certificates are not secrets — they are public trust anchors. Embedding the CA cert in firmware is correct. Passwords are secrets — store in NVS.

---

## 6. Rate Limiting and Retained Messages

**Retained messages on command topics are dangerous**:
- If a device restarts, it will immediately receive the last retained command
- This can cause unexpected actuator activation (lights on, relays triggered)
- Use `retain=false` for all command topics; let HA resend state on reconnect

**QoS guidance**:
| QoS | Use case |
|---|---|
| 0 (fire and forget) | Sensor telemetry — occasional loss is acceptable |
| 1 (at least once) | Commands, availability — must arrive, duplicates tolerable |
| 2 (exactly once) | Rarely needed at home scale; higher overhead |

**Publish rate limits**: don't publish sensor data more than once per 10 seconds under normal conditions. Burst publishing fills broker queues and wastes bandwidth.

```cpp
static uint32_t lastPublishMs = 0;
if (millis() - lastPublishMs > 10000) {
    mqttClient.publish(topic, payload);
    lastPublishMs = millis();
}
```

---

## 7. Security Checklist

### Must-do

- [ ] `allow_anonymous false` in broker config
- [ ] Unique username and password per device
- [ ] TLS enabled (port 8883) — no plaintext 1883 in production
- [ ] ACL file: devices write only to their own namespace
- [ ] CA cert pinned in ESP32 firmware (not `setInsecure()`)
- [ ] MQTT broker bound to internal LAN interface only — no internet exposure
- [ ] No retained command topics

### Nice-to-have (but meaningful)

- [ ] IoT VLAN isolating all ESP32 devices from main LAN
- [ ] Client certificate authentication (mutual TLS)
- [ ] Broker access logs reviewed periodically
- [ ] LWT (Last Will and Testament) for each device's availability topic
- [ ] Unique MQTT client ID per device (use MAC or UUID)
- [ ] Firmware OTA updates signed and verified

---

## See Also

!!! tip "See also"
    - [Home Assistant Security Best Practices](../../best-practices/home-automation/home-assistant-security-best-practices.md) — secure the controller that manages your MQTT devices
    - [ESP32 + MQTT + Home Assistant Integration](../../tutorials/embedded/esp32-mqtt-home-assistant-integration.md) — see these security practices applied in a working firmware example
    - [Embedded Security & OTA](embedded-security-and-ota.md) — NVS credential storage, OTA signing, secure boot
    - [ESP32 Programming Architecture](esp32-programming-architecture.md) — non-blocking reconnect patterns for MQTT
