---
tags:
  - embedded
  - esp32
  - security
---

# Embedded Security & OTA Strategy

Firmware security is not optional. An IoT device on your LAN is a pivot point for every other device on that network. These principles are proportionate — not paranoid.

---

## Why Hardcoding Secrets Is Bad

```cpp
// ❌ Never do this
const char* ssid     = "MyHomeWifi";
const char* password = "hunter2";
const char* apiKey   = "sk-prod-abc123";
```

Hardcoded secrets in source:
- Are committed to git (often public repos)
- Are readable from a flash dump of the device
- Cannot be rotated without a firmware flash
- Are present in every compiled binary

---

## NVS: Non-Volatile Storage for Secrets

The ESP32 has a dedicated NVS (Non-Volatile Storage) partition in flash — a key-value store that persists across reboots and is separate from firmware.

```cpp
#include <Preferences.h>

Preferences prefs;

// Write once (e.g., provisioning mode)
void provisionDevice(const char* ssid, const char* pass) {
    prefs.begin("app", false);  // namespace, read-write
    prefs.putString("wifi_ssid", ssid);
    prefs.putString("wifi_pass", pass);
    prefs.end();
}

// Read at boot
void loadConfig() {
    prefs.begin("app", true);  // read-only
    String ssid = prefs.getString("wifi_ssid", "");
    String pass = prefs.getString("wifi_pass", "");
    prefs.end();
    WiFi.begin(ssid.c_str(), pass.c_str());
}
```

Provisioning strategies:
- **Captive portal**: first boot creates an AP; user connects and enters credentials via browser
- **BLE provisioning**: ESP-IDF includes `wifi_provisioning` component with BLE or SoftAP
- **Serial provisioning**: development mode — accept credentials over UART, write to NVS

---

## Secure Boot

Secure boot ensures only firmware signed with your private key can run on the device.

**ESP32 secure boot v2 (recommended)**:
1. Generate an RSA key pair: `espsecure.py generate_signing_key --version 2 secure_boot_signing_key.pem`
2. Enable `CONFIG_SECURE_BOOT_V2_ENABLED` in `sdkconfig`
3. The bootloader verifies each firmware partition signature at boot
4. Once burned, the signing key's public key hash is written to eFuses — **this is irreversible**

**Implication**: if you lose your signing key, you cannot update the device. Store keys in a hardware security module or encrypted backup.

---

## Flash Encryption

Flash encryption scrambles the contents of flash so a physical dump of the chip cannot be read.

- Enable `CONFIG_FLASH_ENCRYPTION_ENABLED`
- The encryption key is generated on-device and stored in eFuses (never leaves the chip)
- Encrypted flash = NVS credentials are protected from physical extraction
- **Downside**: OTA binaries must be correctly signed; plaintext flashing via UART is disabled

For hobbyist projects: secure boot alone is often sufficient. Flash encryption is for commercial devices.

---

## OTA Update Strategy

OTA (Over-the-Air) updates are required for any device you don't want to physically touch for firmware updates.

**Partition table for OTA**:
```
# partitions.csv
nvs,      data, nvs,  0x9000,  0x5000,
otadata,  data, ota,  0xe000,  0x2000,
app0,     app,  ota_0, 0x10000, 0x200000,
app1,     app,  ota_1, 0x210000,0x200000,
```

Two app partitions (`app0`, `app1`) enable rollback: if the new firmware fails to boot, the bootloader reverts to the last known good partition.

**Update flow**:
```cpp
#include <Update.h>

// Arduino: fetch binary from HTTPS server
void performOTA(const char* url) {
    WiFiClientSecure client;
    client.setCACert(server_root_ca);  // pin the CA cert
    HTTPClient http;
    http.begin(client, url);
    int code = http.GET();
    if (code == 200) {
        int len = http.getSize();
        if (Update.begin(len)) {
            Update.writeStream(http.getStream());
            if (Update.end(true)) {
                ESP.restart();
            }
        }
    }
}
```

**Never**:
- Fetch firmware over plain HTTP
- Skip signature verification on downloaded binary
- Use `setCACert(nullptr)` (disables verification)

---

## Firmware Signing

If using ESP-IDF's update mechanism, sign your firmware binary:

```bash
espsecure.py sign_data \
  --version 2 \
  --keyfile secure_boot_signing_key.pem \
  --output firmware-signed.bin \
  firmware.bin
```

The bootloader verifies this signature before executing. On failure, it falls back to the previous partition.

---

## WiFi Credential Hygiene

- Never hardcode SSID/password
- Use a dedicated IoT VLAN — isolate ESP32 devices from your main LAN
- Set a firewall rule: IoT VLAN → internet only; no lateral movement to other VLANs
- Rotate WiFi passwords periodically; re-provision devices
- Consider WPA3 if your access point supports it

---

## Local Network Attack Surface

Even on a trusted LAN, assume the device can be reached by other devices:

- **TLS for all outbound HTTP**: pin the CA cert or use certificate pinning
- **No open debug endpoints**: disable serial logging in production firmware
- **No telnet/HTTP debug server** exposed on port 80 in production
- **Minimal services**: if the device doesn't need a web server, don't run one
- **Disable JTAG and download mode** in production using eFuse burn:
  ```bash
  espefuse.py burn_efuse JTAG_DISABLE
  ```

---

## See Also

!!! tip "See also"
    - [ESP32 Programming Architecture](esp32-programming-architecture.md) — NVS usage fits into config loading at boot
    - [IAM & RBAC Governance](../security/iam-rbac-abac-governance.md) — access control at the network and service layer
    - [Docker & Compose Best Practices](../docker-infrastructure/docker-and-compose.md) — if running an OTA server
