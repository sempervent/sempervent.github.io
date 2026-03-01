---
tags:
  - security
  - embedded
---

# Home Assistant Security Best Practices

Home Assistant is powerful precisely because it can control locks, alarms, climate, and lighting. That power makes a misconfigured HA instance a meaningful target. These practices apply whether you're running HA OS, HA Supervised, or HA Container.

---

## 1. Default Risks

| Risk | Default state | Impact |
|---|---|---|
| HA accessible on port 8123 | Open on LAN | Any LAN device can attempt login |
| Admin password | Set at first run, often weak | Full system access |
| No MFA | Not enabled by default | Credential stuffing works |
| Unused integrations | Active but forgotten | Expanded attack surface |
| Port forwarded to internet | Done by users for remote access | Direct public exposure |

The single most common mistake: forwarding port 8123 directly to the internet without TLS or authentication hardening. Do not do this. Use a VPN or authenticated reverse proxy instead.

---

## 2. Access Control

### Password and MFA

- Use a **long random passphrase** (≥20 characters) for the admin account — not your WiFi password, not your email password
- Enable **Time-based One-Time Password (TOTP) MFA** for every human user account:
  - Settings → People → [User] → Enable MFA
- Create **separate named accounts** for each human who accesses HA — no shared admin credentials
- Create a **dedicated service account** for automations that need API access (not the admin account)

### Account hygiene

```yaml
# In configuration.yaml (HA 2024+ uses UI for this, but YAML still works)
# Disable legacy API password auth (deprecated, insecure)
homeassistant:
  auth_providers:
    - type: homeassistant   # UI-managed accounts
    # Do NOT add type: legacy_api_password
```

- Review People → Users periodically; remove accounts for users who no longer need access
- Long-lived access tokens for integrations: use minimal-scope tokens; rotate annually

---

## 3. Network Architecture

```
  Internet
      │
      │  (VPN preferred; reverse proxy if VPN not possible)
      ▼
  ┌─────────────────────────────┐
  │  Nginx / Traefik / Caddy   │  ← TLS termination, auth headers
  │  (DMZ or proxy host)        │
  └──────────────┬──────────────┘
                 │  Internal only (no internet route)
                 ▼
  ┌─────────────────────────────┐
  │   Home Assistant            │  ← Never directly internet-facing
  │   (port 8123, LAN only)     │
  └──────────────┬──────────────┘
                 │  MQTT over TLS (port 8883, internal VLAN only)
                 ▼
  ┌─────────────────────────────┐
  │   MQTT Broker (Mosquitto)   │  ← No internet route; no port forward
  │   (LAN/IoT VLAN only)       │
  └──────────────┬──────────────┘
                 │
         ESP32 devices (IoT VLAN)
```

**Key principle**: HA and the MQTT broker should have no direct internet route. Remote access goes through VPN (WireGuard, Tailscale) or an authenticated, TLS-terminated reverse proxy.

**VLAN segmentation** (recommended):
- Main LAN: computers, phones, NAS
- IoT VLAN: ESP32s, smart bulbs, cameras, sensors
- Firewall rule: IoT VLAN → HA/MQTT only; IoT VLAN → internet blocked

---

## 4. TLS and Reverse Proxy

**Option A — Caddy (simplest)**:
```
# Caddyfile
ha.yourdomain.com {
    reverse_proxy homeassistant:8123
    # Caddy handles Let's Encrypt automatically
}
```

**Option B — Nginx**:
```nginx
server {
    listen 443 ssl;
    server_name ha.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/ha.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ha.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass         http://homeassistant:8123;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        # WebSocket support (required for HA)
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
    }
}
# Redirect HTTP → HTTPS
server { listen 80; return 301 https://$host$request_uri; }
```

Configure HA to trust the proxy in `configuration.yaml`:
```yaml
http:
  use_x_forwarded_for: true
  trusted_proxies:
    - 192.168.1.0/24   # your proxy's subnet
  ip_ban_enabled: true
  login_attempts_threshold: 5
```

**HSTS** (`Strict-Transport-Security`) prevents downgrade attacks. Set `max-age=31536000` (1 year) after verifying HTTPS works correctly.

---

## 5. Backups

Losing HA configuration means losing all automations, dashboards, integrations, and entity history. Back up before every major change.

**HA OS / Supervised — built-in snapshots**:
```
Settings → System → Backups → Create Backup
```

Automate with an automation:
```yaml
automation:
  - alias: "Weekly HA Backup"
    trigger:
      - platform: time
        at: "02:00:00"
    condition:
      - condition: time
        weekday: [sun]
    action:
      - service: backup.create
```

**Off-device backup** (critical — on-device backup doesn't survive hardware failure):
- Copy snapshots to NAS via Samba add-on, or use the [Google Drive Backup](https://github.com/sabeechen/hassio-google-drive-backup) add-on
- Or: `rsync` snapshots from HA host to an external machine nightly

**What a backup contains**: all configuration, add-ons, database, automations, entity registry. Does **not** contain credentials from the OS keyring or secrets stored outside HA's `/config` directory.

---

## 6. Add-on Security

HA OS add-ons run as containers with varying privilege levels:

- **Review add-on permissions before installing**: `host_network: true` means the add-on shares the host network stack — be cautious
- **Only install add-ons from the Official and Community add-on repositories** — avoid third-party repos unless you have reviewed their source
- **Disable unused add-ons**: fewer running services = smaller attack surface
- **Check for updates**: outdated add-ons are a common vector for known CVEs

Red flags in add-on configuration:
```yaml
# These require extra scrutiny:
host_network: true   # full network access
privileged: true     # root-equivalent host access
full_access: true    # unrestricted hardware + filesystem
```

---

## 7. Logging and Monitoring

**Login attempt monitoring** — HA logs failed logins; enable IP banning:
```yaml
http:
  ip_ban_enabled: true
  login_attempts_threshold: 5  # ban after 5 failures
```

Bans are logged to `/config/ip_bans.yaml`. Review periodically.

**MQTT activity** — suspicious MQTT patterns to watch for:
- A device publishing to topics outside its ACL scope (broker will reject, but log)
- Unusually high publish rate from a single device
- Connection attempts with unknown client IDs

Enable Mosquitto logging:
```conf
# mosquitto.conf
log_dest file /var/log/mosquitto/mosquitto.log
log_type all   # or: error warning notice information
```

Review with: `grep "Connection Refused" /var/log/mosquitto/mosquitto.log`

**HA notification on login** (useful early warning):
```yaml
automation:
  - alias: "Alert on HA Login"
    trigger:
      - platform: event
        event_type: call_service
        event_data:
          domain: auth
          service: login
    action:
      - service: notify.mobile_app_your_phone
        data:
          message: "HA login from {{ trigger.event.data.context.user_id }}"
```

---

## 8. Zero-Trust Mindset for IoT

**Assume every IoT device is compromised.** Design your network so a compromised ESP32 cannot:
- Access your main LAN (computers, NAS, phone)
- Reach the internet directly
- Reach other IoT devices (disable inter-VLAN communication within IoT segment)
- Authenticate to HA with anything stronger than MQTT credentials

A compromised ESP32 should be able to: publish to its own MQTT topics. That's it.

**Practical implementation**:
1. IoT VLAN → MQTT broker: allowed (port 8883 only)
2. IoT VLAN → internet: blocked
3. IoT VLAN → main LAN: blocked
4. Main LAN → IoT VLAN: blocked (HA talks to MQTT broker, not directly to devices)
5. MQTT broker → HA: allowed (HA subscribes to broker)

If a device is compromised, it can spam its own MQTT topics. ACLs prevent it from writing to other devices' topics. HA's MQTT integration handles malformed payloads gracefully (at worst, a sensor shows an incorrect value).

---

## See Also

!!! tip "See also"
    - [MQTT Security Best Practices](../esp32/mqtt-security-best-practices.md) — broker hardening, ACLs, TLS, topic design
    - [ESP32 + MQTT + Home Assistant Integration](../../tutorials/embedded/esp32-mqtt-home-assistant-integration.md) — end-to-end secure integration tutorial
    - [IAM & RBAC Governance](../security/iam-rbac-abac-governance.md) — access control principles that apply equally to HA user management
    - [Docker & Compose Best Practices](../docker-infrastructure/docker-and-compose.md) — if running HA Container or Mosquitto in Docker
