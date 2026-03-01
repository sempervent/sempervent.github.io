---
tags:
  - deep-dive
  - iot
  - networking
---

# MQTT vs HTTP in IoT Systems: Latency, Power, and Network Realities

*This deep dive examines the protocol-level trade-offs between MQTT and HTTP in IoT deployments. For an implementation example using MQTT on an ESP32, see the [ESP32 tutorials](../tutorials/esp32/index.md).*

**Themes:** Embedded · IoT · Architecture

---

## The Protocol Is Not the Afterthought

In desktop and server software development, the choice of communication protocol is frequently a secondary concern — HTTP is the default, and deviations from it require explicit justification. In IoT systems design, this assumption is often carried in uncritically, and it regularly produces systems that are power-inefficient, fragile under poor network conditions, or operationally opaque at scale.

The properties of MQTT and HTTP differ in ways that are not cosmetic. They reflect different design premises about network reliability, connection state, power budgets, and data multiplicity. Understanding these premises clarifies when each protocol is the correct choice — and when each is being misapplied.

---

## Connection Models: Stateful vs Stateless

The most fundamental difference between MQTT and HTTP is their connection models.

**HTTP** is stateless by protocol design. Each HTTP request is independent: a connection is established, a request is sent, a response is returned, and the connection is closed (or kept alive briefly for pipelining). The server retains no per-client state between requests. Authentication, session context, and prior message history must be reconstructed from each request's headers and body.

```
  HTTP request-response model:
  ────────────────────────────────────────────────────────────────────
  Device                                    Server
    │                                          │
    │──── TCP SYN ─────────────────────────────►│  TCP handshake
    │◄─── SYN-ACK ─────────────────────────────│
    │──── ACK ─────────────────────────────────►│
    │                                          │
    │──── TLS ClientHello ──────────────────────►│  TLS handshake
    │◄─── ServerHello, Certificate ────────────│  (adds 1-2 RTTs)
    │──── Finished ────────────────────────────►│
    │                                          │
    │──── POST /sensor/data ────────────────────►│  Request
    │◄─── 200 OK ──────────────────────────────│  Response
    │                                          │
    │──── TCP FIN ─────────────────────────────►│  Connection teardown
```

For a constrained device sending a temperature reading every 60 seconds, this model requires establishing and tearing down a TCP connection (3-way handshake) and a TLS session (1-2 additional round trips) for each measurement. On a cellular IoT connection with 200 ms RTT, the overhead before any application data is transmitted may be 600–1000 ms and consume orders of magnitude more energy than transmitting the sensor value itself.

**MQTT** is stateful by design. A device establishes a persistent TCP connection to a broker, authenticates once, and then publishes or subscribes to topics over the life of that connection. The broker maintains per-client session state — which topics the client subscribes to, any queued messages for a disconnected client with a persistent session — across the connection lifetime.

```
  MQTT connection model:
  ────────────────────────────────────────────────────────────────────
  Device                        Broker
    │                              │
    │── CONNECT ──────────────────►│  once at startup
    │◄─ CONNACK ───────────────────│
    │                              │
    │── PUBLISH temp/sensor1 ─────►│  30-byte packet
    │                              │  (no response required)
    │── PUBLISH temp/sensor1 ─────►│
    │── PUBLISH temp/sensor1 ─────►│  ... repeating indefinitely
    │                              │
    │── DISCONNECT ───────────────►│  optional clean shutdown
```

After the initial connection, each MQTT publish is a small packet (minimum 2 bytes fixed header + topic + payload) sent over the existing connection. The per-message overhead is the packet itself, not the connection establishment. For a device sending 60 readings per hour, MQTT's overhead is one connection establishment amortized across all readings.

---

## Power Consumption Differences

Power consumption in battery-operated or energy-harvesting IoT devices is determined primarily by radio-on time — the duration the radio transceiver is active. Connection establishment requires the radio to remain on while waiting for responses. MQTT's persistent connection keeps the radio on continuously (unless using MQTT over DTLS with sleep cycles), but eliminates per-message connection overhead.

The comparison depends critically on the measurement interval:

| Measurement interval | Recommended protocol | Rationale |
|---|---|---|
| < 5 minutes | MQTT persistent connection | Radio on < reconnection overhead |
| 5–60 minutes | MQTT with clean session or HTTP | Trade-off depends on sleep depth |
| > 60 minutes | HTTP or MQTT with deep sleep | Connection per wake cycle |
| Event-driven (infrequent) | HTTP | No persistent connection needed |

For devices that deep-sleep between measurements and wake only to transmit, the HTTP model — establish connection, send data, disconnect, sleep — may be more power-efficient because no connection is maintained during sleep. The ESP32 deep sleep current is around 10 µA; maintaining a TCP/TLS connection requires keeping the radio active, consuming 80–400 mA depending on signal strength and protocol.

The MQTT pattern for deep-sleeping devices typically uses short LWT (Last Will and Testament) sessions with clean session flags, effectively approximating the HTTP model with MQTT framing. In this configuration, the power profiles converge.

---

## TLS Handshake Costs

TLS is not optional for production IoT systems. Both MQTT and HTTP can run over TLS, but the TLS handshake cost differs between them in a practically important way.

A TLS 1.3 handshake over a cellular IoT link with 300 ms RTT takes approximately 600–900 ms and 10–20 KB of data exchange. For HTTP, this cost is incurred on every new connection. For MQTT, it is incurred once at connection establishment and not repeated until reconnection.

TLS session resumption (both TLS 1.2 session tickets and TLS 1.3 PSK) reduces reconnection cost for HTTP clients, but requires the server to support and cache session state. In practice, many IoT backend HTTP endpoints do not implement session resumption effectively, and constrained devices may not have sufficient memory for session ticket storage.

The certificate chain validation step is a particular burden on constrained devices. Validating a certificate chain requires the device to have current time (which requires NTP synchronization, itself a network operation), a root CA certificate store, and sufficient CPU and memory for RSA or ECDSA verification. The mbedTLS footprint on an ESP32 for full TLS is approximately 100 KB RAM plus flash for the certificate store.

MQTT over TLS (MQTTS, port 8883) has the same TLS costs per connection. The advantage is amortization: one TLS handshake per device uptime rather than one per measurement.

---

## Broker Centralization: Architectural Implications

MQTT's design requires a broker — a central message routing system (Mosquitto, EMQX, AWS IoT Core, HiveMQ). All messages flow through the broker; devices do not communicate directly. This creates a central point of control, a central point of failure, and a natural observability point.

```
  MQTT publish-subscribe topology:
  ─────────────────────────────────────────────────────────────
                     [MQTT Broker]
                    /      │      \
                   /       │       \
  [Device A]     /    [Device C]    \     [App Server]
  Pub: sensors/A ──── Sub: sensors/# ──── Sub: alerts/#
  Sub: cmd/A ──── Pub: cmd/C ────────── Pub: alerts/device
  [Device B]
  Pub: sensors/B
```

The broker handles:
- **Fan-out**: a single published message can be delivered to arbitrarily many subscribers without the publisher knowing how many subscribers exist
- **QoS**: MQTT's three Quality of Service levels (0: fire-and-forget, 1: at-least-once, 2: exactly-once) are implemented by the broker with acknowledgment and queuing
- **Persistence**: QoS 1/2 messages for offline clients with persistent sessions are stored by the broker until the client reconnects
- **Access control**: topic-based ACLs defining which clients can publish or subscribe to which topic patterns

The broker centralization creates an availability dependency: if the broker is unreachable, devices cannot communicate with backend systems. This is partially mitigated by MQTT's clean/persistent session design — devices reconnect automatically and resume where they left off — but the broker is still a single point of failure unless deployed in a clustered, highly-available configuration.

HTTP's point-to-point model has no inherent broker; devices communicate directly with backend API endpoints. Load balancing, message routing, and fan-out are the responsibility of the backend service layer. This is more complex to implement but avoids a dedicated messaging infrastructure dependency.

---

## Observability Differences

An MQTT broker provides natural observability through its topic structure. A broker's `$SYS` topics (supported by Mosquitto) expose real-time metrics: connected clients, message throughput, byte counts, and subscription counts. A monitoring dashboard can subscribe to `sensors/#` and receive all sensor data from all devices without modifying any device firmware.

```
  MQTT observability through subscription:
  ───────────────────────────────────────────────────────────
  Monitor subscribes to:
    sensors/#        → all sensor readings, all devices
    $SYS/#           → broker metrics
    cmd/#            → all commands issued to devices
    lwt/#            → last will messages (device disconnects)

  No device-side instrumentation changes required.
  The broker is the observability plane.
```

HTTP-based IoT observability requires instrumentation at the backend API layer: logging every request, tracking device IDs, correlating requests to devices. This is not fundamentally harder, but it requires deliberate design. The natural structure of MQTT topics — hierarchical, device-identified, type-segregated — provides a built-in taxonomy for observability that HTTP URLs can replicate but don't provide automatically.

---

## Security Trade-offs

Both protocols can be secured equivalently with TLS and proper authentication, but the attack surface and threat model differ.

**MQTT broker authentication**: clients authenticate with username/password or X.509 client certificates at connection time. The broker enforces topic ACLs. A single compromised credential can publish to topics it is authorized for and subscribe to topics it is authorized for. If ACLs are not correctly configured, a device can subscribe to commands intended for other devices.

**HTTP authentication**: each request carries credentials (Bearer token, API key, mTLS) and is independently authenticated. There is no persistent session to compromise. A stolen credential is useful for exactly one request before token expiry, if short-lived tokens are used.

The MQTT persistent connection model means a compromised device maintains an authorized connection until it disconnects or is explicitly disconnected. The broker must provide mechanisms to forcibly disconnect clients and invalidate sessions (most enterprise MQTT brokers do; Mosquitto with management API can). HTTP's stateless model means credential rotation is immediately effective for subsequent requests.

**MQTT's LWT (Last Will and Testament)** is a security-relevant feature with operational value: devices register a message that the broker publishes on their behalf if they disconnect uncleanly. This enables detection of device failures or attacks that cause unclean disconnection.

---

## Decision Framework

**Choose MQTT when**:
- Devices send many messages with high frequency (>1 per minute)
- The network is unreliable and QoS message delivery guarantees matter
- Multiple backend services need to receive the same device data (pub-sub fan-out)
- Devices need to receive commands from backends without polling
- The operational team can manage or use a managed MQTT broker

**Choose HTTP when**:
- Devices send infrequent messages (hourly, daily)
- Devices are battery-operated with deep sleep cycles between transmissions
- The backend is an existing REST API without MQTT infrastructure
- The device payload is large (HTTP handles arbitrary body sizes without frame limits)
- The team lacks MQTT operational expertise and broker management overhead is prohibitive

**The practical answer for most production IoT systems is not a binary choice**: cloud IoT platforms (AWS IoT Core, Azure IoT Hub, Google Cloud IoT) accept both MQTT and HTTP from devices and route messages to the same backend topics/streams. Devices can use MQTT when frequently connected and HTTP when waking from deep sleep, with the same backend consuming both.

!!! tip "See also"
    - [ESP32 Section Architecture](../adr/0005-esp32-section-architecture.md) — the ADR explaining how this site's ESP32 content is structured
    - [ESP32 Tutorials](../tutorials/esp32/index.md) — implementation examples using MQTT with an ESP32
