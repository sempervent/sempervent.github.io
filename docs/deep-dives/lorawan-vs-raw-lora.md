---
tags:
  - embedded
  - esp32
  - deep-dive
---

# LoRaWAN vs Raw LoRa: Protocol, Power, and Control in Long-Range IoT

*This is a Deep Dive — an analytical exploration of a design tension, not a deployment guide. For firmware configuration and wiring, see [LoRa Best Practices (SX127x)](../best-practices/esp32/lora-best-practices-sx127x.md).*

---

## Opening Thesis

The choice between raw LoRa and LoRaWAN is not a technical detail. It is an architectural commitment that determines your security model, your infrastructure dependencies, your regulatory exposure, and your operational ceiling for the next several years of a deployment. Yet it is routinely treated as an afterthought — a library selection made before the first prototype is soldered.

The tension is real and deliberate. Raw LoRa offers control: a point-to-point radio link you define entirely, encode however you wish, and operate without any intermediary. LoRaWAN offers interoperability: a standardized network stack with built-in session security, adaptive data rate, and the ability to connect to public and private network infrastructure at scale.

Neither is universally superior. The question is not which is better in the abstract but which is architecturally appropriate for a specific deployment context — and that question has a real answer if you are willing to reason through it.

---

## Historical Context

LoRa — *Long Range* — is a radio modulation technique developed by Cycleo, a French startup acquired by Semtech in 2012. The modulation itself, based on chirp spread spectrum, is Semtech's proprietary technology, implemented in the SX127x and SX126x chip families. What Semtech contributed was a physical layer with unusual properties: very low receiver sensitivity (down to −137 dBm on SF12), resilience to interference and multipath fading, and operation in license-free ISM bands. The LoRa PHY made long-range, low-power wireless communication feasible at costs that were previously impossible.

The gap Semtech did not fill was the network stack. Raw LoRa transmits and receives arbitrary payloads — it knows nothing about device identities, join procedures, acknowledgments, or encryption. This suited point-to-point industrial telemetry well but made fleet-scale IoT unwieldy.

The LoRa Alliance was formed in 2015 to address this. Member companies — including IBM, Cisco, Semtech, and dozens of operators and device makers — defined the LoRaWAN specification: a MAC-layer protocol sitting atop the LoRa PHY, adding device activation, session key management, adaptive data rate, and network routing. The first major version, LoRaWAN 1.0, shipped in 2015. Version 1.1 followed in 2017, substantially overhauling the security architecture.

The community ecosystem grew alongside the commercial one. The Things Network — a volunteer-operated, crowd-sourced LoRaWAN network — demonstrated that public LoRaWAN coverage could be built without carrier infrastructure, and inspired deployments in hundreds of cities. By the early 2020s, LoRaWAN was established in both commercial and civil infrastructure contexts: utility metering, supply chain tracking, smart city sensors, and agricultural monitoring.

Raw LoRa, meanwhile, remained the natural choice for private, controlled deployments: a farm monitoring system where the gateway is a Raspberry Pi in a barn; a pair of ESP32 nodes exchanging sensor telemetry across a field; a building automation system with a single gateway and a known device population. The absence of standardization was, in these contexts, a feature — one less dependency, one less attack surface, one less operational concern.

---

## Protocol Stack Differences

The most fundamental distinction is where the intelligence lives.

```
  Raw LoRa                    LoRaWAN
  ─────────────────────       ─────────────────────────
  Application logic           Application logic
      │                           │
  Custom framing                  │
  (your encoding)             LoRaWAN MAC
      │                       (join, ADR, ACK, fcnt)
      │                           │
  LoRa PHY                    LoRa PHY
  (Semtech SX127x)            (Semtech SX127x)
      │                           │
  Radio (ISM band)            Radio (ISM band)
```

In raw LoRa, the application developer owns everything above the physical layer. Payload encoding, framing, acknowledgment, retransmission, addressing, encryption — all of these are the application's responsibility, or are simply absent.

In LoRaWAN, the MAC layer assumes significant responsibilities:

**Device activation**: LoRaWAN devices join a network through one of two procedures:
- *Over-the-Air Activation (OTAA)*: the device sends a Join Request containing a DevEUI (device identifier) and AppEUI (application identifier), signed with an AppKey. The network server issues a Join Accept that establishes two session keys — NwkSKey (network session key) and AppSKey (application session key). These keys encrypt different parts of subsequent traffic.
- *Activation by Personalization (ABP)*: session keys and addresses are provisioned statically, bypassing the join procedure. This is convenient but eliminates forward secrecy and makes key rotation expensive.

**Device classes**: LoRaWAN defines three operational classes that balance power consumption against downlink latency:
- *Class A* (all devices): the device opens two short receive windows immediately after each uplink. The rest of the time it is unreachable. This is the lowest-power mode — the device controls all communication timing.
- *Class B*: adds scheduled receive windows synchronized to a network-wide beacon. The gateway broadcasts a beacon every 128 seconds; devices open receive slots at predictable intervals. Reduces latency at the cost of battery life.
- *Class C*: the device is continuously listening when not transmitting. Zero downlink latency. Suitable for mains-powered devices. Consumes orders of magnitude more power than Class A.

**Adaptive Data Rate (ADR)**: the network server can adjust a device's spreading factor and transmission power based on observed signal quality. ADR optimizes for spectrum efficiency — a device with a reliable link uses a lower SF (shorter airtime), leaving bandwidth for devices at the margin. This is one of LoRaWAN's most practically significant features at scale: without ADR, a fleet of devices defaults to conservative SF settings, potentially saturating the available spectrum.

---

## Spectrum and Regulatory Realities

LoRa operates in license-free ISM bands. "License-free" is often misread as "regulation-free." It is not.

The 868 MHz band used in Europe is governed by ETSI EN 300 220. The most commonly used sub-bands carry a **1% duty cycle** constraint: for every unit of time spent transmitting, the device must remain silent for 99 units. A transmission that occupies 1 second of airtime mandates 99 seconds of silence. At SF12 with 125 kHz bandwidth and a 20-byte payload, a single packet takes approximately 1.4 seconds to transmit. The minimum silence between packets is therefore just under 2.5 minutes.

This is not a guideline. It is a regulatory requirement, and exceeding it is illegal in EU jurisdictions. The rationale is resource sharing: the ISM bands are shared by medical devices, safety equipment, industrial sensors, and consumer electronics. Duty cycle limits prevent any single device class from monopolizing the spectrum.

The United States operates differently. The 915 MHz band (FCC Part 15) does not impose duty cycle limits in the same form. Instead, it requires frequency hopping across at least 50 channels, or direct-sequence spread spectrum meeting specific power spectral density limits. The practical effect on LoRa is that most US deployments use the 915 MHz FHSS implementation defined in LoRaWAN's US915 regional parameters — 64 uplink channels, 8 downlink channels, a defined channel plan.

Raw LoRa operators in the EU must implement duty cycle enforcement themselves. There is no mechanism in the SX127x to enforce duty cycle — the chip will transmit whenever you command it to. This places the compliance burden entirely on the firmware developer. LoRaWAN network stacks include duty cycle management as a standard feature; the MAC layer tracks airtime per sub-band and refuses transmission requests that would violate limits.

The airtime mathematics matter beyond compliance. At SF12, 125 kHz bandwidth, with a 10-byte payload and a Class A receive window:
- Uplink airtime: ~1.5 seconds
- At 1% duty cycle: minimum inter-packet interval of 150 seconds
- Maximum sustainable uplink rate: approximately **24 packets per hour**

This is the fundamental throughput ceiling of LoRa at maximum range settings. Applications designed around frequent updates — one per minute, or continuous streaming — are architecturally incompatible with long-range LoRa. The physics and the regulations agree on this point.

---

## Power Consumption Comparison

Power is where the differences between raw LoRa and LoRaWAN become most operationally concrete. The comparison is not simply "raw LoRa is lighter" — it depends heavily on which LoRaWAN device class is used and how the join procedure is structured.

```
  Energy cost breakdown (approximate, SX1276 at +14 dBm):
  ─────────────────────────────────────────────────────────
  Operation                  Current    Duration    Energy
  ──────────────────────     ────────   ─────────   ───────────
  TX at SF9, 20 bytes        120 mA     ~370 ms     ~44 µAh
  TX at SF12, 20 bytes       120 mA     ~1480 ms    ~178 µAh
  OTAA Join (2 exchanges)    120 mA     ~3000 ms    ~360 µAh
  Class A RX1 window         12 mA      ~50 ms      ~0.2 µAh
  Class A RX2 window         12 mA      ~50 ms      ~0.2 µAh
  Deep sleep (SX1276)        1 µA       —           —
```

For a Class A LoRaWAN device transmitting once every five minutes at SF9:
- TX cost per cycle: ~44 µAh
- RX windows per cycle: ~0.4 µAh
- Sleep current (5 min): ~0.08 µAh
- **Total per cycle: ~45 µAh**

This is functionally indistinguishable from raw LoRa at the same spreading factor and transmit power. The overhead of the LoRaWAN MAC — the FCnt counter, the MIC calculation, the ACK mechanism — adds bytes to the payload but not significant power.

The divergence appears at the join procedure. OTAA requires two radio exchanges before any application data is transmitted. On cold boot or after a network reset, the device must complete a join before operating. For a device that reboots frequently — after a firmware update, a power cycle, or a deep sleep that loses state — join overhead is not negligible. At SF9, an OTAA join consumes approximately the equivalent of 8 normal uplink cycles. For a device that joins once per day, this is irrelevant. For a device that joins on every wake cycle due to a design decision to not persist session state in RTC memory, it is a meaningful penalty.

Class B and Class C devices have fundamentally different power profiles. A Class B device consuming beacon synchronization traffic and holding periodic receive slots may draw 5–10× the energy of an equivalent Class A device, depending on the ping slot interval. A Class C device — continuously listening — is battery-incompatible for most deployments; it belongs on a mains supply.

The raw LoRa advantage in power is most pronounced for devices that transmit infrequently, can tolerate or do not require downlinks, and operate in environments where the join overhead of LoRaWAN would be paid repeatedly. For long-lived Class A deployments with infrequent state changes, the power profiles converge.

---

## Security Model

The security comparison between raw LoRa and LoRaWAN is not subtle. Raw LoRa has no built-in security at all. The physical layer transmits whatever bytes you give it, in plaintext, on a known frequency and spreading factor, receivable by any device within radio range with compatible hardware.

The practical implication: a raw LoRa deployment that does not implement application-layer encryption is broadcasting its data openly. In the context of a private farm with a known gateway and a handful of sensors, this may be an acceptable risk — the attacker would need to be within radio range, which in a rural context may be kilometers of empty field. In an urban or commercial deployment, the risk profile is different.

LoRaWAN 1.1 implements a two-layer security architecture:

```
  LoRaWAN 1.1 Security Layers
  ─────────────────────────────────────────
  Application payload
      │
  [Encrypted with AppSKey — AES-128 CTR]
      │
  LoRaWAN MAC frame
      │
  [MIC computed with NwkSKey — AES-128 CMAC]
      │
  LoRa PHY frame
```

The network session key (NwkSKey / FNwkSIntKey in 1.1) protects the MAC layer: it authenticates the frame counter and generates the Message Integrity Code that the network server verifies. The application session key (AppSKey) encrypts the application payload — critically, the network server never sees plaintext application data. Only the application server, which holds AppSKey, can decrypt the payload. This separation is meaningful: a compromised network server reveals routing metadata but not application content.

The threat model that LoRaWAN does not address is replay attacks on the join procedure in OTAA. LoRaWAN 1.0 was vulnerable to a join replay attack — a recorded Join Request could be replayed to the network server to desync the device. Version 1.1 introduced a DevNonce counter (rather than a random nonce) to prevent this. Deployments still running 1.0 devices remain exposed.

Key management is the operational challenge. LoRaWAN devices ship with AppKey — a 128-bit root key that must be provisioned securely and kept confidential. If AppKey is compromised, all historical and future sessions derived from it are compromised. For commercial IoT fleets, this demands a hardware security module (HSM) in the provisioning pipeline, a join server that handles key derivation, and a rotation strategy for compromised devices. For a hobbyist deployment, it demands at minimum that AppKey not be committed to a public git repository.

Raw LoRa deployments implementing AES encryption face a different key management problem: there is no session layer. The same symmetric key is shared between all devices and the gateway. Key rotation requires reflashing every device. A compromise of any single device reveals the key for all traffic on the network. For small, controlled deployments this may be acceptable; for any deployment where devices are physically accessible to adversaries, it is a meaningful constraint.

The attack surface comparison:

| Vector | Raw LoRa (no encryption) | Raw LoRa (AES) | LoRaWAN 1.1 |
|---|---|---|---|
| Passive eavesdropping | Trivial | Blocked | Blocked |
| Replay attack on data | Easy | Blocked by sequence | FCnt prevents |
| Replay attack on join | N/A | N/A | Blocked in 1.1 |
| Key compromise blast radius | N/A | All devices | Single device |
| Network server compromise | N/A | N/A | Routing metadata only |
| Physical device extraction | Full access | Key exposure | AppKey exposure |

---

## Infrastructure Requirements

Raw LoRa's infrastructure model is minimal by design. A point-to-point link between two SX1276-equipped devices requires nothing beyond the devices themselves. A star topology — multiple sensor nodes reporting to a single gateway — requires a gateway running receive code and, typically, software to forward data to an application (MQTT broker, database, API endpoint). The gateway can be a Raspberry Pi, an ESP32, or any Linux system with an SPI-connected LoRa module.

```
  Raw LoRa Star Topology
  ──────────────────────────────────────
  Sensor Node 1 ─────┐
  Sensor Node 2 ─────┼──► Gateway (RPi / ESP32) ──► MQTT / DB / API
  Sensor Node 3 ─────┘
  (direct radio links, no MAC layer)
```

The operational simplicity is real. The gateway has no concept of device registration. If a node transmits, the gateway receives it (assuming it is listening on the right frequency, spreading factor, and bandwidth). There are no join procedures, no network servers, no device registries. A new sensor node comes online by simply transmitting — the gateway receives its first packet without any prior configuration.

LoRaWAN's infrastructure model is more elaborate:

```
  LoRaWAN Infrastructure
  ──────────────────────────────────────────────────────────
  Device ──► Gateway(s) ──► Network Server ──► App Server
                │               │                   │
           Packet forward   Join server         Application
           (UDP/IP)         Key derivation      Decryption
                            ADR                 Business logic
                            FCnt validation
```

A LoRaWAN deployment requires at minimum: one or more gateways running packet forwarder firmware, a network server handling device authentication and MAC layer functions, and an application server receiving decrypted payload. In a self-hosted deployment, all three can run on a single machine (ChirpStack, for example, packages all components). In a managed deployment, the network server is provided by a third party (The Things Stack, Helium, AWS IoT Core for LoRaWAN).

The gateway in a LoRaWAN deployment is typically multi-channel — capable of receiving on 8 or more channels simultaneously at multiple spreading factors. This is orders of magnitude more capable than the single-channel raw LoRa gateway feasible with a single SX1276. The cost difference is significant: a single-channel LoRa gateway (ESP32 + SX1276) costs a few dollars in components. A proper 8-channel LoRaWAN gateway (SX1302-based) starts around $100–200 and rises considerably for industrial-grade hardware.

The scalability implication is direct: raw LoRa's single-channel gateway becomes a bottleneck as device count grows. With a single receive channel and spreading factor, the gateway can only receive one packet at a time — simultaneous transmissions collide. LoRaWAN's multi-channel gateway, combined with the MAC layer's capacity planning and ADR, is designed to support hundreds to thousands of devices on a single gateway.

---

## Ecosystem and Vendor Considerations

The LoRa PHY chip market is effectively controlled by Semtech, whose SX127x and SX126x families are the dominant hardware. This is a single-vendor dependency at the radio level — there is no LoRa-compatible chip from a competing vendor, because LoRa modulation is patented. The practical implication for longevity is that LoRa-based designs are dependent on Semtech's continued production and pricing of these chips. To date this has not been a meaningful concern, but it is a risk that pure-LoRa deployments carry in a way that WiFi or Bluetooth deployments do not.

Above the PHY, the ecosystem diverges sharply. Raw LoRa has a rich library ecosystem for the SX127x — RadioLib, Arduino-LoRa, Meshtastic (which builds a mesh protocol on top of raw LoRa) — but no standardization of the network layer. Each deployment invents its own framing, addressing, and (if present) security. This means raw LoRa deployments are inherently proprietary above the PHY; integration with third-party systems requires a custom protocol bridge.

LoRaWAN's standardization creates interoperability, but at a complexity cost. The LoRaWAN specification has multiple revisions (1.0, 1.0.2, 1.0.3, 1.1), and not all devices, gateways, and network servers implement the same version or the same optional features. Regional parameters add another dimension of variation. A device certified for EU868 cannot be used on US915 without reconfiguration; the channel plans, duty cycles, and data rate configurations are region-specific.

Network server options range from fully managed (The Things Stack Community, AWS IoT Core for LoRaWAN, Azure IoT Hub) to fully self-hosted open-source (ChirpStack). Managed services introduce a dependency on a third-party operator — if the operator changes pricing, terms of service, or ceases operation, migrating a fleet of registered devices to a new network server requires reflashing every device with new configuration. Self-hosted deployments avoid this dependency but require operational investment in server infrastructure, security maintenance, and backup strategies.

The Helium network introduced a cryptocurrency-incentivized public LoRaWAN network, which at various points in its operation offered broad US coverage. Its architecture introduced additional complexity and dependency layers that are worth understanding before committing to it as a network provider.

---

## Operational Trade-Off Matrix

| Dimension | Raw LoRa | LoRaWAN |
|---|---|---|
| Control over protocol | Full | Constrained by spec |
| Infrastructure complexity | Low | High |
| Gateway cost | Low (single-channel: ~$5) | Higher (8-channel: $100–200+) |
| Device count scalability | Limited by single channel | Hundreds to thousands per gateway |
| Security (default) | None | AES-128 with session key separation |
| Security (hardened) | Manual AES; same key per fleet | Per-device session keys, forward secrecy |
| Duty cycle enforcement | Manual in firmware | Built into MAC layer |
| Regulatory compliance | Manually verified | Structured by regional parameters |
| Downlink capability | Application-defined | Class A/B/C with predictable windows |
| Network interoperability | None (proprietary framing) | Broad (standard MAC, certified devices) |
| Third-party dependency | Minimal | Network server, join server |
| Ecosystem lock-in | Low above PHY | LoRa Alliance spec dependencies |
| Join overhead | None | Present on OTAA; mitigated by ABP |
| Vendor dependency at PHY | Semtech only | Semtech only (identical at radio level) |

---

## When to Choose What

The decision is not technical in isolation — it is contextual. The following is a decision framework, not a flowchart, because the factors do not combine algorithmically. They require judgment.

### Choose Raw LoRa when

**The deployment is private, controlled, and small-scale.** A farm sensor network with ten nodes and a Raspberry Pi gateway is exactly the deployment raw LoRa was designed for. You control every device, the gateway is on your property, and the "network" is the space between your barn and your fields. LoRaWAN's infrastructure overhead provides no benefit here.

**The application requires non-standard protocol behavior.** If your payload format, addressing scheme, or communication pattern does not fit LoRaWAN's uplink/downlink model — for example, a mesh topology where nodes relay for each other — raw LoRa is your starting point. Meshtastic demonstrates that sophisticated protocols can be built on the raw LoRa PHY without the LoRaWAN MAC.

**You want to minimize external dependencies for longevity.** A system built on raw LoRa and a self-hosted gateway has no dependency on any network operator, cloud service, or third-party infrastructure. It will continue to function as long as the hardware does.

**Duty cycle compliance is manageable at your transmission rate.** If you are transmitting one packet every five or ten minutes, duty cycle enforcement is a few lines of firmware. If you are transmitting dozens of packets per minute, you have already left the operating envelope of LoRa regardless of protocol.

### Choose LoRaWAN when

**The deployment involves a fleet of devices across a geographic area.** LoRaWAN's multi-channel gateway and MAC layer capacity management are designed for this. Raw LoRa's single-channel model does not scale beyond a handful of devices without collisions.

**Integration with public or third-party network infrastructure is required.** City-level deployments, utility monitoring, and supply chain tracking often require connectivity through infrastructure you do not control. LoRaWAN provides a standard interface to these networks.

**Regulatory compliance must be demonstrably built-in.** In commercial or regulated deployments, auditors may require documentation that duty cycle compliance is enforced. LoRaWAN's MAC layer provides this structurally; raw LoRa requires custom implementation and verification.

**Per-device security and key management are requirements.** A deployment where physical device extraction is a plausible threat — outdoor sensors, unattended infrastructure — benefits from LoRaWAN's per-device session keys. A compromised device leaks one device's keys, not the fleet's.

**You expect device count to grow over the deployment's lifetime.** Adding a device to a raw LoRa network is trivial today but may require gateway redesign at scale. Adding a device to a LoRaWAN network requires registration in the join server and provisioning of DevEUI/AppKey — more process upfront, more headroom later.

---

## Hybrid Architectures

The binary framing of raw LoRa vs LoRaWAN obscures a practical middle path: use raw LoRa at the edge while adopting LoRaWAN-inspired patterns at the gateway.

```
  Hybrid: Raw LoRa Edge + MQTT Gateway Bridge
  ─────────────────────────────────────────────────────────────
  Sensor Node ────────────────────────────┐
  (raw LoRa, custom framing, AES-128)     │
                                    Gateway (RPi)
  Sensor Node ────────────────────────────┤  ├──► MQTT Broker
  (raw LoRa)                              │  └──► InfluxDB / TimescaleDB
                                          │
  Sensor Node ────────────────────────────┘
```

In this architecture, nodes transmit raw LoRa with application-layer AES encryption. The gateway — running on commodity hardware — decrypts, validates, and forwards to a local MQTT broker. Home Assistant, Node-RED, or any MQTT subscriber can then consume the data. The external network dependency is eliminated; the security model is explicit and controlled.

This pattern trades LoRaWAN's MAC layer features — ADR, Class B/C downlinks, standard network server integration — for complete protocol ownership. It is appropriate when the device count is moderate, the deployment environment is private, and the operational team has the capability to maintain a custom gateway stack.

The inverse hybrid also exists: a LoRaWAN network where the application server bridges to a local MQTT broker or proprietary backend, rather than a managed cloud service. ChirpStack's architecture supports this directly — the application server can publish to any MQTT broker, enabling fully self-hosted LoRaWAN without cloud dependency while retaining the MAC layer's capacity management and security features.

---

## Final Reflection

The persistence of the "LoRaWAN vs raw LoRa" debate in embedded engineering circles reflects something genuine: the two are not simply different implementations of the same goal. They represent different theories of how networked embedded systems should be structured.

Raw LoRa is an argument for minimal infrastructure and maximal control. It says: the complexity of a network stack should be incurred only when its benefits — standardization, interoperability, managed security — are concretely needed, not speculatively anticipated. Build the simplest thing that works, and add complexity only when the deployment demands it.

LoRaWAN is an argument for principled standardization. It says: the decisions that are hard to reverse — security model, addressing, key management, compliance enforcement — should be made once by people who have thought about them carefully, embodied in a specification, and implemented in tested libraries. The cost of that standardization is complexity and constraint; the benefit is not having to rediscover the failure modes of ad-hoc security and spectrum management under operational pressure.

Neither argument is wrong. The tension between them is real and productive. A deployment that starts on raw LoRa and grows beyond its operational ceiling learns something concrete about when LoRaWAN's infrastructure is worth its cost. A deployment that starts on LoRaWAN for a three-node private sensor network learns something equally concrete about the price of unnecessary standardization.

The right choice is the one that fits the deployment's actual requirements — not its anticipated ones, not its aspirational ones. That requires honesty about scale, about security posture, about operational capability, and about the willingness to manage complexity that is not yet necessary.

The hardware underneath is the same. The radio waves are the same. The choice is about what you build around them.

---

*For firmware-level guidance on configuring SX127x modules with ESP32, implementing duty cycle enforcement, and structuring antenna layouts, see [LoRa Best Practices (SX127x)](../best-practices/esp32/lora-best-practices-sx127x.md).*

!!! tip "See also"
    - [LoRa Best Practices (SX127x)](../best-practices/esp32/lora-best-practices-sx127x.md) — spreading factor selection, duty cycle enforcement, antenna layout, firmware state machines
    - [ESP32 Programming Architecture](../best-practices/esp32/esp32-programming-architecture.md) — non-blocking firmware patterns applicable to LoRa node firmware
    - [Power Management & Deep Sleep](../best-practices/esp32/power-management-and-deep-sleep.md) — battery lifetime budgeting for Class A LoRa nodes
    - [Embedded Security & OTA](../best-practices/esp32/embedded-security-and-ota.md) — NVS key storage, OTA strategy for LoRa gateway nodes
