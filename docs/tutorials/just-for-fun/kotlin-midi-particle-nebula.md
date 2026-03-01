---
tags:
  - kotlin
  - midi
  - generative-art
---

# MIDI-Driven Particle Nebula (Kotlin + LWJGL + OpenGL)

**Objective**: Turn live MIDI input into a real-time particle simulation rendered in OpenGL. Every note, velocity, and pedal movement reshapes the cosmos on screen.

---

## What You're Building

- A real-time particle system rendered via LWJGL (OpenGL 3.3+)
- MIDI events from any connected device drive simulation parameters live
- Note-on → burst of particles from a pitch-mapped emitter position
- Velocity → particle lifetime and peak brightness
- Pitch → hue of the emission and horizontal emitter offset
- Aftertouch → turbulence strength (particles wobble)
- Sustain pedal → gravity well pulls particles to center
- Screenshot export to PNG on keypress

---

## Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │  javax.sound.midi                                   │
  │  MidiDevice → Receiver → MidiEventQueue (ConcurrentLinkedQueue)
  └─────────────────────────┬───────────────────────────┘
                            │ poll per frame
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  EventMapper                                        │
  │  note / velocity / pitch / aftertouch → SimParams   │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  ParticleSystem (CPU update v1)                     │
  │  FloatArray pool → update positions + lifetimes     │
  │  upload to VBO → draw GL_POINTS with custom shader  │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  GLFW Window + OpenGL 3.3 core profile              │
  │  Vertex shader: position + size from lifetime       │
  │  Fragment shader: soft circle + hue from attribute  │
  └─────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| JDK | 17+ (LTS) |
| Kotlin | 1.9+ |
| Gradle | 8.x (wrapper) |
| LWJGL | 3.3.3 |
| OS | Linux/macOS/Windows — GLFW works on all three |

**Linux note**: GLFW needs a display server. On headless servers, use Xvfb: `Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99`

---

## Project Layout

```
midi-nebula/
├── build.gradle.kts
├── settings.gradle.kts
└── src/main/kotlin/nebula/
    ├── Main.kt             # GLFW window init + main loop
    ├── MidiInput.kt        # Device enumeration + event queue
    ├── EventMapper.kt      # MIDI → SimParams
    ├── ParticleSystem.kt   # Particle pool + CPU update
    ├── Renderer.kt         # VAO/VBO/shader management
    └── Shaders.kt          # Inline GLSL strings
```

---

## Core Concepts

- **Particle pool**: pre-allocated `FloatArray` of fixed capacity (e.g. 100 000 particles × 8 floats: x, y, vx, vy, life, maxLife, hue, size). No per-frame allocations.
- **VBO upload**: each frame, copy the live portion of the pool into a mapped or `glBufferSubData` VBO. `GL_STREAM_DRAW` hint.
- **MIDI receiver**: `javax.sound.midi.Receiver.send()` is called on a MIDI thread. Use a `ConcurrentLinkedQueue` to decouple it from the render thread.
- **SimParams**: a mutable data class holding current turbulence, gravity, emission rate, base hue — modified by the event mapper, read by the particle update.
- **Soft-circle fragment shader**: discard fragments where `length(gl_PointCoord - 0.5) > 0.5`; apply a radial falloff for a glowing dot.
- **GPU upgrade path**: replace CPU update with a compute shader or transform feedback to keep particle state on the GPU (Section 8).

---

## Implementation

### Step 1 — build.gradle.kts

```kotlin
plugins {
    kotlin("jvm") version "1.9.23"
    application
}

val lwjglVersion = "3.3.3"
val lwjglNatives = "natives-linux"  // change for macos / windows

repositories { mavenCentral() }

dependencies {
    implementation(kotlin("stdlib"))
    listOf("", "-glfw", "-opengl", "-stb").forEach { mod ->
        implementation("org.lwjgl:lwjgl$mod:$lwjglVersion")
        runtimeOnly("org.lwjgl:lwjgl$mod:$lwjglVersion:$lwjglNatives")
    }
}

application { mainClass.set("nebula.MainKt") }
```

### Step 2 — MIDI device enumeration and receiver

```kotlin
// MidiInput.kt
package nebula

import java.util.concurrent.ConcurrentLinkedQueue
import javax.sound.midi.*

data class MidiEvent(val status: Int, val data1: Int, val data2: Int)

class MidiInput(deviceIndex: Int = 0) : AutoCloseable {
    val queue = ConcurrentLinkedQueue<MidiEvent>()
    private val device: MidiDevice

    init {
        val infos = MidiSystem.getMidiDeviceInfo()
        println("Available MIDI devices:")
        infos.forEachIndexed { i, info -> println("  $i: ${info.name}") }

        device = MidiSystem.getMidiDevice(infos[deviceIndex])
        device.open()
        device.transmitter.receiver = Receiver { msg, _ ->
            if (msg is ShortMessage)
                queue.add(MidiEvent(msg.command, msg.data1, msg.data2))
        }
    }

    override fun close() = device.close()
}
```

### Step 3 — SimParams and EventMapper

```kotlin
// EventMapper.kt
package nebula

data class SimParams(
    var turbulence: Float = 0.0f,
    var gravityWell: Float = 0.0f,
    var baseHue: Float = 0.0f,
    var emitterX: Float = 0.0f,
    var emitLifetime: Float = 3.0f,
    var emitBrightness: Float = 1.0f,
    var pendingBurst: Int = 0,
)

fun mapMidi(event: MidiEvent, params: SimParams) {
    val NOTE_ON  = 0x90
    val AFTERTOUCH = 0xA0
    val CC = 0xB0
    val SUSTAIN_CC = 64

    when (event.status and 0xF0) {
        NOTE_ON -> if (event.data2 > 0) {
            params.baseHue      = event.data1 / 127f          // pitch → hue
            params.emitterX     = (event.data1 - 64) / 64f    // center-normalized
            params.emitLifetime = 1.5f + event.data2 / 127f * 4f
            params.emitBrightness = event.data2 / 127f
            params.pendingBurst += 50 + event.data2 / 2
        }
        AFTERTOUCH -> params.turbulence = event.data2 / 127f
        CC -> if (event.data1 == SUSTAIN_CC)
            params.gravityWell = if (event.data2 >= 64) 1.0f else 0.0f
    }
}
```

### Step 4 — Particle pool update (CPU v1)

```kotlin
// ParticleSystem.kt — partial snippet
package nebula

import kotlin.math.*
import kotlin.random.Random

const val MAX_PARTICLES = 100_000
const val STRIDE = 8  // x y vx vy life maxLife hue size

class ParticleSystem {
    val data = FloatArray(MAX_PARTICLES * STRIDE)
    var liveCount = 0

    fun emit(params: SimParams, count: Int) {
        repeat(count) {
            if (liveCount >= MAX_PARTICLES) return
            val i = liveCount * STRIDE
            data[i + 0] = params.emitterX + Random.nextFloat() * 0.1f - 0.05f  // x
            data[i + 1] = Random.nextFloat() * 0.2f - 0.1f                      // y
            data[i + 2] = Random.nextFloat() * 0.02f - 0.01f                    // vx
            data[i + 3] = Random.nextFloat() * 0.04f + 0.01f                    // vy (upward)
            data[i + 4] = params.emitLifetime                                   // life
            data[i + 5] = params.emitLifetime                                   // maxLife
            data[i + 6] = params.baseHue                                         // hue
            data[i + 7] = 4f + params.emitBrightness * 8f                       // size
            liveCount++
        }
    }

    fun update(dt: Float, params: SimParams) {
        var alive = 0
        for (i in 0 until liveCount) {
            val b = i * STRIDE
            data[b + 4] -= dt
            if (data[b + 4] <= 0f) continue  // dead — skip (compact later)

            // turbulence
            data[b + 2] += (Random.nextFloat() - 0.5f) * params.turbulence * 0.005f
            data[b + 3] += (Random.nextFloat() - 0.5f) * params.turbulence * 0.005f

            // gravity well pulls toward origin
            if (params.gravityWell > 0f) {
                data[b + 2] -= data[b + 0] * params.gravityWell * 0.002f
                data[b + 3] -= data[b + 1] * params.gravityWell * 0.002f
            }

            data[b + 0] += data[b + 2]
            data[b + 1] += data[b + 3]

            // compact in-place
            if (alive != i) System.arraycopy(data, b, data, alive * STRIDE, STRIDE)
            alive++
        }
        liveCount = alive
    }
}
```

### Step 5 — Shaders (GLSL inline strings)

```kotlin
// Shaders.kt
package nebula

val VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aLife;
layout(location = 2) in float aMaxLife;
layout(location = 3) in float aHue;
layout(location = 4) in float aSize;

out vec3 vColor;

vec3 hsv2rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
    vec3 m = vec3(v - c);
    if (h < 1.0/6.0) return vec3(c, x, 0) + m;
    if (h < 2.0/6.0) return vec3(x, c, 0) + m;
    if (h < 3.0/6.0) return vec3(0, c, x) + m;
    if (h < 4.0/6.0) return vec3(0, x, c) + m;
    if (h < 5.0/6.0) return vec3(x, 0, c) + m;
    return vec3(c, 0, x) + m;
}

void main() {
    float t = aLife / aMaxLife;
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = aSize * t;
    vColor = hsv2rgb(aHue, 0.8, t);
}
""".trimIndent()

val FRAGMENT_SRC = """
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float alpha = (1.0 - d * 2.0) * vColor.r; // soft edge
    fragColor = vec4(vColor, alpha);
}
""".trimIndent()
```

### Step 6 — GPU upgrade path

Replace the CPU update loop with a transform feedback pass:

```
// v2 sketch (pseudocode):
// 1. Bind transform feedback buffer as output
// 2. Draw particles with an UPDATE vertex shader (reads vel, life; writes updated values)
// 3. Use GL_RASTERIZER_DISCARD to skip fragment stage
// 4. Swap input/output buffers
// 5. Bind output as VBO for RENDER pass
```

This keeps all particle state on the GPU — no `FloatArray` upload each frame. Refer to LWJGL transform feedback examples for the full implementation.

---

## Controls

| Key | Action |
|---|---|
| `R` | Randomize emission hue |
| `G` | Toggle gravity well |
| `C` | Clear all particles |
| `S` | Save screenshot to `output/nebula_<timestamp>.png` |
| `Q` / `Esc` | Quit |

---

## Performance Notes

- Pool size 100 000 particles with CPU update runs at 60 fps on most mid-tier hardware.
- Avoid `ArrayList<Particle>` — object allocation per particle triggers GC every few frames. Use the `FloatArray` pool.
- Use `glEnable(GL_BLEND)` + `GL_ONE` additive blending for the glow effect without sorting.
- Enable `GL_PROGRAM_POINT_SIZE` to allow `gl_PointSize` in the vertex shader.
- Profile with `glBeginQuery(GL_TIME_ELAPSED)` to measure GPU time per frame.

---

## Export

```kotlin
// Partial snippet — framebuffer to PNG via STB
fun saveScreenshot(width: Int, height: Int, path: String) {
    val buffer = BufferUtils.createByteBuffer(width * height * 3)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
    // STB image write (LWJGL stb binding):
    STBImageWrite.stbi_flip_vertically_on_write(true)
    STBImageWrite.stbi_write_png(path, width, height, 3, buffer, width * 3)
}
```

---

## Troubleshooting

**No MIDI devices found**: verify `MidiSystem.getMidiDeviceInfo()` returns non-empty; check that the USB device is recognized by the OS (`aconnect -l` on Linux).

**`UnsatisfiedLinkError` on LWJGL natives**: confirm `lwjglNatives` in `build.gradle.kts` matches your OS (`natives-linux`, `natives-macos`, `natives-windows`).

**Particles not visible**: check `GL_PROGRAM_POINT_SIZE` is enabled; check `gl_PointSize > 0` in vertex shader; check blend mode.

**Simulation runs too fast/slow**: normalize updates by `dt` (delta time in seconds between frames).

!!! tip "See also"
    - [Recursive Cathedral Generator](kotlin-recursive-cathedral.md) — L-system generative art in Kotlin + Processing; same creative territory, different rendering backend
    - [OSC + MQTT + Prometheus + SuperCollider](osc-mqtt-prometheus-supercollider.md) — sonify system metrics; a server-side audio complement to this client-side MIDI project
    - [Pi-Based Sample Library Server](pi-sample-server.md) — USB MIDI on a Raspberry Pi; the hardware layer that feeds this nebula
    - [Pi-Powered Infinite Art Frame](pi-infinite-art-frame-kotlin.md) — display your nebula frames on a wall-mounted Pi screen
