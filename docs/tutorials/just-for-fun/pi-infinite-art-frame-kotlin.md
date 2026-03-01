---
tags:
  - raspberry-pi
  - kotlin
  - generative-art
---

# Pi-Powered Infinite Art Frame (Kotlin on Raspberry Pi)

**Objective**: Turn a Raspberry Pi into a living art installation — a headless JVM process renders generative art frames continuously, shifts palettes by time of day, and exposes a tiny HTTP API for remote control. The display refreshes from disk; no X server required in the render process.

---

## What You're Building

- A headless Kotlin/JVM process renders PNG frames using Java2D offscreen rendering
- Palettes and parameters shift slowly over time (per-minute tick, time-of-day phase)
- Frames are written to a fixed output path; a lightweight viewer on the Pi reads from disk
- A minimal Ktor HTTP server accepts live parameter changes (palette, seed, speed)
- Runs as a systemd service; logs are capped; old frames are cleaned automatically
- GPU upgrade path (OpenGL ES on Pi via JOGL) documented as future work

---

## Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │  Timer (coroutine, per-minute tick)                 │
  │  + TimeOfDay phase (hour → palette zone)            │
  └──────────────────┬──────────────────────────────────┘
                     │ update Params
                     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Renderer (Java2D / BufferedImage offscreen)        │
  │  Generative algorithm (e.g., flow field, fractal)   │
  │  Write frame → /tmp/artframe/latest.png             │
  └──────────────────┬──────────────────────────────────┘
                     │ file write
                     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Display (feh / fim / browser kiosk mode)           │
  │  Reads latest.png; refreshes on file change         │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  Ktor HTTP server (port 8080)                       │
  │  GET  /status         → current params JSON         │
  │  POST /palette/{name} → change palette              │
  │  POST /seed/{n}       → reseed                      │
  │  POST /speed/{f}      → change tick interval        │
  └─────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Raspberry Pi 4 (2 GB+ RAM) | Pi 3 works but is slower; Pi 5 is ideal |
| JDK 17 (ARM64) | `sudo apt install openjdk-17-jdk` |
| Kotlin 1.9+ | Deployed as a fat JAR — no Kotlin install needed on Pi |
| Gradle 8.x | Build on dev machine; deploy JAR to Pi |
| `feh` (image viewer) | `sudo apt install feh` — for file-based display refresh |

**Build on your dev machine; deploy the fat JAR to the Pi.** Do not try to run Gradle on the Pi — it's slow and unnecessary.

```bash
./gradlew shadowJar   # produces build/libs/artframe-all.jar
scp build/libs/artframe-all.jar pi@raspberrypi.local:/opt/artframe/
```

---

## Project Layout

```
pi-artframe/
├── build.gradle.kts
├── settings.gradle.kts
└── src/main/kotlin/artframe/
    ├── Main.kt           # entry point: start renderer + HTTP server
    ├── Params.kt         # shared mutable state (thread-safe)
    ├── TimePhase.kt      # time-of-day → palette zone
    ├── Renderer.kt       # Java2D offscreen render + PNG write
    ├── Algorithms.kt     # generative algorithm(s)
    ├── Palette.kt        # named color palettes
    └── Api.kt            # Ktor route definitions
```

---

## Core Concepts

- **Headless Java2D**: `BufferedImage` + `Graphics2D` works without a display. No X11, no GLFW, no framebuffer access needed. Set `java.awt.headless=true`.
- **Fixed output path**: render to `/tmp/artframe/latest.png`. Atomic write via temp-file rename prevents the viewer reading a half-written PNG.
- **Time-of-day palette**: divide 24 hours into phases (dawn, day, dusk, night). Each phase has a target palette. Interpolate between them over a 30-minute transition window.
- **Per-minute tick**: a coroutine wakes every N seconds (configurable), updates params (nudge seed, advance palette interpolation), and triggers a render.
- **Ktor server**: embedded Ktor with `embeddedServer(Netty, ...)` — no external server needed. ~3 MB dependency.
- **Atomic frame write**: write to `/tmp/artframe/next.png`, then `Files.move(..., ATOMIC_MOVE)` to `latest.png`. Prevents torn reads by the viewer.
- **Disk usage control**: keep only the last N frames in a rotating directory; delete older ones in the cleanup coroutine.

---

## Implementation

### Step 1 — build.gradle.kts

```kotlin
plugins {
    kotlin("jvm") version "1.9.23"
    id("com.github.johnrengelman.shadow") version "8.1.1"
    application
}

repositories { mavenCentral() }

dependencies {
    implementation(kotlin("stdlib"))
    implementation("io.ktor:ktor-server-netty:2.3.11")
    implementation("io.ktor:ktor-server-core:2.3.11")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.11")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.11")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
}

application { mainClass.set("artframe.MainKt") }
```

### Step 2 — Shared params (thread-safe)

```kotlin
// Params.kt
package artframe

import java.util.concurrent.atomic.AtomicReference
import kotlinx.serialization.Serializable

@Serializable
data class ArtParams(
    val seed: Long       = 42L,
    val palette: String  = "dawn",
    val speedSeconds: Int = 10,
    val paletteMix: Float = 0.0f,  // 0.0 = current palette, 1.0 = next
)

// Thread-safe holder — renderer and HTTP handler share this reference
val PARAMS = AtomicReference(ArtParams())
```

### Step 3 — Time-of-day palette phase

```kotlin
// TimePhase.kt
package artframe

import java.time.LocalTime

enum class Phase { DAWN, DAY, DUSK, NIGHT }

fun currentPhase(): Phase {
    val h = LocalTime.now().hour
    return when {
        h in 5..8   -> Phase.DAWN
        h in 9..17  -> Phase.DAY
        h in 18..21 -> Phase.DUSK
        else         -> Phase.NIGHT
    }
}

fun phaseToDefaultPalette(p: Phase) = when(p) {
    Phase.DAWN  -> "dawn"
    Phase.DAY   -> "daylight"
    Phase.DUSK  -> "ember"
    Phase.NIGHT -> "midnight"
}
```

### Step 4 — Renderer (Java2D offscreen)

```kotlin
// Renderer.kt
package artframe

import artframe.Algorithms.drawFlowField
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import javax.imageio.ImageIO

object Renderer {
    private const val W = 1920; private const val H = 1080
    private val outDir = File("/tmp/artframe").also { it.mkdirs() }

    fun render(params: ArtParams) {
        // headless = no display required
        System.setProperty("java.awt.headless", "true")

        val img = BufferedImage(W, H, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics().apply {
            setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        }

        drawFlowField(g, W, H, params)  // your algorithm here
        g.dispose()

        // Atomic write: temp → rename
        val tmp  = File(outDir, "next.png")
        val dest = File(outDir, "latest.png")
        ImageIO.write(img, "PNG", tmp)
        Files.move(tmp.toPath(), dest.toPath(), StandardCopyOption.ATOMIC_MOVE,
                   StandardCopyOption.REPLACE_EXISTING)
    }
}
```

### Step 5 — Flow field algorithm (example)

```kotlin
// Algorithms.kt — partial snippet
package artframe

import artframe.Palette.colorForPalette
import java.awt.BasicStroke
import java.awt.Graphics2D
import kotlin.math.*
import kotlin.random.Random

object Algorithms {
    fun drawFlowField(g: Graphics2D, w: Int, h: Int, params: ArtParams) {
        val rng = Random(params.seed)
        val pal = colorForPalette(params.palette)

        g.color = pal.background
        g.fillRect(0, 0, w, h)
        g.stroke = BasicStroke(1.2f)

        val step = 12; val len = 40; val angScale = 0.004f
        for (sy in 0 until h step step) {
            for (sx in 0 until w step step) {
                var x = sx.toFloat(); var y = sy.toFloat()
                val alpha = (rng.nextFloat() * 180 + 40).toInt()
                g.color = java.awt.Color(pal.strokes[rng.nextInt(pal.strokes.size)].rgb
                    .let { java.awt.Color(it) }.let {
                        java.awt.Color(it.red, it.green, it.blue, alpha)
                    })
                for (s in 0..len) {
                    val angle = sin(x * angScale + params.seed * 0.001f) *
                                cos(y * angScale) * PI.toFloat() * 2f
                    val nx = x + cos(angle) * step * 0.4f
                    val ny = y + sin(angle) * step * 0.4f
                    g.drawLine(x.toInt(), y.toInt(), nx.toInt(), ny.toInt())
                    x = nx; y = ny
                    if (x < 0 || x >= w || y < 0 || y >= h) break
                }
            }
        }
    }
}
```

### Step 6 — Ktor HTTP API

```kotlin
// Api.kt
package artframe

import io.ktor.server.application.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

fun startApi() = embeddedServer(Netty, port = 8080) {
    routing {
        get("/status") {
            call.respondText(Json.encodeToString(PARAMS.get()),
                             contentType = io.ktor.http.ContentType.Application.Json)
        }
        post("/palette/{name}") {
            val name = call.parameters["name"] ?: "dawn"
            PARAMS.updateAndGet { it.copy(palette = name) }
            call.respondText("palette set to $name")
        }
        post("/seed/{n}") {
            val n = call.parameters["n"]?.toLongOrNull() ?: return@post
            PARAMS.updateAndGet { it.copy(seed = n) }
            call.respondText("seed set to $n")
        }
        post("/speed/{f}") {
            val f = call.parameters["f"]?.toIntOrNull() ?: return@post
            PARAMS.updateAndGet { it.copy(speedSeconds = f.coerceIn(1, 3600)) }
            call.respondText("speed set to ${f}s")
        }
    }
}.start(wait = false)
```

### Step 7 — Main entry point + render loop

```kotlin
// Main.kt
package artframe

import kotlinx.coroutines.*

fun main() {
    println("Pi Art Frame starting…")
    startApi()

    runBlocking {
        // Phase-sync: update palette once per minute based on time of day
        launch {
            while (true) {
                val phase = currentPhase()
                val target = phaseToDefaultPalette(phase)
                PARAMS.updateAndGet { it.copy(palette = target) }
                delay(60_000)
            }
        }

        // Render loop
        while (true) {
            val params = PARAMS.get()
            try {
                Renderer.render(params)
                // Drift seed slightly each frame for organic variation
                PARAMS.updateAndGet { it.copy(seed = it.seed + 1) }
            } catch (e: Exception) {
                System.err.println("Render error: ${e.message}")
            }
            delay(params.speedSeconds * 1000L)
        }
    }
}
```

---

## Display Strategy

### Option A — `feh` (simplest, Linux/Pi)

```bash
# Install:
sudo apt install feh

# Kiosk mode: full-screen, reload on file change every 5s
feh --fullscreen --reload 5 /tmp/artframe/latest.png &
```

Add to `~/.config/autostart/artframe-viewer.desktop` for login autostart.

### Option B — Browser kiosk with local HTTP server

Serve `/tmp/artframe/` via a tiny Python server + auto-refreshing HTML:

```bash
# Serve frames:
cd /tmp/artframe && python3 -m http.server 9090 &

# Chromium kiosk:
chromium-browser --kiosk --noerrdialogs \
  "http://localhost:9090/frame.html"
```

```html
<!-- /tmp/artframe/frame.html -->
<html><body style="margin:0;background:#000">
  <img id="f" src="latest.png" style="width:100vw;height:100vh;object-fit:contain">
  <script>setInterval(()=>{document.getElementById('f').src='latest.png?'+Date.now()},5000)</script>
</body></html>
```

---

## systemd Service

```ini
# /etc/systemd/system/artframe.service
[Unit]
Description=Pi Infinite Art Frame
After=network.target

[Service]
User=pi
WorkingDirectory=/opt/artframe
ExecStart=/usr/bin/java -Xmx512m -Djava.awt.headless=true -jar artframe-all.jar
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable artframe
sudo systemctl start artframe
sudo journalctl -u artframe -f
```

### Disk usage control (cron)

```bash
# /etc/cron.hourly/artframe-cleanup
#!/bin/bash
# Keep only the 48 most recent frames; delete older ones
ls -t /tmp/artframe/frame_*.png 2>/dev/null | tail -n +49 | xargs rm -f
```

---

## Controls / Interaction

```bash
# Remote control via HTTP:
curl http://raspberrypi.local:8080/status
curl -X POST http://raspberrypi.local:8080/palette/midnight
curl -X POST http://raspberrypi.local:8080/seed/42
curl -X POST http://raspberrypi.local:8080/speed/30   # render every 30s
```

---

## Performance Notes

- Java2D at 1920×1080 with a flow-field algorithm renders in 1–3 seconds on Pi 4. Acceptable for a 10–30 second update cycle.
- Set `-Xmx512m` to bound heap usage. Java2D uses off-heap for image buffers — monitor RSS, not just heap.
- If render time exceeds the tick interval, renders will queue. Use a semaphore to skip frames rather than queue them.
- **GPU upgrade path**: replace Java2D with JOGL (OpenGL ES 2.0 on Pi) for sub-second renders. More complex; see JOGL + EGL headless setup for ARM Linux.

---

## Troubleshooting

**`java.awt.HeadlessException`**: add `-Djava.awt.headless=true` to the JVM flags in the systemd service (already shown above).

**`/tmp/artframe/latest.png` not updating**: check `journalctl -u artframe` for render errors; verify JVM has write permission to `/tmp/artframe/`.

**feh shows stale image**: ensure `--reload N` is set; alternatively use inotifywait: `inotifywait -m /tmp/artframe/latest.png -e close_write | xargs -I{} feh --reload 0 latest.png`.

**Ktor port 8080 in use**: change in `Api.kt`; confirm with `ss -tlnp | grep 8080`.

**High memory usage**: reduce render resolution (e.g. 1280×720) or reduce algorithm complexity. Check with `systemctl status artframe` → Memory line.

!!! tip "See also"
    - [Pi-Based Sample Library Server](pi-sample-server.md) — another always-on Pi service with an HTTP API; same deployment pattern (systemd, Pi 4, local network)
    - [RKE2 on Raspberry Pi](../docker-infrastructure/rke2-raspberry-pi.md) — run art frame as a Kubernetes Pod if you have a Pi cluster
    - [Recursive Cathedral Generator](kotlin-recursive-cathedral.md) — export cathedral frames into this display pipeline
    - [Cellular Automata Organism Garden](kotlin-cellular-automata-garden.md) — render garden frames on the Pi art display
    - [MIDI-Driven Particle Nebula](kotlin-midi-particle-nebula.md) — send nebula screenshots to the art frame for a live MIDI-to-wall pipeline
