---
tags:
  - generative-art
  - kotlin
---

# Recursive Cathedral Generator (Kotlin + Processing)

Procedurally grow Gothic cathedrals from recursive L-system rules — deterministic seeds, bilateral symmetry, exportable PNG frames. No assets. No meshes. Just math, recursion, and spires that go all the way down.

---

## Table of Contents

1. [What You're Building](#1-what-youre-building)
2. [Output Preview](#2-output-preview)
3. [Architecture](#3-architecture)
4. [Prerequisites](#4-prerequisites)
5. [Project Setup](#5-project-setup)
6. [Core Concept: L-Systems for Architecture](#6-core-concept-l-systems-for-architecture)
7. [Implementation](#7-implementation)
8. [Controls](#8-controls)
9. [Export](#9-export)
10. [Ideas to Mutate It](#10-ideas-to-mutate-it)
11. [Troubleshooting](#11-troubleshooting)
12. [Next Project](#12-next-project)

---

## 1. What You're Building

A generative cathedral silhouette renderer that:

- Uses an **L-system grammar** to grow recursive arch and spire structures
- Renders via **Processing (Java mode, called from Kotlin)**
- Applies **bilateral symmetry** so the result always reads as a building
- Uses a **seed integer** to make any result exactly reproducible
- Adds **controlled jitter** so two nearby seeds produce siblings, not clones
- Exports **PNG frames** at any resolution on keypress

The output is 2D — a filled silhouette with procedural depth implied by layering. Think architectural cross-section, not 3D model.

---

## 2. Output Preview

No images allowed here, so here is what you are aiming for in words and ASCII:

> A tall central spire flanked by recursive sub-spires, each with pointed arch bases, decreasing in height symmetrically outward. Thin lancet windows implied by negative space cuts. Everything emerges from one rule applied recursively 4–6 times.

```
            *
           ***
          *****
         *******                     Seed: 42
            |                        Depth: 5
      *    |||    *                  Jitter: 0.08
     ***   |||   ***
    *****  |||  *****
       |   |||   |
  *   |||  |||  |||   *
 ***  |||  |||  |||  ***
*|||**|||**|||**|||**|||*
 BASE LINE GROUND LEVEL
```

Each run at the same seed is pixel-identical. Changing the seed by 1 shifts arch widths, spire heights, and window placement by small amounts — the family resemblance holds, but no two cathedrals are the same.

---

## 3. Architecture

### Sketch Structure

```
kotlin-cathedral/
├── build.gradle.kts
├── settings.gradle.kts
├── src/
│   └── main/
│       └── kotlin/
│           └── cathedral/
│               ├── Main.kt           # entry point — launches Processing sketch
│               ├── CathedralSketch.kt # Processing PApplet subclass
│               ├── LSystem.kt        # grammar + rewrite engine
│               ├── Turtle.kt         # interprets L-system string → draw calls
│               └── Cathedral.kt      # symmetry, arch/spire rendering logic
└── output/                           # PNG frames land here
```

### How It Fits Together

```
  Main.kt
    └── launches PApplet (CathedralSketch)
          │
          ▼
  CathedralSketch.setup()
    ├── reads seed from args or random
    ├── builds LSystem (grammar + rules)
    ├── iterates grammar N times → string
    └── passes string to Cathedral.draw()
          │
          ▼
  Cathedral.draw()
    ├── splits string at axis for symmetry
    ├── calls Turtle.interpret() for each half
    └── mirrors right half → full silhouette
```

### Deterministic Seeds

```kotlin
// Every random decision flows through one SeededRandom instance
class SeededRandom(seed: Long) {
    private val rng = java.util.Random(seed)
    fun nextFloat(min: Float, max: Float) = min + rng.nextFloat() * (max - min)
    fun nextInt(min: Int, max: Int) = min + rng.nextInt(max - min + 1)
}
```

Pass `--seed 42` at launch. Same seed → identical output across machines.

---

## 4. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| JDK | 17+ | LTS; tested on Temurin 17 and 21 |
| Gradle | 8.x (wrapper) | `./gradlew` handles it — no system install needed |
| Processing core | 4.x | pulled via Gradle dependency |
| OS | Linux / macOS / Windows | Processing runs on all three |

Processing is used as a library (`processing-core` JAR), not as the IDE. You write Kotlin, Processing handles the render loop and windowing.

---

## 5. Project Setup

### `settings.gradle.kts`

```kotlin
rootProject.name = "kotlin-cathedral"
```

### `build.gradle.kts`

```kotlin
plugins {
    kotlin("jvm") version "1.9.23"
    application
}

repositories {
    mavenCentral()
    maven("https://jogamp.org/deployment/maven")
    // Processing core is not on Maven Central — use local JAR or a community mirror
    flatDir { dirs("libs") }
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(files("libs/core.jar"))      // Processing 4 core JAR
    implementation("org.jogamp.jogl:jogl-all:2.4.0:natives-linux-amd64")
}

application {
    mainClass.set("cathedral.MainKt")
}
```

**Getting `core.jar`:**

```bash
# Download Processing 4 for your platform, then copy the core JAR:
cp /path/to/processing-4.x.x/core/library/core.jar libs/
```

Or build it from source. The JAR is not on Maven Central due to licensing; `libs/` is the pragmatic path.

### Folder Bootstrap

```bash
mkdir -p kotlin-cathedral/src/main/kotlin/cathedral kotlin-cathedral/libs kotlin-cathedral/output
cd kotlin-cathedral
gradle wrapper --gradle-version 8.7
```

---

## 6. Core Concept: L-Systems for Architecture

An L-system has three parts:

| Part | Role |
|---|---|
| **Axiom** | Starting string (e.g., `"S"`) |
| **Rules** | Rewrite each symbol → new string |
| **Turtle** | Interprets the final string as drawing commands |

For a cathedral, we map symbols to architectural actions:

| Symbol | Meaning |
|---|---|
| `S` | Spire: draw upward, narrow |
| `A` | Arch: draw a pointed arch base |
| `[` | Push turtle state (branch left) |
| `]` | Pop turtle state (return, branch right) |
| `+` | Rotate left by angle |
| `-` | Rotate right by angle |
| `F` | Move forward (draw) |
| `f` | Move forward (no draw — gap/window) |

A simple cathedral rule:

```
S → S[+SA][-SA]
A → AA
```

After 4 iterations from axiom `S`, you get a branching tree of spires and arches. The turtle then walks this string, placing geometry.

---

## 7. Implementation

### Step A: L-System Grammar Engine

```kotlin
// LSystem.kt
data class LSystem(
    val axiom: String,
    val rules: Map<Char, String>,
    val iterations: Int
) {
    fun generate(): String {
        var current = axiom
        repeat(iterations) {
            current = current.map { rules[it] ?: it.toString() }.joinToString("")
        }
        return current
    }
}

// Usage:
val grammar = LSystem(
    axiom = "S",
    rules = mapOf(
        'S' to "S[+SA][-SA]",
        'A' to "AA"
    ),
    iterations = 5
)
val sentence = grammar.generate()
```

Keep `iterations` at 4–6. At 7+ the string length explodes and render time becomes unreasonable.

### Step B: Turtle Interpreter

```kotlin
// Turtle.kt
import processing.core.PApplet
import java.util.ArrayDeque

data class TurtleState(
    val x: Float, val y: Float,
    val angle: Float, val length: Float
)

class Turtle(
    private val sketch: PApplet,
    private val rng: SeededRandom,
    private val jitter: Float = 0.08f
) {
    private val stack = ArrayDeque<TurtleState>()
    var x = 0f; var y = 0f
    var angle = -90f               // start pointing up
    var length = 80f
    val shrink = 0.65f             // each recursion level shrinks length
    val turnAngle = 25f

    fun interpret(sentence: String) {
        sentence.forEach { cmd ->
            val jitterAngle = rng.nextFloat(-jitter, jitter) * turnAngle
            when (cmd) {
                'F' -> {
                    val nx = x + PApplet.cos(PApplet.radians(angle)) * length
                    val ny = y + PApplet.sin(PApplet.radians(angle)) * length
                    sketch.line(x, y, nx, ny)
                    x = nx; y = ny
                }
                'f' -> {
                    x += PApplet.cos(PApplet.radians(angle)) * length
                    y += PApplet.sin(PApplet.radians(angle)) * length
                }
                '+' -> angle += turnAngle + jitterAngle
                '-' -> angle -= turnAngle + jitterAngle
                '[' -> {
                    stack.push(TurtleState(x, y, angle, length))
                    length *= shrink
                }
                ']' -> {
                    stack.pop().also { s ->
                        x = s.x; y = s.y; angle = s.angle; length = s.length
                    }
                }
            }
        }
    }
}
```

### Step C: Arches, Spires, and Symmetry

The `Cathedral` class takes the turtle output and applies bilateral symmetry — it runs the turtle once for the right half, then mirrors across the vertical axis for the left half.

```kotlin
// Cathedral.kt
import processing.core.PApplet

class Cathedral(
    private val sketch: PApplet,
    private val sentence: String,
    private val seed: Long
) {
    fun draw(centerX: Float, baseY: Float) {
        val rng = SeededRandom(seed)
        sketch.pushMatrix()

        // Right half
        sketch.translate(centerX, baseY)
        drawHalf(sentence, rng, mirrored = false)

        // Mirror left half
        sketch.scale(-1f, 1f)
        drawHalf(sentence, rng.fork(), mirrored = true)

        sketch.popMatrix()
    }

    private fun drawHalf(sentence: String, rng: SeededRandom, mirrored: Boolean) {
        val turtle = Turtle(sketch, rng)
        sketch.strokeWeight(2f)
        sketch.stroke(220f)
        sketch.noFill()
        turtle.interpret(sentence)
        drawGroundLine(sketch)
    }

    private fun drawGroundLine(s: PApplet) {
        s.strokeWeight(3f)
        s.stroke(180f)
        s.line(-s.width / 2f, 0f, s.width / 2f, 0f)
    }
}

// SeededRandom extension for forking a reproducible sibling stream
fun SeededRandom.fork() = SeededRandom(this.nextLong())
```

Arches are drawn by adding `drawArch()` calls when the turtle encounters `'A'` — replace the `'A'` case in the interpreter with a pointed arch shape:

```kotlin
// Partial snippet — add inside Turtle.interpret() when cmd == 'A'
'A' -> {
    val w = length * 0.6f
    val h = length * 1.2f
    // Pointed arch: two bezier curves meeting at apex
    sketch.noFill()
    sketch.beginShape()
    sketch.vertex(x - w, y)
    sketch.bezierVertex(x - w, y - h * 0.5f, x, y - h, x, y - h)
    sketch.endShape()
    sketch.beginShape()
    sketch.vertex(x + w, y)
    sketch.bezierVertex(x + w, y - h * 0.5f, x, y - h, x, y - h)
    sketch.endShape()
}
```

### Step D: Controlled Randomness

Jitter is bounded by `jitter` (a float in `[0, 1]`). At `0.0`, the output is perfectly geometric. At `0.15`, it reads as hand-drafted. At `0.4+`, it collapses into noise.

```kotlin
// CathedralSketch.kt (partial snippet)
val jitter = args.getOrNull(1)?.toFloat() ?: 0.08f

// The SeededRandom instance is created once and passed to all components.
// Every random call consumes from the same stream in a fixed order,
// so seed stability is guaranteed as long as the call order doesn't change.
val rng = SeededRandom(seed)
```

Good jitter range for cathedral-like results: **0.04 – 0.12**.

### Full Sketch Entry Point

```kotlin
// Main.kt
package cathedral

fun main(args: Array<String>) {
    val seed = args.getOrNull(0)?.toLong() ?: System.currentTimeMillis()
    val jitter = args.getOrNull(1)?.toFloat() ?: 0.08f
    processing.core.PApplet.main(arrayOf(
        "--sketch-path=${System.getProperty("user.dir")}",
        "cathedral.CathedralSketch",
        seed.toString(),
        jitter.toString()
    ))
}
```

```kotlin
// CathedralSketch.kt
package cathedral

import processing.core.PApplet

class CathedralSketch : PApplet() {
    private var seed = 42L
    private var jitter = 0.08f
    private lateinit var cathedral: Cathedral

    override fun settings() {
        size(1200, 900)
    }

    override fun setup() {
        background(15)
        colorMode(RGB, 255)
        seed = args?.getOrNull(0)?.toLong() ?: 42L
        jitter = args?.getOrNull(1)?.toFloat() ?: 0.08f
        rebuild()
    }

    override fun draw() {
        // Static render — only redraws on keypress
    }

    private fun rebuild() {
        background(15)
        val grammar = LSystem(
            axiom = "S",
            rules = mapOf(
                'S' to "S[+SA][-SA]",
                'A' to "AA"
            ),
            iterations = 5
        )
        val sentence = grammar.generate()
        cathedral = Cathedral(this, sentence, seed)
        cathedral.draw(width / 2f, height * 0.9f)
    }

    override fun keyPressed() {
        when (key) {
            'r', 'R' -> { seed = System.currentTimeMillis(); rebuild() }
            's', 'S' -> { seed++; rebuild() }
            'e', 'E' -> exportFrame()
            'q', 'Q' -> exit()
        }
    }

    private fun exportFrame() {
        val fname = "output/cathedral_seed${seed}.png"
        save(fname)
        println("Exported: $fname")
    }
}
```

---

## 8. Controls

| Key | Action |
|---|---|
| `R` | Regenerate with new random seed (current time) |
| `S` | Increment seed by 1 (step through sibling cathedrals) |
| `E` | Export current frame as PNG to `output/` |
| `Q` | Quit |

Run it:

```bash
./gradlew run --args="42 0.08"
#                     ^   ^
#                  seed  jitter
```

---

## 9. Export

### PNG Frame Export

`exportFrame()` (bound to `E`) calls Processing's built-in `save()`:

```kotlin
save("output/cathedral_seed${seed}.png")
```

Output resolution equals the sketch window size — set in `settings()`. To render at 4K:

```kotlin
override fun settings() {
    size(3840, 2160)  // 4K — will open a very large window
}
```

### Headless High-Resolution Render

Processing 4 supports headless rendering with the `PDF` renderer. Add the dependency and switch the renderer for batch export:

```kotlin
// Partial snippet — switch renderer for PDF/high-res export
override fun settings() {
    size(4000, 3000, PDF, "output/cathedral_seed${seed}.pdf")
}
```

For PNG at arbitrary resolution without a window, use `createGraphics()`:

```kotlin
// Partial snippet
val pg = createGraphics(4000, 3000)
pg.beginDraw()
pg.background(15)
cathedral.drawTo(pg)  // refactor draw() to accept a PGraphics target
pg.endDraw()
pg.save("output/cathedral_${seed}_4k.png")
```

---

## 10. Ideas to Mutate It

1. **Color by depth** — track recursion depth in the turtle and shift stroke color from white to amber to red as depth increases.
2. **Filled silhouette** — use `beginShape()`/`endShape()` with fills to produce a solid black cutout on a colored sky gradient.
3. **Flying buttresses** — add a second L-system pass that connects adjacent arches with curved struts.
4. **Rose window** — detect the arch apex coordinate; draw a procedural circular mandala at that point using a separate rule set.
5. **Fog of depth** — apply alpha decay inversely proportional to spire height so distant structures fade.
6. **Animated growth** — render one character of the L-system string per frame; the cathedral grows before your eyes.
7. **Exportable SVG** — use Processing's SVG renderer instead of PNG for clean vector output.
8. **Rule evolution** — start with a shallow grammar and mutate one rule per keypress, showing gradual drift.
9. **Floor plan mode** — rotate the turtle 90°; the cathedral becomes a procedural floor plan instead of a silhouette.
10. **Soundtrack** — pipe the seed into a OSC message to SuperCollider; each unique seed triggers a different chord voicing. (See [OSC + MQTT + Prometheus + SuperCollider](osc-mqtt-prometheus-supercollider.md).)

---

## 11. Troubleshooting

### Processing window does not open

- Confirm `core.jar` is in `libs/` and the `flatDir` is declared in `build.gradle.kts`.
- On Linux, ensure `DISPLAY` is set: `export DISPLAY=:0` before running.
- Try `./gradlew run --info` to see the full classpath and pinpoint missing natives.

### `UnsatisfiedLinkError` on JOGL natives

Processing 4 requires platform-specific OpenGL natives. Make sure you pulled the correct native JAR for your platform (`linux-amd64`, `macosx-universal`, `windows-amd64`):

```kotlin
// build.gradle.kts — adjust natives classifier to match your OS
implementation("org.jogamp.jogl:jogl-all:2.4.0:natives-linux-amd64")
implementation("org.jogamp.gluegen:gluegen-rt:2.4.0:natives-linux-amd64")
```

### L-system string becomes too large / OOM

Reduce `iterations`. At depth 6 with the default rule, the string has `3^6 = 729` symbols — fine. At depth 8 it's `3^8 = 6561` — still manageable. At depth 10+ the string can hit millions of characters. Add a guard:

```kotlin
require(iterations <= 7) { "Iteration depth > 7 is unreasonable. Use 4–6." }
```

### Output looks like noise, not a cathedral

- Check `turnAngle` — values above 35° will fold the structure chaotically.
- Check `shrink` — values above 0.85 mean sub-spires are almost the same size as the parent; the structure won't read as hierarchical.
- Check `jitter` — values above 0.2 destroy the geometric quality. Start at 0.0 (perfect geometry) and work up.

### Exported PNG is blank

- `save()` must be called inside or after `draw()`. If `draw()` never runs (static sketch), call `redraw()` once and export from `keyPressed()` after — which is exactly what the example above does.

---

## 12. Next Project

**Next: MIDI-Driven Particle Nebula (Kotlin + Processing)** — coming soon.

Each MIDI note becomes a particle emitter. Velocity → emission rate. Pitch → particle color. Sustain → particle lifetime. The sky fills up when you play.

In the meantime, if you want MIDI on the Pi right now: [Pi-Based Sample Library Server](pi-sample-server.md).

---

*Seed 42. Depth 5. Jitter 0.08. Your cathedral awaits.*

!!! tip "See also"
    - [MIDI-Driven Particle Nebula](kotlin-midi-particle-nebula.md) — the next Kotlin art project: OpenGL particles driven by live MIDI input
    - [Cellular Automata Organism Garden](kotlin-cellular-automata-garden.md) — probabilistic life simulation in Kotlin + Processing; different algorithm, same toolkit
    - [Pi-Powered Infinite Art Frame](pi-infinite-art-frame-kotlin.md) — display cathedral frames on a wall-mounted Pi screen
    - [Fractal Art Explorer (JavaScript)](fractal-art-explorer-js.md) — WebGL Mandelbrot/Julia; same generative-art energy, browser-native
    - [Generative Art in R](../../tutorials/data-science-visualization/r-generative-art.md) — statistical generative art; complementary approach
