---
tags:
  - kotlin
  - generative-art
---

# Cellular Automata Organism Garden (Kotlin + Processing)

**Objective**: Grow a garden of artificial life — starting from Conway's Game of Life rules, then extending into organisms with energy, decay, mutation, and heritable lineage coloring.

---

## What You're Building

- A 2D cellular automata simulation in Kotlin + Processing
- Starts with Conway rules; extends to energy accumulation and mutation
- Cells carry lineage depth — coloring reveals evolutionary history
- Rule sets are swappable at runtime via keyboard
- Seeds and rule configs are exportable for reproducibility
- Frame export for timelapse or animation

---

## Architecture

```
  ┌──────────────────────────────────────────────────────┐
  │  Grid State A (IntArray, flat 2D)                    │
  └────────────────────────┬─────────────────────────────┘
                           │ step(rules, params)
                           ▼
  ┌──────────────────────────────────────────────────────┐
  │  Grid State B (IntArray, double-buffer)              │
  │  Apply rule function per cell                        │
  │  Accumulate energy; decay; maybe mutate              │
  └────────────────────────┬─────────────────────────────┘
                           │ swap A↔B
                           ▼
  ┌──────────────────────────────────────────────────────┐
  │  Colorizer                                           │
  │  lineage depth → hue; energy → brightness            │
  └────────────────────────┬─────────────────────────────┘
                           │
                           ▼
  ┌──────────────────────────────────────────────────────┐
  │  Processing PApplet render (pixel array or rect grid)│
  └──────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| JDK | 17+ |
| Kotlin | 1.9+ |
| Gradle | 8.x (wrapper) |
| Processing core | 4.x (`core.jar` in `libs/`) |

See [Recursive Cathedral Generator](kotlin-recursive-cathedral.md) for instructions on obtaining `core.jar`.

---

## Project Layout

```
ca-garden/
├── build.gradle.kts
├── settings.gradle.kts
├── libs/
│   └── core.jar
└── src/main/kotlin/garden/
    ├── Main.kt           # entry point
    ├── GardenSketch.kt   # PApplet subclass + main loop
    ├── Grid.kt           # double-buffered cell state
    ├── Cell.kt           # cell data model
    ├── Rules.kt          # rule function definitions
    └── Colorizer.kt      # lineage → color mapping
```

---

## Core Concepts

- **Double buffering**: keep two `IntArray` grids (A and B). Write generation N+1 into B while reading from A; swap pointers. Never mutate the grid you're reading.
- **Cell encoding**: pack cell state into a single `Int` — bit fields for alive/dead, energy (0–15), lineage depth (0–255), mutation flag. Avoids per-cell object allocation.
- **Energy model**: living cells accumulate energy each tick from neighbors. Above a threshold, they survive; below, they decay and die.
- **Mutation**: with probability `P_MUTATE`, a surviving cell flips one bit of its rule-lookup index, producing a variant offspring.
- **Lineage depth**: when a cell is born from a parent, `lineage = parent.lineage + 1`. Depth is visualized as hue rotation — young cells are cool blue; ancient lineages are deep red.
- **Rule abstraction**: rules are Kotlin function literals `(neighbors: Int, energy: Int, alive: Boolean) -> CellResult`. Swap the active rule at runtime.
- **Sparse v2**: for large grids, maintain an `activeSet: HashSet<Int>` of cells with live neighbors. Only process those cells each step. Reduces O(W×H) to O(active).

---

## Implementation

### Step 1 — Cell encoding and Grid

```kotlin
// Cell.kt
package garden

// Pack into Int:
// bits 0:    alive (1 = alive)
// bits 1-4:  energy (0-15)
// bits 5-12: lineage depth (0-255)
// bits 13:   mutation flag

inline fun cellAlive(c: Int)    = (c and 0x1) != 0
inline fun cellEnergy(c: Int)   = (c shr 1) and 0xF
inline fun cellLineage(c: Int)  = (c shr 5) and 0xFF
inline fun cellMutant(c: Int)   = (c and 0x2000) != 0

inline fun makeCell(alive: Boolean, energy: Int, lineage: Int, mutant: Boolean): Int =
    (if (alive) 1 else 0) or
    ((energy.coerceIn(0, 15)) shl 1) or
    ((lineage.coerceIn(0, 255)) shl 5) or
    (if (mutant) 0x2000 else 0)
```

```kotlin
// Grid.kt
package garden

class Grid(val cols: Int, val rows: Int) {
    private var current = IntArray(cols * rows)
    private var next    = IntArray(cols * rows)

    operator fun get(x: Int, y: Int) = current[y * cols + x]
    private fun set(x: Int, y: Int, v: Int) { next[y * cols + x] = v }

    fun step(rule: (Int, Int, Int, Boolean) -> Int) {
        for (y in 0 until rows) {
            for (x in 0 until cols) {
                val neighbors = countLiveNeighbors(x, y)
                val c = get(x, y)
                set(x, y, rule(c, neighbors, x * rows + y, /* seed */ 0))
            }
        }
        val tmp = current; current = next; next = tmp
    }

    fun countLiveNeighbors(x: Int, y: Int): Int {
        var n = 0
        for (dy in -1..1) for (dx in -1..1) {
            if (dx == 0 && dy == 0) continue
            val nx = (x + dx + cols) % cols
            val ny = (y + dy + rows) % rows
            if (cellAlive(current[ny * cols + nx])) n++
        }
        return n
    }

    fun randomize(density: Float = 0.3f) {
        for (i in current.indices)
            current[i] = if (Math.random() < density)
                makeCell(true, 5, 0, false) else 0
    }

    fun toArray() = current  // read-only access for rendering
}
```

### Step 2 — Rule definitions

```kotlin
// Rules.kt
package garden

import kotlin.random.Random

data class CellResult(val alive: Boolean, val energy: Int, val lineage: Int, val mutant: Boolean)

// Conway rules (no energy model)
val CONWAY: (Int, Int) -> Int = { cell, n ->
    val alive = cellAlive(cell)
    val nextAlive = if (alive) n in 2..3 else n == 3
    makeCell(nextAlive, if (nextAlive) 5 else 0, cellLineage(cell), false)
}

// Extended: energy + decay + mutation
val ORGANISM: (Int, Int) -> Int = { cell, n ->
    val alive   = cellAlive(cell)
    val energy  = cellEnergy(cell)
    val lineage = cellLineage(cell)
    val mutant  = Random.nextFloat() < 0.0005f

    val nextAlive = when {
        alive  -> n in 2..3 && energy > 0
        !alive -> n == 3
        else   -> false
    }
    val nextEnergy = if (nextAlive) (energy + n - 1).coerceIn(0, 15) else 0
    val nextLineage = if (nextAlive && !alive) (lineage + 1).coerceIn(0, 255) else lineage
    makeCell(nextAlive, nextEnergy, nextLineage, mutant)
}

val RULES = mapOf("conway" to CONWAY, "organism" to ORGANISM)
```

### Step 3 — Colorizer

```kotlin
// Colorizer.kt
package garden

import processing.core.PApplet

fun cellColor(cell: Int, sketch: PApplet): Int {
    if (!cellAlive(cell)) return sketch.color(10f, 10f, 15f)  // near-black background

    val hue      = (cellLineage(cell) / 255f) * 280f          // blue → red arc
    val bright   = 50f + cellEnergy(cell) / 15f * 200f
    val sat      = if (cellMutant(cell)) 80f else 255f

    sketch.colorMode(PApplet.HSB, 360f, 255f, 255f)
    val c = sketch.color(hue, sat, bright)
    sketch.colorMode(PApplet.RGB, 255f, 255f, 255f)
    return c
}
```

### Step 4 — Sketch main loop

```kotlin
// GardenSketch.kt (partial snippet)
package garden

import processing.core.PApplet

class GardenSketch : PApplet() {
    private val COLS = 200; private val ROWS = 150
    private val CELL = 4   // pixels per cell
    private val grid = Grid(COLS, ROWS)
    private var activeRule: (Int, Int) -> Int = ORGANISM
    private var paused = false
    private var frameExport = false

    override fun settings() = size(COLS * CELL, ROWS * CELL)

    override fun setup() {
        background(10); frameRate(30f)
        grid.randomize(0.35f)
    }

    override fun draw() {
        if (!paused) grid.step { c, n -> activeRule(c, n) }
        val cells = grid.toArray()
        loadPixels()
        for (y in 0 until ROWS) for (x in 0 until COLS) {
            val c = cellColor(cells[y * COLS + x], this)
            for (py in 0 until CELL) for (px in 0 until CELL)
                pixels[(y * CELL + py) * width + (x * CELL + px)] = c
        }
        updatePixels()
        if (frameExport) { saveFrame("output/frame-####.png"); frameExport = false }
    }

    override fun keyPressed() {
        when (key) {
            'r', 'R' -> grid.randomize()
            'c', 'C' -> activeRule = CONWAY
            'o', 'O' -> activeRule = ORGANISM
            'p', 'P' -> paused = !paused
            's', 'S' -> frameExport = true
            'q', 'Q' -> exit()
        }
    }
}
```

### Step 5 — Seed + config export for reproducibility

```kotlin
// Partial snippet — export current seed + rule config as JSON
fun exportConfig(seed: Long, ruleName: String, density: Float, path: String) {
    val json = """{"seed":$seed,"rule":"$ruleName","density":$density}"""
    java.io.File(path).writeText(json)
    println("Config saved: $path")
}

// On load:
// val cfg = parseJson(File("config.json").readText())
// random = Random(cfg.seed)
// grid.randomize(cfg.density)
// activeRule = RULES[cfg.rule] ?: ORGANISM
```

---

## Controls

| Key | Action |
|---|---|
| `R` | Re-randomize grid (new seed) |
| `C` | Switch to Conway rules |
| `O` | Switch to Organism rules |
| `P` | Pause / resume |
| `S` | Save current frame to `output/` |
| `E` | Export seed + rule config to `output/config.json` |
| `Q` | Quit |

---

## Performance Notes

- At 200×150 cells × 30 fps, the CPU update loop is fast enough. Profile with `millis()` around `grid.step()`.
- Avoid per-cell object allocation. The `IntArray` bit-pack approach (above) eliminates GC pressure entirely.
- For grids ≥ 500×500, implement the sparse active-set optimization: only process cells that have at least one live neighbor.
- Use `loadPixels()` + direct `pixels[]` write instead of `rect()` per cell — at least 5× faster for dense grids.

---

## Export / Save Output

```kotlin
// Save a PNG frame (Processing built-in):
save("output/garden_frame.png")

// Save a sequence for timelapse (in draw()):
saveFrame("output/frame-####.png")  // #### → zero-padded frame number

// Render a timelapse video from frames (external, FFmpeg):
// ffmpeg -r 30 -i output/frame-%04d.png -c:v libx264 -pix_fmt yuv420p garden.mp4
```

---

## Troubleshooting

**Grid looks identical every run**: `randomize()` uses `Math.random()` by default (not seeded). Replace with `kotlin.random.Random(seed)` and pass an explicit seed.

**FPS drops below 10**: switch to the sparse active-set approach, or reduce cell size / grid dimensions.

**Colors look flat**: check `colorMode` is reset to RGB after HSB calls in `Colorizer.kt`.

**Processing window doesn't open**: confirm `core.jar` is in `libs/` and `flatDir` is declared in `build.gradle.kts`. On Linux, confirm `DISPLAY` is set.

!!! tip "See also"
    - [Recursive Cathedral Generator](kotlin-recursive-cathedral.md) — deterministic generative art via L-systems; pairs well with this probabilistic simulation
    - [Fractal Art Explorer (JavaScript)](fractal-art-explorer-js.md) — real-time GPU generative art in the browser; complementary visual aesthetics
    - [Generative Art in R](../../tutorials/data-science-visualization/r-generative-art.md) — statistical generative art; different discipline, same creative energy
    - [Pi-Powered Infinite Art Frame](pi-infinite-art-frame-kotlin.md) — display the organism garden on a wall-mounted Raspberry Pi screen
