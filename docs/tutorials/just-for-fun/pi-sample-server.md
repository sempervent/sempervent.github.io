---
tags:
  - raspberry-pi
  - audio
  - midi
  - docker
---

# Pi-Based Sample Library Server with Live Audition

**Objective**: Turn a Raspberry Pi into a local sample library server — complete with a web UI, SQLite index, Nginx routing, USB MIDI input, and WebAudio live audition. No cloud. No subscription. Just samples, latency, and a blinking LED.

---

## Table of Contents

1. [Project Overview & Philosophy](#1-project-overview--philosophy)
2. [Hardware Setup](#2-hardware-setup)
3. [System Architecture](#3-system-architecture)
4. [Backend Implementation](#4-backend-implementation)
5. [Frontend: WebAudio + Web MIDI](#5-frontend-webaudio--web-midi)
6. [Live Audition Flow](#6-live-audition-flow)
7. [Sample Organization Strategy](#7-sample-organization-strategy)
8. [Deployment & GitHub Pages Integration](#8-deployment--github-pages-integration)
9. [Enhancements & Roadmap](#9-enhancements--roadmap)
10. [Appendix](#10-appendix)

---

## 1. Project Overview & Philosophy

### What Is This?

This is a self-hosted sample library server running on a Raspberry Pi. It indexes your WAV sample packs into SQLite, serves them over Nginx, and exposes a browser-based UI where you can browse, search, and audition samples in real-time — triggered by your USB MIDI controller.

No DAW required. No latency from a SaaS roundtrip. Just you, your pads, and your Pi.

### Why a Pi?

- **Always on.** Plug it in, it's there. No boot-up ritual.
- **Local network only.** Zero cloud surface area. Your samples stay yours.
- **Cheap.** A Pi 4 with a 1TB SSD costs less than one month of a sample subscription.
- **Hackable.** You can modify every layer of the stack without a terms-of-service.
- **Fun.** There is intrinsic joy in making a $70 computer serve audio.

### Why Not SaaS?

SaaS sample tools are fine products. But they assume you want:

- A browser tab tethered to their servers
- Monthly billing for sounds you already own
- Rate limits on your own creativity
- Someone else's idea of "discovery"

This project is the opposite. You own the index. You own the files. You own the API.

### Architecture Overview

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Raspberry Pi 4                           │
  │                                                             │
  │  ┌───────────┐    ┌──────────────┐    ┌─────────────────┐  │
  │  │  SQLite   │◄───│  Indexer     │◄───│  WAV Files      │  │
  │  │  (index)  │    │  (Python)    │    │  (SSD/USB)      │  │
  │  └─────┬─────┘    └──────────────┘    └─────────────────┘  │
  │        │                                                     │
  │  ┌─────▼─────┐    ┌──────────────┐                          │
  │  │  FastAPI  │◄───│  Nginx       │◄──── LAN HTTP Request    │
  │  │  (API)    │    │  (proxy +    │                          │
  │  └───────────┘    │   static)    │                          │
  │                   └──────────────┘                          │
  │                                                             │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │  USB MIDI (Korg padKontrol, etc.)                   │   │
  │  │  → aconnect → Web MIDI API (browser)                │   │
  │  └─────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
          │
          │  Browser (any device on LAN)
  ┌───────▼──────────────────────────────────────┐
  │  Web UI                                       │
  │  ├── Sample browser (search/filter)           │
  │  ├── WebAudio playback engine                 │
  │  └── Web MIDI pad trigger mapping             │
  └───────────────────────────────────────────────┘
```

### Feature List

| Feature | Status |
|---|---|
| WAV sample indexing via Python | Core |
| SQLite metadata store (BPM, key, vibe, tags) | Core |
| Nginx static file serving + API proxy | Core |
| FastAPI backend with query endpoints | Core |
| Browser-based sample browser UI | Core |
| WebAudio API playback engine | Core |
| Web MIDI API pad trigger input | Core |
| USB MIDI device detection (aconnect) | Core |
| Velocity-to-gain mapping | Core |
| Sample preloading + buffer cache | Core |
| Waveform visualization | Enhancement |
| MIDI clock sync | Roadmap |
| ADSR envelope controls | Roadmap |
| Sample layering | Roadmap |
| Multi-user LAN jam mode | Roadmap |

### Demo Workflow

Here's what a typical session looks like:

1. Pi boots. Nginx is up. FastAPI is running via systemd.
2. You plug in your Korg padKontrol via USB.
3. Open a browser on your laptop: `http://pi.local:8080`
4. The sample browser loads — showing your indexed library with BPM, key, and vibe metadata.
5. You filter by BPM range and vibe tag. Results populate instantly.
6. You click a sample row. It preloads into the WebAudio buffer.
7. You hit a pad. The sample plays. Velocity controls gain.
8. You find a loop you like. You tag it, export the filepath, drag it into your DAW.
9. Repeat. No subscription renewed.

---

## 2. Hardware Setup

### Recommended Hardware

| Component | Recommendation |
|---|---|
| Pi model | Raspberry Pi 4 (4GB or 8GB RAM) |
| OS | Raspberry Pi OS Lite (64-bit) |
| Storage | USB 3.0 SSD (500GB–2TB) — NOT SD card for samples |
| Boot drive | SD card (16GB min) or SSD boot |
| MIDI controller | Korg padKontrol, Arturia BeatStep, or any USB class-compliant device |
| Network | Wired Ethernet preferred for stability; Wi-Fi works |
| Power | Official Pi 4 USB-C PSU (5V/3A) |

### Why Not SD Card for Samples?

SD cards are fine for the OS. They are not fine for a sample library. SD cards have:

- High write amplification under random I/O
- Limited endurance (especially cheap cards)
- Slow random read speeds for large WAV files

Use a USB 3.0 SSD for the sample library. Boot from SD, store samples on SSD.

```bash
# Check SSD is detected
lsblk

# Mount it (example: /dev/sda1)
sudo mkdir -p /mnt/samples
sudo mount /dev/sda1 /mnt/samples

# Add to /etc/fstab for persistent mount
echo '/dev/sda1 /mnt/samples ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab
```

### USB MIDI Device Detection

```bash
# List all USB devices
lsusb

# Example output for Korg padKontrol:
# Bus 001 Device 004: ID 0944:0204 KORG, Inc. padKONTROL

# List MIDI devices via ALSA
aconnect -l

# Expected output:
# client 20: 'padKONTROL' [type=kernel,card=1]
#     0 'padKONTROL MIDI 1'
#     1 'padKONTROL MIDI 2'

# Install ALSA tools if needed
sudo apt-get install -y alsa-utils
```

Web MIDI API handles MIDI directly in the browser for modern Chromium-based browsers — no additional Linux-side routing needed for browser access. `aconnect` is useful for debugging device visibility.

### Audio Latency Considerations

This project uses WebAudio API for playback — audio is rendered in the browser, not on the Pi's audio hardware. This means:

- **Pi audio output is not used.** Your laptop/desktop browser handles DAC output.
- **Latency = network round-trip (LAN) + WebAudio buffer size.**
- On a wired LAN, network latency is typically < 1ms.
- WebAudio buffer latency is configurable — aim for 128–256 sample buffer sizes.
- Total perceived latency goal: < 20ms from pad hit to sound.

```bash
# Verify network latency from client to Pi
ping pi.local

# Expected: < 1ms on wired LAN, 1–5ms on Wi-Fi
```

### Network Configuration

```bash
# Set static IP (recommended for reliability)
# Edit /etc/dhcpcd.conf
sudo nano /etc/dhcpcd.conf

# Add:
# interface eth0
# static ip_address=192.168.1.100/24
# static routers=192.168.1.1
# static domain_name_servers=192.168.1.1

# Enable mDNS for pi.local resolution
sudo apt-get install -y avahi-daemon
sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon

# Verify hostname
hostname  # should return: raspberrypi (or custom name)
```

### Signal Flow (Optional ASCII Diagram)

```
  USB MIDI Controller
         │
         ▼ (USB)
  Raspberry Pi OS
  (ALSA USB MIDI kernel driver)
         │
         ▼ (Web MIDI API — browser access)
  Browser Tab (client device on LAN)
  ├── Web MIDI API receives pad events
  ├── Maps pad → sample ID
  ├── Fetches/retrieves preloaded AudioBuffer
  └── WebAudio API plays buffer → DAC → headphones/speakers

  Browser Tab
         │ (HTTP GET /samples?bpm=...)
         ▼
  Nginx (Pi, port 8080)
  ├── /api/* → FastAPI (port 8000)
  └── /static/* → WAV files on SSD
```

---

## 3. System Architecture

### Directory Layout on Pi

```
/
├── etc/
│   └── nginx/
│       └── sites-enabled/
│           └── pi-sample-server
├── home/
│   └── pi/
│       └── pi-sample-server/
│           ├── api/
│           │   ├── main.py          # FastAPI app
│           │   ├── db.py            # SQLite connection helpers
│           │   └── requirements.txt
│           ├── indexer/
│           │   ├── index_samples.py # Scan + ingest WAV files
│           │   └── extract_meta.py  # WAV metadata extraction
│           └── ui/                  # Static frontend (HTML/CSS/JS)
│               ├── index.html
│               ├── app.js
│               └── style.css
└── mnt/
    └── samples/                     # SSD mount point
        ├── packs/
│       │   ├── kick_pack_01/
│       │   │   ├── kick_120bpm_Am.wav
│       │   │   └── kick_140bpm_Gm.wav
│       │   └── loops_house_01/
│       │       ├── loop_120bpm_Cm_vibe_deep.wav
│       │       └── loop_128bpm_Fm_vibe_tech.wav
        └── samples.db               # SQLite index
```

### Nginx Configuration Strategy

Nginx serves two roles:

1. **Static file serving** — WAV files from `/mnt/samples/packs/` at `/static/`
2. **API proxy** — forwards `/api/` requests to FastAPI on port 8000

```nginx
# /etc/nginx/sites-enabled/pi-sample-server
server {
    listen 8080;
    server_name pi.local 192.168.1.100;

    # Static WAV files
    location /static/ {
        alias /mnt/samples/packs/;
        add_header Accept-Ranges bytes;
        add_header Cache-Control "public, max-age=86400";
        types {
            audio/wav wav;
        }
    }

    # API proxy to FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Frontend UI
    location / {
        root /home/pi/pi-sample-server/ui;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
}
```

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    bpm         REAL,
    key         TEXT,
    vibe        TEXT,
    tags        TEXT,           -- comma-separated tag list
    duration_ms INTEGER,
    filepath    TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_bpm ON samples(bpm);
CREATE INDEX IF NOT EXISTS idx_key ON samples(key);
CREATE INDEX IF NOT EXISTS idx_vibe ON samples(vibe);
```

**Field notes:**

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Auto-incremented primary key |
| `filename` | TEXT | Base filename (e.g., `kick_120bpm_Am.wav`) |
| `bpm` | REAL | Float — allows 120.5 BPM |
| `key` | TEXT | Musical key (e.g., `Am`, `Cm`, `G`) |
| `vibe` | TEXT | Free-form mood tag (e.g., `dark`, `deep`, `punchy`) |
| `tags` | TEXT | Comma-separated tags (e.g., `kick,transient,short`) |
| `duration_ms` | INTEGER | Duration in milliseconds |
| `filepath` | TEXT | Absolute path on Pi filesystem |

### Static File Serving Strategy

WAV files are served directly from Nginx — FastAPI never touches audio bytes. This keeps the API lightweight and allows range requests (for partial audio loading) to be handled natively by Nginx.

Key header: `Accept-Ranges: bytes` — enables browser to fetch partial WAV data for waveform rendering without downloading the full file.

### API Routing Concept

FastAPI handles three endpoint families:

| Endpoint | Method | Description |
|---|---|---|
| `/samples` | GET | List all samples (with optional filters) |
| `/samples?bpm=120&key=Am` | GET | Filter by BPM, key, vibe, tags |
| `/samples/{id}` | GET | Fetch a single sample record |
| `/samples/{id}/url` | GET | Return the Nginx static URL for the WAV file |

All responses are JSON. No audio bytes pass through the API.

---

## 4. Backend Implementation

### Sample Indexing Script

The indexer walks the sample directory, extracts WAV metadata, infers BPM/key from filename conventions, and upserts records into SQLite.

```python
# indexer/index_samples.py
import os
import sqlite3
from extract_meta import extract_wav_meta, parse_filename_meta

DB_PATH = "/mnt/samples/samples.db"
SAMPLES_ROOT = "/mnt/samples/packs"

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            bpm         REAL,
            key         TEXT,
            vibe        TEXT,
            tags        TEXT,
            duration_ms INTEGER,
            filepath    TEXT NOT NULL UNIQUE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON samples(bpm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON samples(key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vibe ON samples(vibe)")
    conn.commit()

def index_directory(conn, root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(dirpath, fname)
            wav_meta = extract_wav_meta(fpath)
            name_meta = parse_filename_meta(fname)
            record = {
                "filename": fname,
                "bpm":      name_meta.get("bpm") or wav_meta.get("bpm"),
                "key":      name_meta.get("key"),
                "vibe":     name_meta.get("vibe"),
                "tags":     name_meta.get("tags"),
                "duration_ms": wav_meta.get("duration_ms"),
                "filepath": fpath,
            }
            conn.execute("""
                INSERT INTO samples (filename, bpm, key, vibe, tags, duration_ms, filepath)
                VALUES (:filename, :bpm, :key, :vibe, :tags, :duration_ms, :filepath)
                ON CONFLICT(filepath) DO UPDATE SET
                    bpm=excluded.bpm,
                    key=excluded.key,
                    vibe=excluded.vibe,
                    tags=excluded.tags,
                    duration_ms=excluded.duration_ms
            """, record)
    conn.commit()

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    index_directory(conn, SAMPLES_ROOT)
    print("Indexing complete.")
    conn.close()
```

### WAV Metadata Extraction

```python
# indexer/extract_meta.py
import wave
import re

def extract_wav_meta(filepath: str) -> dict:
    """Extract duration from WAV header."""
    try:
        with wave.open(filepath, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_ms = int((frames / rate) * 1000)
            return {"duration_ms": duration_ms}
    except Exception:
        return {"duration_ms": None}

def parse_filename_meta(filename: str) -> dict:
    """
    Infer BPM, key, and vibe from filename conventions.

    Expected pattern (flexible):
      <name>_<bpm>bpm_<key>_vibe_<vibe>.<ext>
      e.g.: loop_120bpm_Am_vibe_deep.wav
    """
    meta = {"bpm": None, "key": None, "vibe": None, "tags": None}
    name = filename.lower().replace(".wav", "")
    parts = name.split("_")

    bpm_match = re.search(r"(\d+(?:\.\d+)?)bpm", name)
    if bpm_match:
        meta["bpm"] = float(bpm_match.group(1))

    key_pattern = re.compile(r"\b([a-g](?:#|b)?(?:m|maj|min)?)\b", re.IGNORECASE)
    key_match = key_pattern.search(name)
    if key_match:
        meta["key"] = key_match.group(1)

    if "vibe" in parts:
        idx = parts.index("vibe")
        if idx + 1 < len(parts):
            meta["vibe"] = parts[idx + 1]

    # Tags = all non-numeric, non-bpm, non-key parts
    tag_parts = [p for p in parts if not re.search(r"\d+bpm|vibe", p)]
    meta["tags"] = ",".join(tag_parts) if tag_parts else None

    return meta
```

### FastAPI Endpoints

```python
# api/main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import sqlite3
from typing import Optional

DB_PATH = "/mnt/samples/samples.db"
STATIC_BASE = "http://pi.local:8080/static"

app = FastAPI(title="Pi Sample Server")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/samples")
def list_samples(
    bpm: Optional[float] = Query(None),
    bpm_range: Optional[int] = Query(10, alias="bpm_range"),
    key: Optional[str] = Query(None),
    vibe: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0),
):
    conn = get_db()
    query = "SELECT * FROM samples WHERE 1=1"
    params = []

    if bpm is not None:
        query += " AND bpm BETWEEN ? AND ?"
        params += [bpm - bpm_range, bpm + bpm_range]
    if key:
        query += " AND key = ?"
        params.append(key)
    if vibe:
        query += " AND vibe = ?"
        params.append(vibe)
    if tags:
        for tag in tags.split(","):
            query += " AND tags LIKE ?"
            params.append(f"%{tag.strip()}%")

    query += " LIMIT ? OFFSET ?"
    params += [limit, offset]

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/samples/{sample_id}")
def get_sample(sample_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM samples WHERE id = ?", [sample_id]).fetchone()
    conn.close()
    if row is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    result = dict(row)
    # Append static URL for browser fetch
    rel_path = result["filepath"].replace("/mnt/samples/packs/", "")
    result["url"] = f"{STATIC_BASE}/{rel_path}"
    return result
```

**Run the API:**

```bash
# Install dependencies
pip install fastapi uvicorn

# Start (development)
uvicorn main:app --host 0.0.0.0 --port 8000

# Start via systemd (production — see Appendix)
sudo systemctl start pi-sample-api
```

### Performance Considerations

- **SQLite is fast enough** for a local LAN library of < 100,000 samples.
- Use `WAL` journal mode for concurrent reads while indexing:

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA cache_size=-32000")  # 32MB page cache
conn.execute("PRAGMA synchronous=NORMAL")
```

- **No connection pooling needed** — single-user LAN use case. If multi-user, use `aiosqlite` with FastAPI's async support.
- Index re-runs are idempotent via `ON CONFLICT DO UPDATE`.

### Caching Strategy

- Nginx caches static WAV responses at the HTTP layer (`Cache-Control: max-age=86400`).
- Browser caches decoded `AudioBuffer` objects in-memory — no re-fetch on re-trigger.
- SQLite query results for frequently used BPM ranges can be cached in a Python `dict` or `functools.lru_cache` — acceptable for single-user use.

---

## 5. Frontend: WebAudio + Web MIDI

### HTML Layout

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pi Sample Server</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>🎛 Pi Sample Library</h1>
    <div id="midi-status">MIDI: checking...</div>
  </header>

  <section id="filters">
    <input type="number" id="filter-bpm" placeholder="BPM (e.g. 120)" min="60" max="200">
    <input type="text" id="filter-key" placeholder="Key (e.g. Am)">
    <input type="text" id="filter-vibe" placeholder="Vibe (e.g. dark)">
    <button id="btn-search">Search</button>
  </section>

  <section id="sample-list">
    <table>
      <thead>
        <tr>
          <th>Filename</th><th>BPM</th><th>Key</th><th>Vibe</th>
          <th>Duration</th><th>Actions</th>
        </tr>
      </thead>
      <tbody id="sample-rows"></tbody>
    </table>
  </section>

  <section id="pad-grid">
    <!-- 16 pads — map to loaded samples -->
    <div id="pads"></div>
  </section>

  <script src="app.js"></script>
</body>
</html>
```

### WebAudio Playback Engine

```javascript
// app.js — WebAudio engine
const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
  latencyHint: 'interactive',
  sampleRate: 44100,
});

// Buffer cache: sampleId -> AudioBuffer
const bufferCache = new Map();

async function loadSample(sampleId, url) {
  if (bufferCache.has(sampleId)) return bufferCache.get(sampleId);

  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  bufferCache.set(sampleId, audioBuffer);
  return audioBuffer;
}

function playSample(sampleId, velocity = 127) {
  const buffer = bufferCache.get(sampleId);
  if (!buffer) return;

  // Ensure AudioContext is running (browsers require user gesture)
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const source = audioCtx.createBufferSource();
  const gainNode = audioCtx.createGain();

  source.buffer = buffer;
  gainNode.gain.value = velocity / 127;  // velocity → gain (0.0–1.0)

  source.connect(gainNode);
  gainNode.connect(audioCtx.destination);
  source.start();
}
```

### Sample Browser UI

```javascript
// Fetch samples from API and render table rows
async function searchSamples() {
  const bpm = document.getElementById('filter-bpm').value;
  const key = document.getElementById('filter-key').value;
  const vibe = document.getElementById('filter-vibe').value;

  const params = new URLSearchParams();
  if (bpm)  params.append('bpm', bpm);
  if (key)  params.append('key', key);
  if (vibe) params.append('vibe', vibe);

  const res = await fetch(`/api/samples?${params}`);
  const samples = await res.json();

  const tbody = document.getElementById('sample-rows');
  tbody.innerHTML = '';

  for (const s of samples) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${s.filename}</td>
      <td>${s.bpm ?? '—'}</td>
      <td>${s.key ?? '—'}</td>
      <td>${s.vibe ?? '—'}</td>
      <td>${s.duration_ms ? (s.duration_ms / 1000).toFixed(2) + 's' : '—'}</td>
      <td>
        <button onclick="preloadAndPlay(${s.id})">▶ Play</button>
        <button onclick="assignToPad(${s.id}, 0)">→ Pad</button>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

async function preloadAndPlay(sampleId) {
  const res = await fetch(`/api/samples/${sampleId}`);
  const sample = await res.json();
  await loadSample(sampleId, sample.url);
  playSample(sampleId, 100);
}

document.getElementById('btn-search').addEventListener('click', searchSamples);
```

### Pad Grid

```javascript
// 16-pad grid: padIndex → sampleId
const padMap = new Map(); // padIndex (0–15) → sampleId

function buildPadGrid() {
  const container = document.getElementById('pads');
  for (let i = 0; i < 16; i++) {
    const pad = document.createElement('div');
    pad.className = 'pad';
    pad.dataset.index = i;
    pad.textContent = i + 1;
    pad.addEventListener('click', () => {
      const sid = padMap.get(i);
      if (sid !== undefined) playSample(sid, 100);
    });
    container.appendChild(pad);
  }
}

function assignToPad(sampleId, padIndex) {
  padMap.set(padIndex, sampleId);
  const pad = document.querySelector(`.pad[data-index="${padIndex}"]`);
  if (pad) pad.classList.add('loaded');
}

buildPadGrid();
```

### Web MIDI API Integration

```javascript
// MIDI setup — requires Chromium/Chrome (HTTPS or localhost)
let midiAccess = null;

async function initMIDI() {
  const statusEl = document.getElementById('midi-status');
  try {
    midiAccess = await navigator.requestMIDIAccess({ sysex: false });
    statusEl.textContent = `MIDI: ${midiAccess.inputs.size} device(s) found`;

    for (const input of midiAccess.inputs.values()) {
      input.onmidimessage = handleMIDIMessage;
    }

    // Re-bind on device connect/disconnect
    midiAccess.onstatechange = (event) => {
      const port = event.port;
      if (port.type === 'input' && port.state === 'connected') {
        port.onmidimessage = handleMIDIMessage;
        statusEl.textContent = `MIDI: ${port.name} connected`;
      }
    };
  } catch (err) {
    statusEl.textContent = `MIDI: not available (${err.message})`;
  }
}

function handleMIDIMessage(event) {
  const [status, note, velocity] = event.data;
  const isNoteOn = (status & 0xF0) === 0x90 && velocity > 0;

  if (!isNoteOn) return;

  // Map MIDI note to pad index (Korg padKontrol: notes 36–51)
  const padIndex = note - 36;
  if (padIndex < 0 || padIndex > 15) return;

  const sampleId = padMap.get(padIndex);
  if (sampleId !== undefined) {
    playSample(sampleId, velocity);
  }
}

initMIDI();
```

**Browser MIDI permissions note:**

- Web MIDI API requires HTTPS in production or `localhost`.
- On Chrome: first-visit permission prompt appears — user must click "Allow".
- Firefox does not support Web MIDI natively — requires a plugin or switch to Chrome/Chromium.
- On the Pi itself, serving from `http://pi.local` works in Chrome with `--unsafely-treat-insecure-origin-as-secure` flag, or use a self-signed cert + Nginx TLS.

### Device Selection

```javascript
// Expose a device selector for multi-device setups
function buildDeviceSelector() {
  const select = document.createElement('select');
  select.id = 'midi-device-select';

  for (const [id, input] of midiAccess.inputs) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = input.name;
    select.appendChild(opt);
  }

  select.addEventListener('change', () => {
    for (const input of midiAccess.inputs.values()) {
      input.onmidimessage = null;
    }
    const selected = midiAccess.inputs.get(select.value);
    if (selected) selected.onmidimessage = handleMIDIMessage;
  });

  document.querySelector('#midi-status').after(select);
}
```

---

## 6. Live Audition Flow

### What Happens When a Pad Is Pressed

```
  Pad physical press
       │
       ▼
  USB HID event → OS (kernel)
       │
       ▼
  Web MIDI API (browser) receives MIDIMessageEvent
       │
       ▼
  handleMIDIMessage():
    - Parse status byte → Note On
    - Extract note number → pad index (note - 36)
    - Extract velocity (0–127)
       │
       ▼
  padMap.get(padIndex) → sampleId
       │
       ▼
  playSample(sampleId, velocity):
    - bufferCache.get(sampleId) → AudioBuffer (already decoded)
    - createBufferSource() → connect GainNode → destination
    - gainNode.gain.value = velocity / 127
    - source.start()
       │
       ▼
  AudioContext renders buffer to output device
  (< 5ms from pad press to sound, on wired LAN)
```

### Velocity to Gain Mapping

Linear mapping is simple but perceptually uneven. A square-root curve feels more natural:

```javascript
function velocityToGain(velocity) {
  // Linear: return velocity / 127;
  // Perceptual (sqrt curve):
  return Math.sqrt(velocity / 127);
}
```

This gives better dynamic range at low velocities — soft hits still have audible body.

### Sample Preload Strategy

Never fetch on trigger. Always preload.

```javascript
// Preload all samples currently visible in the browser list
async function preloadVisible(samples) {
  const promises = samples.map(async (s) => {
    if (!bufferCache.has(s.id)) {
      const detail = await fetch(`/api/samples/${s.id}`).then(r => r.json());
      await loadSample(s.id, detail.url);
    }
  });
  await Promise.all(promises);
}
```

Strategy tiers:

| Tier | What | When |
|---|---|---|
| Eager | Load all pad-assigned samples at startup | Always |
| Lazy | Load on first click/play in browser UI | Fine for browsing |
| Background | Preload search results after query | Smooth UX |

### Handling Tempo Matching

This version does not do real-time tempo stretching. Instead:

- Samples are tagged with BPM at index time.
- The UI filter allows you to search by BPM range.
- You play samples that are already at the right tempo.

Future enhancement: use `AudioBufferSourceNode.playbackRate` for coarse tempo shift:

```javascript
// Pseudocode: pitch-shift a 120bpm sample to 130bpm
const ratio = targetBpm / sourceBpm;  // 130/120 = 1.083
source.playbackRate.value = ratio;
// Note: this also shifts pitch — use a pitch-correcting approach for production
```

### Avoiding Audio Glitches

**Common causes and fixes:**

| Cause | Fix |
|---|---|
| AudioContext suspended | Call `audioCtx.resume()` on first user gesture |
| Buffer not preloaded | Always preload before pad assignment |
| GC pauses during playback | Pre-allocate GainNodes or use a pool |
| Network fetch on trigger | Never. Preload only. |
| Wi-Fi jitter | Use wired Ethernet on Pi |
| Clock drift (multiple sources) | Use `audioCtx.currentTime` for scheduled starts |

### Buffer Reuse Strategy

```javascript
// GainNode pool to reduce GC pressure
const gainPool = [];

function getGainNode() {
  if (gainPool.length > 0) return gainPool.pop();
  return audioCtx.createGain();
}

function releaseGainNode(node) {
  node.disconnect();
  gainPool.push(node);
}

function playSamplePooled(sampleId, velocity = 127) {
  const buffer = bufferCache.get(sampleId);
  if (!buffer) return;

  const source = audioCtx.createBufferSource();
  const gain = getGainNode();
  gain.gain.value = velocityToGain(velocity);
  gain.connect(audioCtx.destination);
  source.buffer = buffer;
  source.connect(gain);
  source.onended = () => releaseGainNode(gain);
  source.start();
}
```

### Memory Considerations

- Each decoded `AudioBuffer` uses approximately `duration_seconds × sample_rate × channels × 4 bytes`.
- A 4-second stereo 44.1kHz WAV ≈ 1.4 MB decoded.
- 100 preloaded samples ≈ ~140 MB RAM. This is fine on a modern laptop.
- If running the UI on a low-RAM device (e.g., Pi's own browser), limit preload to 16–32 samples at once.
- Evict least-recently-used buffers if cache size exceeds a threshold:

```javascript
const MAX_CACHED = 64;

function evictIfNeeded() {
  if (bufferCache.size > MAX_CACHED) {
    const firstKey = bufferCache.keys().next().value;
    bufferCache.delete(firstKey);
  }
}
```

---

## 7. Sample Organization Strategy

### Directory Layout for Sample Packs

Each pack lives in its own directory under `/mnt/samples/packs/`. Directory name = pack name.

```
/mnt/samples/packs/
├── kicks_garage_01/
│   ├── kick_80bpm_C_vibe_punchy.wav
│   ├── kick_120bpm_C_vibe_tight.wav
│   └── kick_140bpm_C_vibe_hard.wav
├── loops_house_deep_01/
│   ├── loop_120bpm_Am_vibe_deep.wav
│   ├── loop_124bpm_Fm_vibe_deep.wav
│   └── loop_128bpm_Cm_vibe_groove.wav
├── percussion_breakbeats_01/
│   ├── break_130bpm_vibe_dusty.wav
│   └── break_160bpm_vibe_choppy.wav
└── pads_ambient_01/
    ├── pad_Am_vibe_lush.wav
    └── pad_Dm_vibe_cold.wav
```

### Naming Convention

```
<type>_<bpm>bpm_<key>_vibe_<vibe>[_extra].wav
```

- `type`: what the sample is (`kick`, `snare`, `loop`, `pad`, `break`, `bass`, etc.)
- `bpm`: numeric BPM (integer or float)
- `key`: musical key (`Am`, `Cm`, `G`, `F#m`, etc.) — omit for atonal samples
- `vibe`: one-word mood descriptor (`deep`, `punchy`, `lush`, `dark`, `raw`, `airy`)
- `extra`: optional version or variant suffix

**Examples:**

```
kick_120bpm_vibe_hard.wav
loop_128bpm_Fm_vibe_tech.wav
pad_Am_vibe_lush_v2.wav
break_160bpm_vibe_choppy.wav
```

Samples with no BPM (one-shots, FX) can omit it:

```
fx_riser_vibe_dark.wav
snare_vibe_crispy.wav
```

### Tagging Strategy

Two-tier tagging:

1. **Filename-inferred tags** — extracted automatically by the indexer from the filename tokens.
2. **Manual override tags** — set via a sidecar `.tags` file or inline in the filename.

Sidecar example:

```
# loop_128bpm_Fm_vibe_tech.tags
bass,minimal,groove,fm,house
```

The indexer checks for a matching `.tags` file and merges with filename-derived tags.

### Bulk Import Script Design

```python
# indexer/bulk_import.py
"""
Bulk import WAV files from an external folder into the managed packs directory.
Normalizes filenames to the naming convention.
"""
import os
import shutil
import re
import argparse

def normalize_name(filename: str) -> str:
    """Strip spaces, lowercase, replace illegal chars."""
    name = filename.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_.]+", "", name)
    return name

def bulk_import(source_dir: str, pack_name: str, dest_root: str):
    dest = os.path.join(dest_root, pack_name)
    os.makedirs(dest, exist_ok=True)
    for fname in os.listdir(source_dir):
        if not fname.lower().endswith(".wav"):
            continue
        norm = normalize_name(fname)
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest, norm)
        shutil.copy2(src, dst)
        print(f"  {fname} → {norm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source directory of WAV files")
    parser.add_argument("pack_name", help="Target pack directory name")
    parser.add_argument("--dest", default="/mnt/samples/packs")
    args = parser.parse_args()
    bulk_import(args.source, args.pack_name, args.dest)
```

```bash
python bulk_import.py /path/to/raw/kicks kicks_garage_02
```

### BPM and Key Inference Options

| Method | Accuracy | Complexity |
|---|---|---|
| Filename parsing (primary) | High (if named well) | Low |
| `librosa.beat.beat_track()` | Medium (loops) | Medium |
| Manual tag via UI | Perfect | High effort |
| ML model (future) | High | High |

For quick BPM inference without librosa:

```bash
pip install librosa
python -c "
import librosa
y, sr = librosa.load('loop_unknown.wav', mono=True)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f'Estimated BPM: {tempo:.1f}')
"
```

### Future: ML Auto-Tagging

- Train a simple classifier on spectral features (MFCCs, chroma, spectral centroid).
- Predict vibe and key from audio content rather than filename.
- `scikit-learn` + `librosa` feature extraction → `RandomForestClassifier` or `SVM`.
- Label a few hundred samples by hand, train, auto-tag the rest.
- Store predicted tags with a confidence score in a separate `tags_ml` column.

---

## 8. Deployment & GitHub Pages Integration

### What Lives on the Pi

| Component | Location on Pi | Notes |
|---|---|---|
| WAV sample files | `/mnt/samples/packs/` | On USB SSD |
| SQLite index | `/mnt/samples/samples.db` | On USB SSD |
| FastAPI app | `/home/pi/pi-sample-server/api/` | systemd service |
| Web UI (HTML/JS) | `/home/pi/pi-sample-server/ui/` | Served by Nginx |
| Nginx config | `/etc/nginx/sites-enabled/pi-sample-server` | |
| Indexer scripts | `/home/pi/pi-sample-server/indexer/` | Run manually or via cron |

### What Lives on GitHub Pages

GitHub Pages (this site) hosts the **documentation only** — not the running service.

| Content | Location in repo |
|---|---|
| Project documentation | `docs/tutorials/just-for-fun/pi-sample-server.md` |
| Architecture diagrams | Embedded in this file (ASCII) |
| Setup instructions | Sections 2–4 above |
| Frontend code snippets | Section 5 above |

### systemd Service for FastAPI

```ini
# /etc/systemd/system/pi-sample-api.service
[Unit]
Description=Pi Sample Server API
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/pi-sample-server/api
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-sample-api
sudo systemctl start pi-sample-api
sudo systemctl status pi-sample-api
```

### Nginx TLS (Optional, for Web MIDI on non-localhost)

Web MIDI API requires HTTPS when not on `localhost`. Use a self-signed cert or mkcert:

```bash
# Install mkcert
sudo apt-get install -y libnss3-tools
wget -O mkcert https://dl.filippo.io/mkcert/latest?for=linux/arm64
chmod +x mkcert && sudo mv mkcert /usr/local/bin/

mkcert -install
mkcert pi.local 192.168.1.100

# Outputs: pi.local+1.pem and pi.local+1-key.pem
```

Add TLS to Nginx:

```nginx
server {
    listen 8443 ssl;
    ssl_certificate     /home/pi/pi.local+1.pem;
    ssl_certificate_key /home/pi/pi.local+1-key.pem;
    # ... rest of config same as above
}
```

### Navigation Integration

This document is linked from:
- `docs/tutorials/just-for-fun/index.md` — Just for Fun section listing
- `mkdocs.yml` — nav entry under `🎨 Just for Fun`

To link from another page:

```markdown
[Pi Sample Library Server](pi-sample-server.md)
```

---

## 9. Enhancements & Roadmap

### Real-Time Waveform Visualization

Use the WebAudio `AnalyserNode` to draw a live waveform or frequency spectrum:

```javascript
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 2048;
// Connect: source → analyser → gainNode → destination

const dataArray = new Uint8Array(analyser.frequencyBinCount);
function drawWaveform(canvas) {
  const ctx = canvas.getContext('2d');
  requestAnimationFrame(() => drawWaveform(canvas));
  analyser.getByteTimeDomainData(dataArray);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  for (let i = 0; i < dataArray.length; i++) {
    const x = (i / dataArray.length) * canvas.width;
    const y = (dataArray[i] / 128) * (canvas.height / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
```

### MIDI Clock Sync

Receive MIDI clock messages (status byte `0xF8`) to sync a local BPM counter:

```javascript
// MIDI clock: 24 pulses per quarter note
let clockTicks = 0;
let lastClockTime = null;
let currentBpm = null;

function handleMIDIClock(timestamp) {
  if (lastClockTime !== null) {
    const delta = (timestamp - lastClockTime) / 1000; // seconds
    currentBpm = 60 / (delta * 24);
  }
  lastClockTime = timestamp;
  clockTicks++;
}
```

Display live BPM in the UI and use it to filter samples by proximity to live tempo.

### ADSR Envelope Controls

Add attack, decay, sustain, release controls per pad using `GainNode.gain` automation:

```javascript
function playWithADSR(buffer, { attack, decay, sustain, release }, velocity) {
  const source = audioCtx.createBufferSource();
  const gain = audioCtx.createGain();
  const now = audioCtx.currentTime;
  const peak = velocityToGain(velocity);

  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(peak, now + attack);
  gain.gain.linearRampToValueAtTime(peak * sustain, now + attack + decay);
  // Release triggered on note-off event

  source.buffer = buffer;
  source.connect(gain);
  gain.connect(audioCtx.destination);
  source.start();
  return { source, gain, releaseAt: now + attack + decay };
}
```

### Sample Layering

Assign multiple samples to one pad and trigger all simultaneously:

```javascript
const layerMap = new Map(); // padIndex → [sampleId, ...]

function playLayered(padIndex, velocity) {
  const ids = layerMap.get(padIndex) || [];
  ids.forEach(id => playSample(id, velocity));
}
```

### Multi-User LAN Jam Mode

- Broadcast pad triggers via WebSocket to all connected clients.
- All clients play the same samples in sync.
- Use `audioCtx.currentTime` + a shared timestamp offset to schedule synchronized playback.
- WebSocket server: add a simple `websockets` endpoint to FastAPI.

```python
# Pseudocode addition to main.py
from fastapi import WebSocket

@app.websocket("/ws/jam")
async def jam_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        # Broadcast to all connected clients
        await broadcast(data)
```

### Dockerized Deployment Option

```yaml
# docker-compose.yml (see Appendix for full version)
services:
  api:
    build: ./api
    volumes:
      - /mnt/samples:/mnt/samples:ro
    ports:
      - "8000:8000"
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/pi-sample-server.conf:/etc/nginx/conf.d/default.conf
      - /mnt/samples/packs:/srv/static:ro
      - ./ui:/srv/ui:ro
    ports:
      - "8080:8080"
    depends_on:
      - api
```

### Pi Cluster Mode

Because why not.

- Run multiple Pi nodes, each with a portion of the sample library.
- Each node runs its own FastAPI + Nginx.
- A coordinator node (or the client browser) queries all nodes and merges results.
- Use mDNS (`avahi`) for automatic node discovery.
- Distribute samples by genre or BPM range across nodes.
- This is maximally silly and maximally fun.

```
  pi-node-01 (kicks & drums)      → 192.168.1.101
  pi-node-02 (loops & bass)       → 192.168.1.102
  pi-node-03 (pads & atmospheres) → 192.168.1.103
  coordinator                     → 192.168.1.100

  Client queries /api/samples across all nodes → merges → renders
```

---

## 10. Appendix

### Full SQLite Schema

```sql
-- Full schema with all indexes
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    bpm         REAL,
    key         TEXT,
    vibe        TEXT,
    tags        TEXT,
    duration_ms INTEGER,
    filepath    TEXT NOT NULL UNIQUE,
    pack        TEXT,           -- derived from parent directory name
    imported_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_bpm      ON samples(bpm);
CREATE INDEX IF NOT EXISTS idx_key      ON samples(key);
CREATE INDEX IF NOT EXISTS idx_vibe     ON samples(vibe);
CREATE INDEX IF NOT EXISTS idx_pack     ON samples(pack);
CREATE INDEX IF NOT EXISTS idx_filename ON samples(filename);

-- Optional: ML tag predictions
CREATE TABLE IF NOT EXISTS tags_ml (
    sample_id   INTEGER REFERENCES samples(id),
    tag         TEXT,
    confidence  REAL,
    model       TEXT
);
```

### Full Nginx Config

```nginx
# /etc/nginx/sites-enabled/pi-sample-server
server {
    listen 8080;
    server_name pi.local raspberrypi.local 192.168.1.100;

    client_max_body_size 0;

    # Gzip for API JSON responses
    gzip on;
    gzip_types application/json;

    # Static WAV files (direct, no API)
    location /static/ {
        alias /mnt/samples/packs/;
        add_header Accept-Ranges bytes;
        add_header Cache-Control "public, max-age=86400";
        add_header Access-Control-Allow-Origin "*";
        types {
            audio/wav wav;
            audio/x-wav wav;
        }
        sendfile on;
        tcp_nopush on;
    }

    # FastAPI proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30;
    }

    # WebSocket (jam mode)
    location /ws/ {
        proxy_pass http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Frontend UI
    location / {
        root /home/pi/pi-sample-server/ui;
        index index.html;
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache";
    }
}
```

### Example docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    restart: unless-stopped
    volumes:
      - samples_data:/mnt/samples:ro
    environment:
      - DB_PATH=/mnt/samples/samples.db
      - SAMPLES_ROOT=/mnt/samples/packs
    expose:
      - "8000"

  nginx:
    image: nginx:1.25-alpine
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./nginx/pi-sample-server.conf:/etc/nginx/conf.d/default.conf:ro
      - samples_data:/mnt/samples:ro
      - ./ui:/home/pi/pi-sample-server/ui:ro
    depends_on:
      - api

volumes:
  samples_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/samples
```

**api/Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**api/requirements.txt:**

```
fastapi>=0.110.0
uvicorn>=0.29.0
```

### Troubleshooting

**API returns no results:**

```bash
# Check database has records
sqlite3 /mnt/samples/samples.db "SELECT COUNT(*) FROM samples;"

# Re-run indexer
python /home/pi/pi-sample-server/indexer/index_samples.py

# Check FastAPI logs
sudo journalctl -u pi-sample-api -f
```

**MIDI device not visible in browser:**

```bash
# Verify device is detected by OS
lsusb
aconnect -l

# Check browser: must be Chromium/Chrome
# Check: page must be served over HTTPS or localhost for Web MIDI
# Try: chrome://flags → enable Web MIDI (if needed on older versions)
```

**WAV file not playing (fetch error):**

```bash
# Check Nginx static path
curl -I http://pi.local:8080/static/packs/your_pack/your_file.wav

# Verify Nginx alias path matches actual directory
ls /mnt/samples/packs/

# Check Nginx error log
sudo tail -f /var/log/nginx/error.log
```

**AudioContext suspended (no sound):**

```javascript
// Add this to any user interaction handler (button click, etc.)
if (audioCtx.state === 'suspended') {
  audioCtx.resume().then(() => console.log('AudioContext resumed'));
}
```

### Latency Debugging Tips

| Symptom | Tool | Fix |
|---|---|---|
| High MIDI-to-sound latency | Chrome DevTools → Performance | Reduce buffer size in AudioContext options |
| Audio dropouts | `top` on Pi → CPU | Reduce FastAPI workers or offload indexing |
| Slow sample preload | Browser Network tab | Check WAV file size; use smaller samples |
| Wi-Fi jitter | `ping -i 0.01 pi.local` | Switch to wired Ethernet |
| `AudioContext` not starting | Console errors | Ensure `resume()` is called on user gesture |

**Target latency budget:**

```
  MIDI physical → USB HID → OS:      ~1ms
  Web MIDI API event dispatch:        ~1ms
  JS handler + buffer lookup:        ~0.5ms
  AudioContext schedule + render:    ~5ms (at 128-sample buffer, 44.1kHz)
  ─────────────────────────────────────────
  Total target:                      < 10ms perceived
```

If perceived latency exceeds 20ms, check AudioContext `baseLatency` and `outputLatency`:

```javascript
console.log('Base latency:', audioCtx.baseLatency);
console.log('Output latency:', audioCtx.outputLatency);
```

Adjust `latencyHint` if needed:

```javascript
// 'interactive' = lowest latency; 'balanced' = more stable
const audioCtx = new AudioContext({ latencyHint: 'interactive' });
```

---

*Built with a Pi, a USB cable, and too much free time. Go make something.*

!!! tip "See also"
    - [RKE2 on Raspberry Pi Farm](../docker-infrastructure/rke2-raspberry-pi.md) — put a full Kubernetes cluster on the same hardware
    - [OSC + MQTT + Prometheus + SuperCollider](osc-mqtt-prometheus-supercollider.md) — make your metrics audible; pairs well with the WebAudio engine here
    - [Docker & Compose Best Practices](../../best-practices/docker-infrastructure/docker-and-compose.md) — containerize the FastAPI + Nginx stack
