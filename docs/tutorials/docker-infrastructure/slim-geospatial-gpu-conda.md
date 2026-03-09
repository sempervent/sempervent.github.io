---
tags:
  - docker
  - geospatial
  - performance
---

# Slim Geospatial + GPU Containers: Conda (GDAL) → Production Image Without 12 GB Bloat

**Objective**: Build a Python geospatial + optional GPU stack with Conda (including GDAL, PROJ, GEOS), then ship a production image under 2–4 GB by being deliberate about every layer. No 12 GB monsters. No mystery bloat. No hand-waving about "just use conda-pack."

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Strategy Overview](#2-strategy-overview)
3. [Two Build Targets](#3-two-build-targets)
4. [The Example Application](#4-the-example-application)
5. [Environment Design Rules That Prevent 12 GB](#5-environment-design-rules-that-prevent-12-gb)
6. [environment.yml — CPU and GPU Variants](#6-environmentyml--cpu-and-gpu-variants)
7. [Dockerfile: Multi-Stage Slimming](#7-dockerfile-multi-stage-slimming)
8. [Audit and Prove You Slimmed It](#8-audit-and-prove-you-slimmed-it)
9. [Common Bloat Traps](#9-common-bloat-traps)
10. [Production Hardening](#10-production-hardening)
11. [Troubleshooting](#11-troubleshooting)
12. [See Also](#12-see-also)

---

## 1. The Problem

A naive geospatial + GPU Conda image reaches 8–12 GB before your app code touches the filesystem. Here is why:

| Contributor | Typical size |
|---|---|
| `miniconda3` base image | ~400 MB |
| GDAL + GEOS + PROJ + HDF5 + NetCDF | ~600–900 MB |
| `cuda-toolkit` (full) from Conda | ~2–4 GB |
| `pytorch` with CUDA (default wheel) | ~2.5–3 GB |
| Build tools kept in final image (`gcc`, `make`, `gdal-dev`) | ~300–600 MB |
| Conda package cache left in image | ~500 MB–2 GB |
| Pip wheel cache left in image | ~200–500 MB |
| **Total (worst case)** | **~8–12 GB** |

Every item on that list is avoidable or reducible without dropping functionality.

**Target sizes (realistic, not theoretical):**

| Image | Expected size |
|---|---|
| CPU-only (GDAL + FastAPI) | 1.2–1.8 GB |
| GPU runtime (CUDA runtime base + CuPy) | 2.5–3.5 GB |
| GPU runtime (CUDA runtime base + PyTorch slim) | 3.5–5 GB |

---

## 2. Strategy Overview

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1: builder                                               │
  │  Base: mambaorg/micromamba:1.5-bookworm-slim                    │
  │  - Create env at /opt/env (fixed prefix)                        │
  │  - Install only runtime packages (no -dev, no compiler tools)   │
  │  - Run sanity imports: python -c "from osgeo import gdal"       │
  │  - conda clean -a && pip cache purge                            │
  │  Output: /opt/env/  (~800 MB CPU, ~2 GB GPU)                    │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ COPY --from=builder /opt/env
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 2a: runtime-cpu                                          │
  │  Base: debian:bookworm-slim                                     │
  │  - Install OS runtime deps (libstdc++, libgomp, ca-certs)       │
  │  - COPY /opt/env from builder                                   │
  │  - Set PATH, GDAL_DATA, PROJ_LIB                                │
  │  - Run as nonroot, expose 8000                                  │
  │  Final image: ~1.2–1.8 GB                                       │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 2b: runtime-gpu                                          │
  │  Base: nvidia/cuda:12.4.1-runtime-ubuntu22.04                   │
  │  (CUDA runtime provided by base — DO NOT duplicate from Conda)  │
  │  - Install OS runtime deps                                      │
  │  - COPY /opt/env from builder                                   │
  │  - Set PATH, GDAL_DATA, PROJ_LIB, LD_LIBRARY_PATH              │
  │  Final image: ~2.5–3.5 GB                                       │
  └─────────────────────────────────────────────────────────────────┘
```

**Why NOT scratch?**

GDAL and its dependency tree (GEOS, PROJ, HDF5, NetCDF, libcurl, OpenSSL) involve dozens of shared libraries with complex `.so` version requirements and runtime data files (`GDAL_DATA`, `PROJ_LIB`). Tracking all of them manually is fragile and breaks on every package version bump. The multi-stage approach with a slim Debian or CUDA-runtime base gets you 90% of the size savings with 5% of the maintenance burden. See [Multi-Stage Docker: Conda Build → scratch](multistage-conda-to-scratch.md) if you still want to go all the way.

---

## 3. Two Build Targets

| Target | Base | CUDA | Use case |
|---|---|---|---|
| `runtime-cpu` | `debian:bookworm-slim` | None | Geospatial API on CPU, no GPU hardware |
| `runtime-gpu` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | Runtime only | GPU-accelerated raster processing (CuPy) |

Build them from the same `Dockerfile` using `--target`:

```bash
# CPU image:
docker build --target runtime-cpu -t myapp:cpu .

# GPU image:
docker build --target runtime-gpu -t myapp:gpu .
```

A single builder stage serves both. The env is built once; the final stage just picks a different base.

---

## 4. The Example Application

A FastAPI service that:

1. Accepts a GeoTIFF path (or generates a synthetic raster)
2. Uses GDAL to extract: bounding box, CRS/EPSG, resolution, band statistics
3. Optionally runs a GPU-accelerated band normalization via CuPy if a GPU is present

### Why this example?

It exercises GDAL's C library binding (the hardest part to get right in a slim image), uses PROJ for CRS info, and gives you a working HTTP endpoint to verify the image is functional.

### Project layout

```
slim-geospatial-gpu-conda/
├── Dockerfile
├── environment.cpu.yml
├── environment.gpu.yml
└── app/
    ├── __init__.py
    ├── main.py
    └── raster.py
```

### app/raster.py

```python
# app/raster.py
"""GDAL-based raster inspection utilities."""
from __future__ import annotations

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def synthetic_geotiff(path: str, width: int = 256, height: int = 256) -> None:
    """Write a tiny synthetic single-band GeoTIFF to *path* for testing."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([-180.0, 360.0 / width, 0, 90.0, 0, -180.0 / height])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    data = np.random.rand(height, width).astype(np.float32) * 100
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    ds.FlushCache()
    ds = None


def inspect_raster(path: str) -> dict:
    """Return metadata + band statistics for a GeoTIFF."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Cannot open: {path}")

    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = srs.GetAuthorityCode(None)

    result = {
        "driver": ds.GetDriver().ShortName,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "bands": ds.RasterCount,
        "epsg": epsg,
        "resolution_x": abs(gt[1]),
        "resolution_y": abs(gt[5]),
        "bbox": {
            "west": gt[0],
            "north": gt[3],
            "east": gt[0] + gt[1] * ds.RasterXSize,
            "south": gt[3] + gt[5] * ds.RasterYSize,
        },
        "bands_stats": [],
    }

    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        stats = band.ComputeStatistics(False)  # False = use all pixels
        result["bands_stats"].append({
            "band": i,
            "min": stats[0],
            "max": stats[1],
            "mean": stats[2],
            "stddev": stats[3],
        })

    ds = None
    return result


def gpu_normalize(path: str) -> dict:
    """
    Normalize band 1 to [0, 1] using CuPy if available; fall back to NumPy.
    Returns a summary of the normalized array (not the full array).
    """
    try:
        import cupy as cp
        xp = cp
        backend = "cupy"
    except ImportError:
        xp = np
        backend = "numpy"

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    arr = xp.asarray(data, dtype=xp.float32)
    arr_min, arr_max = float(xp.nanmin(arr)), float(xp.nanmax(arr))
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = xp.zeros_like(arr)

    return {
        "backend": backend,
        "original_min": arr_min,
        "original_max": arr_max,
        "normalized_mean": float(xp.nanmean(norm)),
    }
```

### app/main.py

```python
# app/main.py
import os
import tempfile
from fastapi import FastAPI, HTTPException
from app.raster import synthetic_geotiff, inspect_raster, gpu_normalize

app = FastAPI(title="Geospatial Slim Demo")

# Generate a synthetic raster once at startup for smoke-testing.
_SYNTHETIC = os.path.join(tempfile.gettempdir(), "synthetic.tif")
synthetic_geotiff(_SYNTHETIC)


@app.get("/health")
def health():
    from osgeo import gdal
    return {"status": "ok", "gdal_version": gdal.VersionInfo()}


@app.get("/inspect")
def inspect(path: str = _SYNTHETIC):
    """Return metadata and band statistics for a GeoTIFF."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return inspect_raster(path)


@app.get("/normalize")
def normalize(path: str = _SYNTHETIC):
    """Normalize band 1 to [0,1]. Uses CuPy if GPU is available."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return gpu_normalize(path)
```

### Verify locally (before building the image)

```bash
conda env create -f environment.cpu.yml --prefix /tmp/testenv
/tmp/testenv/bin/pip install fastapi uvicorn
/tmp/testenv/bin/python -c "from osgeo import gdal; print(gdal.VersionInfo())"
/tmp/testenv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

---

## 5. Environment Design Rules That Prevent 12 GB

Apply these rules before writing a single line of `environment.yml`.

### Use micromamba, not conda

```dockerfile
# Bad — slow solver, heavier base:
FROM continuumio/miniconda3 AS builder

# Good — fast solver, minimal base image:
FROM mambaorg/micromamba:1.5-bookworm-slim AS builder
```

micromamba is a C++ reimplementation of conda's package manager. Same package ecosystem, ~10× faster solves, no Python bootstrap overhead.

### Pin Python and every native lib

```yaml
# Bad:
dependencies:
  - gdal
  - python

# Good:
dependencies:
  - python=3.11.*
  - gdal=3.8.*
  - proj=9.3.*
  - geos=3.12.*
```

A minor version bump in GDAL or PROJ can pull in a different libcurl or OpenSSL, adding 100–200 MB silently.

### Use conda-forge consistently — never mix channels without a channel priority rule

```yaml
# Bad — mixing channels without priority causes duplicate packages:
channels:
  - defaults
  - conda-forge

# Good — conda-forge wins everywhere:
channels:
  - conda-forge
  - nodefaults
```

Channel mixing is the #1 source of duplicate native libraries. `nodefaults` prevents the Anaconda defaults channel from leaking in.

### Prefer runtime CUDA libs, not the full toolkit

```yaml
# Bad — ships a 2–4 GB compiler toolchain:
dependencies:
  - cuda-toolkit=12.4.*

# Good — only the runtime your Python binding needs:
dependencies:
  - cuda-runtime=12.4.*  # or let the base image provide it
  - cupy=13.*            # already bundles what it needs
```

For CuPy: install it from conda-forge; it bundles its own CUDA runtime stubs. For the host CUDA runtime, let the `nvidia/cuda` base image provide it — don't duplicate it from Conda.

### Never include `-dev`, `-devel`, or build tools in the runtime env

```yaml
# Bad (dev headers + compiler):
- gdal-dev
- libgdal-dev
- gcc_linux-64
- make

# Good (runtime only):
- gdal=3.8.*
- libgdal=3.8.*  # shared libraries only
```

### Clean before COPY

```dockerfile
# In builder stage, always clean before the COPY layer is sealed:
RUN micromamba clean --all --yes && \
    pip cache purge 2>/dev/null || true && \
    find /opt/env -name "*.pyc" -delete && \
    find /opt/env -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

Conda's package cache stays in `$MAMBA_ROOT_PREFIX/pkgs/`. If you don't clean it and it lives inside `/opt/env`, you ship it. On a GPU env this can be 1–2 GB.

---

<span id="6-environmentyml--cpu-and-gpu-variants"></span>
## 6. environment.yml — CPU and GPU Variants

### environment.cpu.yml

```yaml
# environment.cpu.yml — minimal geospatial Python runtime, no GPU
name: geoapp
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.11.*
  - gdal=3.8.*
  - proj=9.3.*
  - geos=3.12.*
  - numpy=1.26.*
  - rasterio=1.3.*    # optional; remove if you use GDAL directly
  - pip
  - pip:
      - fastapi==0.111.*
      - uvicorn[standard]==0.29.*
      - python-multipart==0.0.9
```

### environment.gpu.yml

```yaml
# environment.gpu.yml — adds CuPy for GPU raster acceleration
# CUDA runtime is provided by the nvidia/cuda base image.
# Do NOT add cuda-toolkit here — it would duplicate and bloat.
name: geoapp
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.11.*
  - gdal=3.8.*
  - proj=9.3.*
  - geos=3.12.*
  - numpy=1.26.*
  - rasterio=1.3.*
  - cupy=13.*         # bundles CUDA stubs; picks up host CUDA at runtime
  - pip
  - pip:
      - fastapi==0.111.*
      - uvicorn[standard]==0.29.*
      - python-multipart==0.0.9
```

**CuPy vs PyTorch:** CuPy adds ~500 MB; PyTorch with CUDA adds 2.5–3 GB. If you only need array operations on raster bands, CuPy is the right choice. Use PyTorch only if you need neural network inference — and even then, consider ONNX Runtime with the CUDA execution provider (~300 MB) instead.

---

## 7. Dockerfile: Multi-Stage Slimming

```dockerfile
# syntax=docker/dockerfile:1.6
# ─── Stage 1: Build Conda environment ────────────────────────────────────────
FROM mambaorg/micromamba:1.5-bookworm-slim AS builder

# Build as root to allow env creation at /opt/env.
# micromamba base image uses $MAMBA_USER by default; override for fixed prefix.
USER root

# Accept build arg to choose cpu or gpu environment file.
ARG ENV_FILE=environment.cpu.yml
COPY ${ENV_FILE} /tmp/environment.yml

# Create the env at a fixed prefix.
# --no-deps is NOT used here — we want full dep resolution.
RUN micromamba env create \
      --file /tmp/environment.yml \
      --prefix /opt/env \
      --yes \
    && micromamba clean --all --yes \
    && /opt/env/bin/pip cache purge 2>/dev/null || true \
    && find /opt/env -name "*.pyc" -delete \
    && find /opt/env -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity check — fail the build early if GDAL import breaks.
RUN /opt/env/bin/python -c "from osgeo import gdal; print('GDAL OK:', gdal.VersionInfo())"
RUN /opt/env/bin/python -c "import numpy; print('NumPy OK:', numpy.__version__)"

# Copy the application into the builder layer (available to both targets below).
COPY app/ /opt/app/

# ─── Stage 2a: CPU runtime ────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime-cpu

# Install only the OS runtime libs that Conda env links against externally.
# libgomp1 = OpenMP (numpy, rasterio), ca-certificates = TLS, libstdc++6 = C++ runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user.
RUN useradd --uid 10001 --gid 0 --no-create-home --shell /sbin/nologin appuser

# Copy the Conda environment from builder.
COPY --from=builder /opt/env /opt/env

# Copy the application.
COPY --from=builder /opt/app /opt/app

# Tell GDAL and PROJ where their data directories are (inside the Conda env).
ENV PATH="/opt/env/bin:$PATH" \
    GDAL_DATA="/opt/env/share/gdal" \
    PROJ_LIB="/opt/env/share/proj" \
    GDAL_CACHEMAX=512 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/app
USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/opt/env/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["/opt/env/bin/uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000"]

# ─── Stage 2b: GPU runtime ────────────────────────────────────────────────────
# Use the CUDA runtime image — NOT the devel image.
# runtime = libcudart + cuBLAS runtime only.
# devel = full toolkit including compiler, nvcc, headers — never ship this.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu

# Ubuntu base needs the same OS libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --gid 0 --no-create-home --shell /sbin/nologin appuser

COPY --from=builder /opt/env /opt/env
COPY --from=builder /opt/app /opt/app

# CuPy looks for CUDA libs in LD_LIBRARY_PATH.
# The nvidia/cuda base image installs libs under /usr/local/cuda/lib64.
# Conda env may also have stubs under /opt/env/lib — include both.
ENV PATH="/opt/env/bin:$PATH" \
    GDAL_DATA="/opt/env/share/gdal" \
    PROJ_LIB="/opt/env/share/proj" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/env/lib" \
    GDAL_CACHEMAX=512 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/app
USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/opt/env/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

ENTRYPOINT ["/opt/env/bin/uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000"]
```

### Build commands

```bash
# CPU image:
docker build \
  --build-arg ENV_FILE=environment.cpu.yml \
  --target runtime-cpu \
  --tag geoapp:cpu \
  --progress=plain \
  .

# GPU image:
docker build \
  --build-arg ENV_FILE=environment.gpu.yml \
  --target runtime-gpu \
  --tag geoapp:gpu \
  --progress=plain \
  .
```

---

## 8. Audit and Prove You Slimmed It

### Size comparison

```bash
docker images geoapp

# Expected output:
# REPOSITORY   TAG    IMAGE ID       CREATED        SIZE
# geoapp       cpu    abc123...      1 minute ago   1.6 GB
# geoapp       gpu    def456...      2 minutes ago  3.1 GB
```

### Inspect layers

```bash
docker history geoapp:cpu

# The largest layer should be the COPY of /opt/env.
# No layer should contain conda pkgs cache or pip wheel cache.
```

### Inspect with dive (if installed)

```bash
dive geoapp:cpu
# Look for: wasted space in layers, files copied that shouldn't be there.
```

### Verify what's in the Conda env by size

```bash
docker run --rm geoapp:cpu \
  /opt/env/bin/conda list --prefix /opt/env \
  2>/dev/null || \
  docker run --rm --entrypoint /opt/env/bin/pip geoapp:cpu \
  list --format=columns
```

A faster size breakdown without conda:

```bash
docker run --rm geoapp:cpu \
  du -sh /opt/env/lib/python3.11/site-packages/* \
  | sort -h | tail -20
```

### Verify the app works

```bash
# Start:
docker run --rm -d -p 8000:8000 --name geo-test geoapp:cpu

# Health check:
curl http://localhost:8000/health
# {"status":"ok","gdal_version":"3080400"}

# Raster inspection (uses synthetic raster generated at startup):
curl "http://localhost:8000/inspect"
# {"driver":"GTiff","width":256,"height":256,"bands":1,"epsg":"4326",...}

# GPU normalization (falls back to NumPy on CPU image):
curl "http://localhost:8000/normalize"
# {"backend":"numpy","original_min":...,"normalized_mean":...}

# Stop:
docker stop geo-test
```

### GPU verify

```bash
# Requires: NVIDIA Container Toolkit installed on the host.
docker run --rm --gpus all geoapp:gpu \
  /opt/env/bin/python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
# Expected: 12040 (or similar CUDA version number)
```

---

## 9. Common Bloat Traps

**Installing `cuda-toolkit` from Conda when the base image already provides CUDA runtime.**
The full toolkit is 2–4 GB and contains compiler tooling you will never use at runtime. Your base image (`nvidia/cuda:*-runtime-*`) provides `libcudart`. CuPy's conda-forge package bundles the stubs it needs. That's all you need.

**Using `-devel` or `*-dev` NVIDIA base images.**
`nvidia/cuda:12.4.1-devel-ubuntu22.04` is 4 GB+. `nvidia/cuda:12.4.1-runtime-ubuntu22.04` is ~800 MB. Always use `runtime`.

**Installing `gdal-dev` or `libgdal-dev` in the runtime image.**
Development headers are needed to compile C extensions against GDAL. If your packages are already compiled (conda-forge provides pre-built binaries), you don't need headers at runtime. Never install `-dev` packages in the `runtime-*` stages.

**Keeping `gcc`, `make`, `cmake`, `ninja` in the final image.**
These are build-time tools. They belong only in the builder stage. If you find yourself adding them to the runtime stage, stop and ask why.

**Mixing `conda-forge` and `defaults` channels without explicit priority.**
This causes duplicate versions of shared libs (libcurl, OpenSSL, zlib) to be installed side by side. Always use `nodefaults` or pin `channel_priority: strict` in `.condarc`.

**Letting pip install Torch with CUDA wheels on top of a Conda env.**
A pip-installed PyTorch CUDA wheel is 2.5–3 GB and ships its own libcuda, libcublas, and cuDNN — duplicating what the base image already has. If you need PyTorch, install it from conda-forge and use the Conda env's GPU bindings.

**Not running `micromamba clean --all --yes` before the final COPY.**
The micromamba/conda package cache (`$MAMBA_ROOT_PREFIX/pkgs/`) contains tarballs of every downloaded package. On a large GPU env this is 1–2 GB. If `/opt/env` and `$MAMBA_ROOT_PREFIX` are the same path, clean before copying.

**Setting `COPY /opt/env /opt/env` before running `conda clean`.**
If you clean after the COPY layer, Docker's layer cache won't help — the bloated layer is already sealed. Always clean inside the builder stage's `RUN` block, before any `COPY` from builder.

---

## 10. Production Hardening

### Non-root user

Already implemented above (`useradd --uid 10001`, `USER 10001`). The UID 10001 avoids collisions with common system UIDs and aligns with distroless conventions.

### Healthcheck

The `HEALTHCHECK` instruction in the Dockerfile uses Python's built-in `urllib.request` — no `curl` dependency in the runtime image:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/opt/env/bin/python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
```

### Environment variables for GDAL/PROJ

These must be set correctly or GDAL will silently fail to interpret CRS information:

```dockerfile
ENV GDAL_DATA="/opt/env/share/gdal" \
    PROJ_LIB="/opt/env/share/proj"
```

Verify them inside the running container:

```bash
docker run --rm geoapp:cpu /opt/env/bin/python -c \
  "from osgeo import gdal, osr; srs = osr.SpatialReference(); print(srs.ImportFromEPSG(4326))"
# Expected: 0 (OGRERR_NONE)
```

### Read-only filesystem

Run with `--read-only` and a tmpfs for the synthetic raster write:

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,size=64m \
  -p 8000:8000 \
  geoapp:cpu
```

The synthetic raster is written to `/tmp` (tmpfs). Everything else is read-only.

### GPU runtime: NVIDIA Container Toolkit assumption

Running `geoapp:gpu` requires:

1. NVIDIA driver ≥ 525 on the host
2. `nvidia-container-toolkit` installed
3. `--gpus all` passed to `docker run`

```bash
docker run --rm --gpus all -p 8000:8000 geoapp:gpu
```

In Kubernetes, use the NVIDIA device plugin and request GPU resources:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## 11. Troubleshooting

### `PROJ: PROJ error: proj_create: Cannot find proj.db`

PROJ can't find its database. Set `PROJ_LIB` to the correct path inside the Conda env:

```bash
# Find the correct path:
docker run --rm geoapp:cpu \
  find /opt/env -name "proj.db" 2>/dev/null
# => /opt/env/share/proj/proj.db

# Fix in Dockerfile:
ENV PROJ_LIB="/opt/env/share/proj"
```

### `GDAL_DATA not found` / `Unable to open EPSG support file`

Same pattern as PROJ. GDAL's data directory contains datum grid files and EPSG definitions:

```bash
# Find it:
docker run --rm geoapp:cpu \
  find /opt/env -name "epsg" -o -name "gcs.csv" 2>/dev/null | head -3
# => /opt/env/share/gdal/epsg, /opt/env/share/gdal/gcs.csv

ENV GDAL_DATA="/opt/env/share/gdal"
```

### `ImportError: libgdal.so.X: cannot open shared object file`

The Conda env's `lib/` directory is not on `LD_LIBRARY_PATH`. The runtime-cpu stage uses `debian:bookworm-slim` which won't automatically find Conda libs:

```dockerfile
# Add to ENV in runtime-cpu:
ENV LD_LIBRARY_PATH="/opt/env/lib"
```

Or run `ldconfig` after copying the env (preferable — avoids runtime env var dependency):

```dockerfile
RUN ldconfig /opt/env/lib
```

### `ModuleNotFoundError: No module named 'app'`

`WORKDIR` is set to `/opt/app` but the module path isn't on `PYTHONPATH`. Fix:

```dockerfile
ENV PYTHONPATH="/opt/app"
```

Or ensure the entrypoint runs from the correct directory:

```dockerfile
WORKDIR /opt/app
ENTRYPOINT ["/opt/env/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CUDA version mismatch: `CUDA error: no kernel image is available for execution on the device`

CuPy was compiled for a different CUDA version than the host driver supports. Check:

```bash
# CuPy's expected CUDA version:
docker run --rm geoapp:gpu \
  /opt/env/bin/python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"

# Host CUDA version:
nvidia-smi | grep "CUDA Version"
```

The CUDA runtime version in the image must be ≤ the driver's supported CUDA version. If mismatched, pin a lower `cupy` version that targets an earlier CUDA toolkit.

### Container is too slow to start (GDAL scanning plugins)

GDAL scans plugin directories at import time. Disable plugin scanning if you don't use GDAL plugins:

```dockerfile
ENV GDAL_DRIVER_PATH="disable"
```

This can cut GDAL import time from ~2s to ~200ms in some environments.

---

## 12. See Also

!!! tip "See also"
    - [Multi-Stage Docker: Conda Build → scratch](multistage-conda-to-scratch.md) — go all the way to a zero-OS image; understand where this tutorial stops short
    - [Slimming GPU Docker Images](slim-gpu-docker-images.md) — GPU slimming patterns for non-geospatial workloads
    - [Slimming TensorFlow GPU Images](slim-tf-gpu-images.md) — same multi-stage discipline applied to TensorFlow
    - [Docker & Compose Best Practices](../../best-practices/docker-infrastructure/docker-and-compose.md) — foundational Bake, profiles, and security patterns
    - [PostGIS Best Practices](../../best-practices/postgres/postgis-best-practices.md) — when geospatial processing belongs in the database, not the container
    - [RKE2 on Raspberry Pi](rke2-raspberry-pi.md) — deploy these containers on a real Kubernetes cluster
    - [GeoParquet Best Practices](../../best-practices/database-data/geoparquet.md) — the columnar-file complement to GDAL raster workflows

---

*A slim geospatial image is a deliberate geospatial image. Measure first, optimize second, and document every byte you removed.*
