---
tags:
  - docker
  - performance
  - security
---

# Multi-Stage Docker: Conda Build → Slim Runtime → scratch

**Objective**: Keep Conda's dependency-resolution ergonomics for building, then ship a final image that starts from `scratch` — zero OS layer, zero shell, zero package manager. You copy in exactly what your app needs to run and nothing else.

scratch is where convenience goes to die. You must supply every file your process will ever open: the Python interpreter, the Conda environment tree, every shared library it links against, TLS certificates if you hit HTTPS, and a minimal `/etc/passwd` if your app checks user identity. Miss one and you get a cryptic runtime crash. Get it right and your image is 80–95% smaller than the Conda base and has almost no CVE surface.

---

## Table of Contents

1. [When You Should NOT Do This](#1-when-you-should-not-do-this)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [Project Layout](#4-project-layout)
5. [environment.yml](#5-environmentyml)
6. [The Application](#6-the-application)
7. [The Dockerfile](#7-the-dockerfile)
8. [Discovering Required Shared Libraries](#8-discovering-required-shared-libraries)
9. [CA Certificates, DNS, and TLS](#9-ca-certificates-dns-and-tls)
10. [Non-Root User in scratch](#10-non-root-user-in-scratch)
11. [Build and Run](#11-build-and-run)
12. [Troubleshooting](#12-troubleshooting)
13. [Security Notes](#13-security-notes)
14. [See Also](#14-see-also)

---

## 1. When You Should NOT Do This

scratch is a deliberate choice that comes with real costs. Be honest with yourself first.

| Situation | Better choice |
|---|---|
| You need `glibc` and don't know every `.so` your app pulls in | `debian:slim` or Wolfi |
| You want a shell for debugging production issues | `gcr.io/distroless/python3` |
| Your app uses native C extensions beyond numpy/scipy basics | `debian:slim` + careful pruning |
| You need timezone data (`/usr/share/zoneinfo`) | `gcr.io/distroless/python3` |
| You need locale data (`/usr/lib/locale`) | `debian:slim` |
| Your build timeline is tight and image size is not critical | Stop here |

**Distroless is usually the right answer** for Python apps that need HTTPS. It gives you glibc, certs, and tzdata — without apt, bash, or other attack surface — at ~20 MB overhead vs scratch. Consider this tutorial a learning exercise and an optimization tool for when you've exhausted the easy alternatives.

---

## 2. Architecture Overview

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1: conda-builder                                         │
  │  Base: continuumio/miniconda3 (or mambaforge)                   │
  │  - conda env create -f environment.yml --prefix /opt/conda/envs/app
  │  - conda-pack (optional) or direct copy                         │
  │  Output: /opt/conda/envs/app/  (full env, ~500MB typical)       │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ COPY --from=conda-builder
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 2: runtime-collector                                     │
  │  Base: debian:bookworm-slim (has ldd, find, cp)                 │
  │  - Copy conda env                                               │
  │  - Run ldd on python + .so modules                              │
  │  - Collect all linked .so files into /runtime/lib               │
  │  - Copy app code into /opt/app                                  │
  │  Output: /runtime/  (stripped runtime tree, ~100–200 MB)        │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ COPY --from=runtime-collector
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 3: scratch (Variant A) or distroless (Variant B)         │
  │  - /runtime/conda/  (env tree)                                  │
  │  - /runtime/lib/    (shared libs)                               │
  │  - /opt/app/        (application code)                          │
  │  - /etc/passwd      (non-root user record)                      │
  │  - /etc/ssl/certs/  (if TLS needed)                             │
  │  ENTRYPOINT: /runtime/conda/bin/python -m app                   │
  │  Final image: ~80–300 MB depending on deps                      │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 3. Prerequisites

- Docker 23+ with BuildKit enabled (default in Docker Desktop and recent Docker Engine)
- A working Conda or Mamba installation (for local testing only — not needed in CI)

Enable BuildKit explicitly if needed:

```bash
export DOCKER_BUILDKIT=1
```

Or set it permanently in `/etc/docker/daemon.json`:

```json
{ "features": { "buildkit": true } }
```

---

## 4. Project Layout

```
conda-to-scratch/
├── Dockerfile
├── environment.yml
└── app/
    ├── __init__.py
    └── main.py
```

Keep it flat. The tutorial mounts nothing; everything is `COPY`-ed into the image at build time.

---

## 5. environment.yml

```yaml
# environment.yml
name: app
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11.*
  - numpy=1.26.*          # native .so extension — makes the .so story real
  - pip
  - pip:
      - fastapi==0.111.*
      - uvicorn[standard]==0.29.*
```

Pin every package. In a scratch image there is no recovery mechanism: if a package pulls in a new `.so` at build time you didn't account for, the container crashes silently at runtime.

---

## 6. The Application

A minimal FastAPI app. Small enough to be irrelevant to the Docker story; large enough to exercise HTTPS client calls if you extend it.

```python
# app/__init__.py
# (empty)
```

```python
# app/main.py
import numpy as np
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "numpy_version": np.__version__}

@app.get("/compute")
def compute():
    arr = np.arange(1_000_000, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}
```

Run locally (outside Docker) to confirm it works before fighting the Dockerfile:

```bash
conda env create -f environment.yml
conda activate app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 7. The Dockerfile

### Variant A — Pure scratch (no shell, minimal attack surface)

```dockerfile
# syntax=docker/dockerfile:1.6
# ─── Stage 1: Build the Conda environment ────────────────────────────────────
FROM continuumio/miniconda3:24.1.2-0 AS conda-builder

# Use a fixed prefix — never build into the base env.
# Conda envs are NOT fully relocatable by default; we'll address this below.
ENV CONDA_ENV_PREFIX=/opt/conda/envs/app

COPY environment.yml /tmp/environment.yml

RUN conda env create \
      --file /tmp/environment.yml \
      --prefix ${CONDA_ENV_PREFIX} \
    && conda clean --all --yes \
    && find ${CONDA_ENV_PREFIX} -name "*.pyc" -delete \
    && find ${CONDA_ENV_PREFIX} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ─── Stage 2: Collect the runtime (shared libs + env) ────────────────────────
FROM debian:bookworm-slim AS runtime-collector

COPY --from=conda-builder /opt/conda/envs/app /runtime/conda

# Install tooling needed to walk the dynamic link graph.
# These tools do NOT end up in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      binutils \
      file \
    && rm -rf /var/lib/apt/lists/*

# Collect all shared libraries that the Python interpreter and installed
# .so extension modules link against.
#
# Strategy:
#   1. Find all ELF binaries and .so files in the env
#   2. Run ldd on each; extract the resolved library paths
#   3. Copy them into /runtime/lib (preserving symlinks)
#
# This will catch glibc, libstdc++, libgomp (numpy), etc.
RUN mkdir -p /runtime/lib && \
    find /runtime/conda \( -name "*.so" -o -name "*.so.*" -o -type f -name "python3*" \) \
      -exec file {} \; \
    | grep "ELF" \
    | awk -F: '{print $1}' \
    | sort -u > /tmp/elf_files.txt && \
    xargs -a /tmp/elf_files.txt ldd 2>/dev/null \
    | awk '/=>/ && !/not found/ {print $3}' \
    | sort -u \
    | grep -v "^$" > /tmp/libs.txt && \
    while IFS= read -r lib; do \
        if [ -f "$lib" ]; then \
            cp --preserve=links "$lib" /runtime/lib/ 2>/dev/null || true; \
        fi; \
    done < /tmp/libs.txt

# Copy the application code
COPY app/ /opt/app/

# ─── Stage 3A: scratch — zero OS layer ───────────────────────────────────────
FROM scratch AS final

# Conda env (Python interpreter + all packages)
COPY --from=runtime-collector /runtime/conda /runtime/conda

# System shared libraries collected in stage 2
COPY --from=runtime-collector /runtime/lib /runtime/lib

# Application code
COPY --from=runtime-collector /opt/app /opt/app

# Minimal passwd: scratch has no users. Without this, getpwuid() calls fail.
# Format: name:password:UID:GID:GECOS:home:shell
COPY --from=runtime-collector /etc/passwd /etc/passwd

# If your app makes outbound HTTPS calls, copy CA certs (see Section 9).
# COPY --from=runtime-collector /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Tell the dynamic linker where to find libraries.
# Without this, the interpreter finds your libs even on scratch.
ENV LD_LIBRARY_PATH=/runtime/lib:/runtime/conda/lib

# Run as non-root (UID 65532 — see Section 10)
USER 65532

EXPOSE 8000

# No shell in scratch — must use exec form (JSON array), never string form.
ENTRYPOINT ["/runtime/conda/bin/python", "-m", "uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000"]
```

### Variant B — distroless Python (debug-friendlier, cert-aware)

Use this when scratch is too painful but you still want a minimal, hardened image:

```dockerfile
# syntax=docker/dockerfile:1.6
# ─── Stage 1: Build ──────────────────────────────────────────────────────────
FROM continuumio/miniconda3:24.1.2-0 AS conda-builder

ENV CONDA_ENV_PREFIX=/opt/conda/envs/app
COPY environment.yml /tmp/environment.yml

RUN conda env create \
      --file /tmp/environment.yml \
      --prefix ${CONDA_ENV_PREFIX} \
    && conda clean --all --yes \
    && find ${CONDA_ENV_PREFIX} -name "*.pyc" -delete

# ─── Stage 2: distroless final ───────────────────────────────────────────────
# distroless/python3 includes: glibc, libstdc++, libgomp, CA certs, tzdata.
# It does NOT include: bash, sh, apt, pip, or any package manager.
FROM gcr.io/distroless/python3-debian12:nonroot AS final

COPY --from=conda-builder /opt/conda/envs/app /runtime/conda
COPY app/ /opt/app/

ENV PYTHONPATH=/opt/app
ENV LD_LIBRARY_PATH=/runtime/conda/lib

EXPOSE 8000
ENTRYPOINT ["/runtime/conda/bin/python", "-m", "uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000"]
```

Variant B gives you: working TLS, working timezone data, a read-only filesystem that's easy to enable, and a nonroot user — without needing to manually collect `.so` files.

---

## 8. Discovering Required Shared Libraries

The library collection script in Stage 2 is the most fragile part of this build. Here is a deeper look at each approach so you can fix it when it breaks.

### Method 1: ldd walk (used above)

```bash
# Inside the runtime-collector stage or locally on the built env:
ldd /runtime/conda/bin/python3.11

# Example output:
#   linux-vdso.so.1 (0x00007ffd...)       ← kernel virtual DSO, skip
#   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
#   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
#   /lib64/ld-linux-x86-64.so.2           ← the dynamic loader itself
```

**Critical:** `/lib64/ld-linux-x86-64.so.2` is the dynamic loader. Without it, you get:
```
exec /runtime/conda/bin/python3.11: no such file or directory
```
even though the file clearly exists. Copy it explicitly:

```bash
# In runtime-collector (add to the RUN block):
cp /lib64/ld-linux-x86-64.so.2 /runtime/lib/ld-linux-x86-64.so.2
```

And create the expected path in scratch via a symlink — but scratch has no shell to create symlinks. Instead, structure your copy target to match `/lib64/`:

```dockerfile
# In scratch stage — copy the loader to the path glibc expects:
COPY --from=runtime-collector /runtime/lib/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
```

### Method 2: lddtree (more thorough, handles transitive deps)

```bash
# In runtime-collector, install patchelf + lddtree:
apt-get install -y --no-install-recommends python3-pyelftools

# lddtree walks the full transitive dependency tree:
lddtree /runtime/conda/bin/python3.11
```

### Method 3: strace at runtime (discover what's missing after the fact)

Run the scratch container once with strace in a debug sidecar to see exactly which paths are `open()`-ed. More practical for iterating.

### Checking numpy specifically

numpy's `.so` extensions link against `libgomp` (OpenMP) and `libgfortran`. Don't miss them:

```bash
# Find numpy's binary extensions:
find /runtime/conda/lib/python3.11/site-packages/numpy -name "*.so" | head -5

# Check each:
ldd /runtime/conda/lib/python3.11/site-packages/numpy/core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so
# => libgomp.so.1, libgfortran.so.5, libblas.so.3, etc.
```

Add each discovered library to your `/runtime/lib` copy step.

---

## 9. CA Certificates, DNS, and TLS

### What breaks in scratch for HTTPS

In scratch there is no `/etc/ssl/`, no `/etc/resolv.conf`, no `/etc/hosts`, and no NSS configuration. When your app makes an HTTPS call:

| What's missing | Symptom |
|---|---|
| `/etc/ssl/certs/ca-certificates.crt` | `SSL: CERTIFICATE_VERIFY_FAILED` |
| `/etc/resolv.conf` | DNS lookup hangs or fails (Docker injects this at runtime — usually OK) |
| `/etc/hosts` | `localhost` resolution fails (Docker injects this too — usually OK) |
| `/etc/nsswitch.conf` | `getaddrinfo` uses wrong resolution order |

### Fix: copy the CA bundle from the collector stage

The `runtime-collector` stage is `debian:bookworm-slim` which has `ca-certificates` installed by default:

```dockerfile
# In runtime-collector — verify it exists:
RUN ls /etc/ssl/certs/ca-certificates.crt

# In the scratch stage — uncomment this line:
COPY --from=runtime-collector /etc/ssl/certs/ca-certificates.crt \
     /etc/ssl/certs/ca-certificates.crt
```

Then set the environment variable so Python's `ssl` module finds it:

```dockerfile
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### Fix: copy nsswitch.conf

Without this, `getaddrinfo` may skip DNS and fail to resolve names:

```dockerfile
COPY --from=runtime-collector /etc/nsswitch.conf /etc/nsswitch.conf
```

A minimal nsswitch.conf for a containerized app:

```
# /etc/nsswitch.conf (minimal)
hosts: files dns
```

---

## 10. Non-Root User in scratch

scratch has no OS, so it has no `/etc/passwd`, no `useradd`, and no concept of users beyond UIDs. If your app (or any library) calls `getpwuid()`, it will crash unless you supply a passwd entry.

### Create the minimal passwd file in the collector stage

```dockerfile
# In runtime-collector — create a nonroot user record:
RUN echo "nonroot:x:65532:65532:nonroot:/home/nonroot:/sbin/nologin" \
    >> /etc/passwd \
    && echo "nonroot:x:65532:" >> /etc/group \
    && mkdir -p /home/nonroot \
    && chown 65532:65532 /home/nonroot
```

### Copy into scratch

```dockerfile
COPY --from=runtime-collector /etc/passwd /etc/passwd
COPY --from=runtime-collector /etc/group  /etc/group
# If app writes to home:
COPY --from=runtime-collector --chown=65532:65532 /home/nonroot /home/nonroot
```

### Set USER

```dockerfile
USER 65532
```

Using the UID directly (not the name) avoids a `getpwnam()` call during image startup.

---

## 11. Build and Run

### Build

```bash
# Build Variant A (scratch):
docker build \
  --target final \
  --tag myapp:scratch \
  --progress=plain \
  .

# Build Variant B (distroless):
docker build \
  --file Dockerfile.distroless \
  --tag myapp:distroless \
  --progress=plain \
  .
```

### Run

```bash
# Variant A — scratch (no shell available for exec):
docker run --rm -p 8000:8000 myapp:scratch

# Test it:
curl http://localhost:8000/health
# {"status":"ok","numpy_version":"1.26.x"}

curl http://localhost:8000/compute
# {"mean":499999.5,"std":288675.13...}
```

### Verify image size

```bash
docker images myapp

# Expected approximate sizes:
# REPOSITORY   TAG          IMAGE ID       SIZE
# myapp        scratch      abc123...      ~180 MB   (Python + numpy env)
# myapp        distroless   def456...      ~220 MB   (+ distroless base layer)
# (base conda)              ghi789...      ~1.1 GB   (for comparison)
```

```bash
# Inspect the scratch image layers — should be very few:
docker history myapp:scratch

# Check what's in it (requires a running container or image extract):
docker save myapp:scratch | tar -tv | grep -v "^l" | sort -k5 -h | tail -20
```

---

## 12. Troubleshooting

### `exec /runtime/conda/bin/python: no such file or directory`

The file exists but the dynamic loader is missing. The kernel can't start the ELF binary without the interpreter path listed in the ELF header (`PT_INTERP` segment).

```bash
# Check what interpreter the binary expects:
file /runtime/conda/bin/python3.11
# => ELF 64-bit LSB executable, ... interpreter /lib64/ld-linux-x86-64.so.2

# Fix: copy the loader into the scratch image at the exact expected path:
COPY --from=runtime-collector /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
```

### `error while loading shared libraries: libX.so.Y: cannot open shared object file`

A shared library was missed in the collection step. Find it in the collector stage:

```bash
# Run an interactive collector container to debug:
docker run --rm -it --entrypoint bash \
  $(docker build -q --target runtime-collector .) \
  -c "ldd /runtime/conda/bin/python3.11"

# Then add the missing .so to /runtime/lib in the RUN block.
```

### `SSL: CERTIFICATE_VERIFY_FAILED`

CA certs are missing. Add the COPY step from Section 9.

```bash
# Quick test inside the container (distroless or debug build):
python -c "import ssl; print(ssl.get_default_verify_paths())"
```

### `getaddrinfo failed` / DNS not resolving

Check that Docker is injecting `/etc/resolv.conf` (it does by default). If you're using a custom DNS setup:

```bash
docker run --rm --dns 8.8.8.8 myapp:scratch
```

### Timezone / locale errors

scratch has no timezone data. Options:

1. Set `TZ=UTC` environment variable — most apps work fine with UTC
2. Copy `/usr/share/zoneinfo` from the collector stage (adds ~40 MB)
3. Switch to distroless (tzdata included)

```dockerfile
# UTC only — clean and predictable:
ENV TZ=UTC
```

### `ImportError: cannot import name 'X' from 'numpy'`

The numpy `.so` extension loaded but found a missing transitive library (`libgomp`, `libgfortran`, `libblas`). Run ldd on the specific numpy extension that fails and add the missing library.

---

## 13. Security Notes

**Minimize attack surface by design.** scratch has no shell, no package manager, no coreutils, no curl. An attacker who achieves code execution inside the container has nothing to pivot with — no wget, no bash, no netstat. This is the primary security advantage of scratch beyond image size.

**Pin everything.** Every unpinned package is a non-deterministic surface:

```yaml
# Bad — unpinned:
dependencies:
  - numpy

# Good — fully pinned:
dependencies:
  - numpy=1.26.4
  - python=3.11.9
```

**SBOM + image provenance.** Generate a software bill of materials at build time:

```bash
# Docker BuildKit SBOM (requires buildx):
docker buildx build \
  --sbom=true \
  --provenance=mode=max \
  --tag myapp:scratch \
  --output type=image,push=false \
  .

# Or use syft after build:
syft myapp:scratch -o spdx-json > sbom.spdx.json
```

**Scan before shipping:**

```bash
grype myapp:scratch
# or
trivy image myapp:scratch
```

scratch images typically produce far fewer CVEs than Debian/Alpine bases because there is simply less software present to have vulnerabilities.

**Don't embed secrets.** No `.env` files, no credentials in `environment.yml`, no API keys in `ARG` values (they appear in `docker history`). Use runtime secrets via environment variables or Docker secrets.

---

## 14. See Also

!!! tip "See also"
    - [Docker & Compose Best Practices](../../best-practices/docker-infrastructure/docker-and-compose.md) — the conceptual foundation; Bake, profiles, and security by default
    - [Slimming GPU Docker Images](slim-gpu-docker-images.md) — adjacent slim-image techniques for CUDA workloads
    - [Slimming TensorFlow GPU Images](slim-tf-gpu-images.md) — same multi-stage approach for TF GPU containers
    - [RKE2 on Raspberry Pi](rke2-raspberry-pi.md) — deploy your scratch image to a real Kubernetes cluster
    - [Compose Profiles Polyglot Stack](compose-profiles-polyglot-stack.md) — orchestrate scratch services alongside heavier services with Compose profiles
    - [SBOMs, Trivy Scans, and CVE Mitigation](../../best-practices/docker-infrastructure/docker-sbom-trivy-cve-mitigation.md) — scan your scratch image before it ships

---

*scratch is honest: it holds exactly what you put in it, and runs exactly what you built. No magic. No baggage. No excuses.*
