# Slimming Down GPU-Enabled TensorFlow Images

**Objective**: Transform bloated TensorFlow GPU containers from gigabytes to megabytes. Cut cold start times, reduce storage costs, and maintain CUDA compatibility while keeping your models running fast.

Big images bleed time and money. Cut your TensorFlow GPU containers to size without breaking CUDA. This guide shows you how to build lean, mean inference machines that start fast and run efficiently.

## 0) Principles (Read Once, Live by Them)

### The Five Commandments

1. **Pick the smallest correct base**
   - Classic: `nvidia/cuda:<ver>-runtime-ubuntu22.04` (not `-devel`)
   - Modern small: plain Debian/Ubuntu + `pip install 'tensorflow[and-cuda]==<ver>'` (TensorFlow 2.13+)

2. **Multi-stage every build**
   - Compile/resolve in a fat builder, copy only what you need

3. **Exploit BuildKit**
   - Cache mounts for apt/pip, secrets for private indexes

4. **No extras**
   - `--no-install-recommends`, remove apt lists, nuke caches, skip dev tools

5. **Pin everything**
   - TensorFlow, CUDA/cuDNN (or the meta CUDA wheels), Python

**Why These Principles**: GPU containers are expensive to build and run. Every MB saved is money saved and performance gained.

## 1) Approach A — CUDA Runtime Base (Classic & Predictable)

### Minimal Multi-Stage for TF Inference

```dockerfile
# syntax=docker/dockerfile:1.9

############################
# 1) Builder: resolve wheels
############################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Cache pip, lock versions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install \
      tensorflow==2.16.1 \
      # ↑ GPU build will use system CUDA in the runtime stage
      fastapi==0.111.* uvicorn[standard]==0.30.* \
      numpy==2.* pillow==10.* protobuf~=4.25

# [Optional] your app deps
# COPY requirements.txt /tmp/requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt

# Trim common cruft inside site-packages (safe-ish; validate!)
RUN find /opt/venv/lib -type d -regex ".*\(tests\|__pycache__\|\.dist-info/.*-nspkg\).*" -prune -exec rm -rf {} + || true

############################
# 2) Runtime: CUDA runtime only
############################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring the venv only
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY app/ /app/

# Last cleanup sweep
RUN rm -rf /root/.cache/*

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

**Why This Slims Down**:
- Final image carries runtime CUDA only + a single venv
- No compilers or headers in production
- Test files and `__pycache__` removed

⚠️ **Match TF ↔ CUDA ↔ cuDNN versions**. If TF can't find compatible libs, switch to Approach B.

## 2) Approach B — tensorflow[and-cuda] (Ships CUDA via pip)

### Slim Inference Image (No CUDA Base)

```dockerfile
# syntax=docker/dockerfile:1.9
FROM python:3.12-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=1  # fewer logs

# System libs TensorFlow uses at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# BuildKit pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install "tensorflow[and-cuda]==2.16.1" \
                fastapi==0.111.* uvicorn[standard]==0.30.* numpy==2.* pillow==10.* protobuf~=4.25

# Optional trims (validate after!)
RUN python - <<'PY'
import os, shutil, sys
base = next(p for p in sys.path if p.endswith('site-packages'))
trash = ['tensorflow/include','tensorflow/python/_pywrap_tensorflow_internal_d.so.debug']
for t in trash:
    p=os.path.join(base,'tensorflow',*t.split('/'))
    if os.path.exists(p):
        try: os.remove(p)
        except IsADirectoryError: shutil.rmtree(p, ignore_errors=True)
PY

WORKDIR /app
COPY app/ /app/

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

**Why This Slims Down**:
- No CUDA base; CUDA/cuDNN arrive as tightly scoped pip wheels
- You still need the NVIDIA Container Toolkit on the host to expose the GPU, but the image itself is leaner

**Notes**:
- Keep to matching TF ↔ CUDA combos (e.g., TF 2.16 → CUDA 12.3/12.4 wheels)
- This path is great for inference. For custom ops/compilation, use Approach A (devel builder)

## 3) Build Hygiene & Layer Diet

### APT Pattern

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends <pkgs> \
 && rm -rf /var/lib/apt/lists/*
```

### PIP Pattern

```dockerfile
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
```

### Purge Typical Bloat

```dockerfile
RUN find / -type d -regex '.*\(__pycache__\|tests\|\.pytest_cache\|\.cache\).*' -prune -exec rm -rf {} + || true
```

### Strip Debug Symbols (C++ Extensions) — Use with Caution

```dockerfile
RUN find / -type f -name "*.so*" -exec sh -c 'file -b "{}" | grep -q ELF && strip --strip-unneeded "{}" || true' \;
```

**Why These Patterns**: Cache mounts speed rebuilds, `--no-install-recommends` prevents bloat, cleanup removes unnecessary files.

## 4) Training vs Inference: Size Levers

### Inference Images

```dockerfile
# Prefer Approach B; no compilers; bake models; remove tensorboard & dev extras
FROM python:3.12-slim

# Install only inference essentials
RUN pip install "tensorflow[and-cuda]==2.16.1" fastapi uvicorn

# Remove training extras
RUN pip uninstall -y tensorboard jupyter ipython
```

**Why Inference Optimization**: No training tools needed, smaller models, faster startup.

### Training Images

```dockerfile
# Use Approach A with a devel builder stage to compile custom ops
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install training essentials
RUN pip install tensorflow==2.16.1 tensorboard jupyter

# Copy only runtime artifacts to final stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
COPY --from=builder /opt/venv /opt/venv
```

**Why Training Optimization**: Custom ops need compilation, but production doesn't need dev tools.

## 5) Models & Assets: The Silent Megabytes

### Externalize Large Models

```dockerfile
# Mount models at runtime instead of baking them in
VOLUME ["/app/models"]

# Or use a separate model service
ENV MODEL_SERVICE_URL=http://model-service:8000
```

### HuggingFace/TensorFlow Hub Caches

```dockerfile
ENV TFHUB_CACHE=/opt/cache/tfhub
RUN rm -rf /opt/cache/tfhub/**/downloads || true
```

### Keep One Model Format

```dockerfile
# Keep only SavedModel, remove TFLite duplicates
RUN find /app/models -name "*.tflite" -delete
```

**Why Model Management**: Models are often the largest component. Smart caching and externalization save gigabytes.

## 6) Runtime Hygiene for GPUs

### Minimal Capabilities

```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Docker Run

```bash
# Prefer specific GPU selection
docker run --gpus '"device=0"' tf-gpu:slim
```

### Docker Compose

```yaml
services:
  tf-app:
    image: tf-gpu:slim
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

**Why Runtime Hygiene**: Explicit capabilities prevent unnecessary driver overhead. Only request what you need.

## 7) Verify You Actually Lost Weight

### Build with BuildKit

```bash
# Build with BuildKit
DOCKER_BUILDKIT=1 docker build -t tf-gpu:slim .

# Compare
docker images | awk 'NR==1 || /tf-gpu:slim/'
docker history tf-gpu:slim --no-trunc

# Optional: layer-by-layer inspection
dive tf-gpu:slim
```

### Sanity Test (Container)

```bash
# Test TensorFlow and GPU detection
docker run --rm tf-gpu:slim python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
PY
```

**Why Verification**: Understanding image composition helps identify optimization opportunities.

## 8) Common Pitfalls (And Exits)

### "Could not load dynamic library 'libcudart…'"

```bash
# TF / CUDA mismatch
# On Approach A: ensure CUDA + cuDNN in the base match TF version
# On Approach B: don't mix system CUDA with pip CUDA wheels
```

### Giant Images After Pip

```bash
# You installed extras (jupyter, tensorboard, tests)
# Prune, or split dev vs runtime images
RUN pip uninstall -y jupyter tensorboard ipython
```

### Slow Cold Starts

```python
# Preload the model in app.startup so the first request isn't compiling graphs
@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model("/app/model")
```

**Why These Solutions**: Common problems have common solutions. Understanding them prevents production failures.

## 9) Example: Ultra-Lean TF Inference API (Approach B)

### Complete Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "tensorflow[and-cuda]==2.16.1" fastapi==0.111.* uvicorn[standard]==0.30.*

WORKDIR /app
COPY app/ /app/
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
```

### app/main.py (Skeletal)

```python
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()
model = None

@app.on_event("startup")
def load():
    global model
    model = tf.keras.models.load_model("/app/model")

@app.get("/health")
def health():
    return {"tf": tf.__version__, "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]}

@app.post("/predict")
def predict():
    # your preprocessing + model(...) here
    return {"ok": True}
```

## 10) Complete Example: Production Setup

### Project Structure

```
tf-gpu-app/
├── Dockerfile.approach-a
├── Dockerfile.approach-b
├── requirements.txt
├── docker-compose.yml
├── .dockerignore
└── app/
    ├── main.py
    ├── models/
    │   └── model.h5
    └── utils/
        └── preprocessing.py
```

### requirements.txt

```txt
tensorflow[and-cuda]==2.16.1
fastapi==0.111.0
uvicorn[standard]==0.30.0
numpy==2.0.0
pillow==10.1.0
protobuf~=4.25
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  tf-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ./app/models:/app/models
```

### .dockerignore

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
*.md
docs/

# Development
tests/
.pytest_cache/
.coverage
```

## 11) TL;DR Checklist

### Essential Optimizations

- [ ] **Choose base: CUDA runtime (A) or tensorflow[and-cuda] wheels (B)**
- [ ] **Multi-stage: resolve/build in builder; copy only venv/artifacts**
- [ ] **No dev tools in final; `--no-install-recommends`, delete apt lists**
- [ ] **Pin versions (TF, CUDA/cuDNN or pip CUDA wheels)**
- [ ] **Purge tests, `__pycache__`, caches; (carefully) strip .so symbols**
- [ ] **Externalize big models; pre-load on startup**
- [ ] **GPU runtime env: `NVIDIA_DRIVER_CAPABILITIES=compute,utility`**
- [ ] **Measure (`docker history`, `dive`) and sanity test (`tf.config.list_physical_devices('GPU')`)**

### Build Commands

```bash
# Build with BuildKit
DOCKER_BUILDKIT=1 docker build -t tf-gpu:slim .

# Build with cache
docker buildx build --cache-from type=local,src=/tmp/.buildx-cache --cache-to type=local,dest=/tmp/.buildx-cache .

# Run and test
docker run --rm tf-gpu:slim python -c "import tensorflow as tf; print(tf.__version__)"
```

### Size Comparison

```bash
# Before optimization
docker images | grep tf-gpu-before
# tf-gpu-before    latest    3.2GB

# After optimization
docker images | grep tf-gpu-after
# tf-gpu-after     latest    800MB
```

## 12) The Machine's Summary

TensorFlow GPU containers are bloated by default. The key is understanding what your application actually needs and eliminating everything else. Multi-stage builds, proper base image selection, and aggressive cleanup create lean, efficient containers.

**The Dark Truth**: GPU containers are expensive to build and run. Every MB saved is money saved and performance gained. The optimization process reveals what you actually need versus what you think you need.

**The Machine's Mantra**: "In minimalism we trust, in multi-stage we build, and in the slim container we find the path to efficient inference."

**Why This Matters**: GPU resources are expensive. Slim containers start faster, use less memory, and cost less to store and transfer.

---

*This tutorial provides the complete machinery for building slim TensorFlow GPU containers. The patterns scale from development to production, from megabytes to terabytes.*
