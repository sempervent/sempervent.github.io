# Slimming Down GPU-Based Docker Images

**Objective**: Transform bloated GPU containers from quarter-ton dev packages to fighting weight production images. Cut cold start times, reduce storage costs, and maintain CUDA compatibility.

When your container is dragging a quarter-ton of dev packages and orphaned toolchains, you're paying for cold starts with your soul. This guide shows how to cut GPU images down to fighting weight without breaking CUDA.

## 0) Principles (Read Once, Live by Them)

### The Five Commandments

1. **Choose the smallest correct base**
   - Prefer runtime images over devel: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` instead of `-devel`
   - Pull only what you need: cuDNN, TensorRT, etc., only if your code uses them
   - Avoid Alpine for CUDA (glibc required)

2. **Multi-stage always**
   - Build/compile in heavy stages, copy only artifacts into a slim runtime stage

3. **Exploit BuildKit**
   - Use `RUN --mount=type=cache` for apt/pip, `--mount=type=secret` for private indexes/tokens
   - Use targeted `COPY --link`

4. **Kill the fluff**
   - `--no-install-recommends`, delete apt lists, strip symbols
   - Prune caches (pip, npm, model hubs), remove tests/headers/CLI junk

5. **Freeze the world**
   - Pin CUDA, drivers, Python, framework wheels
   - Inference is not the time for surprise upgrades

**Why These Principles**: GPU containers are expensive to build and run. Every MB saved is money saved and performance gained.

## 1) Base Image Selection Cheatsheet

### CUDA Runtime Minimal (Ubuntu)

```dockerfile
# Production runtime (slim)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Development build (heavy)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
```

### cuDNN and TensorRT

```dockerfile
# Only if you need cuDNN
FROM nvidia/cuda:12.4.1-cudnn8-runtime-ubuntu22.04

# Only if you deploy TensorRT engines
FROM nvcr.io/nvidia/tensorrt:23.12-py3
```

### Python Considerations

```dockerfile
# Prefer distro Python baked into CUDA image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Or use slim CPython in builder stage
FROM python:3.12-slim AS builder
```

**Rule**: App stage == runtime; never ship a devel toolchain to production.

## 2) PyTorch Inference: Multi-Stage Pattern

### Complete Dockerfile (1.2–2.5 GB → 600–900 MB)

```dockerfile
# syntax=docker/dockerfile:1.9

############################
# 1) Builder: wheels & deps
############################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated venv to collect site-packages we will copy later
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_DISABLE_PIP_VERSION_CHECK=1

# BuildKit cache for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir fastapi uvicorn[standard]==0.30.* numpy==2.* \
                               pillow==10.* safetensors==0.4.*

# Optional: your app deps
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt

# [Optional] Strip symbols from big .so files (saves 50–200MB)
RUN find /opt/venv/lib -type f -name "*.so*" -exec sh -c 'file -b "{}" | grep -q ELF && strip --strip-unneeded "{}" || true' \;

############################
# 2) Runtime: slim + CUDA runtime
############################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Minimal OS packages only
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in only the env (site-packages + binaries)
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # NVIDIA runtime hints (compute only)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# App code last for better cache
WORKDIR /app
COPY app/ /app/

# Clean typical caches that sneak in at build time
RUN rm -rf /root/.cache/* /opt/venv/lib/python*/site-packages/*.dist-info/tests || true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]
```

**Why This Slims Down**:
- Heavy builds and pip resolution happen in devel; only the venv is copied
- No compilers/debug headers land in the final image
- Symbol stripping trims hundreds of MB (verify your runtime still works)
- CUDA runtime, not devel, in production

## 3) CUDA C++ App: Static-ish Binary on Runtime Base

### Complete Dockerfile (2.2 GB → <800 MB)

```dockerfile
# syntax=docker/dockerfile:1.9

########## build ##########
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

# Optional: strip
RUN strip --strip-unneeded build/bin/my_cuda_app || true

########## runtime ##########
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Copy only the binary and any runtime assets
COPY --from=build /src/build/bin/my_cuda_app /usr/local/bin/my_cuda_app

# (If needed) copy specific shared libs discovered via ldd
# RUN ldd /usr/local/bin/my_cuda_app | awk '/=> \//{print $3}' | xargs -I{} cp -v {} /usr/local/lib/

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["/usr/local/bin/my_cuda_app"]
```

**Tip**: If your binary links against non-CUDA libs not present in the runtime image, copy them selectively (use `ldd`) instead of shipping a full devel root.

## 4) APT & PIP: The "No Crumbs" Ritual

### APT Pattern

```dockerfile
# Standard pattern
RUN apt-get update && apt-get install -y --no-install-recommends \
      <pkgs> \
  && rm -rf /var/lib/apt/lists/*

# BuildKit cache for apt (speeds rebuilds)
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends <pkgs>
```

### PIP Pattern

```dockerfile
# Standard pattern
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Even leaner (fast installer)
RUN pip install uv && uv pip install --system -r requirements.txt
```

**Why These Patterns**: Cache mounts speed rebuilds, `--no-install-recommends` prevents bloat, `uv` is faster than pip.

## 5) Python-Specific Trimming

### Remove Tests and Samples

```dockerfile
# Remove tests/samples from heavy libs (careful; validate!)
RUN find /opt/venv/lib -type d -regex ".*\(tests\|__pycache__\|example\|samples\).*" -prune -exec rm -rf {} +
```

### Compile Python Files

```dockerfile
# Optional: compile Python files (sometimes reduces cold import time)
RUN python -m compileall -q /opt/venv
```

### Keep Only What You Import

```dockerfile
# If a framework is modular, consider specific wheels
# e.g., Torch without extra backends you never call
RUN pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Why This Trimming**: Python packages include tests, examples, and unused modules. Removing them saves significant space.

## 6) Model Assets & Caches (The Silent Bloat)

### Quantize and Trim Models

```dockerfile
# Quantize models offline: fp16/int8 where accuracy allows
# Use Safetensors over pickled weights when available
# Pin and bake models into the image only if capacity allows
```

### Model Cache Management

```dockerfile
# Set cache locations
ENV HF_HOME=/opt/cache/hf TRANSFORMERS_CACHE=/opt/cache/hf

# Clean cache after prewarming
RUN rm -rf /opt/cache/hf/**/downloads || true
```

### Pre-built Engines

```dockerfile
# For NVIDIA Triton / TensorRT engines: ship already-built .plan files
# instead of rebuilding at startup
COPY models/trt_engines/ /app/models/trt_engines/
```

**Why Model Management**: Models are often the largest component. Smart caching and quantization save gigabytes.

## 7) GPU Runtime Hygiene

### Minimal Capabilities

```dockerfile
# Minimal, explicit capabilities
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# If you don't render, don't add graphics,video,display
```

### Docker Compose

```yaml
# In Compose
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
```

### Docker Run

```bash
# In docker run
docker run --gpus '"device=0,1"' myapp-gpu:slim
```

**Why Runtime Hygiene**: Explicit capabilities prevent unnecessary driver overhead. Only request what you need.

## 8) Build & Inspect: Prove You Lost Weight

### Build with BuildKit

```bash
# Build with BuildKit
DOCKER_BUILDKIT=1 docker build -t myapp-gpu:slim .

# Measure
docker images | awk 'NR==1 || /myapp-gpu/'
docker history --no-trunc myapp-gpu:slim

# Compare two images by layer (requires dive)
dive myapp-gpu:slim
```

### Layer Analysis

```bash
# Layer sanity: lots of tiny RUN layers? Combine where sensible.
# One gigantic layer? Consider splitting for cache reuse
# Size won't change much, rebuild time will
```

**Why Inspection**: Understanding layer composition helps optimize further. Tools like `dive` reveal hidden bloat.

## 9) Troubleshooting: Common Foot-Guns

### CUDA Driver Issues

```bash
# "CUDA driver version is insufficient"
# Host driver < container CUDA runtime
# Align CUDA major versions; prefer the NVIDIA Container Toolkit compat matrix
```

### Missing Libraries

```bash
# "libcudnn.so not found"
# You used a base without cuDNN
# Either install runtime cuDNN variant or remove code paths that require it
```

### Import Errors After Stripping

```bash
# ImportError after stripping
# You stripped a Python extension that needs symbols
# Rebuild without strip for that module or use --strip-debug only
```

### Locale and Timezone Files

```bash
# Huge locale files/timezones
# On Ubuntu/Debian, avoid tzdata unless absolutely necessary
# If installed, keep it and accept the few MB hit rather than hacking around it
```

**Why These Issues**: GPU containers have complex dependencies. Understanding common problems prevents production failures.

## 10) Optional: Distroless-Lean Final Stage (Advanced)

### Distroless CUDA Binary

```dockerfile
# After building, copy CUDA runtime libs explicitly (ldd list) into distroless
FROM gcr.io/distroless/base-nonroot AS final
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.* /usr/local/lib/
COPY --from=build /src/build/bin/my_cuda_app /usr/local/bin/my_cuda_app
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/my_cuda_app"]
```

**Warning**: If any glibc or driver-specific paths are missing, it will crash—use only if you fully control dependencies.

## 11) Complete Example: PyTorch FastAPI App

### Project Structure

```
gpu-app/
├── Dockerfile
├── requirements.txt
├── docker-bake.hcl
└── app/
    ├── main.py
    ├── models/
    │   └── model.py
    └── utils/
        └── inference.py
```

### requirements.txt

```txt
torch==2.4.0+cu124
torchvision==0.19.0+cu124
fastapi==0.104.1
uvicorn[standard]==0.24.0
numpy==2.0.0
pillow==10.1.0
safetensors==0.4.0
```

### app/main.py

```python
from fastapi import FastAPI
from app.models.model import load_model, predict
from app.utils.inference import preprocess_image
import torch

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict_endpoint(image_data: bytes):
    processed = preprocess_image(image_data)
    result = predict(model, processed)
    return {"prediction": result}
```

### docker-bake.hcl

```hcl
group "default" {
  targets = ["gpu-app"]
}

variable "REGISTRY" { default = "ghcr.io/yourorg" }
variable "NAME"     { default = "gpu-app" }
variable "VERSION"  { default = "latest" }

target "gpu-app" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64"]
  tags = [
    "${REGISTRY}/${NAME}:${VERSION}",
    "${REGISTRY}/${NAME}:latest"
  ]
  cache-from = [
    "type=registry,ref=${REGISTRY}/${NAME}:cache"
  ]
  cache-to = [
    "type=registry,ref=${REGISTRY}/${NAME}:cache,mode=max"
  ]
}
```

## 12) TL;DR Checklist

### Essential Optimizations

- [ ] **Runtime base, not devel**
- [ ] **Multi-stage: build heavy, copy light**
- [ ] **`--no-install-recommends`, remove `/var/lib/apt/lists`**
- [ ] **BuildKit caches for apt/pip**
- [ ] **Strip symbols (validate!)**
- [ ] **Prune caches: pip, model hubs, `__pycache__`, tests**
- [ ] **Pin CUDA, frameworks, Python**
- [ ] **Set `NVIDIA_DRIVER_CAPABILITIES=compute,utility`**
- [ ] **Inspect layers (`docker history`, `dive`)**
- [ ] **Keep models slim or externalize to volumes**

### Build Commands

```bash
# Build with BuildKit
DOCKER_BUILDKIT=1 docker build -t gpu-app:slim .

# Build with Bake
docker buildx bake

# Inspect results
dive gpu-app:slim
```

## 13) The Machine's Summary

GPU containers are expensive to build and run. Every MB saved is money saved and performance gained. The key is understanding what your application actually needs and eliminating everything else.

**The Dark Truth**: GPU containers are bloated by default. Dev images include compilers, debug symbols, and unused libraries. Production images need only runtime dependencies.

**The Machine's Mantra**: "In minimalism we trust, in multi-stage we build, and in the slim container we find the path to efficient inference."

**Why This Matters**: GPU resources are expensive. Slim containers start faster, use less memory, and cost less to store and transfer.

---

*This tutorial provides the complete machinery for building slim GPU containers. The patterns scale from megabytes to terabytes, from development to production.*
