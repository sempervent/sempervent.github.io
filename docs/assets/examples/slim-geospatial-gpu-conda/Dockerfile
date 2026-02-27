# syntax=docker/dockerfile:1.6
# Multi-stage geospatial + GPU Conda image.
# Usage:
#   CPU: docker build --build-arg ENV_FILE=environment.cpu.yml --target runtime-cpu -t geoapp:cpu .
#   GPU: docker build --build-arg ENV_FILE=environment.gpu.yml --target runtime-gpu -t geoapp:gpu .

# ─── Stage 1: Build Conda environment ────────────────────────────────────────
FROM mambaorg/micromamba:1.5-bookworm-slim AS builder

USER root

ARG ENV_FILE=environment.cpu.yml
COPY ${ENV_FILE} /tmp/environment.yml

RUN micromamba env create \
      --file /tmp/environment.yml \
      --prefix /opt/env \
      --yes \
    && micromamba clean --all --yes \
    && /opt/env/bin/pip cache purge 2>/dev/null || true \
    && find /opt/env -name "*.pyc" -delete \
    && find /opt/env -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

RUN /opt/env/bin/python -c "from osgeo import gdal; print('GDAL OK:', gdal.VersionInfo())"
RUN /opt/env/bin/python -c "import numpy; print('NumPy OK:', numpy.__version__)"

COPY app/ /opt/app/

# ─── Stage 2a: CPU runtime ────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --gid 0 --no-create-home --shell /sbin/nologin appuser

COPY --from=builder /opt/env /opt/env
COPY --from=builder /opt/app /opt/app

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
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --gid 0 --no-create-home --shell /sbin/nologin appuser

COPY --from=builder /opt/env /opt/env
COPY --from=builder /opt/app /opt/app

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
