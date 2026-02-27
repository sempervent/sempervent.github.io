# Geospatial Slim Demo — Local Run

## Prerequisites

- Conda or Micromamba installed locally
- NVIDIA GPU + driver ≥ 525 (GPU path only)

## Run CPU image locally

```bash
# From repo root (slim-geospatial-gpu-conda/)
micromamba env create -f environment.cpu.yml --prefix /tmp/geoapp-env
/tmp/geoapp-env/bin/pip install fastapi uvicorn[standard]

GDAL_DATA="$(find /tmp/geoapp-env -name 'epsg' | head -1 | xargs dirname)" \
PROJ_LIB="$(find /tmp/geoapp-env -name 'proj.db' | head -1 | xargs dirname)" \
/tmp/geoapp-env/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Build and run via Docker

```bash
# CPU:
docker build --build-arg ENV_FILE=environment.cpu.yml --target runtime-cpu -t geoapp:cpu .
docker run --rm -p 8000:8000 geoapp:cpu

# GPU:
docker build --build-arg ENV_FILE=environment.gpu.yml --target runtime-gpu -t geoapp:gpu .
docker run --rm --gpus all -p 8000:8000 geoapp:gpu
```

## Endpoints

- `GET /health` — GDAL version check
- `GET /inspect` — Metadata + stats for synthetic raster
- `GET /normalize` — Band normalization (CuPy if GPU available, NumPy otherwise)
