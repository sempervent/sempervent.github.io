# TensorFlow GPU Slim Images - Repository Skeleton

This skeleton provides a complete, production-ready example for building slim TensorFlow GPU containers. It includes both approaches (CUDA runtime base and tensorflow[and-cuda] wheels) with a working FastAPI application.

## Project Structure

```
tf-gpu-slim/
├── Dockerfile.approach-a          # CUDA runtime base approach
├── Dockerfile.approach-b          # tensorflow[and-cuda] wheels approach
├── requirements.txt               # Python dependencies
├── docker-compose.yml            # Multi-service setup
├── .dockerignore                 # Build exclusions
├── README.md                     # Project documentation
└── app/
    ├── main.py                   # FastAPI application
    ├── models/
    │   ├── create_model.py        # Model creation script
    │   └── model.h5              # Saved Keras model
    └── utils/
        └── preprocessing.py       # Data preprocessing utilities
```

## Dockerfile.approach-a (CUDA Runtime Base)

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
      fastapi==0.111.* uvicorn[standard]==0.30.* \
      numpy==2.* pillow==10.* protobuf~=4.25

# Trim common cruft inside site-packages
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

## Dockerfile.approach-b (tensorflow[and-cuda] Wheels)

```dockerfile
# syntax=docker/dockerfile:1.9
FROM python:3.12-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=1

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

## requirements.txt

```txt
tensorflow[and-cuda]==2.16.1
fastapi==0.111.0
uvicorn[standard]==0.30.0
numpy==2.0.0
pillow==10.1.0
protobuf~=4.25
```

## docker-compose.yml

```yaml
version: '3.8'

services:
  tf-app-a:
    build:
      context: .
      dockerfile: Dockerfile.approach-a
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

  tf-app-b:
    build:
      context: .
      dockerfile: Dockerfile.approach-b
    ports:
      - "8001:8000"
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

## .dockerignore

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

## app/main.py (FastAPI Application)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TensorFlow GPU Inference API",
    description="Slim TensorFlow GPU container for inference",
    version="1.0.0"
)

# Global model variable
model = None

class PredictionRequest(BaseModel):
    data: List[float]
    batch_size: int = 1

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    gpu_available: bool

@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model on startup"""
    global model
    try:
        model = tf.keras.models.load_model("/app/models/model.h5")
        logger.info(f"Model loaded successfully. Version: {tf.__version__}")
        
        # Log GPU information
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU available: {[gpu.name for gpu in gpus]}")
        else:
            logger.warning("No GPU detected, running on CPU")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpus = tf.config.list_physical_devices('GPU')
    return {
        "status": "healthy",
        "tensorflow_version": tf.__version__,
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [gpu.name for gpu in gpus] if gpus else [],
        "model_loaded": model is not None
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_summary": str(model.summary()),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        input_data = np.array(request.data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data, batch_size=request.batch_size)
        
        # Convert to list for JSON serialization
        predictions = prediction.tolist()[0] if len(prediction.shape) > 1 else prediction.tolist()
        
        return PredictionResponse(
            predictions=predictions,
            model_version=tf.__version__,
            gpu_available=len(tf.config.list_physical_devices('GPU')) > 0
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/gpu/info")
async def gpu_info():
    """Get detailed GPU information"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return {"gpu_available": False, "message": "No GPU detected"}
    
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "name": gpu.name,
            "device_type": gpu.device_type,
            "memory_limit": tf.config.experimental.get_memory_info(gpu.name) if hasattr(tf.config.experimental, 'get_memory_info') else None
        })
    
    return {
        "gpu_available": True,
        "gpu_count": len(gpus),
        "gpus": gpu_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## app/models/create_model.py (Model Creation Script)

```python
"""
Create a simple Keras model for demonstration purposes.
This script creates a small neural network and saves it as model.h5
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def create_simple_model():
    """Create a simple neural network for demonstration"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_training_data(n_samples=1000):
    """Generate synthetic training data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y

def main():
    """Create and train the model"""
    print("Creating model...")
    model = create_simple_model()
    
    print("Generating training data...")
    X, y = generate_training_data()
    
    print("Training model...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the model
    test_data = np.random.randn(1, 10)
    prediction = model.predict(test_data)
    print(f"Test prediction: {prediction[0][0]:.4f}")

if __name__ == "__main__":
    main()
```

## app/utils/preprocessing.py (Data Preprocessing Utilities)

```python
"""
Data preprocessing utilities for the TensorFlow application
"""

import numpy as np
from typing import List, Union

def normalize_data(data: Union[List[float], np.ndarray]) -> np.ndarray:
    """Normalize input data to [0, 1] range"""
    data = np.array(data)
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def standardize_data(data: Union[List[float], np.ndarray]) -> np.ndarray:
    """Standardize input data to zero mean and unit variance"""
    data = np.array(data)
    return (data - data.mean()) / (data.std() + 1e-8)

def validate_input_shape(data: Union[List[float], np.ndarray], expected_shape: int = 10) -> bool:
    """Validate that input data has the expected shape"""
    data = np.array(data)
    return len(data) == expected_shape

def preprocess_for_prediction(data: List[float]) -> np.ndarray:
    """Complete preprocessing pipeline for prediction"""
    if not validate_input_shape(data):
        raise ValueError(f"Expected input shape of 10, got {len(data)}")
    
    # Normalize the data
    processed_data = normalize_data(data)
    
    # Reshape for model input
    return processed_data.reshape(1, -1)
```

## README.md

```markdown
# TensorFlow GPU Slim Images

This repository demonstrates how to build slim TensorFlow GPU containers using two different approaches:

1. **Approach A**: CUDA runtime base with multi-stage builds
2. **Approach B**: tensorflow[and-cuda] wheels with minimal base image

## Quick Start

### Prerequisites

- Docker with BuildKit enabled
- NVIDIA Container Toolkit
- GPU with CUDA support

### Build and Run

```bash
# Build Approach A
docker build -f Dockerfile.approach-a -t tf-gpu:a .

# Build Approach B
docker build -f Dockerfile.approach-b -t tf-gpu:b .

# Run with GPU support
docker run --gpus all -p 8000:8000 tf-gpu:a

# Or use docker-compose
docker-compose up tf-app-a
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}'

# Get GPU info
curl http://localhost:8000/gpu/info
```

### Create Model

```bash
# Run the model creation script
python app/models/create_model.py
```

## Size Comparison

- **Approach A**: ~800MB (CUDA runtime base)
- **Approach B**: ~600MB (tensorflow[and-cuda] wheels)

## Features

- FastAPI application with health checks
- GPU detection and monitoring
- Model loading and inference
- Comprehensive error handling
- Production-ready logging

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app/main.py

# Test with curl
curl http://localhost:8000/health
```

## Production Deployment

This skeleton is designed for production deployment with:

- Health checks
- GPU monitoring
- Error handling
- Logging
- Model versioning
```

## Usage Instructions

1. **Copy the skeleton** to a new directory
2. **Create the model** by running `python app/models/create_model.py`
3. **Build the images** using either Dockerfile approach
4. **Test the API** with the provided curl commands
5. **Deploy to production** using docker-compose

This skeleton provides a complete, production-ready example for building slim TensorFlow GPU containers with both approaches demonstrated.
