# Local LLM Deployments: Methods, Quantization, and OpenAI-Compatible Serving

**Objective**: A ruthless field guide to running modern LLMs on your own hardware. CPU to GPU, tiny to obnoxious, with real benchmarks and a path to RAG.

## Mental Model: What You're Actually Choosing

### Inference Stack Components
- **Model Format**: GGUF, GPTQ, AWQ, FP16/BF16
- **Runtime**: llama.cpp, vLLM, TGI, Ollama
- **Serving API**: OpenAI-compatible, native REST
- **Orchestration**: Docker Compose profiles, resource management

### Three Canonical Paths

| Path | Format | Runtime | Best For | Memory | Throughput |
|------|--------|---------|----------|--------|------------|
| **Ollama** | GGUF | Ollama | Easy setup, Mac M-series | 5-16GB | Medium |
| **llama.cpp** | GGUF | llama.cpp | CPU-only, edge deployment | 4-12GB | Low |
| **vLLM/TGI** | HF Models | vLLM/TGI | GPU serving, high throughput | 8-24GB | High |

### Other Worthwhile Options
- **LM Studio**: Desktop UI for GGUF models
- **KoboldCpp**: Roleplay and longform generation
- **MLC**: WebGPU inference for browsers
- **ONNX Runtime/OpenVINO**: Intel CPU/iGPU optimization
- **TensorRT-LLM**: NVIDIA maximum speed (enterprise)

## Model Formats & Quantization Crash Kit

### GGUF Quantization (Safest Local Bet)
| Quantization | Quality | Memory | Use Case |
|--------------|---------|--------|----------|
| **Q4_K_M** | Good | 5-6GB (8B) | Chat, tooling, minimal VRAM |
| **Q5_K_M** | Better | 6-8GB (8B) | Reasoning, writing |
| **Q6_K** | High | 8-10GB (8B) | Code, analysis |
| **Q8_K** | Near-FP16 | 12-14GB (8B) | Best quality, heavy |

### GPU-Optimized Formats
- **GPTQ**: 4-bit weight-only, exllama kernels
- **AWQ**: 4-bit weight-only, AWQ kernels
- **FP16/BF16**: Best quality, needs big VRAM

### Memory Planning Quick Sheet
```bash
# 8B Model Memory Requirements
Q4_K_M:  ~5-6 GB RAM/VRAM
Q6_K:    ~8-10 GB RAM/VRAM  
Q8_K:    ~12-14 GB RAM/VRAM
FP16:    ~16 GB VRAM

# 13B Model Memory Requirements
Q4_K_M:  ~8-10 GB RAM/VRAM
Q6_K:    ~12-14 GB RAM/VRAM
FP16:    ~24 GB VRAM

# 70B Model Memory Requirements
Q4_K_M:  ~40-45 GB RAM/VRAM
FP16:    "Don't."
```

## Docker Compose: Profile the Zoo

### docker-compose.yml
```yaml
version: "3.9"
x-hc: &hc { interval: 5s, timeout: 3s, retries: 40 }

services:
  # --- OLLAMA + UI ---
  ollama:
    image: ollama/ollama:latest
    profiles: ["ollama"]
    environment: 
      - OLLAMA_KEEP_ALIVE=24h
    volumes: 
      - ollama:/root/.ollama
    ports: 
      - "11434:11434"
    healthcheck: 
      test: ["CMD","curl","-sf","http://localhost:11434/api/tags"]
      <<: *hc

  openwebui:
    image: ghcr.io/open-webui/open-webui:latest
    profiles: ["ollama"]
    depends_on: 
      ollama: 
        condition: service_healthy
    environment: 
      - OLLAMA_BASE_URL=http://ollama:11434
    ports: 
      - "8080:8080"

  # --- llama.cpp OpenAI-compatible server (CPU/GPU) ---
  llamacpp:
    image: ghcr.io/ggerganov/llama.cpp:server
    profiles: ["llamacpp"]
    command: 
      - "-m"
      - "/models/model.gguf"
      - "-c"
      - "4096"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8001"
      - "--api"
    volumes: 
      - "./models:/models"
    ports: 
      - "8001:8001"

  # --- vLLM (GPU) ---
  vllm:
    image: vllm/vllm-openai:latest
    profiles: ["vllm"]
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices: 
            - capabilities: ["gpu"]
    command: >
      --model meta-llama/Meta-Llama-3-8B-Instruct
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.85
    ports: 
      - "8002:8000"

  # --- TGI (GPU) ---
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    profiles: ["tgi"]
    deploy:
      resources:
        reservations:
          devices: 
            - capabilities: ["gpu"]
    environment:
      - MODEL_ID=NousResearch/Hermes-3-Llama-3.1-8B
    ports: 
      - "8003:80"

  # --- RAG bits: Qdrant + embeddings + demo API ---
  qdrant:
    image: qdrant/qdrant:latest
    profiles: ["rag"]
    ports: 
      - "6333:6333"
      - "6334:6334"
    volumes: 
      - qdrant:/qdrant/storage

  e5-small:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-0.4
    profiles: ["rag"]
    environment: 
      - MODEL_ID=intfloat/e5-small-v2
    ports: 
      - "8808:80"

  rag-api:
    build: ./rag-api
    profiles: ["rag"]
    environment:
      OPENAI_BASE_URL: http://vllm:8000
      QDRANT_URL: http://qdrant:6333
      EMB_URL: http://e5-small:80
    ports: 
      - "8010:8010"

  # --- bench pod ---
  bench:
    build: ./bench
    profiles: ["bench"]
    command: ["bash","-lc","python bench.py"]
    environment:
      OPENAI_ENDPOINT: http://localhost:8002/v1
      MODEL: meta-llama/Meta-Llama-3-8B-Instruct

volumes:
  qdrant:
  ollama:
```

### Operator Recipes
```bash
# 1) One-command UX (Ollama + OpenWebUI)
docker compose --profile ollama up -d
# Pull & run a model
curl http://localhost:11434/api/pull -d '{"name":"llama3.1:8b"}'

# 2) OpenAI-compatible llama.cpp
docker compose --profile llamacpp up -d
curl http://localhost:8001/v1/models

# 3) GPU throughput (vLLM)
docker compose --profile vllm up -d
curl http://localhost:8002/v1/models

# 4) TGI alternative
docker compose --profile tgi up -d

# 5) Minimal RAG (Qdrant + embeddings + demo FastAPI)
docker compose --profile rag --profile vllm up -d

# 6) Benchmark
docker compose --profile bench up --build
```

## OpenAI-Compatible Calls (Switch Backends Freely)

### vLLM (High Throughput)
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages":[{"role":"user","content":"Explain locality-sensitive hashing."}],
    "temperature": 0.2,
    "max_tokens": 512
  }'
```

### llama.cpp Server (CPU/GPU)
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages":[{"role":"user","content":"Explain locality-sensitive hashing."}],
    "temperature": 0.2,
    "max_tokens": 512
  }'
```

### Ollama (Native API)
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"haiku about cache misses","stream":false}'
```

## Methodologies: Choose Your Poison

### A) Ollama (GGUF; Easiest)
**Pros**: Single endpoint, model zoo, CPU/GPU, fast setup, Mac M-series optimization
**Cons**: Less tunable than raw vLLM/TGI; mixed plugin ecosystem
**Use when**: Prototyping, local UX apps (OpenWebUI), mixed OS fleet

### B) llama.cpp (GGUF; Absolute Portability)
**Pros**: Smallest memory footprint per quality; insane quant variety; CPU-only feasible
**Cons**: Lower throughput vs GPU servers; long context windows can get slow
**Use when**: Edge/air-gapped, CPU-only nodes, deterministic packaging

### C) vLLM (GPU; High Throughput)
**Pros**: PagedAttention = large context + strong throughput; OpenAI API out of the box
**Cons**: GPU required; model weights usually FP16/BF16 or GPTQ/AWQ variants
**Use when**: Serving, multi-tenant, latency targets

### D) TGI (GPU; Rock-Solid Production)
**Pros**: Battle-tested with HF models, tensor parallel, tokenizer server
**Cons**: Bigger footprint, config verbosity
**Use when**: Enterprise serving with HF ecosystem

## Prompting & Sampling Defaults

### Sane Defaults (Don't Sabotage Yourself)
```json
{
  "temperature": 0.2,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "max_tokens": 512
}
```

### Use Case Specific Settings
```json
// Coding/Agents
{
  "temperature": 0.0,
  "top_p": 0.95,
  "max_tokens": 2048
}

// Creative Writing
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024
}

// Reasoning/Analysis
{
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 1024
}
```

## RAG Mini-Stack (Works with Any Server)

### rag-api/Dockerfile
```dockerfile
FROM python:3.11-slim

RUN pip install fastapi uvicorn requests qdrant-client

WORKDIR /app
COPY main.py .

CMD ["python", "main.py"]
```

### rag-api/main.py
```python
from fastapi import FastAPI
import requests, uuid, json
from qdrant_client import QdrantClient
from qdrant_client.http import models

APP = FastAPI()
QDRANT = QdrantClient(url="http://qdrant:6333")
COL = "docs"
EMB = "http://e5-small:80/embed"
OAI = "http://vllm:8000/v1"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

def embed(texts):
    r = requests.post(EMB, json={"inputs": texts})
    return r.json()["embeddings"]

@APP.post("/ingest")
def ingest(doc: dict):
    chunks = [doc["text"][i:i+800] for i in range(0, len(doc["text"]), 800)]
    vecs = embed(chunks)
    QDRANT.upsert(collection_name=COL, points=[
        models.PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": c}) 
        for v, c in zip(vecs, chunks)
    ])
    return {"ok": True}

@APP.post("/ask")
def ask(q: dict):
    qvec = embed([q["question"]])[0]
    hits = QDRANT.search(collection_name=COL, query_vector=qvec, limit=4)
    context = "\n\n".join([h.payload["text"] for h in hits])
    prompt = f"Answer using only the context.\n\nContext:\n{context}\n\nQ: {q['question']}\nA:"
    r = requests.post(f"{OAI}/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    })
    return r.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=8010)
```

## Benchmarks (Latency, Throughput, Token-per-Second)

### bench/Dockerfile
```dockerfile
FROM python:3.11-slim

RUN pip install requests

WORKDIR /app
COPY bench.py .

CMD ["python", "bench.py"]
```

### bench/bench.py
```python
import os, time, requests

ENDPOINT = os.getenv("OPENAI_ENDPOINT", "http://localhost:8002/v1")
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Summarize: " + "lorem ipsum " * 200

print(f"Benchmarking {ENDPOINT} with model {MODEL}")

# Latency test
t0 = time.time()
r = requests.post(f"{ENDPOINT}/chat/completions", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.2,
    "max_tokens": 512
}, timeout=120)

dt = time.time() - t0
out = r.json()["choices"][0]["message"]["content"]
tps = 512 / dt

print(f"Latency: {dt:.2f}s  ~{tps:.1f} tok/s  len={len(out)}")

# Throughput test (concurrent requests)
import concurrent.futures
import threading

def single_request():
    t0 = time.time()
    r = requests.post(f"{ENDPOINT}/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 100
    }, timeout=60)
    return time.time() - t0

# Run 5 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(single_request) for _ in range(5)]
    times = [f.result() for f in futures]

avg_time = sum(times) / len(times)
print(f"Concurrent avg: {avg_time:.2f}s per request")
```

### Benchmark Interpretation Rules
- **If tps < 8 on CPU**: Step down quantization or batch size; consider GPU
- **If GPU OOM**: Reduce max context, use quantized weights (GPTQ/AWQ), or try paged KV (vLLM)
- **For multi-client serving**: Prefer vLLM or TGI over Ollama/llama.cpp

## Security, Licensing, and Tribal Scars

### Model Licenses
- **Llama 3.1**: Meta license (commercial use allowed)
- **Code Llama**: Meta license (commercial use allowed)
- **Mistral**: Apache 2.0 (permissive)
- **Qwen**: Tongyi Qianwen license (check restrictions)
- **Always verify**: Don't ship proprietary weights where you shouldn't

### Security Best Practices
```bash
# Don't expose your local server to the internet naked
# Use reverse proxy + auth
nginx -c nginx.conf

# Keep prompt/response logs off shared disks
# If you run agents, rate-limit tools and cap budgets
```

### Production Considerations
- **Rate Limiting**: Implement request throttling
- **Authentication**: Add API keys or OAuth
- **Monitoring**: Log usage and performance metrics
- **Backup**: Model weights and configurations

## Failure Modes & Fixes

### Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| **Garbage output** | Wrong chat template or over-quantized weights | Check model compatibility, try higher quant |
| **Stalls/timeout** | KV cache thrash | Cut max tokens or context; upgrade VRAM |
| **GPU OOM on first token** | Model too large for VRAM | Use `--gpu-memory-utilization 0.75`, smaller batch |
| **Bad RAG answers** | Chunking too big; embedding mismatch | Re-rank with small cross-encoder |

### Debugging Commands
```bash
# Check GPU memory usage
nvidia-smi

# Monitor llama.cpp server
docker logs llamacpp

# Test vLLM health
curl http://localhost:8002/health

# Check Ollama models
curl http://localhost:11434/api/tags
```

### Performance Tuning
```bash
# vLLM memory optimization
--gpu-memory-utilization 0.85
--max-model-len 8192
--dtype bfloat16

# llama.cpp CPU optimization
--threads 8
--batch-size 512
--ctx-size 4096

# Ollama optimization
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=1
```

## Trade-off Tables

### Memory vs Quality vs Speed
| Method | Memory (8B) | Quality | Speed | Use Case |
|--------|-------------|---------|-------|----------|
| **Q4_K_M** | 5-6GB | Good | Fast | Chat, tooling |
| **Q6_K** | 8-10GB | Better | Medium | Reasoning |
| **Q8_K** | 12-14GB | High | Slow | Analysis |
| **FP16** | 16GB | Best | Fast | Production |

### Backend Comparison
| Backend | Setup | Memory | Throughput | API | Best For |
|---------|-------|--------|------------|-----|----------|
| **Ollama** | Easy | Medium | Medium | Native | Prototyping |
| **llama.cpp** | Medium | Low | Low | OpenAI | Edge/CPU |
| **vLLM** | Hard | High | High | OpenAI | Serving |
| **TGI** | Hard | High | High | Custom | Enterprise |

## TL;DR Runbook

```bash
# Easiest path: Ollama + OpenWebUI
docker compose --profile ollama up -d
curl http://localhost:11434/api/pull -d '{"name":"llama3.1:8b"}'
# Open http://localhost:8080

# Need OpenAI API + throughput: vLLM (GPU)
docker compose --profile vllm up -d
curl http://localhost:8002/v1/models

# CPU-only edge: llama.cpp with GGUF Q4_K
docker compose --profile llamacpp up -d
# Place model.gguf in ./models/

# Add RAG: Qdrant + embeddings + demo API
docker compose --profile rag --profile vllm up -d
curl http://localhost:8010/docs

# Benchmark, then raise quant quality until speed and output are both acceptable
docker compose --profile bench up --build
```

### Quick Start Checklist
1. **Choose your poison**: Ollama (easy), llama.cpp (CPU), vLLM (GPU)
2. **Pick quantization**: Q4_K_M (start), Q6_K (better), Q8_K (best)
3. **Test with benchmarks**: Latency, throughput, quality
4. **Add RAG if needed**: Qdrant + embeddings + FastAPI
5. **Productionize**: Security, monitoring, scaling

---

*This tutorial provides the complete machinery for running LLMs locally. Each component is production-ready, copy-paste runnable, and designed for real-world deployment with proper trade-offs and failure handling.*
