# Compose Profiles at Scale: Toggling a Polyglot Data/Geo/ML City

**Objective**: One compose.yml. Many realities. Flip `--profile gpu` and torch tensors; `--profile tiles` and mint vector tiles. Profiles as power, not garnish.

## Architecture & Profiles Map

```mermaid
flowchart LR
  subgraph core[Core]
    PG[(PostGIS)]
    S3[(MinIO)]
    NX[Nginx]
  end

  subgraph realtime[realtime]
    KF[(Kafka)]
    API[FastAPI GW]
  end

  subgraph batch[batch]
    PF[Prefect]
    DS[Dask]
  end

  subgraph analytics[analytics]
    CH[(ClickHouse)]
  end

  subgraph observability[observability]
    PR[Prometheus]
    LK[(Loki)]
    TP[(Tempo)]
    GF[Grafana]
    PM[Promtail]
  end

  subgraph tiles[tiles]
    MR[Martin]
    ML[MapLibre UI]
  end

  subgraph mlops[mlops]
    MF[MLflow]
  end

  subgraph search[search]
    OS[(OpenSearch)]
    OD[Dashboards]
  end

  subgraph gpu[gpu]
    JP[Jupyter CUDA]
  end

  API --> PG
  KF --> API
  PF --> DS
  DS --> PG
  CH --> PG
  GF --> PR & LK & TP
  MR --> PG
  MF --> S3
  JP --> S3
  NX --> API & ML & GF
```

### Profile-to-Service Matrix

| Profile | Services | Purpose |
|---------|----------|---------|
| `core` | PostGIS, MinIO, Nginx | Always-on backbone |
| `realtime` | Kafka, FastAPI Gateway | Event streaming & APIs |
| `batch` | Prefect, Dask | Workflow orchestration & compute |
| `analytics` | ClickHouse | Read-heavy analytics |
| `observability` | Prometheus, Loki, Tempo, Grafana, Promtail | Monitoring & telemetry |
| `tiles` | Martin, MapLibre UI | Vector tile serving & visualization |
| `mlops` | MLflow | ML experiment tracking |
| `search` | OpenSearch, Dashboards | Full-text search & analytics |
| `gpu` | Jupyter CUDA | GPU-accelerated notebooks |

## .env and Secrets

### .env (Sample)

```bash
# Core services
POSTGRES_PASSWORD=postgres
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
S3_ENDPOINT=http://minio:9000
S3_BUCKET=artifacts

# MLflow configuration
MLFLOW_BACKEND_DB=postgresql+psycopg2://postgres:postgres@postgres:5432/mlflow

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# ClickHouse
CLICKHOUSE_DB=analytics
```

### Secrets Management

```bash
# For production, use Docker secrets instead of .env
# Create secrets files
echo "your-secure-password" | docker secret create postgres_password -
echo "your-minio-key" | docker secret create minio_access_key -
echo "your-minio-secret" | docker secret create minio_secret_key -

# Reference in compose.yml
secrets:
  postgres_password:
    external: true
  minio_access_key:
    external: true
  minio_secret_key:
    external: true
```

**When to use SOPS**: For Git-committed configs with encrypted secrets. Use `env_file:` for dev, `secrets:` for production.

## The Compose File (Profiles as Circuit Breakers)

```yaml
# docker-compose.yml
version: "3.9"

x-health: &hc
  interval: 5s
  timeout: 3s
  retries: 20

networks:
  edge:
    driver: bridge
  data:
    driver: bridge
  observ:
    driver: bridge

volumes:
  pgdata:
  minio-data:
  clickhouse-data:
  opensearch-data:
  loki-data:
  tempo-data:

services:
  # --- CORE SERVICES (always-on) ---
  postgres:
    image: postgis/postgis:15-3.4
    container_name: postgres
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: app
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d app"]
      <<: *hc
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks: [data]

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/ready"]
      <<: *hc
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    networks: [data]

  nginx:
    image: nginx:alpine
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    ports:
      - "80:80"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro
    networks: [edge, data]

  # --- REALTIME PROFILE ---
  kafka:
    image: bitnami/kafka:3.7
    profiles: ["realtime"]
    environment:
      KAFKA_ENABLE_KRAFT: "yes"
      KAFKA_CFG_PROCESS_ROLES: "broker,controller"
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: "CONTROLLER"
      KAFKA_CFG_LISTENERS: "PLAINTEXT://:9092,CONTROLLER://:9093"
      KAFKA_CFG_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:9092"
      ALLOW_PLAINTEXT_LISTENER: "yes"
    ports:
      - "9092:9092"
    healthcheck:
      test: ["CMD", "bash", "-lc", "echo > /dev/tcp/localhost/9092"]
      <<: *hc
    networks: [data]

  api:
    image: ghcr.io/yourorg/fastapi-gw:latest
    profiles: ["realtime"]
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_started
    environment:
      DB_DSN: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/app
      KAFKA_BOOTSTRAP: kafka:9092
    ports:
      - "8000:8000"
    networks: [edge, data]

  # --- BATCH PROFILE ---
  prefect:
    image: prefecthq/prefect:2-latest
    profiles: ["batch"]
    command: ["prefect", "server", "start", "--host", "0.0.0.0"]
    environment:
      PREFECT_API_URL: http://prefect:4200/api
    ports:
      - "4200:4200"
    networks: [data]

  dask-scheduler:
    image: daskdev/dask:latest
    profiles: ["batch"]
    command: ["dask-scheduler", "--host", "0.0.0.0"]
    ports:
      - "8786:8786"
      - "8787:8787"
    networks: [data]

  dask-worker:
    image: daskdev/dask:latest
    profiles: ["batch"]
    command: ["dask-worker", "dask-scheduler:8786", "--nthreads", "2", "--memory-limit", "2GB"]
    depends_on:
      - dask-scheduler
    deploy:
      replicas: 2
    networks: [data]

  # --- ANALYTICS PROFILE ---
  clickhouse:
    image: clickhouse/clickhouse-server:24.8
    profiles: ["analytics"]
    environment:
      CLICKHOUSE_DB: ${CLICKHOUSE_DB:-analytics}
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    networks: [data]

  # --- OBSERVABILITY PROFILE ---
  prometheus:
    image: prom/prometheus:latest
    profiles: ["observability"]
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    networks: [observ, data]

  loki:
    image: grafana/loki:2.9.8
    profiles: ["observability"]
    command: ["-config.file=/etc/loki/local-config.yaml"]
    volumes:
      - ./observability/loki.yaml:/etc/loki/local-config.yaml:ro
      - loki-data:/loki
    ports:
      - "3100:3100"
    networks: [observ]

  promtail:
    image: grafana/promtail:2.9.8
    profiles: ["observability"]
    command: ["-config.file=/etc/promtail/config.yml"]
    volumes:
      - ./observability/promtail.yml:/etc/promtail/config.yml:ro
      - /var/run/docker.sock:/var/run/docker.sock
    networks: [observ]

  tempo:
    image: grafana/tempo:latest
    profiles: ["observability"]
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./observability/tempo.yaml:/etc/tempo.yaml:ro
      - tempo-data:/tmp/tempo
    ports:
      - "3200:3200"
    networks: [observ]

  grafana:
    image: grafana/grafana:latest
    profiles: ["observability"]
    depends_on:
      prometheus:
        condition: service_started
      loki:
        condition: service_started
      tempo:
        condition: service_started
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GF_SECURITY_ADMIN_PASSWORD}
    ports:
      - "3000:3000"
    volumes:
      - ./observability/grafana/provisioning:/etc/grafana/provisioning:ro
    networks: [edge, observ]

  # --- TILES PROFILE ---
  martin:
    image: ghcr.io/maplibre/martin:latest
    profiles: ["tiles"]
    command: ["--listen-addr", "0.0.0.0:3001", "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/app"]
    ports:
      - "3001:3001"
    depends_on:
      postgres:
        condition: service_healthy
    networks: [edge, data]

  maplibre:
    image: nginx:alpine
    profiles: ["tiles"]
    volumes:
      - ./maplibre:/usr/share/nginx/html:ro
    ports:
      - "8080:80"
    networks: [edge]

  # --- MLOPS PROFILE ---
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    profiles: ["mlops"]
    command: >
      mlflow server 
      --backend-store-uri ${MLFLOW_BACKEND_DB}
      --default-artifact-root s3://${S3_BUCKET}/mlflow/ 
      --host 0.0.0.0 
      --port 5000
    ports:
      - "5000:5000"
    environment:
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
      MLFLOW_S3_ENDPOINT_URL: ${S3_ENDPOINT}
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    networks: [edge, data]

  # MinIO bucket initialization
  minio-init:
    image: minio/mc
    profiles: ["mlops", "gpu"]
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      sh -c "
      mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} &&
      mc mb -p local/${S3_BUCKET} || true
      "
    networks: [data]

  # --- SEARCH PROFILE ---
  opensearch:
    image: opensearchproject/opensearch:2
    profiles: ["search"]
    environment:
      - "discovery.type=single-node"
      - "plugins.security.disabled=true"
    ports:
      - "9200:9200"
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    networks: [data]

  dashboards:
    image: opensearchproject/opensearch-dashboards:2
    profiles: ["search"]
    environment:
      - "OPENSEARCH_HOSTS=['http://opensearch:9200']"
    ports:
      - "5601:5601"
    depends_on:
      opensearch:
        condition: service_started
    networks: [edge, data]

  # --- GPU PROFILE ---
  jupyter-gpu:
    image: nvidia/cuda:12.3.2-cudnn-runtime-ubuntu22.04
    profiles: ["gpu"]
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - JUPYTER_ENABLE_LAB=yes
    command: >
      bash -lc "
      apt-get update && 
      apt-get install -y python3-pip && 
      pip3 install jupyterlab boto3 s3fs && 
      jupyter lab --ip=0.0.0.0 --allow-root --no-browser
      "
    ports:
      - "8888:8888"
    depends_on:
      minio:
        condition: service_healthy
    networks: [edge, data]
```

## Profile Recipes (Operator Cheatsheet)

### Minimal Core Only (DB + S3 + Front)

```bash
# Just the essentials
docker compose up -d
```

**What you get**: PostGIS, MinIO, Nginx. Perfect for development or when you just need data storage.

### Realtime API + Kafka

```bash
# Event streaming and APIs
docker compose --profile realtime up -d
```

**What you get**: Core + Kafka + FastAPI Gateway. Ready for real-time data ingestion and API endpoints.

### Batch Compute (Prefect + Dask)

```bash
# Workflow orchestration and distributed compute
docker compose --profile batch up -d
```

**What you get**: Core + Prefect server + Dask scheduler + 2 Dask workers. Perfect for ETL pipelines and batch processing.

### Analytics with ClickHouse

```bash
# Read-heavy analytics
docker compose --profile analytics up -d
```

**What you get**: Core + ClickHouse. Ready for OLAP queries and analytics dashboards.

### Observability Cockpit

```bash
# Full monitoring stack
docker compose --profile observability up -d
```

**What you get**: Core + Prometheus + Loki + Tempo + Grafana + Promtail. Complete observability stack.

**Access**: Grafana at http://localhost:3000 (admin/admin by default—change it!)

### Tileserver + Static MapLibre UI

```bash
# Vector tile serving and visualization
docker compose --profile tiles up -d
```

**What you get**: Core + Martin tileserver + MapLibre static UI.

**Access**: 
- Martin: http://localhost:3001
- MapLibre demo: http://localhost:8080

### MLOps with MLflow

```bash
# ML experiment tracking
docker compose --profile mlops up -d
```

**What you get**: Core + MLflow server with MinIO artifact store.

**Access**: MLflow UI at http://localhost:5000

### Search (OpenSearch + Dashboards)

```bash
# Full-text search and analytics
docker compose --profile search up -d
```

**What you get**: Core + OpenSearch + OpenSearch Dashboards.

**Access**: 
- OpenSearch: http://localhost:9200
- Dashboards: http://localhost:5601

### GPU Notebook

```bash
# GPU-accelerated Jupyter
docker compose --profile gpu up -d
```

**What you get**: Core + Jupyter Lab with CUDA support.

**Access**: Jupyter Lab at http://localhost:8888 (check logs for token)

### All-Out Mayhem (Not for 8GB Laptops)

```bash
# Everything at once
docker compose \
  --profile realtime --profile batch --profile analytics \
  --profile observability --profile tiles --profile mlops \
  --profile search --profile gpu up -d
```

**What you get**: The entire polyglot city. Use with caution and plenty of RAM.

## Health, Readiness, and Dependencies

### Healthcheck Best Practices

```yaml
# Good: Specific, fast healthcheck
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U postgres -d app"]
  interval: 5s
  timeout: 3s
  retries: 20

# Bad: Generic, slow healthcheck
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Dependency Chains

```yaml
# Postgres must be healthy before Martin starts
martin:
  depends_on:
    postgres:
      condition: service_healthy

# All observability services must be running before Grafana
grafana:
  depends_on:
    prometheus:
      condition: service_started
    loki:
      condition: service_started
    tempo:
      condition: service_started
```

### Common Race Conditions

**Kafka broker readiness**: Use `service_started` instead of `service_healthy` for Kafka dependencies.

**MinIO bucket creation**: The `minio-init` service handles this automatically for MLflow and GPU profiles.

**PostGIS extensions**: Ensure PostGIS is fully initialized before Martin tries to connect.

## Networking & Security Notes

### Network Segmentation

```yaml
networks:
  edge:    # Public-facing services (Nginx, Grafana, APIs)
  data:    # Database and storage services
  observ:  # Telemetry services (Prometheus, Loki, Tempo)
```

### Security Considerations

```bash
# Rotate Grafana admin password
export GF_SECURITY_ADMIN_PASSWORD="your-secure-password"

# Use Docker secrets for production
docker secret create postgres_password ./secrets/postgres_password.txt

# For GPU: Ensure NVIDIA Container Toolkit is installed
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Nginx Configuration

```nginx
# nginx/default.conf
upstream api {
    server api:8000;
}

upstream grafana {
    server grafana:3000;
}

upstream martin {
    server martin:3001;
}

server {
    listen 80;
    
    location /api/ {
        proxy_pass http://api/;
    }
    
    location /grafana/ {
        proxy_pass http://grafana/;
    }
    
    location /tiles/ {
        proxy_pass http://martin/;
    }
}
```

## Observability Wiring (Quick Dashboards)

### Prometheus Configuration

```yaml
# observability/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
  
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']
```

### Grafana Provisioning

```yaml
# observability/grafana/provisioning/datasources/datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    
  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
```

### Loki Configuration

```yaml
# observability/loki.yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

## Failure Modes (Tiny, Honest Postmortems)

### Kafka Starts But Clients Can't Connect

**Symptom**: `Connection refused` when trying to connect to Kafka from applications.

**Root Cause**: `KAFKA_CFG_ADVERTISED_LISTENERS` misconfiguration.

**Fix**: Ensure advertised listeners match the service name:
```yaml
KAFKA_CFG_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:9092"
```

### Martin Shows 500s

**Symptom**: Martin tileserver returns HTTP 500 errors.

**Root Cause**: Wrong PostGIS DSN or missing spatial indexes.

**Fix**: Check PostGIS connection string and ensure spatial indexes exist:
```sql
CREATE INDEX IF NOT EXISTS idx_geom ON your_table USING gist(geom);
```

### MLflow Artifacts Failing

**Symptom**: MLflow can't store artifacts in MinIO.

**Root Cause**: S3 endpoint misconfiguration or bucket doesn't exist.

**Fix**: Verify `MLFLOW_S3_ENDPOINT_URL` and ensure bucket is created:
```bash
mc mb -p local/artifacts
```

### GPU Container Idle

**Symptom**: Jupyter GPU container starts but no web interface.

**Root Cause**: Jupyter token in logs, port mapping, or NVIDIA runtime issues.

**Fix**: 
1. Check logs for Jupyter token: `docker logs jupyter-gpu`
2. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`
3. Check port mapping: `docker port jupyter-gpu`

### ClickHouse Connection Refused

**Symptom**: Can't connect to ClickHouse from applications.

**Root Cause**: ClickHouse not fully initialized or wrong port.

**Fix**: Wait for ClickHouse to be ready and use port 9000 for native protocol, 8123 for HTTP.

## TL;DR Runbook

### Essential Commands

```bash
# Start with core services
docker compose up -d

# Add one subsystem at a time
docker compose --profile realtime up -d
docker compose --profile observability up -d
docker compose --profile tiles up -d

# Check service health
docker compose ps

# View logs
docker compose logs -f martin

# Stop everything
docker compose down -v
```

### Production Checklist

- [ ] Use Docker secrets instead of .env files
- [ ] Rotate default passwords (Grafana, Postgres, MinIO)
- [ ] Configure proper healthchecks
- [ ] Set up log rotation
- [ ] Monitor resource usage
- [ ] Backup volumes regularly
- [ ] Use external networks for multi-compose setups

### Development Workflow

1. **Start core**: `docker compose up -d`
2. **Add profiles incrementally**: `--profile realtime --profile observability`
3. **Use healthchecks**: Wait for services to be healthy before adding dependent services
4. **Monitor logs**: `docker compose logs -f [service]`
5. **Clean up**: `docker compose down -v` to remove volumes

---

*This tutorial demonstrates how Docker Compose profiles can orchestrate complex, multi-technology stacks with surgical precision. Each profile represents a complete subsystem that can be toggled independently, enabling operators to build exactly the infrastructure they need without the overhead of unused services.*
