# Monitoring with Grafana, Prometheus & Node Exporter

This tutorial establishes a complete monitoring stack using Grafana for visualization, Prometheus for metrics collection and time-series storage, and Node Exporter for system metrics. We start simple with a single machine setup, then scale to multiple machines where Prometheus scrapes metrics from remote endpoints.

## 1. Single-Machine Setup (Prometheus + Grafana + Node Exporter)

### Node Exporter Installation

```bash
curl -LO https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-*.linux-amd64.tar.gz
tar xvf node_exporter-*.linux-amd64.tar.gz
cd node_exporter-*
./node_exporter
```

**Why:** Node Exporter exposes system metrics at http://localhost:9100/metrics. This provides CPU, memory, disk, and network statistics in Prometheus format.

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "node"
    static_configs:
      - targets: ["localhost:9100"]
```

Run Prometheus:

```bash
./prometheus --config.file=prometheus.yml
```

**Why:** Prometheus scrapes metrics from Node Exporter every 15 seconds and stores them in its time-series database. Access the UI at http://localhost:9090.

### Grafana Quickstart

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Access Grafana at http://localhost:3000 (admin/admin). Add Prometheus as a datasource (http://localhost:9090).

**Why:** Grafana provides rich dashboards and alerting capabilities. This stack gets you monitoring dashboards in minutes with live system metrics.

## 2. Scaling to Multiple Machines

### Install Node Exporter on Every Node

```bash
# On each target machine
curl -LO https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-*.linux-amd64.tar.gz
tar xvf node_exporter-*.linux-amd64.tar.gz
cd node_exporter-*
sudo ./node_exporter
```

### Update Prometheus Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "nodes"
    static_configs:
      - targets:
          - "10.0.1.10:9100"
          - "10.0.1.11:9100"
          - "10.0.1.12:9100"
```

Restart Prometheus:

```bash
systemctl restart prometheus
```

**Why:** Prometheus centralizes metrics collection by scraping all Node Exporters. This enables monitoring of entire infrastructure from a single point.

## 3. Docker Compose with Profiles

```yaml
# docker-compose.yaml
version: "3.9"
name: monitoring

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    profiles: ["single-node", "multi-node"]

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on: [prometheus]
    profiles: ["single-node", "multi-node"]

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    profiles: ["single-node"]

  # Example of remote node simulation
  node-exporter-remote:
    image: prom/node-exporter:latest
    ports:
      - "9200:9100"
    profiles: ["multi-node"]
```

**Usage:**

```bash
# Single machine
docker compose --profile single-node up -d

# Multi-machine (simulated extra node)
docker compose --profile multi-node up -d
```

**Why:** Profiles enable different deployment scenarios without duplicating configuration. Single-node for development, multi-node for production-like setups.

## 4. Dashboards & Next Steps

### Import Node Exporter Dashboard

1. Access Grafana at http://localhost:3000
2. Go to "+" → Import
3. Enter dashboard ID: 1860 (official Node Exporter dashboard)
4. Select Prometheus datasource

### Grafana Provisioning

Create `grafana/provisioning/dashboards/dashboard.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

**Why:** Automated dashboard provisioning ensures consistent monitoring setup across environments. The Node Exporter dashboard provides comprehensive system metrics visualization.

## 5. TL;DR (Quickstart)

```bash
# 1. Start exporters + prometheus + grafana
docker compose --profile single-node up -d

# 2. Access dashboards
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000

# 3. Scale out
docker compose --profile multi-node up -d
# Update prometheus.yml with node IPs
```

**Why:** This sequence establishes a complete monitoring stack in under 5 minutes. Each command builds on the previous, ensuring a deterministic setup that scales from development to production.
