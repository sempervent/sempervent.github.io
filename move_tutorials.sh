#!/bin/bash
# Move Tutorial files to their appropriate topic-based directories

echo "Moving Tutorial files to topic-based directories..."

# Create tutorial directories
mkdir -p docs/tutorials/python-development
mkdir -p docs/tutorials/rust-development
mkdir -p docs/tutorials/docker-infrastructure
mkdir -p docs/tutorials/database-data-engineering
mkdir -p docs/tutorials/ml-ai
mkdir -p docs/tutorials/system-administration
mkdir -p docs/tutorials/data-science-visualization
mkdir -p docs/tutorials/development-tools
mkdir -p docs/tutorials/quick-start
mkdir -p docs/tutorials/just-for-fun

# Quick Start
echo "Moving Quick Start files..."
mv docs/tutorials/creating-mkdocs-github-site.md docs/tutorials/quick-start/
mv docs/tutorials/monitoring-with-grafana-prometheus.md docs/tutorials/quick-start/

# Python Development
echo "Moving Python Development files..."
mv docs/tutorials/psycopg2-to-psycopg3-migration.md docs/tutorials/python-development/
mv docs/tutorials/ruff-check-ignore-pyproject.md docs/tutorials/python-development/
mv docs/tutorials/r-shiny-geoapp.md docs/tutorials/python-development/
mv docs/tutorials/click-to-fastapi-conversion.md docs/tutorials/python-development/
mv docs/tutorials/websocket-chat-fastapi.md docs/tutorials/python-development/
mv docs/tutorials/js-glitch-observatory.md docs/tutorials/python-development/
mv docs/tutorials/chaos-engineering-k8s-python.md docs/tutorials/python-development/

# Rust Development
echo "Moving Rust Development files..."
mv docs/tutorials/rust-csr-parquet-db.md docs/tutorials/rust-development/
mv docs/tutorials/rust-event-sourcing.md docs/tutorials/rust-development/

# Docker & Infrastructure
echo "Moving Docker & Infrastructure files..."
mv docs/tutorials/slim-gpu-docker-images.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/slim-tf-gpu-images.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/slim-tf-gpu-images-skeleton.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/rke2-raspberry-pi.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/zfs-tank-nvme.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/ansible-slurm-raspberrypi.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/harbor-registry-setup.md docs/tutorials/docker-infrastructure/
mv docs/tutorials/ansible-dask-heterogeneous.md docs/tutorials/docker-infrastructure/

# Database & Data Engineering
echo "Moving Database & Data Engineering files..."
mv docs/tutorials/postgis-geometry-indexing.md docs/tutorials/database-data-engineering/
mv docs/tutorials/postgis-raster-indexing.md docs/tutorials/database-data-engineering/
mv docs/tutorials/postgis-raster-vector-workflows.md docs/tutorials/database-data-engineering/
mv docs/tutorials/alembic-migrations.md docs/tutorials/database-data-engineering/
mv docs/tutorials/postgres-pooling.md docs/tutorials/database-data-engineering/
mv docs/tutorials/parquet-s3-fdw.md docs/tutorials/database-data-engineering/
mv docs/tutorials/geoparquet-with-polars.md docs/tutorials/database-data-engineering/
mv docs/tutorials/real-time-data-processing.md docs/tutorials/database-data-engineering/
mv docs/tutorials/kafka-timescaledb-iot.md docs/tutorials/database-data-engineering/
mv docs/tutorials/apache-spark-mastery.md docs/tutorials/database-data-engineering/
mv docs/tutorials/apache-iceberg-mastery.md docs/tutorials/database-data-engineering/
mv docs/tutorials/graph-vs-vector-databases.md docs/tutorials/database-data-engineering/
mv docs/tutorials/geospatial-knowledge-graph.md docs/tutorials/database-data-engineering/
mv docs/tutorials/duckdb-parquet-data-quality.md docs/tutorials/database-data-engineering/

# Machine Learning & AI
echo "Moving Machine Learning & AI files..."
mv docs/tutorials/mlflow-api-experiments.md docs/tutorials/ml-ai/
mv docs/tutorials/onnx-browser-inference.md docs/tutorials/ml-ai/
mv docs/tutorials/rag-ollama-db.md docs/tutorials/ml-ai/
mv docs/tutorials/mcp-mlflow-toolchain.md docs/tutorials/ml-ai/
mv docs/tutorials/semantic-ml-training.md docs/tutorials/ml-ai/

# System Administration
echo "Moving System Administration files..."
mv docs/tutorials/ipxe-multi-boot.md docs/tutorials/system-administration/
mv docs/tutorials/remote-dev-tmux-screen.md docs/tutorials/system-administration/
mv docs/tutorials/prefect-fifo-redis.md docs/tutorials/system-administration/
mv docs/tutorials/awk-unix-text-processing.md docs/tutorials/system-administration/

# Data Science & Visualization
echo "Moving Data Science & Visualization files..."
mv docs/tutorials/jupyter-notebook-best-practices-geo.md docs/tutorials/data-science-visualization/
mv docs/tutorials/mermaid-diagrams.md docs/tutorials/data-science-visualization/
mv docs/tutorials/latex-tikz-diagrams.md docs/tutorials/data-science-visualization/
mv docs/tutorials/r-generative-art.md docs/tutorials/data-science-visualization/

# Development Tools
echo "Moving Development Tools files..."
mv docs/tutorials/jq-json-parsing-mastery.md docs/tutorials/development-tools/
mv docs/tutorials/find-files-parquet-fdw.md docs/tutorials/development-tools/
mv docs/tutorials/mosquitto-mqtt-python.md docs/tutorials/development-tools/
mv docs/tutorials/python-udp.md docs/tutorials/development-tools/
mv docs/tutorials/python-modbus-devices.md docs/tutorials/development-tools/

# Just for Fun
echo "Moving Just for Fun files..."
mv docs/tutorials/terminal-to-gif.md docs/tutorials/just-for-fun/
mv docs/tutorials/redis-midi-music.md docs/tutorials/just-for-fun/
mv docs/tutorials/postgis-webgl-art.md docs/tutorials/just-for-fun/
mv docs/tutorials/iot-ipfs-graphql-blender.md docs/tutorials/just-for-fun/
mv docs/tutorials/mqtt-timescaledb-websockets-threejs.md docs/tutorials/just-for-fun/
mv docs/tutorials/fastify-kafka-clickhouse-wasm-webgpu.md docs/tutorials/just-for-fun/
mv docs/tutorials/gonzo-prometheus-exporter.md docs/tutorials/just-for-fun/
mv docs/tutorials/git-weather-node-redis-ipfs-webrtc.md docs/tutorials/just-for-fun/
mv docs/tutorials/selenium-grid-docker-python.md docs/tutorials/just-for-fun/
mv docs/tutorials/martin-postgis-tiling.md docs/tutorials/just-for-fun/
mv docs/tutorials/managing-people-software-dev.md docs/tutorials/just-for-fun/

echo "All Tutorial files have been moved to their appropriate directories!"
echo "Directory structure:"
echo "  docs/tutorials/quick-start/ - Getting started guides"
echo "  docs/tutorials/python-development/ - Python development tutorials"
echo "  docs/tutorials/rust-development/ - Rust development tutorials"
echo "  docs/tutorials/docker-infrastructure/ - Docker and infrastructure tutorials"
echo "  docs/tutorials/database-data-engineering/ - Database and data engineering tutorials"
echo "  docs/tutorials/ml-ai/ - Machine learning and AI tutorials"
echo "  docs/tutorials/system-administration/ - System administration tutorials"
echo "  docs/tutorials/data-science-visualization/ - Data science and visualization tutorials"
echo "  docs/tutorials/development-tools/ - Development tools tutorials"
echo "  docs/tutorials/just-for-fun/ - Creative and experimental tutorials"