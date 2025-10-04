# Tutorials

**Objective**: Master complex technical implementations through step-by-step guides. When you need to implement specific technologies, when you want to follow proven patterns, when you need copy-paste runnable examples—these tutorials become your weapon of choice.

This collection provides comprehensive, hands-on tutorials for implementing key technologies and workflows. Each tutorial includes complete code examples, configuration files, and production-ready patterns.

## 🚀 Quick Start

### Getting Started
- **[Creating MkDocs GitHub Site](creating-mkdocs-github-site.md)** - Set up and deploy your own MkDocs site on GitHub Pages
- **[Monitoring with Grafana & Prometheus](monitoring-with-grafana-prometheus.md)** - Build a robust observability stack

## R Development
- **[Generative Art in R](r-generative-art.md)** - Creating beautiful algorithmic art with mathematical patterns


## 🐍 Python Development

### Core Python
- **[psycopg2 to psycopg 3 Migration](psycopg2-to-psycopg3-migration.md)** - Migrate to modern PostgreSQL driver with async support
- **[Ruff Check Ignore in pyproject.toml](ruff-check-ignore-pyproject.md)** - Configure Python linting exceptions

### Web Development
- **[R Shiny Geospatial App](r-shiny-geoapp.md)** - Build and deploy interactive spatial web applications
- **[Click CLI to FastAPI Conversion](click-to-fastapi-conversion.md)** - Convert Click CLIs into FastAPI endpoints
- **[WebSocket Chat with FastAPI](websocket-chat-fastapi.md)** - Real-time chat application with WebSockets
- **[Glitch Observatory (JS)](js-glitch-observatory.md)** - Visualizing entropy in JavaScript with chaotic art and sound
- **[Chaos Engineering with Kubernetes and Python](chaos-engineering-k8s-python.md)** - Controlled failure injection for system resilience

## 🦀 Rust Development

### Systems Programming
- **[Rust + CSR (Parse & Build from Parquet/DB)](rust-csr-parquet-db.md)** - Master sparse matrix operations in Rust
- **[Event-Sourcing in Rust](rust-event-sourcing.md)** - High-performance event-sourcing engine with async streams

## 🐳 Containerization & Infrastructure

### Docker & Orchestration
- **[Slimming GPU Docker Images](slim-gpu-docker-images.md)** - Optimize GPU-based containers for production
- **[Slimming TensorFlow GPU Images](slim-tf-gpu-images.md)** - Cut down TensorFlow GPU images to fighting weight
- **[TensorFlow GPU Slim Images - Repository Skeleton](slim-tf-gpu-images-skeleton.md)** - Complete production-ready example for building slim TensorFlow GPU containers
- **[Conda to Docker Migration](conda-to-docker-migration.md)** - Migrate conda environments to Docker

### Kubernetes & Clustering
- **[RKE2 on Raspberry Pi Farm](rke2-raspberry-pi.md)** - Set up a Raspberry Pi cluster running RKE2
- **[ZFS Tank with OS on NVMe](zfs-tank-nvme.md)** - Build enterprise-grade storage with ZFS

### Infrastructure Automation
- **[Deploy SLURM with Ansible on Raspberry Pi Cluster](ansible-slurm-raspberrypi.md)** - Build a HPC cluster with SLURM job scheduler
- **[Harbor Container Registry Setup](harbor-registry-setup.md)** - Enterprise container registry with network storage
- **[Dask with Ansible (CPU + GPU)](ansible-dask-heterogeneous.md)** - Heterogeneous distributed computing cluster


## 🗄️ Database & Data Engineering

### PostgreSQL & PostGIS
- **[PostGIS Geometry Indexing](postgis-geometry-indexing.md)** - Optimize spatial queries in PostGIS
- **[PostGIS Raster Indexing](postgis-raster-indexing.md)** - Efficiently manage and query raster data
- **[Raster–Vector Workflows](postgis-raster-vector-workflows.md)** - Combine raster and vector data for advanced analysis
- **[Alembic Migrations](alembic-migrations.md)** - Master database schema evolution with Alembic
- **[PostgreSQL Pooling (PgBouncer + FastAPI)](postgres-pooling.md)** - Optimize database connections for high-concurrency applications

### Data Processing & Analytics
- **[parquet_s3_fdw with Local, MinIO, Vast, and AWS](parquet-s3-fdw.md)** - Query Parquet directly from object storage
- **[GeoParquet with Polars](geoparquet-with-polars.md)** - High-performance spatial analytics with Polars
- **[Real-Time Data Processing](real-time-data-processing.md)** - Implement Kafka and TimescaleDB for streaming data
- **[Kafka + TimescaleDB IoT Streaming](kafka-timescaledb-iot.md)** - Stream simulated IoT telemetry into TimescaleDB

### Big Data Processing
- **[Apache Spark Mastery](apache-spark-mastery.md)** - Master distributed data processing with Apache Spark
- **[Apache Iceberg Mastery](apache-iceberg-mastery.md)** - Build, break, and bend ACID tables in the data lake

### Database Technologies
- **[Graph vs Vector Databases](graph-vs-vector-databases.md)** - When relationships meet similarity in modern data applications
- **[Geospatial Knowledge Graph](geospatial-knowledge-graph.md)** - Build intelligent location-aware systems with PostGIS + Neo4j
- **[DuckDB Parquet Data Quality](duckdb-parquet-data-quality.md)** - Inspect and validate Parquet files for data quality issues

## 🤖 Machine Learning & AI

### ML Operations
- **[MLflow API Experiments](mlflow-api-experiments.md)** - Master MLflow for experiment tracking and model management
- **[ONNX Browser Inference](onnx-browser-inference.md)** - Deploy machine learning models directly in the browser
- **[RAG with Ollama + Database](rag-ollama-db.md)** - Build intelligent data interfaces with retrieval-augmented generation
- **[MCP ↔ MLflow Toolchain](mcp-mlflow-toolchain.md)** - Data-agnostic LLM experimentation with Model Context Protocol
- **[Semantic ML Training](semantic-ml-training.md)** - Master semantic machine learning from embeddings to production

## 🔧 System Administration

### Network & Boot Management
- **[iPXE Multi-System Booting](ipxe-multi-boot.md)** - Create network boot images for multiple systems
- **[Remote Dev with tmux & screen](remote-dev-tmux-screen.md)** - Bulletproof remote development for when the wire goes dead

### Workflow Orchestration
- **[FIFO Prefect Flow with Redis](prefect-fifo-redis.md)** - Implement FIFO greedy scheduler patterns in Prefect

### Unix Tools
- **[AWK Unix Text Processing](awk-unix-text-processing.md)** - Master AWK for parsing ls, ps aux, and system data

## 📊 Data Science & Visualization

### Jupyter & Notebooks
- **[Jupyter Notebook Best Practices (Geospatial Edition)](jupyter-notebook-best-practices-geo.md)** - Master Jupyter for geospatial data exploration

### Visualization & Diagrams
- **[Mermaid Diagrams in MkDocs](mermaid-diagrams.md)** - Create beautiful diagrams with Mermaid
- **[TikZ Diagrams in LaTeX](latex-tikz-diagrams.md)** - Create professional diagrams with LaTeX and TikZ

## 🛠️ Development Tools

### JSON Processing
- **[jq JSON Parsing Mastery](jq-json-parsing-mastery.md)** - Master command-line JSON manipulation with `jq`

### File Management
- **[find_files for parquet_s3_fdw](find-files-parquet-fdw.md)** - Build customizable file discovery for Parquet workflows

### Messaging & Communication
- **[Mosquitto + Python (MQTT Best Practices)](mosquitto-mqtt-python.md)** - Master MQTT messaging for IoT and event-driven applications
- **[Python UDP Messaging](python-udp.md)** - Master UDP networking for high-performance, real-time applications
- **[Python Modbus Device Communication](python-modbus-devices.md)** - Industrial device communication with PLCs, sensors, and automation equipment

## Just for Fun
- **[Terminal to GIF](terminal-to-gif.md)** - Capturing command-line magic and converting to animated GIFs
- **[Redis Streams + Web MIDI](redis-midi-music.md)** - Procedural MIDI jams with Redis Streams and Web MIDI API
- **[PostGIS Rasters + WebGL Art](postgis-webgl-art.md)** - From elevation models to shader dreams with PostGIS and WebGL
- **[IoT + IPFS + GraphQL + Blender](iot-ipfs-graphql-blender.md)** - IoT data in the metaverse with decentralized storage and 3D visualization
- **[MQTT + TimescaleDB + WebSockets + Three.js](mqtt-timescaledb-websockets-threejs.md)** - From sensors to sculptures with real-time IoT data as 3D geometry
- **[Fastify + Kafka + ClickHouse + WASM + WebGPU](fastify-kafka-clickhouse-wasm-webgpu.md)** - Real-time anomaly radar with streaming analytics and GPU visualization
- **[Gonzo Prometheus Exporter](gonzo-prometheus-exporter.md)** - Building a surreal Prometheus metrics exporter in Docker (weird metrics, real monitoring)
- **[Git Commit Weather Station](git-weather-node-redis-ipfs-webrtc.md)** - Node.js + RedisTimeSeries + IPFS + WebRTC (in Docker)
- **[Selenium Grid (Docker) + Python](selenium-grid-docker-python.md)** - ChromeDriver headless testing with Docker Compose

## 📚 Tutorial Categories

### By Technology
- **Database**: PostgreSQL, PostGIS, TimescaleDB, Kafka, Graph, Vector, DuckDB
- **Python**: FastAPI, psycopg, Alembic, Polars
- **Rust**: Sparse matrices, performance computing
- **Docker**: Container optimization, GPU workloads
- **Kubernetes**: RKE2, Raspberry Pi clusters
- **Infrastructure**: ZFS, iPXE, network booting
- **Messaging**: MQTT, Mosquitto, UDP, real-time communication
- **Big Data**: Apache Spark, Apache Iceberg, distributed processing, data engineering
- **System Administration**: AWK, Unix utilities, text processing, system monitoring
- **Machine Learning**: ONNX browser inference, client-side ML, web-based AI, RAG systems
- **Geospatial**: PostGIS, Neo4j, knowledge graphs, spatial reasoning

### By Use Case
- **Data Engineering**: ETL pipelines, streaming data, spatial analytics
- **Web Development**: APIs, interactive applications, deployment
- **System Administration**: Clustering, storage, network management, remote development
- **Machine Learning**: Experiment tracking, model deployment, MCP-MLflow toolchains, semantic ML
- **Visualization**: Diagrams, notebooks, interactive tools

### By Complexity
- **Beginner**: Basic setup and configuration
- **Intermediate**: Production deployment and optimization
- **Advanced**: Performance tuning and enterprise patterns

## 🎯 How to Use These Tutorials

1. **Choose Your Technology**: Browse by technology or use case
2. **Follow Step-by-Step**: Each tutorial provides complete implementation
3. **Adapt to Your Needs**: Modify configurations for your environment
4. **Scale to Production**: Use advanced patterns for enterprise deployment

## 🔗 Related Resources

- **[Best Practices](../best-practices/)** - Senior-level implementation patterns
- **[Professional Profile](../about.md)** - Background and experience
- **[Projects Portfolio](../projects.md)** - Real-world implementations

---

*These tutorials provide the complete machinery for implementing key technologies and workflows. Each guide includes production-ready examples, configuration files, and best practices for enterprise deployment.*
