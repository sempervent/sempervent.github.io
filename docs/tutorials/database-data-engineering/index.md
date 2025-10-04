# Database & Data Engineering Tutorials

**Objective**: Master complex database and data engineering implementations through step-by-step guides. When you need to implement specific database technologies, when you want to follow proven patterns, when you need copy-paste runnable examples—these tutorials become your weapon of choice.

## PostgreSQL & PostGIS

- **[PostGIS Geometry Indexing](postgis-geometry-indexing.md)** - Optimize spatial queries in PostGIS
- **[PostGIS Raster Indexing](postgis-raster-indexing.md)** - Efficiently manage and query raster data
- **[Raster–Vector Workflows](postgis-raster-vector-workflows.md)** - Combine raster and vector data for advanced analysis
- **[Alembic Migrations](alembic-migrations.md)** - Master database schema evolution with Alembic
- **[PostgreSQL Pooling (PgBouncer + FastAPI)](postgres-pooling.md)** - Optimize database connections for high-concurrency applications
- **[Solr + Postgres JSONB Search](solr-postgres-jsonb-search.md)** - Supercharge relational data with Solr's facets, suggesters, geo, and highlighting—using JSONB as source

## Data Processing & Analytics

- **[parquet_s3_fdw with Local, MinIO, Vast, and AWS](parquet-s3-fdw.md)** - Query Parquet directly from object storage
- **[GeoParquet with Polars](geoparquet-with-polars.md)** - High-performance spatial analytics with Polars
- **[Real-Time Data Processing](real-time-data-processing.md)** - Implement Kafka and TimescaleDB for streaming data
- **[Kafka + TimescaleDB IoT Streaming](kafka-timescaledb-iot.md)** - Stream simulated IoT telemetry into TimescaleDB
- **[Pulsar → Flink → Pinot (Realtime OLAP) + Superset](pulsar-flink-pinot-superset.md)** - A clean, fast streaming stack with sub-second analytics
- **[H3 + Tile38 + NATS + DuckDB](h3-tile38-nats-duckdb.md)** - Real-time geofences with hex aggregation and offline analytics
- **[H3 Raster to Hex](h3-raster-to-hex.md)** - Converting geospatial rasters to hexagonal grids for spatial analysis
- **[IPFS + SurrealDB + Meilisearch + NATS + Deno + Svelte](ipfs-surreal-meili-nats-deno-svelte.md)** - Content-addressed knowledge with verifiable blobs and fast search

## Orchestration & Pipelines

- **[Go-Glue OSM → PostGIS → Tiles Pipeline](go-osm-tiling-pipeline.md)** - A realistic Go CLI that orchestrates fetching OSM, loading PostGIS, generating tiles, and publishing to object storage

## Big Data Processing

- **[Apache Spark Mastery](apache-spark-mastery.md)** - Master distributed data processing with Apache Spark
- **[Apache Iceberg Mastery](apache-iceberg-mastery.md)** - Build, break, and bend ACID tables in the data lake

## Database Technologies

- **[Graph vs Vector Databases](graph-vs-vector-databases.md)** - When relationships meet similarity in modern data applications
- **[Geospatial Knowledge Graph](geospatial-knowledge-graph.md)** - Build intelligent location-aware systems with PostGIS + Neo4j
- **[DuckDB Parquet Data Quality](duckdb-parquet-data-quality.md)** - Inspect and validate Parquet files for data quality issues

---

*These tutorials provide the complete machinery for implementing key database and data engineering technologies and workflows. Each guide includes production-ready examples, configuration files, and best practices for enterprise deployment.*
