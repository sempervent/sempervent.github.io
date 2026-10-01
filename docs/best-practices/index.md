# Best Practices

Notes on how systems behave in the wild: trade-offs, failure modes, and the parts of operations that survive contact with production.

These pages answer *why* and *when* more often than *follow these steps in order*. Code shows up when it clarifies a pattern — the point is judgment, not a runbook.

## Start here

- [Geospatial system architecture](geospatial/geospatial-system-design.md)
- [Failure-oriented system design](operations/failure-oriented-design.md)
- [System resilience and concurrency](operations-monitoring/system-resilience-and-concurrency.md)
- [PostGIS patterns](postgres/postgis-best-practices.md)
- [Parquet and GeoParquet](database-data/parquet.md) · [GeoParquet](database-data/geoparquet.md)

## Languages and runtimes

- [Python](python/index.md)
- [Rust](rust/index.md)
- [Go](go/index.md)
- [R](r/index.md)

## Data and platforms

- [Database and data management](database-data/index.md)
- [PostgreSQL](postgres/index.md)
- [Machine learning and AI](ml-ai/index.md)
- [Docker and infrastructure](docker-infrastructure/index.md)
- [Git and version control](git/index.md)

## Architecture and operations

- [Architecture and design](architecture-design/index.md)
- [Operations and monitoring](operations-monitoring/index.md)
- [Data governance](data-governance/index.md)
- [Security](security/index.md)

## Embedded and home lab

- [ESP32 and embedded](esp32/index.md)
- [Power electronics](embedded/index.md)
- [Home automation](home-automation/index.md)

## Diagrams

- [Systems diagramming](diagrams/systems-diagramming-best-practices.md)
- [Mermaid → SVG workflow](diagrams/svg-workflow-generation.md)

## Creative reference

- [Creative and fun patterns](creative-fun/index.md) — idempotency, Celery notes, and similar (not the same as [Just for Fun tutorials](../tutorials/just-for-fun/index.md))

---

*Pick a section that matches the problem; cross-links inside each guide point to [tutorials](../tutorials/index.md) when you need a hands-on walkthrough.*
