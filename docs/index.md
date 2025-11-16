<div class="hero-section">

# Geospatial Systems, Data, and Strange Machinery

*Building production-scale systems at the intersection of geospatial data, cloud infrastructure, and distributed computing*

<div class="hero-actions">
  [Best Practices →](best-practices/index.md){ .md-button .md-button--primary }
  [Tutorials →](tutorials/index.md){ .md-button .md-button--primary }
  [Projects →](projects.md){ .md-button .md-button--primary }
</div>

</div>

---

## Welcome

This is my technical studio—a curated collection of best practices, tutorials, and experiments from building production systems at **Oak Ridge National Laboratory** and beyond. Here you'll find deep dives into geospatial data engineering, distributed systems architecture, and the practical patterns that make complex systems reliable.

!!! tip "What to Expect"
    This site is organized into **Best Practices** (conceptual guides and patterns) and **Tutorials** (step-by-step implementations). Both are written for engineers who need production-ready solutions, not just examples.

---

## Quick Navigation

<div class="card-grid">

<div class="card">

### 🎯 Best Practices

Production-grade patterns, architectures, and methodologies for building reliable distributed systems.

**[Explore Best Practices →](best-practices/index.md)**

**Highlights:**
- [System Resilience & Concurrency](best-practices/operations-monitoring/system-resilience-and-concurrency.md)
- [Configuration Management](best-practices/operations-monitoring/configuration-management.md)
- [Release Management & Progressive Delivery](best-practices/operations-monitoring/release-management-and-progressive-delivery.md)
- [IAM & RBAC Governance](best-practices/security/iam-rbac-abac-governance.md)

</div>

<div class="card">

### 📚 Tutorials

Step-by-step guides with copy-paste examples for implementing key technologies and patterns.

**[Browse Tutorials →](tutorials/index.md)**

**Popular:**
- [PostGIS Geometry Indexing](tutorials/database-data-engineering/postgis-geometry-indexing.md)
- [PostgreSQL Auditing with PgAudit](tutorials/database-data-engineering/postgres-pgaudit-pgcron-auditing.md)
- [RKE2 on Raspberry Pi](tutorials/docker-infrastructure/rke2-raspberry-pi.md)
- [ONNX Browser Inference](tutorials/ml-ai/onnx-browser-inference.md)

</div>

<div class="card">

### 🏗️ Projects

Technical implementations, experiments, and creative explorations at the edge of what's possible.

**[View Projects →](projects.md)**

**Featured:**
- Final Fantasy Football (semantic learning)
- Where I've Been (travel visualization)
- This Is A Casino (trading platform)
- Decentralized Content Reward System

</div>

</div>

---

## 📌 Start Here

If you're new to this site, these foundational guides will give you the most value:

1. **[ADR and Technical Decision Governance](best-practices/architecture-design/adr-decision-governance.md)** — How we make and document architectural decisions
2. **[Configuration Management](best-practices/operations-monitoring/configuration-management.md)** — Managing configs across multi-environment systems
3. **[System Resilience & Concurrency](best-practices/operations-monitoring/system-resilience-and-concurrency.md)** — Patterns for building resilient distributed systems
4. **[IAM & RBAC Governance](best-practices/security/iam-rbac-abac-governance.md)** — Identity and access management across heterogeneous stacks

---

## Featured Deep Dives

### Parquet & Data Warehouses

**[Fast queries & cheap storage](best-practices/database-data/parquet.md)** — How to lay out partitions, size files and row groups, enable predicate pushdown & column pruning, serve efficiently over S3 byte-range, and wire up `parquet_s3_fdw` in Postgres for pushdown.

### Geospatial Data Engineering

**[PostGIS Best Practices](best-practices/postgres/postgis-best-practices.md)** — Production patterns for spatial indexing, raster workflows, and large-scale geospatial data management.

### Release Management

**[Release Management & Progressive Delivery](best-practices/operations-monitoring/release-management-and-progressive-delivery.md)** — Safe deployment strategies for coordinating changes across applications, databases, data pipelines, and ML systems.

---

## What I'm Working On

Currently focused on:

- **Advanced geospatial data warehouse architectures** using GeoParquet and PostGIS
- **Real-time IoT tracking systems** with Kafka, TimescaleDB, and geospatial processing
- **Production-grade configuration management** for multi-environment distributed systems
- **ML model deployment pipelines** with ONNX, MLflow, and progressive delivery

---

## About This Site

This documentation represents years of building production systems, learning from failures, and refining patterns that actually work. Everything here is battle-tested in real environments—from air-gapped clusters to cloud-native architectures.

The content is organized to be **immediately useful**: copy-paste examples, production-ready configurations, and clear explanations of trade-offs. No fluff, no theory without practice.

**[Learn more about me →](about.md)**

---

## Latest Updates

- **2025**: Comprehensive guides on release management, configuration governance, and IAM/RBAC patterns
- **2024**: Deep dives into system resilience, caching strategies, and secrets management
- **2023**: Expanded geospatial and data engineering tutorials

*For the most up-to-date content, check the [Best Practices](best-practices/index.md) and [Tutorials](tutorials/index.md) sections.*

---

## Connect

- **GitHub**: [@sempervent](https://github.com/sempervent)
- **LinkedIn**: [Joshua N. Grant](https://linkedin.com/in/joshuanagrant)
- **Email**: [jngrant@live.com](mailto:jngrant@live.com)
- **Blog**: [Not Just a Datum](https://notjustadatum.blogspot.com)

[Contact & Collaboration →](getting-started.md)
