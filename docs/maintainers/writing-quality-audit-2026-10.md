# Writing quality audit (2026-10)

Generated 2026-09-30 by `scripts/generate_writing_audit.py`.
Disposition is heuristic — review before bulk deletes.

## Summary counts

- **KEEP**: 142
- **LIGHT EDIT**: 189
- **REWRITE**: 36
- **MERGE**: 0
- **MOVE**: 0
- **ARCHIVE/REMOVE**: 22

## Pages

| Path | Category | Disposition | Reason | Verify commands? | Overlap |
| --- | --- | --- | --- | --- | --- |
| best-practices/architecture/cost-aware-systems.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/architecture-design/adr-decision-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, Key Takeaways; 1598 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/api-gateway-architecture.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers, gateway to | if retained |  |
| best-practices/architecture-design/api-governance-interface-stability.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/architecture-fitness-functions-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2045 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/cache-topology-architecture.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/caching-performance.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/architecture-design/capacity-planning-and-workload-modeling.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/ci-cd-pipelines.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/architecture-design/cloud-architecture.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/architecture-design/cognitive-load-developer-experience.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2224 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/cost-aware-architecture-and-efficiency-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/data-mesh-architecture.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/documentation.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters; 1184 lines (catalog risk) | if retained |  |
| best-practices/architecture-design/environment-config-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2548 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/event-driven-architecture.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2843 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/index.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/architecture-design/multi-cloud-federation-portability.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/multi-region-dr-strategy.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/polyglot-interoperability-design.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/protobuf-python.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready; 1375 lines (catalog risk) | if retained |  |
| best-practices/architecture-design/rdf-owl-metadata-automation.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/architecture-design/reference-architecture-diagrams.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/repository-standardization-and-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2180 lines (catalog risk) | n/a |  |
| best-practices/architecture-design/secrets-management.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/architecture-design/service-decomposition-strategy.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/streaming-architecture-patterns.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/system-taxonomy-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/architecture-design/temporal-governance-and-time-synchronization.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2132 lines (catalog risk) | n/a |  |
| best-practices/creative-fun/celery-best-practices.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/creative-fun/idempotency-and-dedup.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/creative-fun/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/creative-fun/latex.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/creative-fun/time-hygiene.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/creative-fun/yaml-recipe-format.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready; 1015 lines (catalog risk) | if retained |  |
| best-practices/data/metadata-control-plane.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/data/reproducible-data-pipelines.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/data-governance/data-freshness-sla-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/data-governance/data-lineage-contracts.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/data-governance/data-quality-sla-validation-observability.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/data-governance/data-retention-archival-lifecycle-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/data-governance/data-validation-and-contract-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2957 lines (catalog risk) | n/a |  |
| best-practices/data-governance/index.md | best-practices | LIGHT EDIT | fingerprints: complete framework | if retained |  |
| best-practices/data-governance/metadata-provenance-contracts.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2885 lines (catalog risk) | n/a |  |
| best-practices/data-processing/spark/scaling-spark.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/data-processing/spark/spark-modern-architecture.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/data-processing/spark/spark-on-kubernetes.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/data-processing/spark/spark-performance-tuning.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/data-processing/spark/when-to-use-spark.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/database-data/ai-ml-geospatial-knowledge-graph.md | best-practices | ARCHIVE/REMOVE | fingerprints: This guide provides, What This Guide Covers, Why This Matters; 2036 lines (catalog risk) | n/a |  |
| best-practices/database-data/aws-serverless-geospatial.md | best-practices | LIGHT EDIT | fingerprints: This guide provides, production-ready | if retained |  |
| best-practices/database-data/data-engineering.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/database-data/data-lake-governance.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters; 1004 lines (catalog risk) | if retained |  |
| best-practices/database-data/data-lineage-contracts.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/database-data/database-migrations.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters; 1003 lines (catalog risk) | if retained |  |
| best-practices/database-data/database-optimization.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/database-data/etl-pipeline-design.md | best-practices | LIGHT EDIT | fingerprints: This guide provides, production-ready | if retained |  |
| best-practices/database-data/geoparquet-data-warehouses.md | best-practices | LIGHT EDIT | fingerprints: This guide provides, production-ready | if retained |  |
| best-practices/database-data/geoparquet.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/database-data/geospatial-benchmarking.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters; 1053 lines (catalog risk) | if retained |  |
| best-practices/database-data/geospatial-data-engineering.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/database-data/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/database-data/lake-vs-lakehouse-vs-warehouse.md | best-practices | REWRITE | fingerprints: Why This Matters | if retained |  |
| best-practices/database-data/parquet.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/database-data/patroni-postgres-ha.md | best-practices | REWRITE | fingerprints: Why This Matters, enterprise-grade | yes |  |
| best-practices/database-data/semantic-layer-engineering.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/diagrams/svg-workflow-generation.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/diagrams/systems-diagramming-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/docker-infrastructure/ansible-inventory-management.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/docker-infrastructure/ansible-performance-optimization.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters, enterprise-grade | if retained |  |
| best-practices/docker-infrastructure/ansible-playbook-design.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/docker-infrastructure/ansible-security-hardening.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters, enterprise-grade | if retained |  |
| best-practices/docker-infrastructure/conda-to-docker-migration.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/docker-infrastructure/docker-and-compose.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/docker-infrastructure/docker-sbom-trivy-cve-mitigation.md | best-practices | LIGHT EDIT | 1582 lines (catalog risk) | if retained |  |
| best-practices/docker-infrastructure/index.md | best-practices | LIGHT EDIT | fingerprints: enterprise-grade | if retained |  |
| best-practices/docker-infrastructure/jinja-best-practices.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/docker-infrastructure/nginx-best-practices.md | best-practices | REWRITE | fingerprints: What This Guide Covers, production-ready; 1765 lines (catalog risk) | if retained |  |
| best-practices/docker-infrastructure/nginx-production.md | best-practices | REWRITE | fingerprints: Why This Matters, enterprise-grade, production-ready; 1184 lines (catalog risk) | if retained |  |
| best-practices/docker-infrastructure/tmux-advanced.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/embedded/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/embedded/power-electronics-for-esp32.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/e-ink-display-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/embedded-security-and-ota.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/esp32-hardware-and-electrical-safety.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/esp32-programming-architecture.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/esp32-s3-and-c3-architecture-notes.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/esp32-safety-checklist-printable.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/lora-best-practices-sx127x.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/mqtt-security-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/power-management-and-deep-sleep.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/esp32/sensor-integration-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/geospatial/geospatial-system-design.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/git/git-production.md | best-practices | REWRITE | fingerprints: Why This Matters, enterprise-grade | if retained |  |
| best-practices/git/git-workflows-collaboration.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/git/gitflow-best-practices.md | best-practices | REWRITE | fingerprints: production-ready; 1351 lines (catalog risk) | if retained |  |
| best-practices/git/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/go/go-api-design.md | best-practices | LIGHT EDIT | 1104 lines (catalog risk) | if retained |  |
| best-practices/go/go-cicd-pipelines.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/go-concurrency-optimization.md | best-practices | LIGHT EDIT | 1035 lines (catalog risk) | if retained |  |
| best-practices/go/go-concurrency-patterns.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/go-containerization.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/go-data-processing.md | best-practices | LIGHT EDIT | 1213 lines (catalog risk) | if retained |  |
| best-practices/go/go-database-patterns.md | best-practices | LIGHT EDIT | 1247 lines (catalog risk) | if retained |  |
| best-practices/go/go-dev-environment.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/go/go-error-handling.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/go/go-geospatial-development.md | best-practices | LIGHT EDIT | 1135 lines (catalog risk) | if retained |  |
| best-practices/go/go-memory-management.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/go-microservices.md | best-practices | LIGHT EDIT | 1042 lines (catalog risk) | if retained |  |
| best-practices/go/go-monitoring-observability.md | best-practices | LIGHT EDIT | 1256 lines (catalog risk) | if retained |  |
| best-practices/go/go-performance-tuning.md | best-practices | LIGHT EDIT | 1073 lines (catalog risk) | if retained |  |
| best-practices/go/go-testing-best-practices.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/go-web-services.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/go/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/go/tui-applications.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/home-automation/home-assistant-security-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/home-automation/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/index.md | index | KEEP | low template signal | if retained |  |
| best-practices/ml-ai/embeddings-and-vector-databases.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready; 1229 lines (catalog risk) | if retained |  |
| best-practices/ml-ai/index.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/ml-ai/mcp-fastapi-stack.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/ml-ai/ml-systems-architecture-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/ml-ai/onnx-model-optimization.md | best-practices | REWRITE | fingerprints: Why This Matters, enterprise-grade, production-ready; 1143 lines (catalog risk) | if retained |  |
| best-practices/ml-ai/prompting-llms.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/ml-ai/r-data-exploration.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| best-practices/ml-ai/vibe-to-agentic.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/operations/failure-oriented-design.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/operations/system-resilience-and-concurrency.md | best-practices | REWRITE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2389 lines (catalog risk) | if retained |  |
| best-practices/operations-monitoring/blast-radius-risk-modeling.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/operations-monitoring/chaos-engineering-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/operations-monitoring/configuration-drift-detection-prevention.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 3003 lines (catalog risk) | n/a |  |
| best-practices/operations-monitoring/configuration-management.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 1601 lines (catalog risk) | n/a |  |
| best-practices/operations-monitoring/environment-promotion-drift-governance.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/operations-monitoring/grafana-prometheus-loki-observability.md | best-practices | ARCHIVE/REMOVE | fingerprints: This guide provides, production-ready; 2068 lines (catalog risk) | n/a |  |
| best-practices/operations-monitoring/grafana.md | best-practices | REWRITE | fingerprints: Why This Matters, enterprise-grade; 1116 lines (catalog risk) | if retained |  |
| best-practices/operations-monitoring/index.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/operations-monitoring/logging-observability.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/operations-monitoring/observability-driven-development.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/operations-monitoring/operational-resilience-and-incident-response.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2347 lines (catalog risk) | n/a |  |
| best-practices/operations-monitoring/performance-monitoring.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/operations-monitoring/release-management-and-progressive-delivery.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 1885 lines (catalog risk) | n/a |  |
| best-practices/operations-monitoring/secrets-config.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/operations-monitoring/system-resilience-and-concurrency.md | best-practices | REWRITE | fingerprints: production-ready | if retained |  |
| best-practices/operations-monitoring/testing-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/operations-monitoring/unified-observability-architecture.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/performance/end-to-end-caching-strategy.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2560 lines (catalog risk) | n/a |  |
| best-practices/performance/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/postgres/fdw-postgres.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | yes |  |
| best-practices/postgres/index.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgis-best-practices.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-api-development.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-backup-recovery.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-cloud-integration.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-configuration-management.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-constraints-validation.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-containerization.md | best-practices | LIGHT EDIT | low template signal | yes |  |
| best-practices/postgres/postgres-data-pipeline-integration.md | best-practices | LIGHT EDIT | low template signal | yes |  |
| best-practices/postgres/postgres-data-types.md | best-practices | LIGHT EDIT | low template signal | yes |  |
| best-practices/postgres/postgres-database-design.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-deployment-strategies.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-dev-environment.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-event-driven.md | best-practices | LIGHT EDIT | low template signal | yes |  |
| best-practices/postgres/postgres-extensions.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-fulltext-search.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-indexing-strategies.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-json-jsonb.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-large-objects.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-maintenance-vacuum.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-monitoring-observability.md | best-practices | LIGHT EDIT | low template signal | yes |  |
| best-practices/postgres/postgres-partitioning.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-performance-tuning.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-pooling.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | yes |  |
| best-practices/postgres/postgres-replication-ha.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-scaling-strategies.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-security-best-practices.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-timeseries.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-transactions-concurrency.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/postgres/postgres-troubleshooting.md | best-practices | KEEP | low template signal | yes |  |
| best-practices/python/api-development.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/python/dx-architecture-and-golden-paths.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/python/fastapi-geospatial.md | best-practices | LIGHT EDIT | fingerprints: This guide provides, production-ready | if retained |  |
| best-practices/python/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/pytest-best-practices.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| best-practices/python/python-api-design.md | best-practices | LIGHT EDIT | 1111 lines (catalog risk) | if retained |  |
| best-practices/python/python-async-best-practices.md | best-practices | REWRITE | fingerprints: Why This Matters; 1270 lines (catalog risk) | if retained |  |
| best-practices/python/python-async-programming.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-caching-strategies.md | best-practices | LIGHT EDIT | 1003 lines (catalog risk) | if retained |  |
| best-practices/python/python-cicd-pipelines.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-code-quality.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-compliance.md | best-practices | LIGHT EDIT | 1053 lines (catalog risk) | if retained |  |
| best-practices/python/python-concurrency-patterns.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-containerization.md | best-practices | LIGHT EDIT | 1161 lines (catalog risk) | if retained |  |
| best-practices/python/python-data-analysis.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-data-processing.md | best-practices | LIGHT EDIT | 1114 lines (catalog risk) | if retained |  |
| best-practices/python/python-data-storage.md | best-practices | LIGHT EDIT | 1088 lines (catalog risk) | if retained |  |
| best-practices/python/python-database-patterns.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-dev-environment.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/python-error-handling.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-geospatial-development.md | best-practices | LIGHT EDIT | 1049 lines (catalog risk) | if retained |  |
| best-practices/python/python-machine-learning.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-memory-management.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-microservices.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-monitoring-observability.md | best-practices | LIGHT EDIT | 1103 lines (catalog risk) | if retained |  |
| best-practices/python/python-package-development.md | best-practices | LIGHT EDIT | 1025 lines (catalog risk) | if retained |  |
| best-practices/python/python-package.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/python-performance-tuning.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/python-production-deployment.md | best-practices | LIGHT EDIT | 1150 lines (catalog risk) | if retained |  |
| best-practices/python/python-secrets-management.md | best-practices | LIGHT EDIT | 1033 lines (catalog risk) | if retained |  |
| best-practices/python/python-security-best-practices.md | best-practices | LIGHT EDIT | 1052 lines (catalog risk) | if retained |  |
| best-practices/python/python-testing-best-practices.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-threading-and-multiprocessing.md | best-practices | REWRITE | fingerprints: Why This Matters, production-ready; 1089 lines (catalog risk) | if retained |  |
| best-practices/python/python-type-hints.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-web-performance.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/python/python-web-services.md | best-practices | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| best-practices/python/tui-applications.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/typing-in-python.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/python/web-performance-optimization.md | best-practices | LIGHT EDIT | fingerprints: This guide provides | if retained |  |
| best-practices/r/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/r/r-big-data-processing.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-cicd-pipelines.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-containerization.md | best-practices | LIGHT EDIT | 1016 lines (catalog risk) | if retained |  |
| best-practices/r/r-data-analysis-workflows.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-data-exploration.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-data-processing.md | best-practices | LIGHT EDIT | 1062 lines (catalog risk) | if retained |  |
| best-practices/r/r-database-integration.md | best-practices | LIGHT EDIT | 1081 lines (catalog risk) | if retained |  |
| best-practices/r/r-dev-environment.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/r/r-geospatial-analysis.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-interactive-applications.md | best-practices | LIGHT EDIT | 1084 lines (catalog risk) | if retained |  |
| best-practices/r/r-machine-learning.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-memory-management.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-package-development.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/r/r-parallel-computing.md | best-practices | LIGHT EDIT | 1077 lines (catalog risk) | if retained |  |
| best-practices/r/r-performance-tuning.md | best-practices | LIGHT EDIT | 1034 lines (catalog risk) | if retained |  |
| best-practices/r/r-production-deployment.md | best-practices | LIGHT EDIT | 1239 lines (catalog risk) | if retained |  |
| best-practices/r/r-reporting-workflows.md | best-practices | LIGHT EDIT | 1010 lines (catalog risk) | if retained |  |
| best-practices/r/r-statistical-modeling.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/r/r-testing-best-practices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/r/r-visualization-best-practices.md | best-practices | LIGHT EDIT | 1051 lines (catalog risk) | if retained |  |
| best-practices/rust/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-api-design.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-caching-strategies.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-cicd-pipelines.md | best-practices | LIGHT EDIT | 1042 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-code-quality.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-compliance.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-concurrency-patterns.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-containerization.md | best-practices | LIGHT EDIT | 1020 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-data-analysis.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-data-processing.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-data-storage.md | best-practices | LIGHT EDIT | 1107 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-database-patterns.md | best-practices | LIGHT EDIT | 1004 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-dev-environment.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-error-handling.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-generics-traits.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-geospatial-development.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-machine-learning.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-memory-management.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-microservices.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-monitoring-observability.md | best-practices | LIGHT EDIT | 1160 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-ownership-borrowing.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-package-development.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-performance-tuning.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-production-deployment.md | best-practices | LIGHT EDIT | 1022 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-secrets-management.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-security-best-practices.md | best-practices | LIGHT EDIT | 1027 lines (catalog risk) | if retained |  |
| best-practices/rust/rust-testing-best-practices.md | best-practices | LIGHT EDIT | low template signal | if retained |  |
| best-practices/rust/rust-unsafe-programming.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-web-performance.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/rust-web-services.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/rust/tui-applications.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/security/encryption-lifecycle-and-crypto-rotation.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/security/iam-rbac-abac-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2648 lines (catalog risk) | n/a |  |
| best-practices/security/identity-federation-authz-authn-architecture.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/security/index.md | best-practices | KEEP | low template signal | if retained |  |
| best-practices/security/secrets-governance.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2867 lines (catalog risk) | n/a |  |
| best-practices/security/secure-by-design-polyglot.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/security/secure-sandboxing-and-multi-tenant-isolation.md | best-practices | LIGHT EDIT | fingerprints: What This Guide Covers | if retained |  |
| best-practices/testing/end-to-end-testing-strategy.md | best-practices | ARCHIVE/REMOVE | fingerprints: complete framework, This guide provides, What This Guide Covers; 2295 lines (catalog risk) | n/a |  |
| best-practices/testing/index.md | best-practices | KEEP | low template signal | if retained |  |
| tutorials/data-science-visualization/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/data-science-visualization/jupyter-notebook-best-practices-geo.md | best-practices | LIGHT EDIT | fingerprints: Why This Matters, production-ready | if retained |  |
| tutorials/data-science-visualization/latex-tikz-diagrams.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/data-science-visualization/mermaid-diagrams.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/data-science-visualization/r-generative-art.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/database-data-engineering/alembic-migrations.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1151 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/apache-iceberg-mastery.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/database-data-engineering/apache-spark-mastery.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/database-data-engineering/duckdb-parquet-data-quality.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters, production-ready; 1069 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/geoparquet-with-polars.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/database-data-engineering/geospatial-knowledge-graph.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready; 1094 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/go-osm-tiling-pipeline.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/database-data-engineering/graph-vs-vector-databases.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1047 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/h3-raster-to-hex.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/database-data-engineering/h3-tile38-nats-duckdb.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/database-data-engineering/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/database-data-engineering/ipfs-surreal-meili-nats-deno-svelte.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/database-data-engineering/kafka-timescaledb-iot.md | tutorials | REWRITE | fingerprints: This tutorial provides, production-ready; 1048 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/openmaptiles-us-dark-z12.md | tutorials | LIGHT EDIT | fingerprints: This tutorial provides; 1084 lines (catalog risk) | if retained |  |
| tutorials/database-data-engineering/parquet-s3-fdw.md | tutorials | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| tutorials/database-data-engineering/postgis-geometry-indexing.md | tutorials | LIGHT EDIT | low template signal | yes |  |
| tutorials/database-data-engineering/postgis-raster-indexing.md | tutorials | KEEP | low template signal | yes |  |
| tutorials/database-data-engineering/postgis-raster-vector-workflows.md | tutorials | KEEP | low template signal | yes |  |
| tutorials/database-data-engineering/postgres-lakehouse-pglake-parquet-fdw.md | tutorials | KEEP | low template signal | yes |  |
| tutorials/database-data-engineering/postgres-pgaudit-pgcron-auditing.md | tutorials | REWRITE | fingerprints: This tutorial provides, production-ready; 1307 lines (catalog risk) | yes |  |
| tutorials/database-data-engineering/postgres-pooling.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | yes |  |
| tutorials/database-data-engineering/pulsar-flink-pinot-superset.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/database-data-engineering/real-time-data-processing.md | tutorials | LIGHT EDIT | fingerprints: This tutorial provides, production-ready | if retained |  |
| tutorials/database-data-engineering/solr-postgres-jsonb-search.md | tutorials | LIGHT EDIT | low template signal | yes |  |
| tutorials/development-tools/find-files-parquet-fdw.md | tutorials | LIGHT EDIT | fingerprints: This tutorial provides, Why This Matters | if retained |  |
| tutorials/development-tools/go-tech-mixer.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/development-tools/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/development-tools/jq-json-parsing-mastery.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/development-tools/mosquitto-mqtt-python.md | tutorials | REWRITE | fingerprints: Why This Matters; 1247 lines (catalog risk) | if retained |  |
| tutorials/development-tools/python-modbus-devices.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1195 lines (catalog risk) | if retained |  |
| tutorials/development-tools/python-udp.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1091 lines (catalog risk) | if retained |  |
| tutorials/development-tools/tauri-rqlite-syncthing.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/diagrams/layered-systems-diagrams-mermaid-to-svg.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/diagrams/mermaid-to-svg-workflow-pipeline.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/docker-infrastructure/ansible-dask-heterogeneous.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1107 lines (catalog risk) | if retained |  |
| tutorials/docker-infrastructure/ansible-rke2-rancher-pgo-prefect.md | tutorials | KEEP | low template signal | yes |  |
| tutorials/docker-infrastructure/ansible-slurm-raspberrypi.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| tutorials/docker-infrastructure/compose-profiles-polyglot-stack.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/docker-infrastructure/harbor-registry-setup.md | tutorials | ARCHIVE/REMOVE | fingerprints: Why This Matters, enterprise-grade, production-ready; 1302 lines (catalog risk) | n/a |  |
| tutorials/docker-infrastructure/index.md | tutorials | LIGHT EDIT | fingerprints: enterprise-grade, production-ready | if retained |  |
| tutorials/docker-infrastructure/multistage-conda-to-scratch.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/docker-infrastructure/rke2-raspberry-pi.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | yes |  |
| tutorials/docker-infrastructure/slim-geospatial-gpu-conda.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/docker-infrastructure/slim-gpu-docker-images.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/docker-infrastructure/slim-tf-gpu-images-skeleton.md | tutorials | LIGHT EDIT | fingerprints: production-ready | if retained |  |
| tutorials/docker-infrastructure/slim-tf-gpu-images.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/docker-infrastructure/zfs-tank-nvme.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/embedded/esp32-eink-sensor-monitor.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/embedded/esp32-mqtt-home-assistant-integration.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/embedded/esp32-rf-room-light-controller.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/embedded/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/go-development/building-a-go-tui.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/go-development/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/index.md | index | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/fastify-kafka-clickhouse-wasm-webgpu.md | just-for-fun | ARCHIVE/REMOVE | fingerprints: weapon of choice, complete machinery, This tutorial provides; 1173 lines (catalog risk) | n/a |  |
| tutorials/just-for-fun/fractal-art-explorer-js.md | just-for-fun | KEEP | fingerprints: complete machinery, This tutorial provides, production-ready | if retained |  |
| tutorials/just-for-fun/git-weather-node-redis-ipfs-webrtc.md | just-for-fun | KEEP | fingerprints: complete machinery, This guide provides, Why This Matters | if retained |  |
| tutorials/just-for-fun/go-auth-scratch-compose.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/gonzo-prometheus-exporter.md | just-for-fun | KEEP | fingerprints: complete machinery, This guide provides, Why This Matters | if retained |  |
| tutorials/just-for-fun/index.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/iot-ipfs-graphql-blender.md | just-for-fun | ARCHIVE/REMOVE | fingerprints: weapon of choice, complete machinery, This tutorial provides; 1208 lines (catalog risk) | n/a |  |
| tutorials/just-for-fun/js-glitch-observatory.md | just-for-fun | REWRITE | fingerprints: weapon of choice, complete machinery, This tutorial provides | if retained |  |
| tutorials/just-for-fun/kotlin-cellular-automata-garden.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/kotlin-midi-particle-nebula.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/kotlin-recursive-cathedral.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/managing-people-software-dev.md | just-for-fun | KEEP | fingerprints: complete machinery, This guide provides, Why This Matters | if retained |  |
| tutorials/just-for-fun/martin-postgis-tiling.md | just-for-fun | KEEP | fingerprints: complete machinery, This guide provides, Why This Matters | yes |  |
| tutorials/just-for-fun/mqtt-timescaledb-websockets-threejs.md | just-for-fun | REWRITE | fingerprints: weapon of choice, complete machinery, This tutorial provides; 1190 lines (catalog risk) | if retained |  |
| tutorials/just-for-fun/osc-mqtt-prometheus-supercollider.md | just-for-fun | KEEP | fingerprints: complete machinery, This tutorial provides, production-ready | if retained |  |
| tutorials/just-for-fun/pi-infinite-art-frame-kotlin.md | just-for-fun | KEEP | low template signal | if retained |  |
| tutorials/just-for-fun/pi-sample-server.md | just-for-fun | KEEP | 1635 lines (catalog risk) | if retained |  |
| tutorials/just-for-fun/postgis-webgl-art.md | just-for-fun | KEEP | fingerprints: weapon of choice, complete machinery, This tutorial provides | yes |  |
| tutorials/just-for-fun/redis-midi-music.md | just-for-fun | KEEP | fingerprints: weapon of choice, complete machinery, This tutorial provides | if retained |  |
| tutorials/just-for-fun/selenium-grid-docker-python.md | just-for-fun | KEEP | fingerprints: complete machinery, This guide provides, Why This Matters; 1029 lines (catalog risk) | if retained |  |
| tutorials/just-for-fun/terminal-to-gif.md | just-for-fun | KEEP | fingerprints: weapon of choice, complete machinery, This tutorial provides | if retained |  |
| tutorials/ml-ai/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/ml-ai/local-llm-deployments.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/ml-ai/mcp-mlflow-toolchain.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready; 1193 lines (catalog risk) | if retained |  |
| tutorials/ml-ai/mlflow-api-experiments.md | tutorials | REWRITE | fingerprints: Why This Matters, enterprise-grade; 1289 lines (catalog risk) | if retained |  |
| tutorials/ml-ai/onnx-browser-inference.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready; 1037 lines (catalog risk) | if retained |  |
| tutorials/ml-ai/rag-ollama-db.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready; 1326 lines (catalog risk) | if retained |  |
| tutorials/ml-ai/semantic-ml-training.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready; 1210 lines (catalog risk) | if retained |  |
| tutorials/python-development/advanced-nicegui-architecture.md | tutorials | REWRITE | fingerprints: Key Takeaways, production-ready; 1519 lines (catalog risk) | yes |  |
| tutorials/python-development/building-a-python-tui.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/python-development/chaos-engineering-k8s-python.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1111 lines (catalog risk) | if retained |  |
| tutorials/python-development/click-to-fastapi-conversion.md | tutorials | LIGHT EDIT | low template signal | if retained |  |
| tutorials/python-development/distributed-nicegui-redis.md | tutorials | LIGHT EDIT | 1621 lines (catalog risk) | yes |  |
| tutorials/python-development/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/python-development/nicegui-class-based-pages.md | tutorials | KEEP | 1126 lines (catalog risk) | yes |  |
| tutorials/python-development/psycopg2-to-psycopg3-migration.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/python-development/r-shiny-geoapp.md | tutorials | LIGHT EDIT | fingerprints: This tutorial provides, production-ready | if retained |  |
| tutorials/python-development/ruff-check-ignore-pyproject.md | tutorials | REWRITE | fingerprints: Why This Matters, enterprise-grade, production-ready | if retained |  |
| tutorials/python-development/websocket-chat-fastapi.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters; 1083 lines (catalog risk) | if retained |  |
| tutorials/quick-start/creating-mkdocs-github-site.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/quick-start/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/quick-start/monitoring-with-grafana-prometheus.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/rust-development/building-a-rust-tui.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/rust-development/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/rust-development/rust-csr-parquet-db.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/rust-development/rust-event-sourcing.md | tutorials | REWRITE | fingerprints: Why This Matters; 1228 lines (catalog risk) | if retained |  |
| tutorials/system-administration/awk-unix-text-processing.md | tutorials | LIGHT EDIT | fingerprints: Why This Matters | if retained |  |
| tutorials/system-administration/index.md | tutorials | KEEP | low template signal | if retained |  |
| tutorials/system-administration/ipxe-multi-boot.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
| tutorials/system-administration/prefect-fifo-redis.md | tutorials | LIGHT EDIT | 1025 lines (catalog risk) | if retained |  |
| tutorials/system-administration/remote-dev-tmux-screen.md | tutorials | REWRITE | fingerprints: Why This Matters, production-ready | if retained |  |
