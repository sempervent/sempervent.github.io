---
tags:
  - deep-dive
  - infrastructure
  - storage
  - data-engineering
  - performance
---

# The Physics of Storage Systems: Latency, Throughput, and the Material Limits of Data

*See also: [Parquet vs CSV, ORC, and Avro](parquet-vs-csv-orc-avro.md) — how columnar file formats exploit storage physics to minimize IO amplification, and [The End of the Data Warehouse?](the-end-of-the-data-warehouse.md) — how cloud object storage economics are reshaping analytical system architecture.*

**Themes:** Storage · Infrastructure · Economics

---

## Opening Thesis

Storage architecture is constrained by physics long before it is constrained by software. The choice of storage medium, the access pattern of the workload, and the distance between compute and data are physical facts that no abstraction layer eliminates and no software optimization overcomes. A columnar query engine processing Parquet files on remote object storage is not slower than the same engine on local NVMe because of software inefficiency — it is slower because light travels at a finite speed, disk seek times are governed by rotational mechanics, and NAND flash has write amplification properties that are intrinsic to its architecture. Understanding storage systems requires understanding these physical constraints before evaluating the software abstractions that operate within them.

---

## Historical Context

### HDD and the Rotational Limit

Hard disk drives (HDDs) store data on magnetic platters that spin at 5,400–15,000 RPM. Reading or writing data requires positioning the read/write head over the correct track (seek) and waiting for the correct sector to rotate under the head (rotational latency). Typical seek time: 3–10ms. Typical rotational latency at 7,200 RPM: 4.2ms average. Combined average access latency: 7–14ms.

For sequential access — reading data from a contiguous region of disk — HDD performance is adequate: sustained throughput of 80–200 MB/s is typical. For random access — reading data scattered across many non-contiguous locations — HDD performance collapses. Each random read requires a seek plus rotational latency, producing effective random IOPS of 50–200 operations per second. This physical constraint made row-oriented database access patterns (read an entire row to answer a query) architecturally rational for HDDs: a full row is stored contiguously, a single seek retrieves the entire record.

The relational database B-tree index was designed for this constraint: rather than scanning a full table sequentially (expensive in time, cheap in seek operations), an index narrows the search to a small number of pages, each requiring a seek. The HDD seek time was the organizing principle of database index design.

### SSD and the NAND Architecture

NAND flash SSDs eliminate rotational mechanics. There is no seek time — data at any location is addressable with equal access time, bounded by the flash controller's response time (0.1–1ms for random reads, lower for sequential). Sustained sequential throughput of SATA SSDs: 500–600 MB/s. NVMe SSDs on PCIe 4.0: 3,000–7,000 MB/s sequential, with random IOPS of 500,000–1,000,000.

NAND flash has physical properties that create write amplification: flash cells can only be written once per erase cycle, and erase cycles operate on blocks larger than write units (pages). Writing a small amount of data to an SSD requires reading the entire block, erasing it, modifying the relevant pages, and rewriting the block. This write amplification (the ratio of data written to the SSD to data logically written by the application) ranges from 1.5x to 10x depending on the write pattern and the SSD's wear-leveling algorithm. Write amplification is invisible to the application but degrades SSD endurance over time and throttles write throughput under sustained write-intensive workloads.

### NVMe and the PCIe Interface

SATA SSDs, despite their NAND flash storage medium, were initially constrained by the SATA interface (600 MB/s maximum). NVMe (Non-Volatile Memory Express) is a protocol designed specifically for flash storage, using PCIe lanes directly. NVMe eliminates the SATA interface bottleneck and reduces command overhead (NVMe has 2 command sets vs SATA's 2,048), enabling the full performance of NAND flash to reach the CPU.

NVMe drives on PCIe 4.0 x4: 6,500–7,000 MB/s sequential read, 600,000–1,000,000 IOPS random read with 4KB blocks. These figures represent a 10x improvement over SATA SSDs and a 50x improvement over HDDs for random workloads. The storage performance improvement has made compute — not IO — the bottleneck for many workloads that were previously IO-bound, fundamentally changing the optimization target for query engine design.

### Network-Attached Storage and Object Storage

Network-attached storage (NAS, SAN) introduced the latency of a network between compute and storage. Modern NVMe-oF (NVMe over Fabrics) on InfiniBand achieves single-digit microsecond latency, approaching local NVMe. iSCSI over 10GbE adds 0.1–1ms of network latency. These forms of networked block storage present the storage as if it were a local device — the filesystem abstraction is preserved, and the network latency is hidden but not eliminated.

Cloud object storage (S3, GCS, Azure Blob) is fundamentally different: it is not a filesystem and it is not block storage. It is a key-value store accessed via HTTP. Object storage latency for a metadata operation (checking existence, listing objects) is 10–100ms. First-byte latency for reads: 20–200ms. Sustained sequential throughput per connection: 100–250 MB/s, but dramatically higher with parallel connections. Object storage does not support random writes — objects must be written atomically in a single PUT operation. These characteristics make object storage poorly suited for workloads requiring low-latency random access and well-suited for large sequential read workloads where the HTTP overhead amortizes over large reads.

---

## Latency Layers

```
  Storage latency hierarchy (approximate, per-operation):
  ──────────────────────────────────────────────────────────────────────
  CPU L1 cache          0.5  ns   │████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  CPU L2 cache          7    ns   │█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  CPU L3 cache          40   ns   │██████░░░░░░░░░░░░░░░░░░░░░░░░░░░
  RAM (DRAM)            100  ns   │███████░░░░░░░░░░░░░░░░░░░░░░░░░░
  NVMe SSD (local)      100  μs   │████████░░░░░░░░░░░░░░░░░░░░░░░░░
  SATA SSD (local)      500  μs   │█████████░░░░░░░░░░░░░░░░░░░░░░░░
  Network (LAN, 1ms)    1    ms   │██████████░░░░░░░░░░░░░░░░░░░░░░░
  HDD (random)          10   ms   │███████████░░░░░░░░░░░░░░░░░░░░░░
  Object storage        50   ms   │████████████░░░░░░░░░░░░░░░░░░░░░
  Object storage (cold) 5+   s    │████████████████████████████████░

  Each level: ~10–100x slower than the level above.
  A cache miss that reaches object storage: 5–6 orders of magnitude
  slower than a cache hit. Software cannot close this gap.
```

**IO amplification**: for column-oriented file formats stored on disk, a query that reads one column from a multi-column table reads only the bytes corresponding to that column — IO is proportional to the data actually needed. For row-oriented formats, a query that reads one column must read the entire row, even for columns not needed by the query — IO is proportional to total row width. At 100GB table sizes, the IO amplification of row-oriented formats versus columnar formats for selective column reads is commonly 10x–100x.

**The implication for query design**: the storage medium determines which access patterns are tolerable. On HDD, sequential scans are acceptable; random seeks are prohibitive. On NVMe SSD, random reads are tolerable; sequential scans are fast. On object storage, parallel range requests over large objects are the optimal pattern; many small random reads produce per-request overhead that dominates over data transfer time.

---

## Throughput vs IOPS

**Sequential access throughput** is the bottleneck for analytical workloads: full table scans, large batch exports, columnar file reads. Sequential throughput determines how quickly bulk data can be processed. NVMe SSDs have high sequential throughput (6+ GB/s) but this throughput is shared across all workloads on the machine. Object storage has effectively unlimited aggregate throughput but per-connection sequential throughput is bounded by network bandwidth.

**IOPS (Input/Output Operations Per Second)** is the bottleneck for transactional workloads: database index lookups, key-value reads, record updates. IOPS determines how many discrete storage operations can be completed per second. NVMe SSDs provide 1M+ IOPS, enabling highly concurrent transactional access. Object storage IOPS are bounded by request latency — at 50ms per operation, maximum single-thread IOPS is 20; with 1000 parallel connections, 20,000 IOPS.

**The columnar vs row storage interaction with throughput and IOPS**: columnar storage (Parquet, ORC) is designed for high-throughput sequential access — read column stripes in parallel, decompress them, apply predicates. Row storage (PostgreSQL heap, CSV) is designed for IOPS-bound transactional access — look up specific records by key, read entire rows, update them. The storage medium and the access pattern are coupled: deploying columnar analytics queries against row-oriented storage on IOPS-constrained media, or deploying OLTP queries against columnar storage, both produce poor performance for physical reasons rather than software limitations.

**Caching layers**: caching addresses the latency gap between storage tiers. OS page cache retains recently accessed disk pages in RAM, making repeated reads of the same data free (RAM latency). Database buffer pools (PostgreSQL shared_buffers, MySQL innodb_buffer_pool) provide application-controlled caching of database pages. Distributed caches (Redis, Memcached) extend caching across nodes for horizontally scaled applications. The effectiveness of each caching layer is governed by the workload's locality: workloads with high temporal locality (the same data is accessed repeatedly in a short window) benefit enormously from caching; workloads with low locality (analytics queries that scan different data every time) benefit minimally.

---

## Cloud Object Storage Reality

Cloud object storage is the dominant storage substrate for modern data systems because its economics are compelling: $0.023/GB/month (S3 Standard) with effectively unlimited scale, no hardware management, and 11 nines of durability. Its performance characteristics are distinctive and frequently misunderstood.

**Eventual consistency**: object storage provides strong read-after-write consistency for individual objects (AWS S3 achieved this in 2020; GCS has had it since inception), but bucket-level operations (listing objects after a write, reading metadata after an update) may observe transient inconsistency windows in some configurations. Multi-part uploads, concurrent writes, and cross-region replication introduce consistency windows that require explicit handling in data engineering workflows.

**Range requests**: the HTTP GET Range header allows clients to request specific byte ranges of an object without downloading the entire object. Range requests are the mechanism by which columnar file formats achieve selective IO on object storage: a Parquet reader sends a range request for the column chunk corresponding to the needed column rather than downloading the entire file. The effectiveness of range requests depends on file layout: Parquet files with large row groups minimize the number of range requests needed for a scan; many small files require many requests each with significant per-request overhead.

**Cold vs hot tiers**: object storage tiering (S3 Standard → S3 Infrequent Access → S3 Glacier → S3 Glacier Deep Archive) trades access cost for storage cost. S3 Standard at $0.023/GB/month has millisecond-latency access. S3 Glacier Instant Retrieval at $0.004/GB/month has millisecond-latency access. S3 Glacier Flexible Retrieval at $0.0036/GB/month has hours-latency retrieval. Data that is accessed infrequently but must be queryable benefits from Instant Retrieval; data that is archived for compliance with rare access benefits from Deep Archive. The economic optimization of storage tiering is non-trivial for large data estates and requires lifecycle policies calibrated to actual access patterns.

---

## Data Format Interaction

The interaction between file format and storage medium is the primary lever for storage performance optimization in analytical systems.

**Parquet scan efficiency**: Parquet's columnar layout with row group metadata (min/max statistics per column per row group) enables two levels of predicate pushdown: file-level metadata filtering (skip entire files that cannot contain matching rows), and row-group-level filtering (skip row groups within a file based on statistics). For a query `WHERE event_date = '2024-01-15'` on a date-partitioned Parquet table, the query engine reads only the file for January 15 and, within that file, only the row groups whose date range includes that date. The IO reduction is proportional to the selectivity of the predicate — a highly selective predicate on a well-organized Parquet dataset reduces IO by 99% relative to a full scan.

**Cloud-Optimized GeoTIFF HTTP range requests**: COG files store internal tile indexes and tile data such that a client can request the index via a range request, determine which tiles contain the area of interest, and request only those tiles via additional range requests. Three HTTP requests are sufficient to retrieve a spatial subset of a multi-gigabyte COG without downloading the full file. This format-level optimization enables spatial analytics directly against object storage without data movement.

**Index locality**: database indexes (B-trees, hash indexes, BRIN) concentrate the data required for point lookups into a small number of pages, reducing the IO cost of lookup from a full table scan to a logarithmic number of page reads. Index effectiveness depends on the storage medium: on HDD, each page read is a seek-plus-rotation (10ms), so indexes with deep trees (many levels, many seeks) perform poorly; on NVMe SSD, each page read is 100μs, making deeper trees acceptable.

*See also: [Parquet vs CSV, ORC, and Avro](parquet-vs-csv-orc-avro.md) — the file format properties that determine IO efficiency on each storage tier.*

---

## Economic Implications

**Storage tiering**: the economic case for cold storage is straightforward for data that is not accessed frequently but must be retained. A petabyte of data in S3 Glacier Deep Archive costs $1/TB/month vs $23/TB/month in S3 Standard — a 23x cost reduction. The retrieval cost (time and per-GB fee) must be weighed against the access frequency. For regulatory archive data accessed once per year for audit purposes, Glacier Deep Archive is economically dominant. For analytical data accessed weekly by data scientists, S3 Standard or S3 Intelligent-Tiering is appropriate.

**Egress costs**: cloud providers charge for data transferred out of their network ($0.09/GB for AWS), but not for data transferred within a region to cloud services. An analytical architecture where compute (Spark cluster, Athena query) runs in the same region as storage (S3) incurs no egress charges for reads. Data exported to on-premises systems, to other cloud providers, or to another AWS region incurs egress fees that can dominate the storage cost for large datasets. Multi-cloud or hybrid architectures must account for egress cost as a structural expense, not an edge case.

**Lifecycle policies**: object storage lifecycle policies automatically transition objects between storage tiers based on age and access patterns. A lifecycle policy that transitions objects from S3 Standard to S3 Infrequent Access after 30 days, and to S3 Glacier Instant Retrieval after 90 days, reduces storage cost for historical data without manual intervention. The design of lifecycle policies requires understanding the actual access patterns of the data — a policy optimized for expected access patterns that do not materialize wastes retrieval fees on cold-tiered data that analysts actually need frequently.

---

## Decision Framework

| Workload Type | Optimal Storage Tier | Format Recommendation |
|---|---|---|
| OLTP (transactions, point lookups) | Local NVMe SSD | Row-oriented (PostgreSQL heap, RocksDB) |
| OLAP (full-table analytical scans) | Local NVMe or S3 | Columnar Parquet / ORC |
| ML training data (large batch reads) | Local NVMe or EBS | Columnar or binary (TFRecord, Parquet) |
| Spatial analytics (large rasters) | S3 (COG) | Cloud-Optimized GeoTIFF |
| Archival / compliance | S3 Glacier | Parquet (compressed) or Avro (schema evolution) |
| Interactive BI dashboards | Local cache + SSD | Columnar with aggressive statistics |
| Real-time event processing | In-memory (Kafka) | Binary (Avro, Protobuf) |

**When to optimize for latency**: workloads with human interaction in the response path (interactive dashboards, API responses, real-time scoring) require sub-100ms storage operations. These workloads require local SSD or in-memory caching; object storage cannot satisfy low-latency requirements without a caching layer.

**When to optimize for throughput**: batch processing, large-scale analytics, and ML training are throughput-constrained. Maximizing throughput means using columnar formats with large row groups, parallelizing IO across many connections, and minimizing per-operation overhead by reading large blocks.

**When to accept cold storage**: data accessed less than once per month for non-latency-sensitive use cases (compliance audit, historical analysis, disaster recovery) is appropriately stored in cold tiers. The economic savings are significant; the retrieval latency is the only cost.

**When to introduce caching layers**: when the same data is read repeatedly by different queries or users, caching eliminates redundant storage IO. Database buffer pools and Redis caches are appropriate for frequently-accessed hot data. For analytical workloads, columnar in-memory engines (DuckDB, Polars) that load and cache Parquet data for the duration of an analysis session reduce repeated object storage requests.

!!! tip "See also"
    - [Parquet vs CSV, ORC, and Avro](parquet-vs-csv-orc-avro.md) — columnar file format design and its interaction with storage media
    - [The End of the Data Warehouse?](the-end-of-the-data-warehouse.md) — object storage economics and how they reshape analytical architecture
    - [DuckDB vs PostgreSQL vs Spark](duckdb-vs-postgres-vs-spark.md) — storage access patterns for single-node vs distributed analytical engines
    - [Geospatial File Format Choices](geospatial-file-format-choices.md) — COG range request mechanics and spatial storage access patterns
