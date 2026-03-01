---
tags:
  - deep-dive
  - data-engineering
  - performance
---

# Parquet vs CSV vs ORC vs Avro: Columnar Reality and the Economics of Data

*This is a Deep Dive — analytical and comparative. For operational guidance on working with Parquet, see the [Parquet Best Practices](../best-practices/database-data/parquet.md) page.*

**Themes:** Data Formats · Storage · Economics

---

## The Fundamental Question

Every data format encodes the same fact: that a set of values exists in some structured relationship to one another. The meaningful differences between formats are not aesthetic. They are physical — they determine which bytes must be read for a given query, how well the data compresses, whether a schema can evolve without rewriting the dataset, and whether a streaming consumer can process it incrementally. These physical properties have direct and substantial economic implications at scale.

The contemporary data format landscape is dominated by four formats with genuinely different design lineages and different primary constituencies: CSV, Avro, ORC, and Parquet. Understanding why each exists, and what assumptions underlie its design, is more useful than benchmarking numbers, which are sensitive to workload, hardware, and configuration in ways that make direct comparison treacherous.

---

## Storage Layout: The Row-Column Axis

The most fundamental property of a storage format is whether it organizes records by row or by column. This choice determines the relationship between the query's access pattern and the data's physical layout.

```
  Row-oriented storage (CSV, Avro row):
  ──────────────────────────────────────────────────────────────
  [id=1, name="Alice", age=30, salary=85000]
  [id=2, name="Bob",   age=24, salary=72000]
  [id=3, name="Carol", age=41, salary=95000]

  To compute avg(salary): must read all three complete rows.
  Bytes read includes id, name, age — irrelevant to the query.

  Columnar storage (Parquet, ORC):
  ──────────────────────────────────────────────────────────────
  [id: 1, 2, 3]
  [name: "Alice", "Bob", "Carol"]
  [age: 30, 24, 41]
  [salary: 85000, 72000, 95000]   ← only this column is read

  To compute avg(salary): read only the salary column.
  No wasted IO on irrelevant columns.
```

At the scale of a 100-column, 100-million-row dataset where an analytical query touches five columns, the difference in IO amplification between row and columnar storage is roughly 20x. This is not a theoretical advantage — it directly determines query latency and, in cloud environments, the cost of data egress and object storage reads.

---

## CSV: The Universal Lingua Franca

CSV (Comma-Separated Values) has no governing specification that is universally implemented. Its persistent dominance in data exchange is not a product of technical merit. It is a product of universality: every programming language, every database, every spreadsheet application, every data tool that has ever existed can read and write CSV without a library dependency.

This universality extracts a cost at every dimension of technical performance:

- **No schema**: column types are absent. A CSV file cannot reliably distinguish an integer from a string without external documentation or inference. Type inference is heuristic and brittle.
- **No compression awareness**: CSV is unaware of its own content structure and cannot apply column-specific compression strategies.
- **No predicate pushdown**: to filter rows matching a condition, every row must be read and parsed.
- **No schema evolution**: adding a column without breaking existing readers requires negotiated convention (new column appended last, readers tolerant of extra fields).
- **No encoding of nulls**: null representation varies by convention (`""`, `\N`, `NULL`, empty, `-`).

CSV's genuine advantages are the other side of the same properties: it is human-readable, trivially inspectable with any text editor, and requires no deserialization library. For data exchange between systems with different technical ecosystems — exporting from a financial system to a reporting tool, sharing a dataset with a domain expert who uses Excel — CSV remains correct precisely because it makes no assumptions about the receiver's tooling.

**CSV is not failing. It is being used for the wrong workloads.**

---

## Avro: Schema Evolution as Primary Design Goal

Apache Avro emerged from the Hadoop ecosystem in 2009, designed primarily for data serialization in distributed systems — specifically for Kafka and HDFS interchange. Its central innovation is not storage efficiency but schema evolution with backward and forward compatibility.

Avro stores its schema in the file header (or separately in a schema registry, as in the Confluent model) and encodes data as binary. The schema defines field names, types, and default values. Avro's schema resolution rules define precisely how a writer schema and a reader schema interact: which fields can be added with defaults, which can be removed while maintaining forward compatibility, which changes are breaking.

```
  Avro File Layout:
  ─────────────────────────────────────────────────────────
  [Magic bytes][Schema JSON][Sync marker]
  [Block: count, compressed bytes][Sync marker]
  [Block: count, compressed bytes][Sync marker]
  ...

  Schema travels with data (self-describing).
  Records are stored row by row within blocks.
```

The row-oriented layout makes Avro well-suited for streaming: a consumer can deserialize records one at a time without materializing the full dataset. Kafka's schema registry pattern — where each message carries a schema ID and the schema is retrieved from a central registry — extends this to high-throughput event streams.

Avro's weakness is analytical workloads. Reading a single field from an Avro file requires parsing every record in full. For a 100-field schema where an analysis touches three fields, the IO and CPU cost is approximately 33x what a columnar format would incur.

**Avro's design target is correct serialization over a wire or event bus. It was never intended to be a storage format for analytical queries.**

---

## ORC: Columnar Design for Hive

ORC (Optimized Row Columnar) was created at Hortonworks in 2013 for Hive workloads on HDFS. It is genuinely columnar and includes significant optimizations for analytical queries on Hadoop-era infrastructure:

- **Column-level statistics**: min, max, sum, and count stored in file metadata, enabling predicate pushdown at the file level without reading any row data
- **Lightweight indexing**: row group indices and bloom filters enabling fine-grained IO skipping within a file
- **ACID transaction support**: ORC is the underlying format for Hive ACID tables, supporting insert, update, and delete operations with transaction isolation
- **Stripe-based layout**: data is organized into stripes (default 250 MB), each containing column data for a subset of rows

```
  ORC File Layout:
  ─────────────────────────────────────────────────────────────
  [File Header]
  [Stripe 1: Index → Column Data → Stripe Footer]
  [Stripe 2: Index → Column Data → Stripe Footer]
  ...
  [File Footer: Column Statistics, Stripe Positions]
  [Postscript: Compression Codec, Footer Length]
```

ORC's primary constituency is the Hive/MapReduce/Spark-on-HDFS ecosystem. Its ACID support makes it the correct format for data warehouse tables that require mutation. Its bloom filters and column statistics provide meaningful query acceleration for high-selectivity filters on large tables.

ORC's weakness relative to Parquet is ecosystem breadth. While Spark, Hive, and Presto support ORC well, the Python and R analytical ecosystems, cloud-native tools like Athena and Redshift Spectrum, and the emerging lakehouse tools (Iceberg, Delta Lake) default to Parquet. ORC is technically competitive; it lost the ecosystem vote.

---

## Parquet: The Columnar Standard

Apache Parquet was designed at Twitter and Cloudera in 2013, influenced by the Dremel paper from Google. Its design choices reflect a specific set of priorities: columnar storage with nested schema support, aggressive compression through column-specific codecs, and broad ecosystem compatibility.

```
  Parquet File Layout:
  ─────────────────────────────────────────────────────────────────────
  [Magic: PAR1]
  [Row Group 1]
    [Column Chunk: col_A → Page headers + compressed pages]
    [Column Chunk: col_B → Page headers + compressed pages]
    ...
  [Row Group 2]
    ...
  [File Footer: Schema, Row Group Metadata, Column Statistics]
  [Footer Length][Magic: PAR1]
```

Parquet's key properties:

**Nested types via Dremel encoding**: Parquet can represent arbitrarily nested schemas — arrays of structs, maps, optional nested fields — using Dremel's repetition and definition levels. This allows JSON-like data structures to be stored in columnar format without flattening, which is the critical enabler for analytics on semi-structured data.

**Column-specific compression**: each column uses the compression codec best suited to its type and cardinality. Low-cardinality string columns (status codes, categories) use dictionary encoding; high-cardinality integers use delta encoding or PLAIN. Compression ratios for analytical datasets frequently reach 10:1 or better.

**Predicate pushdown**: row group statistics (min/max) and optional bloom filters enable the reader to skip entire row groups whose data cannot satisfy the query predicate. On a well-partitioned, sorted dataset, a highly selective filter may skip 99% of the file's physical bytes.

**Schema evolution**: Parquet supports adding and removing columns. Column identity is by name in most implementations, so column additions are backward compatible. Column removal or type changes are not.

**Parquet dominates the lakehouse ecosystem because it was designed for the same access patterns that lakehouse architectures serve**: large-scale analytical queries over immutable datasets, with partitioning and statistics enabling IO skipping at multiple levels.

---

## Compression Behavior and Economics

The compression story is not uniformly better in columnar formats. It is specifically better for analytical workloads with many columns and selective queries.

| Format | Typical compression (mixed data) | Column-specific codec | Dictionary encoding |
|---|---|---|---|
| CSV | 3–5x (gzip) | No | No |
| Avro | 2–4x (snappy/deflate) | No (per-block) | No |
| ORC | 8–15x | Yes | Yes |
| Parquet | 8–20x | Yes | Yes |

The compression advantage of columnar formats compounds with cardinality. A status column with three values ("active", "inactive", "pending") stored in a columnar format with dictionary encoding represents every value as a single-byte integer. The same column in CSV uses 6–10 bytes per value. At one billion rows, the difference is roughly 6–9 GB vs 60–80 MB.

The practical economic implication in cloud-native deployments is substantial. Object storage (S3, GCS, Azure Blob) charges per byte stored and per byte read. Query engines (Athena, BigQuery) charge per byte scanned. A 10x compression ratio and a 5x IO reduction (from predicate pushdown and column pruning) represent a potential 50x reduction in query cost on a cloud-native analytical workload.

---

## Schema Evolution Comparison

Schema evolution — the ability to modify a dataset's schema over time while maintaining compatibility with existing readers and writers — is a first-class operational concern for any data system that runs for more than a few months.

| Format | Add column | Remove column | Rename column | Change type |
|---|---|---|---|---|
| CSV | Convention only | Convention only | Breaking | Breaking |
| Avro | Supported (with default) | Supported | Breaking | Conditional |
| ORC | Supported | Supported | Breaking | Limited |
| Parquet | Supported | Supported (by convention) | Breaking | Limited |

CSV has no schema evolution story. Schema changes in CSV are managed entirely by convention — a decision between the writer and every reader — with no enforcement mechanism.

Avro's schema evolution is the most formal: the Avro specification defines precise compatibility rules, and the Confluent Schema Registry enforces backward, forward, or full compatibility at schema registration time. For event-driven systems where schema changes are frequent and producers and consumers evolve independently, this is a meaningful advantage.

Parquet and ORC support column additions through metadata. Existing files written without a new column simply read null or a configured default for that column when read through a schema that includes it. Column renames and type changes typically require dataset migration.

---

## Streaming vs Batch Implications

The streaming/batch distinction maps almost directly onto the row/columnar axis.

Streaming systems — Kafka, Kinesis, Flink, Spark Streaming — process records incrementally. Each event must be deserialized and processed before the next arrives. Row-oriented formats (Avro, Protobuf) are natural here: deserialization of a single record is fast and allocation is bounded.

Columnar formats are fundamentally batch-oriented. A Parquet file cannot be usefully read until the file footer is written — the footer contains the column statistics and offsets that make the columnar layout navigable. Reading partial Parquet files, or using Parquet for message-by-message streaming, requires either buffering (introducing latency) or accepting the absence of statistics (losing IO pushdown benefits).

The practical implication for data platform design: landing zones for streaming data should use Avro or Protobuf. Compaction jobs convert this data to Parquet for analytical consumption. This is the standard Lambda and Kappa architecture pattern — the format boundary corresponds to the streaming/batch boundary.

---

## Cloud-Native Implications

Object storage (S3, GCS, Azure Blob) is optimized for sequential reads of large objects, not random access. Columnar formats with large row groups (128 MB is the Parquet default) align well with this: a column chunk is a contiguous byte range, readable in a single HTTP range request.

Cloud-Optimized GeoTIFF (COG) demonstrates the HTTP range request pattern applied to raster data — the same principle applies to Parquet on object storage: a well-designed Parquet file allows a query engine to read only the column chunks and row groups it needs with minimal round trips. This is why Parquet, not ORC or Avro, is the native format for Athena, Redshift Spectrum, BigQuery external tables, and Snowflake external stages.

The emergence of table formats (Delta Lake, Apache Iceberg, Apache Hudi) layering transaction semantics on top of Parquet files on object storage is the logical extension: Parquet provides the columnar IO model; the table format provides ACID transactions, schema evolution enforcement, time travel, and partition evolution. ORC's ACID support (within Hive) becomes less distinctive when the table format layer provides equivalent semantics for Parquet.

---

## When Each Format Is the Right Answer

| Workload | Recommended format | Reasoning |
|---|---|---|
| Data exchange with heterogeneous consumers | CSV | Universality > performance |
| Kafka event streaming | Avro | Schema evolution + row-at-a-time deserialization |
| Hive/MapReduce ACID warehouse | ORC | Native Hive ACID support |
| Analytical queries on S3/GCS | Parquet | Columnar IO + cloud-native ecosystem |
| Lakehouse (Delta/Iceberg) | Parquet | Mandatory — table formats require it |
| ML feature stores | Parquet | Columnar reads for feature vectors |
| Protobuf/gRPC services | Protobuf | Not a storage format; wire protocol only |
| Inter-service configuration | JSON/YAML | Human readability; low volume |

---

## Decision Framework

**Start with the query pattern, not the format.** If queries access few columns from wide tables with many rows, columnar formats yield 5–20x improvements in IO and cost. If every query reads most columns from every row (e.g., a row-per-transaction system where each transaction is audited in full), the columnar advantage diminishes.

**Start with the streaming or batch boundary.** If data arrives as a stream and must be processable before the batch window closes, the landing format must be row-oriented. The compaction to columnar is a separate, scheduled operation.

**Start with the ecosystem.** If your toolchain is Spark + Hive + HDFS, ORC is deeply integrated and technically competitive. If your toolchain is any cloud-native analytical service plus Python, Parquet is the default and deviating from it requires justification.

**CSV is always available as a fallback.** For ad-hoc data exchange, experimentation, and one-time transfers, the operational cost of format conversion is rarely justified. The question is not whether CSV is inferior to Parquet — it is — but whether the workload justifies the engineering investment of moving to a binary format.

!!! tip "See also"
    - [Geospatial File Format Choices](geospatial-file-format-choices.md) — how columnar principles extend to raster and vector geospatial data
    - [Lakehouse vs Warehouse vs Database](lakehouse-vs-warehouse-vs-database.md) — the architectural context in which Parquet dominates
    - [The End of the Data Warehouse?](the-end-of-the-data-warehouse.md) — open table formats (Delta, Iceberg, Hudi) that build on Parquet and reshape the warehouse landscape
    - [Parquet Best Practices](../best-practices/database-data/parquet.md) — operational guidance for working with Parquet in production
