---
tags:
  - deep-dive
  - geospatial
  - data-engineering
  - spatial-systems
  - architecture
---

# The Operational Geometry of Spatial Systems: Hex Grids, File Formats, and the Shape of Computation

*See also: [Geospatial File Format Choices](geospatial-file-format-choices.md) — the file format layer on which spatial analytical systems depend, and [Polars vs Pandas for Geospatial Data](polars-vs-pandas-geospatial.md) — the dataframe computation layer for spatial analytics.*

**Themes:** Spatial · Architecture · Data Formats

---

## Opening Thesis

Spatial systems are shaped by geometry before they are shaped by code. The choice of spatial index determines which queries are fast and which are slow. The choice of file format determines whether a spatial query touches 10MB or 10GB of data. The choice of partitioning strategy determines whether a system scales to billions of geometries or degrades at millions. These are not implementation details to be addressed after the architecture is designed — they are the architecture. An engineer who selects PostGIS without understanding the interaction between geometry complexity and B-tree index effectiveness, or who adopts GeoParquet without understanding row group spatial clustering, will build a system whose performance characteristics are determined by accident rather than design. The geometry of the data is the geometry of the computation.

---

## Spatial Indexing Models

The fundamental problem in spatial indexing is answering containment and proximity queries efficiently: "which geometries intersect this bounding box?", "which points are within 1km of this coordinate?", "which polygons contain this point?" The challenge is that spatial relationships are two-dimensional — a one-dimensional sorted index (B-tree) cannot efficiently express spatial proximity because proximity in two dimensions is not preserved by any one-dimensional ordering.

```
  Spatial indexing approaches compared:
  ──────────────────────────────────────────────────────────────────
  Quadtree (recursive 2D partitioning):
  ┌───────────┬───────────┐
  │  NW quad  │  NE quad  │
  │  ┌──┬──┐  │           │
  │  │  │  │  │   (empty) │
  │  ├──┼──┤  │           │
  │  │  │  │  │           │
  │  └──┴──┘  │           │
  ├───────────┼───────────┤
  │  SW quad  │  SE quad  │
  │           │           │
  └───────────┴───────────┘
  Strength: hierarchical, natural for zoom levels
  Weakness: poor for non-uniform distributions (empty quads waste space)

  R-tree (minimum bounding rectangle grouping):
  ┌─────────────────────────────────────────────┐
  │  MBR 1 (contains 5 geometries)              │
  │  ┌─────────┐   ┌─────────────┐             │
  │  │ MBR 1.1 │   │  MBR 1.2   │             │
  │  │ [●][●]  │   │ [△][□][●]  │             │
  │  └─────────┘   └─────────────┘             │
  └─────────────────────────────────────────────┘
  Strength: adapts to data distribution, efficient range queries
  Weakness: insertion/deletion complexity, overlap in dense datasets

  H3 hex grid (discrete global tessellation):
  Each cell has a unique 64-bit integer ID at each resolution
  ⬡ ⬡ ⬡ ⬡ ⬡        Resolution 5: ~252 km² cells
  ⬡ ⬡ ⬡ ⬡ ⬡        Resolution 9: ~0.1 km² cells
  ⬡ ⬡ ⬡ ⬡ ⬡        Resolution 12: ~0.3 m² cells

  Strength: uniform cell area, O(1) neighbor lookup, easily joinable
  Weakness: approximates geometry, not exact for boundary cases
```

### Quadtrees

The quadtree recursively subdivides space into four equal quadrants until each quadrant contains at most a defined number of geometries. Quadtrees are intuitive and map naturally to tile-based web maps: zoom level 0 is the entire world, zoom level N is a 2^N × 2^N grid of tiles. Their critical weakness is sensitivity to data distribution: if all geometries are concentrated in one geographic region (urban datasets, regional sensors), the quadtree develops deep subtrees for the dense region and shallow subtrees for empty regions, producing unbalanced performance.

### R-Trees

The R-tree groups geometries by their minimum bounding rectangles (MBRs), creating a tree of nested bounding boxes. A query traverses the tree, pruning subtrees whose MBR does not intersect the query region. R-trees are the dominant spatial index in relational databases (PostGIS GiST indexes, SQLite Spatialite). Their performance degrades when MBRs overlap significantly — a common occurrence in dense urban datasets with complex polygons — because queries cannot prune overlapping subtrees without examining both.

PostGIS uses GIST-based R-trees with Hilbert curve ordering for spatial joins. The Hilbert curve is a space-filling curve that preserves spatial locality better than naive row-major ordering, reducing the number of index node reads for range queries.

### H3 Hex Grids

Uber's H3 is a discrete global grid system that tessellates the Earth into hexagonal cells at 16 resolutions (resolution 0: 122 cells covering the globe; resolution 15: ~0.9 m² cells). Each cell has a unique 64-bit integer identifier. H3's key properties for computational systems:

**Uniform area**: H3 cells at a given resolution have approximately equal area (hexagons have less area variance than square grids at equivalent resolution). This property makes H3-aggregated statistics comparable across cells without area normalization.

**O(1) neighbor lookup**: finding the 6 immediate neighbors of an H3 cell is a bitwise operation on the cell ID — no index traversal required. Ring queries (all cells within k hops) execute in O(k²) time with simple arithmetic.

**Integer joinability**: H3 cell IDs are 64-bit integers. Joining datasets on H3 cells is a standard integer join — fast in columnar databases (DuckDB, BigQuery, Snowflake), fast in dataframes (Polars, Pandas), and indexable with standard B-tree indexes. This joinability is the primary operational advantage of H3 over traditional spatial indexes for analytics workloads.

**Resolution hierarchy**: H3 cells have a parent-child relationship: each resolution-N cell is the parent of 7 resolution-(N+1) cells. Multi-resolution aggregation (computing statistics at different spatial granularities) is a tree traversal over this hierarchy.

The limitation of H3 is approximation: geometries must be converted to H3 cell coverage (the set of cells that cover the geometry, at a chosen resolution). The conversion introduces approximation error proportional to the ratio of geometry size to cell size. Boundary effects — geometries that straddle cell boundaries — require including boundary cells, which over-approximates the covered area.

---

## File Format Interaction

The spatial query performance of file-based analytical systems is determined as much by file format choice as by query engine choice.

**COG vs GeoParquet**: Cloud-Optimized GeoTIFF and GeoParquet address different data modalities. COG is for raster data — gridded arrays of values (elevation, satellite imagery, land cover, temperature). GeoParquet is for vector data — geometries (points, lines, polygons) with attribute columns. A system that processes both raster and vector data requires both formats: COG for the raster analysis layer, GeoParquet for the vector feature layer.

Within the vector domain, the distinction between GeoParquet and traditional PostGIS database tables is the compute-storage separation: GeoParquet files on object storage are readable by any query engine that supports GeoParquet (DuckDB with spatial extension, Apache Sedona, GeoPandas), while PostGIS tables are accessible only through the PostgreSQL connection protocol. For analytical workloads that do not require the transactional consistency of a database, GeoParquet provides comparable query performance for full scans (with DuckDB) and dramatically better scalability for concurrent readers.

**Database vs file-based spatial queries**: PostGIS excels at transactional workloads requiring point-in-polygon queries on arbitrary geometries with precise topology (using GEOS algorithms): "is this GPS coordinate inside this exact county boundary?" For this query type, the PostGIS GIST index is more efficient than GeoParquet because it operates on stored, indexed geometries rather than requiring a scan of Parquet row groups.

GeoParquet excels at aggregation workloads on large feature sets: "what is the average elevation of all points in each H3 cell?" This query benefits from columnar storage (read only the geometry and elevation columns), spatial partitioning (read only the row groups covering the target region), and DuckDB's vectorized execution engine. A DuckDB query over GeoParquet can process hundreds of millions of point geometries faster than PostGIS for this access pattern.

---

## Partitioning and Scale

Spatial partitioning — organizing data such that spatially proximate records are stored together — is the primary mechanism for limiting IO in spatial analytical queries.

**Spatial bucketing**: H3 partitioning divides a dataset by H3 cell ID at a chosen resolution. Records assigned to H3 cell X are stored in the partition for cell X. A query that filters by H3 cell reads only the relevant partition; a query that filters by a bounding box reads only the partitions that overlap the bounding box. The efficiency of H3 partitioning depends on resolution choice: too coarse (large cells) produces partitions that are large and IO-expensive; too fine (small cells) produces many partitions with high overhead per partition.

**Sharding by region**: for globally distributed datasets, geographic sharding (North America, Europe, Asia-Pacific) provides a coarse partitioning that limits cross-shard query fan-out for region-specific analytical workloads. Region-level sharding is a practical partitioning strategy when the data has natural geographic affinity (regional business operations, country-specific regulatory domains) and queries primarily target single regions.

**Multi-resolution strategies**: some spatial analytical systems maintain data at multiple H3 resolutions simultaneously — detailed data at resolution 10 (fine-grained, for local analysis), aggregated data at resolution 6 (coarse, for regional analysis), and pre-aggregated national statistics at resolution 2. This materialized multi-resolution pyramid trades storage cost for query latency: regional queries hit the resolution-6 table rather than requiring aggregation over the resolution-10 table. The trade-off is the same as in any analytical pre-aggregation strategy: storage cost and update lag increase, query latency decreases.

---

## Routing and Graph Systems

Spatial routing systems — computing optimal paths between locations through a network — are a specialized class of spatial systems with distinct architectural requirements.

**Network graphs vs raster cost surfaces**: routing can be modeled as a graph problem (road network: nodes are intersections, edges are road segments with travel time weights) or as a raster problem (cross-country navigation: a grid of cells where each cell has a traversal cost). Graph-based routing (Valhalla, OSRM, GraphHopper) is optimal for road and transit networks where the legal travel paths are defined by a discrete network. Raster cost surface routing is optimal for off-road movement (hiking, off-road vehicles, animal movement ecology) where the travel cost is a continuous function of terrain.

Graph-based routing systems precompute routing structures (Contraction Hierarchies, OSRM's preprocessing pipeline, Valhalla tile extracts) that enable sub-second point-to-point routing on continental road networks. The precomputation transforms O(N²) routing problems into O(log N) queries using graph decomposition. The trade-off is update cost: road network changes require recomputing the hierarchy, which takes hours for continental datasets. Dynamic routing (incorporating real-time traffic, live road closures) requires incremental hierarchy updates or dynamic overlay routing on top of the static precomputed structure.

**Precomputation vs dynamic computation**: for static spatial analysis (finding the nearest facility, computing catchment areas), precomputation produces query results in milliseconds at the cost of storage and update latency. For dynamic queries (routing around a live incident, computing accessibility for a changing transit schedule), dynamic computation is required. The boundary between "precompute" and "compute dynamically" is determined by the frequency of updates to the underlying network and the latency budget of the query.

---

## Storage and Compute Trade-Offs

**Columnar geometry storage**: GeoParquet stores geometries as WKB (Well-Known Binary) in a byte-array column. WKB geometries are variable-length — a point is 21 bytes, a complex polygon with thousands of vertices is kilobytes. Columnar storage groups WKB values together, enabling efficient column-level compression (LZ4, Zstandard) and enabling query engines to read geometry columns independently of attribute columns. For queries that filter on non-spatial attributes (selecting points by category or time range) and then apply spatial predicates, columnar storage allows the non-spatial filter to be applied before loading geometries, reducing the volume of WKB parsing required.

**Bounding box pruning**: GeoParquet's geometry column statistics (minimum bounding boxes at the row group level) enable the same predicate pushdown for spatial queries that Parquet column statistics enable for value-range queries. A spatial query with a bounding box filter reads only the row groups whose recorded bounding box intersects the query bounding box. For spatially sorted datasets (data organized by H3 cell ID or Hilbert curve order), bounding box pruning eliminates most row groups for regional queries, producing IO proportional to the result size rather than the table size.

**IO minimization**: the practical techniques for minimizing IO in spatial analytical systems converge on two principles: (1) organize data such that spatially proximate records are stored together (spatial sorting, H3 partitioning), and (2) use file formats whose metadata structures support spatial predicate pushdown (GeoParquet bounding box statistics, COG tile indexes). Columnar format + spatial sort + bounding box statistics produces a system where query IO scales with query result size, not table size — the performance characteristic of an indexed database, achieved with open file formats on object storage.

---

## Decision Framework

| Workload | Recommended Approach |
|---|---|
| Transactional point-in-polygon queries (< 1M features) | PostGIS with GIST R-tree index |
| Analytical aggregation on large vector datasets (> 10M features) | GeoParquet + DuckDB spatial extension |
| Raster analytics (imagery, elevation, environmental grids) | COG on S3 + range request clients (rasterio, rio-cogeo) |
| Uniform spatial aggregation (heatmaps, density analysis) | H3 grid at appropriate resolution + columnar store |
| Road network routing | Valhalla (flexible tiles) or OSRM (fastest preprocessing) |
| Off-road / terrain routing | Raster cost surface + Dijkstra on grid |
| Multi-scale spatial analysis | H3 resolution pyramid with pre-aggregated tiers |
| Spatial joins across large datasets | DuckDB spatial join on GeoParquet or BigQuery / Snowflake spatial functions |

**When to use hex grids**: H3 is appropriate when the analysis requires: uniform spatial aggregation independent of administrative boundaries, joinability with non-spatial data using integer keys, multi-resolution hierarchical analysis, or spatial indexing in columnar databases. H3 is not appropriate when exact geometric containment is required (polygon boundaries matter), when the geometries are too small relative to the cell size to be accurately represented, or when the data is inherently raster.

**When to use database indexing**: PostGIS R-tree indexing is appropriate for transactional spatial queries requiring exact topology (legal parcel boundaries, precise address geocoding, utility network connectivity), point-in-polygon queries on complex polygon boundaries at low to moderate volume, and spatial updates requiring transactional consistency.

**When to use raster precomputation**: precomputed raster cost surfaces and raster tile indexes are appropriate when the spatial function (elevation, slope, land cover, demographic density) is continuous over space, when the resolution is determined by the data source rather than by query granularity, and when the read-to-write ratio is high (the raster is computed once and queried many times).

**When to use file-based analytics**: GeoParquet + DuckDB is appropriate for analytical workloads requiring: ad hoc spatial queries over large feature sets, integration with non-spatial analytical pipelines (Parquet-based data warehouses, Polars dataframes), or multi-engine access to the same spatial data (different teams using different query engines over the same stored files).

!!! tip "See also"
    - [Geospatial File Format Choices](geospatial-file-format-choices.md) — COG, GeoParquet, and WKB/WKT format mechanics and trade-offs
    - [Polars vs Pandas for Geospatial Data](polars-vs-pandas-geospatial.md) — the dataframe layer for spatial analytics and its ecosystem maturity
    - [The Physics of Storage Systems](the-physics-of-storage-systems.md) — the IO latency and throughput constraints that govern spatial query performance
    - [DuckDB vs PostgreSQL vs Spark](duckdb-vs-postgres-vs-spark.md) — the analytical query engine choices for spatial workloads
