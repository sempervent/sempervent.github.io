# Geospatial System Architecture

**Objective**: Establish foundational patterns for designing geospatial data systems: primitives, indexing, pipelines, storage, and performance.

## Core geospatial primitives

### Coordinates

Coordinates are the basic unit of location. Store and transmit in a well-defined coordinate reference system (CRS). Prefer a single internal CRS (e.g. WGS 84 / OGC:CRS84 for global lon/lat) and transform at ingest or display as needed. Document CRS in metadata; in [GeoParquet](../database-data/geoparquet.md) use the `geo` metadata block and PROJJSON when CRS is not default.

### Projections

Projections map the ellipsoid to the plane. Use equal-area or conformal projections appropriate to the area and analysis (e.g. UTM for regional, Web Mercator for tiling). Avoid mixing CRSs in the same pipeline without explicit conversion and document all transformations.

### Vector vs raster

- **Vector**: Points, lines, polygons; good for discrete features, attributes, and topology. Stored as WKB/WKT, GeoJSON, or in spatial databases (PostGIS).
- **Raster**: Grids of cells (imagery, DEMs, coverages); good for continuous phenomena and pixel-level analysis. Stored as GeoTIFF, COG, or in raster stores.

Choose representation based on query and analysis patterns. Hybrid pipelines (raster → vector extraction, or vector → rasterization) are common; see [PostGIS Raster–Vector Workflows](../../tutorials/database-data-engineering/postgis-raster-vector-workflows.md).

## Spatial indexing

### H3

[H3](https://h3geo.org/) is a hexagonal global grid. It provides stable, hierarchical cell IDs useful for aggregation, joins, and tiling. Use for global or continental datasets where hex topology and equal-area-ish properties matter. See [H3 Raster to Hex](../../tutorials/database-data-engineering/h3-raster-to-hex.md) and [H3 + Tile38 + NATS + DuckDB](../../tutorials/database-data-engineering/h3-tile38-nats-duckdb.md).

### Quadtrees

Quadtrees recursively subdivide space into four quadrants. They suit hierarchical tiling (e.g. web maps), level-of-detail, and spatial partitioning. Implementation and behavior depend on split rules and balance.

### R-trees

R-trees index bounding boxes (and often full geometries) in a tree of minimum bounding rectangles. They are the standard for "which geometries intersect this area?" in PostGIS and many spatial databases. Tune with appropriate fill factor and type (e.g. GiST, SP-GiST). See [PostGIS Geometry Indexing](../../tutorials/database-data-engineering/postgis-geometry-indexing.md) and [The Operational Geometry of Spatial Systems](../../deep-dives/the-operational-geometry-of-spatial-systems.md).

## Spatial data pipelines

```mermaid
flowchart LR
    A[Satellite / Sensors]
    B[Raster Processing]
    C[Vector Extraction]
    D[Spatial Indexing]
    E[Query Services]

    A --> B --> C --> D --> E
```

- **Ingest**: Satellite or sensor data (raster or vector) lands in raw or normalized form.
- **Raster processing**: Calibration, mosaicking, reprojection, COG generation, or derivation of products.
- **Vector extraction**: Feature extraction from rasters (e.g. contours, classification boundaries) or normalization of vector sources.
- **Spatial indexing**: Build H3/quadtree/R-tree indexes or partition by spatial keys for fast lookup.
- **Query services**: APIs, tiles, or analytical queries over indexed data.

For tiling pipelines see [Go-Glue OSM → PostGIS → Tiles](../../tutorials/database-data-engineering/go-osm-tiling-pipeline.md) and [Martin + PostGIS Tiling](../../tutorials/just-for-fun/martin-postgis-tiling.md). For format choices see [Geospatial File Format Choices](../../deep-dives/geospatial-file-format-choices.md).

## Storage considerations

### GeoParquet

Use [GeoParquet](../database-data/geoparquet.md) for analytical and lake-style storage: WKB geometry, `geo` metadata, CRS, and partitioning by space and time. Enables predicate pushdown and columnar efficiency with [parquet_s3_fdw](../../tutorials/database-data-engineering/parquet-s3-fdw.md) and [GeoParquet with Polars](../../tutorials/database-data-engineering/geoparquet-with-polars.md).

### Spatial databases

PostGIS (and other spatial databases) provide indexing, spatial SQL, and transaction safety. Use for serving, transactional workloads, and when you need topology or complex spatial queries. See [PostGIS Best Practices](../postgres/postgis-best-practices.md) and [Geospatial Data Engineering](../database-data/geospatial-data-engineering.md).

### Tiling strategies

For web and low-latency access, precompute tiles (vector or raster) and serve from object storage or CDN. Partition and index source data so tile generation is incremental and efficient. Balance tile size, zoom levels, and storage cost.

## Performance considerations

- **Spatial joins**: Expensive at scale. Reduce with spatial indexing, partitioning, and filter pushdown. Pre-aggregate or materialize when join patterns are stable.
- **Indexing strategies**: Match index type to queries (point vs polygon, containment vs proximity). Maintain statistics and vacuum; monitor index usage.
- **Chunked datasets**: Process large rasters and vector sets in chunks or partitions to control memory and parallelism. Use tiled rasters (e.g. COG) and partition keys (e.g. H3, grid) for scalability.

For benchmarking and comparison see [Geospatial Benchmarking](../database-data/geospatial-benchmarking.md) and [Polars vs Pandas for Geospatial Data](../../deep-dives/polars-vs-pandas-geospatial.md).

## See also

- [GeoParquet](../database-data/geoparquet.md) and [GeoParquet Data Warehouses](../database-data/geoparquet-data-warehouses.md) — format and warehouse patterns
- [Geospatial Data Engineering](../database-data/geospatial-data-engineering.md) — pipelines and engineering patterns
- [PostGIS Best Practices](../postgres/postgis-best-practices.md) — database-side spatial
- [Reproducible Data Pipelines](../data/reproducible-data-pipelines.md) — determinism in spatial pipelines
- [Geospatial File Format Choices](../../deep-dives/geospatial-file-format-choices.md) — format tradeoffs
- [The Operational Geometry of Spatial Systems](../../deep-dives/the-operational-geometry-of-spatial-systems.md) — indexing and geometry
- [Geospatial Data Mesh, Cost & Capacity](../../tutorials/best-practices-integration/geospatial-data-mesh-cost-capacity.md) — data mesh and cost in geospatial
