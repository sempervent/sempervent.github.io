---
tags:
  - deep-dive
  - geospatial
  - data-engineering
---

# Geospatial File Format Choices: GeoTIFF, Cloud-Optimized GeoTIFF, Parquet, and Beyond

*See also: [Parquet vs CSV vs ORC vs Avro](parquet-vs-csv-orc-avro.md) — the general columnar format landscape that GeoParquet extends into the geospatial domain.*

**Themes:** Data Formats · Spatial · Storage

---

## Spatial Data's Dual Nature

Geospatial data is not a monolithic category. It encompasses two fundamentally different data models — raster and vector — that have distinct storage requirements, distinct query patterns, and consequently distinct format histories. Conflating them leads to format choices optimized for the wrong model.

**Raster data** represents the world as a grid of cells, where each cell has a value (or set of values across spectral bands) corresponding to a geographic location. Satellite imagery, elevation models, temperature fields, and land cover classifications are raster data. The natural representation is a dense array, potentially very large, with spatial metadata defining the geographic extent and projection.

**Vector data** represents the world as discrete geometric objects — points, lines, and polygons — each optionally carrying attribute values. Administrative boundaries, roads, buildings, well locations, and species occurrence records are vector data. The natural representation is a table: each row is a feature with a geometry column and attribute columns.

These two models require different format considerations. Attempting to use raster formats for vector data or vice versa is technically possible in degenerate cases but uniformly inadvisable.

---

## Raster Formats: From GeoTIFF to COG

### GeoTIFF: The Raster Standard

GeoTIFF is a TIFF file with embedded georeferencing metadata — coordinate reference system (CRS), spatial extent, pixel resolution, and optionally additional metadata. It emerged in the 1990s and has been the dominant raster format in scientific and geospatial computing for over two decades. Its ubiquity is near-total: every GIS tool, every remote sensing library, every satellite data archive defaults to GeoTIFF.

A standard GeoTIFF stores pixel values as a 2D (or 3D, for multi-band) array. The file layout is sequential:

```
  Standard GeoTIFF file layout:
  ──────────────────────────────────────────────────────────────────
  [TIFF Header]
  [Image File Directory (IFD): metadata, offsets]
  [Image Data: rows of pixels, band by band or interleaved]
  [Optional: additional overviews (lower-resolution versions)]
```

The standard GeoTIFF's primary limitation for cloud access is that the image data is stored contiguously by row. To access a spatial subset — say, the upper-left quadrant of a 10 GB global mosaic — a client must know which byte range corresponds to that region, and this information requires reading the entire IFD and potentially some of the image data to determine. HTTP clients accessing the file from object storage (S3) must either download the entire file or make many small requests, each with the latency of an HTTP round trip.

### Cloud-Optimized GeoTIFF (COG): Enabling HTTP Range Requests

Cloud-Optimized GeoTIFF (COG) is a GeoTIFF with a specific internal organization designed to enable efficient partial access over HTTP. The COG specification requires:

1. **Overviews first**: the file begins with lower-resolution overview (pyramid) levels, from coarsest to finest.
2. **Internal tiling**: image data is organized into tiles (typically 256×256 or 512×512 pixels) rather than rows.
3. **Header at front**: all IFDs and tile offsets are placed at the beginning of the file, so a client can determine byte ranges for any tile with a single initial HTTP request.

```
  Cloud-Optimized GeoTIFF file layout:
  ──────────────────────────────────────────────────────────────────
  [TIFF Header]
  [IFDs: full resolution + overview IFDs]  ← known offsets upfront
  [Overview IFD data (coarsest first)]
     [Overview tiles: 8x zoom out]
     [Overview tiles: 4x zoom out]
     [Overview tiles: 2x zoom out]
  [Full resolution tiles]                  ← accessed by offset
     [Tile (0,0)][Tile (0,1)][Tile (1,0)]...

  Client workflow for spatial subset:
  1. HTTP GET bytes 0–16KB → read header + IFDs
  2. Compute which tiles intersect area of interest
  3. HTTP GET byte ranges for those tiles → read only relevant data
```

The COG organization enables a map tile server or a spatial analysis tool to read only the bytes corresponding to the area of interest, at the appropriate resolution, with two or three HTTP requests. A 100 GB global raster accessed as a COG requires reading perhaps 1–10 MB for a typical analysis region. The same file as a standard GeoTIFF would require downloading the full 100 GB or complex server-side subsetting.

COG represents the application of the same HTTP range request principle that makes Parquet efficient on object storage (reading column chunks by byte range) to the raster domain. The architectural insight is the same: organize files so that the metadata needed to locate relevant data is available cheaply, and the relevant data itself is contiguous on disk.

---

## Vector Formats: From Shapefile to GeoParquet

### The Shapefile's Persistence

The Shapefile format, developed by ESRI in the early 1990s, is among the most widely criticized file formats in active use and also among the most persistently used. It stores geometry and attributes across a mandatory minimum of three files: `.shp` (geometry), `.dbf` (attributes in dBASE format), `.shx` (spatial index). Additional files (`.prj` for CRS, `.cpg` for encoding) are technically optional but practically required.

Its limitations are well-documented: attribute names are limited to 10 characters, field types are limited (no native boolean, date, or time types), files cannot exceed 2 GB, Unicode support is unreliable, and the multi-file structure is operationally fragile.

The shapefile persists for the same reason CSV persists: it is universally supported. Every GIS tool, every GIS professional, every government data portal speaks Shapefile. The cost of conversion to a more capable format must be paid at every data exchange boundary, and the network effects of ubiquitous support are difficult to overcome.

### GeoJSON: Web-Native but Verbose

GeoJSON, formalized in 2016 (RFC 7946), represents geographic features as JSON with a standardized geometry encoding. It is the native format for web mapping APIs (Leaflet, Mapbox GL, Google Maps) and is widely used for data interchange in web applications.

GeoJSON's weaknesses for large datasets mirror JSON's general weaknesses for data storage: verbose text encoding of numeric coordinates (floating point numbers as decimal strings), no binary encoding option, no spatial indexing, and no schema enforcement. A million-feature vector dataset as GeoJSON may be 10–50x larger than the equivalent in a binary format, with no compression applied.

### GeoPackage and GeoParquet: Two Paths to Modernity

**GeoPackage** (OGC standard, 2013) stores vector and raster data in a SQLite database. It provides schema enforcement, multiple layers per file, spatial indexing (via R-tree), transactions, and a single-file distribution model. GeoPackage is the modern replacement for Shapefile in contexts where a file-based format is required — it is supported by QGIS, GDAL, ArcGIS, and most modern GIS tools.

**GeoParquet** is an emerging specification (v1.0, 2023) that extends Apache Parquet with a standardized encoding of geometric data and spatial metadata. A GeoParquet file is a valid Parquet file with a `geometry` column encoded in WKB (Well-Known Binary) and metadata in the file's schema specifying the CRS and geometry types.

```
  GeoParquet file layout:
  ─────────────────────────────────────────────────────────
  [Parquet file footer (schema, row group metadata)]
  [Row group N]
    [geometry column: WKB-encoded geometries]
    [attr column 1: values]
    [attr column 2: values]
    ...
  [Row group 1]
    ...

  Spatial metadata in Parquet key-value metadata:
  {
    "geo": {
      "version": "1.0.0",
      "primary_column": "geometry",
      "columns": {
        "geometry": {
          "encoding": "WKB",
          "crs": "OGC:CRS84",
          "bbox": [xmin, ymin, xmax, ymax]
        }
      }
    }
  }
```

GeoParquet inherits all of Parquet's properties: columnar storage, compression, predicate pushdown on attribute columns, cloud-native accessibility, and ecosystem integration with Spark, DuckDB, PyArrow, and Pandas. Spatial predicate pushdown (filter by bounding box or geometry intersection) is not yet standardized in GeoParquet v1.0 but is implemented in tools like DuckDB's spatial extension through bounding box column statistics.

---

## Spatial Indexing: File vs Database Trade-offs

Spatial queries — "find all features within this polygon" — require spatial indexing to avoid full dataset scans. The indexing story differs significantly between file formats and database formats.

| Approach | Spatial index | Query performance | Operational complexity |
|---|---|---|---|
| Shapefile + `.shx` | Limited | Poor at large scale | Low |
| GeoPackage (SQLite) | R-tree | Good for file-scale queries | Low |
| PostGIS (PostgreSQL) | GIST + SP-GiST | Excellent | Moderate |
| GeoParquet on S3 + DuckDB | Bbox statistics (partial) | Good; improving | Low-Moderate |
| GeoParquet on S3 + Sedona | Full spatial index | Very good at scale | Moderate-High |

For vector datasets under a few million features, file-based formats with internal indexing (GeoPackage, PostGIS) provide adequate spatial query performance. For continental- or global-scale datasets (hundreds of millions of features), distributed spatial processing (Apache Sedona on Spark, or Trino with spatial functions) is typically required.

---

## Compression Trade-offs in Spatial Data

Spatial data compresses non-uniformly. Attribute columns compress well with standard Parquet codecs — dictionary encoding for categorical columns (land cover class, administrative level), delta encoding for ordered numeric columns (elevation values in sorted order). Geometry columns are more challenging.

WKB-encoded geometries contain coordinate values as 64-bit IEEE floats. Adjacent geometries in a spatially sorted file tend to share high-order bits in their coordinate representations (features near each other have similar longitude and latitude values), which creates compression opportunities that delta encoding can exploit. GeoParquet's specification allows encodings other than WKB (including WKBZ for 3D), and future versions may standardize columnar-native geometry encodings designed for better compression.

Raster data compression is better understood and more mature: COG supports LZW, DEFLATE, ZSTD, JPEG, WebP, and LERC encoding at the tile level. Lossy compression (JPEG, WebP) is appropriate for visual imagery; lossless compression (ZSTD) is appropriate for scientific data where exact values matter.

---

## Decision Framework

**Raster data, cloud storage**: Cloud-Optimized GeoTIFF is the default. For data that will be accessed by map tile servers, cloud processing pipelines, or spatial analysis tools, COG's HTTP range request support eliminates the need for server-side subsetting and enables direct access from S3 without a tile server.

**Raster data, local processing**: Standard GeoTIFF is fine. If the data lives on local storage and is processed sequentially, the COG reorganization provides no benefit.

**Vector data, data exchange or small dataset (<10M features)**: GeoPackage if a single file is required; GeoJSON if web-API compatibility is required; Shapefile only if forced by legacy system requirements.

**Vector data, analytical workloads or large dataset (>10M features)**: GeoParquet for cloud-native access, Parquet-native tools, and columnar attribute access. PostGIS if complex spatial queries with indexing are required and operational overhead is acceptable.

**Shapefile**: acceptable only for legacy interoperability requirements. No new systems should be designed around Shapefile as a primary format.

**The spatial domain is converging on the same principles as the general data engineering domain**: open columnar formats on object storage with HTTP range access, table format metadata for governance, and engine-agnostic access through standardized specifications. The COG principle for raster and the GeoParquet principle for vector are manifestations of the same architectural insight applied to spatial data's specific properties.

!!! tip "See also"
    - [Parquet vs CSV vs ORC vs Avro](parquet-vs-csv-orc-avro.md) — the general columnar format landscape that GeoParquet extends
    - [DuckDB vs PostgreSQL vs Spark](duckdb-vs-postgres-vs-spark.md) — query engine selection for spatial analytics
    - [Polars vs Pandas for Geospatial Data](polars-vs-pandas-geospatial.md) — dataframe execution model trade-offs for processing GeoParquet and other spatial formats
    - [The Operational Geometry of Spatial Systems](the-operational-geometry-of-spatial-systems.md) — how H3 indexing, spatial partitioning, and query engine selection interact with file format choice
