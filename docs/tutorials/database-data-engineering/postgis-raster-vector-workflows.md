# Raster–Vector Workflows in PostGIS

This tutorial establishes the definitive approach to raster-vector hybrid workflows in PostGIS that scale to production datasets. We combine rasters (DEM, coverage, land cover, imagery) with vectors (zones, polygons, points) for analysis without grinding your server into dust.

**Why hybrid queries matter:** Zonal statistics enable watershed analysis, suitability modeling requires land cover extraction, and DEM sampling powers elevation-based routing. These workflows are the foundation of modern geospatial analysis.

## 1. Zonal Statistics (vector polygons over raster tiles)

### Example: average DEM elevation inside a watershed polygon

```sql
-- Performance trick: filter rasters by rast && geom before clipping
SELECT 
  w.id,
  w.name,
  ST_SummaryStats(ST_Clip(d.rast, w.geom)) as elevation_stats
FROM watersheds w
JOIN dem d ON ST_Intersects(d.rast, w.geom)
WHERE w.id = :watershed_id;
```

**Why:** The `ST_Intersects` filter uses the raster envelope index to eliminate irrelevant tiles before expensive clipping operations. This prevents processing thousands of tiles that don't intersect the watershed.

### Tile-by-tile vs union aggregation

```sql
-- Tile-by-tile aggregation (preferred for large polygons)
SELECT 
  w.id,
  AVG(ST_SummaryStats(ST_Clip(d.rast, w.geom)).mean) as avg_elevation,
  COUNT(*) as tile_count
FROM watersheds w
JOIN dem d ON ST_Intersects(d.rast, w.geom)
WHERE w.id = :watershed_id
GROUP BY w.id;

-- ST_Union approach (only for small polygons)
SELECT 
  w.id,
  ST_SummaryStats(ST_Clip(ST_Union(d.rast), w.geom)) as elevation_stats
FROM watersheds w
JOIN dem d ON ST_Intersects(d.rast, w.geom)
WHERE w.id = :watershed_id
GROUP BY w.id;
```

**Why:** Tile-by-tile aggregation scales to large polygons without memory exhaustion. ST_Union works for small areas but fails on massive watersheds that intersect thousands of tiles.

## 2. Point Sampling (extract raster values at vector points)

### Example: get elevation for GPS points

```sql
-- Join pattern: JOIN rasters r ON ST_Intersects(r.rast, pt.geom)
SELECT 
  p.id,
  p.name,
  ST_Value(d.rast, 1, p.geom) as elevation
FROM gps_points p
JOIN dem d ON ST_Intersects(d.rast, p.geom)
WHERE p.acquisition_date = :date;
```

**Why:** The spatial join uses envelope indexes on both rasters and points for efficient filtering. ST_Value extracts the pixel value at the exact point location without expensive interpolation.

### Multi-band sampling

```sql
-- Extract values from multiple raster bands
SELECT 
  p.id,
  ST_Value(landcover.rast, 1, p.geom) as landcover_class,
  ST_Value(dem.rast, 1, p.geom) as elevation,
  ST_Value(ndvi.rast, 1, p.geom) as ndvi_value
FROM gps_points p
JOIN landcover ON ST_Intersects(landcover.rast, p.geom)
JOIN dem ON ST_Intersects(dem.rast, p.geom)
JOIN ndvi ON ST_Intersects(ndvi.rast, p.geom)
WHERE p.id = :point_id;
```

**Why:** Multi-band sampling enables comprehensive environmental analysis at point locations. Separate joins for each raster band ensure proper spatial alignment and efficient index usage.

## 3. Masking Rasters With Vector Geometries

### Example: crop landcover raster by protected-area polygon

```sql
-- Crop raster to polygon boundary
SELECT 
  ST_Clip(lc.rast, pa.geom) as masked_raster
FROM landcover lc
JOIN protected_areas pa ON ST_Intersects(lc.rast, pa.geom)
WHERE pa.id = :protected_area_id;
```

**Why:** ST_Clip creates NULL values outside the geometry boundary, producing smaller, analysis-ready rasters. This reduces memory usage and focuses analysis on relevant areas.

### Batch masking with tile filtering

```sql
-- Efficient batch masking with envelope filtering
SELECT 
  pa.id,
  ST_Union(ST_Clip(lc.rast, pa.geom)) as masked_landcover
FROM protected_areas pa
JOIN landcover lc ON ST_Intersects(lc.rast, pa.geom)
WHERE pa.region = :region
GROUP BY pa.id;
```

**Why:** Batch processing with spatial filtering enables efficient masking of multiple polygons. ST_Union assembles clipped tiles into complete masked rasters.

## 4. Rasterizing Vectors (burning features into raster grids)

### Example: convert polygons to raster for overlay with DEM

```sql
-- Create alignment with a reference raster, then rasterize
WITH ref_raster AS (
  SELECT ST_Envelope(ST_Union(rast)) as bounds
  FROM dem 
  WHERE ST_Intersects(rast, ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 3857))
),
raster_template AS (
  SELECT ST_AsRaster(
    (SELECT bounds FROM ref_raster),
    ST_ScaleX((SELECT rast FROM dem LIMIT 1)),
    ST_ScaleY((SELECT rast FROM dem LIMIT 1)),
    ST_UpperLeftX((SELECT rast FROM dem LIMIT 1)),
    ST_UpperLeftY((SELECT rast FROM dem LIMIT 1)),
    0, 0, 3857
  ) as template
)
SELECT ST_AsRaster(
  lc.geom, 
  (SELECT template FROM raster_template),
  'landcover_class',
  '8BUI',
  0
) as landcover_raster
FROM landcover_polygons lc;
```

**Why:** Consistent cell alignment is critical for raster operations. The reference raster template ensures proper spatial registration between vector-derived and original rasters.

### Rasterizing with attribute values

```sql
-- Burn polygon attributes into raster cells
SELECT ST_AsRaster(
  p.geom,
  ref.rast,
  p.suitability_score::text,
  '32BF',
  p.suitability_score
) as suitability_raster
FROM suitability_polygons p
CROSS JOIN (
  SELECT rast FROM dem LIMIT 1
) ref;
```

**Why:** Attribute-based rasterization enables continuous value mapping from discrete polygons. This supports suitability modeling and continuous surface generation.

## 5. Hybrid Overlays (reclassify + join)

### Example: calculate mean elevation per land cover class

```sql
-- Workflow: rasterize polygons → overlay with DEM raster → aggregate
WITH landcover_raster AS (
  SELECT ST_AsRaster(
    lc.geom,
    dem.rast,
    lc.class_id::text,
    '8BUI',
    lc.class_id
  ) as rast
  FROM landcover_polygons lc
  JOIN dem ON ST_Intersects(dem.rast, lc.geom)
),
overlay_result AS (
  SELECT ST_MapAlgebra(
    lc.rast, dem.rast,
    '([rast1] = 1) * [rast2]',  -- Only where landcover = 1
    '32BF'
  ) as elevation_forest
  FROM landcover_raster lc
  JOIN dem ON ST_Intersects(lc.rast, dem.rast)
)
SELECT 
  ST_SummaryStats(elevation_forest) as forest_elevation_stats
FROM overlay_result;
```

**Why:** Pre-rasterizing vector classes enables efficient overlay operations. ST_MapAlgebra performs conditional raster operations without expensive per-pixel vector intersection tests.

### Multi-class overlay analysis

```sql
-- Calculate statistics for each land cover class
WITH class_elevation AS (
  SELECT 
    ST_MapAlgebra(
      lc.rast, dem.rast,
      '([rast1] = ' || class_id || ') * [rast2]',
      '32BF'
    ) as class_elevation,
    class_id
  FROM landcover_raster lc
  JOIN dem ON ST_Intersects(lc.rast, dem.rast)
  CROSS JOIN (SELECT unnest(ARRAY[1,2,3,4,5]) as class_id) classes
)
SELECT 
  class_id,
  ST_SummaryStats(class_elevation) as elevation_stats
FROM class_elevation
GROUP BY class_id;
```

**Why:** Multi-class analysis enables comprehensive land cover assessment. Dynamic MapAlgebra expressions process each class separately for detailed statistical analysis.

## 6. Indexing & Partitioning Tricks

### Always tile rasters and index envelopes

```sql
-- Tile rasters on ingest
INSERT INTO dem_tiled (rast)
SELECT ST_Tile(rast, 256, 256)
FROM dem_monolithic;

-- Index raster envelopes
CREATE INDEX dem_tiled_envelope_gist 
ON dem_tiled USING GIST (ST_Envelope(rast));

-- Index vector geometries
CREATE INDEX watersheds_geom_gist 
ON watersheds USING GIST (geom);
```

**Why:** Tiled storage enables efficient spatial filtering. Envelope indexes on rasters and GiST indexes on vectors provide the foundation for fast spatial joins.

### Subdivide large vector polygons

```sql
-- For zonal stats on huge polygons
CREATE TABLE watersheds_subdivided AS
SELECT 
  id,
  ST_Subdivide(geom, 1000) as geom  -- 1000 vertex limit
FROM watersheds;

CREATE INDEX watersheds_sub_geom_gist 
ON watersheds_subdivided USING GIST (geom);
```

**Why:** ST_Subdivide breaks large polygons into manageable pieces, preventing expensive clipping operations with monster geometries. This enables efficient processing of complex watershed boundaries.

### Partition large raster mosaics

```sql
-- Partition by region or acquisition date
CREATE TABLE dem_partitioned (
  id serial,
  region_id integer,
  acquisition_date date,
  rast raster
) PARTITION BY LIST (region_id);

CREATE TABLE dem_region_1 PARTITION OF dem_partitioned 
FOR VALUES IN (1);

CREATE INDEX dem_region_1_envelope_gist 
ON dem_region_1 USING GIST (ST_Envelope(rast));
```

**Why:** Partitioning enables constraint exclusion and parallel processing. Regional partitions optimize queries that focus on specific geographic areas.

## 7. Performance Rituals

### Use constraint exclusion on raster partitions

```sql
-- Enable constraint exclusion
SET constraint_exclusion = partition;

-- Query with partition key for optimal performance
SELECT ST_SummaryStats(ST_Clip(d.rast, w.geom))
FROM dem_partitioned d
JOIN watersheds w ON ST_Intersects(d.rast, w.geom)
WHERE d.region_id = 1  -- Partition pruning
  AND w.id = :watershed_id;
```

**Why:** Constraint exclusion eliminates irrelevant partitions before spatial operations. This reduces I/O and processing time for region-specific queries.

### Cache expensive clips with materialized views

```sql
-- Cache expensive operations
CREATE MATERIALIZED VIEW watershed_elevation_cache AS
SELECT 
  w.id,
  w.name,
  ST_Union(ST_Clip(d.rast, w.geom)) as elevation_raster
FROM watersheds w
JOIN dem d ON ST_Intersects(d.rast, w.geom)
GROUP BY w.id, w.name;

CREATE INDEX wsec_geom_gist 
ON watershed_elevation_cache USING GIST (ST_Envelope(elevation_raster));

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY watershed_elevation_cache;
```

**Why:** Materialized views cache expensive clipping operations for repeated analysis. CONCURRENTLY refresh prevents blocking during updates.

### Verify with EXPLAIN (ANALYZE, BUFFERS)

```sql
-- Always verify index usage
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT ST_Value(d.rast, 1, p.geom)
FROM gps_points p
JOIN dem d ON ST_Intersects(d.rast, p.geom)
WHERE p.acquisition_date = :date;

-- Look for:
-- - Index Cond on envelope operations
-- - Low buffer usage
-- - Fast execution times
```

**Why:** EXPLAIN ANALYZE reveals actual performance characteristics. Index conditions confirm spatial index usage; buffer statistics indicate I/O efficiency.

## 8. Advanced Workflows

### Temporal raster-vector analysis

```sql
-- Analyze land cover change over time
WITH temporal_analysis AS (
  SELECT 
    p.id,
    ST_Value(lc_2020.rast, 1, p.geom) as landcover_2020,
    ST_Value(lc_2021.rast, 1, p.geom) as landcover_2021
  FROM sample_points p
  JOIN landcover_2020 lc_2020 ON ST_Intersects(lc_2020.rast, p.geom)
  JOIN landcover_2021 lc_2021 ON ST_Intersects(lc_2021.rast, p.geom)
)
SELECT 
  landcover_2020,
  landcover_2021,
  COUNT(*) as point_count
FROM temporal_analysis
GROUP BY landcover_2020, landcover_2021;
```

**Why:** Temporal analysis requires consistent spatial alignment across time periods. Separate joins for each time period ensure proper temporal comparison.

### Suitability modeling workflow

```sql
-- Multi-criteria suitability analysis
WITH suitability_factors AS (
  SELECT 
    ST_MapAlgebra(elevation.rast, slope.rast, '[rast1] * 0.4 + [rast2] * 0.6', '32BF') as combined_score
  FROM elevation_raster elevation
  JOIN slope_raster slope ON ST_Intersects(elevation.rast, slope.rast)
),
suitability_zones AS (
  SELECT 
    z.id,
    ST_SummaryStats(ST_Clip(sf.combined_score, z.geom)) as zone_suitability
  FROM suitability_factors sf
  JOIN analysis_zones z ON ST_Intersects(sf.combined_score, z.geom)
)
SELECT 
  id,
  zone_suitability.mean as avg_suitability,
  zone_suitability.stddev as suitability_variability
FROM suitability_zones;
```

**Why:** Multi-criteria analysis combines multiple raster layers with weighted factors. ST_MapAlgebra enables complex mathematical operations on aligned raster grids.

## 9. TL;DR Checklist

- **rast && geom before ST_Clip/ST_Value** - Use spatial filtering before expensive operations
- **Tile rasters + GiST on envelopes** - Enable efficient spatial joins
- **Subdivide large vector polygons** - Prevent expensive clipping with monster geometries
- **Rasterize vectors ahead of time** - Cache expensive vector-to-raster conversions
- **Keep SRID and alignment consistent** - Ensure proper spatial operations
- **Measure with EXPLAIN** - Never trust assumptions about performance
- **Use constraint exclusion** - Optimize partitioned raster queries
- **Cache expensive operations** - Materialize views for repeated analysis

**Why:** This checklist ensures optimal raster-vector workflow performance. Each item addresses a critical aspect of hybrid spatial analysis and prevents common performance pitfalls.

## 10. Common Anti-Patterns

### Don't ST_Union rasters in hot paths

```sql
-- Bad: Union in query
SELECT ST_SummaryStats(ST_Union(ST_Clip(d.rast, w.geom)))
FROM dem d
JOIN watersheds w ON ST_Intersects(d.rast, w.geom)
WHERE w.id = :watershed_id;

-- Good: Tile-by-tile aggregation
SELECT ST_SummaryStats(ST_Clip(d.rast, w.geom))
FROM dem d
JOIN watersheds w ON ST_Intersects(d.rast, w.geom)
WHERE w.id = :watershed_id;
```

**Why:** ST_Union in queries causes memory exhaustion on large polygons. Tile-by-tile aggregation scales to any polygon size.

### Don't skip spatial indexes

```sql
-- Bad: No spatial filtering
SELECT ST_Value(d.rast, 1, p.geom)
FROM dem d, gps_points p
WHERE ST_Intersects(d.rast, p.geom);

-- Good: Proper spatial join
SELECT ST_Value(d.rast, 1, p.geom)
FROM gps_points p
JOIN dem d ON ST_Intersects(d.rast, p.geom);
```

**Why:** Explicit spatial joins enable index usage. Implicit joins often result in sequential scans that kill performance.

### Don't ignore SRID consistency

```sql
-- Bad: Mixed SRIDs
SELECT ST_Value(d.rast, 1, p.geom)
FROM dem d
JOIN gps_points p ON ST_Intersects(d.rast, p.geom);

-- Good: Consistent SRIDs
SELECT ST_Value(d.rast, 1, ST_Transform(p.geom, 3857))
FROM dem d
JOIN gps_points p ON ST_Intersects(d.rast, ST_Transform(p.geom, 3857));
```

**Why:** SRID mismatches cause coordinate system errors and poor performance. Transform coordinates to match raster SRID for accurate results.
