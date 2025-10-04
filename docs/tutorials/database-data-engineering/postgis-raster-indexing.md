# PostGIS Raster Indexing & Coverage Mosaics — Best Practices

This tutorial establishes the definitive approach to PostGIS raster handling that doesn't grind your server into dust. We enforce tiled storage, proper indexing, and queryable mosaics. Never stuff an entire GeoTIFF into one row; let the database breathe.

**Goal:** Handle rasters in PostGIS without performance degradation. Keep them tiled, indexed, and queryable as mosaics for efficient spatial operations.

## 0. Raster Columns: Schema Discipline

### Always tile on ingest

```sql
-- Rasters must be stored in chunks (tiles), not as monolithic images
-- Default: ST_Tile(rast, 256, 256) is a good start
-- Smaller tiles → more rows; larger tiles → fewer rows but slower random access
```

**Why:** Tiled storage enables efficient spatial queries and prevents memory exhaustion. Monolithic rasters break database performance and scalability.

### Column setup

```sql
-- Use raster type
-- Add SRID check constraints for consistency
ALTER TABLE dem ADD CONSTRAINT dem_srid_check CHECK (ST_SRID(rast) = 3857);
```

**Why:** SRID constraints ensure spatial consistency and enable proper coordinate transformations. Raster type provides optimized storage and operations.

## 1. Indexing Rasters

### A) Spatial index on raster envelope

```sql
CREATE INDEX dem_rast_gist
  ON dem
  USING GIST (ST_ConvexHull(rast));
```

**Why:** Rasters don't index directly. Index the geometry envelope of each tile (ST_ConvexHull or ST_Envelope) to enable spatial filtering.

### B) Per-band constraints (optional)

```sql
-- If you often query by band
ALTER TABLE dem ADD COLUMN band1_mean double precision
  GENERATED ALWAYS AS (ST_SummaryStats(rast, 1, TRUE)).mean STORED;

CREATE INDEX dem_band1_brin ON dem USING BRIN (band1_mean);
```

**Why:** BRIN on numeric summaries helps range queries ("tiles where band1 > X"). Generated columns precompute expensive band statistics.

## 2. Coverage Mosaics (stitching tiles)

### A) Virtual mosaics (views)

```sql
CREATE VIEW dem_mosaic AS
SELECT ST_Union(rast) AS rast
FROM dem;
```

**Why:** Logical union for small extents. Not scalable for huge datasets—use for demos or small coverages where memory usage is controlled.

### B) On-the-fly mosaics

```sql
SELECT ST_Union(rast)
FROM dem
WHERE ST_Intersects(ST_Envelope(rast), ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 3857));
```

**Why:** Use the envelope index to pull just the needed tiles, then merge. This approach scales to large datasets by filtering before union operations.

### C) ST_Tile + ST_AddBand

```sql
-- For multi-band mosaics, tile consistently and use ST_AddBand when assembling
-- Ensure consistent tile sizes and coordinate systems across bands
```

**Why:** Consistent tiling enables efficient multi-band operations. ST_AddBand provides proper band management during mosaic assembly.

## 3. Query Patterns

### A) Clip & resample

```sql
SELECT ST_AsTIFF(ST_Clip(rast, bbox.geom)) 
FROM dem, (SELECT ST_MakeEnvelope(:xmin,:ymin,:xmax,:ymax,3857) AS geom) bbox
WHERE rast && bbox.geom;
```

**Why:** Index accelerates &&; clip avoids reading whole tiles. This pattern enables efficient extraction of raster subsets.

### B) Value at a point

```sql
SELECT ST_Value(rast, 1, ST_Transform(pt.geom, 3857))
FROM dem, (SELECT ST_SetSRID(ST_MakePoint(:x, :y), 4326) AS geom) pt
WHERE ST_Intersects(rast, ST_Transform(pt.geom, 3857));
```

**Why:** Point value extraction requires spatial intersection testing. Transform coordinates to match raster SRID for accurate results.

### C) Zonal stats

```sql
SELECT ST_SummaryStats(ST_Clip(rast, zone.geom))
FROM dem d
JOIN zones z ON ST_Intersects(d.rast, z.geom)
WHERE z.id = :zone_id;
```

**Why:** Zonal statistics require clipping rasters to zone boundaries. Spatial joins enable efficient zone-based analysis.

## 4. Performance Rituals

```sql
-- Vacuum & analyze raster tables after load
VACUUM ANALYZE dem;

-- Use constraint exclusion with partitioned raster tables (by region, tile grid, or acquisition date)
-- Consider out-db storage (ST_FromGDALRaster) if rasters are too large for in-db
-- When exporting, use ST_AsTIFF or raster2pgsql -t 256x256
```

**Why:** Regular maintenance ensures optimal performance. Out-db storage prevents database bloat for large raster datasets. Consistent tiling enables efficient export operations.

## 5. Partitioning & Large Coverages

### Partition by space and time

```sql
-- Partition by space: state, region, H3 index, or UTM zone
-- Partition by time: acquisition date or year
-- Each partition gets its own raster index

CREATE TABLE dem_2020 PARTITION OF dem FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
CREATE INDEX dem_2020_rast_gist ON dem_2020 USING GIST (ST_Envelope(rast));
```

**Why:** Partitioning enables efficient query pruning and maintenance. Separate indexes per partition optimize spatial operations for large datasets.

## 6. Don'ts (performance killers)

- Don't load a single raster per row unless it's already tiled
- Don't run ST_Union over millions of tiles interactively
- Don't skip spatial indexes on raster envelopes; seq scans will kill you
- Don't use rasters where vectorizing (contours, grid cells) would be faster for your query

**Why:** These anti-patterns break raster performance and scalability. Monolithic rasters and unindexed operations cause memory exhaustion and slow queries.

## 7. Diagnostics

### Check index usage

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT ST_Value(rast, 1, pt.geom)
FROM dem, (SELECT ST_SetSRID(ST_MakePoint(:x, :y), 3857) AS geom) pt
WHERE rast && pt.geom;
```

**Why:** Look for Index Cond: (st_convexhull(rast) && ...). This confirms spatial index usage and identifies performance bottlenecks in raster operations.

### Performance monitoring

```sql
-- Monitor raster table sizes and index usage
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename LIKE '%dem%';

-- Check index effectiveness
SELECT indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename = 'dem';
```

**Why:** Monitoring enables proactive performance management. Track table growth and index usage to identify optimization opportunities.

## 8. Advanced Patterns

### A) Multi-resolution pyramids

```sql
-- Create multiple resolution levels for efficient visualization
CREATE TABLE dem_pyramid AS
SELECT 
  ST_Resample(rast, ST_ScaleX(rast)/2, ST_ScaleY(rast)/2) as rast,
  'level_1' as resolution
FROM dem
WHERE ST_Width(rast) > 512;

CREATE INDEX dem_pyramid_gist ON dem_pyramid USING GIST (ST_Envelope(rast));
```

**Why:** Multi-resolution pyramids enable efficient visualization at different zoom levels. Lower resolution tiles reduce data transfer and processing time.

### B) Temporal raster sequences

```sql
-- Partition by acquisition date for temporal analysis
CREATE TABLE dem_temporal (
  id serial,
  acquisition_date date,
  rast raster
) PARTITION BY RANGE (acquisition_date);

-- Create partitions for each year
CREATE TABLE dem_2020 PARTITION OF dem_temporal 
FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
```

**Why:** Temporal partitioning enables efficient time-series analysis. Separate partitions optimize queries by time range and enable independent maintenance.

### C) Raster algebra operations

```sql
-- Perform raster algebra on tiled datasets
SELECT ST_MapAlgebra(rast1, rast2, '[rast1] + [rast2]') as result
FROM dem1 d1
JOIN dem2 d2 ON ST_Intersects(d1.rast, d2.rast)
WHERE ST_Intersects(d1.rast, ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 3857));
```

**Why:** Raster algebra enables complex spatial analysis operations. Tiled storage ensures memory efficiency during computation.

## 9. Export and Integration

### A) Efficient export patterns

```sql
-- Export specific regions with proper tiling
SELECT ST_AsTIFF(ST_Union(rast), 'GTiff') as tiff_data
FROM dem
WHERE ST_Intersects(rast, ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 3857));
```

### B) GDAL integration

```bash
# Export with consistent tiling
raster2pgsql -t 256x256 -I -C -M dem.tif | psql -d your_db

# Import with proper indexing
raster2pgsql -t 256x256 -I -C -M -s 3857 dem.tif dem | psql -d your_db
```

**Why:** Consistent tiling during import/export maintains performance. GDAL integration provides seamless workflow with external tools.

## 10. TL;DR Checklist

- **Store rasters as tiled chunks** (256x256 default)
- **Add GiST index on ST_ConvexHull(rast)** for spatial operations
- **Use ST_DWithin, &&, ST_Intersects** with bounding boxes
- **Mosaic with ST_Union only on filtered subsets** to control memory usage
- **Partition large coverages by space/time** for scalability
- **Out-db rasters for monster datasets** to prevent database bloat
- **Always ANALYZE after ingest** for optimal query planning

**Why:** This checklist ensures optimal PostGIS raster performance. Each item addresses a critical aspect of raster database management and query optimization.

## 11. Worked Examples

### A) Elevation profile extraction

```sql
-- Extract elevation values along a line
WITH line_points AS (
  SELECT ST_LineInterpolatePoint(route_geom, s/ST_Length(route_geom)) as pt
  FROM generate_series(0, ST_Length(route_geom)::int) as s
)
SELECT ST_Value(d.rast, 1, lp.pt) as elevation
FROM dem d, line_points lp
WHERE ST_Intersects(d.rast, lp.pt)
ORDER BY ST_LineLocatePoint(route_geom, lp.pt);
```

### B) Watershed analysis

```sql
-- Calculate flow accumulation from DEM
SELECT ST_MapAlgebra(
  ST_Union(rast), 
  '[rast] * 0.001', 
  '32BF'
) as flow_accumulation
FROM dem
WHERE ST_Intersects(rast, watershed_boundary);
```

### C) Change detection

```sql
-- Compare two raster time periods
SELECT ST_MapAlgebra(
  d1.rast, d2.rast, 
  '[rast1] - [rast2]', 
  '32BF'
) as difference
FROM dem_2020 d1
JOIN dem_2021 d2 ON ST_Intersects(d1.rast, d2.rast);
```

**Why:** These examples demonstrate common raster analysis patterns with proper spatial indexing and memory management. They provide templates for real-world geospatial analysis workflows.
