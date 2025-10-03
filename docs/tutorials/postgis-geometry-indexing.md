# PostGIS Geometry Indexing — Best Practices

This tutorial establishes the definitive approach to PostGIS geometry indexing that makes spatial queries fast, predictable, and boring. We enforce correct SRIDs, SARGable predicates, proper index classes (GiST/SP-GiST), ruthless query hygiene, and maintenance that doesn't lie.

**TL;DR:** GiST on geometry + clean predicates wins most of the time. Don't wrap the indexed column in functions. Use KNN for nearest neighbors. Partition only when you must. Verify with EXPLAIN (ANALYZE, BUFFERS) or you're telling yourself bedtime stories.

## 0. Pre-flight: Schema Discipline

### Pick the right column type

```sql
-- Enforce a specific SRID and geometry type
ALTER TABLE roads
  ALTER COLUMN geom TYPE geometry(LineString, 3857)
  USING ST_Transform(geom, 3857);

ALTER TABLE roads
  ADD CONSTRAINT roads_geom_is_linestring_3857
  CHECK (GeometryType(geom) = 'LINESTRING' AND ST_SRID(geom) = 3857);
```

**Why:** The planner's selectivity estimates and index operators are only trustworthy when types and SRIDs are consistent. Use geometry unless you explicitly need ellipsoidal math—then geography.

### Don't index lies

```sql
-- Load data, then index
-- If you must transform geometries, do it once (ETL or generated columns), not per-query
```

**Why:** Load data first, then index. Transform geometries once during ETL, not per-query to maintain index effectiveness.

## 1. Index Classes: When to Use What

### A) GiST (Generalized Search Tree) — default for geometry

```sql
CREATE INDEX roads_geom_gist
  ON roads
  USING GIST (geom);

-- For 3D/4D operators, specify ND ops
CREATE INDEX points_geom_gist_nd
  ON points USING GIST (geom gist_geometry_ops_nd);
```

**Why:** GiST supports bounding-box filtering and distance ordering; it's the workhorse for most spatial predicates (ST_Intersects, &&, KNN <->, etc.).

### B) SP-GiST (Space-Partitioned GiST) — points, quadtree/k-d tree vibe

```sql
CREATE INDEX pts_geom_spgist
  ON points
  USING SPGIST (geom);
```

**Why:** Use for dense point datasets with lots of KNN/box queries. Often beats GiST on huge point clouds, but not always faster for mixed geometry types.

### C) BRIN — only if your data are spatially clustered on disk

```sql
-- Works great if rows are physically ordered by space (e.g., pre-binned by tile/H3)
-- Typical approach: generated columns for tile IDs, then BRIN/RANGE indexes on those
```

**Why:** BRIN only pays off when data is spatially clustered on disk. Use generated columns for tile IDs rather than direct geometry indexing.

## 2. Query Hygiene: Keep It SARGable

### A) Bounding boxes first, then exact tests

```sql
-- Good: SARGable, uses index
SELECT * FROM parcels
WHERE geom && ST_MakeEnvelope(-85, 35, -84.5, 35.5, 4326)
  AND ST_Intersects(geom, ST_MakeEnvelope(-85, 35, -84.5, 35.5, 4326));

-- Bad: function wraps the indexed column = index denial
-- WHERE ST_Intersects(ST_Buffer(geom, 10), :poly);
```

**Why:** Bounding box operations (&&) are fast and use indexes effectively. Avoid wrapping indexed columns in functions.

### B) Distance & radius searches (2D geometry)

```sql
-- Use ST_DWithin for fast "within radius" queries
SELECT *
FROM poi
WHERE ST_DWithin(
  geom,
  ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
  :meters
);
```

**Why:** ST_DWithin is index-accelerated; ST_Distance(geom, ...) < r often isn't. This enables efficient radius-based queries.

### C) KNN nearest neighbors (ORDER BY <->)

```sql
-- N nearest
SELECT id, name
FROM poi
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 3857)
LIMIT 10;
```

**Why:** KNN requires GiST (or SP-GiST) index and matching SRIDs. This provides efficient nearest neighbor queries without distance calculations.

### D) Keep the column naked

```sql
-- Do: ST_Intersects(geom, const_geom)
-- Don't: ST_Intersects(ST_Transform(geom, 4326), const_geom)

-- Fix: add a generated column in the target SRID if needed
ALTER TABLE parcels
  ADD COLUMN geom_4326 geometry(MultiPolygon, 4326)
  GENERATED ALWAYS AS (ST_Transform(geom, 4326)) STORED;

CREATE INDEX parcels_geom4326_gist
  ON parcels USING GIST (geom_4326);
```

**Why:** Transform the literal or use generated columns to avoid per-row transformations that break index usage.

## 3. Composite & Partial Indexes (when they pay)

### A) Filter + space (common subset + spatial)

```sql
-- Example: only "active" features are queried 99% of the time
CREATE INDEX roads_active_geom_gist
  ON roads USING GIST (geom)
  WHERE status = 'active';
```

**Why:** Smaller index, tighter selectivity. Planner can skip irrelevant rows and focus on the most commonly queried subset.

### B) Multi-column to guide planner

```sql
-- Example: city boundary joins and multipurpose filters
CREATE INDEX parcels_city_geom_gist
  ON parcels USING GIST (geom)
  INCLUDE (city_id);
```

**Why:** The INCLUDE column can make index-only plans possible for common queries, reducing I/O and improving performance.

## 4. Geometry Size & Validity

```sql
-- Validate once on ingest; don't weaponize ST_IsValid in queries
-- Consider ST_Subdivide for monstrous polygons to speed intersects/contains during ETL
-- Multi vs. single geometries: normalize (e.g., always MultiPolygon) to avoid polymorphic pain
```

**Why:** Validate geometries during ETL, not at query time. Normalize geometry types to avoid polymorphic issues and improve query planning.

## 5. Partitioning Strategy (only if the data demand it)

### When to partition

```sql
-- Example: list partitioning by state FIPS
CREATE TABLE parcels (
  id bigserial primary key,
  state_fips smallint not null,
  geom geometry(MultiPolygon, 3857) not null,
  ...
) PARTITION BY LIST (state_fips);

CREATE TABLE parcels_47 PARTITION OF parcels FOR VALUES IN (47); -- Tennessee
CREATE INDEX parcels_47_geom_gist ON parcels_47 USING GIST (geom);
```

**Why:** Partition only when table > hundreds of millions of rows with skewed geography. Use stable keys (state FIPS, tile id) for partitioning, not geometry directly.

## 6. Maintenance & Build Order

### Bulk load ritual

```sql
-- 1. Load data (COPY is king)
-- 2. ANALYZE (so the planner stops guessing)
ANALYZE parcels;

-- 3. Build indexes (buffered build is automatic)
CREATE INDEX CONCURRENTLY parcels_geom_gist ON parcels USING GIST (geom);

-- 4. CLUSTER (optional) to pack locality (or use pg_repack)
CLUSTER parcels USING parcels_geom_gist;
```

### Autovacuum tuning (heavy write tables)

```sql
-- Increase autovacuum_vacuum_cost_limit, lower autovacuum_vacuum_scale_factor
-- Analyze proactively after large batches: ANALYZE VERBOSE parcels;
```

### Statistics depth (complex distributions)

```sql
ALTER TABLE parcels ALTER COLUMN geom SET STATISTICS 1000;
ANALYZE parcels (geom);
```

**Why:** Better statistics lead to better query plans, especially for skewed spatial distributions. Proper maintenance order ensures optimal performance.

## 7. Geography Column Notes (if you truly need it)

```sql
-- Index the geography column with GiST
CREATE INDEX poi_geog_gist
  ON poi USING GIST (geog);

-- Use ST_DWithin(geog, geog, meters) for radius queries; it's index-accelerated
-- Avoid per-row ST_Transform back-and-forth; pick one type and stand by it
```

**Why:** Geography columns require GiST indexing for performance. Avoid constant transformations between geometry and geography types.

## 8. Operator Quick Reference (index-friendly)

- `&&`: bounding box intersect (fast, cheap)
- `&&&`: n-D bounding box intersect (use ND opclass)
- `<->`: KNN distance operator (ORDER BY)
- `@`, `~` variants: containment/within for boxes (rarely needed directly)
- **Prefer:** ST_Intersects, ST_DWithin, ST_Covers/Within with literals matching SRID
- **Avoid:** wrapping geom in ST_Buffer, ST_Simplify, ST_Transform inside WHERE

**Why:** These operators are designed for index usage. Avoid function-wrapped columns in WHERE clauses.

## 9. Diagnostic Playbook

### A) Is the index used?

```sql
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT *
FROM parcels
WHERE geom && ST_MakeEnvelope(:xmin,:ymin,:xmax,:ymax, 3857)
  AND ST_Intersects(geom, ST_MakeEnvelope(:xmin,:ymin,:xmax,:ymax, 3857));
```

**Why:** Look for Index Cond on gist/spgist and recheck lines (normal for GIS). This confirms index usage and identifies performance bottlenecks.

### B) Why is my KNN slow?

- Ensure matching SRID and GiST/SP-GiST index exists
- ORDER BY geom <-> :pt LIMIT N must not include extra functions on geom

### C) Still slow?

- Materialize expensive predicates (generated columns)
- Increase stats target, re-ANALYZE
- Consider SP-GiST for dense points; consider partitioning when data volume demands it

**Why:** Systematic diagnosis identifies the root cause of performance issues. Measure with EXPLAIN ANALYZE before optimizing.

## 10. Patterns That Scale

### A) Generated columns for speed

```sql
-- Store centroids for cheap KNN prefilter on polygons
ALTER TABLE parcels
  ADD COLUMN centroid geometry(Point, 3857)
  GENERATED ALWAYS AS (ST_Centroid(geom)) STORED;

CREATE INDEX parcels_centroid_gist ON parcels USING GIST (centroid);
```

**Why:** Use case: Quick KNN on centroid, then exact polygon checks. Generated columns precompute expensive operations.

### B) Envelope columns for BRIN/RANGE partition pruning

```sql
ALTER TABLE parcels
  ADD COLUMN xmin double precision GENERATED ALWAYS AS (ST_XMin(geom)) STORED,
  ADD COLUMN xmax double precision GENERATED ALWAYS AS (ST_XMax(geom)) STORED,
  ADD COLUMN ymin double precision GENERATED ALWAYS AS (ST_YMin(geom)) STORED,
  ADD COLUMN ymax double precision GENERATED ALWAYS AS (ST_YMax(geom)) STORED;

-- Optional BRIN indexes when rows are roughly clustered by space
CREATE INDEX parcels_xmin_brin ON parcels USING BRIN (xmin);
CREATE INDEX parcels_ymin_brin ON parcels USING BRIN (ymin);
```

**Why:** BRIN pays off only when insertion order matches spatial order. Generated envelope columns enable efficient spatial filtering.

## 11. Don'ts (graveyard of performance)

- Don't ST_Transform(geom, …) in WHERE; transform the literal or use a generated column
- Don't index both geometry and geography for the same data unless you have a concrete plan for each
- Don't create redundant GiST indexes (one per geometry column is enough)
- Don't rely on ST_Distance < r; use ST_DWithin
- Don't forget ANALYZE after bulk loads; the planner is blind without stats

**Why:** These anti-patterns break index usage and degrade performance. Avoid them to maintain query efficiency.

## 12. Worked Recipes

### A) Window query: features intersecting a viewport

```sql
WITH bbox AS (
  SELECT ST_MakeEnvelope(:xmin,:ymin,:xmax,:ymax, 3857) AS win
)
SELECT id, attrs
FROM parcels p, bbox b
WHERE p.geom && b.win
  AND ST_Intersects(p.geom, b.win);
```

### B) N nearest restaurants to a click

```sql
SELECT id, name
FROM restaurants
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:x, :y), 3857)
LIMIT 20;
```

### C) Radius search (500 m)

```sql
SELECT id, name
FROM hydrants
WHERE ST_DWithin(
  geom,
  ST_SetSRID(ST_MakePoint(:x, :y), 3857),
  500
);
```

**Why:** These patterns demonstrate common spatial query patterns with proper index usage. They provide templates for real-world applications.

## 13. TL;DR Checklist

- **Schema:** geometry(..., SRID) fixed + CHECK constraints
- **Index:** CREATE INDEX ... USING GIST (geom); (SP-GiST for massive points)
- **Queries:** && + ST_Intersects or ST_DWithin; no functions on geom
- **KNN:** ORDER BY geom <-> :pt LIMIT n (matching SRID)
- **Analyze:** ANALYZE after loads; raise stats target when skewed
- **Partition:** only when necessary; use simple keys; per-partition GiST
- **Generated columns:** for transformed/derived geometries; index those
- **Verify:** with EXPLAIN (ANALYZE, BUFFERS). If it isn't measured, it didn't happen

**Why:** This checklist ensures optimal PostGIS performance. Each item addresses a critical aspect of spatial query optimization.
