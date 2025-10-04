# Martin + PostGIS: Tables, Views, Functions & Raster Tiles (2025 Edition)

**Objective**: Martin generates MVT tiles on the fly from PostGIS and can also serve tiles from MBTiles/PMTiles archives; we'll use that plus PostgreSQL table + function sources to serve both vector and raster tiles.

Martin generates MVT tiles on the fly from PostGIS and can also serve tiles from MBTiles/PMTiles archives; we'll use that plus PostgreSQL table + function sources to serve both vector and raster tiles.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand Martin's architecture**
   - Table sources: auto-discovered PostGIS tables/views
   - Function sources: parameterized SQL returning bytea
   - File sources: MBTiles/PMTiles archives
   - TileJSON endpoints for client integration

2. **Master PostGIS geometry handling**
   - Use Web Mercator (SRID 3857) for web tiles
   - Create spatial indexes on geometry columns
   - Optimize with ST_SimplifyVW for low zooms
   - Use ST_TileEnvelope for tile bounds

3. **Implement proper function signatures**
   - Vector functions: (z, x, y) → bytea MVT
   - Raster functions: (z, x, y) → bytea PNG/JPEG
   - Return (bytea, text) for ETag support
   - Use ST_AsMVT and ST_AsPNG for encoding

4. **Optimize for performance**
   - Index geometry columns
   - Use views to curate attributes
   - Implement caching with ETags
   - Pre-tile raster data when possible

5. **Plan for production**
   - Monitor tile generation performance
   - Use CDN caching for static tiles
   - Implement proper error handling
   - Keep Martin updated

**Why These Principles**: Martin tile serving requires understanding PostGIS geometry handling, function signatures, and performance optimization. Understanding these patterns prevents tile generation chaos and enables reliable map services.

## 1) Quick Architecture

### Martin + PostGIS Flow

```mermaid
flowchart LR
    A[(PostGIS)] -->|tables/views/functions| M[Martin Tile Server]
    M -->|/tiles/{z}/{x}/{y}.mvt| C1[MapLibre Vector]
    M -->|/rasters/{z}/{x}/{y}.png| C2[MapLibre Raster]
    M -->|/tilejson| T[TileJSON endpoints]
    
    subgraph "Data Sources"
        T1[Tables]
        V1[Views]
        F1[Functions]
        R1[Rasters]
    end
    
    subgraph "Tile Formats"
        MVT[MVT Vector]
        PNG[PNG Raster]
        JPEG[JPEG Raster]
    end
    
    subgraph "Client Integration"
        C1
        C2
        T
    end
    
    T1 --> M
    V1 --> M
    F1 --> M
    R1 --> M
    M --> MVT
    M --> PNG
    M --> JPEG
```

**Why Architecture Matters**: Understanding Martin's data flow enables efficient tile generation and client integration. Understanding these patterns prevents tile chaos and enables reliable map services.

## 2) Install & Run (Docker Compose)

### Complete Docker Compose Setup

```yaml
# docker-compose.yml
version: "3.9"

services:
  # PostGIS database
  db:
    image: postgis/postgis:15-3.4
    container_name: postgis-martin
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: gis
      POSTGRES_USER: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgis-data:/var/lib/postgresql/data
      - ./init-data.sql:/docker-entrypoint-initdb.d/init-data.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d gis"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Martin tile server
  martin:
    image: ghcr.io/maplibre/martin:latest
    container_name: martin-tiles
    command: [
      "--listen-addr", "0.0.0.0:3000",
      "--postgres-connection-string", "postgresql://postgres:postgres@db:5432/gis"
    ]
    ports:
      - "3000:3000"
    depends_on:
      db:
        condition: service_healthy
    environment:
      - MARTIN_LOG_LEVEL=info
    restart: unless-stopped

  # Optional: MapLibre demo client
  demo:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./demo:/usr/share/nginx/html
    depends_on:
      - martin
    restart: unless-stopped

volumes:
  postgis-data:
```

**Why Docker Compose Matters**: Orchestrated services enable complete tile serving stack deployment. Understanding these patterns prevents service chaos and enables reliable map services.

### Data Initialization

```sql
-- init-data.sql
-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;

-- Create sample places table
CREATE TABLE IF NOT EXISTS places (
    gid SERIAL PRIMARY KEY,
    name VARCHAR(255),
    geom GEOMETRY(POINT, 4326)
);

-- Insert sample data
INSERT INTO places (name, geom) VALUES
('London', ST_GeomFromText('POINT(-0.1276 51.5074)', 4326)),
('Paris', ST_GeomFromText('POINT(2.3522 48.8566)', 4326)),
('Berlin', ST_GeomFromText('POINT(13.4050 52.5200)', 4326)),
('Madrid', ST_GeomFromText('POINT(-3.7038 40.4168)', 4326)),
('Rome', ST_GeomFromText('POINT(12.4964 41.9028)', 4326));

-- Transform to Web Mercator and create spatial index
ALTER TABLE places ADD COLUMN geom_webmerc GEOMETRY(POINT, 3857);
UPDATE places SET geom_webmerc = ST_Transform(geom, 3857);
CREATE INDEX idx_places_geom_webmerc ON places USING GIST(geom_webmerc);

-- Create optimized view for Martin
CREATE OR REPLACE VIEW v_places AS
SELECT 
    gid,
    name,
    geom_webmerc AS geom
FROM places;

-- Create sample raster data (DEM)
CREATE TABLE IF NOT EXISTS dem (
    rid SERIAL PRIMARY KEY,
    rast RASTER
);

-- Insert sample raster (simplified)
INSERT INTO dem (rast) VALUES (
    ST_AddBand(
        ST_MakeEmptyRaster(256, 256, -20037508.34, 20037508.34, 156543.033928041, -156543.033928041, 0, 0, 3857),
        '32BF'::text, 0, NULL
    )
);

-- Create raster spatial index
CREATE INDEX idx_dem_rast ON dem USING GIST(ST_ConvexHull(rast));

-- Analyze tables
ANALYZE places;
ANALYZE dem;
```

**Why Data Initialization Matters**: Proper PostGIS setup enables efficient tile generation and spatial queries. Understanding these patterns prevents data chaos and enables reliable map services.

## 3) Table Source (Vector MVT)

### Basic Table Source

```sql
-- Create a more comprehensive places table
CREATE TABLE IF NOT EXISTS cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    country VARCHAR(255),
    population INTEGER,
    geom GEOMETRY(POINT, 4326)
);

-- Insert sample cities data
INSERT INTO cities (name, country, population, geom) VALUES
('London', 'United Kingdom', 8982000, ST_GeomFromText('POINT(-0.1276 51.5074)', 4326)),
('Paris', 'France', 2161000, ST_GeomFromText('POINT(2.3522 48.8566)', 4326)),
('Berlin', 'Germany', 3669000, ST_GeomFromText('POINT(13.4050 52.5200)', 4326)),
('Madrid', 'Spain', 3223000, ST_GeomFromText('POINT(-3.7038 40.4168)', 4326)),
('Rome', 'Italy', 2873000, ST_GeomFromText('POINT(12.4964 41.9028)', 4326)),
('Amsterdam', 'Netherlands', 872000, ST_GeomFromText('POINT(4.9041 52.3676)', 4326)),
('Vienna', 'Austria', 1911000, ST_GeomFromText('POINT(16.3738 48.2082)', 4326)),
('Prague', 'Czech Republic', 1309000, ST_GeomFromText('POINT(14.4378 50.0755)', 4326));

-- Transform to Web Mercator
ALTER TABLE cities ADD COLUMN geom_webmerc GEOMETRY(POINT, 3857);
UPDATE cities SET geom_webmerc = ST_Transform(geom, 3857);

-- Create spatial index
CREATE INDEX idx_cities_geom_webmerc ON cities USING GIST(geom_webmerc);

-- Create optimized view for Martin
CREATE OR REPLACE VIEW v_cities AS
SELECT 
    id,
    name,
    country,
    population,
    geom_webmerc AS geom
FROM cities;

-- Analyze table
ANALYZE cities;
```

**Why Table Sources Matter**: PostGIS tables with geometry columns are automatically discovered by Martin. Understanding these patterns prevents tile generation chaos and enables reliable map services.

### Testing Table Source

```bash
# Check Martin status
curl http://localhost:3000/

# Get TileJSON for cities
curl "http://localhost:3000/tilejson.json?src=postgres&layer=v_cities"

# Request a specific tile
curl "http://localhost:3000/tiles/v_cities/2/1/1.mvt" -o tile.mvt

# Check tile content
file tile.mvt
```

**Why Testing Matters**: Verifying tile endpoints ensures proper Martin configuration and data access. Understanding these patterns prevents tile chaos and enables reliable map services.

## 4) Function Source (Vector MVT)

### Vector Function Source

```sql
-- Create a function that returns MVT tiles
CREATE OR REPLACE FUNCTION mvt_cities(z integer, x integer, y integer)
RETURNS bytea
LANGUAGE SQL AS $$
  WITH bounds AS (
    SELECT ST_TileEnvelope(z, x, y) AS env
  ),
  clipped AS (
    SELECT 
      id,
      name,
      country,
      population,
      ST_AsMVTGeom(geom, (SELECT env FROM bounds)) AS geom
    FROM v_cities
    WHERE geom && (SELECT env FROM bounds)
  )
  SELECT ST_AsMVT(clipped, 'cities', 4096, 'geom') FROM clipped;
$$;

-- Create a function with ETag support for caching
CREATE OR REPLACE FUNCTION mvt_cities_cached(z integer, x integer, y integer)
RETURNS TABLE(tile bytea, etag text)
LANGUAGE SQL AS $$
  WITH bounds AS (
    SELECT ST_TileEnvelope(z, x, y) AS env
  ),
  clipped AS (
    SELECT 
      id,
      name,
      country,
      population,
      ST_AsMVTGeom(geom, (SELECT env FROM bounds)) AS geom
    FROM v_cities
    WHERE geom && (SELECT env FROM bounds)
  ),
  mvt AS (
    SELECT ST_AsMVT(clipped, 'cities', 4096, 'geom') AS tile FROM clipped
  )
  SELECT 
    mvt.tile,
    md5(mvt.tile::text) AS etag
  FROM mvt;
$$;
```

**Why Function Sources Matter**: Parameterized SQL functions enable custom tile generation and caching. Understanding these patterns prevents tile generation chaos and enables reliable map services.

### Advanced Vector Function

```sql
-- Create a function with zoom-based simplification
CREATE OR REPLACE FUNCTION mvt_cities_simplified(z integer, x integer, y integer)
RETURNS bytea
LANGUAGE SQL AS $$
  WITH bounds AS (
    SELECT ST_TileEnvelope(z, x, y) AS env
  ),
  clipped AS (
    SELECT 
      id,
      name,
      country,
      population,
      CASE 
        WHEN z < 8 THEN ST_AsMVTGeom(ST_SimplifyVW(geom, 1000), (SELECT env FROM bounds))
        ELSE ST_AsMVTGeom(geom, (SELECT env FROM bounds))
      END AS geom
    FROM v_cities
    WHERE geom && (SELECT env FROM bounds)
  )
  SELECT ST_AsMVT(clipped, 'cities', 4096, 'geom') FROM clipped;
$$;
```

**Why Advanced Functions Matter**: Zoom-based simplification enables efficient tile generation at different zoom levels. Understanding these patterns prevents tile generation chaos and enables reliable map services.

## 5) Raster Tiles from PostGIS

### Raster Function Source

```sql
-- Create a function that returns PNG raster tiles
CREATE OR REPLACE FUNCTION png_dem(z integer, x integer, y integer)
RETURNS bytea
LANGUAGE plpgsql AS $$
DECLARE
  env geometry;
  out_rast raster;
BEGIN
  -- Get tile envelope in Web Mercator
  SELECT ST_TileEnvelope(z, x, y) INTO env;
  
  -- Merge intersecting rasters, clip to tile, and resample to 256x256
  SELECT ST_Resample(
           ST_Union(ST_Clip(rast, env, true)),
           ST_AddBand(
             ST_MakeEmptyRaster(256, 256, ST_XMin(env), ST_YMax(env), 
                               156543.033928041/z, -156543.033928041/z, 0, 0, 3857),
             '32BF'::text, 0, NULL
           )
         )
    INTO out_rast
  FROM dem
  WHERE ST_Intersects(rast, env);

  -- Return NULL if no raster data
  IF out_rast IS NULL THEN
    RETURN NULL;
  END IF;

  -- Return PNG bytes
  RETURN ST_AsPNG(out_rast);
END;
$$;

-- Create a function with color mapping
CREATE OR REPLACE FUNCTION png_dem_colored(z integer, x integer, y integer)
RETURNS bytea
LANGUAGE plpgsql AS $$
DECLARE
  env geometry;
  out_rast raster;
BEGIN
  SELECT ST_TileEnvelope(z, x, y) INTO env;
  
  SELECT ST_Resample(
           ST_Union(ST_Clip(rast, env, true)),
           ST_AddBand(
             ST_MakeEmptyRaster(256, 256, ST_XMin(env), ST_YMax(env), 
                               156543.033928041/z, -156543.033928041/z, 0, 0, 3857),
             '32BF'::text, 0, NULL
           )
         )
    INTO out_rast
  FROM dem
  WHERE ST_Intersects(rast, env);

  IF out_rast IS NULL THEN
    RETURN NULL;
  END IF;

  -- Apply color map
  out_rast := ST_ColorMap(out_rast, 1, 'greys', 'EXP');
  
  RETURN ST_AsPNG(out_rast);
END;
$$;
```

**Why Raster Functions Matter**: PostGIS raster functions enable dynamic tile generation from spatial data. Understanding these patterns prevents tile generation chaos and enables reliable map services.

### Testing Raster Functions

```bash
# Test raster function
curl "http://localhost:3000/tiles/png_dem/2/1/1.png" -o dem_tile.png

# Test colored raster
curl "http://localhost:3000/tiles/png_dem_colored/2/1/1.png" -o dem_colored_tile.png

# Check tile content
file dem_tile.png
```

**Why Raster Testing Matters**: Verifying raster tile generation ensures proper PostGIS raster functions and Martin integration. Understanding these patterns prevents tile chaos and enables reliable map services.

## 6) MapLibre Client Integration

### HTML Demo Page

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Martin + PostGIS Demo</title>
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
        body { margin: 0; padding: 0; }
        #map { width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <script>
        const map = new maplibregl.Map({
            container: 'map',
            style: {
                version: 8,
                sources: {
                    // Vector source from PostGIS table
                    cities: {
                        type: "vector",
                        url: "http://localhost:3000/tilejson.json?src=postgres&layer=v_cities"
                    },
                    // Raster source from PostGIS function
                    dem: {
                        type: "raster",
                        tiles: ["http://localhost:3000/tiles/png_dem/{z}/{x}/{y}.png"],
                        tileSize: 256
                    }
                },
                layers: [
                    // Base layer
                    {
                        id: "background",
                        type: "background",
                        paint: {
                            "background-color": "#f0f0f0"
                        }
                    },
                    // Raster layer
                    {
                        id: "dem",
                        type: "raster",
                        source: "dem",
                        paint: {
                            "raster-opacity": 0.6
                        }
                    },
                    // Vector layer
                    {
                        id: "cities",
                        type: "circle",
                        source: "cities",
                        "source-layer": "cities",
                        paint: {
                            "circle-radius": {
                                "base": 1.75,
                                "stops": [[12, 2], [22, 180]]
                            },
                            "circle-color": "#00d1b2",
                            "circle-stroke-width": 1,
                            "circle-stroke-color": "#ffffff"
                        }
                    },
                    // City labels
                    {
                        id: "city-labels",
                        type: "symbol",
                        source: "cities",
                        "source-layer": "cities",
                        layout: {
                            "text-field": ["get", "name"],
                            "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"],
                            "text-size": 12,
                            "text-anchor": "top",
                            "text-offset": [0, 1]
                        },
                        paint: {
                            "text-color": "#333333",
                            "text-halo-color": "#ffffff",
                            "text-halo-width": 1
                        }
                    }
                ]
            },
            center: [0, 0],
            zoom: 2
        });

        // Add popup on click
        map.on('click', 'cities', (e) => {
            const coordinates = e.lngLat;
            const properties = e.features[0].properties;
            
            new maplibregl.Popup()
                .setLngLat(coordinates)
                .setHTML(`
                    <div>
                        <h3>${properties.name}</h3>
                        <p><strong>Country:</strong> ${properties.country}</p>
                        <p><strong>Population:</strong> ${properties.population?.toLocaleString() || 'N/A'}</p>
                    </div>
                `)
                .addTo(map);
        });

        // Change cursor on hover
        map.on('mouseenter', 'cities', () => {
            map.getCanvas().style.cursor = 'pointer';
        });

        map.on('mouseleave', 'cities', () => {
            map.getCanvas().style.cursor = '';
        });
    </script>
</body>
</html>
```

**Why MapLibre Integration Matters**: Client integration enables interactive map visualization with Martin tiles. Understanding these patterns prevents client chaos and enables reliable map services.

## 7) Performance & Operational Tips

### Optimization Strategies

```sql
-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_cities_geom_webmerc ON cities USING GIST(geom_webmerc);
CREATE INDEX CONCURRENTLY idx_cities_name ON cities (name);
CREATE INDEX CONCURRENTLY idx_cities_country ON cities (country);

-- Create materialized view for complex queries
CREATE MATERIALIZED VIEW mv_cities_stats AS
SELECT 
    country,
    COUNT(*) as city_count,
    AVG(population) as avg_population,
    ST_Collect(geom_webmerc) as geom
FROM cities
GROUP BY country;

-- Create index on materialized view
CREATE INDEX idx_mv_cities_stats_geom ON mv_cities_stats USING GIST(geom);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_cities_stats;
```

**Why Optimization Matters**: Proper indexing and materialized views enable efficient tile generation and spatial queries. Understanding these patterns prevents performance chaos and enables reliable map services.

### Caching Configuration

```sql
-- Function with ETag support for HTTP caching
CREATE OR REPLACE FUNCTION mvt_cities_with_etag(z integer, x integer, y integer)
RETURNS TABLE(tile bytea, etag text)
LANGUAGE SQL AS $$
  WITH bounds AS (
    SELECT ST_TileEnvelope(z, x, y) AS env
  ),
  clipped AS (
    SELECT 
      id,
      name,
      country,
      population,
      ST_AsMVTGeom(geom, (SELECT env FROM bounds)) AS geom
    FROM v_cities
    WHERE geom && (SELECT env FROM bounds)
  ),
  mvt AS (
    SELECT ST_AsMVT(clipped, 'cities', 4096, 'geom') AS tile FROM clipped
  )
  SELECT 
    mvt.tile,
    md5(mvt.tile::text) AS etag
  FROM mvt;
$$;
```

**Why Caching Matters**: ETag support enables HTTP caching and reduces server load. Understanding these patterns prevents performance chaos and enables reliable map services.

## 8) MBTiles/PMTiles Integration

### File Source Configuration

```yaml
# docker-compose.files.yml
version: "3.9"

services:
  martin:
    image: ghcr.io/maplibre/martin:latest
    container_name: martin-files
    command: [
      "--listen-addr", "0.0.0.0:3000",
      "--postgres-connection-string", "postgresql://postgres:postgres@db:5432/gis",
      "--file-source", "/data/mbtiles",
      "--file-source", "/data/pmtiles"
    ]
    ports:
      - "3000:3000"
    volumes:
      - ./mbtiles:/data/mbtiles
      - ./pmtiles:/data/pmtiles
    depends_on:
      - db
    restart: unless-stopped
```

**Why File Sources Matter**: MBTiles/PMTiles integration enables serving pre-generated tiles alongside PostGIS sources. Understanding these patterns prevents tile chaos and enables reliable map services.

## 9) Troubleshooting

### Common Issues

```bash
# Check Martin status
curl http://localhost:3000/

# Check PostGIS connection
docker exec -it postgis-martin psql -U postgres -d gis -c "SELECT version();"

# Check table discovery
curl "http://localhost:3000/catalog"

# Check specific layer
curl "http://localhost:3000/tilejson.json?src=postgres&layer=v_cities"

# Test tile generation
curl "http://localhost:3000/tiles/v_cities/2/1/1.mvt" -v

# Check raster tiles
curl "http://localhost:3000/tiles/png_dem/2/1/1.png" -v
```

**Why Troubleshooting Matters**: Proper debugging enables identification and resolution of tile generation issues. Understanding these patterns prevents tile chaos and enables reliable map services.

### Debugging Queries

```sql
-- Check table structure
\d v_cities

-- Check spatial reference system
SELECT ST_SRID(geom) FROM v_cities LIMIT 1;

-- Check geometry validity
SELECT COUNT(*) FROM v_cities WHERE NOT ST_IsValid(geom);

-- Check tile envelope
SELECT ST_AsText(ST_TileEnvelope(2, 1, 1));

-- Test MVT generation
SELECT ST_AsMVT(q, 'cities', 4096, 'geom') FROM (
  SELECT id, name, country, population, 
         ST_AsMVTGeom(geom, ST_TileEnvelope(2, 1, 1)) AS geom
  FROM v_cities
  WHERE geom && ST_TileEnvelope(2, 1, 1)
) q;
```

**Why Debugging Queries Matter**: SQL debugging enables identification of geometry and tile generation issues. Understanding these patterns prevents tile chaos and enables reliable map services.

## 10) TL;DR Runbook

### Essential Commands

```bash
# Start the stack
docker compose up -d

# Check Martin status
curl http://localhost:3000/

# Get TileJSON
curl "http://localhost:3000/tilejson.json?src=postgres&layer=v_cities"

# Request vector tile
curl "http://localhost:3000/tiles/v_cities/2/1/1.mvt" -o tile.mvt

# Request raster tile
curl "http://localhost:3000/tiles/png_dem/2/1/1.png" -o tile.png

# Check catalog
curl "http://localhost:3000/catalog"
```

### Essential Patterns

```yaml
# Essential Martin + PostGIS patterns
martin_patterns:
  "docker_compose": "Start PostGIS + Martin with Docker Compose",
  "table_sources": "Create PostGIS tables/views with geometry columns",
  "function_sources": "Create SQL functions returning bytea MVT/PNG",
  "spatial_indexes": "Create GIST indexes on geometry columns",
  "web_mercator": "Use SRID 3857 for web tiles",
  "tile_envelope": "Use ST_TileEnvelope for tile bounds",
  "mvt_generation": "Use ST_AsMVT for vector tiles",
  "raster_tiles": "Use ST_AsPNG for raster tiles",
  "caching": "Implement ETag support for HTTP caching",
  "optimization": "Use views, indexes, and materialized views"
```

### Quick Reference

```sql
-- Essential Martin + PostGIS operations
-- 1. Create table with geometry
CREATE TABLE places (id SERIAL, name TEXT, geom GEOMETRY(POINT, 4326));
ALTER TABLE places ADD COLUMN geom_webmerc GEOMETRY(POINT, 3857);
UPDATE places SET geom_webmerc = ST_Transform(geom, 3857);

-- 2. Create spatial index
CREATE INDEX idx_places_geom ON places USING GIST(geom_webmerc);

-- 3. Create view for Martin
CREATE VIEW v_places AS SELECT id, name, geom_webmerc AS geom FROM places;

-- 4. Create MVT function
CREATE FUNCTION mvt_places(z int, x int, y int) RETURNS bytea AS $$
  SELECT ST_AsMVT(q, 'places', 4096, 'geom') FROM (
    SELECT id, name, ST_AsMVTGeom(geom, ST_TileEnvelope(z, x, y)) AS geom
    FROM v_places WHERE geom && ST_TileEnvelope(z, x, y)
  ) q;
$$ LANGUAGE SQL;

-- 5. Create raster function
CREATE FUNCTION png_dem(z int, x int, y int) RETURNS bytea AS $$
  SELECT ST_AsPNG(ST_Resample(ST_Union(ST_Clip(rast, ST_TileEnvelope(z, x, y))), ...))
  FROM dem WHERE ST_Intersects(rast, ST_TileEnvelope(z, x, y));
$$ LANGUAGE plpgsql;
```

**Why This Runbook**: These patterns cover 90% of Martin + PostGIS needs. Master these before exploring advanced tile serving scenarios.

## 11) The Machine's Summary

Martin + PostGIS tile serving requires understanding geometry handling, function signatures, and performance optimization. When used correctly, Martin enables efficient tile generation, prevents tile chaos, and provides insights into spatial data visualization. The key is understanding PostGIS geometry, Martin's auto-discovery, and proper function implementation.

**The Dark Truth**: Without proper Martin understanding, your tiles remain slow and unreliable. Martin + PostGIS is your weapon. Use it wisely.

**The Machine's Mantra**: "In the geometry we trust, in the functions we find efficiency, and in the tiles we find the path to spatial data visualization."

**Why This Matters**: Martin + PostGIS enables efficient tile serving that can handle complex spatial data, prevent tile generation chaos, and provide insights into spatial patterns while ensuring technical accuracy and reliability.

---

*This guide provides the complete machinery for Martin + PostGIS tile serving. The patterns scale from simple table sources to complex raster functions, from basic MVT generation to advanced caching strategies.*
