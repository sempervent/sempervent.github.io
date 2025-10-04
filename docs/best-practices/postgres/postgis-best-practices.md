# PostGIS Best Practices

**Objective**: Master senior-level PostGIS spatial data patterns for production systems. When you need to handle geospatial data, when you want to optimize spatial queries, when you need enterprise-grade spatial analysis strategies—these best practices become your weapon of choice.

## Core Principles

- **Spatial Indexing**: Use appropriate spatial indexes for performance
- **Coordinate Systems**: Choose correct CRS for your use case
- **Data Types**: Select optimal geometry vs geography types
- **Query Optimization**: Optimize spatial queries for performance
- **Data Quality**: Ensure spatial data integrity and validation

## PostGIS Setup and Configuration

### PostGIS Extension Installation

```sql
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Check PostGIS version and capabilities
SELECT PostGIS_Version();
SELECT PostGIS_GEOS_Version();
SELECT PostGIS_Lib_Version();
SELECT PostGIS_Scripts_Build_Date();

-- Check available spatial reference systems
SELECT srid, auth_name, auth_srid, srtext 
FROM spatial_ref_sys 
WHERE auth_name = 'EPSG' 
ORDER BY auth_srid;
```

### Spatial Reference System Configuration

```sql
-- Create tables with appropriate SRID
CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL,
    geom GEOMETRY(POINT, 4326),              -- WGS84 for global data
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE buildings (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    height_meters INTEGER,
    geom GEOMETRY(POLYGON, 3857),            -- Web Mercator for web mapping
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE administrative_boundaries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    level VARCHAR(50) NOT NULL,
    geom GEOMETRY(MULTIPOLYGON, 4326),       -- WGS84 for global boundaries
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create geography table for global calculations
CREATE TABLE global_events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(200) NOT NULL,
    location GEOGRAPHY(POINT, 4326),         -- Geography for global distance calculations
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

## Spatial Data Types and Storage

### Geometry vs Geography

```sql
-- Use GEOMETRY for local/regional data
CREATE TABLE local_parcels (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) UNIQUE NOT NULL,
    owner_name VARCHAR(200),
    area_sq_meters NUMERIC(12,2),
    geom GEOMETRY(POLYGON, 4326),            -- Geometry for local operations
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Use GEOGRAPHY for global data and distance calculations
CREATE TABLE global_sensors (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) UNIQUE NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326),          -- Geography for global operations
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO cities (name, country, geom) VALUES
    ('New York', 'USA', ST_GeomFromText('POINT(-74.0059 40.7128)', 4326)),
    ('London', 'UK', ST_GeomFromText('POINT(-0.1276 51.5074)', 4326)),
    ('Tokyo', 'Japan', ST_GeomFromText('POINT(139.6917 35.6895)', 4326));

INSERT INTO global_sensors (sensor_id, sensor_type, location) VALUES
    ('SENSOR_001', 'temperature', ST_GeogFromText('POINT(-74.0059 40.7128)')),
    ('SENSOR_002', 'humidity', ST_GeogFromText('POINT(-0.1276 51.5074)')),
    ('SENSOR_003', 'pressure', ST_GeogFromText('POINT(139.6917 35.6895)'));
```

### Spatial Data Validation

```sql
-- Create function to validate spatial data
CREATE OR REPLACE FUNCTION validate_geometry(geom GEOMETRY)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if geometry is valid
    IF NOT ST_IsValid(geom) THEN
        RETURN FALSE;
    END IF;
    
    -- Check if geometry is not empty
    IF ST_IsEmpty(geom) THEN
        RETURN FALSE;
    END IF;
    
    -- Check if geometry has correct dimension
    IF ST_Dimension(geom) < 0 THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add validation constraints
ALTER TABLE cities ADD CONSTRAINT cities_geom_valid 
    CHECK (validate_geometry(geom));

ALTER TABLE buildings ADD CONSTRAINT buildings_geom_valid 
    CHECK (validate_geometry(geom));

-- Create function to fix invalid geometries
CREATE OR REPLACE FUNCTION fix_invalid_geometry(geom GEOMETRY)
RETURNS GEOMETRY AS $$
BEGIN
    IF ST_IsValid(geom) THEN
        RETURN geom;
    ELSE
        RETURN ST_MakeValid(geom);
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Spatial Indexing Strategies

### GIST Indexes for Spatial Data

```sql
-- Create spatial indexes
CREATE INDEX idx_cities_geom_gist ON cities USING GIST (geom);
CREATE INDEX idx_buildings_geom_gist ON buildings USING GIST (geom);
CREATE INDEX idx_administrative_boundaries_geom_gist ON administrative_boundaries USING GIST (geom);
CREATE INDEX idx_global_sensors_location_gist ON global_sensors USING GIST (location);

-- Create partial spatial indexes
CREATE INDEX idx_cities_geom_gist_active ON cities USING GIST (geom) 
WHERE created_at > CURRENT_DATE - INTERVAL '1 year';

-- Create spatial indexes with specific operators
CREATE INDEX idx_buildings_geom_gist_ops ON buildings USING GIST (geom gist_geometry_ops_2d);
```

### Spatial Index Performance

```sql
-- Analyze spatial index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%_geom_gist%' OR indexname LIKE '%_location_gist%'
ORDER BY idx_scan DESC;

-- Check spatial index statistics
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE attname LIKE '%geom%' OR attname LIKE '%location%';
```

## Spatial Query Optimization

### Spatial Query Patterns

```sql
-- Point-in-polygon queries
SELECT 
    c.name as city_name,
    c.country,
    ab.name as boundary_name,
    ab.level
FROM cities c
JOIN administrative_boundaries ab ON ST_Contains(ab.geom, c.geom)
WHERE ab.level = 'country';

-- Distance-based queries
SELECT 
    s1.sensor_id as sensor1,
    s2.sensor_id as sensor2,
    ST_Distance(s1.location, s2.location) as distance_meters
FROM global_sensors s1
CROSS JOIN global_sensors s2
WHERE s1.sensor_id < s2.sensor_id
AND ST_DWithin(s1.location, s2.location, 1000)  -- Within 1km
ORDER BY distance_meters;

-- Spatial joins with performance optimization
SELECT 
    b.name as building_name,
    b.height_meters,
    c.name as city_name,
    ST_Distance(b.geom, c.geom) as distance_meters
FROM buildings b
JOIN cities c ON ST_DWithin(b.geom, c.geom, 10000)  -- Within 10km
WHERE b.height_meters > 50
ORDER BY distance_meters;
```

### Spatial Aggregation

```sql
-- Spatial aggregation functions
SELECT 
    ab.name as boundary_name,
    ab.level,
    COUNT(c.id) as city_count,
    AVG(ST_Area(ab.geom)) as avg_area,
    ST_Union(c.geom) as city_union_geom
FROM administrative_boundaries ab
LEFT JOIN cities c ON ST_Contains(ab.geom, c.geom)
GROUP BY ab.id, ab.name, ab.level, ab.geom;

-- Spatial clustering
SELECT 
    ST_ClusterKMeans(geom, 5) as cluster_id,
    COUNT(*) as point_count,
    ST_Centroid(ST_Collect(geom)) as cluster_center
FROM cities
GROUP BY ST_ClusterKMeans(geom, 5);
```

## Coordinate Reference Systems

### CRS Transformation

```sql
-- Transform between coordinate systems
SELECT 
    name,
    geom as original_geom,
    ST_Transform(geom, 3857) as web_mercator_geom,
    ST_Transform(geom, 32633) as utm_geom
FROM cities
WHERE name = 'New York';

-- Create function for CRS transformation
CREATE OR REPLACE FUNCTION transform_to_web_mercator(geom GEOMETRY)
RETURNS GEOMETRY AS $$
BEGIN
    RETURN ST_Transform(geom, 3857);
END;
$$ LANGUAGE plpgsql;

-- Use CRS transformation in queries
SELECT 
    name,
    ST_Area(ST_Transform(geom, 3857)) as area_sq_meters
FROM administrative_boundaries
WHERE level = 'state';
```

### CRS Best Practices

```sql
-- Create tables with appropriate CRS
CREATE TABLE local_roads (
    id SERIAL PRIMARY KEY,
    road_name VARCHAR(200) NOT NULL,
    road_type VARCHAR(50) NOT NULL,
    geom GEOMETRY(LINESTRING, 4326),         -- WGS84 for global roads
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE local_parcels (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) UNIQUE NOT NULL,
    owner_name VARCHAR(200),
    geom GEOMETRY(POLYGON, 4326),            -- WGS84 for global parcels
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to get appropriate CRS for region
CREATE OR REPLACE FUNCTION get_regional_crs(geom GEOMETRY)
RETURNS INTEGER AS $$
DECLARE
    centroid POINT;
    longitude NUMERIC;
    latitude NUMERIC;
BEGIN
    centroid := ST_Centroid(geom);
    longitude := ST_X(centroid);
    latitude := ST_Y(centroid);
    
    -- Return appropriate UTM zone
    IF latitude >= 0 THEN
        RETURN 32600 + (FLOOR((longitude + 180) / 6) + 1);
    ELSE
        RETURN 32700 + (FLOOR((longitude + 180) / 6) + 1);
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Spatial Analysis Functions

### Geometric Operations

```sql
-- Create spatial analysis functions
CREATE OR REPLACE FUNCTION calculate_building_density(
    boundary_geom GEOMETRY,
    building_geom GEOMETRY
)
RETURNS NUMERIC AS $$
DECLARE
    building_count INTEGER;
    boundary_area NUMERIC;
BEGIN
    -- Count buildings within boundary
    SELECT COUNT(*) INTO building_count
    FROM buildings
    WHERE ST_Within(geom, boundary_geom);
    
    -- Calculate boundary area in square meters
    boundary_area := ST_Area(ST_Transform(boundary_geom, 3857));
    
    -- Return density (buildings per square kilometer)
    RETURN (building_count / (boundary_area / 1000000));
END;
$$ LANGUAGE plpgsql;

-- Use spatial analysis function
SELECT 
    ab.name as boundary_name,
    calculate_building_density(ab.geom, b.geom) as building_density
FROM administrative_boundaries ab
CROSS JOIN buildings b
WHERE ab.level = 'city';
```

### Spatial Relationships

```sql
-- Spatial relationship queries
SELECT 
    c.name as city_name,
    b.name as building_name,
    ST_Distance(c.geom, b.geom) as distance_meters,
    CASE 
        WHEN ST_Contains(c.geom, b.geom) THEN 'Inside'
        WHEN ST_Intersects(c.geom, b.geom) THEN 'Intersects'
        WHEN ST_Touches(c.geom, b.geom) THEN 'Touches'
        ELSE 'Separate'
    END as spatial_relationship
FROM cities c
CROSS JOIN buildings b
WHERE ST_DWithin(c.geom, b.geom, 1000)
ORDER BY distance_meters;
```

## Spatial Data Import/Export

### Data Import Functions

```sql
-- Create function to import spatial data from GeoJSON
CREATE OR REPLACE FUNCTION import_geojson_features(
    geojson_data JSONB,
    target_table TEXT
)
RETURNS INTEGER AS $$
DECLARE
    feature JSONB;
    feature_count INTEGER := 0;
    geom_wkt TEXT;
    properties JSONB;
BEGIN
    -- Process each feature in the GeoJSON
    FOR feature IN SELECT * FROM jsonb_array_elements(geojson_data->'features') LOOP
        -- Extract geometry as WKT
        geom_wkt := ST_AsText(ST_GeomFromGeoJSON(feature->'geometry'));
        
        -- Extract properties
        properties := feature->'properties';
        
        -- Insert into target table
        EXECUTE format('INSERT INTO %I (geom, properties) VALUES (ST_GeomFromText(%L, 4326), %L)',
                      target_table, geom_wkt, properties);
        
        feature_count := feature_count + 1;
    END LOOP;
    
    RETURN feature_count;
END;
$$ LANGUAGE plpgsql;
```

### Data Export Functions

```sql
-- Create function to export spatial data to GeoJSON
CREATE OR REPLACE FUNCTION export_to_geojson(
    table_name TEXT,
    geom_column TEXT DEFAULT 'geom',
    where_clause TEXT DEFAULT '1=1'
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    features JSONB;
BEGIN
    -- Build GeoJSON features
    EXECUTE format('
        SELECT jsonb_build_object(
            ''type'', ''FeatureCollection'',
            ''features'', jsonb_agg(
                jsonb_build_object(
                    ''type'', ''Feature'',
                    ''geometry'', ST_AsGeoJSON(%I)::jsonb,
                    ''properties'', row_to_json(t) - ''%I''
                )
            )
        )
        FROM (
            SELECT * FROM %I WHERE %s
        ) t
    ', geom_column, geom_column, table_name, where_clause) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
```

## Spatial Data Quality

### Data Quality Validation

```sql
-- Create spatial data quality validation
CREATE OR REPLACE FUNCTION validate_spatial_data_quality(
    geom GEOMETRY
)
RETURNS TABLE (
    validation_check TEXT,
    result BOOLEAN,
    message TEXT
) AS $$
BEGIN
    -- Check if geometry is valid
    RETURN QUERY SELECT 'valid_geometry', ST_IsValid(geom), 
        CASE WHEN ST_IsValid(geom) THEN 'Geometry is valid' ELSE 'Geometry is invalid' END;
    
    -- Check if geometry is not empty
    RETURN QUERY SELECT 'not_empty', NOT ST_IsEmpty(geom),
        CASE WHEN NOT ST_IsEmpty(geom) THEN 'Geometry is not empty' ELSE 'Geometry is empty' END;
    
    -- Check if geometry has correct dimension
    RETURN QUERY SELECT 'correct_dimension', ST_Dimension(geom) >= 0,
        CASE WHEN ST_Dimension(geom) >= 0 THEN 'Geometry has correct dimension' ELSE 'Geometry has incorrect dimension' END;
    
    -- Check if geometry is not too complex
    RETURN QUERY SELECT 'not_too_complex', ST_NPoints(geom) < 10000,
        CASE WHEN ST_NPoints(geom) < 10000 THEN 'Geometry is not too complex' ELSE 'Geometry is too complex' END;
END;
$$ LANGUAGE plpgsql;
```

### Spatial Data Cleaning

```sql
-- Create function to clean spatial data
CREATE OR REPLACE FUNCTION clean_spatial_data(
    geom GEOMETRY
)
RETURNS GEOMETRY AS $$
BEGIN
    -- Remove duplicate points
    geom := ST_RemoveRepeatedPoints(geom);
    
    -- Simplify geometry if too complex
    IF ST_NPoints(geom) > 1000 THEN
        geom := ST_Simplify(geom, 0.0001);
    END IF;
    
    -- Fix invalid geometries
    IF NOT ST_IsValid(geom) THEN
        geom := ST_MakeValid(geom);
    END IF;
    
    -- Remove empty geometries
    IF ST_IsEmpty(geom) THEN
        RETURN NULL;
    END IF;
    
    RETURN geom;
END;
$$ LANGUAGE plpgsql;
```

## Spatial Performance Monitoring

### Spatial Query Performance

```python
# monitoring/spatial_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class SpatialMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_spatial_index_usage(self):
        """Get spatial index usage statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    WHERE indexname LIKE '%_geom_gist%' OR indexname LIKE '%_location_gist%'
                    ORDER BY idx_scan DESC
                """)
                
                spatial_indexes = cur.fetchall()
                return spatial_indexes
                
        except Exception as e:
            self.logger.error(f"Error getting spatial index usage: {e}")
            return []
        finally:
            conn.close()
    
    def analyze_spatial_query_performance(self):
        """Analyze spatial query performance."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Test point-in-polygon query performance
                start_time = datetime.now()
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM cities c
                    JOIN administrative_boundaries ab ON ST_Contains(ab.geom, c.geom)
                """)
                pip_time = (datetime.now() - start_time).total_seconds()
                
                # Test distance query performance
                start_time = datetime.now()
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM global_sensors s1
                    CROSS JOIN global_sensors s2
                    WHERE ST_DWithin(s1.location, s2.location, 1000)
                """)
                distance_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'point_in_polygon_time': pip_time,
                    'distance_query_time': distance_time
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing spatial query performance: {e}")
            return {}
        finally:
            conn.close()
    
    def get_spatial_data_statistics(self):
        """Get spatial data statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE column_name LIKE '%geom%' OR column_name LIKE '%location%'
                    ORDER BY table_name, column_name
                """)
                
                spatial_columns = cur.fetchall()
                return spatial_columns
                
        except Exception as e:
            self.logger.error(f"Error getting spatial data statistics: {e}")
            return []
        finally:
            conn.close()
    
    def generate_spatial_report(self):
        """Generate comprehensive spatial data report."""
        spatial_indexes = self.get_spatial_index_usage()
        query_performance = self.analyze_spatial_query_performance()
        spatial_columns = self.get_spatial_data_statistics()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'spatial_indexes': spatial_indexes,
            'query_performance': query_performance,
            'spatial_columns': spatial_columns
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = SpatialMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_spatial_report()
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. Create spatial tables
CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    geom GEOMETRY(POINT, 4326)
);

-- 3. Create spatial indexes
CREATE INDEX idx_cities_geom_gist ON cities USING GIST (geom);

-- 4. Insert spatial data
INSERT INTO cities (name, geom) VALUES
    ('New York', ST_GeomFromText('POINT(-74.0059 40.7128)', 4326));

-- 5. Query spatial data
SELECT name, ST_AsText(geom) as geometry
FROM cities
WHERE ST_DWithin(geom, ST_GeomFromText('POINT(-74.0059 40.7128)', 4326), 1000);
```

### Essential Patterns

```python
# Complete PostGIS spatial data setup
def setup_postgis_spatial_data():
    # 1. PostGIS extension setup
    # 2. Spatial data types and storage
    # 3. Spatial indexing strategies
    # 4. Spatial query optimization
    # 5. Coordinate reference systems
    # 6. Spatial analysis functions
    # 7. Data import/export
    # 8. Spatial data quality and monitoring
    
    print("PostGIS spatial data setup complete!")
```

---

*This guide provides the complete machinery for PostGIS spatial data excellence. Each pattern includes implementation examples, spatial analysis strategies, and real-world usage patterns for enterprise PostGIS spatial systems.*
