# GeoParquet Data Warehouses

**Objective**: Build modern geospatial data warehouses using GeoParquet format for optimal performance and interoperability.

GeoParquet represents the future of geospatial data storage—columnar, compressed, and cross-platform compatible. This guide covers building production-ready geospatial data warehouses that scale.

## 1) GeoParquet Fundamentals

### Core Architecture

```python
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import Point
import pandas as pd

# Load and optimize geospatial data
gdf = gpd.read_file('data.geojson')
gdf = gdf.set_crs('EPSG:4326')

# Transform to optimal projection for analytics
gdf['geometry'] = gdf['geometry'].to_crs('EPSG:3857')

# Write optimized GeoParquet
gdf.to_parquet('optimized_data.parquet', engine='pyarrow')
```

**Why**: GeoParquet combines the performance of Parquet's columnar storage with native geospatial support, enabling fast analytical queries across multiple platforms.

### Schema Design Patterns

```python
# Define consistent schema
schema = {
    'id': 'int64',
    'name': 'string',
    'geometry': 'geometry',
    'properties': 'struct<key1:string,key2:double>',
    'created_at': 'timestamp',
    'updated_at': 'timestamp'
}

# Enforce schema on write
gdf = gdf.astype(schema)
```

**Why**: Consistent schemas prevent data quality issues and enable predictable query performance across different analytical engines.

## 2) Spatial Indexing Strategies

### R-Tree Indexing

```python
# Create spatial index for fast spatial queries
gdf = gdf.set_index('id')
gdf = gdf.set_geometry('geometry')

# Build spatial index
gdf.sindex  # Creates R-tree index automatically

# Optimize for common query patterns
gdf['bounds'] = gdf.bounds
gdf['centroid'] = gdf.centroid
```

**Why**: Spatial indexes dramatically improve query performance for spatial operations like intersections, contains, and distance calculations.

### Partitioning Strategies

```python
# Partition by spatial regions
def spatial_partition(geom):
    """Partition by spatial grid"""
    bounds = geom.bounds
    x_tile = int(bounds[0] // 1000)  # 1km tiles
    y_tile = int(bounds[1] // 1000)
    return f"x{x_tile}_y{y_tile}"

gdf['spatial_partition'] = gdf.geometry.apply(spatial_partition)

# Write partitioned data
gdf.to_parquet('partitioned_data.parquet', partition_cols=['spatial_partition'])
```

**Why**: Spatial partitioning enables parallel processing and reduces I/O for spatial queries by only reading relevant partitions.

## 3) Performance Optimization

### Compression and Encoding

```python
# Optimize compression for geospatial data
compression_config = {
    'compression': 'snappy',  # Fast compression
    'use_dictionary': True,  # Dictionary encoding for categorical data
    'use_statistics': True,  # Column statistics for predicate pushdown
}

gdf.to_parquet(
    'optimized.parquet',
    engine='pyarrow',
    compression='snappy',
    use_dictionary=True
)
```

**Why**: Proper compression reduces storage costs and improves query performance by reducing I/O bandwidth requirements.

### Column Pruning

```python
# Design for column pruning
essential_columns = ['id', 'geometry', 'name']
analytical_columns = ['population', 'area', 'density']
metadata_columns = ['created_at', 'updated_at', 'source']

# Structure data for efficient column access
gdf = gdf[essential_columns + analytical_columns + metadata_columns]
```

**Why**: Column pruning allows analytical engines to read only the columns needed for specific queries, dramatically improving performance.

## 4) Cross-Platform Compatibility

### Standardized Metadata

```python
# Ensure GeoParquet compliance
import geoparquet

# Validate GeoParquet format
geoparquet.validate_geoparquet('data.parquet')

# Add required metadata
metadata = {
    'geo': {
        'version': '1.0.0',
        'primary_column': 'geometry',
        'columns': {
            'geometry': {
                'encoding': 'WKB',
                'crs': 'EPSG:3857'
            }
        }
    }
}
```

**Why**: Standardized metadata ensures compatibility across different geospatial libraries and analytical engines.

### Multi-Engine Support

```python
# Test compatibility across engines
engines = ['pyarrow', 'fastparquet']

for engine in engines:
    df = pd.read_parquet('data.parquet', engine=engine)
    print(f"{engine}: {len(df)} records loaded")
```

**Why**: Multi-engine support ensures your data warehouse works with different analytical tools and cloud platforms.

## 5) Data Quality and Validation

### Geometry Validation

```python
# Validate geometries before storage
def validate_geometry(geom):
    """Validate and fix geometry issues"""
    if not geom.is_valid:
        geom = geom.buffer(0)  # Fix self-intersections
    return geom

gdf['geometry'] = gdf['geometry'].apply(validate_geometry)

# Check for invalid geometries
invalid_count = (~gdf['geometry'].is_valid).sum()
print(f"Invalid geometries: {invalid_count}")
```

**Why**: Invalid geometries cause query failures and performance issues. Validation ensures data quality and reliability.

### Schema Evolution

```python
# Handle schema evolution gracefully
def migrate_schema(df, target_schema):
    """Migrate DataFrame to target schema"""
    for col, dtype in target_schema.items():
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(dtype)
    return df

# Apply schema migration
gdf = migrate_schema(gdf, schema)
```

**Why**: Schema evolution allows your data warehouse to adapt to changing requirements without breaking existing queries.

## 6) Production Deployment

### Cloud Storage Optimization

```python
# Optimize for cloud storage
def optimize_for_cloud(df, target_size_mb=128):
    """Optimize DataFrame for cloud storage"""
    current_size = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    if current_size > target_size_mb:
        # Split into smaller chunks
        chunk_size = len(df) // (current_size // target_size_mb + 1)
        return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    return [df]

chunks = optimize_for_cloud(gdf, target_size_mb=128)
```

**Why**: Cloud storage costs scale with file size and access patterns. Optimizing chunk sizes improves performance and reduces costs.

### Monitoring and Alerting

```python
# Monitor data warehouse health
def monitor_warehouse_health(parquet_path):
    """Monitor GeoParquet warehouse health"""
    import os
    
    stats = {
        'file_size_mb': os.path.getsize(parquet_path) / 1024 / 1024,
        'record_count': len(pd.read_parquet(parquet_path)),
        'last_modified': os.path.getmtime(parquet_path)
    }
    
    # Alert on anomalies
    if stats['file_size_mb'] > 1000:  # > 1GB
        print("WARNING: Large file detected")
    
    return stats
```

**Why**: Monitoring ensures your data warehouse remains performant and reliable as it scales.

## 7) Advanced Patterns

### Time-Series Geospatial Data

```python
# Handle time-series geospatial data
def create_temporal_partition(df, time_col='timestamp'):
    """Create temporal partitions for time-series data"""
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    
    return df

# Partition by time and space
gdf = create_temporal_partition(gdf)
gdf.to_parquet(
    'temporal_spatial.parquet',
    partition_cols=['year', 'month', 'spatial_partition']
)
```

**Why**: Temporal partitioning enables efficient time-range queries and supports common analytical patterns like trend analysis.

### Multi-Resolution Data

```python
# Create multi-resolution datasets
def create_resolution_levels(gdf, levels=[0.1, 0.5, 1.0]):
    """Create multiple resolution levels"""
    results = {}
    
    for level in levels:
        simplified = gdf.copy()
        simplified['geometry'] = simplified['geometry'].simplify(level)
        results[f'level_{level}'] = simplified
    
    return results

# Store multiple resolutions
resolution_levels = create_resolution_levels(gdf)
for level, data in resolution_levels.items():
    data.to_parquet(f'data_{level}.parquet')
```

**Why**: Multi-resolution data supports different use cases—high resolution for detailed analysis, low resolution for overview visualizations.

## 8) TL;DR Quickstart

```python
# 1. Load and optimize geospatial data
gdf = gpd.read_file('data.geojson')
gdf = gdf.set_crs('EPSG:4326')
gdf['geometry'] = gdf['geometry'].to_crs('EPSG:3857')

# 2. Add spatial partitioning
gdf['spatial_partition'] = gdf.geometry.apply(spatial_partition)

# 3. Write optimized GeoParquet
gdf.to_parquet(
    'optimized.parquet',
    engine='pyarrow',
    compression='snappy',
    partition_cols=['spatial_partition']
)

# 4. Validate and monitor
geoparquet.validate_geoparquet('optimized.parquet')
stats = monitor_warehouse_health('optimized.parquet')
```

## 9) Anti-Patterns to Avoid

- **Don't store geometries in WKT format**—use binary encoding for performance
- **Don't ignore coordinate reference systems**—always specify and validate CRS
- **Don't store large geometries without simplification**—use appropriate resolution levels
- **Don't skip validation**—validate geometries before storage
- **Don't ignore partitioning**—spatial partitioning is essential for performance

**Why**: These anti-patterns lead to poor performance, data quality issues, and compatibility problems across different analytical engines.

---

*This guide provides the foundation for building production-ready GeoParquet data warehouses that scale with your analytical needs.*
