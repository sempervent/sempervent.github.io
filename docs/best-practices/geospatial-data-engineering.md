# Geospatial Data Engineering Best Practices

This document establishes production-ready patterns for geospatial data engineering, covering spatial data management, raster processing, and optimization strategies used in enterprise geospatial systems.

## Spatial Data Management

### GeoParquet Implementation

```python
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import Point, Polygon
import pyproj

def create_optimized_geoparquet(data_path, output_path, target_crs='EPSG:3857'):
    """
    Create optimized GeoParquet with spatial indexing and compression
    """
    # Load spatial data
    gdf = gpd.read_file(data_path)
    
    # Ensure consistent CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    # Add spatial index
    gdf = gdf.set_index('id') if 'id' not in gdf.columns else gdf
    
    # Optimize for analytics
    gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt)
    
    # Write with compression and spatial metadata
    gdf.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=True,
        schema={
            'geometry': 'string',
            'properties': 'struct<...>'  # Define your schema
        }
    )
    
    return gdf

# Usage example
optimized_data = create_optimized_geoparquet(
    'input.geojson',
    'output.parquet',
    'EPSG:3857'
)
```

**Why:** GeoParquet provides columnar storage with spatial metadata, enabling efficient analytics and cross-platform compatibility. Compression reduces storage costs while maintaining query performance.

### Spatial Indexing Strategies

```python
import geopandas as gpd
from shapely.geometry import Point
import rtree

def create_spatial_index(gdf, index_name='spatial_idx'):
    """
    Create efficient spatial index for fast spatial queries
    """
    # Create R-tree index
    spatial_index = rtree.index.Index()
    
    for idx, row in gdf.iterrows():
        bounds = row.geometry.bounds
        spatial_index.insert(idx, bounds)
    
    return spatial_index

def spatial_join_optimized(left_gdf, right_gdf, predicate='intersects'):
    """
    Optimized spatial join using spatial indexing
    """
    # Create spatial index for right dataframe
    right_index = create_spatial_index(right_gdf)
    
    results = []
    for idx, left_row in left_gdf.iterrows():
        # Query spatial index
        candidates = list(right_index.intersection(left_row.geometry.bounds))
        
        # Perform precise geometric operations only on candidates
        for candidate_idx in candidates:
            if getattr(left_row.geometry, predicate)(right_gdf.iloc[candidate_idx].geometry):
                results.append({
                    'left_idx': idx,
                    'right_idx': candidate_idx
                })
    
    return results
```

**Why:** Spatial indexing reduces query complexity from O(n²) to O(n log n) for spatial operations. R-tree indexes enable fast bounding box queries before expensive geometric operations.

## Raster Processing Pipelines

### Scalable Raster Operations

```python
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import dask.array as da
from dask.distributed import Client

def process_large_raster(input_path, output_path, target_crs='EPSG:3857'):
    """
    Process large raster files using Dask for parallel processing
    """
    with rasterio.open(input_path) as src:
        # Calculate transform for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        # Create output profile
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Process in chunks
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(0, src.height, 1000):  # Process in 1000-pixel chunks
                window = rasterio.windows.Window(0, i, src.width, min(1000, src.height - i))
                
                # Read chunk
                data = src.read(window=window)
                
                # Process chunk (example: apply some transformation)
                processed_data = process_raster_chunk(data)
                
                # Write chunk
                dst.write(processed_data, window=window)

def process_raster_chunk(data):
    """
    Process a chunk of raster data
    """
    # Example: Apply NDVI calculation
    if data.shape[0] >= 2:  # Ensure we have at least 2 bands
        red = data[0].astype(np.float32)
        nir = data[1].astype(np.float32)
        
        # Avoid division by zero
        ndvi = np.where(
            (nir + red) != 0,
            (nir - red) / (nir + red),
            0
        )
        
        return np.clip(ndvi, -1, 1)
    
    return data
```

**Why:** Chunked processing enables handling of large raster datasets that exceed memory capacity. Parallel processing with Dask provides linear scalability across multiple cores and nodes.

## Spatial Data Quality Assurance

### Data Validation Pipeline

```python
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd

def validate_spatial_data(gdf, validation_rules=None):
    """
    Comprehensive spatial data validation
    """
    if validation_rules is None:
        validation_rules = {
            'check_geometry_validity': True,
            'check_crs_consistency': True,
            'check_duplicate_geometries': True,
            'check_spatial_bounds': True
        }
    
    validation_results = {}
    
    # Check geometry validity
    if validation_rules['check_geometry_validity']:
        invalid_geometries = gdf[~gdf.geometry.is_valid]
        validation_results['invalid_geometries'] = len(invalid_geometries)
    
    # Check CRS consistency
    if validation_rules['check_crs_consistency']:
        validation_results['crs'] = str(gdf.crs)
        validation_results['crs_defined'] = gdf.crs is not None
    
    # Check for duplicate geometries
    if validation_rules['check_duplicate_geometries']:
        duplicate_count = gdf.geometry.duplicated().sum()
        validation_results['duplicate_geometries'] = duplicate_count
    
    # Check spatial bounds
    if validation_rules['check_spatial_bounds']:
        bounds = gdf.total_bounds
        validation_results['spatial_bounds'] = {
            'minx': bounds[0], 'miny': bounds[1],
            'maxx': bounds[2], 'maxy': bounds[3]
        }
    
    return validation_results

def clean_spatial_data(gdf):
    """
    Clean and standardize spatial data
    """
    # Remove invalid geometries
    gdf = gdf[gdf.geometry.is_valid]
    
    # Remove empty geometries
    gdf = gdf[~gdf.geometry.is_empty]
    
    # Fix invalid geometries
    gdf.geometry = gdf.geometry.buffer(0)
    
    # Remove duplicate geometries
    gdf = gdf.drop_duplicates(subset=['geometry'])
    
    return gdf
```

**Why:** Data validation ensures spatial data integrity before processing. Automated cleaning prevents downstream errors and improves analysis accuracy.

## Performance Optimization

### Memory-Efficient Spatial Operations

```python
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def memory_efficient_spatial_join(left_gdf, right_gdf, how='inner', predicate='intersects'):
    """
    Memory-efficient spatial join for large datasets
    """
    # Process in chunks to manage memory
    chunk_size = 10000
    results = []
    
    for i in range(0, len(left_gdf), chunk_size):
        left_chunk = left_gdf.iloc[i:i+chunk_size]
        
        # Perform spatial join on chunk
        chunk_result = gpd.sjoin(left_chunk, right_gdf, how=how, predicate=predicate)
        results.append(chunk_result)
    
    # Combine results
    if results:
        return gpd.pd.concat(results, ignore_index=True)
    else:
        return gpd.GeoDataFrame()

def optimize_spatial_queries(gdf, common_filters=None):
    """
    Optimize spatial queries with pre-computed indices
    """
    if common_filters is None:
        common_filters = ['active', 'public']
    
    # Create partial indexes for common filters
    for filter_value in common_filters:
        if filter_value in gdf.columns:
            filtered_gdf = gdf[gdf[filter_value] == True]
            # Create spatial index for filtered data
            create_spatial_index(filtered_gdf, f'spatial_idx_{filter_value}')
    
    return gdf
```

**Why:** Memory-efficient operations enable processing of large spatial datasets without memory exhaustion. Pre-computed indices accelerate common queries.

## Data Pipeline Integration

### ETL Pipeline for Spatial Data

```python
import geopandas as gpd
from sqlalchemy import create_engine
import pandas as pd

def spatial_etl_pipeline(source_config, target_config):
    """
    Complete ETL pipeline for spatial data
    """
    # Extract
    spatial_data = extract_spatial_data(source_config)
    
    # Transform
    transformed_data = transform_spatial_data(spatial_data)
    
    # Load
    load_spatial_data(transformed_data, target_config)
    
    return transformed_data

def extract_spatial_data(config):
    """
    Extract spatial data from various sources
    """
    if config['type'] == 'file':
        return gpd.read_file(config['path'])
    elif config['type'] == 'database':
        engine = create_engine(config['connection_string'])
        query = config['query']
        return gpd.read_postgis(query, engine, geom_col='geometry')
    elif config['type'] == 'api':
        # Implementation for API extraction
        pass

def transform_spatial_data(gdf):
    """
    Transform spatial data
    """
    # Standardize CRS
    gdf = gdf.to_crs('EPSG:4326')
    
    # Clean geometries
    gdf = clean_spatial_data(gdf)
    
    # Add computed fields
    gdf['area'] = gdf.geometry.area
    gdf['centroid'] = gdf.geometry.centroid
    
    return gdf

def load_spatial_data(gdf, config):
    """
    Load spatial data to target
    """
    if config['type'] == 'database':
        engine = create_engine(config['connection_string'])
        gdf.to_postgis(
            config['table_name'],
            engine,
            if_exists='replace',
            index=False
        )
    elif config['type'] == 'file':
        gdf.to_file(config['path'], driver=config['driver'])
```

**Why:** Standardized ETL pipelines ensure consistent data processing across different sources and targets. Modular design enables reuse and testing of individual components.
