# Storing, Using, and Optimizing GeoParquet with Polars

This tutorial establishes the definitive approach to GeoParquet storage and querying with Polars for production-scale geospatial datasets. GeoParquet is the future of vector storage—columnar, efficient, cloud-native. Polars is a fast DataFrame engine that can exploit these structures without the memory overhead of traditional geospatial tools.

**Why this matters:** Traditional shapefiles and GeoJSON are dead weight in the cloud era. GeoParquet provides columnar storage with spatial metadata, while Polars delivers the performance needed for billion-point datasets without grinding your infrastructure into dust.

## 1. Writing GeoParquet with Polars

### Convert geometry and write to GeoParquet

```python
import polars as pl
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pyarrow as pa
import pyarrow.parquet as pq

# Create sample geospatial data
data = {
    "id": [1, 2, 3, 4, 5],
    "name": ["Point A", "Point B", "Point C", "Point D", "Point E"],
    "geometry": [
        Point(0, 0),
        Point(1, 1), 
        Point(2, 2),
        Point(3, 3),
        Point(4, 4)
    ],
    "category": ["A", "B", "A", "C", "B"],
    "value": [10.5, 20.3, 15.7, 8.9, 25.1]
}

# Convert to GeoPandas for geometry handling
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

# Convert to Polars with WKB geometry
df = pl.DataFrame({
    "id": gdf["id"],
    "name": gdf["name"], 
    "geometry": gdf.geometry.to_wkb(),
    "category": gdf["category"],
    "value": gdf["value"]
})

# Write to GeoParquet with spatial metadata
df.write_parquet(
    "data.parquet",
    compression="zstd",
    row_group_size=128_000_000  # 128MB row groups
)

# Verify schema includes geometry metadata
print(df.schema)
```

**Why:** WKB (Well-Known Binary) geometry format provides maximum portability and performance. GeoParquet preserves spatial metadata while enabling columnar storage benefits. ZSTD compression reduces storage costs for archival data.

## 2. Reading GeoParquet with Polars

### Efficient column pruning and lazy loading

```python
# Read with column pruning - only load what you need
df = pl.read_parquet(
    "data.parquet", 
    columns=["id", "geometry", "category"]
)

# Lazy loading for large datasets
df_lazy = pl.scan_parquet("data.parquet")

# Apply filters before materialization
filtered = (df_lazy
    .filter(pl.col("category") == "A")
    .select(["id", "geometry"])
    .collect()
)

print(f"Loaded {len(filtered)} rows with only required columns")
```

**Why:** Column pruning reduces I/O by 60-80% for wide tables. Lazy evaluation enables predicate pushdown and query optimization. Only materialize data when you need it.

## 3. Spatial Operations with polars_st

### Bounding box filtering and spatial predicates

```python
# Install: pip install polars-st
import polars_st as plst

# Create bounding box for spatial filtering
bbox = pl.DataFrame({
    "minx": [0.5],
    "miny": [0.5], 
    "maxx": [3.5],
    "maxy": [3.5]
})

# Spatial intersection query
spatial_query = (df_lazy
    .filter(pl.col("geometry").st_intersects(bbox))
    .select(["id", "name", "geometry"])
    .collect()
)

# Point-in-polygon test
point = pl.DataFrame({
    "x": [1.5],
    "y": [1.5]
})

pip_result = (df_lazy
    .filter(pl.col("geometry").st_contains(point))
    .collect()
)

# Distance-based filtering
distance_query = (df_lazy
    .filter(pl.col("geometry").st_dwithin(point, 2.0))
    .select(["id", "name"])
    .collect()
)
```

**Why:** polars_st provides native spatial operations optimized for Polars' execution engine. Spatial predicates push down to the storage layer, eliminating unnecessary data transfer. Distance queries enable efficient radius-based analysis.

## 4. Partitioning & Row Groups

### Hive-style partitioning for spatial data

```python
# Create partitioned dataset by spatial key
import os

# Partition by FIPS code (state-level partitioning)
partitioned_data = df.with_columns(
    pl.col("geometry").st_centroid().st_x().alias("centroid_x"),
    pl.col("geometry").st_centroid().st_y().alias("centroid_y")
).with_columns(
    pl.when(pl.col("centroid_x") < -100)
    .then(pl.lit("west"))
    .otherwise(pl.lit("east"))
    .alias("region")
)

# Write partitioned data
partitioned_data.write_parquet(
    "partitioned_data/",
    partition_by=["region"],
    compression="snappy"  # Faster for interactive queries
)

# Read from partitioned dataset
df_partitioned = pl.scan_parquet("partitioned_data/region=west/*.parquet")

# Partition pruning - only reads relevant partitions
filtered_partition = (df_partitioned
    .filter(pl.col("category") == "A")
    .collect()
)
```

**Why:** Spatial partitioning enables query pruning and parallel processing. Hive-style partitioning works with cloud storage and enables efficient directory-based filtering. Row group filtering reduces I/O for selective queries.

## 5. Optimization Practices

### Predicate pushdown and query optimization

```python
# Use scan_parquet for large datasets - enables predicate pushdown
large_df = pl.scan_parquet("large_dataset.parquet")

# Predicate pushdown - filters applied at storage layer
optimized_query = (large_df
    .filter(pl.col("road_type") == "highway")  # Pushed to storage
    .filter(pl.col("state") == "CA")            # Pushed to storage  
    .filter(pl.col("geometry").st_intersects(bbox))  # Spatial pushdown
    .select(["id", "geometry", "road_name"])    # Column pruning
    .collect()
)

# Compression tuning for different workloads
# Archival: ZSTD (better compression)
df.write_parquet(
    "archival.parquet",
    compression="zstd",
    compression_level=6,
    row_group_size=256_000_000  # 256MB for archival
)

# Interactive: Snappy (faster decompression)
df.write_parquet(
    "interactive.parquet", 
    compression="snappy",
    row_group_size=64_000_000   # 64MB for interactive
)
```

**Why:** Predicate pushdown eliminates unnecessary data transfer by filtering at the storage layer. Compression choice balances storage costs vs. query performance. Row group size affects both compression ratio and query granularity.

## 6. Cloud/Remote Storage

### S3-compatible storage with authentication

```python
# Configure S3 access
import os
os.environ["AWS_ACCESS_KEY_ID"] = "your_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Read from S3 with fsspec URLs
s3_df = pl.scan_parquet("s3://your-bucket/geospatial/roads.parquet")

# Partitioned S3 data with Hive partitioning
partitioned_s3 = pl.scan_parquet("s3://your-bucket/roads/year=2025/month=01/*.parquet")

# MinIO setup (local S3-compatible)
minio_df = pl.scan_parquet("s3://minio-bucket/data.parquet")

# Authentication via environment or config
# AWS credentials: ~/.aws/credentials
# MinIO: set endpoint via fsspec
```

**Why:** Cloud storage enables scalable geospatial data lakes. S3-compatible APIs work across AWS, MinIO, and other object stores. Partitioned storage enables efficient query pruning and cost optimization.

## 7. Benchmarks & Scaling Notes

### Performance comparison and scaling characteristics

```python
# Benchmark: Polars vs Pandas on GeoParquet
import time
import pandas as pd

# Large dataset benchmark
def benchmark_read():
    start = time.time()
    
    # Polars lazy evaluation
    polars_result = (pl.scan_parquet("large_dataset.parquet")
        .filter(pl.col("category") == "A")
        .select(["id", "geometry"])
        .collect()
    )
    
    polars_time = time.time() - start
    
    # Pandas equivalent (materializes everything)
    start = time.time()
    pandas_df = pd.read_parquet("large_dataset.parquet")
    pandas_result = pandas_df[pandas_df["category"] == "A"][["id", "geometry"]]
    pandas_time = time.time() - start
    
    print(f"Polars: {polars_time:.2f}s, Pandas: {pandas_time:.2f}s")
    print(f"Speedup: {pandas_time/polars_time:.1f}x")

# Memory usage comparison
def memory_usage():
    # Polars lazy - minimal memory
    lazy_df = pl.scan_parquet("large_dataset.parquet")
    print(f"Lazy DataFrame memory: {lazy_df.estimated_size()}")
    
    # Pandas - materializes everything
    pandas_df = pd.read_parquet("large_dataset.parquet")
    print(f"Pandas memory: {pandas_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

**Why:** Polars lazy evaluation provides 3-10x speedup over Pandas for large datasets. Memory usage is dramatically lower due to columnar storage and lazy evaluation. Geometry operations are still CPU-bound, but Polars reduces I/O bottlenecks.

## 8. Advanced Spatial Patterns

### Complex spatial analysis workflows

```python
# Multi-layer spatial joins
roads_df = pl.scan_parquet("s3://data/roads.parquet")
buildings_df = pl.scan_parquet("s3://data/buildings.parquet")

# Spatial join with distance threshold
spatial_join = (roads_df
    .join(
        buildings_df,
        on=pl.col("geometry").st_dwithin(pl.col("geometry"), 100),  # 100m buffer
        how="inner"
    )
    .select(["road_id", "building_id", "distance"])
    .collect()
)

# Aggregated spatial statistics
spatial_stats = (roads_df
    .group_by("road_type")
    .agg([
        pl.col("geometry").st_length().sum().alias("total_length"),
        pl.col("geometry").st_area().sum().alias("total_area"),
        pl.count().alias("road_count")
    ])
    .collect()
)

# Temporal spatial analysis
temporal_df = pl.scan_parquet("s3://data/temporal/*.parquet")

# Time-based spatial filtering
recent_data = (temporal_df
    .filter(pl.col("timestamp") >= pl.lit("2025-01-01"))
    .filter(pl.col("geometry").st_intersects(bbox))
    .group_by("date")
    .agg([pl.count().alias("daily_count")])
    .collect()
)
```

**Why:** Complex spatial workflows benefit from Polars' query optimization. Multi-layer joins enable sophisticated spatial analysis. Temporal patterns require efficient time-based filtering combined with spatial predicates.

## 9. Production Deployment

### Large-scale deployment considerations

```python
# Production configuration for billion-point datasets
def production_setup():
    # Configure Polars for large datasets
    pl.Config.set_streaming_chunk_size(1_000_000)  # 1M row chunks
    pl.Config.set_fmt_str_lengths(50)  # Truncate long strings
    
    # Memory management
    pl.Config.set_verbose(True)  # Enable query optimization logging
    
    # Cloud storage optimization
    storage_options = {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": "us-east-1"
    }
    
    # Read with optimized settings
    df = pl.scan_parquet(
        "s3://production-bucket/geospatial/",
        storage_options=storage_options
    )
    
    return df

# Monitoring and debugging
def debug_query_plan():
    # Show query execution plan
    lazy_df = pl.scan_parquet("large_dataset.parquet")
    optimized = (lazy_df
        .filter(pl.col("category") == "A")
        .select(["id", "geometry"])
    )
    
    # Print execution plan
    print(optimized.explain())
    
    # Profile query performance
    import cProfile
    cProfile.run("optimized.collect()")
```

**Why:** Production deployment requires careful memory management and query optimization. Streaming chunk size affects memory usage vs. performance tradeoffs. Query planning reveals optimization opportunities.

## 10. TL;DR Checklist

- **Always use scan_parquet for big GeoParquet** - enables lazy evaluation and predicate pushdown
- **Partition by spatial key (e.g., FIPS, tile ID)** - enables query pruning and parallel processing  
- **Prune columns aggressively (columns=[…])** - reduces I/O by 60-80% for wide tables
- **Tune row group size for workload** - 64MB for interactive, 256MB for archival
- **Use lazy queries for predicate pushdown** - filters applied at storage layer
- **Store geometry in WKB for max portability** - binary format enables efficient operations
- **Choose compression wisely** - Snappy for interactive, ZSTD for archival
- **Monitor memory usage** - streaming chunks prevent OOM on large datasets
- **Profile query plans** - identify optimization opportunities with explain()
- **Use cloud storage partitioning** - enables cost-effective data lakes

**Why:** This checklist ensures optimal GeoParquet performance with Polars. Each item addresses a critical aspect of large-scale geospatial data processing and prevents common performance pitfalls.

## 11. Anti-Patterns to Avoid

### Common mistakes that kill performance

```python
# Bad: Materializing everything upfront
df = pl.read_parquet("huge_dataset.parquet")  # Loads entire dataset
result = df.filter(pl.col("category") == "A")

# Good: Lazy evaluation with predicate pushdown
result = (pl.scan_parquet("huge_dataset.parquet")
    .filter(pl.col("category") == "A")
    .collect()
)

# Bad: No column pruning
df = pl.read_parquet("wide_dataset.parquet")  # Loads all columns
result = df.select(["id", "geometry"])

# Good: Column pruning
result = pl.read_parquet("wide_dataset.parquet", columns=["id", "geometry"])

# Bad: No spatial partitioning
df = pl.scan_parquet("unpartitioned_data.parquet")

# Good: Spatial partitioning
df = pl.scan_parquet("partitioned_data/region=west/*.parquet")
```

**Why:** These anti-patterns cause memory exhaustion and slow queries. Lazy evaluation and column pruning are essential for large datasets. Spatial partitioning enables efficient query processing.

The combination of GeoParquet and Polars eliminates the traditional bottlenecks of geospatial data processing. Your billion-point datasets deserve better than memory-bound tools that grind to a halt.
