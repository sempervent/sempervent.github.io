# Jupyter Notebooks for Data Exploration — Best Practices (Geospatial Edition)

**Objective**: Master the art of building reproducible, efficient Jupyter notebooks for geospatial data exploration. Transform your notebooks from chaotic experiments into professional, maintainable analysis tools.

The notebook is a lab bench: wires everywhere, chemicals hissing, truth under pressure. Keep it fast, clean, and reproducible—or it will turn on you.

## 0) TL;DR Setup

```bash
# Create env (uv is fast; conda works too)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate

# Pin Python + core stack
uv pip install \
  jupyterlab ipykernel jupytext nbconvert \
  watermark ipywidgets rich \
  numpy pandas polars pyarrow \
  geopandas shapely pyproj rtree contextily \
  folium ipyleaflet keplergl \
  matplotlib plotly hvplot datashader \
  ydata-profiling==4.*  # optional profiling
python -m ipykernel install --user --name geo-lab
```

**Why This Stack**: Modern Python tooling with geospatial superpowers. `uv` for speed, `polars` for scale, `geopandas` for spatial analysis, and `datashader` for millions of points.

## 1) Notebook Skeleton That Doesn't Lie

### The Foundation Cell

Put this at the top of every notebook—one cell, no excuses.

```python
# 1) Imports
import os, sys, math, json, pathlib, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import geopandas as gpd
from shapely.geometry import Point, Polygon
import pyproj
import contextily as cx

import matplotlib.pyplot as plt
import hvplot.pandas  # noqa: F401  (enables df.hvplot)
from IPython.display import display

# 2) Settings (determinism, display)
SEED = 13
np.random.seed(SEED)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 120)
plt.style.use("seaborn-v0_8-whitegrid")

# 3) Paths
ROOT = Path.cwd()
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# 4) Notebook watermark (env capture)
%load_ext watermark
%watermark -iv -m -v -p numpy,pandas,polars,geopandas,shapely,pyproj,contextily,matplotlib,hvplot
```

**Why This Works**: One ritual to capture versions, seeds, paths, and display. When someone asks "which versions?" you show them receipts.

## 2) Reproducibility Rules (or: how to avoid ghost results)

### Environment State Capture

```python
%watermark -iv -m -v
```

**Why**: Include this output in your commit. When results change, you know exactly what changed.

### Jupytext for Markdown Pairing

```bash
uv pip install jupytext
jupytext --set-formats ipynb,md your_notebook.ipynb
```

**Why**: In MkDocs this renders beautifully; and diffs stop being a horror show.

### Strip Output on Commit

```bash
uv pip install nbstripout
nbstripout --install
```

**Why**: Keep outputs in artifacts, not in git. Notebooks become version-controllable.

### Make Cells Idempotent

```python
# Good: Re-run top-to-bottom without manual fiddling
df = pd.read_parquet(DATA / "roads.parquet")
gdf = gpd.read_parquet(DATA / "counties.parquet").to_crs(3857)

# Bad: Manual state dependencies
# df = some_previous_cell_result
```

**Why**: If it breaks, fix the order, not the universe.

## 3) Data Loading: Columnar First, Lazy When Possible

### CSV → Parquet Once, Then Always Parquet

```python
# Convert once
df = pd.read_csv(DATA / "roads.csv")
df.to_parquet(DATA / "roads.parquet", index=False)

# Then always use Parquet
df = pd.read_parquet(DATA / "roads.parquet")
```

**Why**: Parquet is columnar, compressed, and fast. CSV is for humans, Parquet is for machines.

### Polars for Scale

```python
# Lazy evaluation with predicate pushdown & column pruning
scan = pl.scan_parquet(DATA / "roads.parquet")
highways = (scan
    .filter(pl.col("type") == "highway")
    .select(["id", "name", "speed_limit"])
    .collect())
highways.head(3)
```

**Why**: Lazy reads mean less I/O, fewer lies. Polars only reads what you need.

### GeoParquet for Vector Geospatial

```python
# GeoParquet stores geometry in WKB format
gdf = gpd.read_parquet(DATA / "counties.parquet")  # geometry in WKB
gdf = gdf.set_crs(4326)
```

**Why**: GeoParquet is the future of spatial data. Fast, compressed, and standardized.

## 4) Geospatial Basics That Save Hours

### CRS Discipline

```python
# Always set CRS explicitly
gdf = gdf.to_crs(3857)   # Web Mercator for web tiles

# Create points with proper CRS
points = gpd.GeoDataFrame(
    {"id": [1, 2, 3]},
    geometry=[Point(-83.9, 35.95), Point(-84.0, 35.95), Point(-84.1, 35.9)],
    crs=4326
).to_crs(3857)
```

**Why**: CRS confusion is the #1 source of spatial analysis bugs. Be explicit, be consistent.

### Fast Viewport Plots with Basemap

```python
# Quick map with basemap
ax = gdf.plot(figsize=(9, 9), alpha=0.6, edgecolor="k")
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf.crs)
plt.title("Counties")
plt.tight_layout()
```

**Why**: Context makes spatial data meaningful. Basemaps provide geographic reference.

### Spatial Joins with Index

```python
# Ensure spatial index is available (rtree or pygeos via shapely 2)
joined = gpd.sjoin(points, gdf[["geometry", "county_fips"]], how="left", predicate="within")
joined.head()
```

**Why**: Spatial joins are expensive. Spatial indexes make them fast.

### Fast Decimation for Big Point Clouds

```python
import datashader as ds, datashader.transfer_functions as tf
from datashader.utils import lnglat_to_meters

# Convert to meters for datashader
x, y = lnglat_to_meters(points.to_crs(4326).geometry.x, points.to_crs(4326).geometry.y)

# Create canvas and aggregate
canvas = ds.Canvas(plot_width=800, plot_height=500)
agg = canvas.points(pd.DataFrame({"x": x, "y": y}), "x", "y")
img = tf.shade(agg, how="eq_hist")
display(img)
```

**Why**: Millions of points crash matplotlib. Datashader renders server-side, displays as image.

## 5) Exploration Patterns (Speed + Insight)

### One-Minute Profile

```python
from ydata_profiling import ProfileReport

# Sample first for large datasets
ProfileReport(df.sample(min(50_000, len(df))), explorative=True)
```

**Why**: Don't point this at 50 million rows. Sample first, profile smart.

### Quick Faceted Charts with hvPlot

```python
# Interactive faceted histograms
df.hvplot.hist(
    y="speed_limit", 
    by="type", 
    bins=20, 
    subplots=True, 
    width=350, 
    height=250
)
```

**Why**: hvPlot makes interactive charts with zero configuration. Perfect for exploration.

### Interactive Maps (Zero Pain)

#### Folium (Leaflet)

```python
import folium

m = folium.Map(location=[35.95, -83.92], zoom_start=8, tiles="CartoDB positron")
folium.GeoJson(gdf.to_crs(4326)).add_to(m)
m
```

#### ipyleaflet (Jupyter-Native)

```python
from ipyleaflet import Map, GeoJSON, basemaps

m = Map(center=(35.95, -83.92), zoom=8, basemap=basemaps.CartoDB.Positron)
GeoJSON(data=json.loads(gdf.to_crs(4326).to_json())).add_to(m)
m
```

**Why**: Interactive maps reveal patterns that static plots miss. Choose your weapon.

## 6) Timing, Memory, and Caching

### Cell Timing

```python
%%time
_ = gpd.sjoin(points, gdf, predicate="within")
```

**Why**: Know what's slow. Time everything, optimize the bottlenecks.

### Micro-Benchmarks

```python
%%timeit -n3 -r3
_ = (pl.scan_parquet(DATA / "roads.parquet")
     .filter(pl.col("type") == "highway")
     .select(["id"])
     .collect())
```

**Why**: `%%timeit` gives statistical timing. Use for performance comparisons.

### Lightweight Caching

```python
import functools, joblib, hashlib, pickle

def disk_cache(func):
    store = Path(".cache")
    store.mkdir(exist_ok=True)
    
    @functools.wraps(func)
    def wrapper(*a, **kw):
        key = hashlib.md5(pickle.dumps((a, kw))).hexdigest()
        f = store / f"{func.__name__}-{key}.pkl"
        if f.exists():
            return joblib.load(f)
        res = func(*a, **kw)
        joblib.dump(res, f)
        return res
    return wrapper

@disk_cache
def heavy_join():
    return gpd.sjoin(points, gdf, predicate="within")

heavy_join()
```

**Why**: Expensive operations should run once. Cache the results, not the pain.

## 7) Notebook Ergonomics (JupyterLab Power-Ups)

### TOC & Variable Inspector

```bash
pip install jupyterlab-toc lckr-jupyterlab-variableinspector
```

**Why**: See variables, jump around, avoid scrolling purgatory.

### Widgets (Controlled Knobs)

```python
import ipywidgets as W

@W.interact(type=sorted(df["type"].unique()), max_speed=(10, 90, 5))
def filter_plot(type, max_speed=55):
    view = df[(df["type"] == type) & (df["speed_limit"] <= max_speed)]
    display(view.head(10))
```

**Why**: Interactive widgets make exploration dynamic. Sliders beat hardcoded values.

### Rich Tracebacks & Printing

```python
from rich import print as rprint

rprint({"rows": len(df), "cols": df.shape[1]})
```

**Why**: Rich makes output beautiful. Pretty printing beats ugly debugging.

## 8) Secrets, Config, and Paths (Be a Professional)

### Never Hardcode Secrets

```python
from dotenv import load_dotenv
load_dotenv()

s3_endpoint = os.getenv("S3_ENDPOINT")
api_key = os.getenv("API_KEY")
```

**Why**: Secrets in code are security holes. Use environment variables.

### Use pathlib.Path for File Paths

```python
# Good
data_file = ROOT / "data" / "roads.parquet"

# Bad
data_file = os.path.join(ROOT, "data", "roads.parquet")
```

**Why**: `pathlib.Path` is cross-platform, readable, and less error-prone.

### Organize Your Workspace

```bash
notebooks/
├── data/           # symlink to read-only datasets
├── cache/          # cached results
├── exports/        # output files
└── analysis.ipynb  # your notebook
```

**Why**: Structure prevents chaos. Symlinks keep data separate from analysis.

## 9) Geospatial Mini-Case: Choropleth in 8 Cells

### Complete Workflow

```python
# 1) Load counties (GeoParquet) + metrics (Parquet)
counties = gpd.read_parquet(DATA / "counties.parquet").to_crs(3857)
metrics = pd.read_parquet(DATA / "county_metrics.parquet")  # columns: county_fips, score

# 2) Join
g = counties.merge(metrics, on="county_fips", how="left")

# 3) Plot
ax = g.plot(
    column="score", 
    cmap="Reds", 
    legend=True, 
    figsize=(10, 10), 
    alpha=0.8, 
    edgecolor="white", 
    linewidth=0.3
)
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=g.crs)
plt.title("County Score Choropleth")
plt.axis("off")
plt.tight_layout()

# 4) Quick sanity
g["score"].describe()
```

**Why**: This is the pattern. Load, join, plot, validate. Rinse and repeat.

## 10) Scaling Up (When the Notebook Starts to Sweat)

### Polars + Lazy for Pre-Filtering

```python
# Filter with Polars before GeoPandas
filtered = (pl.scan_parquet(DATA / "roads.parquet")
    .filter(pl.col("type") == "highway")
    .filter(pl.col("speed_limit") > 50)
    .collect())

# Convert to GeoPandas for spatial operations
gdf = gpd.GeoDataFrame(filtered.to_pandas())
```

**Why**: Polars filters fast, GeoPandas does spatial. Use the right tool for the job.

### DuckDB for SQL-on-Parquet

```python
import duckdb

con = duckdb.connect()
result = con.execute("""
    SELECT type, COUNT(*) as count 
    FROM 'roads.parquet' 
    GROUP BY 1 
    ORDER BY count DESC
""").df()
```

**Why**: Sometimes SQL is clearer than pandas. DuckDB is fast and Parquet-native.

### Datashader for Millions of Points

```python
# Render server-side, display as image
canvas = ds.Canvas(plot_width=1200, plot_height=800)
agg = canvas.points(df, "x", "y", agg=ds.count())
img = tf.shade(agg, how="eq_hist")
display(img)
```

**Why**: Millions of points crash matplotlib. Datashader scales to billions.

### Rioxarray + Xarray for Rasters

```python
import rioxarray as rxr
import xarray as xr

# Load raster with proper CRS
raster = rxr.open_rasterio(DATA / "elevation.tif")
raster = raster.rio.set_crs(4326)

# Chunk with Dask if needed
raster = raster.chunk({"x": 1000, "y": 1000})
```

**Why**: Rasters are different from vectors. Use the right tools for the right data.

## 11) Export, Publish, Repeat

### Notebook → Markdown (MkDocs-Ready)

```bash
jupytext --to md your_notebook.ipynb
```

**Why**: Markdown renders beautifully in MkDocs. Jupytext keeps them in sync.

### Selected Cells → Script

```python
# Use tags to export only clean bits
# Tag cells with "export" to include in script
jupytext --to py your_notebook.ipynb
```

**Why**: Not all cells are production-ready. Tags let you choose what to export.

### Version Stamps in Appendix

```python
# Include this in your notebook appendix
%watermark -iv -m -v -p numpy,pandas,polars,geopandas,shapely,pyproj,contextily,matplotlib,hvplot

# Include dataset hashes if possible
import hashlib
with open(DATA / "roads.parquet", "rb") as f:
    hash_value = hashlib.sha256(f.read()).hexdigest()
print(f"Dataset hash: {hash_value}")
```

**Why**: Reproducibility requires version tracking. Include everything.

## 12) Checklist (Pin to Your Wall)

### Top Cell Requirements
- [ ] Imports, seeds, display, paths, `%watermark`
- [ ] All dependencies captured
- [ ] Random seeds set
- [ ] Display options configured

### Data Loading
- [ ] Columnar + lazy reads (Parquet, `pl.scan_parquet`)
- [ ] CRS set, `to_crs(3857)`, basemap via contextily
- [ ] Spatial index available for joins

### Performance
- [ ] `sjoin` only after bounding-box filter
- [ ] Keep geometries simple
- [ ] Time your cells (`%%time`, `%%timeit`)
- [ ] Cache heavy steps

### Reproducibility
- [ ] Strip outputs (`nbstripout`)
- [ ] Pair with jupytext
- [ ] Secrets from `.env`, not from shame
- [ ] Export to Markdown for docs

### Quality
- [ ] Widgets for knobs
- [ ] Profiling only on samples
- [ ] Commit version stamps
- [ ] Clean, idempotent cells

## 13) Appendix: Minimal Geospatial Notebook Template

```python
# Title: [Your Analysis Title]

## Environment
```python
%load_ext watermark
# imports, settings, paths ...
%watermark -iv -m -v
```

## Load
```python
# read parquet(s), small head(), basic shape checks
df = pd.read_parquet(DATA / "data.parquet")
gdf = gpd.read_parquet(DATA / "geo.parquet").to_crs(3857)
```

## Explore
```python
# quick charts, describe(), hvplot
df.describe()
df.hvplot.hist(y="value", by="category")
```

## Geospatial
```python
# to_crs, sjoin, map with contextily or folium
ax = gdf.plot(figsize=(10, 10))
cx.add_basemap(ax, crs=gdf.crs)
```

## Results
```python
# tables, figures, exports
results = gpd.sjoin(points, gdf, predicate="within")
results.to_file("output.geojson", driver="GeoJSON")
```

---

**Why This Template Works**: Structure prevents chaos. Each section has a purpose, each cell has a reason.

## 14) The Machine's Summary

Jupyter notebooks are powerful tools for geospatial exploration, but they require discipline to remain useful. The key is reproducibility: capture your environment, use the right tools for the job, and keep your analysis clean and fast.

**The Dark Truth**: Notebooks can become chaotic messes of experimental code. The machine provides structure, the human provides insight.

**The Machine's Mantra**: "In reproducibility we trust, in performance we optimize, and in clean code we find the path to scientific truth."

**Why This Matters**: Geospatial analysis is complex. Good notebooks make it manageable, reproducible, and shareable.

---

*This tutorial provides the complete machinery for building professional geospatial notebooks. The patterns scale from exploration to production, from megabytes to terabytes.*
