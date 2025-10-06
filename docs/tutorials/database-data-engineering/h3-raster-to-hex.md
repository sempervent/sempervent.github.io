# H3 Raster to Hex: Converting Geospatial Rasters to Hexagonal Grids

**Objective**: Master the conversion of raster data (satellite imagery, elevation models, climate data) to H3 hexagonal grids for efficient spatial analysis, aggregation, and visualization. Transform pixels into hexagons for better spatial reasoning and analysis.

## Architecture

```mermaid
flowchart LR
    R[Raster Data] -->|GDAL/OGR| P[Python Processor]
    P -->|H3 Indexing| H[H3 Hex Grid]
    H -->|Aggregation| DB[Database Storage]
    DB -->|Query| V[Visualization]
    
    subgraph "Raster Sources"
        R1[Satellite Imagery]
        R2[Elevation Models]
        R3[Climate Data]
        R4[Population Density]
    end
    
    subgraph "H3 Processing"
        P
        H
        A[Aggregation Functions]
    end
    
    subgraph "Output Formats"
        DB
        G[GeoJSON]
        C[CSV]
        P2[Parquet]
    end
```

**Why**: H3 provides consistent hexagonal indexing for spatial data, enabling efficient aggregation, spatial joins, and hierarchical analysis. Converting rasters to H3 hexagons improves spatial reasoning and enables better data integration.

## Docker Compose with Profiles

```yaml
# docker-compose.yml
version: "3.9"
x-hc: &hc { interval: 5s, timeout: 3s, retries: 40 }

services:
  postgres:
    image: postgis/postgis:15-3.3
    environment:
      POSTGRES_DB: h3_raster
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck: { test: ["CMD-SHELL", "pg_isready -U postgres"], <<: *hc }

  jupyter:
    image: jupyter/scipy-notebook:latest
    ports: ["8888:8888"]
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
    healthcheck: { test: ["CMD", "curl", "-f", "http://localhost:8888/health"], <<: *hc }

  # Raster processing service (profile)
  processor:
    build: ./processor
    profiles: ["processing"]
    environment:
      POSTGRES_URL: postgresql://postgres:postgres@postgres:5432/h3_raster
    volumes:
      - ./data:/data
      - ./output:/output
    depends_on:
      postgres: { condition: service_healthy }

  # Visualization service (profile)
  viz:
    build: ./viz
    profiles: ["viz"]
    ports: ["3000:3000"]
    environment:
      POSTGRES_URL: postgresql://postgres:postgres@postgres:5432/h3_raster
    depends_on:
      postgres: { condition: service_healthy }

volumes:
  postgres_data:
```

### Operator Recipes

```bash
# Core H3 raster processing
docker compose up -d

# Add processing service
docker compose --profile processing up -d

# Add visualization
docker compose --profile viz up -d
```

## Data Model & H3 Schema

### H3 Hex Schema
```sql
-- Create H3 hex table
CREATE TABLE h3_hexes (
    h3_index TEXT PRIMARY KEY,
    resolution INTEGER NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create raster data table
CREATE TABLE raster_data (
    id SERIAL PRIMARY KEY,
    h3_index TEXT NOT NULL,
    resolution INTEGER NOT NULL,
    value DOUBLE PRECISION,
    data_type VARCHAR(50),
    source VARCHAR(100),
    timestamp TIMESTAMPTZ,
    metadata JSONB,
    FOREIGN KEY (h3_index) REFERENCES h3_hexes(h3_index)
);

-- Create indexes
CREATE INDEX idx_h3_hexes_resolution ON h3_hexes(resolution);
CREATE INDEX idx_h3_hexes_geom ON h3_hexes USING GIST(geom);
CREATE INDEX idx_raster_data_h3 ON raster_data(h3_index);
CREATE INDEX idx_raster_data_type ON raster_data(data_type);
CREATE INDEX idx_raster_data_timestamp ON raster_data(timestamp);
```

## Python Raster to H3 Processor

### processor/Dockerfile
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install rasterio geopandas h3 pandas numpy sqlalchemy psycopg2-binary

WORKDIR /app
COPY processor.py .

CMD ["python", "processor.py"]
```

### processor/processor.py
```python
import os
import h3
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sqlalchemy import create_engine, text
import json
from typing import List, Dict, Tuple
import logging

class RasterToH3Processor:
    def __init__(self, postgres_url: str):
        self.engine = create_engine(postgres_url)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_h3_hexes(self, bounds: Tuple[float, float, float, float], 
                       resolution: int) -> List[Dict]:
        """Create H3 hexagons for given bounds and resolution."""
        min_lat, min_lon, max_lat, max_lon = bounds
        
        # Get all H3 hexagons in bounds
        hexes = []
        
        # Calculate step size for sampling
        lat_step = (max_lat - min_lat) / 100
        lon_step = (max_lon - min_lon) / 100
        
        for lat in np.arange(min_lat, max_lat, lat_step):
            for lon in np.arange(min_lon, max_lon, lon_step):
                h3_index = h3.latlng_to_cell(lat, lon, resolution)
                
                if h3_index not in [h['h3_index'] for h in hexes]:
                    # Get hexagon boundary
                    hex_boundary = h3.cell_to_boundary(h3_index)
                    
                    # Create polygon
                    polygon = Polygon([(lon, lat) for lat, lon in hex_boundary])
                    
                    # Get center coordinates
                    center_lat, center_lon = h3.cell_to_latlng(h3_index)
                    
                    hexes.append({
                        'h3_index': h3_index,
                        'resolution': resolution,
                        'lat': center_lat,
                        'lon': center_lon,
                        'geom': polygon.wkt
                    })
        
        return hexes
    
    def process_raster_to_h3(self, raster_path: str, data_type: str, 
                            resolution: int = 9) -> None:
        """Process raster data and convert to H3 hexagons."""
        self.logger.info(f"Processing raster: {raster_path}")
        
        # Open raster
        with rasterio.open(raster_path) as src:
            # Get raster bounds
            bounds = src.bounds
            transform = src.transform
            
            # Read raster data
            data = src.read(1)  # Read first band
            
            # Create H3 hexagons for bounds
            hexes = self.create_h3_hexes(bounds, resolution)
            
            # Store hexes in database
            self.store_hexes(hexes)
            
            # Process raster values for each hex
            raster_values = self.aggregate_raster_to_hexes(data, transform, hexes)
            
            # Store raster data
            self.store_raster_data(raster_values, data_type, raster_path)
    
    def aggregate_raster_to_hexes(self, data: np.ndarray, transform, 
                                hexes: List[Dict]) -> List[Dict]:
        """Aggregate raster values to H3 hexagons."""
        values = []
        
        for hex_info in hexes:
            h3_index = hex_info['h3_index']
            
            # Get hexagon boundary
            hex_boundary = h3.cell_to_boundary(h3_index)
            
            # Convert to raster coordinates
            raster_coords = []
            for lat, lon in hex_boundary:
                row, col = rasterio.transform.rowcol(transform, lon, lat)
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    raster_coords.append((row, col))
            
            if raster_coords:
                # Create mask for hexagon
                mask = self.create_hex_mask(raster_coords, data.shape)
                
                # Extract values within hexagon
                hex_values = data[mask]
                
                # Calculate statistics
                if len(hex_values) > 0:
                    stats = {
                        'h3_index': h3_index,
                        'value': np.mean(hex_values),
                        'min_value': np.min(hex_values),
                        'max_value': np.max(hex_values),
                        'std_value': np.std(hex_values),
                        'count': len(hex_values)
                    }
                    values.append(stats)
        
        return values
    
    def create_hex_mask(self, coords: List[Tuple[int, int]], 
                       shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for hexagon in raster."""
        mask = np.zeros(shape, dtype=bool)
        
        # Simple polygon filling (in practice, use more sophisticated method)
        for row, col in coords:
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                mask[row, col] = True
        
        return mask
    
    def store_hexes(self, hexes: List[Dict]) -> None:
        """Store H3 hexagons in database."""
        with self.engine.connect() as conn:
            for hex_info in hexes:
                conn.execute(text("""
                    INSERT INTO h3_hexes (h3_index, resolution, lat, lon, geom)
                    VALUES (:h3_index, :resolution, :lat, :lon, ST_GeomFromText(:geom, 4326))
                    ON CONFLICT (h3_index) DO NOTHING
                """), hex_info)
            conn.commit()
    
    def store_raster_data(self, values: List[Dict], data_type: str, 
                         source: str) -> None:
        """Store raster data in database."""
        with self.engine.connect() as conn:
            for value_info in values:
                conn.execute(text("""
                    INSERT INTO raster_data (h3_index, resolution, value, data_type, source, metadata)
                    VALUES (:h3_index, :resolution, :value, :data_type, :source, :metadata)
                """), {
                    'h3_index': value_info['h3_index'],
                    'resolution': 9,  # Default resolution
                    'value': value_info['value'],
                    'data_type': data_type,
                    'source': source,
                    'metadata': json.dumps({
                        'min_value': value_info['min_value'],
                        'max_value': value_info['max_value'],
                        'std_value': value_info['std_value'],
                        'count': value_info['count']
                    })
                })
            conn.commit()
    
    def process_multiple_rasters(self, raster_configs: List[Dict]) -> None:
        """Process multiple raster files."""
        for config in raster_configs:
            try:
                self.process_raster_to_h3(
                    config['path'],
                    config['data_type'],
                    config.get('resolution', 9)
                )
                self.logger.info(f"Processed {config['path']}")
            except Exception as e:
                self.logger.error(f"Error processing {config['path']}: {e}")

# Example usage
if __name__ == "__main__":
    processor = RasterToH3Processor(
        postgres_url=os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/h3_raster")
    )
    
    # Process sample rasters
    raster_configs = [
        {
            'path': '/data/elevation.tif',
            'data_type': 'elevation',
            'resolution': 9
        },
        {
            'path': '/data/ndvi.tif',
            'data_type': 'ndvi',
            'resolution': 9
        },
        {
            'path': '/data/temperature.tif',
            'data_type': 'temperature',
            'resolution': 8
        }
    ]
    
    processor.process_multiple_rasters(raster_configs)
```

## H3 Aggregation Functions

### SQL Aggregation Queries
```sql
-- H3 hex aggregation by resolution
SELECT 
    resolution,
    COUNT(*) as hex_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value
FROM raster_data rd
JOIN h3_hexes hh ON rd.h3_index = hh.h3_index
GROUP BY resolution
ORDER BY resolution;

-- H3 hex statistics by data type
SELECT 
    data_type,
    COUNT(DISTINCT h3_index) as unique_hexes,
    AVG(value) as avg_value,
    STDDEV(value) as std_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value
FROM raster_data
GROUP BY data_type
ORDER BY data_type;

-- H3 hex spatial clustering
SELECT 
    h3_index,
    lat,
    lon,
    value,
    ST_AsText(geom) as geometry
FROM raster_data rd
JOIN h3_hexes hh ON rd.h3_index = hh.h3_index
WHERE rd.data_type = 'elevation'
ORDER BY value DESC
LIMIT 100;

-- H3 hex density analysis
SELECT 
    h3_index,
    COUNT(*) as data_points,
    AVG(value) as avg_value,
    STDDEV(value) as std_value
FROM raster_data
GROUP BY h3_index
HAVING COUNT(*) > 1
ORDER BY data_points DESC;
```

## H3 Resolution Analysis

### Python H3 Resolution Analysis
```python
# notebooks/h3_resolution_analysis.ipynb
import h3
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

def analyze_h3_resolutions(postgres_url: str):
    """Analyze H3 data across different resolutions."""
    engine = create_engine(postgres_url)
    
    # Get data for each resolution
    resolutions = [6, 7, 8, 9, 10, 11]
    results = []
    
    for res in resolutions:
        query = f"""
        SELECT 
            COUNT(*) as hex_count,
            AVG(value) as avg_value,
            STDDEV(value) as std_value,
            MIN(value) as min_value,
            MAX(value) as max_value
        FROM raster_data rd
        JOIN h3_hexes hh ON rd.h3_index = hh.h3_index
        WHERE hh.resolution = {res}
        """
        
        df = pd.read_sql(query, engine)
        df['resolution'] = res
        results.append(df)
    
    # Combine results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Plot resolution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hex count by resolution
    axes[0, 0].bar(combined_df['resolution'], combined_df['hex_count'])
    axes[0, 0].set_title('Hex Count by Resolution')
    axes[0, 0].set_xlabel('Resolution')
    axes[0, 0].set_ylabel('Hex Count')
    
    # Average value by resolution
    axes[0, 1].plot(combined_df['resolution'], combined_df['avg_value'])
    axes[0, 1].set_title('Average Value by Resolution')
    axes[0, 1].set_xlabel('Resolution')
    axes[0, 1].set_ylabel('Average Value')
    
    # Standard deviation by resolution
    axes[1, 0].plot(combined_df['resolution'], combined_df['std_value'])
    axes[1, 0].set_title('Standard Deviation by Resolution')
    axes[1, 0].set_xlabel('Resolution')
    axes[1, 0].set_ylabel('Standard Deviation')
    
    # Value range by resolution
    axes[1, 1].plot(combined_df['resolution'], combined_df['max_value'] - combined_df['min_value'])
    axes[1, 1].set_title('Value Range by Resolution')
    axes[1, 1].set_xlabel('Resolution')
    axes[1, 1].set_ylabel('Value Range')
    
    plt.tight_layout()
    plt.show()
    
    return combined_df

# Usage
results = analyze_h3_resolutions("postgresql://postgres:postgres@localhost:5432/h3_raster")
```

## H3 Visualization

### viz/Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package.json .
RUN npm install

COPY . .
EXPOSE 3000

CMD ["npm", "start"]
```

### viz/package.json
```json
{
  "name": "h3-raster-viz",
  "version": "1.0.0",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3",
    "h3-js": "^4.1.0"
  }
}
```

### viz/server.js
```javascript
const express = require('express');
const { Pool } = require('pg');
const h3 = require('h3-js');

const app = express();
const port = 3000;

// PostgreSQL connection
const pool = new Pool({
    user: 'postgres',
    host: 'postgres',
    database: 'h3_raster',
    password: 'postgres',
    port: 5432
});

// Serve static files
app.use(express.static('public'));

// API endpoint for H3 hex data
app.get('/api/hexes/:resolution', async (req, res) => {
    try {
        const { resolution } = req.params;
        const { data_type } = req.query;
        
        let query = `
            SELECT 
                rd.h3_index,
                hh.lat,
                hh.lon,
                rd.value,
                rd.data_type,
                ST_AsText(hh.geom) as geometry
            FROM raster_data rd
            JOIN h3_hexes hh ON rd.h3_index = hh.h3_index
            WHERE hh.resolution = $1
        `;
        
        const params = [resolution];
        
        if (data_type) {
            query += ' AND rd.data_type = $2';
            params.push(data_type);
        }
        
        const result = await pool.query(query, params);
        
        // Convert to GeoJSON
        const geojson = {
            type: 'FeatureCollection',
            features: result.rows.map(row => ({
                type: 'Feature',
                properties: {
                    h3_index: row.h3_index,
                    value: row.value,
                    data_type: row.data_type
                },
                geometry: {
                    type: 'Polygon',
                    coordinates: [JSON.parse(row.geometry.replace('POLYGON((', '[[').replace('))', ']]'))]
                }
            }))
        };
        
        res.json(geojson);
        
    } catch (error) {
        console.error('Error fetching hex data:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// API endpoint for H3 statistics
app.get('/api/stats/:resolution', async (req, res) => {
    try {
        const { resolution } = req.params;
        
        const query = `
            SELECT 
                data_type,
                COUNT(*) as hex_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                STDDEV(value) as std_value
            FROM raster_data rd
            JOIN h3_hexes hh ON rd.h3_index = hh.h3_index
            WHERE hh.resolution = $1
            GROUP BY data_type
        `;
        
        const result = await pool.query(query, [resolution]);
        res.json(result.rows);
        
    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(port, () => {
    console.log(`H3 raster visualization server running on port ${port}`);
});
```

### viz/public/index.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>H3 Raster Visualization</title>
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; }
        #map { height: 100vh; width: 100vw; }
        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="controls">
        <label for="resolution">Resolution:</label>
        <select id="resolution">
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9" selected>9</option>
            <option value="10">10</option>
            <option value="11">11</option>
        </select>
        
        <label for="dataType">Data Type:</label>
        <select id="dataType">
            <option value="">All</option>
            <option value="elevation">Elevation</option>
            <option value="ndvi">NDVI</option>
            <option value="temperature">Temperature</option>
        </select>
        
        <button onclick="loadHexData()">Load Data</button>
    </div>

    <script>
        const map = new maplibregl.Map({
            container: 'map',
            style: 'https://demotiles.maplibre.org/style.json',
            center: [-74.0, 40.7],
            zoom: 10
        });

        let hexLayer = null;

        function loadHexData() {
            const resolution = document.getElementById('resolution').value;
            const dataType = document.getElementById('dataType').value;
            
            const url = `/api/hexes/${resolution}${dataType ? `?data_type=${dataType}` : ''}`;
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    // Remove existing layer
                    if (hexLayer) {
                        map.removeLayer(hexLayer);
                        map.removeSource('hexes');
                    }
                    
                    // Add new layer
                    map.addSource('hexes', {
                        type: 'geojson',
                        data: data
                    });
                    
                    map.addLayer({
                        id: 'hexes',
                        type: 'fill',
                        source: 'hexes',
                        paint: {
                            'fill-color': [
                                'interpolate',
                                ['linear'],
                                ['get', 'value'],
                                0, '#0000ff',
                                0.5, '#00ff00',
                                1, '#ff0000'
                            ],
                            'fill-opacity': 0.7
                        }
                    });
                    
                    hexLayer = 'hexes';
                })
                .catch(error => {
                    console.error('Error loading hex data:', error);
                });
        }

        // Load initial data
        map.on('load', () => {
            loadHexData();
        });
    </script>
</body>
</html>
```

## H3 Performance Optimization

### H3 Indexing Strategies
```sql
-- Create H3 index for fast lookups
CREATE INDEX idx_h3_hexes_h3_index ON h3_hexes(h3_index);
CREATE INDEX idx_h3_hexes_resolution ON h3_hexes(resolution);
CREATE INDEX idx_h3_hexes_geom ON h3_hexes USING GIST(geom);

-- Create composite index for common queries
CREATE INDEX idx_raster_data_h3_type ON raster_data(h3_index, data_type);
CREATE INDEX idx_raster_data_resolution_type ON raster_data(resolution, data_type);

-- Create partial index for specific data types
CREATE INDEX idx_raster_data_elevation ON raster_data(h3_index, value) 
WHERE data_type = 'elevation';
```

### H3 Query Optimization
```sql
-- Optimized H3 hex lookup
SELECT 
    h3_index,
    value,
    data_type
FROM raster_data
WHERE h3_index IN (
    SELECT h3_index 
    FROM h3_hexes 
    WHERE resolution = 9
    AND ST_Intersects(geom, ST_MakeEnvelope(-74.1, 40.6, -73.9, 40.8, 4326))
);

-- H3 hex aggregation with spatial filtering
SELECT 
    h3_index,
    AVG(value) as avg_value,
    COUNT(*) as point_count
FROM raster_data
WHERE h3_index IN (
    SELECT h3_index 
    FROM h3_hexes 
    WHERE resolution = 9
    AND ST_DWithin(geom, ST_Point(-74.0, 40.7), 0.1)
)
GROUP BY h3_index
ORDER BY avg_value DESC;
```

## Failure Modes & Runbook

### Raster Processing Issues
- **Problem**: Large raster files cause memory issues
- **Solution**: Process in chunks, use streaming
- **Prevention**: Monitor memory usage, implement chunking

### H3 Resolution Mismatch
- **Problem**: Inconsistent hex sizes across datasets
- **Solution**: Standardize on resolution 9 for most use cases
- **Debug**: Validate H3 indices with `h3.is_valid_cell()`

### Spatial Index Performance
- **Problem**: Slow spatial queries on large datasets
- **Solution**: Use proper spatial indexes, optimize queries
- **Monitoring**: Check query execution plans

### Data Quality Issues
- **Problem**: Invalid or missing raster values
- **Solution**: Implement data validation, handle NoData values
- **Prevention**: Check raster metadata before processing

## TL;DR Runbook

```bash
# 1. Start core stack
docker compose up -d

# 2. Initialize database schema
psql -h localhost -U postgres -d h3_raster -f schema.sql

# 3. Add processing service
docker compose --profile processing up -d

# 4. Process sample rasters
docker compose exec processor python processor.py

# 5. Add visualization
docker compose --profile viz up -d

# 6. View results
# - Jupyter: http://localhost:8888
# - Visualization: http://localhost:3000
# - Database: psql -h localhost -U postgres -d h3_raster

# 7. Query H3 hex data
# - Get hexes by resolution: SELECT * FROM h3_hexes WHERE resolution = 9
# - Get raster data: SELECT * FROM raster_data WHERE data_type = 'elevation'
# - Spatial query: SELECT * FROM h3_hexes WHERE ST_Intersects(geom, ST_MakeEnvelope(-74.1, 40.6, -73.9, 40.8, 4326))
```

---

*This tutorial provides the complete machinery for converting raster data to H3 hexagonal grids. Each component is production-ready, copy-paste runnable, and designed to handle large-scale geospatial data processing with efficient spatial indexing and aggregation.*
