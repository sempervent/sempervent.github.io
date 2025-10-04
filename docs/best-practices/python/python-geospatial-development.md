# Python Geospatial Development Best Practices

**Objective**: Master senior-level Python geospatial development patterns for production systems. When you need to build spatial data applications, when you want to implement geospatial analysis workflows, when you need enterprise-grade geospatial strategies—these best practices become your weapon of choice.

## Core Principles

- **Spatial Accuracy**: Ensure precise spatial calculations and transformations
- **Performance**: Optimize for large geospatial datasets
- **Standards**: Follow OGC and industry standards
- **Visualization**: Create effective geospatial visualizations
- **Integration**: Seamlessly integrate with spatial databases

## Geospatial Data Processing

### Spatial Data Structures

```python
# python/01-spatial-data-structures.py

"""
Spatial data structures and geometric operations
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, cascaded_union
from shapely.affinity import translate, rotate, scale
from shapely.validation import make_valid
import pyproj
from pyproj import CRS, Transformer
import folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometryType(Enum):
    """Geometry type enumeration"""
    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"

class SpatialDataProcessor:
    """Spatial data processing utilities"""
    
    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = CRS.from_string(crs)
        self.processor_metrics = {}
    
    def create_point(self, x: float, y: float, crs: str = None) -> Point:
        """Create a point geometry"""
        point = Point(x, y)
        
        if crs and crs != self.crs.to_string():
            # Transform to target CRS
            transformer = Transformer.from_crs(self.crs, CRS.from_string(crs))
            x_transformed, y_transformed = transformer.transform(x, y)
            point = Point(x_transformed, y_transformed)
        
        return point
    
    def create_polygon(self, coordinates: List[Tuple[float, float]], 
                      crs: str = None) -> Polygon:
        """Create a polygon geometry"""
        polygon = Polygon(coordinates)
        
        if crs and crs != self.crs.to_string():
            # Transform coordinates
            transformer = Transformer.from_crs(self.crs, CRS.from_string(crs))
            transformed_coords = [transformer.transform(x, y) for x, y in coordinates]
            polygon = Polygon(transformed_coords)
        
        return polygon
    
    def create_linestring(self, coordinates: List[Tuple[float, float]], 
                         crs: str = None) -> LineString:
        """Create a linestring geometry"""
        linestring = LineString(coordinates)
        
        if crs and crs != self.crs.to_string():
            # Transform coordinates
            transformer = Transformer.from_crs(self.crs, CRS.from_string(crs))
            transformed_coords = [transformer.transform(x, y) for x, y in coordinates]
            linestring = LineString(transformed_coords)
        
        return linestring
    
    def transform_geometry(self, geometry, from_crs: str, to_crs: str) -> Any:
        """Transform geometry between coordinate systems"""
        transformer = Transformer.from_crs(from_crs, to_crs)
        
        if geometry.geom_type == "Point":
            x, y = transformer.transform(geometry.x, geometry.y)
            return Point(x, y)
        elif geometry.geom_type == "Polygon":
            coords = list(geometry.exterior.coords)
            transformed_coords = [transformer.transform(x, y) for x, y in coords]
            return Polygon(transformed_coords)
        else:
            # Handle other geometry types
            return geometry
    
    def calculate_distance(self, geom1: Any, geom2: Any, 
                          unit: str = "meters") -> float:
        """Calculate distance between geometries"""
        # Ensure geometries are in projected CRS for accurate distance calculation
        if self.crs.is_geographic:
            # Transform to UTM for distance calculation
            utm_crs = self._get_utm_crs(geom1.centroid.x, geom1.centroid.y)
            geom1_utm = self.transform_geometry(geom1, self.crs.to_string(), utm_crs)
            geom2_utm = self.transform_geometry(geom2, self.crs.to_string(), utm_crs)
        else:
            geom1_utm = geom1
            geom2_utm = geom2
        
        distance = geom1_utm.distance(geom2_utm)
        
        if unit == "kilometers":
            return distance / 1000
        elif unit == "miles":
            return distance / 1609.34
        else:
            return distance
    
    def calculate_area(self, geometry: Any, unit: str = "square_meters") -> float:
        """Calculate area of geometry"""
        if geometry.geom_type not in ["Polygon", "MultiPolygon"]:
            return 0.0
        
        # Ensure geometry is in projected CRS for accurate area calculation
        if self.crs.is_geographic:
            utm_crs = self._get_utm_crs(geometry.centroid.x, geometry.centroid.y)
            geometry_projected = self.transform_geometry(geometry, self.crs.to_string(), utm_crs)
        else:
            geometry_projected = geometry
        
        area = geometry_projected.area
        
        if unit == "square_kilometers":
            return area / 1_000_000
        elif unit == "acres":
            return area / 4046.86
        else:
            return area
    
    def calculate_bounds(self, geometries: List[Any]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for list of geometries"""
        if not geometries:
            return (0, 0, 0, 0)
        
        # Get bounds for each geometry
        bounds_list = [geom.bounds for geom in geometries]
        
        # Calculate overall bounds
        minx = min(bounds[0] for bounds in bounds_list)
        miny = min(bounds[1] for bounds in bounds_list)
        maxx = max(bounds[2] for bounds in bounds_list)
        maxy = max(bounds[3] for bounds in bounds_list)
        
        return (minx, miny, maxx, maxy)
    
    def _get_utm_crs(self, lon: float, lat: float) -> str:
        """Get UTM CRS for given coordinates"""
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        return f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"

class SpatialOperations:
    """Spatial operations and analysis"""
    
    def __init__(self):
        self.operation_metrics = {}
    
    def spatial_join(self, left_gdf: gpd.GeoDataFrame, right_gdf: gpd.GeoDataFrame, 
                    how: str = "inner", predicate: str = "intersects") -> gpd.GeoDataFrame:
        """Perform spatial join between GeoDataFrames"""
        result = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
        
        self.operation_metrics["spatial_join"] = {
            "left_features": len(left_gdf),
            "right_features": len(right_gdf),
            "result_features": len(result),
            "join_type": how,
            "predicate": predicate
        }
        
        return result
    
    def buffer_analysis(self, geometries: List[Any], buffer_distance: float, 
                       crs: str = None) -> List[Any]:
        """Create buffers around geometries"""
        if crs:
            # Transform to projected CRS for accurate buffering
            transformer = Transformer.from_crs("EPSG:4326", crs)
            buffered_geometries = []
            
            for geom in geometries:
                if geom.geom_type == "Point":
                    x, y = transformer.transform(geom.x, geom.y)
                    geom_projected = Point(x, y)
                else:
                    # Handle other geometry types
                    geom_projected = geom
                
                buffered = geom_projected.buffer(buffer_distance)
                buffered_geometries.append(buffered)
            
            return buffered_geometries
        else:
            return [geom.buffer(buffer_distance) for geom in geometries]
    
    def intersection_analysis(self, geom1: Any, geom2: Any) -> Dict[str, Any]:
        """Analyze intersection between geometries"""
        intersection = geom1.intersection(geom2)
        
        result = {
            "intersects": geom1.intersects(geom2),
            "intersection_geometry": intersection,
            "intersection_area": intersection.area if hasattr(intersection, 'area') else 0,
            "intersection_type": intersection.geom_type if hasattr(intersection, 'geom_type') else None
        }
        
        return result
    
    def union_analysis(self, geometries: List[Any]) -> Any:
        """Perform union of multiple geometries"""
        if not geometries:
            return None
        
        if len(geometries) == 1:
            return geometries[0]
        
        # Use unary_union for efficient union operation
        union_result = unary_union(geometries)
        
        self.operation_metrics["union"] = {
            "input_geometries": len(geometries),
            "result_geometry_type": union_result.geom_type,
            "result_area": union_result.area if hasattr(union_result, 'area') else 0
        }
        
        return union_result
    
    def clip_analysis(self, target_geom: Any, clip_geom: Any) -> Any:
        """Clip target geometry with clip geometry"""
        if not target_geom.intersects(clip_geom):
            return None
        
        clipped = target_geom.intersection(clip_geom)
        
        self.operation_metrics["clip"] = {
            "target_area": target_geom.area if hasattr(target_geom, 'area') else 0,
            "clip_area": clip_geom.area if hasattr(clip_geom, 'area') else 0,
            "result_area": clipped.area if hasattr(clipped, 'area') else 0
        }
        
        return clipped
    
    def nearest_neighbor_analysis(self, points: List[Point], 
                                 target_points: List[Point]) -> List[Dict[str, Any]]:
        """Find nearest neighbors between point sets"""
        results = []
        
        for point in points:
            distances = [point.distance(target) for target in target_points]
            nearest_idx = np.argmin(distances)
            nearest_point = target_points[nearest_idx]
            distance = distances[nearest_idx]
            
            results.append({
                "source_point": point,
                "nearest_target": nearest_point,
                "distance": distance,
                "target_index": nearest_idx
            })
        
        return results

class GeospatialVisualizer:
    """Geospatial visualization utilities"""
    
    def __init__(self):
        self.visualization_metrics = {}
    
    def create_interactive_map(self, gdf: gpd.GeoDataFrame, 
                             center: Tuple[float, float] = None,
                             zoom_start: int = 10) -> folium.Map:
        """Create interactive Folium map"""
        if center is None:
            # Calculate center from data
            bounds = gdf.total_bounds
            center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        
        m = folium.Map(location=center, zoom_start=zoom_start)
        
        # Add geometries to map
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == "Point":
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=f"Feature {idx}"
                ).add_to(m)
            elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    popup=f"Feature {idx}"
                ).add_to(m)
        
        return m
    
    def create_plotly_map(self, gdf: gpd.GeoDataFrame, 
                         color_column: str = None,
                         size_column: str = None) -> go.Figure:
        """Create Plotly map visualization"""
        if color_column is None:
            color_column = gdf.columns[0] if len(gdf.columns) > 1 else None
        
        # Convert geometries to coordinates
        lats, lons = [], []
        colors, sizes = [], []
        
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == "Point":
                lats.append(row.geometry.y)
                lons.append(row.geometry.x)
                colors.append(row[color_column] if color_column else "blue")
                sizes.append(row[size_column] if size_column else 10)
            elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                # Extract polygon coordinates
                if row.geometry.geom_type == "Polygon":
                    coords = list(row.geometry.exterior.coords)
                else:
                    coords = []
                    for poly in row.geometry.geoms:
                        coords.extend(list(poly.exterior.coords))
                
                if coords:
                    lats.extend([coord[1] for coord in coords])
                    lons.extend([coord[0] for coord in coords])
                    colors.extend([row[color_column] if color_column else "blue"] * len(coords))
                    sizes.extend([row[size_column] if size_column else 10] * len(coords))
        
        fig = go.Figure()
        
        if color_column:
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Viridis'
                ),
                text=gdf[color_column].tolist(),
                hovertemplate=f"{color_column}: %{{text}}<br>Lat: %{{lat}}<br>Lon: %{{lon}}<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(size=sizes)
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=np.mean(lats), lon=np.mean(lons)),
                zoom=10
            ),
            title="Geospatial Data Visualization"
        )
        
        return fig
    
    def create_choropleth_map(self, gdf: gpd.GeoDataFrame, 
                            value_column: str,
                            color_scale: str = "Viridis") -> go.Figure:
        """Create choropleth map"""
        fig = px.choropleth_mapbox(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color=value_column,
            color_continuous_scale=color_scale,
            mapbox_style="open-street-map",
            center=dict(lat=gdf.geometry.centroid.y.mean(), 
                      lon=gdf.geometry.centroid.x.mean()),
            zoom=10
        )
        
        fig.update_layout(
            title=f"Choropleth Map - {value_column}",
            margin=dict(r=0, t=0, l=0, b=0)
        )
        
        return fig

class SpatialDatabase:
    """Spatial database operations"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.db_metrics = {}
    
    def create_spatial_table(self, table_name: str, 
                           geometry_column: str = "geometry",
                           crs: str = "EPSG:4326") -> bool:
        """Create spatial table with PostGIS"""
        # This would be implemented with actual database connection
        # For now, return success
        return True
    
    def insert_spatial_data(self, table_name: str, gdf: gpd.GeoDataFrame) -> bool:
        """Insert spatial data into database"""
        # This would be implemented with actual database connection
        # For now, return success
        self.db_metrics["inserted_features"] = len(gdf)
        return True
    
    def spatial_query(self, table_name: str, 
                     geometry: Any, 
                     operation: str = "intersects") -> gpd.GeoDataFrame:
        """Perform spatial query"""
        # This would be implemented with actual database connection
        # For now, return empty GeoDataFrame
        return gpd.GeoDataFrame()
    
    def spatial_index_analysis(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze spatial index performance"""
        # Calculate spatial index statistics
        bounds = gdf.total_bounds
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        
        result = {
            "total_features": len(gdf),
            "bounds": bounds,
            "area": area,
            "density": len(gdf) / area if area > 0 else 0,
            "geometry_types": gdf.geometry.geom_type.value_counts().to_dict()
        }
        
        self.db_metrics["spatial_index"] = result
        return result

# Usage examples
def example_geospatial_development():
    """Example geospatial development usage"""
    # Create spatial data processor
    processor = SpatialDataProcessor(crs="EPSG:4326")
    
    # Create sample geometries
    point1 = processor.create_point(-74.0, 40.7)  # New York
    point2 = processor.create_point(-73.9, 40.8)  # Nearby point
    
    polygon_coords = [(-74.1, 40.6), (-73.9, 40.6), (-73.9, 40.8), (-74.1, 40.8), (-74.1, 40.6)]
    polygon = processor.create_polygon(polygon_coords)
    
    # Calculate distance
    distance = processor.calculate_distance(point1, point2, unit="meters")
    print(f"Distance between points: {distance:.2f} meters")
    
    # Calculate area
    area = processor.calculate_area(polygon, unit="square_kilometers")
    print(f"Polygon area: {area:.6f} square kilometers")
    
    # Create GeoDataFrame
    data = {
        'name': ['Point 1', 'Point 2', 'Polygon 1'],
        'value': [10, 20, 30]
    }
    geometries = [point1, point2, polygon]
    
    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
    print(f"GeoDataFrame created with {len(gdf)} features")
    
    # Spatial operations
    spatial_ops = SpatialOperations()
    
    # Buffer analysis
    buffered_points = spatial_ops.buffer_analysis([point1, point2], 1000)  # 1km buffer
    print(f"Created buffers for {len(buffered_points)} points")
    
    # Intersection analysis
    intersection = spatial_ops.intersection_analysis(point1, polygon)
    print(f"Point intersects polygon: {intersection['intersects']}")
    
    # Union analysis
    union_result = spatial_ops.union_analysis(buffered_points)
    print(f"Union result type: {union_result.geom_type}")
    
    # Visualization
    visualizer = GeospatialVisualizer()
    
    # Create interactive map
    folium_map = visualizer.create_interactive_map(gdf)
    print("Interactive map created")
    
    # Create Plotly map
    plotly_fig = visualizer.create_plotly_map(gdf, color_column='value')
    print("Plotly map created")
    
    # Spatial database operations
    spatial_db = SpatialDatabase("postgresql://user:pass@localhost/db")
    
    # Create spatial table
    table_created = spatial_db.create_spatial_table("spatial_features")
    print(f"Spatial table created: {table_created}")
    
    # Insert data
    data_inserted = spatial_db.insert_spatial_data("spatial_features", gdf)
    print(f"Data inserted: {data_inserted}")
    
    # Spatial index analysis
    index_analysis = spatial_db.spatial_index_analysis(gdf)
    print(f"Spatial index analysis: {index_analysis}")
```

### Geospatial Analysis

```python
# python/02-geospatial-analysis.py

"""
Advanced geospatial analysis patterns and workflows
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SpatialAnalysis:
    """Advanced spatial analysis utilities"""
    
    def __init__(self):
        self.analysis_metrics = {}
    
    def point_in_polygon_analysis(self, points: gpd.GeoDataFrame, 
                                 polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Analyze which points are inside which polygons"""
        # Perform spatial join
        result = gpd.sjoin(points, polygons, how='left', predicate='within')
        
        self.analysis_metrics["point_in_polygon"] = {
            "total_points": len(points),
            "total_polygons": len(polygons),
            "points_in_polygons": len(result.dropna(subset=['index_right'])),
            "coverage_percentage": (len(result.dropna(subset=['index_right'])) / len(points)) * 100
        }
        
        return result
    
    def spatial_clustering(self, points: gpd.GeoDataFrame, 
                          algorithm: str = "dbscan",
                          **kwargs) -> gpd.GeoDataFrame:
        """Perform spatial clustering on points"""
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in points.geometry])
        
        if algorithm == "dbscan":
            eps = kwargs.get('eps', 0.01)
            min_samples = kwargs.get('min_samples', 5)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(coords)
            
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Add cluster labels to GeoDataFrame
        points_with_clusters = points.copy()
        points_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.analysis_metrics["spatial_clustering"] = {
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "total_points": len(points),
            "clustered_points": len(points) - n_noise
        }
        
        return points_with_clusters
    
    def spatial_density_analysis(self, points: gpd.GeoDataFrame, 
                                cell_size: float = 0.01) -> gpd.GeoDataFrame:
        """Analyze spatial density using grid cells"""
        # Create grid
        bounds = points.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Create grid cells
        grid_cells = []
        x_coords = np.arange(x_min, x_max, cell_size)
        y_coords = np.arange(y_min, y_max, cell_size)
        
        for x in x_coords:
            for y in y_coords:
                cell = Polygon([
                    (x, y), (x + cell_size, y), 
                    (x + cell_size, y + cell_size), (x, y + cell_size)
                ])
                grid_cells.append(cell)
        
        # Create grid GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=points.crs)
        
        # Count points in each cell
        spatial_join = gpd.sjoin(grid_gdf, points, how='left', predicate='contains')
        point_counts = spatial_join.groupby(spatial_join.index).size()
        
        # Add density information
        grid_gdf['point_count'] = point_counts.reindex(grid_gdf.index, fill_value=0)
        grid_gdf['density'] = grid_gdf['point_count'] / (cell_size ** 2)
        
        self.analysis_metrics["density_analysis"] = {
            "grid_cells": len(grid_gdf),
            "cell_size": cell_size,
            "max_density": grid_gdf['density'].max(),
            "mean_density": grid_gdf['density'].mean(),
            "high_density_cells": len(grid_gdf[grid_gdf['density'] > grid_gdf['density'].quantile(0.9)])
        }
        
        return grid_gdf
    
    def spatial_autocorrelation(self, gdf: gpd.GeoDataFrame, 
                               value_column: str) -> Dict[str, Any]:
        """Calculate spatial autocorrelation (Moran's I)"""
        from libpysal.weights import Queen
        from esda.moran import Moran
        
        # Create spatial weights
        w = Queen.from_dataframe(gdf)
        
        # Calculate Moran's I
        moran = Moran(gdf[value_column], w)
        
        result = {
            "morans_i": moran.I,
            "p_value": moran.p_norm,
            "z_score": moran.z_norm,
            "significant": moran.p_norm < 0.05,
            "interpretation": "positive" if moran.I > 0 else "negative"
        }
        
        self.analysis_metrics["spatial_autocorrelation"] = result
        return result
    
    def spatial_interpolation(self, points: gpd.GeoDataFrame, 
                            value_column: str,
                            method: str = "idw",
                            grid_resolution: int = 100) -> gpd.GeoDataFrame:
        """Perform spatial interpolation"""
        from scipy.interpolate import griddata
        
        # Extract coordinates and values
        coords = np.array([[point.x, point.y] for point in points.geometry])
        values = points[value_column].values
        
        # Create interpolation grid
        bounds = points.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Perform interpolation
        if method == "idw":
            # Inverse Distance Weighting
            grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            interpolated_values = griddata(coords, values, grid_points, method='cubic')
            interpolated_values = interpolated_values.reshape(X_grid.shape)
        
        # Create result GeoDataFrame
        result_data = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                if not np.isnan(interpolated_values[i, j]):
                    point = Point(x_grid[j], y_grid[i])
                    result_data.append({
                        'geometry': point,
                        'interpolated_value': interpolated_values[i, j]
                    })
        
        result_gdf = gpd.GeoDataFrame(result_data, crs=points.crs)
        
        self.analysis_metrics["spatial_interpolation"] = {
            "method": method,
            "grid_resolution": grid_resolution,
            "interpolated_points": len(result_gdf),
            "min_value": result_gdf['interpolated_value'].min(),
            "max_value": result_gdf['interpolated_value'].max()
        }
        
        return result_gdf

class NetworkAnalysis:
    """Network analysis utilities"""
    
    def __init__(self):
        self.network_metrics = {}
    
    def create_network_graph(self, edges: gpd.GeoDataFrame, 
                           nodes: gpd.GeoDataFrame = None) -> Dict[str, Any]:
        """Create network graph from edges and nodes"""
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        if nodes is not None:
            for idx, node in nodes.iterrows():
                G.add_node(idx, **node.drop('geometry').to_dict())
        else:
            # Extract nodes from edges
            all_nodes = set()
            for edge in edges.geometry:
                if edge.geom_type == "LineString":
                    all_nodes.update(edge.coords)
            
            for i, node_coords in enumerate(all_nodes):
                G.add_node(i, x=node_coords[0], y=node_coords[1])
        
        # Add edges
        for idx, edge in edges.iterrows():
            if edge.geometry.geom_type == "LineString":
                coords = list(edge.geometry.coords)
                for i in range(len(coords) - 1):
                    start_node = coords[i]
                    end_node = coords[i + 1]
                    
                    # Find closest nodes in graph
                    start_idx = self._find_closest_node(G, start_node)
                    end_idx = self._find_closest_node(G, end_node)
                    
                    if start_idx != end_idx:
                        G.add_edge(start_idx, end_idx, **edge.drop('geometry').to_dict())
        
        self.network_metrics["network_graph"] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(G)
        }
        
        return {"graph": G, "metrics": self.network_metrics["network_graph"]}
    
    def _find_closest_node(self, G, coords: Tuple[float, float], 
                          tolerance: float = 0.001) -> int:
        """Find closest node to given coordinates"""
        min_distance = float('inf')
        closest_node = None
        
        for node_id, node_data in G.nodes(data=True):
            if 'x' in node_data and 'y' in node_data:
                distance = np.sqrt((node_data['x'] - coords[0])**2 + (node_data['y'] - coords[1])**2)
                if distance < min_distance and distance < tolerance:
                    min_distance = distance
                    closest_node = node_id
        
        if closest_node is None:
            # Create new node
            new_node_id = G.number_of_nodes()
            G.add_node(new_node_id, x=coords[0], y=coords[1])
            return new_node_id
        
        return closest_node
    
    def shortest_path_analysis(self, G, start_node: int, end_node: int) -> Dict[str, Any]:
        """Find shortest path between nodes"""
        import networkx as nx
        
        try:
            path = nx.shortest_path(G, start_node, end_node)
            path_length = nx.shortest_path_length(G, start_node, end_node)
            
            result = {
                "path": path,
                "path_length": path_length,
                "path_exists": True
            }
        except nx.NetworkXNoPath:
            result = {
                "path": [],
                "path_length": float('inf'),
                "path_exists": False
            }
        
        return result
    
    def centrality_analysis(self, G) -> Dict[str, Any]:
        """Calculate network centrality measures"""
        import networkx as nx
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        result = {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality,
            "most_central_node": max(degree_centrality, key=degree_centrality.get),
            "network_diameter": nx.diameter(G) if nx.is_connected(G) else None
        }
        
        return result

class GeospatialWorkflow:
    """Complete geospatial workflow"""
    
    def __init__(self):
        self.workflow_metrics = {}
        self.spatial_processor = SpatialDataProcessor()
        self.spatial_ops = SpatialOperations()
        self.spatial_analysis = SpatialAnalysis()
        self.network_analysis = NetworkAnalysis()
    
    def complete_analysis(self, points: gpd.GeoDataFrame, 
                         polygons: gpd.GeoDataFrame = None,
                         edges: gpd.GeoDataFrame = None) -> Dict[str, Any]:
        """Perform complete geospatial analysis"""
        results = {}
        
        # Basic spatial analysis
        if polygons is not None:
            pip_result = self.spatial_analysis.point_in_polygon_analysis(points, polygons)
            results["point_in_polygon"] = pip_result
        
        # Spatial clustering
        clustered_points = self.spatial_analysis.spatial_clustering(points)
        results["spatial_clustering"] = clustered_points
        
        # Density analysis
        density_grid = self.spatial_analysis.spatial_density_analysis(points)
        results["density_analysis"] = density_grid
        
        # Spatial interpolation
        if len(points) > 1:
            value_column = points.select_dtypes(include=[np.number]).columns[0]
            interpolated = self.spatial_analysis.spatial_interpolation(points, value_column)
            results["spatial_interpolation"] = interpolated
        
        # Network analysis
        if edges is not None:
            network_result = self.network_analysis.create_network_graph(edges)
            results["network_analysis"] = network_result
        
        # Compile metrics
        self.workflow_metrics = {
            "input_points": len(points),
            "input_polygons": len(polygons) if polygons is not None else 0,
            "input_edges": len(edges) if edges is not None else 0,
            "analysis_completed": list(results.keys())
        }
        
        return results
    
    def export_results(self, results: Dict[str, Any], output_dir: str) -> bool:
        """Export analysis results"""
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for analysis_name, result in results.items():
                if isinstance(result, gpd.GeoDataFrame):
                    result.to_file(f"{output_dir}/{analysis_name}.geojson", driver="GeoJSON")
                elif isinstance(result, dict) and "graph" in result:
                    # Export network graph
                    import networkx as nx
                    nx.write_gml(result["graph"], f"{output_dir}/{analysis_name}_network.gml")
            
            return True
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False

# Usage examples
def example_geospatial_analysis():
    """Example geospatial analysis usage"""
    # Create sample data
    np.random.seed(42)
    
    # Create random points
    n_points = 100
    points_data = {
        'id': range(n_points),
        'value': np.random.normal(50, 15, n_points),
        'category': np.random.choice(['A', 'B', 'C'], n_points)
    }
    
    # Create random point geometries
    points_geometry = [Point(np.random.uniform(-74.1, -73.9), 
                           np.random.uniform(40.6, 40.8)) for _ in range(n_points)]
    
    points_gdf = gpd.GeoDataFrame(points_data, geometry=points_geometry, crs="EPSG:4326")
    
    # Create sample polygons
    polygon_coords = [
        [(-74.05, 40.65), (-73.95, 40.65), (-73.95, 40.75), (-74.05, 40.75), (-74.05, 40.65)],
        [(-73.9, 40.7), (-73.8, 40.7), (-73.8, 40.8), (-73.9, 40.8), (-73.9, 40.7)]
    ]
    
    polygons_data = {'id': [1, 2], 'name': ['Area 1', 'Area 2']}
    polygons_geometry = [Polygon(coords) for coords in polygon_coords]
    polygons_gdf = gpd.GeoDataFrame(polygons_data, geometry=polygons_geometry, crs="EPSG:4326")
    
    # Perform spatial analysis
    spatial_analysis = SpatialAnalysis()
    
    # Point in polygon analysis
    pip_result = spatial_analysis.point_in_polygon_analysis(points_gdf, polygons_gdf)
    print(f"Points in polygons: {len(pip_result.dropna(subset=['index_right']))}")
    
    # Spatial clustering
    clustered_points = spatial_analysis.spatial_clustering(points_gdf, algorithm="dbscan", eps=0.01)
    print(f"Clusters found: {clustered_points['cluster'].nunique()}")
    
    # Density analysis
    density_grid = spatial_analysis.spatial_density_analysis(points_gdf, cell_size=0.01)
    print(f"Density grid created with {len(density_grid)} cells")
    
    # Spatial interpolation
    interpolated = spatial_analysis.spatial_interpolation(points_gdf, 'value')
    print(f"Interpolated surface created with {len(interpolated)} points")
    
    # Network analysis
    network_analysis = NetworkAnalysis()
    
    # Create sample edges
    edge_coords = [
        LineString([(-74.0, 40.7), (-73.9, 40.7)]),
        LineString([(-73.9, 40.7), (-73.8, 40.8)]),
        LineString([(-74.0, 40.7), (-74.0, 40.8)])
    ]
    
    edges_data = {'id': [1, 2, 3], 'weight': [1, 2, 1]}
    edges_gdf = gpd.GeoDataFrame(edges_data, geometry=edge_coords, crs="EPSG:4326")
    
    # Create network graph
    network_result = network_analysis.create_network_graph(edges_gdf)
    print(f"Network created with {network_result['metrics']['nodes']} nodes and {network_result['metrics']['edges']} edges")
    
    # Complete workflow
    workflow = GeospatialWorkflow()
    complete_results = workflow.complete_analysis(points_gdf, polygons_gdf, edges_gdf)
    print(f"Complete analysis completed: {list(complete_results.keys())}")
    
    # Export results
    export_success = workflow.export_results(complete_results, "geospatial_output")
    print(f"Results exported: {export_success}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Spatial data processing
processor = SpatialDataProcessor(crs="EPSG:4326")
point = processor.create_point(-74.0, 40.7)
polygon = processor.create_polygon([(-74.1, 40.6), (-73.9, 40.6), (-73.9, 40.8), (-74.1, 40.8)])

# 2. Spatial operations
spatial_ops = SpatialOperations()
buffered = spatial_ops.buffer_analysis([point], 1000)
intersection = spatial_ops.intersection_analysis(point, polygon)

# 3. Spatial analysis
spatial_analysis = SpatialAnalysis()
clustered = spatial_analysis.spatial_clustering(points_gdf)
density_grid = spatial_analysis.spatial_density_analysis(points_gdf)

# 4. Visualization
visualizer = GeospatialVisualizer()
folium_map = visualizer.create_interactive_map(gdf)
plotly_fig = visualizer.create_plotly_map(gdf, color_column='value')

# 5. Complete workflow
workflow = GeospatialWorkflow()
results = workflow.complete_analysis(points_gdf, polygons_gdf, edges_gdf)
```

### Essential Patterns

```python
# Complete geospatial setup
def setup_geospatial_development():
    """Setup complete geospatial development environment"""
    
    # Spatial data processor
    processor = SpatialDataProcessor()
    
    # Spatial operations
    spatial_ops = SpatialOperations()
    
    # Spatial analysis
    spatial_analysis = SpatialAnalysis()
    
    # Network analysis
    network_analysis = NetworkAnalysis()
    
    # Geospatial visualizer
    visualizer = GeospatialVisualizer()
    
    # Spatial database
    spatial_db = SpatialDatabase("postgresql://user:pass@localhost/db")
    
    # Complete workflow
    workflow = GeospatialWorkflow()
    
    print("Geospatial development setup complete!")
```

---

*This guide provides the complete machinery for Python geospatial development. Each pattern includes implementation examples, spatial analysis strategies, and real-world usage patterns for enterprise geospatial management.*
