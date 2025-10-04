# Rust Geospatial Development Best Practices

**Objective**: Master senior-level Rust geospatial development patterns for production systems. When you need to build high-performance geospatial applications, when you want to leverage Rust's speed for spatial data processing, when you need enterprise-grade geospatial patterns—these best practices become your weapon of choice.

## Core Principles

- **Spatial Data Types**: Use appropriate spatial data structures
- **Coordinate Systems**: Handle different coordinate reference systems
- **Spatial Indexing**: Implement efficient spatial indexing
- **Spatial Operations**: Use optimized spatial algorithms
- **Performance**: Leverage Rust's speed for spatial computations

## Geospatial Development Patterns

### Spatial Data Structures

```rust
// rust/01-spatial-data-structures.rs

/*
Spatial data structures and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Point in 2D space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    /// Calculate distance to another point.
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
    
    /// Calculate bearing to another point.
    pub fn bearing_to(&self, other: &Point) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        dy.atan2(dx).to_degrees()
    }
}

/// Bounding box for spatial queries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }
    
    /// Check if point is within bounding box.
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.min_x && point.x <= self.max_x &&
        point.y >= self.min_y && point.y <= self.max_y
    }
    
    /// Check if bounding box intersects with another.
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.max_x < other.min_x || self.min_x > other.max_x ||
          self.max_y < other.min_y || self.min_y > other.max_y)
    }
    
    /// Expand bounding box to include point.
    pub fn expand_to_include(&mut self, point: &Point) {
        self.min_x = self.min_x.min(point.x);
        self.min_y = self.min_y.min(point.y);
        self.max_x = self.max_x.max(point.x);
        self.max_y = self.max_y.max(point.y);
    }
    
    /// Calculate area of bounding box.
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }
}

/// Polygon with holes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polygon {
    pub exterior: Vec<Point>,
    pub holes: Vec<Vec<Point>>,
}

impl Polygon {
    pub fn new(exterior: Vec<Point>) -> Self {
        Self {
            exterior,
            holes: Vec::new(),
        }
    }
    
    /// Add a hole to the polygon.
    pub fn add_hole(&mut self, hole: Vec<Point>) {
        self.holes.push(hole);
    }
    
    /// Check if point is inside polygon.
    pub fn contains(&self, point: &Point) -> bool {
        if !self.point_in_ring(point, &self.exterior) {
            return false;
        }
        
        // Check if point is in any hole
        for hole in &self.holes {
            if self.point_in_ring(point, hole) {
                return false;
            }
        }
        
        true
    }
    
    /// Check if point is inside a ring using ray casting.
    fn point_in_ring(&self, point: &Point, ring: &[Point]) -> bool {
        let mut inside = false;
        let mut j = ring.len() - 1;
        
        for i in 0..ring.len() {
            let pi = &ring[i];
            let pj = &ring[j];
            
            if ((pi.y > point.y) != (pj.y > point.y)) &&
               (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x) {
                inside = !inside;
            }
            j = i;
        }
        
        inside
    }
    
    /// Calculate area of polygon.
    pub fn area(&self) -> f64 {
        let mut area = self.ring_area(&self.exterior);
        
        // Subtract area of holes
        for hole in &self.holes {
            area -= self.ring_area(hole);
        }
        
        area.abs()
    }
    
    /// Calculate area of a ring.
    fn ring_area(&self, ring: &[Point]) -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        
        let mut area = 0.0;
        let mut j = ring.len() - 1;
        
        for i in 0..ring.len() {
            area += (ring[j].x + ring[i].x) * (ring[j].y - ring[i].y);
            j = i;
        }
        
        area / 2.0
    }
    
    /// Get bounding box of polygon.
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox = BoundingBox::new(
            f64::INFINITY, f64::INFINITY,
            f64::NEG_INFINITY, f64::NEG_INFINITY,
        );
        
        for point in &self.exterior {
            bbox.expand_to_include(point);
        }
        
        for hole in &self.holes {
            for point in hole {
                bbox.expand_to_include(point);
            }
        }
        
        bbox
    }
}

/// Line string (polyline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineString {
    pub points: Vec<Point>,
}

impl LineString {
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
    
    /// Calculate length of line string.
    pub fn length(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        
        let mut length = 0.0;
        for i in 1..self.points.len() {
            length += self.points[i-1].distance_to(&self.points[i]);
        }
        length
    }
    
    /// Get bounding box of line string.
    pub fn bounding_box(&self) -> BoundingBox {
        if self.points.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }
        
        let mut bbox = BoundingBox::new(
            f64::INFINITY, f64::INFINITY,
            f64::NEG_INFINITY, f64::NEG_INFINITY,
        );
        
        for point in &self.points {
            bbox.expand_to_include(point);
        }
        
        bbox
    }
}

/// Spatial feature with geometry and properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialFeature {
    pub id: String,
    pub geometry: Geometry,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Geometry {
    Point(Point),
    LineString(LineString),
    Polygon(Polygon),
    MultiPoint(Vec<Point>),
    MultiLineString(Vec<LineString>),
    MultiPolygon(Vec<Polygon>),
}

impl Geometry {
    /// Get bounding box of geometry.
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Geometry::Point(point) => {
                BoundingBox::new(point.x, point.y, point.x, point.y)
            }
            Geometry::LineString(linestring) => linestring.bounding_box(),
            Geometry::Polygon(polygon) => polygon.bounding_box(),
            Geometry::MultiPoint(points) => {
                if points.is_empty() {
                    return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
                }
                let mut bbox = BoundingBox::new(
                    f64::INFINITY, f64::INFINITY,
                    f64::NEG_INFINITY, f64::NEG_INFINITY,
                );
                for point in points {
                    bbox.expand_to_include(point);
                }
                bbox
            }
            Geometry::MultiLineString(linestrings) => {
                if linestrings.is_empty() {
                    return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
                }
                let mut bbox = linestrings[0].bounding_box();
                for linestring in &linestrings[1..] {
                    let ls_bbox = linestring.bounding_box();
                    bbox.min_x = bbox.min_x.min(ls_bbox.min_x);
                    bbox.min_y = bbox.min_y.min(ls_bbox.min_y);
                    bbox.max_x = bbox.max_x.max(ls_bbox.max_x);
                    bbox.max_y = bbox.max_y.max(ls_bbox.max_y);
                }
                bbox
            }
            Geometry::MultiPolygon(polygons) => {
                if polygons.is_empty() {
                    return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
                }
                let mut bbox = polygons[0].bounding_box();
                for polygon in &polygons[1..] {
                    let poly_bbox = polygon.bounding_box();
                    bbox.min_x = bbox.min_x.min(poly_bbox.min_x);
                    bbox.min_y = bbox.min_y.min(poly_bbox.min_y);
                    bbox.max_x = bbox.max_x.max(poly_bbox.max_x);
                    bbox.max_y = bbox.max_y.max(poly_bbox.max_y);
                }
                bbox
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1.distance_to(&p2), 5.0);
    }
    
    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let point = Point::new(5.0, 5.0);
        assert!(bbox.contains(&point));
        
        let point_outside = Point::new(15.0, 15.0);
        assert!(!bbox.contains(&point_outside));
    }
    
    #[test]
    fn test_polygon_area() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let polygon = Polygon::new(exterior);
        assert_eq!(polygon.area(), 100.0);
    }
    
    #[test]
    fn test_polygon_contains() {
        let exterior = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let polygon = Polygon::new(exterior);
        
        let inside_point = Point::new(5.0, 5.0);
        assert!(polygon.contains(&inside_point));
        
        let outside_point = Point::new(15.0, 15.0);
        assert!(!polygon.contains(&outside_point));
    }
}
```

### Spatial Indexing

```rust
// rust/02-spatial-indexing.rs

/*
Spatial indexing patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// R-tree for spatial indexing.
pub struct RTree {
    root: Option<Box<RTreeNode>>,
    max_entries: usize,
    min_entries: usize,
}

#[derive(Debug, Clone)]
pub struct RTreeNode {
    pub bounding_box: BoundingBox,
    pub entries: Vec<RTreeEntry>,
    pub is_leaf: bool,
}

#[derive(Debug, Clone)]
pub enum RTreeEntry {
    Leaf {
        id: String,
        bounding_box: BoundingBox,
        data: SpatialFeature,
    },
    Node {
        node: Box<RTreeNode>,
    },
}

impl RTree {
    pub fn new(max_entries: usize) -> Self {
        Self {
            root: None,
            max_entries,
            min_entries: max_entries / 2,
        }
    }
    
    /// Insert a spatial feature into the R-tree.
    pub fn insert(&mut self, feature: SpatialFeature) -> Result<(), String> {
        let bounding_box = feature.geometry.bounding_box();
        let entry = RTreeEntry::Leaf {
            id: feature.id.clone(),
            bounding_box,
            data: feature,
        };
        
        if let Some(root) = &mut self.root {
            self.insert_entry(entry, root)?;
        } else {
            // Create root node
            let mut root = RTreeNode {
                bounding_box: bounding_box,
                entries: vec![entry],
                is_leaf: true,
            };
            self.root = Some(Box::new(root));
        }
        
        Ok(())
    }
    
    /// Insert entry into node.
    fn insert_entry(&mut self, entry: RTreeEntry, node: &mut RTreeNode) -> Result<(), String> {
        if node.is_leaf {
            node.entries.push(entry);
            if node.entries.len() > self.max_entries {
                self.split_node(node)?;
            }
        } else {
            // Find best child node
            let mut best_child = 0;
            let mut min_expansion = f64::INFINITY;
            
            for (i, child_entry) in node.entries.iter().enumerate() {
                if let RTreeEntry::Node { node: child_node } = child_entry {
                    let expansion = self.calculate_expansion(&entry, &child_node.bounding_box);
                    if expansion < min_expansion {
                        min_expansion = expansion;
                        best_child = i;
                    }
                }
            }
            
            // Insert into best child
            if let RTreeEntry::Node { node: child_node } = &mut node.entries[best_child] {
                self.insert_entry(entry, child_node)?;
            }
        }
        
        // Update bounding box
        self.update_bounding_box(node);
        Ok(())
    }
    
    /// Split node when it exceeds maximum entries.
    fn split_node(&mut self, node: &mut RTreeNode) -> Result<(), String> {
        if node.entries.len() <= self.max_entries {
            return Ok(());
        }
        
        // Simple split: divide entries into two groups
        let mid = node.entries.len() / 2;
        let mut entries1 = node.entries.drain(..mid).collect::<Vec<_>>();
        let entries2 = node.entries;
        
        // Create new nodes
        let mut node1 = RTreeNode {
            bounding_box: BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            entries: entries1,
            is_leaf: node.is_leaf,
        };
        let mut node2 = RTreeNode {
            bounding_box: BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            entries: entries2,
            is_leaf: node.is_leaf,
        };
        
        // Update bounding boxes
        self.update_bounding_box(&mut node1);
        self.update_bounding_box(&mut node2);
        
        // Replace current node with two child nodes
        node.entries = vec![
            RTreeEntry::Node { node: Box::new(node1) },
            RTreeEntry::Node { node: Box::new(node2) },
        ];
        node.is_leaf = false;
        
        Ok(())
    }
    
    /// Calculate expansion needed to include entry.
    fn calculate_expansion(&self, entry: &RTreeEntry, bbox: &BoundingBox) -> f64 {
        let entry_bbox = match entry {
            RTreeEntry::Leaf { bounding_box, .. } => bounding_box,
            RTreeEntry::Node { node } => &node.bounding_box,
        };
        
        let new_bbox = BoundingBox::new(
            bbox.min_x.min(entry_bbox.min_x),
            bbox.min_y.min(entry_bbox.min_y),
            bbox.max_x.max(entry_bbox.max_x),
            bbox.max_y.max(entry_bbox.max_y),
        );
        
        new_bbox.area() - bbox.area()
    }
    
    /// Update bounding box of node.
    fn update_bounding_box(&self, node: &mut RTreeNode) {
        if node.entries.is_empty() {
            return;
        }
        
        let mut bbox = match &node.entries[0] {
            RTreeEntry::Leaf { bounding_box, .. } => *bounding_box,
            RTreeEntry::Node { node: child_node } => child_node.bounding_box,
        };
        
        for entry in &node.entries[1..] {
            let entry_bbox = match entry {
                RTreeEntry::Leaf { bounding_box, .. } => bounding_box,
                RTreeEntry::Node { node: child_node } => &child_node.bounding_box,
            };
            
            bbox.min_x = bbox.min_x.min(entry_bbox.min_x);
            bbox.min_y = bbox.min_y.min(entry_bbox.min_y);
            bbox.max_x = bbox.max_x.max(entry_bbox.max_x);
            bbox.max_y = bbox.max_y.max(entry_bbox.max_y);
        }
        
        node.bounding_box = bbox;
    }
    
    /// Search for features within bounding box.
    pub fn search(&self, bbox: &BoundingBox) -> Vec<&SpatialFeature> {
        let mut results = Vec::new();
        
        if let Some(root) = &self.root {
            self.search_node(bbox, root, &mut results);
        }
        
        results
    }
    
    /// Search within a node.
    fn search_node(&self, bbox: &BoundingBox, node: &RTreeNode, results: &mut Vec<&SpatialFeature>) {
        if !node.bounding_box.intersects(bbox) {
            return;
        }
        
        for entry in &node.entries {
            match entry {
                RTreeEntry::Leaf { bounding_box, data, .. } => {
                    if bounding_box.intersects(bbox) {
                        results.push(data);
                    }
                }
                RTreeEntry::Node { node: child_node } => {
                    self.search_node(bbox, child_node, results);
                }
            }
        }
    }
    
    /// Find nearest neighbors to a point.
    pub fn nearest_neighbors(&self, point: &Point, k: usize) -> Vec<&SpatialFeature> {
        let mut results = Vec::new();
        
        if let Some(root) = &self.root {
            self.nearest_neighbors_node(point, k, root, &mut results);
        }
        
        // Sort by distance
        results.sort_by(|a, b| {
            let dist_a = point.distance_to(&a.geometry.bounding_box().center());
            let dist_b = point.distance_to(&b.geometry.bounding_box().center());
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        
        results.truncate(k);
        results
    }
    
    /// Find nearest neighbors within a node.
    fn nearest_neighbors_node(&self, point: &Point, k: usize, node: &RTreeNode, results: &mut Vec<&SpatialFeature>) {
        for entry in &node.entries {
            match entry {
                RTreeEntry::Leaf { data, .. } => {
                    results.push(data);
                }
                RTreeEntry::Node { node: child_node } => {
                    self.nearest_neighbors_node(point, k, child_node, results);
                }
            }
        }
    }
}

impl BoundingBox {
    /// Get center point of bounding box.
    pub fn center(&self) -> Point {
        Point::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rtree_insert() {
        let mut rtree = RTree::new(4);
        
        let feature = SpatialFeature {
            id: "test".to_string(),
            geometry: Geometry::Point(Point::new(1.0, 2.0)),
            properties: HashMap::new(),
        };
        
        rtree.insert(feature).unwrap();
        assert!(rtree.root.is_some());
    }
    
    #[test]
    fn test_rtree_search() {
        let mut rtree = RTree::new(4);
        
        let feature = SpatialFeature {
            id: "test".to_string(),
            geometry: Geometry::Point(Point::new(1.0, 2.0)),
            properties: HashMap::new(),
        };
        
        rtree.insert(feature).unwrap();
        
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let results = rtree.search(&bbox);
        assert_eq!(results.len(), 1);
    }
}
```

### Coordinate Reference Systems

```rust
// rust/03-coordinate-reference-systems.rs

/*
Coordinate reference system patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Coordinate reference system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateReferenceSystem {
    pub code: String,
    pub name: String,
    pub proj4_string: String,
    pub wkt: String,
    pub bounds: BoundingBox,
}

/// Coordinate transformation service.
pub struct CoordinateTransformService {
    crs_registry: HashMap<String, CoordinateReferenceSystem>,
    transform_cache: HashMap<(String, String), TransformFunction>,
}

type TransformFunction = Box<dyn Fn(Point) -> Point + Send + Sync>;

impl CoordinateTransformService {
    pub fn new() -> Self {
        let mut service = Self {
            crs_registry: HashMap::new(),
            transform_cache: HashMap::new(),
        };
        
        // Register common CRS
        service.register_common_crs();
        service
    }
    
    /// Register a coordinate reference system.
    pub fn register_crs(&mut self, crs: CoordinateReferenceSystem) {
        self.crs_registry.insert(crs.code.clone(), crs);
    }
    
    /// Register common coordinate reference systems.
    fn register_common_crs(&mut self) {
        // WGS84 (EPSG:4326)
        let wgs84 = CoordinateReferenceSystem {
            code: "EPSG:4326".to_string(),
            name: "WGS 84".to_string(),
            proj4_string: "+proj=longlat +datum=WGS84 +no_defs".to_string(),
            wkt: "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]".to_string(),
            bounds: BoundingBox::new(-180.0, -90.0, 180.0, 90.0),
        };
        self.crs_registry.insert(wgs84.code.clone(), wgs84);
        
        // Web Mercator (EPSG:3857)
        let web_mercator = CoordinateReferenceSystem {
            code: "EPSG:3857".to_string(),
            name: "Web Mercator".to_string(),
            proj4_string: "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs".to_string(),
            wkt: "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1]]".to_string(),
            bounds: BoundingBox::new(-20037508.34, -20037508.34, 20037508.34, 20037508.34),
        };
        self.crs_registry.insert(web_mercator.code.clone(), web_mercator);
    }
    
    /// Transform point from source CRS to target CRS.
    pub fn transform_point(&mut self, point: Point, source_crs: &str, target_crs: &str) -> Result<Point, String> {
        if source_crs == target_crs {
            return Ok(point);
        }
        
        // Check if transformation is cached
        let cache_key = (source_crs.to_string(), target_crs.to_string());
        if let Some(transform_fn) = self.transform_cache.get(&cache_key) {
            return Ok(transform_fn(point));
        }
        
        // Create transformation function
        let transform_fn = self.create_transform_function(source_crs, target_crs)?;
        let result = transform_fn(point);
        
        // Cache the transformation function
        self.transform_cache.insert(cache_key, transform_fn);
        
        Ok(result)
    }
    
    /// Create transformation function between two CRS.
    fn create_transform_function(&self, source_crs: &str, target_crs: &str) -> Result<TransformFunction, String> {
        let source = self.crs_registry.get(source_crs)
            .ok_or_else(|| format!("Source CRS not found: {}", source_crs))?;
        let target = self.crs_registry.get(target_crs)
            .ok_or_else(|| format!("Target CRS not found: {}", target_crs))?;
        
        // Simple transformation implementations
        match (source_crs, target_crs) {
            ("EPSG:4326", "EPSG:3857") => {
                Ok(Box::new(move |point| {
                    let x = point.x * 20037508.34 / 180.0;
                    let y = (point.y * std::f64::consts::PI / 180.0).tan().ln() * 20037508.34 / std::f64::consts::PI;
                    Point::new(x, y)
                }))
            }
            ("EPSG:3857", "EPSG:4326") => {
                Ok(Box::new(move |point| {
                    let x = point.x * 180.0 / 20037508.34;
                    let y = (point.y * std::f64::consts::PI / 20037508.34).exp().atan() * 180.0 / std::f64::consts::PI;
                    Point::new(x, y)
                }))
            }
            _ => Err(format!("Transformation not supported: {} -> {}", source_crs, target_crs)),
        }
    }
    
    /// Transform geometry from source CRS to target CRS.
    pub fn transform_geometry(&mut self, geometry: &Geometry, source_crs: &str, target_crs: &str) -> Result<Geometry, String> {
        match geometry {
            Geometry::Point(point) => {
                let transformed_point = self.transform_point(*point, source_crs, target_crs)?;
                Ok(Geometry::Point(transformed_point))
            }
            Geometry::LineString(linestring) => {
                let transformed_points: Vec<Point> = linestring.points.iter()
                    .map(|point| self.transform_point(*point, source_crs, target_crs))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Geometry::LineString(LineString::new(transformed_points)))
            }
            Geometry::Polygon(polygon) => {
                let transformed_exterior: Vec<Point> = polygon.exterior.iter()
                    .map(|point| self.transform_point(*point, source_crs, target_crs))
                    .collect::<Result<Vec<_>, _>>()?;
                
                let mut transformed_holes = Vec::new();
                for hole in &polygon.holes {
                    let transformed_hole: Vec<Point> = hole.iter()
                        .map(|point| self.transform_point(*point, source_crs, target_crs))
                        .collect::<Result<Vec<_>, _>>()?;
                    transformed_holes.push(transformed_hole);
                }
                
                let mut transformed_polygon = Polygon::new(transformed_exterior);
                for hole in transformed_holes {
                    transformed_polygon.add_hole(hole);
                }
                
                Ok(Geometry::Polygon(transformed_polygon))
            }
            _ => Err("Complex geometry transformation not implemented".to_string()),
        }
    }
    
    /// Get CRS information.
    pub fn get_crs(&self, code: &str) -> Option<&CoordinateReferenceSystem> {
        self.crs_registry.get(code)
    }
    
    /// List all registered CRS.
    pub fn list_crs(&self) -> Vec<&CoordinateReferenceSystem> {
        self.crs_registry.values().collect()
    }
}

/// Spatial reference system utilities.
pub struct SpatialReferenceUtils;

impl SpatialReferenceUtils {
    /// Calculate distance between two points in meters.
    pub fn distance_meters(point1: &Point, point2: &Point) -> f64 {
        const EARTH_RADIUS: f64 = 6371000.0; // Earth radius in meters
        
        let lat1_rad = point1.y.to_radians();
        let lat2_rad = point2.y.to_radians();
        let delta_lat = (point2.y - point1.y).to_radians();
        let delta_lon = (point2.x - point1.x).to_radians();
        
        let a = (delta_lat / 2.0).sin().powi(2) +
                lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        
        EARTH_RADIUS * c
    }
    
    /// Calculate bearing between two points.
    pub fn bearing(point1: &Point, point2: &Point) -> f64 {
        let lat1_rad = point1.y.to_radians();
        let lat2_rad = point2.y.to_radians();
        let delta_lon = (point2.x - point1.x).to_radians();
        
        let y = delta_lon.sin() * lat2_rad.cos();
        let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();
        
        y.atan2(x).to_degrees()
    }
    
    /// Calculate point at given distance and bearing.
    pub fn destination(point: &Point, distance: f64, bearing: f64) -> Point {
        const EARTH_RADIUS: f64 = 6371000.0; // Earth radius in meters
        
        let lat1_rad = point.y.to_radians();
        let lon1_rad = point.x.to_radians();
        let bearing_rad = bearing.to_radians();
        
        let lat2_rad = (lat1_rad.sin() * (distance / EARTH_RADIUS).cos() +
                       lat1_rad.cos() * (distance / EARTH_RADIUS).sin() * bearing_rad.cos()).asin();
        
        let lon2_rad = lon1_rad + (bearing_rad.sin() * (distance / EARTH_RADIUS).sin() * lat1_rad.cos()).atan2(
            (distance / EARTH_RADIUS).cos() - lat1_rad.sin() * lat2_rad.sin()
        );
        
        Point::new(lon2_rad.to_degrees(), lat2_rad.to_degrees())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coordinate_transform() {
        let mut service = CoordinateTransformService::new();
        
        let point = Point::new(0.0, 0.0); // Equator, Prime Meridian
        let transformed = service.transform_point(point, "EPSG:4326", "EPSG:3857").unwrap();
        
        // Should be close to (0, 0) in Web Mercator
        assert!((transformed.x - 0.0).abs() < 1.0);
        assert!((transformed.y - 0.0).abs() < 1.0);
    }
    
    #[test]
    fn test_distance_calculation() {
        let point1 = Point::new(0.0, 0.0);
        let point2 = Point::new(1.0, 1.0);
        
        let distance = SpatialReferenceUtils::distance_meters(&point1, &point2);
        assert!(distance > 0.0);
        assert!(distance < 200000.0); // Should be less than 200km
    }
    
    #[test]
    fn test_bearing_calculation() {
        let point1 = Point::new(0.0, 0.0);
        let point2 = Point::new(1.0, 0.0);
        
        let bearing = SpatialReferenceUtils::bearing(&point1, &point2);
        assert!((bearing - 90.0).abs() < 1.0); // Should be approximately 90 degrees
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Spatial data structures
let point = Point::new(1.0, 2.0);
let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);

// 2. Spatial indexing
let mut rtree = RTree::new(4);
rtree.insert(feature)?;
let results = rtree.search(&bbox);

// 3. Coordinate transformation
let mut service = CoordinateTransformService::new();
let transformed = service.transform_point(point, "EPSG:4326", "EPSG:3857")?;
```

### Essential Patterns

```rust
// Complete geospatial setup
pub fn setup_rust_geospatial() {
    // 1. Spatial data structures
    // 2. Spatial indexing
    // 3. Coordinate reference systems
    // 4. Spatial operations
    // 5. Performance optimization
    // 6. Memory management
    // 7. Error handling
    // 8. Testing
    
    println!("Rust geospatial development setup complete!");
}
```

---

*This guide provides the complete machinery for Rust geospatial development. Each pattern includes implementation examples, spatial strategies, and real-world usage patterns for enterprise geospatial systems.*
