# Go Geospatial Development Best Practices

**Objective**: Master senior-level Go geospatial development patterns for production systems. When you need to build location-aware applications, when you want to process spatial data efficiently, when you need enterprise-grade geospatial patterns—these best practices become your weapon of choice.

## Core Principles

- **Spatial Data Types**: Handle points, lines, polygons, and complex geometries
- **Spatial Indexing**: Optimize spatial queries with proper indexing
- **Coordinate Systems**: Manage different coordinate reference systems (CRS)
- **Spatial Operations**: Perform geometric calculations and spatial analysis
- **Performance Optimization**: Handle large spatial datasets efficiently

## Spatial Data Types

### Geometry Types

```go
// internal/geospatial/geometry.go
package geospatial

import (
    "encoding/json"
    "fmt"
    "math"
)

// Point represents a 2D point
type Point struct {
    X, Y float64
}

// NewPoint creates a new point
func NewPoint(x, y float64) *Point {
    return &Point{X: x, Y: y}
}

// Distance calculates the distance to another point
func (p *Point) Distance(other *Point) float64 {
    dx := p.X - other.X
    dy := p.Y - other.Y
    return math.Sqrt(dx*dx + dy*dy)
}

// MarshalJSON implements json.Marshaler
func (p *Point) MarshalJSON() ([]byte, error) {
    return json.Marshal([2]float64{p.X, p.Y})
}

// UnmarshalJSON implements json.Unmarshaler
func (p *Point) UnmarshalJSON(data []byte) error {
    var coords [2]float64
    if err := json.Unmarshal(data, &coords); err != nil {
        return err
    }
    p.X = coords[0]
    p.Y = coords[1]
    return nil
}

// LineString represents a line string
type LineString struct {
    Points []Point
}

// NewLineString creates a new line string
func NewLineString(points []Point) *LineString {
    return &LineString{Points: points}
}

// Length calculates the length of the line string
func (ls *LineString) Length() float64 {
    if len(ls.Points) < 2 {
        return 0
    }
    
    length := 0.0
    for i := 1; i < len(ls.Points); i++ {
        length += ls.Points[i-1].Distance(&ls.Points[i])
    }
    return length
}

// Polygon represents a polygon
type Polygon struct {
    ExteriorRing []Point
    InteriorRings [][]Point
}

// NewPolygon creates a new polygon
func NewPolygon(exteriorRing []Point, interiorRings [][]Point) *Polygon {
    return &Polygon{
        ExteriorRing:  exteriorRing,
        InteriorRings: interiorRings,
    }
}

// Area calculates the area of the polygon
func (p *Polygon) Area() float64 {
    return p.ringArea(p.ExteriorRing) - p.interiorRingsArea()
}

// ringArea calculates the area of a ring
func (p *Polygon) ringArea(ring []Point) float64 {
    if len(ring) < 3 {
        return 0
    }
    
    area := 0.0
    n := len(ring)
    
    for i := 0; i < n; i++ {
        j := (i + 1) % n
        area += ring[i].X * ring[j].Y
        area -= ring[j].X * ring[i].Y
    }
    
    return math.Abs(area) / 2.0
}

// interiorRingsArea calculates the area of interior rings
func (p *Polygon) interiorRingsArea() float64 {
    area := 0.0
    for _, ring := range p.InteriorRings {
        area += p.ringArea(ring)
    }
    return area
}

// Contains checks if the polygon contains a point
func (p *Polygon) Contains(point *Point) bool {
    if !p.ringContains(point, p.ExteriorRing, true) {
        return false
    }
    
    // Check if point is in any interior ring
    for _, ring := range p.InteriorRings {
        if p.ringContains(point, ring, false) {
            return false
        }
    }
    
    return true
}

// ringContains checks if a ring contains a point
func (p *Polygon) ringContains(point *Point, ring []Point, exterior bool) bool {
    if len(ring) < 3 {
        return false
    }
    
    inside := false
    n := len(ring)
    
    for i, j := 0, n-1; i < n; j, i = i, i+1 {
        if ((ring[i].Y > point.Y) != (ring[j].Y > point.Y)) &&
            (point.X < (ring[j].X-ring[i].X)*(point.Y-ring[i].Y)/(ring[j].Y-ring[i].Y)+ring[i].X) {
            inside = !inside
        }
    }
    
    return inside == exterior
}

// Geometry represents a generic geometry
type Geometry interface {
    Type() string
    Bounds() *Bounds
    Area() float64
    Length() float64
    Contains(other Geometry) bool
    Intersects(other Geometry) bool
}

// Bounds represents a bounding box
type Bounds struct {
    MinX, MinY, MaxX, MaxY float64
}

// NewBounds creates a new bounds
func NewBounds(minX, minY, maxX, maxY float64) *Bounds {
    return &Bounds{
        MinX: minX,
        MinY: minY,
        MaxX: maxX,
        MaxY: maxY,
    }
}

// Width returns the width of the bounds
func (b *Bounds) Width() float64 {
    return b.MaxX - b.MinX
}

// Height returns the height of the bounds
func (b *Bounds) Height() float64 {
    return b.MaxY - b.MinY
}

// Contains checks if the bounds contain a point
func (b *Bounds) Contains(point *Point) bool {
    return point.X >= b.MinX && point.X <= b.MaxX &&
           point.Y >= b.MinY && point.Y <= b.MaxY
}

// Intersects checks if the bounds intersect with another bounds
func (b *Bounds) Intersects(other *Bounds) bool {
    return !(b.MaxX < other.MinX || b.MinX > other.MaxX ||
             b.MaxY < other.MinY || b.MinY > other.MaxY)
}
```

### Coordinate Reference Systems

```go
// internal/geospatial/crs.go
package geospatial

import (
    "fmt"
    "math"
)

// CRS represents a coordinate reference system
type CRS struct {
    EPSGCode int
    Name     string
    Proj4    string
}

// Common CRS definitions
var (
    WGS84 = CRS{
        EPSGCode: 4326,
        Name:     "WGS 84",
        Proj4:    "+proj=longlat +datum=WGS84 +no_defs",
    }
    
    WebMercator = CRS{
        EPSGCode: 3857,
        Name:     "Web Mercator",
        Proj4:    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs",
    }
    
    UTMZone10N = CRS{
        EPSGCode: 32610,
        Name:     "UTM Zone 10N",
        Proj4:    "+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs",
    }
)

// Transform transforms coordinates between CRS
func Transform(point *Point, from, to CRS) (*Point, error) {
    if from.EPSGCode == to.EPSGCode {
        return point, nil
    }
    
    // Simple transformation for common cases
    if from.EPSGCode == 4326 && to.EPSGCode == 3857 {
        return transformWGS84ToWebMercator(point), nil
    }
    
    if from.EPSGCode == 3857 && to.EPSGCode == 4326 {
        return transformWebMercatorToWGS84(point), nil
    }
    
    return nil, fmt.Errorf("unsupported transformation from %d to %d", from.EPSGCode, to.EPSGCode)
}

// transformWGS84ToWebMercator transforms from WGS84 to Web Mercator
func transformWGS84ToWebMercator(point *Point) *Point {
    x := point.X * 20037508.34 / 180
    y := math.Log(math.Tan((90+point.Y)*math.Pi/360)) / (math.Pi/180)
    y = y * 20037508.34 / 180
    return &Point{X: x, Y: y}
}

// transformWebMercatorToWGS84 transforms from Web Mercator to WGS84
func transformWebMercatorToWGS84(point *Point) *Point {
    x := point.X / 20037508.34 * 180
    y := point.Y / 20037508.34 * 180
    y = 180/math.Pi * (2*math.Atan(math.Exp(y*math.Pi/180)) - math.Pi/2)
    return &Point{X: x, Y: y}
}
```

## Spatial Indexing

### R-Tree Index

```go
// internal/geospatial/rtree.go
package geospatial

import (
    "fmt"
    "math"
    "sort"
)

// RTreeNode represents a node in the R-tree
type RTreeNode struct {
    Bounds    *Bounds
    Children  []*RTreeNode
    Data      []Geometry
    IsLeaf    bool
    MaxEntries int
}

// NewRTreeNode creates a new R-tree node
func NewRTreeNode(maxEntries int, isLeaf bool) *RTreeNode {
    return &RTreeNode{
        Children:   make([]*RTreeNode, 0),
        Data:      make([]Geometry, 0),
        IsLeaf:    isLeaf,
        MaxEntries: maxEntries,
    }
}

// Insert inserts a geometry into the R-tree
func (n *RTreeNode) Insert(geometry Geometry) error {
    if n.IsLeaf {
        n.Data = append(n.Data, geometry)
        n.updateBounds()
        
        if len(n.Data) > n.MaxEntries {
            return n.split()
        }
        return nil
    }
    
    // Find the best child to insert into
    bestChild := n.findBestChild(geometry.Bounds())
    
    if err := bestChild.Insert(geometry); err != nil {
        return err
    }
    
    n.updateBounds()
    return nil
}

// Search searches for geometries that intersect with the given bounds
func (n *RTreeNode) Search(bounds *Bounds) []Geometry {
    var results []Geometry
    
    if !n.Bounds.Intersects(bounds) {
        return results
    }
    
    if n.IsLeaf {
        for _, geometry := range n.Data {
            if geometry.Bounds().Intersects(bounds) {
                results = append(results, geometry)
            }
        }
    } else {
        for _, child := range n.Children {
            results = append(results, child.Search(bounds)...)
        }
    }
    
    return results
}

// findBestChild finds the best child to insert into
func (n *RTreeNode) findBestChild(bounds *Bounds) *RTreeNode {
    if len(n.Children) == 0 {
        return nil
    }
    
    bestChild := n.Children[0]
    bestArea := n.Children[0].Bounds.union(bounds).Area()
    
    for _, child := range n.Children[1:] {
        area := child.Bounds.union(bounds).Area()
        if area < bestArea {
            bestArea = area
            bestChild = child
        }
    }
    
    return bestChild
}

// split splits a leaf node when it exceeds max entries
func (n *RTreeNode) split() error {
    if !n.IsLeaf {
        return fmt.Errorf("cannot split non-leaf node")
    }
    
    // Simple split: divide data in half
    mid := len(n.Data) / 2
    
    leftData := n.Data[:mid]
    rightData := n.Data[mid:]
    
    n.Data = leftData
    
    rightChild := NewRTreeNode(n.MaxEntries, true)
    rightChild.Data = rightData
    rightChild.updateBounds()
    
    n.Children = append(n.Children, rightChild)
    n.IsLeaf = false
    
    n.updateBounds()
    return nil
}

// updateBounds updates the bounds of the node
func (n *RTreeNode) updateBounds() {
    if n.IsLeaf {
        if len(n.Data) == 0 {
            n.Bounds = nil
            return
        }
        
        bounds := n.Data[0].Bounds()
        for _, geometry := range n.Data[1:] {
            bounds = bounds.union(geometry.Bounds())
        }
        n.Bounds = bounds
    } else {
        if len(n.Children) == 0 {
            n.Bounds = nil
            return
        }
        
        bounds := n.Children[0].Bounds
        for _, child := range n.Children[1:] {
            bounds = bounds.union(child.Bounds)
        }
        n.Bounds = bounds
    }
}

// Area returns the area of the bounds
func (b *Bounds) Area() float64 {
    return b.Width() * b.Height()
}

// union returns the union of two bounds
func (b *Bounds) union(other *Bounds) *Bounds {
    if b == nil {
        return other
    }
    if other == nil {
        return b
    }
    
    return &Bounds{
        MinX: math.Min(b.MinX, other.MinX),
        MinY: math.Min(b.MinY, other.MinY),
        MaxX: math.Max(b.MaxX, other.MaxX),
        MaxY: math.Max(b.MaxY, other.MaxY),
    }
}

// RTree represents an R-tree spatial index
type RTree struct {
    Root *RTreeNode
    MaxEntries int
}

// NewRTree creates a new R-tree
func NewRTree(maxEntries int) *RTree {
    return &RTree{
        Root:       NewRTreeNode(maxEntries, true),
        MaxEntries: maxEntries,
    }
}

// Insert inserts a geometry into the R-tree
func (rt *RTree) Insert(geometry Geometry) error {
    return rt.Root.Insert(geometry)
}

// Search searches for geometries that intersect with the given bounds
func (rt *RTree) Search(bounds *Bounds) []Geometry {
    return rt.Root.Search(bounds)
}

// NearestNeighbor finds the nearest neighbor to a point
func (rt *RTree) NearestNeighbor(point *Point) Geometry {
    bounds := &Bounds{
        MinX: point.X - 0.001,
        MinY: point.Y - 0.001,
        MaxX: point.X + 0.001,
        MaxY: point.Y + 0.001,
    }
    
    candidates := rt.Search(bounds)
    if len(candidates) == 0 {
        return nil
    }
    
    nearest := candidates[0]
    minDistance := point.Distance(nearest.(*Point))
    
    for _, candidate := range candidates[1:] {
        distance := point.Distance(candidate.(*Point))
        if distance < minDistance {
            minDistance = distance
            nearest = candidate
        }
    }
    
    return nearest
}
```

### Spatial Hash

```go
// internal/geospatial/spatial_hash.go
package geospatial

import (
    "fmt"
    "math"
)

// SpatialHash represents a spatial hash index
type SpatialHash struct {
    CellSize float64
    Cells    map[string][]Geometry
}

// NewSpatialHash creates a new spatial hash
func NewSpatialHash(cellSize float64) *SpatialHash {
    return &SpatialHash{
        CellSize: cellSize,
        Cells:    make(map[string][]Geometry),
    }
}

// Insert inserts a geometry into the spatial hash
func (sh *SpatialHash) Insert(geometry Geometry) {
    bounds := geometry.Bounds()
    
    minX := math.Floor(bounds.MinX / sh.CellSize)
    maxX := math.Ceil(bounds.MaxX / sh.CellSize)
    minY := math.Floor(bounds.MinY / sh.CellSize)
    maxY := math.Ceil(bounds.MaxY / sh.CellSize)
    
    for x := minX; x <= maxX; x++ {
        for y := minY; y <= maxY; y++ {
            key := fmt.Sprintf("%.0f,%.0f", x, y)
            sh.Cells[key] = append(sh.Cells[key], geometry)
        }
    }
}

// Search searches for geometries that intersect with the given bounds
func (sh *SpatialHash) Search(bounds *Bounds) []Geometry {
    var results []Geometry
    seen := make(map[Geometry]bool)
    
    minX := math.Floor(bounds.MinX / sh.CellSize)
    maxX := math.Ceil(bounds.MaxX / sh.CellSize)
    minY := math.Floor(bounds.MinY / sh.CellSize)
    maxY := math.Ceil(bounds.MaxY / sh.CellSize)
    
    for x := minX; x <= maxX; x++ {
        for y := minY; y <= maxY; y++ {
            key := fmt.Sprintf("%.0f,%.0f", x, y)
            if geometries, exists := sh.Cells[key]; exists {
                for _, geometry := range geometries {
                    if !seen[geometry] && geometry.Bounds().Intersects(bounds) {
                        results = append(results, geometry)
                        seen[geometry] = true
                    }
                }
            }
        }
    }
    
    return results
}

// Clear clears the spatial hash
func (sh *SpatialHash) Clear() {
    sh.Cells = make(map[string][]Geometry)
}
```

## Spatial Operations

### Spatial Analysis

```go
// internal/geospatial/spatial_analysis.go
package geospatial

import (
    "math"
    "sort"
)

// SpatialAnalyzer provides spatial analysis functions
type SpatialAnalyzer struct{}

// NewSpatialAnalyzer creates a new spatial analyzer
func NewSpatialAnalyzer() *SpatialAnalyzer {
    return &SpatialAnalyzer{}
}

// ConvexHull calculates the convex hull of a set of points
func (sa *SpatialAnalyzer) ConvexHull(points []Point) []Point {
    if len(points) < 3 {
        return points
    }
    
    // Sort points by x-coordinate, then by y-coordinate
    sortedPoints := make([]Point, len(points))
    copy(sortedPoints, points)
    sort.Slice(sortedPoints, func(i, j int) bool {
        if sortedPoints[i].X == sortedPoints[j].X {
            return sortedPoints[i].Y < sortedPoints[j].Y
        }
        return sortedPoints[i].X < sortedPoints[j].X
    })
    
    // Build lower hull
    lower := make([]Point, 0)
    for _, point := range sortedPoints {
        for len(lower) >= 2 && sa.cross(lower[len(lower)-2], lower[len(lower)-1], point) <= 0 {
            lower = lower[:len(lower)-1]
        }
        lower = append(lower, point)
    }
    
    // Build upper hull
    upper := make([]Point, 0)
    for i := len(sortedPoints) - 1; i >= 0; i-- {
        for len(upper) >= 2 && sa.cross(upper[len(upper)-2], upper[len(upper)-1], sortedPoints[i]) <= 0 {
            upper = upper[:len(upper)-1]
        }
        upper = append(upper, sortedPoints[i])
    }
    
    // Remove duplicate points
    hull := make([]Point, 0)
    hull = append(hull, lower...)
    hull = append(hull, upper[1:len(upper)-1]...)
    
    return hull
}

// cross calculates the cross product
func (sa *SpatialAnalyzer) cross(o, a, b Point) float64 {
    return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
}

// Centroid calculates the centroid of a set of points
func (sa *SpatialAnalyzer) Centroid(points []Point) Point {
    if len(points) == 0 {
        return Point{}
    }
    
    var sumX, sumY float64
    for _, point := range points {
        sumX += point.X
        sumY += point.Y
    }
    
    return Point{
        X: sumX / float64(len(points)),
        Y: sumY / float64(len(points)),
    }
}

// BoundingBox calculates the bounding box of a set of points
func (sa *SpatialAnalyzer) BoundingBox(points []Point) *Bounds {
    if len(points) == 0 {
        return nil
    }
    
    minX, minY := points[0].X, points[0].Y
    maxX, maxY := points[0].X, points[0].Y
    
    for _, point := range points[1:] {
        if point.X < minX {
            minX = point.X
        }
        if point.X > maxX {
            maxX = point.X
        }
        if point.Y < minY {
            minY = point.Y
        }
        if point.Y > maxY {
            maxY = point.Y
        }
    }
    
    return &Bounds{
        MinX: minX,
        MinY: minY,
        MaxX: maxX,
        MaxY: maxY,
    }
}

// DistanceMatrix calculates the distance matrix between points
func (sa *SpatialAnalyzer) DistanceMatrix(points []Point) [][]float64 {
    n := len(points)
    matrix := make([][]float64, n)
    
    for i := 0; i < n; i++ {
        matrix[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            if i == j {
                matrix[i][j] = 0
            } else {
                matrix[i][j] = points[i].Distance(&points[j])
            }
        }
    }
    
    return matrix
}

// NearestNeighbors finds the k nearest neighbors of a point
func (sa *SpatialAnalyzer) NearestNeighbors(point *Point, points []Point, k int) []Point {
    if k >= len(points) {
        return points
    }
    
    // Calculate distances
    distances := make([]struct {
        point    Point
        distance float64
    }, len(points))
    
    for i, p := range points {
        distances[i] = struct {
            point    Point
            distance float64
        }{
            point:    p,
            distance: point.Distance(&p),
        }
    }
    
    // Sort by distance
    sort.Slice(distances, func(i, j int) bool {
        return distances[i].distance < distances[j].distance
    })
    
    // Return k nearest neighbors
    result := make([]Point, k)
    for i := 0; i < k; i++ {
        result[i] = distances[i].point
    }
    
    return result
}
```

### Spatial Queries

```go
// internal/geospatial/spatial_queries.go
package geospatial

import (
    "context"
    "database/sql"
    "fmt"
)

// SpatialQueries provides spatial query functions
type SpatialQueries struct {
    db *sql.DB
}

// NewSpatialQueries creates a new spatial queries instance
func NewSpatialQueries(db *sql.DB) *SpatialQueries {
    return &SpatialQueries{db: db}
}

// FindWithinDistance finds geometries within a distance of a point
func (sq *SpatialQueries) FindWithinDistance(ctx context.Context, table string, point *Point, distance float64) ([]Geometry, error) {
    query := fmt.Sprintf(`
        SELECT id, ST_AsText(geom) as wkt
        FROM %s
        WHERE ST_DWithin(geom, ST_GeomFromText('POINT(%f %f)', 4326), %f)
    `, table, point.X, point.Y, distance)
    
    rows, err := sq.db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var geometries []Geometry
    for rows.Next() {
        var id int64
        var wkt string
        if err := rows.Scan(&id, &wkt); err != nil {
            return nil, err
        }
        
        // Parse WKT and create geometry
        geometry, err := sq.parseWKT(wkt)
        if err != nil {
            continue
        }
        
        geometries = append(geometries, geometry)
    }
    
    return geometries, nil
}

// FindIntersecting finds geometries that intersect with a given geometry
func (sq *SpatialQueries) FindIntersecting(ctx context.Context, table string, geometry Geometry) ([]Geometry, error) {
    wkt := sq.geometryToWKT(geometry)
    
    query := fmt.Sprintf(`
        SELECT id, ST_AsText(geom) as wkt
        FROM %s
        WHERE ST_Intersects(geom, ST_GeomFromText('%s', 4326))
    `, table, wkt)
    
    rows, err := sq.db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var geometries []Geometry
    for rows.Next() {
        var id int64
        var wkt string
        if err := rows.Scan(&id, &wkt); err != nil {
            return nil, err
        }
        
        // Parse WKT and create geometry
        geometry, err := sq.parseWKT(wkt)
        if err != nil {
            continue
        }
        
        geometries = append(geometries, geometry)
    }
    
    return geometries, nil
}

// FindContained finds geometries that are contained within a given geometry
func (sq *SpatialQueries) FindContained(ctx context.Context, table string, geometry Geometry) ([]Geometry, error) {
    wkt := sq.geometryToWKT(geometry)
    
    query := fmt.Sprintf(`
        SELECT id, ST_AsText(geom) as wkt
        FROM %s
        WHERE ST_Contains(ST_GeomFromText('%s', 4326), geom)
    `, table, wkt)
    
    rows, err := sq.db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var geometries []Geometry
    for rows.Next() {
        var id int64
        var wkt string
        if err := rows.Scan(&id, &wkt); err != nil {
            return nil, err
        }
        
        // Parse WKT and create geometry
        geometry, err := sq.parseWKT(wkt)
        if err != nil {
            continue
        }
        
        geometries = append(geometries, geometry)
    }
    
    return geometries, nil
}

// parseWKT parses a WKT string into a geometry
func (sq *SpatialQueries) parseWKT(wkt string) (Geometry, error) {
    // Simple WKT parser for points
    if wkt[:6] == "POINT(" {
        // Parse point coordinates
        coords := wkt[6 : len(wkt)-1]
        var x, y float64
        if _, err := fmt.Sscanf(coords, "%f %f", &x, &y); err != nil {
            return nil, err
        }
        return NewPoint(x, y), nil
    }
    
    return nil, fmt.Errorf("unsupported WKT type: %s", wkt)
}

// geometryToWKT converts a geometry to WKT
func (sq *SpatialQueries) geometryToWKT(geometry Geometry) string {
    switch g := geometry.(type) {
    case *Point:
        return fmt.Sprintf("POINT(%f %f)", g.X, g.Y)
    default:
        return ""
    }
}
```

## Performance Optimization

### Spatial Index Optimization

```go
// internal/geospatial/performance.go
package geospatial

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

// SpatialIndexOptimizer provides spatial index optimization
type SpatialIndexOptimizer struct {
    db *sql.DB
}

// NewSpatialIndexOptimizer creates a new spatial index optimizer
func NewSpatialIndexOptimizer(db *sql.DB) *SpatialIndexOptimizer {
    return &SpatialIndexOptimizer{db: db}
}

// CreateSpatialIndex creates a spatial index on a table
func (sio *SpatialIndexOptimizer) CreateSpatialIndex(ctx context.Context, table, column, indexName string) error {
    query := fmt.Sprintf("CREATE INDEX %s ON %s USING GIST (%s)", indexName, table, column)
    
    _, err := sio.db.ExecContext(ctx, query)
    return err
}

// AnalyzeTable analyzes a table for query optimization
func (sio *SpatialIndexOptimizer) AnalyzeTable(ctx context.Context, table string) error {
    query := fmt.Sprintf("ANALYZE %s", table)
    _, err := sio.db.ExecContext(ctx, query)
    return err
}

// VacuumTable vacuums a table to reclaim space
func (sio *SpatialIndexOptimizer) VacuumTable(ctx context.Context, table string) error {
    query := fmt.Sprintf("VACUUM ANALYZE %s", table)
    _, err := sio.db.ExecContext(ctx, query)
    return err
}

// GetTableStats gets table statistics
func (sio *SpatialIndexOptimizer) GetTableStats(ctx context.Context, table string) (*TableStats, error) {
    query := `
        SELECT 
            schemaname,
            tablename,
            attname,
            n_distinct,
            correlation
        FROM pg_stats
        WHERE tablename = $1
    `
    
    rows, err := sio.db.QueryContext(ctx, query, table)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    stats := &TableStats{
        TableName: table,
        Columns:   make(map[string]*ColumnStats),
    }
    
    for rows.Next() {
        var schemaName, tableName, attName string
        var nDistinct float64
        var correlation float64
        
        if err := rows.Scan(&schemaName, &tableName, &attName, &nDistinct, &correlation); err != nil {
            return nil, err
        }
        
        stats.Columns[attName] = &ColumnStats{
            Name:         attName,
            NDistinct:    nDistinct,
            Correlation:  correlation,
        }
    }
    
    return stats, nil
}

// TableStats represents table statistics
type TableStats struct {
    TableName string                  `json:"table_name"`
    Columns   map[string]*ColumnStats `json:"columns"`
}

// ColumnStats represents column statistics
type ColumnStats struct {
    Name        string  `json:"name"`
    NDistinct   float64 `json:"n_distinct"`
    Correlation float64 `json:"correlation"`
}

// SpatialQueryProfiler provides spatial query profiling
type SpatialQueryProfiler struct {
    db *sql.DB
}

// NewSpatialQueryProfiler creates a new spatial query profiler
func NewSpatialQueryProfiler(db *sql.DB) *SpatialQueryProfiler {
    return &SpatialQueryProfiler{db: db}
}

// ProfileQuery profiles a spatial query
func (sqp *SpatialQueryProfiler) ProfileQuery(ctx context.Context, query string) (*QueryProfile, error) {
    // Enable query profiling
    if _, err := sqp.db.ExecContext(ctx, "SET enable_seqscan = off"); err != nil {
        return nil, err
    }
    
    start := time.Now()
    
    rows, err := sqp.db.QueryContext(ctx, query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    // Count rows
    rowCount := 0
    for rows.Next() {
        rowCount++
    }
    
    duration := time.Since(start)
    
    // Get query plan
    plan, err := sqp.getQueryPlan(ctx, query)
    if err != nil {
        return nil, err
    }
    
    return &QueryProfile{
        Query:     query,
        Duration:  duration,
        RowCount:  rowCount,
        Plan:      plan,
    }, nil
}

// getQueryPlan gets the query execution plan
func (sqp *SpatialQueryProfiler) getQueryPlan(ctx context.Context, query string) (string, error) {
    explainQuery := fmt.Sprintf("EXPLAIN ANALYZE %s", query)
    
    row := sqp.db.QueryRowContext(ctx, explainQuery)
    
    var plan string
    if err := row.Scan(&plan); err != nil {
        return "", err
    }
    
    return plan, nil
}

// QueryProfile represents a query profile
type QueryProfile struct {
    Query    string        `json:"query"`
    Duration time.Duration `json:"duration"`
    RowCount int           `json:"row_count"`
    Plan     string        `json:"plan"`
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. Spatial data types
point := NewPoint(-122.4194, 37.7749) // San Francisco
lineString := NewLineString([]Point{point1, point2, point3})
polygon := NewPolygon(exteriorRing, interiorRings)

// 2. Coordinate transformations
transformedPoint, err := Transform(point, WGS84, WebMercator)

// 3. Spatial indexing
rtree := NewRTree(10)
rtree.Insert(geometry)
results := rtree.Search(bounds)

// 4. Spatial analysis
analyzer := NewSpatialAnalyzer()
hull := analyzer.ConvexHull(points)
centroid := analyzer.Centroid(points)

// 5. Spatial queries
spatialQueries := NewSpatialQueries(db)
geometries, err := spatialQueries.FindWithinDistance(ctx, "locations", point, 1000.0)
```

### Essential Patterns

```go
// Spatial hash indexing
spatialHash := NewSpatialHash(100.0) // 100m cells
spatialHash.Insert(geometry)
results := spatialHash.Search(bounds)

// Performance optimization
optimizer := NewSpatialIndexOptimizer(db)
err := optimizer.CreateSpatialIndex(ctx, "locations", "geom", "idx_locations_geom")
err = optimizer.AnalyzeTable(ctx, "locations")

// Query profiling
profiler := NewSpatialQueryProfiler(db)
profile, err := profiler.ProfileQuery(ctx, "SELECT * FROM locations WHERE ST_DWithin(geom, $1, $2)")
```

---

*This guide provides the complete machinery for building geospatial applications in Go. Each pattern includes implementation examples, performance considerations, and real-world usage patterns for enterprise deployment.*
