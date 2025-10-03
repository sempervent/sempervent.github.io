# Geospatial Knowledge Graph: From Raw Geometry to Reasoning

**Objective**: Build a geospatial knowledge graph that understands place relationships. Not just points on a map, but connections: roads connecting cities, facilities within flood zones, sensor streams near pipelines, routes avoiding hazards. This tutorial combines PostGIS (heavy geometry math) with Neo4j (graph model + pathfinding/semantics) to create an intelligent geospatial system.

When you need to understand spatial relationships at scale, when you want to query "shortest safe route avoiding flood polygons within 24h", when you're building location-aware applications that reason about connectivity and proximity—geospatial knowledge graphs become your weapon of choice.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand the two-tier architecture**
   - PostGIS for authoritative geometry and spatial operations
   - Neo4j for semantic relationships and graph analytics
   - ETL bridge for computing and materializing relationships

2. **Master spatial relationships**
   - WITHIN, INTERSECTS, nearest neighbors
   - Road graph weights and connectivity
   - Distance calculations and proximity analysis

3. **Know your graph patterns**
   - Friend-of-a-friend associations
   - Shortest path algorithms
   - Centrality and community detection
   - Temporal and conditional relationships

4. **Validate everything**
   - Geometry validity and spatial indexing
   - Graph consistency and relationship integrity
   - Performance under scale
   - Data quality and provenance

5. **Plan for production**
   - Scalable ETL pipelines
   - Monitoring and observability
   - Security and access control
   - Integration with existing systems

**Why These Principles**: Geospatial knowledge graphs provide the foundation for intelligent location-aware applications. Understanding spatial relationships, mastering graph patterns, and following best practices is essential for building production-ready geospatial systems.

## 1) What is a Geospatial Knowledge Graph? (The Foundation)

### Core Concepts

```yaml
# Geospatial knowledge graphs combine spatial and semantic reasoning
geospatial_kg:
  spatial_layer:
    - "PostGIS for authoritative geometry"
    - "Spatial indexing and operations"
    - "Distance calculations and proximity"
    - "Geometric relationship computation"
  
  semantic_layer:
    - "Neo4j for relationship modeling"
    - "Graph analytics and pathfinding"
    - "Semantic reasoning and inference"
    - "Query optimization and caching"
  
  automation:
    - "ETL bridge for relationship computation"
    - "Automated spatial relationship discovery"
    - "Dynamic graph materialization"
    - "Real-time spatial reasoning"
```

### Architecture Overview

```mermaid
graph TB
    subgraph "Data Sources"
        A[Raw Geo Data]
        B[OpenStreetMap]
        C[Sensor Streams]
        D[Real-time Events]
    end
    
    subgraph "Spatial Processing Layer"
        E[(PostGIS)]
        F[Spatial Indexes]
        G[Geometry Operations]
        H[Distance Calculations]
    end
    
    subgraph "ETL Bridge"
        I[Relationship Computation]
        J[Spatial Queries]
        K[Data Transformation]
        L[Graph Materialization]
    end
    
    subgraph "Graph Layer"
        M[(Neo4j)]
        N[Graph Analytics]
        O[Pathfinding]
        P[Semantic Reasoning]
    end
    
    subgraph "Application Layer"
        Q[Spatial Queries]
        R[Route Optimization]
        S[Proximity Analysis]
        T[Impact Assessment]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    
    style E fill:#064,stroke:#0a0,color:#fff
    style M fill:#233,stroke:#59f,color:#fff
    style I fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Q fill:#fce4ec,stroke:#880e4f,stroke-width:2px
```

### Why This Architecture Matters

```cypher
// Geospatial knowledge graphs enable complex spatial reasoning
MATCH (c:City)-[:CONNECTED_TO {km: distance}]->(other:City)
WHERE distance < 50
WITH c, collect(other) as nearby_cities
MATCH (c)-[:NEAR]->(f:Facility)
WHERE f.kind = 'hospital'
RETURN c.name, nearby_cities, f.name as nearest_hospital

// This query combines:
// - Graph connectivity (CONNECTED_TO)
// - Spatial proximity (NEAR)
// - Semantic filtering (hospital)
// - Distance constraints (50km)
```

**Why Geospatial Knowledge Graphs Matter**: They enable applications to understand not just "where" but "how and why" things are connected. They provide the foundation for intelligent location-aware systems that can reason about spatial relationships and optimize routes, assess impacts, and discover patterns.

## 2) Docker Compose Setup (The Infrastructure)

### Complete Stack Configuration

```yaml
# docker-compose.yml
version: "3.9"
name: geospatial-kg

services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geo
      POSTGRES_USER: geo
      POSTGRES_PASSWORD: geo
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U geo -d geo"]
      interval: 5s
      timeout: 3s
      retries: 20
    volumes:
      - pgdata:/var/lib/postgresql/data
    profiles: ["postgis", "full"]

  neo4j:
    image: neo4j:5.23
    environment:
      NEO4J_AUTH: neo4j/test123
      NEO4J_server_memory_heap_initial__size: 1G
      NEO4J_server_memory_heap_max__size: 2G
      NEO4J_server_default__advertised__address: localhost
      NEO4J_PLUGINS: '["graph-data-science"]'
    ports:
      - "7474:7474"   # Browser
      - "7687:7687"   # Bolt
    volumes:
      - neo4jdata:/data
    profiles: ["neo4j", "full"]

  etl:
    image: python:3.11-slim
    depends_on:
      postgis:
        condition: service_healthy
      neo4j:
        condition: service_started
    working_dir: /app
    volumes:
      - ./scripts:/app/scripts
    command: >
      bash -lc "
        pip install --no-cache-dir psycopg[binary] shapely==2.0.4 geojson neo4j==5.23.0 numpy &&
        python scripts/gen_data.py &&
        python scripts/load_postgis.py &&
        python scripts/build_kg.py &&
        tail -f /dev/null
      "
    profiles: ["etl", "full"]

volumes:
  pgdata: {}
  neo4jdata: {}
```

### Stack Management

```bash
# Start the complete stack
docker compose --profile full up -d

# Start individual services
docker compose --profile postgis up -d
docker compose --profile neo4j up -d
docker compose --profile etl up -d

# Monitor logs
docker compose logs -f etl

# Access services
# Neo4j Browser: http://localhost:7474 (neo4j / test123)
# PostGIS: localhost:5432 (geo / geo)
```

**Why This Setup Matters**: The two-tier architecture separates concerns: PostGIS handles geometry, Neo4j handles relationships. This enables each system to excel at what it does best while maintaining data consistency and performance.

## 3) Data Generation (The Foundation)

### Synthetic Geospatial Data

```python
# scripts/gen_data.py
import json, random
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, mapping

random.seed(7)
out = Path("scripts/out")
out.mkdir(parents=True, exist_ok=True)

# Cities (random around SF Bay area)
bbox = (-122.7, 37.2, -121.7, 37.9)
def rand_pt():
    x = random.uniform(bbox[0], bbox[2])
    y = random.uniform(bbox[1], bbox[3])
    return (x, y)

cities = []
for i, name in enumerate(["Alpha","Bravo","Charlie","Delta","Echo","Foxtrot","Golf","Hotel","India","Juliet"], start=1):
    x, y = rand_pt()
    cities.append({"id": f"C{i}", "name": name, "geom": mapping(Point(x, y))})

# Facilities near cities
facilities = []
for i in range(1, 21):
    c = random.choice(cities)
    px, py = c["geom"]["coordinates"]
    # Jitter ~0.01 degrees (~1km)
    fx, fy = px + random.uniform(-0.01, 0.01), py + random.uniform(-0.01, 0.01)
    facilities.append({
        "id": f"F{i}",
        "kind": random.choice(["warehouse", "hospital", "substation"]),
        "geom": mapping(Point(fx, fy))
    })

# Regions (polygons)
regions = []
for i, nm in enumerate(["NorthZone", "SouthZone", "EastPark", "WestReserve"], start=1):
    cx, cy = rand_pt()
    poly = Point(cx, cy).buffer(0.08, resolution=32)  # ~few km radius
    regions.append({
        "id": f"R{i}",
        "name": nm,
        "geom": mapping(Polygon(poly.exterior.coords))
    })

# Roads: connect random pairs of cities
roads = []
pairs = set()
for i in range(1, 20):
    a, b = random.sample(cities, 2)
    key = tuple(sorted([a["id"], b["id"]]))
    if key in pairs:
        continue
    pairs.add(key)
    pa = a["geom"]["coordinates"]
    pb = b["geom"]["coordinates"]
    # Slightly bend the line
    mid = (
        (pa[0] + pb[0]) / 2 + random.uniform(-0.02, 0.02),
        (pa[1] + pb[1]) / 2 + random.uniform(-0.02, 0.02)
    )
    geom = LineString([pa, mid, pb])
    roads.append({
        "id": f"E{i}",
        "a": a["id"],
        "b": b["id"],
        "geom": mapping(geom)
    })

# Export as GeoJSON
for name, data in [("cities", cities), ("facilities", facilities), ("regions", regions), ("roads", roads)]:
    Path(out / f"{name}.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {k: v for k, v in d.items() if k != 'geom'},
                "geometry": d["geom"]
            } for d in data]
        }, indent=2)
    )

print("Generated scripts/out/*.geojson")
```

### Data Model Schema

```mermaid
erDiagram
    City ||--o{ Road : "CONNECTS"
    City {
        string id
        string name
        point location
    }
    Facility }o--|| Region : "WITHIN"
    City ||--o{ Facility : "NEAR"
    Road {
        string id
        float km
        string a
        string b
    }
    Region {
        string id
        string name
        polygon geom
    }
    Facility {
        string id
        string kind
        point location
    }
```

**Why This Data Model Matters**: The schema captures essential geospatial relationships: connectivity (roads), containment (facilities within regions), proximity (cities near facilities), and hierarchy (cities connected by roads). This provides the foundation for complex spatial reasoning.

## 4) PostGIS Loading (The Spatial Foundation)

### Database Schema and Indexing

```python
# scripts/load_postgis.py
import json, os, time
import psycopg
from pathlib import Path

dsn = "postgresql://geo:geo@postgis:5432/geo"

schema_sql = """
CREATE EXTENSION IF NOT EXISTS postgis;

-- Drop existing tables
DROP TABLE IF EXISTS cities CASCADE;
DROP TABLE IF EXISTS facilities CASCADE;
DROP TABLE IF EXISTS regions CASCADE;
DROP TABLE IF EXISTS roads CASCADE;

-- Create tables with proper geometry types
CREATE TABLE cities(
    id text primary key, 
    name text, 
    geom geometry(Point, 4326)
);

CREATE TABLE facilities(
    id text primary key, 
    kind text, 
    geom geometry(Point, 4326)
);

CREATE TABLE regions(
    id text primary key, 
    name text, 
    geom geometry(Polygon, 4326)
);

CREATE TABLE roads(
    id text primary key, 
    a text, 
    b text, 
    geom geometry(LineString, 4326)
);

-- Create spatial indexes for performance
CREATE INDEX ON cities USING GIST (geom);
CREATE INDEX ON facilities USING GIST (geom);
CREATE INDEX ON regions USING GIST (geom);
CREATE INDEX ON roads USING GIST (geom);

-- Create additional indexes for common queries
CREATE INDEX ON cities (name);
CREATE INDEX ON facilities (kind);
CREATE INDEX ON regions (name);
"""

def load(fc_path, table, cols):
    """Load GeoJSON features into PostGIS"""
    fc = json.loads(Path(fc_path).read_text())
    rows = []
    for f in fc["features"]:
        props = f["properties"]
        geom = json.dumps(f["geometry"])
        rows.append(tuple([props[c] for c in cols] + [geom]))
    return rows

# Load data into PostGIS
with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(schema_sql)
        conn.commit()

    paths = Path("scripts/out")
    with conn.cursor() as cur:
        # Cities
        for rid, name, geom in load(paths / "cities.geojson", "cities", ["id", "name"]):
            cur.execute(
                "INSERT INTO cities(id, name, geom) VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (rid, name, geom)
            )
        
        # Facilities
        for rid, kind, geom in load(paths / "facilities.geojson", "facilities", ["id", "kind"]):
            cur.execute(
                "INSERT INTO facilities(id, kind, geom) VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (rid, kind, geom)
            )
        
        # Regions
        for rid, name, geom in load(paths / "regions.geojson", "regions", ["id", "name"]):
            cur.execute(
                "INSERT INTO regions(id, name, geom) VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (rid, name, geom)
            )
        
        # Roads
        for rid, a, b, geom in load(paths / "roads.geojson", "roads", ["id", "a", "b"]):
            cur.execute(
                "INSERT INTO roads(id, a, b, geom) VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
                (rid, a, b, geom)
            )
    
    conn.commit()

print("Loaded GeoJSON into PostGIS with spatial indexes")
```

### Spatial Relationship Queries

```sql
-- Find facilities within regions
SELECT f.id, r.id, r.name
FROM facilities f
JOIN regions r ON ST_Within(f.geom, r.geom);

-- Find facilities near cities (within 2km)
SELECT c.id AS city, f.id AS facility, 
       ST_DistanceSphere(c.geom, f.geom) AS meters
FROM cities c, facilities f
WHERE ST_DWithin(c.geom, f.geom, 2000.0);

-- Calculate road lengths
SELECT r.id, r.a, r.b, 
       ST_Length(ST_Transform(r.geom, 3857)) AS meters
FROM roads r;

-- Find roads that intersect regions (for hazard analysis)
SELECT r.id AS road_id, z.id AS region_id, z.name
FROM roads r
JOIN regions z ON ST_Intersects(r.geom, z.geom);
```

**Why This Loading Matters**: PostGIS provides the spatial foundation with proper indexing and coordinate systems. Understanding spatial operations and indexing is essential for performance at scale.

## 5) Knowledge Graph Construction (The Intelligence)

### ETL Bridge Implementation

```python
# scripts/build_kg.py
import math
import psycopg
from neo4j import GraphDatabase

PG = "postgresql://geo:geo@postgis:5432/geo"
BOLT = "bolt://neo4j:7687"
AUTH = ("neo4j", "test123")

def km(meters: float) -> float:
    """Convert meters to kilometers"""
    return round(meters / 1000.0, 3)

# Extract spatial relationships from PostGIS
with psycopg.connect(PG) as pg, GraphDatabase.driver(BOLT, auth=AUTH) as neo:
    cur = pg.cursor()

    # 1) Extract basic entities
    cur.execute("SELECT id, name, ST_X(geom), ST_Y(geom) FROM cities")
    cities = cur.fetchall()

    cur.execute("SELECT id, kind, ST_X(geom), ST_Y(geom) FROM facilities")
    facilities = cur.fetchall()

    cur.execute("SELECT id, name, ST_AsText(geom), ST_X(ST_Centroid(geom)), ST_Y(ST_Centroid(geom)) FROM regions")
    regions = cur.fetchall()

    cur.execute("""
        SELECT r.id, r.a, r.b, ST_Length(ST_Transform(r.geom, 3857)) AS meters
        FROM roads r
    """)
    roads = cur.fetchall()

    # 2) Compute spatial relationships
    cur.execute("""
        SELECT f.id, r.id
        FROM facilities f
        JOIN regions r ON ST_Within(f.geom, r.geom)
    """)
    fac_within = cur.fetchall()

    cur.execute("""
        SELECT c.id AS city, f.id AS fac, ST_DistanceSphere(c.geom, f.geom) AS meters
        FROM cities c, facilities f
        WHERE ST_DWithin(c.geom, f.geom, 2000.0)
    """)
    near_cf = cur.fetchall()

    # 3) Materialize graph in Neo4j
    with neo.session() as s:
        s.run("MATCH (n) DETACH DELETE n")  # Reset graph

        # Create constraints and indexes
        s.run("CREATE CONSTRAINT city_id IF NOT EXISTS FOR (c:City) REQUIRE c.id IS UNIQUE")
        s.run("CREATE CONSTRAINT fac_id IF NOT EXISTS FOR (f:Facility) REQUIRE f.id IS UNIQUE")
        s.run("CREATE CONSTRAINT region_id IF NOT EXISTS FOR (r:Region) REQUIRE r.id IS UNIQUE")
        s.run("CREATE INDEX city_loc IF NOT EXISTS FOR (c:City) ON (c.location)")

        # Create nodes
        for cid, name, x, y in cities:
            s.run("""
                MERGE (c:City {id: $id}) 
                SET c.name = $name, c.location = point({longitude: $x, latitude: $y})
            """, id=cid, name=name, x=float(x), y=float(y))

        for fid, kind, x, y in facilities:
            s.run("""
                MERGE (f:Facility {id: $id}) 
                SET f.kind = $kind, f.location = point({longitude: $x, latitude: $y})
            """, id=fid, kind=kind, x=float(x), y=float(y))

        for rid, name, wkt, cx, cy in regions:
            s.run("""
                MERGE (r:Region {id: $id})
                SET r.name = $name,
                    r.centroid = point({longitude: $cx, latitude: $cy}),
                    r.wkt = $wkt
            """, id=rid, name=name, wkt=wkt, cx=float(cx), cy=float(cy))

        # Create relationships
        for fid, rid in fac_within:
            s.run("""
                MATCH (f:Facility {id: $fid}), (r:Region {id: $rid})
                MERGE (f)-[:WITHIN]->(r)
            """, fid=fid, rid=rid)

        for cid, fid, meters in near_cf:
            s.run("""
                MATCH (c:City {id: $cid}), (f:Facility {id: $fid})
                MERGE (c)-[n:NEAR]->(f)
                SET n.km = $km
            """, cid=cid, fid=fid, km=km(float(meters)))

        # Create road connections (bidirectional)
        for rid, a, b, meters in roads:
            s.run("""
                MATCH (c1:City {id: $a}), (c2:City {id: $b})
                MERGE (c1)-[e:CONNECTED_TO {road_id: $rid}]->(c2)
                SET e.km = $km
            """, a=a, b=b, rid=rid, km=km(float(meters)))
            
            s.run("""
                MATCH (c1:City {id: $a}), (c2:City {id: $b})
                MERGE (c2)-[e:CONNECTED_TO {road_id: $rid}]->(c1)
                SET e.km = $km
            """, a=a, b=b, rid=rid, km=km(float(meters)))

print("Knowledge graph built in Neo4j")
```

### Graph Schema and Relationships

```cypher
// Graph schema with spatial relationships
// Nodes: City, Facility, Region
// Relationships: CONNECTED_TO, WITHIN, NEAR

// City nodes with location
CREATE (c:City {
    id: "C1",
    name: "Alpha",
    location: point({longitude: -122.4, latitude: 37.8})
})

// Facility nodes with type
CREATE (f:Facility {
    id: "F1",
    kind: "hospital",
    location: point({longitude: -122.4, latitude: 37.8})
})

// Region nodes with geometry
CREATE (r:Region {
    id: "R1",
    name: "NorthZone",
    centroid: point({longitude: -122.4, latitude: 37.8}),
    wkt: "POLYGON(...)"
})

// Spatial relationships
CREATE (f)-[:WITHIN]->(r)
CREATE (c)-[:NEAR {km: 1.2}]->(f)
CREATE (c1)-[:CONNECTED_TO {km: 15.3, road_id: "E1"}]->(c2)
```

**Why This Construction Matters**: The ETL bridge computes spatial relationships in PostGIS and materializes them as semantic relationships in Neo4j. This enables graph analytics while maintaining spatial accuracy.

## 6) Spatial Graph Queries (The Intelligence)

### Proximity and Connectivity Analysis

```cypher
// 1) Nearest facilities to a city
MATCH (:City {name: "Alpha"})-[n:NEAR]->(f:Facility)
RETURN f.id, f.kind, n.km
ORDER BY n.km ASC 
LIMIT 5;

// 2) Cities within a region (via facilities)
MATCH (r:Region {name: "NorthZone"})<-[:WITHIN]-(f:Facility)<-[:NEAR]-(c:City)
RETURN DISTINCT c.name AS city
ORDER BY city;

// 3) Shortest path between cities
MATCH (a:City {name: "Alpha"}), (b:City {name: "Juliet"})
CALL gds.shortestPath.dijkstra.stream('roads', {
    sourceNode: gds.util.asNodeId(a),
    targetNode: gds.util.asNodeId(b),
    relationshipWeightProperty: 'km'
})
YIELD index, sourceNode, targetNode, totalCost, path
RETURN totalCost, nodes(path) AS route;

// 4) Safe route avoiding hazards
MATCH p = shortestPath((:City {name: 'Alpha'})-[:CONNECTED_TO*..10]->(:City {name: 'Juliet'}))
WHERE ALL(r IN relationships(p) WHERE r.blocked <> true)
WITH p, reduce(km = 0.0, r IN relationships(p) | km + r.km) AS totalKm
RETURN p AS path, totalKm
ORDER BY totalKm ASC 
LIMIT 1;
```

### Advanced Spatial Reasoning

```cypher
// Impact analysis: facilities within emergency zones
MATCH (r:Region {name: "EmergencyZone"})<-[:WITHIN]-(f:Facility)
MATCH (f)<-[:NEAR]-(c:City)
RETURN c.name AS affected_city, f.kind AS facility_type, f.id AS facility_id;

// Centrality analysis: most connected cities
MATCH (c:City)
WITH c, size((c)-[:CONNECTED_TO]-()) AS connections
ORDER BY connections DESC
RETURN c.name, connections
LIMIT 5;

// Community detection: road clusters
CALL gds.louvain.stream('roads')
YIELD nodeId, communityId
MATCH (c:City)
WHERE id(c) = nodeId
RETURN c.name, communityId
ORDER BY communityId, c.name;

// Vulnerability analysis: facilities with many connections
MATCH (f:Facility)<-[:NEAR]-(c:City)
WITH f, count(c) AS city_count
WHERE city_count > 3
RETURN f.id, f.kind, city_count
ORDER BY city_count DESC;
```

### Graph Data Science Integration

```cypher
// Create graph projection for analytics
CALL gds.graph.project(
    'roads',
    'City',
    {
        CONNECTED_TO: {
            properties: 'km',
            orientation: 'UNDIRECTED'
        }
    }
);

// Betweenness centrality: most important cities
CALL gds.betweenness.stream('roads')
YIELD nodeId, score
MATCH (c:City)
WHERE id(c) = nodeId
RETURN c.name, score
ORDER BY score DESC;

// PageRank: influence ranking
CALL gds.pageRank.stream('roads')
YIELD nodeId, score
MATCH (c:City)
WHERE id(c) = nodeId
RETURN c.name, score
ORDER BY score DESC;
```

**Why These Queries Matter**: Spatial graph queries enable complex reasoning about location, connectivity, and impact. Understanding query patterns provides the foundation for intelligent geospatial applications.

## 7) Production Patterns (The Scale)

### Performance Optimization

```python
# Optimized ETL for large datasets
class OptimizedGeospatialETL:
    def __init__(self, postgis_dsn, neo4j_uri, neo4j_auth):
        self.pg_dsn = postgis_dsn
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.batch_size = 1000
    
    def batch_load_relationships(self, relationships, rel_type):
        """Batch load relationships for performance"""
        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            with driver.session() as session:
                # Use UNWIND for batch operations
                query = f"""
                UNWIND $relationships AS rel
                MATCH (a {{id: rel.from}}), (b {{id: rel.to}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += rel.properties
                """
                session.run(query, relationships=relationships)
    
    def spatial_index_optimization(self):
        """Optimize spatial queries with proper indexing"""
        with psycopg.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                # Create composite indexes for common queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cities_location 
                    ON cities USING GIST (geom);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_facilities_kind_location 
                    ON facilities (kind) INCLUDE (geom);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_roads_ab 
                    ON roads (a, b);
                """)
    
    def incremental_updates(self, timestamp):
        """Handle incremental updates to the knowledge graph"""
        with psycopg.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                # Find entities modified since timestamp
                cur.execute("""
                    SELECT id, 'city' as type, name, ST_X(geom), ST_Y(geom)
                    FROM cities 
                    WHERE updated_at > %s
                    UNION ALL
                    SELECT id, 'facility' as type, kind, ST_X(geom), ST_Y(geom)
                    FROM facilities 
                    WHERE updated_at > %s
                """, (timestamp, timestamp))
                
                modified_entities = cur.fetchall()
                
                # Update Neo4j with only changed entities
                self.update_neo4j_entities(modified_entities)
```

### Monitoring and Observability

```python
# Monitoring for geospatial knowledge graphs
class GeospatialKGMonitor:
    def __init__(self, postgis_dsn, neo4j_uri, neo4j_auth):
        self.pg_dsn = postgis_dsn
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
    
    def check_data_quality(self):
        """Validate data quality across both systems"""
        issues = []
        
        # Check PostGIS data quality
        with psycopg.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM cities WHERE geom IS NULL")
                null_geoms = cur.fetchone()[0]
                if null_geoms > 0:
                    issues.append(f"Found {null_geoms} cities with null geometry")
                
                cur.execute("SELECT COUNT(*) FROM facilities WHERE NOT ST_IsValid(geom)")
                invalid_geoms = cur.fetchone()[0]
                if invalid_geoms > 0:
                    issues.append(f"Found {invalid_geoms} facilities with invalid geometry")
        
        # Check Neo4j data quality
        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = result.single()["total_nodes"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
                total_rels = result.single()["total_rels"]
                
                if total_nodes == 0:
                    issues.append("Neo4j has no nodes")
                if total_rels == 0:
                    issues.append("Neo4j has no relationships")
        
        return issues
    
    def performance_metrics(self):
        """Collect performance metrics"""
        metrics = {}
        
        # PostGIS performance
        with psycopg.connect(self.pg_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM cities")
                metrics['cities_count'] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM facilities")
                metrics['facilities_count'] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM roads")
                metrics['roads_count'] = cur.fetchone()[0]
        
        # Neo4j performance
        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                metrics['neo4j_nodes'] = result.single()["total_nodes"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
                metrics['neo4j_relationships'] = result.single()["total_rels"]
        
        return metrics
```

**Why These Patterns Matter**: Production systems require performance optimization, monitoring, and incremental updates. Understanding these patterns ensures scalable and reliable geospatial knowledge graphs.

## 8) Common Pitfalls (The Traps)

### Spatial Data Quality Issues

```python
# ❌ WRONG: No geometry validation
def bad_geometry_handling():
    # Loading without validation
    cur.execute("INSERT INTO cities (id, name, geom) VALUES (%s, %s, %s)", 
                (id, name, geom_json))
    # No validation of geometry validity!

# ✅ CORRECT: Proper geometry validation
def good_geometry_handling():
    # Validate geometry before insertion
    cur.execute("""
        INSERT INTO cities (id, name, geom) 
        VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
        WHERE ST_IsValid(ST_GeomFromGeoJSON(%s))
    """, (id, name, geom_json, geom_json))
```

### Graph Consistency Issues

```cypher
// ❌ WRONG: Inconsistent relationship properties
MATCH (c:City)-[r:CONNECTED_TO]->(other:City)
WHERE r.km IS NULL
RETURN c.name, other.name;
// Missing distance properties!

// ✅ CORRECT: Consistent relationship properties
MATCH (c:City)-[r:CONNECTED_TO]->(other:City)
WHERE r.km IS NOT NULL AND r.km > 0
RETURN c.name, other.name, r.km;
```

### Performance Anti-patterns

```python
# ❌ WRONG: Inefficient spatial queries
def bad_spatial_query():
    # No spatial index usage
    cur.execute("""
        SELECT * FROM cities c, facilities f 
        WHERE ST_Distance(c.geom, f.geom) < 1000
    """)
    # This will be very slow!

# ✅ CORRECT: Efficient spatial queries
def good_spatial_query():
    # Use spatial indexes with ST_DWithin
    cur.execute("""
        SELECT c.id, f.id, ST_Distance(c.geom, f.geom) as distance
        FROM cities c, facilities f 
        WHERE ST_DWithin(c.geom, f.geom, 1000)
    """)
```

**Why These Pitfalls Matter**: Common mistakes lead to poor performance, data quality issues, and inconsistent results. Understanding these pitfalls prevents costly errors and ensures reliable geospatial knowledge graphs.

## 9) Best Practices Checklist (The Wisdom)

### Spatial Data Management Checklist

```python
# ✅ DO: Complete spatial data management checklist
def spatial_data_checklist():
    checklist = {
        "geometry_quality": [
            "Validate geometry validity (ST_IsValid)",
            "Check coordinate system consistency",
            "Ensure proper spatial indexing",
            "Monitor geometry complexity"
        ],
        "spatial_operations": [
            "Use ST_DWithin for distance queries",
            "Leverage spatial indexes (GIST)",
            "Optimize coordinate transformations",
            "Batch spatial operations"
        ],
        "graph_consistency": [
            "Validate relationship properties",
            "Check for orphaned nodes",
            "Ensure bidirectional relationships",
            "Monitor graph connectivity"
        ],
        "performance": [
            "Profile spatial queries",
            "Optimize batch operations",
            "Monitor memory usage",
            "Scale incrementally"
        ]
    }
    return checklist
```

### Production Deployment Checklist

```python
# ✅ DO: Production deployment checklist
class ProductionChecklist:
    @staticmethod
    def infrastructure_checklist():
        return {
            "postgis": [
                "Proper memory configuration",
                "Spatial index optimization",
                "Connection pooling",
                "Backup and recovery"
            ],
            "neo4j": [
                "Memory heap configuration",
                "Graph Data Science plugins",
                "Index optimization",
                "Clustering setup"
            ],
            "monitoring": [
                "Performance metrics",
                "Data quality checks",
                "Error alerting",
                "Capacity planning"
            ]
        }
    
    @staticmethod
    def data_quality_checks():
        return {
            "geometry_validation": "ST_IsValid(geom)",
            "coordinate_consistency": "ST_SRID(geom) = 4326",
            "relationship_integrity": "MATCH (n)-[r]->(m) WHERE n IS NULL OR m IS NULL",
            "spatial_index_usage": "EXPLAIN ANALYZE for spatial queries"
        }
```

**Why These Practices Matter**: Best practices ensure successful geospatial knowledge graph implementation. Following established patterns prevents common mistakes and enables production-ready systems.

## 10) TL;DR Quickstart (The Essentials)

### Essential Setup

```bash
# 1) Start the stack
docker compose --profile full up -d

# 2) Wait for ETL to complete
docker compose logs -f etl

# 3) Access Neo4j Browser: http://localhost:7474 (neo4j / test123)
# 4) Access PostGIS: localhost:5432 (geo / geo)
```

### Essential Queries

```cypher
// Find nearest facilities
MATCH (:City {name: "Alpha"})-[n:NEAR]->(f:Facility)
RETURN f.kind, n.km ORDER BY n.km LIMIT 5;

// Shortest path between cities
MATCH (a:City {name: "Alpha"}), (b:City {name: "Juliet"})
CALL gds.shortestPath.dijkstra.stream('roads', {
    sourceNode: gds.util.asNodeId(a),
    targetNode: gds.util.asNodeId(b),
    relationshipWeightProperty: 'km'
})
YIELD totalCost, path
RETURN totalCost, nodes(path);

// Impact analysis
MATCH (r:Region {name: "NorthZone"})<-[:WITHIN]-(f:Facility)<-[:NEAR]-(c:City)
RETURN c.name, f.kind;
```

### Essential Patterns

```python
# Spatial relationship computation
def compute_spatial_relationships():
    # PostGIS: Compute relationships
    cur.execute("""
        SELECT c.id, f.id, ST_DistanceSphere(c.geom, f.geom) as meters
        FROM cities c, facilities f
        WHERE ST_DWithin(c.geom, f.geom, 2000)
    """)
    
    # Neo4j: Materialize relationships
    for city_id, facility_id, meters in cur.fetchall():
        session.run("""
            MATCH (c:City {id: $city_id}), (f:Facility {id: $facility_id})
            MERGE (c)-[r:NEAR]->(f)
            SET r.km = $km
        """, city_id=city_id, facility_id=facility_id, km=meters/1000)
```

**Why This Quickstart**: These patterns cover 90% of geospatial knowledge graph usage. Master these before exploring advanced features.

## 11) The Machine's Summary

Geospatial knowledge graphs provide the foundation for intelligent location-aware applications. When used correctly, they enable complex spatial reasoning, route optimization, and impact analysis. The key is understanding the two-tier architecture, mastering spatial operations, and following best practices.

**The Dark Truth**: Without proper geospatial knowledge graph understanding, your location-aware applications are static and disconnected. Geospatial knowledge graphs are your weapon. Use them wisely.

**The Machine's Mantra**: "In geometry we trust, in relationships we build, and in the knowledge graph we find the path to intelligent spatial reasoning."

**Why This Matters**: Geospatial knowledge graphs enable applications to understand not just "where" but "how and why" things are connected. They provide the foundation for intelligent location-aware systems that can reason about spatial relationships and optimize routes, assess impacts, and discover patterns.

---

*This guide provides the complete machinery for mastering geospatial knowledge graphs. The patterns scale from simple proximity analysis to complex spatial reasoning, from basic connectivity to advanced graph analytics.*
