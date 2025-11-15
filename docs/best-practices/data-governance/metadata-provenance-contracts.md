# Metadata Standards, Schema Governance & Data Provenance Contracts: Best Practices for Distributed Analytics Systems

**Objective**: Master production-grade metadata management, schema governance, and data provenance for distributed analytics ecosystems. When you need to ensure data quality, track lineage, enforce contracts, and maintain reproducibility across Postgres, Parquet, MLflow, and ETL pipelines—this guide provides the complete framework.

## Introduction

Metadata, schema governance, and provenance are the foundation of trustworthy, reproducible analytics. Without them, data becomes a liability: schemas drift, lineage is lost, contracts are violated, and reproducibility fails. This guide provides a complete framework for building metadata-driven, contract-enforced, provenance-tracked data systems.

**What This Guide Covers**:
- Unified metadata model (dataset, column, operational)
- Data contracts (definition, types, enforcement)
- Schema versioning strategies (SemVer, storage patterns)
- Provenance and lineage tracking
- Automated validation workflows
- Metadata storage architecture
- Data quality and SLA governance
- Integration with existing systems

**Prerequisites**:
- Understanding of data pipelines, ETL/ELT, and data storage formats
- Familiarity with Postgres, Parquet, and data processing frameworks
- Experience with version control and CI/CD

## Why Metadata, Schema Governance & Provenance Are Critical

### The Problem: Data Without Context

Modern analytics ecosystems are distributed:
- **Data sources**: Postgres tables, Parquet files, object stores, APIs
- **Processing**: Prefect workflows, Spark jobs, MLflow pipelines, DuckDB queries
- **Consumers**: ML models, GIS applications, APIs, dashboards, reports

**Without metadata and provenance**:
- **Schema drift** breaks downstream systems silently
- **Lineage is lost**, making debugging and auditing impossible
- **Reproducibility fails** when you can't identify which data version was used
- **Trust erodes** when data quality is unknown
- **Governance breaks** when changes aren't tracked or validated

### The Cost of Poor Metadata

**Real-World Failures**:

1. **Schema Drift**: A Parquet column type changes from `int32` to `int64`, breaking ML inference pipelines that expect `int32`
2. **Lost Lineage**: A data quality issue is discovered, but you can't trace which ETL job or source data caused it
3. **Version Confusion**: An ML model is trained on `roads_v1`, but production uses `roads_v2` with different schema
4. **Contract Violations**: An API consumer expects `risk_level ∈ {low, medium, high}`, but receives `{1, 2, 3}`
5. **Spatial Reference Errors**: GeoParquet files use different CRS without metadata, causing misaligned visualizations

### The Value of Metadata-Driven Systems

**With proper metadata and provenance**:
- **Reproducibility**: Know exactly which data version, schema version, and code version produced each result
- **Auditability**: Trace data lineage from source to consumption
- **Quality Assurance**: Automated validation catches schema drift and contract violations
- **Governance**: Schema changes are reviewed, versioned, and tracked
- **Trust**: Data consumers can verify schema, quality, and provenance before use

### System Integration

```mermaid
graph TB
    subgraph Sources["Data Sources"]
        PG["PostgreSQL<br/>(PostGIS)"]
        Parquet["Parquet/GeoParquet<br/>(S3/MinIO)"]
        API["APIs"]
    end
    
    subgraph Metadata["Metadata Layer"]
        Registry["Metadata Registry<br/>(Postgres + JSON-LD)"]
        Contracts["Data Contracts<br/>(YAML/JSON)"]
        SchemaVers["Schema Versions<br/>(SemVer)"]
    end
    
    subgraph Processing["Processing Layer"]
        ETL["ETL Pipelines<br/>(Prefect, Spark)"]
        ML["ML Pipelines<br/>(MLflow, ONNX)"]
        GIS["GIS Processing<br/>(PostGIS, GeoPandas)"]
    end
    
    subgraph Provenance["Provenance Tracking"]
        Lineage["Lineage Graph<br/>(Postgres + JSON-LD)"]
        Events["Provenance Events<br/>(ETL logs, MLflow)"]
    end
    
    subgraph Validation["Validation Layer"]
        SchemaVal["Schema Validation<br/>(Great Expectations, Polars)"]
        ContractVal["Contract Validation<br/>(CI/CD, Prefect)"]
        QualityVal["Quality Checks<br/>(Null ratios, uniqueness)"]
    end
    
    subgraph Consumers["Data Consumers"]
        Models["ML Models<br/>(ONNX inference)"]
        Dashboards["Dashboards<br/>(Grafana, NiceGUI)"]
        APIs["APIs<br/>(FastAPI)"]
    end
    
    PG --> Registry
    Parquet --> Registry
    API --> Registry
    
    Registry --> Contracts
    Registry --> SchemaVers
    
    Sources --> ETL
    Sources --> ML
    Sources --> GIS
    
    ETL --> Lineage
    ML --> Lineage
    GIS --> Lineage
    
    ETL --> Events
    ML --> Events
    
    Lineage --> Registry
    Events --> Registry
    
    ETL --> SchemaVal
    ML --> SchemaVal
    
    SchemaVal --> ContractVal
    ContractVal --> QualityVal
    
    QualityVal --> Models
    QualityVal --> Dashboards
    QualityVal --> APIs
    
    Models --> Registry
    Dashboards --> Registry
    APIs --> Registry
```

### Integration Points

**GeoParquet**:
- Metadata embedded in Parquet file footer
- CRS, spatial extents, column semantics
- Version tracking via directory structure

**PostgreSQL**:
- Schema versioning via schema names (`roads_v1`, `roads_v2`)
- Metadata tables for dataset and column metadata
- CHECK constraints for data quality

**MLflow**:
- Model metadata references training data schema version
- Provenance chain from data → model → inference

**ETL Pipelines**:
- Prefect tasks record lineage events
- Schema validation before/after transformations
- Contract enforcement at pipeline boundaries

**JSON-LD Dynamic Forms**:
- Schema-driven form generation
- Validation against metadata contracts
- Provenance recording for user inputs

## Core Metadata Model

### Unified Metadata Standard

A unified metadata model enables consistent governance across all data formats and systems.

### Dataset-Level Metadata

**Core Fields**:

```json
{
  "@context": "https://schema.org/",
  "@type": "Dataset",
  "identifier": "roads-v2",
  "name": "Road Network Dataset",
  "description": "Comprehensive road network data for the United States",
  "version": "2.1.0",
  "schemaVersion": "2.0.0",
  "owner": {
    "name": "Data Engineering Team",
    "email": "data-eng@example.com"
  },
  "tags": ["transportation", "infrastructure", "geospatial"],
  "domains": ["GIS", "ML", "Analytics"],
  "spatialExtent": {
    "type": "Polygon",
    "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
    "crs": "EPSG:4326"
  },
  "temporalExtent": {
    "startDate": "2020-01-01T00:00:00Z",
    "endDate": "2024-12-31T23:59:59Z"
  },
  "dataQuality": {
    "completeness": 0.98,
    "accuracy": 0.95,
    "consistency": 0.97,
    "timeliness": 0.99
  },
  "accessLevel": "internal",
  "updateFrequency": "daily",
  "storage": {
    "type": "parquet",
    "location": "s3://data-lake/datasets/roads/v2/",
    "format": "geoparquet",
    "partitioning": ["year", "month"]
  },
  "provenance": {
    "parents": [
      {
        "identifier": "osm-roads-v1",
        "version": "1.5.0",
        "relationship": "derived_from"
      }
    ],
    "transformation": {
      "code": "git://github.com/org/etl-pipelines@abc123",
      "environment": "sha256:def456...",
      "executedBy": "prefect-job-789",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  },
  "lineage": {
    "upstream": ["osm-roads-v1", "census-boundaries-v1"],
    "downstream": ["ml-risk-model-v3", "api-roads-v1"]
  },
  "operational": {
    "lastUpdated": "2024-01-15T10:30:00Z",
    "etlJobId": "prefect-job-789",
    "recordCount": 12500000,
    "fileCount": 365,
    "totalSizeBytes": 10737418240,
    "checksum": "sha256:abc123...",
    "validationStatus": "passed",
    "validationTimestamp": "2024-01-15T10:31:00Z"
  }
}
```

**YAML Format**:

```yaml
# datasets/roads-v2/metadata.yaml
identifier: roads-v2
name: Road Network Dataset
description: Comprehensive road network data for the United States
version: 2.1.0
schemaVersion: 2.0.0

owner:
  name: Data Engineering Team
  email: data-eng@example.com

tags:
  - transportation
  - infrastructure
  - geospatial

domains:
  - GIS
  - ML
  - Analytics

spatialExtent:
  type: Polygon
  coordinates: [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]]
  crs: EPSG:4326

temporalExtent:
  startDate: 2020-01-01T00:00:00Z
  endDate: 2024-12-31T23:59:59Z

dataQuality:
  completeness: 0.98
  accuracy: 0.95
  consistency: 0.97
  timeliness: 0.99

accessLevel: internal
updateFrequency: daily

storage:
  type: parquet
  location: s3://data-lake/datasets/roads/v2/
  format: geoparquet
  partitioning:
    - year
    - month

provenance:
  parents:
    - identifier: osm-roads-v1
      version: 1.5.0
      relationship: derived_from
  transformation:
    code: git://github.com/org/etl-pipelines@abc123
    environment: sha256:def456...
    executedBy: prefect-job-789
    timestamp: 2024-01-15T10:30:00Z

lineage:
  upstream:
    - osm-roads-v1
    - census-boundaries-v1
  downstream:
    - ml-risk-model-v3
    - api-roads-v1

operational:
  lastUpdated: 2024-01-15T10:30:00Z
  etlJobId: prefect-job-789
  recordCount: 12500000
  fileCount: 365
  totalSizeBytes: 10737418240
  checksum: sha256:abc123...
  validationStatus: passed
  validationTimestamp: 2024-01-15T10:31:00Z
```

### Column-Level Metadata

**Schema Definition**:

```json
{
  "schema": {
    "version": "2.0.0",
    "columns": [
      {
        "name": "road_id",
        "type": {
          "sql": "BIGINT",
          "parquet": "int64",
          "semantic": "identifier"
        },
        "description": "Unique identifier for each road segment",
        "nullable": false,
        "constraints": {
          "primaryKey": true,
          "unique": true
        },
        "validRange": null,
        "categoryMappings": null,
        "foreignKey": null,
        "geospatial": null
      },
      {
        "name": "road_name",
        "type": {
          "sql": "TEXT",
          "parquet": "string",
          "semantic": "name"
        },
        "description": "Official name of the road",
        "nullable": true,
        "constraints": {
          "maxLength": 255
        },
        "validRange": null,
        "categoryMappings": null,
        "foreignKey": null,
        "geospatial": null
      },
      {
        "name": "risk_level",
        "type": {
          "sql": "VARCHAR(20)",
          "parquet": "string",
          "semantic": "category"
        },
        "description": "Risk level classification",
        "nullable": false,
        "constraints": {
          "allowedValues": ["low", "medium", "high"]
        },
        "validRange": null,
        "categoryMappings": {
          "low": 1,
          "medium": 2,
          "high": 3
        },
        "foreignKey": null,
        "geospatial": null
      },
      {
        "name": "geometry",
        "type": {
          "sql": "GEOMETRY(LINESTRING, 4326)",
          "parquet": "binary",
          "semantic": "geometry"
        },
        "description": "Road segment geometry",
        "nullable": false,
        "constraints": {
          "spatialType": "LINESTRING",
          "crs": "EPSG:4326",
          "validGeometry": true
        },
        "validRange": null,
        "categoryMappings": null,
        "foreignKey": null,
        "geospatial": {
          "type": "LINESTRING",
          "crs": "EPSG:4326",
          "dimension": 2,
          "hasZ": false,
          "hasM": false
        }
      },
      {
        "name": "created_at",
        "type": {
          "sql": "TIMESTAMPTZ",
          "parquet": "timestamp[us, tz=UTC]",
          "semantic": "timestamp"
        },
        "description": "Record creation timestamp",
        "nullable": false,
        "constraints": {
          "timezone": "UTC"
        },
        "validRange": {
          "min": "2020-01-01T00:00:00Z",
          "max": "2024-12-31T23:59:59Z"
        },
        "categoryMappings": null,
        "foreignKey": null,
        "geospatial": null
      }
    ]
  }
}
```

### Operational Metadata

**Runtime Metrics**:

```json
{
  "operational": {
    "lastUpdated": "2024-01-15T10:30:00Z",
    "etlJobId": "prefect-job-789",
    "etlJobRunId": "prefect-run-abc123",
    "recordCount": 12500000,
    "fileCount": 365,
    "totalSizeBytes": 10737418240,
    "checksum": "sha256:abc123...",
    "validationStatus": "passed",
    "validationTimestamp": "2024-01-15T10:31:00Z",
    "qualityMetrics": {
      "nullRatios": {
        "road_name": 0.02,
        "risk_level": 0.0
      },
      "uniqueness": {
        "road_id": 1.0
      },
      "spatialValidity": {
        "geometry": 0.98
      },
      "valueDistribution": {
        "risk_level": {
          "low": 0.6,
          "medium": 0.3,
          "high": 0.1
        }
      }
    },
    "performance": {
      "ingestionDurationSeconds": 1200,
      "validationDurationSeconds": 45,
      "recordsPerSecond": 10416
    }
  }
}
```

### JSON-LD Schema Template

**Semantic Web Integration**:

```json
{
  "@context": {
    "schema": "https://schema.org/",
    "dc": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "dqv": "http://www.w3.org/ns/dqv#"
  },
  "@type": "schema:Dataset",
  "@id": "https://data.example.com/datasets/roads-v2",
  "schema:identifier": "roads-v2",
  "schema:name": "Road Network Dataset",
  "schema:description": "Comprehensive road network data for the United States",
  "schema:version": "2.1.0",
  "schema:creator": {
    "@type": "schema:Organization",
    "schema:name": "Data Engineering Team"
  },
  "schema:keywords": ["transportation", "infrastructure", "geospatial"],
  "schema:spatialCoverage": {
    "@type": "schema:Place",
    "schema:geo": {
      "@type": "schema:GeoShape",
      "schema:polygon": "[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]"
    }
  },
  "schema:temporalCoverage": {
    "@type": "schema:DateTime",
    "schema:startDate": "2020-01-01T00:00:00Z",
    "schema:endDate": "2024-12-31T23:59:59Z"
  },
  "prov:wasDerivedFrom": {
    "@id": "https://data.example.com/datasets/osm-roads-v1",
    "@type": "prov:Entity"
  },
  "prov:wasGeneratedBy": {
    "@type": "prov:Activity",
    "prov:startedAtTime": "2024-01-15T10:30:00Z",
    "prov:used": "https://github.com/org/etl-pipelines@abc123"
  },
  "dqv:hasQualityMeasurement": {
    "@type": "dqv:QualityMeasurement",
    "dqv:completeness": 0.98,
    "dqv:accuracy": 0.95
  }
}
```

## Data Contracts

### What Is a Data Contract?

A **data contract** is a formal, versioned agreement between data producers and consumers that defines:
- **Schema**: Column names, types, constraints
- **Required fields**: Fields that must be present
- **Valid values**: Allowed values for categorical fields
- **Constraints**: Data type constraints, ranges, formats
- **Allowed breaking changes**: What changes are allowed without version bump
- **Behavioral expectations**: Business rules (e.g., "risk_level always ∈ {low, medium, high}")
- **SLA**: Data delivery frequency, latency, availability
- **Ownership**: Who owns the data, who can modify it

### Contract Types

#### API Contract

**Example**:

```yaml
# contracts/api-roads-v1.yaml
contract:
  identifier: api-roads-v1
  version: 1.0.0
  type: api
  owner: API Team
  consumers:
    - frontend-dashboard
    - mobile-app
  
  schema:
    version: 1.0.0
    requiredFields:
      - road_id
      - road_name
      - geometry
    fields:
      - name: road_id
        type: integer
        description: Unique road identifier
        nullable: false
      - name: road_name
        type: string
        description: Road name
        nullable: true
        maxLength: 255
      - name: risk_level
        type: string
        description: Risk level
        nullable: false
        allowedValues:
          - low
          - medium
          - high
      - name: geometry
        type: object
        description: GeoJSON geometry
        nullable: false
        format: geojson
        constraints:
          type: LineString
          crs: EPSG:4326
  
  behavioralExpectations:
    - field: risk_level
      rule: "Must be one of: low, medium, high"
      validation: enum
  
  sla:
    updateFrequency: daily
    latency: p95 < 100ms
    availability: 99.9%
  
  breakingChanges:
    allowed:
      - Adding new optional fields
      - Removing deprecated fields (with 30-day notice)
    notAllowed:
      - Changing field types
      - Removing required fields
      - Changing allowed values for categorical fields
```

#### ETL Contract

**Example**:

```yaml
# contracts/etl-roads-v2.yaml
contract:
  identifier: etl-roads-v2
  version: 2.0.0
  type: etl
  owner: Data Engineering Team
  consumers:
    - ml-risk-model
    - gis-analysis
  
  schema:
    version: 2.0.0
    requiredFields:
      - road_id
      - geometry
      - risk_level
    fields:
      - name: road_id
        type:
          sql: BIGINT
          parquet: int64
        description: Unique road identifier
        nullable: false
        constraints:
          primaryKey: true
          unique: true
      - name: geometry
        type:
          sql: GEOMETRY(LINESTRING, 4326)
          parquet: binary
        description: Road geometry
        nullable: false
        constraints:
          spatialType: LINESTRING
          crs: EPSG:4326
          validGeometry: true
      - name: risk_level
        type:
          sql: VARCHAR(20)
          parquet: string
        description: Risk level
        nullable: false
        constraints:
          allowedValues: [low, medium, high]
  
  qualityRequirements:
    completeness:
      road_id: 1.0
      geometry: 1.0
      risk_level: 1.0
    uniqueness:
      road_id: 1.0
    spatialValidity:
      geometry: 0.98
  
  sla:
    updateFrequency: daily
    backfillWindow: 7 days
    correctionWorkflow: automated
  
  breakingChanges:
    allowed:
      - Adding new optional columns
      - Extending allowed values (backward compatible)
    notAllowed:
      - Changing column types
      - Removing columns
      - Changing required columns to optional
```

#### Geospatial Dataset Contract

**Example**:

```yaml
# contracts/geospatial-roads-v2.yaml
contract:
  identifier: geospatial-roads-v2
  version: 2.0.0
  type: geospatial
  owner: GIS Team
  consumers:
    - map-visualization
    - spatial-analysis
  
  schema:
    version: 2.0.0
    format: geoparquet
    spatialReference:
      crs: EPSG:4326
      dimension: 2
      hasZ: false
      hasM: false
    geometryColumn: geometry
    geometryType: LINESTRING
    requiredFields:
      - road_id
      - geometry
    fields:
      - name: road_id
        type: int64
        description: Unique road identifier
        nullable: false
      - name: geometry
        type: binary
        description: Road geometry (WKB)
        nullable: false
        constraints:
          spatialType: LINESTRING
          crs: EPSG:4326
  
  spatialRequirements:
    extent:
      minX: -180
      minY: -90
      maxX: 180
      maxY: 90
    crsConsistency: true
    geometryValidity: 0.98
  
  sla:
    updateFrequency: daily
    spatialAccuracy: 10 meters
    coverage: complete
```

#### Model Input Contract

**Example**:

```yaml
# contracts/ml-risk-model-input-v1.yaml
contract:
  identifier: ml-risk-model-input-v1
  version: 1.0.0
  type: model_input
  owner: ML Team
  model: risk-model-v3
  
  schema:
    version: 1.0.0
    requiredFields:
      - road_id
      - geometry
      - road_type
      - traffic_volume
    fields:
      - name: road_id
        type: int64
        description: Road identifier
        nullable: false
      - name: geometry
        type: binary
        description: Road geometry
        nullable: false
        constraints:
          spatialType: LINESTRING
          crs: EPSG:4326
      - name: road_type
        type: string
        description: Road type
        nullable: false
        constraints:
          allowedValues: [highway, arterial, local]
      - name: traffic_volume
        type: float64
        description: Average daily traffic
        nullable: false
        constraints:
          min: 0
          max: 100000
  
  preprocessing:
    - step: normalize_geometry
      description: Normalize geometry to EPSG:4326
    - step: encode_road_type
      description: One-hot encode road_type
  
  validation:
    - check: schema_match
      description: Verify schema matches contract
    - check: value_ranges
      description: Verify values within allowed ranges
    - check: spatial_validity
      description: Verify geometry is valid
```

#### Model Output Contract

**Example**:

```yaml
# contracts/ml-risk-model-output-v1.yaml
contract:
  identifier: ml-risk-model-output-v1
  version: 1.0.0
  type: model_output
  owner: ML Team
  model: risk-model-v3
  
  schema:
    version: 1.0.0
    requiredFields:
      - road_id
      - risk_level
      - risk_score
      - confidence
    fields:
      - name: road_id
        type: int64
        description: Road identifier
        nullable: false
      - name: risk_level
        type: string
        description: Predicted risk level
        nullable: false
        constraints:
          allowedValues: [low, medium, high]
      - name: risk_score
        type: float64
        description: Risk score (0-1)
        nullable: false
        constraints:
          min: 0
          max: 1
      - name: confidence
        type: float64
        description: Prediction confidence (0-1)
        nullable: false
        constraints:
          min: 0
          max: 1
  
  behavioralExpectations:
    - field: risk_level
      rule: "Must match risk_score: low if < 0.33, medium if 0.33-0.66, high if > 0.66"
      validation: business_rule
```

### Contract Enforcement

#### Automated Schema Validation

**Great Expectations Example**:

```python
# validation/great_expectations/roads_v2_suite.py
import great_expectations as ge
from great_expectations.dataset import PandasDataset

def create_roads_v2_expectation_suite():
    """Create expectation suite for roads-v2 dataset"""
    suite = ge.ExpectationSuite("roads_v2")
    
    # Schema expectations
    suite.expect_column_to_exist("road_id")
    suite.expect_column_to_exist("road_name")
    suite.expect_column_to_exist("risk_level")
    suite.expect_column_to_exist("geometry")
    
    # Type expectations
    suite.expect_column_values_to_be_of_type("road_id", "int64")
    suite.expect_column_values_to_be_of_type("road_name", "object")
    suite.expect_column_values_to_be_of_type("risk_level", "object")
    
    # Constraint expectations
    suite.expect_column_values_to_be_unique("road_id")
    suite.expect_column_values_to_not_be_null("road_id")
    suite.expect_column_values_to_not_be_null("risk_level")
    suite.expect_column_values_to_be_in_set(
        "risk_level",
        ["low", "medium", "high"]
    )
    
    # Quality expectations
    suite.expect_column_values_to_not_be_null("road_name", mostly=0.98)
    suite.expect_column_value_lengths_to_be_between("road_name", 1, 255)
    
    return suite
```

**Polars Validation**:

```python
# validation/polars/validate_roads_v2.py
import polars as pl
from typing import Dict, Any

def validate_roads_v2_schema(df: pl.DataFrame, contract: Dict[str, Any]) -> bool:
    """Validate DataFrame against contract schema"""
    schema = contract["schema"]
    
    # Check required fields
    required_fields = schema["requiredFields"]
    missing_fields = set(required_fields) - set(df.columns)
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Check field types
    for field in schema["fields"]:
        field_name = field["name"]
        if field_name not in df.columns:
            continue
        
        expected_type = field["type"]["parquet"]
        actual_type = str(df[field_name].dtype)
        
        # Type mapping
        type_map = {
            "int64": "Int64",
            "string": "Utf8",
            "float64": "Float64",
            "bool": "Boolean",
            "timestamp[us, tz=UTC]": "Datetime"
        }
        
        if type_map.get(expected_type) != actual_type:
            raise ValueError(
                f"Field {field_name}: expected {expected_type}, got {actual_type}"
            )
        
        # Check constraints
        if "constraints" in field:
            constraints = field["constraints"]
            
            # Check allowed values
            if "allowedValues" in constraints:
                allowed = set(constraints["allowedValues"])
                actual = set(df[field_name].unique().to_list())
                invalid = actual - allowed
                if invalid:
                    raise ValueError(
                        f"Field {field_name}: invalid values {invalid}"
                    )
            
            # Check nullability
            if not field.get("nullable", False):
                null_count = df[field_name].null_count()
                if null_count > 0:
                    raise ValueError(
                        f"Field {field_name}: {null_count} null values found"
                    )
    
    return True
```

**Postgres CHECK Constraints**:

```sql
-- Postgres schema with contract enforcement
CREATE TABLE roads_v2 (
    road_id BIGINT PRIMARY KEY,
    road_name TEXT,
    risk_level VARCHAR(20) NOT NULL,
    geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Contract enforcement via CHECK constraints
    CONSTRAINT risk_level_valid CHECK (risk_level IN ('low', 'medium', 'high')),
    CONSTRAINT road_name_length CHECK (LENGTH(road_name) <= 255),
    CONSTRAINT geometry_valid CHECK (ST_IsValid(geometry)),
    CONSTRAINT created_at_range CHECK (
        created_at >= '2020-01-01'::timestamptz AND
        created_at <= '2024-12-31'::timestamptz
    )
);

-- Index for performance
CREATE INDEX idx_roads_v2_risk_level ON roads_v2(risk_level);
CREATE INDEX idx_roads_v2_geometry ON roads_v2 USING GIST(geometry);
```

**Prefect Validation Task**:

```python
# prefect/tasks/validate_contract.py
from prefect import task
import polars as pl
from validation.polars.validate_roads_v2 import validate_roads_v2_schema
import yaml

@task
def validate_data_contract(
    data_path: str,
    contract_path: str
) -> bool:
    """Validate data against contract"""
    # Load contract
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    # Load data
    df = pl.read_parquet(data_path)
    
    # Validate
    try:
        validate_roads_v2_schema(df, contract)
        return True
    except ValueError as e:
        raise ValueError(f"Contract validation failed: {e}")
```

**CI/CD Validation**:

```yaml
# .github/workflows/validate-data-contract.yml
name: Validate Data Contract

on:
  pull_request:
    paths:
      - 'data/**'
      - 'contracts/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install polars pyyaml
      
      - name: Validate data contracts
        run: |
          python scripts/validate_contracts.py \
            --data-dir data/ \
            --contracts-dir contracts/
```

## Schema Versioning Strategy

### Semantic Versioning for Schemas

**Version Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (removed fields, type changes, constraint changes)
- **MINOR**: Backward-compatible additions (new optional fields, extended allowed values)
- **PATCH**: Metadata-only or constraint tweaks (description updates, documentation)

**Examples**:

- `1.0.0` → `2.0.0`: Removed `old_field`, changed `risk_level` from `int` to `string` (BREAKING)
- `1.0.0` → `1.1.0`: Added optional `new_field` (NON-BREAKING)
- `1.0.0` → `1.0.1`: Updated field description (NON-BREAKING)

### Versioned Parquet/GeoParquet Directories

**Directory Structure**:

```
s3://data-lake/
  datasets/
    roads/
      v1/
        year=2020/
          month=01/
            roads-2020-01.parquet
        year=2020/
          month=02/
            roads-2020-02.parquet
        metadata.yaml
        schema.json
      v2/
        year=2020/
          month=01/
            roads-2020-01.parquet
        year=2020/
          month=02/
            roads-2020-02.parquet
        metadata.yaml
        schema.json
      latest -> v2/
```

**Metadata File**:

```yaml
# datasets/roads/v2/metadata.yaml
identifier: roads-v2
version: 2.1.0
schemaVersion: 2.0.0
storage:
  location: s3://data-lake/datasets/roads/v2/
  format: geoparquet
  partitioning:
    - year
    - month
```

### Versioned Postgres Schemas

**Schema Naming Convention**:

```sql
-- Create versioned schemas
CREATE SCHEMA roads_v1;
CREATE SCHEMA roads_v2;

-- Create tables in versioned schemas
CREATE TABLE roads_v1.roads (
    road_id BIGINT PRIMARY KEY,
    road_name TEXT,
    risk_level INT,  -- Old: integer
    geometry GEOMETRY(LINESTRING, 4326)
);

CREATE TABLE roads_v2.roads (
    road_id BIGINT PRIMARY KEY,
    road_name TEXT,
    risk_level VARCHAR(20),  -- New: string with allowed values
    geometry GEOMETRY(LINESTRING, 4326),
    created_at TIMESTAMPTZ  -- New: optional field
);

-- Create view for latest version
CREATE VIEW roads AS SELECT * FROM roads_v2.roads;

-- Migration script
INSERT INTO roads_v2.roads (road_id, road_name, risk_level, geometry, created_at)
SELECT 
    road_id,
    road_name,
    CASE risk_level
        WHEN 1 THEN 'low'
        WHEN 2 THEN 'medium'
        WHEN 3 THEN 'high'
        ELSE 'unknown'
    END::VARCHAR(20) AS risk_level,
    geometry,
    now() AS created_at
FROM roads_v1.roads;
```

### Model-Bound Schema Versions

**MLflow Model Metadata**:

```python
# mlflow/model_metadata.py
import mlflow
import mlflow.sklearn
from datetime import datetime

def log_model_with_schema(
    model,
    training_data_schema_version: str,
    contract_version: str
):
    """Log MLflow model with schema version metadata"""
    with mlflow.start_run():
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log schema version metadata
        mlflow.log_param("training_data_schema_version", training_data_schema_version)
        mlflow.log_param("contract_version", contract_version)
        mlflow.log_param("schema_version", "2.0.0")
        
        # Log contract reference
        mlflow.log_artifact("contracts/ml-risk-model-input-v1.yaml")
        mlflow.log_artifact("contracts/ml-risk-model-output-v1.yaml")
        
        # Log provenance
        mlflow.log_param("training_data", "roads-v2")
        mlflow.log_param("training_data_version", "2.1.0")
        mlflow.log_param("code_version", "git://github.com/org/ml-models@abc123")
```

**ONNX Inference Schema Verification**:

```python
# onnx/schema_verification.py
import onnxruntime as ort
import numpy as np
from validation.polars.validate_roads_v2 import validate_roads_v2_schema

def verify_input_schema(
    input_data: np.ndarray,
    model_path: str,
    contract_path: str
) -> bool:
    """Verify input data matches model's expected schema"""
    # Load model
    session = ort.InferenceSession(model_path)
    
    # Get expected input schema from model metadata
    input_meta = session.get_inputs()[0]
    expected_shape = input_meta.shape
    expected_type = input_meta.type
    
    # Verify input matches
    if input_data.shape != tuple(expected_shape):
        raise ValueError(
            f"Input shape mismatch: expected {expected_shape}, got {input_data.shape}"
        )
    
    # Load and validate against contract
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    # Validate data against contract
    validate_roads_v2_schema(input_data, contract)
    
    return True
```

## Provenance & Lineage Tracking

### Lineage Events

**Event Structure**:

```json
{
  "eventId": "lineage-event-abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "eventType": "transformation",
  "inputs": [
    {
      "datasetId": "osm-roads-v1",
      "version": "1.5.0",
      "location": "s3://data-lake/datasets/osm-roads/v1/",
      "checksum": "sha256:def456..."
    }
  ],
  "outputs": [
    {
      "datasetId": "roads-v2",
      "version": "2.1.0",
      "location": "s3://data-lake/datasets/roads/v2/",
      "checksum": "sha256:abc123..."
    }
  ],
  "transformation": {
    "code": "git://github.com/org/etl-pipelines@abc123",
    "codeHash": "sha256:code123...",
    "environment": "sha256:env456...",
    "executedBy": "prefect-job-789",
    "executionId": "prefect-run-abc123"
  },
  "user": "data-eng@example.com",
  "cluster": "rke2-prod",
  "node": "worker-01"
}
```

### Provenance Storage Methods

#### Postgres Provenance Tables

**Schema**:

```sql
-- Provenance event table
CREATE TABLE provenance_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type VARCHAR(50) NOT NULL,
    transformation_code TEXT,
    transformation_code_hash TEXT,
    environment_hash TEXT,
    executed_by TEXT,
    execution_id TEXT,
    user_email TEXT,
    cluster_name TEXT,
    node_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Input/output relationships
CREATE TABLE provenance_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES provenance_events(event_id),
    relationship_type VARCHAR(50) NOT NULL,  -- 'input' or 'output'
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    location TEXT,
    checksum TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX idx_provenance_events_timestamp ON provenance_events(timestamp);
CREATE INDEX idx_provenance_events_event_type ON provenance_events(event_type);
CREATE INDEX idx_provenance_relationships_dataset ON provenance_relationships(dataset_id, dataset_version);
CREATE INDEX idx_provenance_relationships_event ON provenance_relationships(event_id);
```

**Insert Lineage Event**:

```python
# provenance/postgres_lineage.py
import psycopg2
from datetime import datetime
import uuid

def record_lineage_event(
    inputs: list,
    outputs: list,
    transformation: dict,
    user: str,
    cluster: str,
    node: str
):
    """Record lineage event in Postgres"""
    conn = psycopg2.connect("postgresql://user:pass@localhost/metadata")
    cur = conn.cursor()
    
    event_id = uuid.uuid4()
    
    # Insert event
    cur.execute("""
        INSERT INTO provenance_events (
            event_id, timestamp, event_type,
            transformation_code, transformation_code_hash,
            environment_hash, executed_by, execution_id,
            user_email, cluster_name, node_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        event_id,
        datetime.utcnow(),
        "transformation",
        transformation["code"],
        transformation["codeHash"],
        transformation["environment"],
        transformation["executedBy"],
        transformation["executionId"],
        user,
        cluster,
        node
    ))
    
    # Insert input relationships
    for input_dataset in inputs:
        cur.execute("""
            INSERT INTO provenance_relationships (
                event_id, relationship_type,
                dataset_id, dataset_version,
                location, checksum
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            event_id,
            "input",
            input_dataset["datasetId"],
            input_dataset["version"],
            input_dataset["location"],
            input_dataset["checksum"]
        ))
    
    # Insert output relationships
    for output_dataset in outputs:
        cur.execute("""
            INSERT INTO provenance_relationships (
                event_id, relationship_type,
                dataset_id, dataset_version,
                location, checksum
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            event_id,
            "output",
            output_dataset["datasetId"],
            output_dataset["version"],
            output_dataset["location"],
            output_dataset["checksum"]
        ))
    
    conn.commit()
    cur.close()
    conn.close()
```

#### Parquet Lineage Files

**Lineage Metadata in Parquet**:

```python
# provenance/parquet_lineage.py
import pyarrow.parquet as pq
import pyarrow as pa
import json

def embed_lineage_in_parquet(
    parquet_path: str,
    lineage_event: dict
):
    """Embed lineage metadata in Parquet file"""
    # Read existing Parquet file
    table = pq.read_table(parquet_path)
    
    # Add lineage metadata to custom metadata
    custom_metadata = {
        "lineage": json.dumps(lineage_event),
        "provenance": json.dumps({
            "inputs": lineage_event["inputs"],
            "transformation": lineage_event["transformation"]
        })
    }
    
    # Write with custom metadata
    pq.write_table(
        table,
        parquet_path,
        metadata_collector=lambda metadata: metadata.update(custom_metadata)
    )
```

#### JSON-LD Provenance Graphs

**Provenance Graph**:

```json
{
  "@context": {
    "prov": "http://www.w3.org/ns/prov#",
    "schema": "https://schema.org/"
  },
  "@graph": [
    {
      "@id": "prov:Activity-transformation-abc123",
      "@type": "prov:Activity",
      "prov:startedAtTime": "2024-01-15T10:30:00Z",
      "prov:endedAtTime": "2024-01-15T10:50:00Z",
      "prov:used": {
        "@id": "prov:Entity-osm-roads-v1",
        "@type": "prov:Entity",
        "schema:identifier": "osm-roads-v1",
        "schema:version": "1.5.0"
      },
      "prov:wasAssociatedWith": {
        "@id": "prov:Agent-data-eng",
        "@type": "prov:Person",
        "schema:email": "data-eng@example.com"
      },
      "prov:generated": {
        "@id": "prov:Entity-roads-v2",
        "@type": "prov:Entity",
        "schema:identifier": "roads-v2",
        "schema:version": "2.1.0"
      }
    }
  ]
}
```

#### MLflow Metadata

**MLflow Provenance**:

```python
# mlflow/provenance.py
import mlflow

def log_provenance(
    training_data: str,
    training_data_version: str,
    code_version: str,
    environment_hash: str
):
    """Log provenance to MLflow"""
    with mlflow.start_run():
        # Log data provenance
        mlflow.log_param("training_data", training_data)
        mlflow.log_param("training_data_version", training_data_version)
        mlflow.log_param("code_version", code_version)
        mlflow.log_param("environment_hash", environment_hash)
        
        # Log parent runs (if applicable)
        # mlflow.set_tag("mlflow.parentRunId", parent_run_id)
```

#### Prefect Lineage Integration

**Prefect Task with Lineage**:

```python
# prefect/tasks/etl_with_lineage.py
from prefect import task, flow
from provenance.postgres_lineage import record_lineage_event

@task
def transform_roads(input_path: str, output_path: str):
    """Transform roads data with lineage tracking"""
    # Load input
    df = pl.read_parquet(input_path)
    
    # Transform
    df_transformed = df.with_columns([
        pl.col("risk_level").cast(pl.Utf8).alias("risk_level")
    ])
    
    # Write output
    df_transformed.write_parquet(output_path)
    
    # Record lineage
    record_lineage_event(
        inputs=[{
            "datasetId": "osm-roads-v1",
            "version": "1.5.0",
            "location": input_path,
            "checksum": compute_checksum(input_path)
        }],
        outputs=[{
            "datasetId": "roads-v2",
            "version": "2.1.0",
            "location": output_path,
            "checksum": compute_checksum(output_path)
        }],
        transformation={
            "code": "git://github.com/org/etl-pipelines@abc123",
            "codeHash": compute_code_hash(),
            "environment": compute_environment_hash(),
            "executedBy": "prefect-job-789",
            "executionId": prefect.context.get("run_id")
        },
        user="data-eng@example.com",
        cluster="rke2-prod",
        node="worker-01"
    )
    
    return output_path

@flow
def etl_roads_flow():
    """ETL flow with lineage tracking"""
    input_path = "s3://data-lake/datasets/osm-roads/v1/"
    output_path = "s3://data-lake/datasets/roads/v2/"
    
    transform_roads(input_path, output_path)
```

### Querying Provenance

**Trace Dataset Backward**:

```sql
-- Find all upstream datasets for a given dataset
WITH RECURSIVE upstream_lineage AS (
    -- Base case: start with target dataset
    SELECT 
        pr.dataset_id,
        pr.dataset_version,
        pr.event_id,
        0 AS depth
    FROM provenance_relationships pr
    WHERE pr.dataset_id = 'roads-v2'
      AND pr.relationship_type = 'output'
    
    UNION
    
    -- Recursive case: find inputs to events that produced outputs
    SELECT 
        pr_input.dataset_id,
        pr_input.dataset_version,
        pr_input.event_id,
        ul.depth + 1
    FROM upstream_lineage ul
    JOIN provenance_relationships pr_output
        ON ul.event_id = pr_output.event_id
        AND pr_output.relationship_type = 'output'
    JOIN provenance_events pe
        ON pr_output.event_id = pe.event_id
    JOIN provenance_relationships pr_input
        ON pe.event_id = pr_input.event_id
        AND pr_input.relationship_type = 'input'
    WHERE ul.depth < 10  -- Limit recursion depth
)
SELECT DISTINCT
    dataset_id,
    dataset_version,
    depth
FROM upstream_lineage
ORDER BY depth, dataset_id;
```

**Show All Downstream Dependencies**:

```sql
-- Find all downstream datasets for a given dataset
WITH RECURSIVE downstream_lineage AS (
    -- Base case: start with source dataset
    SELECT 
        pr.dataset_id,
        pr.dataset_version,
        pr.event_id,
        0 AS depth
    FROM provenance_relationships pr
    WHERE pr.dataset_id = 'osm-roads-v1'
      AND pr.relationship_type = 'input'
    
    UNION
    
    -- Recursive case: find outputs from events that used inputs
    SELECT 
        pr_output.dataset_id,
        pr_output.dataset_version,
        pr_output.event_id,
        dl.depth + 1
    FROM downstream_lineage dl
    JOIN provenance_relationships pr_input
        ON dl.event_id = pr_input.event_id
        AND pr_input.relationship_type = 'input'
    JOIN provenance_events pe
        ON pr_input.event_id = pe.event_id
    JOIN provenance_relationships pr_output
        ON pe.event_id = pr_output.event_id
        AND pr_output.relationship_type = 'output'
    WHERE dl.depth < 10
)
SELECT DISTINCT
    dataset_id,
    dataset_version,
    depth
FROM downstream_lineage
ORDER BY depth, dataset_id;
```

**Provenance DAG Visualization**:

```mermaid
graph TD
    OSM["osm-roads-v1<br/>v1.5.0"] -->|input| T1[Transform<br/>ETL Job 789]
    Census["census-boundaries-v1<br/>v1.0.0"] -->|input| T1
    T1 -->|output| Roads["roads-v2<br/>v2.1.0"]
    Roads -->|input| ML[ML Training<br/>MLflow Run 456]
    ML -->|output| Model["risk-model-v3<br/>v3.0.0"]
    Roads -->|input| API[API Service<br/>FastAPI]
    Model -->|input| Inference[Inference<br/>ONNX Runtime]
```

## Validation Workflows

### Great Expectations Schema Enforcement

**Expectation Suite**:

```python
# validation/great_expectations/roads_v2_suite.py
import great_expectations as ge
from great_expectations.dataset import PandasDataset

def create_roads_v2_expectation_suite():
    """Create comprehensive expectation suite"""
    suite = ge.ExpectationSuite("roads_v2")
    
    # Schema expectations
    suite.expect_table_column_count_to_equal(5)
    suite.expect_table_columns_to_match_ordered_list([
        "road_id", "road_name", "risk_level", "geometry", "created_at"
    ])
    
    # Column existence
    for col in ["road_id", "road_name", "risk_level", "geometry", "created_at"]:
        suite.expect_column_to_exist(col)
    
    # Type expectations
    suite.expect_column_values_to_be_of_type("road_id", "int64")
    suite.expect_column_values_to_be_of_type("road_name", "object")
    suite.expect_column_values_to_be_of_type("risk_level", "object")
    
    # Constraint expectations
    suite.expect_column_values_to_be_unique("road_id")
    suite.expect_column_values_to_not_be_null("road_id")
    suite.expect_column_values_to_not_be_null("risk_level")
    suite.expect_column_values_to_be_in_set(
        "risk_level",
        ["low", "medium", "high"]
    )
    
    # Quality expectations
    suite.expect_column_values_to_not_be_null("road_name", mostly=0.98)
    suite.expect_column_value_lengths_to_be_between("road_name", 1, 255)
    
    # Spatial expectations (custom expectation)
    suite.expect_column_values_to_be_valid_geometry("geometry")
    
    return suite
```

**Validation Checkpoint**:

```python
# validation/great_expectations/checkpoints/roads_v2_checkpoint.py
from great_expectations.checkpoint import Checkpoint

checkpoint = Checkpoint(
    name="roads_v2_checkpoint",
    config_version=1,
    expectation_suite_name="roads_v2",
    validations=[
        {
            "batch_request": {
                "datasource_name": "s3_datasource",
                "data_asset_name": "roads_v2"
            }
        }
    ]
)
```

### Polars/DuckDB Schema Validation

**Polars Validation**:

```python
# validation/polars/validate_schema.py
import polars as pl
from typing import Dict, Any, List

def validate_schema(
    df: pl.DataFrame,
    contract: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """Validate DataFrame against contract schema"""
    errors = []
    schema = contract["schema"]
    
    # Check required fields
    required_fields = set(schema["requiredFields"])
    actual_fields = set(df.columns)
    missing_fields = required_fields - actual_fields
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")
    
    # Check field types and constraints
    for field in schema["fields"]:
        field_name = field["name"]
        if field_name not in df.columns:
            continue
        
        # Type validation
        expected_type = field["type"]["parquet"]
        actual_type = str(df[field_name].dtype)
        type_map = {
            "int64": "Int64",
            "string": "Utf8",
            "float64": "Float64",
            "bool": "Boolean"
        }
        if type_map.get(expected_type) != actual_type:
            errors.append(
                f"Field {field_name}: type mismatch "
                f"(expected {expected_type}, got {actual_type})"
            )
        
        # Constraint validation
        if "constraints" in field:
            constraints = field["constraints"]
            
            # Allowed values
            if "allowedValues" in constraints:
                allowed = set(constraints["allowedValues"])
                actual = set(df[field_name].unique().to_list())
                invalid = actual - allowed
                if invalid:
                    errors.append(
                        f"Field {field_name}: invalid values {invalid}"
                    )
            
            # Nullability
            if not field.get("nullable", False):
                null_count = df[field_name].null_count()
                if null_count > 0:
                    errors.append(
                        f"Field {field_name}: {null_count} null values "
                        f"(not allowed)"
                    )
    
    return len(errors) == 0, errors
```

**DuckDB Validation**:

```sql
-- DuckDB schema validation
CREATE TABLE validation_results (
    field_name TEXT,
    check_type TEXT,
    status TEXT,
    message TEXT,
    checked_at TIMESTAMPTZ DEFAULT now()
);

-- Validate schema
INSERT INTO validation_results (field_name, check_type, status, message)
SELECT
    'road_id' AS field_name,
    'type_check' AS check_type,
    CASE
        WHEN typeof(road_id) = 'BIGINT' THEN 'pass'
        ELSE 'fail'
    END AS status,
    'Type check: ' || typeof(road_id) AS message
FROM roads_v2
LIMIT 1;

-- Validate constraints
INSERT INTO validation_results (field_name, check_type, status, message)
SELECT
    'risk_level' AS field_name,
    'allowed_values' AS check_type,
    CASE
        WHEN risk_level IN ('low', 'medium', 'high') THEN 'pass'
        ELSE 'fail'
    END AS status,
    'Allowed values check' AS message
FROM roads_v2
WHERE risk_level NOT IN ('low', 'medium', 'high')
LIMIT 1;
```

### Postgres CHECK Constraints

**Schema with Constraints**:

```sql
-- Postgres table with comprehensive constraints
CREATE TABLE roads_v2 (
    road_id BIGINT PRIMARY KEY,
    road_name TEXT,
    risk_level VARCHAR(20) NOT NULL,
    geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Type constraints
    CONSTRAINT road_id_type CHECK (road_id IS NOT NULL),
    
    -- Value constraints
    CONSTRAINT risk_level_valid CHECK (risk_level IN ('low', 'medium', 'high')),
    CONSTRAINT road_name_length CHECK (LENGTH(road_name) <= 255),
    
    -- Spatial constraints
    CONSTRAINT geometry_valid CHECK (ST_IsValid(geometry)),
    CONSTRAINT geometry_type CHECK (ST_GeometryType(geometry) = 'ST_LineString'),
    CONSTRAINT geometry_crs CHECK (ST_SRID(geometry) = 4326),
    
    -- Temporal constraints
    CONSTRAINT created_at_range CHECK (
        created_at >= '2020-01-01'::timestamptz AND
        created_at <= '2024-12-31'::timestamptz
    )
);

-- Validation function
CREATE OR REPLACE FUNCTION validate_roads_v2()
RETURNS TABLE (
    field_name TEXT,
    check_type TEXT,
    status TEXT,
    message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'risk_level'::TEXT,
        'allowed_values'::TEXT,
        CASE
            WHEN COUNT(*) FILTER (WHERE risk_level NOT IN ('low', 'medium', 'high')) = 0
            THEN 'pass'::TEXT
            ELSE 'fail'::TEXT
        END,
        'Allowed values check'::TEXT
    FROM roads_v2;
END;
$$ LANGUAGE plpgsql;
```

### JSON Schema Validation

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["road_id", "risk_level", "geometry"],
  "properties": {
    "road_id": {
      "type": "integer",
      "minimum": 1
    },
    "road_name": {
      "type": "string",
      "maxLength": 255
    },
    "risk_level": {
      "type": "string",
      "enum": ["low", "medium", "high"]
    },
    "geometry": {
      "type": "object",
      "required": ["type", "coordinates"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["LineString"]
        },
        "coordinates": {
          "type": "array",
          "items": {
            "type": "array",
            "items": {
              "type": "number"
            }
          }
        }
      }
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

**Python Validation**:

```python
# validation/json_schema/validate_json.py
import jsonschema
import json

def validate_json_against_schema(data: dict, schema_path: str) -> bool:
    """Validate JSON data against JSON Schema"""
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"JSON validation failed: {e.message}")
```

### Prefect Validation Tasks

**Validation Flow**:

```python
# prefect/flows/validate_data_flow.py
from prefect import task, flow
from validation.polars.validate_schema import validate_schema
import yaml

@task
def load_contract(contract_path: str) -> dict:
    """Load data contract"""
    with open(contract_path, 'r') as f:
        return yaml.safe_load(f)

@task
def validate_data(data_path: str, contract: dict) -> tuple[bool, list]:
    """Validate data against contract"""
    df = pl.read_parquet(data_path)
    return validate_schema(df, contract)

@task
def quarantine_data(data_path: str, reason: str):
    """Move invalid data to quarantine"""
    # Move to quarantine bucket/table
    pass

@flow
def validate_data_flow(data_path: str, contract_path: str):
    """Data validation flow"""
    contract = load_contract(contract_path)
    is_valid, errors = validate_data(data_path, contract)
    
    if not is_valid:
        quarantine_data(data_path, "; ".join(errors))
        raise ValueError(f"Validation failed: {errors}")
    
    return True
```

### CI/CD Validation Pipeline

**GitHub Actions**:

```yaml
# .github/workflows/validate-data.yml
name: Validate Data Contracts

on:
  pull_request:
    paths:
      - 'data/**'
      - 'contracts/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install polars pyyaml jsonschema
      
      - name: Validate data contracts
        run: |
          python scripts/validate_contracts.py \
            --data-dir data/ \
            --contracts-dir contracts/ \
            --fail-on-error
```

## Metadata & Provenance Storage Architecture

### Unified Storage System

**Architecture**:

```mermaid
graph TB
    subgraph Storage["Metadata Storage"]
        PG["PostgreSQL<br/>Metadata Tables"]
        S3["S3/Object Store<br/>Metadata Bundles"]
        JSONLD["JSON-LD<br/>Semantic Layer"]
        ParquetMeta["Parquet<br/>Embedded Metadata"]
        MLflowMeta["MLflow<br/>Model Metadata"]
    end
    
    subgraph Registry["Metadata Registry"]
        API["Metadata API<br/>(FastAPI)"]
        Query["Query Interface<br/>(SQL + GraphQL)"]
    end
    
    subgraph Consumers["Consumers"]
        ETL["ETL Pipelines"]
        ML["ML Pipelines"]
        GIS["GIS Applications"]
        Dashboards["Dashboards"]
    end
    
    PG --> API
    S3 --> API
    JSONLD --> API
    ParquetMeta --> API
    MLflowMeta --> API
    
    API --> Query
    Query --> ETL
    Query --> ML
    Query --> GIS
    Query --> Dashboards
```

### Postgres Metadata Tables

**Schema**:

```sql
-- Dataset metadata table
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    owner_email TEXT NOT NULL,
    tags TEXT[],
    domains TEXT[],
    spatial_extent JSONB,
    temporal_extent JSONB,
    data_quality JSONB,
    access_level TEXT NOT NULL,
    update_frequency TEXT,
    storage_location TEXT NOT NULL,
    storage_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Column metadata table
CREATE TABLE columns (
    column_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    column_name TEXT NOT NULL,
    sql_type TEXT,
    parquet_type TEXT,
    semantic_type TEXT,
    description TEXT,
    nullable BOOLEAN NOT NULL DEFAULT true,
    constraints JSONB,
    valid_range JSONB,
    category_mappings JSONB,
    foreign_key JSONB,
    geospatial JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(dataset_id, column_name)
);

-- Operational metadata table
CREATE TABLE operational_metadata (
    operational_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    last_updated TIMESTAMPTZ NOT NULL,
    etl_job_id TEXT,
    etl_job_run_id TEXT,
    record_count BIGINT,
    file_count INTEGER,
    total_size_bytes BIGINT,
    checksum TEXT,
    validation_status TEXT,
    validation_timestamp TIMESTAMPTZ,
    quality_metrics JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX idx_datasets_version ON datasets(dataset_id, version);
CREATE INDEX idx_datasets_tags ON datasets USING GIN(tags);
CREATE INDEX idx_columns_dataset ON columns(dataset_id);
CREATE INDEX idx_operational_dataset ON operational_metadata(dataset_id, last_updated);
```

### S3/Object Store Metadata Bundles

**Directory Structure**:

```
s3://metadata-registry/
  datasets/
    roads-v2/
      metadata.yaml
      schema.json
      contract.yaml
      lineage.jsonld
      quality-report.json
  schemas/
    roads-v2/
      v2.0.0/
        schema.json
        contract.yaml
      v2.1.0/
        schema.json
        contract.yaml
  contracts/
    api-roads-v1.yaml
    etl-roads-v2.yaml
    ml-risk-model-input-v1.yaml
```

### JSON-LD Metadata Layer

**Semantic Metadata Store**:

```json
{
  "@context": {
    "schema": "https://schema.org/",
    "dc": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#"
  },
  "@graph": [
    {
      "@id": "https://data.example.com/datasets/roads-v2",
      "@type": "schema:Dataset",
      "schema:identifier": "roads-v2",
      "schema:version": "2.1.0",
      "schema:name": "Road Network Dataset",
      "prov:wasDerivedFrom": {
        "@id": "https://data.example.com/datasets/osm-roads-v1"
      }
    }
  ]
}
```

### Parquet Embedded Metadata

**Metadata in Parquet Footer**:

```python
# metadata/parquet_metadata.py
import pyarrow.parquet as pq
import pyarrow as pa
import json

def embed_metadata_in_parquet(
    parquet_path: str,
    metadata: dict
):
    """Embed metadata in Parquet file"""
    table = pq.read_table(parquet_path)
    
    # Add custom metadata
    custom_metadata = {
        "dataset_id": metadata["identifier"],
        "version": metadata["version"],
        "schema_version": metadata["schemaVersion"],
        "provenance": json.dumps(metadata["provenance"]),
        "quality": json.dumps(metadata["dataQuality"])
    }
    
    # Write with metadata
    pq.write_table(
        table,
        parquet_path,
        metadata_collector=lambda metadata: metadata.update(custom_metadata)
    )
```

### MLflow Metadata Integration

**Model Metadata**:

```python
# mlflow/metadata_integration.py
import mlflow

def log_dataset_metadata(
    dataset_id: str,
    dataset_version: str,
    schema_version: str
):
    """Log dataset metadata to MLflow"""
    mlflow.log_param("training_dataset_id", dataset_id)
    mlflow.log_param("training_dataset_version", dataset_version)
    mlflow.log_param("training_schema_version", schema_version)
    
    # Log contract reference
    mlflow.log_artifact("contracts/ml-risk-model-input-v1.yaml")
```

### Metadata API

**FastAPI Metadata Service**:

```python
# api/metadata_service.py
from fastapi import FastAPI, HTTPException
from typing import Optional
import psycopg2

app = FastAPI()

@app.get("/datasets/{dataset_id}")
def get_dataset_metadata(
    dataset_id: str,
    version: Optional[str] = None
):
    """Get dataset metadata"""
    conn = psycopg2.connect("postgresql://user:pass@localhost/metadata")
    cur = conn.cursor()
    
    if version:
        cur.execute("""
            SELECT * FROM datasets
            WHERE dataset_id = %s AND version = %s
        """, (dataset_id, version))
    else:
        cur.execute("""
            SELECT * FROM datasets
            WHERE dataset_id = %s
            ORDER BY version DESC
            LIMIT 1
        """, (dataset_id,))
    
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": row[0],
        "name": row[1],
        "version": row[3],
        "schema_version": row[4],
        # ... other fields
    }

@app.get("/datasets/{dataset_id}/lineage")
def get_dataset_lineage(dataset_id: str):
    """Get dataset lineage"""
    # Query provenance tables
    pass
```

## Data Quality & SLA Governance

### Required Metrics

**Quality Metrics**:

```yaml
# governance/quality_requirements.yaml
qualityRequirements:
  completeness:
    road_id: 1.0
    geometry: 1.0
    risk_level: 1.0
    road_name: 0.98
  
  uniqueness:
    road_id: 1.0
  
  spatialValidity:
    geometry: 0.98
  
  valueDistribution:
    risk_level:
      low: [0.5, 0.7]
      medium: [0.2, 0.4]
      high: [0.05, 0.15]
  
  timeliness:
    maxAgeHours: 24
  
  accuracy:
    spatialAccuracyMeters: 10
```

### Required Checks

**Validation Checklist**:

```python
# governance/validation_checklist.py
VALIDATION_CHECKS = [
    {
        "name": "schema_validation",
        "description": "Verify schema matches contract",
        "required": True
    },
    {
        "name": "completeness_check",
        "description": "Verify required fields are not null",
        "required": True
    },
    {
        "name": "uniqueness_check",
        "description": "Verify primary key uniqueness",
        "required": True
    },
    {
        "name": "spatial_validity",
        "description": "Verify geometry validity",
        "required": True
    },
    {
        "name": "value_distribution",
        "description": "Verify value distribution within expected ranges",
        "required": False
    }
]
```

### SLA Definitions

**SLA Policy**:

```yaml
# governance/sla_policy.yaml
sla:
  updateFrequency: daily
  updateWindow: "00:00-06:00 UTC"
  backfillWindow: 7 days
  correctionWorkflow: automated
  errorBudget:
    availability: 0.99
    dataQuality: 0.95
    timeliness: 0.98
  consequences:
    - trigger: "availability < 0.99 for 3 consecutive days"
      action: "escalate to data engineering lead"
    - trigger: "dataQuality < 0.95"
      action: "quarantine dataset, notify consumers"
```

### Review Workflow

**Schema Change Review Process**:

```mermaid
graph LR
    Propose["Propose Schema<br/>Change"] --> Review["Technical Review<br/>(Data Eng)"]
    Review -->|Approved| Contract["Update Contract<br/>(Version Bump)"]
    Review -->|Rejected| Propose
    Contract --> Test["Test in<br/>Staging"]
    Test -->|Pass| Deploy["Deploy to<br/>Production"]
    Test -->|Fail| Contract
    Deploy --> Notify["Notify<br/>Consumers"]
```

## Integration With Existing Repo Content

### PostGIS Ingestion

**Metadata for PostGIS Tables**:

```sql
-- PostGIS table with metadata
CREATE TABLE roads_v2 (
    road_id BIGINT PRIMARY KEY,
    road_name TEXT,
    risk_level VARCHAR(20) NOT NULL,
    geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Add metadata comment
COMMENT ON TABLE roads_v2 IS 'Road network dataset v2.1.0';
COMMENT ON COLUMN roads_v2.geometry IS 'Road geometry in EPSG:4326';
COMMENT ON COLUMN roads_v2.risk_level IS 'Risk level: low, medium, or high';

-- Register in metadata registry
INSERT INTO datasets (
    dataset_id, name, version, schema_version,
    storage_location, storage_type
) VALUES (
    'roads-v2',
    'Road Network Dataset',
    '2.1.0',
    '2.0.0',
    'postgresql://localhost/mydb.roads_v2',
    'postgres'
);
```

### Parquet FDWs

**Metadata for Parquet FDW**:

```sql
-- Foreign table with metadata
CREATE FOREIGN TABLE roads_v2_parquet (
    road_id BIGINT,
    road_name TEXT,
    risk_level VARCHAR(20),
    geometry BYTEA,
    created_at TIMESTAMPTZ
) SERVER parquet_s3_fdw
OPTIONS (
    filename 's3://data-lake/datasets/roads/v2/',
    format 'parquet'
);

-- Metadata registration
INSERT INTO datasets (
    dataset_id, name, version, schema_version,
    storage_location, storage_type
) VALUES (
    'roads-v2',
    'Road Network Dataset',
    '2.1.0',
    '2.0.0',
    's3://data-lake/datasets/roads/v2/',
    'parquet'
);
```

### MLflow Model Metadata

**Model with Schema Version**:

```python
# mlflow/model_with_metadata.py
import mlflow

def train_and_log_model(
    training_data: str,
    training_data_version: str,
    schema_version: str
):
    """Train model and log with metadata"""
    with mlflow.start_run():
        # Train model
        model = train_model(training_data)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metadata
        mlflow.log_param("training_data", training_data)
        mlflow.log_param("training_data_version", training_data_version)
        mlflow.log_param("schema_version", schema_version)
        mlflow.log_param("contract_version", "ml-risk-model-input-v1")
        
        # Log contract
        mlflow.log_artifact("contracts/ml-risk-model-input-v1.yaml")
```

### ONNX Inference Panels

**Schema Verification in Inference**:

```python
# onnx/inference_with_schema.py
import onnxruntime as ort
from validation.polars.validate_schema import validate_schema

def infer_with_schema_verification(
    input_data: np.ndarray,
    model_path: str,
    contract_path: str
):
    """Run inference with schema verification"""
    # Verify input schema
    validate_schema(input_data, contract_path)
    
    # Load model
    session = ort.InferenceSession(model_path)
    
    # Run inference
    outputs = session.run(None, {"input": input_data})
    
    return outputs
```

### Kafka/Timescale Pipelines

**Provenance in Kafka**:

```python
# kafka/provenance_tracking.py
from kafka import KafkaProducer
import json

def produce_with_provenance(
    topic: str,
    data: dict,
    provenance: dict
):
    """Produce Kafka message with provenance"""
    producer = KafkaProducer(
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    message = {
        "data": data,
        "provenance": provenance,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    producer.send(topic, message)
    producer.flush()
```

### JSON-LD Driven NiceGUI Forms

**Schema-Driven Form Generation**:

```python
# nicegui/schema_driven_form.py
from nicegui import ui
import json

def create_form_from_schema(schema_path: str):
    """Create NiceGUI form from JSON-LD schema"""
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    form_data = {}
    
    for field in schema["schema"]["columns"]:
        field_name = field["name"]
        field_type = field["type"]["semantic"]
        
        if field_type == "identifier":
            form_data[field_name] = ui.number(
                label=field["description"],
                value=0
            )
        elif field_type == "name":
            form_data[field_name] = ui.input(
                label=field["description"],
                placeholder="Enter value"
            )
        elif field_type == "category":
            allowed_values = field["constraints"]["allowedValues"]
            form_data[field_name] = ui.select(
                label=field["description"],
                options=allowed_values
            )
    
    return form_data
```

## Anti-Patterns

### No Owner for a Dataset

**Symptom**: Dataset exists but no one knows who owns it or who to contact for changes.

**Why It's Dangerous**: Changes can't be approved, issues can't be resolved, quality degrades.

**How to Fix**: Assign owner in metadata, create ownership review process.

**Preventative Best Practice**: Require owner assignment when creating dataset metadata.

### "Schema by Accident"

**Symptom**: Schema evolves organically without documentation or versioning.

**Why It's Dangerous**: Schema drift breaks downstream systems, no way to track changes.

**How to Fix**: Document current schema, create version, establish change process.

**Preventative Best Practice**: Require schema documentation and versioning from the start.

### No Versioning

**Symptom**: Datasets overwritten in place, no version history.

**Why It's Dangerous**: Can't reproduce past results, can't roll back changes.

**How to Fix**: Implement versioned storage (directories, schemas, tags).

**Preventative Best Practice**: Always version datasets and schemas.

### Breaking Changes Pushed Silently

**Symptom**: Schema changes break consumers without notification.

**Why It's Dangerous**: Production systems fail, data quality issues go undetected.

**How to Fix**: Implement contract validation, versioning, and change notifications.

**Preventative Best Practice**: Require contract updates and consumer notification for breaking changes.

### Silent Field Repurposing

**Symptom**: Field meaning changes without documentation or version bump.

**Why It's Dangerous**: Consumers use field incorrectly, results are wrong.

**How to Fix**: Document field semantics, version schema on semantic changes.

**Preventative Best Practice**: Treat semantic changes as breaking changes.

### No Lineage Tracking

**Symptom**: Can't trace data from source to consumption.

**Why It's Dangerous**: Can't debug issues, can't audit data flow, can't ensure reproducibility.

**How to Fix**: Implement provenance tracking in ETL pipelines.

**Preventative Best Practice**: Record lineage for all transformations.

### Mixing Spatial Reference Systems Without Metadata

**Symptom**: GeoParquet files use different CRS without recording it.

**Why It's Dangerous**: Spatial analysis produces wrong results, visualizations misaligned.

**How to Fix**: Record CRS in metadata, validate CRS consistency.

**Preventative Best Practice**: Always record CRS in metadata, validate on ingestion.

### Storing Metadata Only in Code Comments

**Symptom**: Metadata exists only in code, not in searchable registry.

**Why It's Dangerous**: Metadata is hard to find, can't be queried, gets out of sync.

**How to Fix**: Move metadata to dedicated registry (Postgres, JSON-LD, etc.).

**Preventative Best Practice**: Store metadata in dedicated registry from the start.

### Copy-Paste ETL Pipelines with No Provenance Trail

**Symptom**: ETL code copied without recording lineage.

**Why It's Dangerous**: Can't trace data flow, can't debug issues, can't ensure reproducibility.

**How to Fix**: Refactor to use shared libraries, record provenance in all pipelines.

**Preventative Best Practice**: Build provenance tracking into ETL framework.

## Final Summary

### Complete Metadata Checklist

**Dataset Creation**:
- [ ] Dataset identifier assigned
- [ ] Name and description documented
- [ ] Owner assigned
- [ ] Tags and domains defined
- [ ] Spatial/temporal extents recorded
- [ ] Schema version created
- [ ] Contract defined
- [ ] Storage location documented
- [ ] Access level set
- [ ] Update frequency defined

**Schema Definition**:
- [ ] All columns documented
- [ ] Types defined (SQL, Parquet, semantic)
- [ ] Constraints specified
- [ ] Valid ranges defined
- [ ] Category mappings documented
- [ ] Foreign keys identified
- [ ] Geospatial metadata recorded

**Provenance**:
- [ ] Input datasets identified
- [ ] Transformation code versioned
- [ ] Environment hash recorded
- [ ] Execution metadata logged
- [ ] Lineage events stored

**Quality**:
- [ ] Quality metrics defined
- [ ] Validation checks implemented
- [ ] SLA defined
- [ ] Monitoring configured

### Data Contract Creation Checklist

**Contract Definition**:
- [ ] Contract identifier assigned
- [ ] Version set
- [ ] Type defined (API, ETL, geospatial, model)
- [ ] Owner assigned
- [ ] Consumers identified

**Schema Specification**:
- [ ] Required fields listed
- [ ] Field types specified
- [ ] Constraints defined
- [ ] Valid values documented
- [ ] Behavioral expectations specified

**Enforcement**:
- [ ] Validation implemented
- [ ] CI/CD checks configured
- [ ] Quarantine process defined
- [ ] Notification process established

### Schema Review Workflow

**Change Proposal**:
1. Create schema change proposal (PR, issue, ADR)
2. Document breaking vs non-breaking changes
3. Identify affected consumers
4. Propose migration strategy

**Review Process**:
1. Technical review (data engineering team)
2. Consumer impact assessment
3. Contract update review
4. Testing in staging

**Approval & Deployment**:
1. Approve change
2. Update contract
3. Version schema
4. Deploy to production
5. Notify consumers

### Readiness Scorecard

**Metadata Maturity**:
- [ ] All datasets have metadata
- [ ] All schemas are versioned
- [ ] All contracts are defined
- [ ] Provenance is tracked
- [ ] Quality metrics are monitored

**Governance**:
- [ ] Change process defined
- [ ] Review workflow established
- [ ] SLA policies in place
- [ ] Monitoring configured

**Integration**:
- [ ] ETL pipelines record provenance
- [ ] ML models reference schema versions
- [ ] APIs validate contracts
- [ ] CI/CD enforces contracts

### GitHub/GitLab Workflow Suggestion

**Repository Structure**:

```
data-governance/
├── datasets/
│   ├── roads-v2/
│   │   ├── metadata.yaml
│   │   ├── schema.json
│   │   └── lineage.jsonld
│   └── ...
├── contracts/
│   ├── api-roads-v1.yaml
│   ├── etl-roads-v2.yaml
│   └── ...
├── schemas/
│   ├── roads-v2/
│   │   ├── v2.0.0/
│   │   └── v2.1.0/
│   └── ...
└── validation/
    ├── great_expectations/
    ├── polars/
    └── ...
```

**Workflow**:

```yaml
# .github/workflows/data-governance.yml
name: Data Governance

on:
  pull_request:
    paths:
      - 'datasets/**'
      - 'contracts/**'
      - 'schemas/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate contracts
        run: python scripts/validate_contracts.py
      - name: Validate schemas
        run: python scripts/validate_schemas.py
      - name: Check metadata completeness
        run: python scripts/check_metadata.py
```

### Future Extensibility

**RDF/OWL Integration**:
- Extend JSON-LD metadata to full RDF
- Use OWL for semantic relationships
- Implement SHACL for constraint validation

**SHACL Validation**:
- Define SHACL shapes for data contracts
- Validate RDF data against shapes
- Integrate with existing validation workflows

**JSON-LD Forms**:
- Generate forms from JSON-LD schemas
- Validate form inputs against contracts
- Record form submissions with provenance

## See Also

- **[RDF/OWL Metadata Automation](../architecture-design/rdf-owl-metadata-automation.md)** - Dynamic knowledge graphs
- **[Data Engineering Best Practices](../database-data/data-engineering.md)** - ETL/ELT patterns
- **[GeoParquet Best Practices](../database-data/geoparquet.md)** - Geospatial data formats
- **[PostgreSQL Best Practices](../postgres/index.md)** - Database patterns

---

*This guide provides a complete framework for metadata, schema governance, and data provenance. Start with metadata model and contracts, then implement validation and provenance tracking. The goal is trustworthy, reproducible, auditable data systems.*

