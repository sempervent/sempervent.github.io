# Solr + Postgres JSONB: Faceted, Full-Text, and Geo Search at Speed

**Objective**: Master the art of hybrid search—keeping relational truth in Postgres while unleashing Solr's full-text, faceting, and geospatial powers on JSONB data. When you need lightning-fast search with facets, suggestions, and geo queries, when you want to combine the reliability of Postgres with the search prowess of Solr—this tutorial becomes your weapon of choice.

## Overview

We'll keep truth in Postgres and let Solr hunt—fast, fuzzy, facet-happy. JSONB becomes the launch pad; Solr becomes the engine. This hybrid approach gives you the best of both worlds: Postgres handles relational constraints and complex filters, while Solr delivers blazing-fast text search, faceting, suggestions, and geospatial queries.

## Architecture: Hybrid Search

```mermaid
flowchart LR
  A[(Postgres<br/>JSONB source)] -->|ETL (Python)| S[(Solr Core)]
  U[FastAPI / Client] -->|filters (org, dates)| A
  U -->|q=fulltext, facets, suggest| S
  S -->|ranked doc IDs| U
  A -->|final rows by IDs| U
```

**Why it works**: Postgres handles relational truth and strict filters; Solr handles text scoring, facets, synonyms, & geo—then we intersect.

## Docker Compose Stack

```yaml
# docker-compose.yml
version: "3.9"

x-hc: &hc { interval: 5s, timeout: 3s, retries: 30 }

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: app
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL","pg_isready -U postgres -d app"]
      <<: *hc
    volumes: ["pgdata:/var/lib/postgresql/data"]

  solr:
    image: solr:9
    command: bash -lc "precreate-core appcore && solr-foreground"
    ports: ["8983:8983"]
    healthcheck:
      test: ["CMD","curl","-sf","http://localhost:8983/solr/admin/cores?action=STATUS"]
      <<: *hc
    volumes: ["solrdata:/var/solr"]

  fastapi:
    build: ./fastapi
    depends_on:
      postgres: { condition: service_healthy }
      solr: { condition: service_healthy }
    environment:
      DB_DSN: postgresql://postgres:postgres@postgres:5432/app
      SOLR_URL: http://solr:8983/solr/appcore
    ports: ["8000:8000"]

  seed:
    build: ./seed
    depends_on:
      postgres: { condition: service_healthy }
      solr: { condition: service_healthy }
    command: ["python","seed_and_index.py"]

  # Optional profiles:
  kafka:
    image: bitnami/kafka:3.7
    profiles: ["cdc"]
    environment:
      KAFKA_ENABLE_KRAFT: "yes"
      KAFKA_CFG_PROCESS_ROLES: "broker,controller"
      KAFKA_CFG_CONTROLLER_LISTENER_NAMES: "CONTROLLER"
      KAFKA_CFG_LISTENERS: "PLAINTEXT://:9092,CONTROLLER://:9093"
      KAFKA_CFG_ADVERTISED_LISTENERS: "PLAINTEXT://kafka:9092"
      ALLOW_PLAINTEXT_LISTENER: "yes"
    ports: ["9092:9092"]

  grafana:
    image: grafana/grafana:latest
    profiles: ["viz"]
    ports: ["3000:3000"]

  maplibre:
    image: nginx:alpine
    profiles: ["geo"]
    volumes: ["./maplibre:/usr/share/nginx/html:ro"]
    ports: ["8080:80"]

volumes:
  pgdata:
  solrdata:
```

### Operator Recipes

```bash
# Core (DB + Solr + FastAPI + seeding/indexing)
docker compose up -d --build

# Add CDC path later:
docker compose --profile cdc up -d

# Visualization or geo demo:
docker compose --profile viz up -d
docker compose --profile geo up -d
```

## Postgres Model: JSONB + Relational Spine

```sql
-- Create the articles table with JSONB payload
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org TEXT NOT NULL,
  published_at TIMESTAMPTZ NOT NULL,
  author TEXT,
  geom GEOGRAPHY(Point,4326),        -- optional geo hook
  doc JSONB NOT NULL,                 -- semi-structured payload
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Filter indexes for "hybrid search intersection" stage
CREATE INDEX ON articles (org);
CREATE INDEX ON articles (published_at);
CREATE INDEX ON articles USING GIST (geom); -- if geo included

-- JSONB indexes for common queries
CREATE INDEX ON articles USING GIN (doc);
CREATE INDEX ON articles USING GIN ((doc->'tags'));

-- Example seed data
INSERT INTO articles (org, published_at, author, geom, doc)
VALUES
('the-hollow-paper', now() - interval '1 day', 'K. Wolfe', ST_GeogFromText('POINT(-73.9857 40.7484)'),
 '{"title":"Night Markets","body":"Feral neon and steam.","tags":["city","night"],"lang":"en"}'),
('the-hollow-paper', now() - interval '2 days', 'D. Lake', ST_GeogFromText('POINT(-122.4194 37.7749)'),
 '{"title":"Salt Maps","body":"Cartography of appetite.","tags":["map","salt"],"lang":"en"}'),
('urban-echo', now() - interval '3 days', 'M. Chen', ST_GeogFromText('POINT(-74.0060 40.7128)'),
 '{"title":"Underground Networks","body":"Tunnels beneath the city.","tags":["infrastructure","urban"],"lang":"en"}'),
('data-dreams', now() - interval '4 days', 'A. Kim', ST_GeogFromText('POINT(-122.4194 37.7749)'),
 '{"title":"Algorithmic Landscapes","body":"Code shapes the world.","tags":["tech","future"],"lang":"en"}');
```

## Solr Core & Schema Configuration

### Schema Definition

```json
{
  "add-field": [
    {"name":"id","type":"string","stored":true,"indexed":true},
    {"name":"org_s","type":"string","stored":true,"indexed":true},
    {"name":"published_dt","type":"pdate","stored":true,"indexed":true},
    {"name":"author_s","type":"string","stored":true,"indexed":true},
    {"name":"loc_p","type":"location","stored":true,"indexed":true},
    {"name":"title_t","type":"text_en","stored":true,"indexed":true},
    {"name":"body_t","type":"text_en","stored":false,"indexed":true},
    {"name":"tags_ss","type":"strings","stored":true,"indexed":true},
    {"name":"lang_s","type":"string","stored":true,"indexed":true}
  ],
  "add-copy-field":[
    {"source":"title_t","dest":"_text_"},
    {"source":"body_t","dest":"_text_"}
  ],
  "add-dynamic-field":[
    {"name":"*_t","type":"text_en","stored":false,"indexed":true},
    {"name":"*_s","type":"string","stored":true,"indexed":true},
    {"name":"*_ss","type":"strings","stored":true,"indexed":true},
    {"name":"*_i","type":"pint","stored":true,"indexed":true},
    {"name":"*_dt","type":"pdate","stored":true,"indexed":true},
    {"name":"*_p","type":"location","stored":true,"indexed":true}
  ]
}
```

### Synonyms Configuration

```text
# solr/appcore/conf/synonyms.txt
geo, geospatial, mapping, cartography
night, nocturnal, evening
city, urban, metropolitan
tech, technology, digital
```

### Suggester Configuration

```xml
<!-- solr/appcore/conf/solrconfig.xml -->
<SearchComponent name="suggest" class="solr.SuggestComponent">
  <lst name="suggester">
    <str name="name">titleSuggester</str>
    <str name="lookupImpl">FuzzyLookupFactory</str>
    <str name="dictionaryImpl">DocumentDictionaryFactory</str>
    <str name="field">title_t</str>
    <str name="suggestAnalyzerFieldType">text_en</str>
  </lst>
</SearchComponent>

<requestHandler name="/suggest" class="solr.SearchHandler" startup="lazy">
  <lst name="components"><str>suggest</str></lst>
  <lst name="defaults">
    <bool name="suggest">true</bool>
    <str name="suggest.dictionary">titleSuggester</str>
  </lst>
</requestHandler>
```

## Python Indexer: JSONB → Solr Documents

```python
# seed/seed_and_index.py
import os
import json
import time
import psycopg2
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@postgres:5432/app")
SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/appcore")

def rows_since(conn: psycopg2.extensions.connection, ts: datetime) -> List[Dict[str, Any]]:
    """Fetch rows changed since timestamp."""
    with conn.cursor() as cur:
        cur.execute("""
          SELECT id::text, org, published_at, author,
                 ST_X(ST_AsText(geom::geometry)) AS lon,
                 ST_Y(ST_AsText(geom::geometry)) AS lat,
                 doc
          FROM articles
          WHERE updated_at > %s
          ORDER BY updated_at
        """, (ts,))
        
        docs = []
        for row in cur:
            doc = {
                "id": row[0],
                "org_s": row[1],
                "published_dt": row[2].isoformat() if row[2] else None,
                "author_s": row[3],
                "loc_p": f"{row[5]},{row[4]}" if row[4] and row[5] else None,
                "title_t": row[6].get("title"),
                "body_t": row[6].get("body"),
                "tags_ss": row[6].get("tags", []),
                "lang_s": row[6].get("lang")
            }
            # Remove None values
            doc = {k: v for k, v in doc.items() if v is not None}
            docs.append(doc)
        
        return docs

def solr_upsert(docs: List[Dict[str, Any]]) -> None:
    """Upsert documents to Solr."""
    if not docs:
        return
    
    payload = {
        "add": [
            {"doc": doc, "commitWithin": 2000} 
            for doc in docs
        ]
    }
    
    try:
        r = requests.post(f"{SOLR_URL}/update", json=payload, timeout=30)
        r.raise_for_status()
        
        # Commit the changes
        requests.get(f"{SOLR_URL}/update?commit=true", timeout=10)
        print(f"✅ Indexed {len(docs)} documents")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Solr update failed: {e}")
        raise

def setup_solr_schema() -> None:
    """Setup Solr schema via Schema API."""
    schema_config = {
        "add-field": [
            {"name":"id","type":"string","stored":True,"indexed":True},
            {"name":"org_s","type":"string","stored":True,"indexed":True},
            {"name":"published_dt","type":"pdate","stored":True,"indexed":True},
            {"name":"author_s","type":"string","stored":True,"indexed":True},
            {"name":"loc_p","type":"location","stored":True,"indexed":True},
            {"name":"title_t","type":"text_en","stored":True,"indexed":True},
            {"name":"body_t","type":"text_en","stored":False,"indexed":True},
            {"name":"tags_ss","type":"strings","stored":True,"indexed":True},
            {"name":"lang_s","type":"string","stored":True,"indexed":True}
        ],
        "add-copy-field":[
            {"source":"title_t","dest":"_text_"},
            {"source":"body_t","dest":"_text_"}
        ]
    }
    
    try:
        r = requests.post(f"{SOLR_URL}/schema", json=schema_config, timeout=30)
        r.raise_for_status()
        print("✅ Solr schema configured")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Schema setup failed (may already exist): {e}")

def main():
    """Main indexing loop."""
    print("🚀 Starting Postgres → Solr indexer")
    
    # Setup Solr schema
    setup_solr_schema()
    
    # Connect to Postgres
    conn = psycopg2.connect(DB_DSN)
    checkpoint = datetime.utcnow() - timedelta(days=365)
    
    print(f"📊 Starting from checkpoint: {checkpoint.isoformat()}")
    
    try:
        while True:
            docs = rows_since(conn, checkpoint)
            if docs:
                solr_upsert(docs)
                checkpoint = datetime.utcnow()
                print(f"🔄 Updated checkpoint: {checkpoint.isoformat()}")
            else:
                print("⏳ No new documents, waiting...")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("🛑 Indexer stopped by user")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
```

## FastAPI Hybrid Search Endpoint

```python
# fastapi/app.py
import os
import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
import psycopg2
import requests
from urllib.parse import urlencode

DB_DSN = os.getenv("DB_DSN")
SOLR_URL = os.getenv("SOLR_URL")

app = FastAPI(title="Hybrid Search API", version="1.0.0")

def pg_candidates(org: Optional[str], date_from: Optional[str], 
                 date_to: Optional[str], tags: Optional[List[str]]) -> List[str]:
    """Phase 1: Get candidate IDs from Postgres filters."""
    sql = """
      SELECT id::text FROM articles
      WHERE (%s::text IS NULL OR org = %s)
        AND (%s::timestamptz IS NULL OR published_at >= %s)
        AND (%s::timestamptz IS NULL OR published_at <= %s)
        AND (%s::text[] IS NULL OR doc->'tags' ?| %s)
      LIMIT 5000
    """
    
    with psycopg2.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, (org, org, date_from, date_from, date_to, date_to, tags, tags))
        return [row[0] for row in cur.fetchall()]

@app.get("/search")
def search(
    q: str = "",
    org: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags: Optional[List[str]] = Query(default=None),
    near: Optional[str] = None,  # "lat,lon,radius"
    facets: bool = True,
    page: int = 1,
    size: int = 10
):
    """
    Hybrid search endpoint combining Postgres filters with Solr search.
    
    - q: Full-text search query
    - org: Organization filter
    - date_from/date_to: Date range filter
    - tags: Tag filters
    - near: Geo proximity (lat,lon,radius_km)
    - facets: Enable faceting
    - page/size: Pagination
    """
    
    # Phase 1: Postgres filters → candidate IDs
    candidate_ids = pg_candidates(org, date_from, date_to, tags)
    if not candidate_ids:
        return {"hits": [], "facets": {}, "total": 0}
    
    # Phase 2: Solr search with ID intersection
    solr_params = {
        "q": q or "*:*",
        "rows": size,
        "start": (page - 1) * size,
        "hl": "true",
        "hl.fl": "title_t,body_t",
        "defType": "edismax",
        "qf": "title_t^3 body_t^1",
        "fq": f"id:({' '.join(candidate_ids[:1000])})"  # Cap fq size
    }
    
    # Add geo search if specified
    if near:
        try:
            lat, lon, radius = near.split(",")
            solr_params.update({
                "sfield": "loc_p",
                "pt": f"{lat},{lon}",
                "d": radius,
                "sort": "geodist() asc"
            })
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid near format. Use: lat,lon,radius")
    
    # Add faceting
    if facets:
        solr_params.update({
            "facet": "true",
            "facet.field": ["org_s", "tags_ss"],
            "facet.range": "published_dt",
            "facet.range.gap": "+1DAY",
            "facet.range.start": "NOW-30DAYS/DAY",
            "facet.range.end": "NOW/DAY"
        })
    
    # Fire Solr query
    try:
        r = requests.get(f"{SOLR_URL}/select", params=solr_params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Solr query failed: {e}")
    
    # Process results
    hits = []
    highlighting = data.get("highlighting", {})
    
    for doc in data["response"]["docs"]:
        hit = {
            "id": doc["id"],
            "org": doc.get("org_s"),
            "title": doc.get("title_t"),
            "author": doc.get("author_s"),
            "tags": doc.get("tags_ss", []),
            "score": doc.get("score"),
            "snippet": " ".join(highlighting.get(doc["id"], {}).get("body_t", [])[:1]) if highlighting else None
        }
        hits.append(hit)
    
    # Extract facets
    facets_out = data.get("facet_counts", {}) if facets else {}
    
    return {
        "hits": hits,
        "facets": facets_out,
        "total": data["response"]["numFound"],
        "page": page,
        "size": size
    }

@app.get("/suggest")
def suggest(q: str = Query(..., min_length=1)):
    """Get search suggestions from Solr."""
    try:
        r = requests.get(f"{SOLR_URL}/suggest", params={
            "suggest": "true",
            "suggest.dictionary": "titleSuggester",
            "suggest.q": q
        }, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {e}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "hybrid-search"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Solr Features Gallery

### A) Full-Text Ranking with eDismax

```bash
# Basic text search with field boosting
curl "http://localhost:8983/solr/appcore/select?q=market steam&qf=title_t^3 body_t^1&defType=edismax"

# Phrase boosting for exact matches
curl "http://localhost:8983/solr/appcore/select?q=night market&pf=title_t^5&qf=title_t^3 body_t^1&defType=edismax"

# Synonym expansion (geo ≈ geospatial)
curl "http://localhost:8983/solr/appcore/select?q=geo mapping&defType=edismax"
```

### B) Faceted Search

```bash
# Field facets
curl "http://localhost:8983/solr/appcore/select?q=*:*&facet=true&facet.field=org_s&facet.field=tags_ss"

# Range facets for dates
curl "http://localhost:8983/solr/appcore/select?q=*:*&facet=true&facet.range=published_dt&facet.range.gap=+1DAY&facet.range.start=NOW-7DAYS/DAY&facet.range.end=NOW/DAY"
```

### C) Highlighting

```bash
# Get highlighted snippets
curl "http://localhost:8983/solr/appcore/select?q=neon steam&hl=true&hl.fl=title_t,body_t&hl.snippets=2"
```

### D) Search Suggestions

```bash
# Get title suggestions
curl "http://localhost:8983/solr/appcore/suggest?suggest=true&suggest.dictionary=titleSuggester&suggest.q=sal"
```

### E) Geospatial Queries

```bash
# Find documents within 10km of NYC
curl "http://localhost:8983/solr/appcore/select?q=*:*&sfield=loc_p&pt=40.7128,-74.0060&d=10&fq={!geofilt}&sort=geodist() asc"

# Geo distance with custom radius
curl "http://localhost:8983/solr/appcore/select?q=*:*&sfield=loc_p&pt=37.7749,-122.4194&d=5&fq={!geofilt}"
```

### F) JSON Facet API (Advanced)

```bash
# Nested facets: top tags per organization
curl "http://localhost:8983/solr/appcore/select?q=*:*&json.facet={
  \"orgs\": {
    \"type\": \"terms\",
    \"field\": \"org_s\",
    \"facet\": {
      \"top_tags\": {
        \"type\": \"terms\",
        \"field\": \"tags_ss\",
        \"limit\": 3
      }
    }
  }
}"
```

## Optional CDC Integration

### Kafka Connect Configuration

```json
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "postgres",
    "database.password": "postgres",
    "database.dbname": "app",
    "database.server.name": "app",
    "table.include.list": "public.articles",
    "plugin.name": "pgoutput"
  }
}
```

### Solr Sink Consumer

```python
# cdc/solr_sink.py
import json
import requests
from kafka import KafkaConsumer
from typing import Dict, Any

SOLR_URL = "http://solr:8983/solr/appcore"

def process_kafka_message(message: Dict[str, Any]) -> None:
    """Process CDC message and update Solr."""
    op = message.get("op")
    after = message.get("after", {})
    before = message.get("before", {})
    
    if op == "c":
        # Create - index new document
        doc = transform_to_solr_doc(after)
        solr_upsert([doc])
    elif op == "u":
        # Update - reindex document
        doc = transform_to_solr_doc(after)
        solr_upsert([doc])
    elif op == "d":
        # Delete - remove from Solr
        doc_id = before.get("id")
        solr_delete(doc_id)

def transform_to_solr_doc(row: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Postgres row to Solr document."""
    # Implementation similar to indexer
    pass

def solr_upsert(docs: List[Dict[str, Any]]) -> None:
    """Upsert documents to Solr."""
    payload = {"add": [{"doc": doc} for doc in docs]}
    requests.post(f"{SOLR_URL}/update", json=payload)

def solr_delete(doc_id: str) -> None:
    """Delete document from Solr."""
    payload = {"delete": {"id": doc_id}}
    requests.post(f"{SOLR_URL}/update", json=payload)

# Kafka consumer
consumer = KafkaConsumer('app.public.articles', 
                        bootstrap_servers=['kafka:9092'],
                        value_deserializer=lambda m: json.loads(m.decode('utf-8')))

for message in consumer:
    process_kafka_message(message.value)
```

## Operations & Failure Modes

### Idempotency Checklist

- ✅ **Checkpointing**: Use `updated_at` timestamps for incremental indexing
- ✅ **Upsert Strategy**: Solr upserts with `commitWithin` for batching
- ✅ **Retry Logic**: Exponential backoff for failed Solr operations
- ✅ **Schema Evolution**: Dynamic fields handle new JSONB keys gracefully

### Backpressure Management

```python
# Batch size limits
MAX_BATCH_SIZE = 1000
COMMIT_WITHIN_MS = 2000

# Rate limiting
import time
time.sleep(0.1)  # 100ms between batches
```

### Schema Drift Handling

```python
# Dynamic field mapping for new JSONB keys
def map_jsonb_to_solr(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Map JSONB fields to Solr dynamic fields."""
    solr_doc = {}
    
    for key, value in doc.items():
        if isinstance(value, str):
            solr_doc[f"{key}_s"] = value
        elif isinstance(value, list):
            solr_doc[f"{key}_ss"] = value
        elif isinstance(value, int):
            solr_doc[f"{key}_i"] = value
        elif isinstance(value, datetime):
            solr_doc[f"{key}_dt"] = value.isoformat()
    
    return solr_doc
```

### Security Hardening

```nginx
# nginx.conf - Protect Solr admin
server {
    listen 80;
    server_name solr.example.com;
    
    location /solr/admin {
        deny all;
        return 403;
    }
    
    location /solr/appcore/select {
        proxy_pass http://solr:8983;
        proxy_set_header Host $host;
    }
}
```

### Monitoring & Alerting

```python
# monitoring/solr_metrics.py
import requests
import time
from prometheus_client import Gauge, Counter, start_http_server

# Metrics
solr_docs_total = Gauge('solr_documents_total', 'Total documents in Solr')
solr_query_duration = Gauge('solr_query_duration_seconds', 'Query duration')
solr_errors_total = Counter('solr_errors_total', 'Total Solr errors')

def collect_solr_metrics():
    """Collect Solr metrics for Prometheus."""
    try:
        r = requests.get("http://solr:8983/solr/appcore/admin/mbeans")
        data = r.json()
        
        # Extract document count
        docs = data['solr-mbeans']['CORE']['appcore']['searcher']['numDocs']
        solr_docs_total.set(docs)
        
    except Exception as e:
        solr_errors_total.inc()
        print(f"Metrics collection failed: {e}")

if __name__ == "__main__":
    start_http_server(8001)
    while True:
        collect_solr_metrics()
        time.sleep(30)
```

### Disaster Recovery

```bash
#!/bin/bash
# disaster_recovery.sh

# 1. Backup Solr core
curl "http://localhost:8983/solr/admin/cores?action=BACKUP&core=appcore&name=backup_$(date +%Y%m%d)"

# 2. Full reindex from Postgres
python seed_and_index.py --full-reindex

# 3. Verify index integrity
curl "http://localhost:8983/solr/appcore/select?q=*:*&rows=0"
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Stand up the stack
docker compose up -d --build

# 2. Check services
curl http://localhost:8000/health
curl http://localhost:8983/solr/admin/cores

# 3. Test search
curl "http://localhost:8000/search?q=night&facets=true"

# 4. Test suggestions
curl "http://localhost:8000/suggest?q=sal"
```

### Essential Patterns

```python
# Complete hybrid search setup
def setup_hybrid_search():
    # 1. Postgres JSONB source
    # 2. Solr schema with dynamic fields
    # 3. Python indexer (incremental)
    # 4. FastAPI hybrid endpoint
    # 5. Solr features (facets, suggest, geo)
    # 6. Optional CDC pipeline
    # 7. Monitoring & ops
    
    print("Hybrid search setup complete!")
```

### Key Commands

```bash
# Core operations
docker compose up -d --build                    # Start core stack
docker compose --profile cdc up -d              # Add CDC
docker compose --profile viz up -d              # Add monitoring
docker compose --profile geo up -d              # Add geo demo

# Search examples
curl "http://localhost:8000/search?q=market&org=the-hollow-paper"
curl "http://localhost:8000/search?near=40.7128,-74.0060,10&facets=true"
curl "http://localhost:8000/suggest?q=underground"

# Solr admin
curl "http://localhost:8983/solr/appcore/select?q=*:*&facet=true&facet.field=org_s"
```

---

*This tutorial provides the complete machinery for hybrid Postgres-Solr search. Each pattern includes implementation examples, search strategies, and real-world usage patterns for enterprise search systems.*
