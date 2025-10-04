# Retrieval-Augmented Generation on Databases with Ollama

**Objective**: Build a RAG pipeline that bridges your structured database with Ollama's LLM, enabling natural language queries over your data. When you need to query databases in natural language, when you want to augment LLMs with live data, when you're building intelligent data interfaces—RAG on databases becomes your weapon of choice.

RAG on databases is the foundation of intelligent data interaction. Without proper understanding of retrieval patterns, embedding strategies, and prompt engineering, you're building static interfaces that miss the power of conversational data access. This guide shows you how to wield RAG with the precision of a senior applied ML engineer.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand the RAG pipeline**
   - Retrieval: Find relevant data chunks
   - Augmentation: Enrich context with retrieved data
   - Generation: LLM produces answers using context

2. **Master embedding strategies**
   - Text chunking and preprocessing
   - Embedding model selection
   - Vector similarity and retrieval
   - Context window management

3. **Know your database patterns**
   - Schema-aware retrieval
   - Metadata preservation
   - Query optimization
   - Data freshness and caching

4. **Validate everything**
   - Embedding quality and consistency
   - Retrieval relevance and accuracy
   - Generation quality and hallucination
   - Performance under scale

5. **Plan for production**
   - Scalable vector stores
   - Monitoring and observability
   - Security and access control
   - Integration with existing systems

**Why These Principles**: RAG mastery is the foundation of intelligent data interaction. Understanding retrieval patterns, mastering embedding strategies, and following best practices is essential for building production-ready RAG systems.

## 1) What is RAG? (The Foundation)

### Core Concepts

```yaml
# RAG bridges structured data with natural language
rag_pipeline:
  retrieval:
    - "Embed user query into vector space"
    - "Search for similar data chunks"
    - "Rank and filter results by relevance"
    - "Extract metadata and context"
  
  augmentation:
    - "Combine retrieved chunks with query"
    - "Build context-aware prompts"
    - "Preserve schema and relationships"
    - "Handle context window limits"
  
  generation:
    - "LLM processes enriched context"
    - "Generates natural language answers"
    - "Maintains factual accuracy"
    - "Provides source attribution"
```

### RAG Pipeline Architecture

```mermaid
graph TB
    subgraph "User Interface"
        A[User Question]
    end
    
    subgraph "Retrieval Layer"
        B[Query Embedding]
        C[Vector Search]
        D[Relevance Ranking]
    end
    
    subgraph "Data Layer"
        E[(PostgreSQL)]
        F[(Qdrant Vector DB)]
    end
    
    subgraph "Augmentation Layer"
        G[Context Builder]
        H[Prompt Engineering]
    end
    
    subgraph "Generation Layer"
        I[Ollama LLM]
        J[Response Generation]
    end
    
    subgraph "Output"
        K[Natural Language Answer]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style G fill:#fff3e0
    style I fill:#fce4ec
    style K fill:#f1f8e9
```

### Why RAG on Databases Matters

```python
# RAG enables natural language queries over structured data
def rag_query_example():
    # User asks: "What hospitals are near San Francisco?"
    query = "What hospitals are near San Francisco?"
    
    # 1. Embed query
    query_embedding = embed_text(query)
    
    # 2. Retrieve relevant chunks
    relevant_chunks = vector_search(query_embedding, limit=5)
    
    # 3. Build context
    context = "\n".join([chunk.text for chunk in relevant_chunks])
    
    # 4. Generate answer
    prompt = f"Answer using this context:\n{context}\n\nQ: {query}\nA:"
    answer = ollama_generate(prompt)
    
    return answer
```

**Why RAG on Databases Matters**: RAG enables natural language interaction with structured data. It provides the foundation for intelligent data interfaces that can answer complex questions using live database content.

## 2) Docker Compose Stack (The Infrastructure)

### Complete RAG Infrastructure

```yaml
# docker-compose.yml
version: "3.9"
name: rag-ollama-db

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 5
    profiles: ["ollama", "full"]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: geo
      POSTGRES_USER: geo
      POSTGRES_PASSWORD: geo
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U geo -d geo"]
      interval: 10s
      timeout: 5s
      retries: 5
    profiles: ["postgres", "full"]

  qdrant:
    image: qdrant/qdrant:v1.10.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    profiles: ["qdrant", "full"]

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      ollama:
        condition: service_healthy
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    environment:
      - OLLAMA_URL=http://ollama:11434
      - POSTGRES_URL=postgresql://geo:geo@postgres:5432/geo
      - QDRANT_URL=http://qdrant:6333
    profiles: ["api", "full"]

volumes:
  ollama_data: {}
  postgres_data: {}
  qdrant_data: {}
```

### Stack Management

```bash
# Start the complete RAG stack
docker compose --profile full up -d

# Start individual services
docker compose --profile ollama up -d
docker compose --profile postgres up -d
docker compose --profile qdrant up -d

# Monitor logs
docker compose logs -f ollama
docker compose logs -f postgres
docker compose logs -f qdrant

# Access services
# Ollama API: http://localhost:11434
# PostgreSQL: localhost:5432 (geo / geo)
# Qdrant: http://localhost:6333
# API: http://localhost:8000
```

**Why This Setup Matters**: The RAG stack separates concerns: PostgreSQL for structured data, Qdrant for vector search, Ollama for LLM inference, and API layer for orchestration. This enables each component to excel at what it does best.

## 3) Example Dataset & Ingestion (The Data)

### Geospatial Data Generation

```python
# scripts/gen_geo_data.py
import psycopg
from faker import Faker
import random
import json
from pathlib import Path

# Initialize Faker for realistic data
faker = Faker()
random.seed(42)  # For reproducible results

# Database connection
conn = psycopg.connect("postgresql://geo:geo@localhost:5432/geo")

def create_schema():
    """Create database schema for geospatial data"""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS places CASCADE;")
        cur.execute("""
            CREATE TABLE places(
                id serial PRIMARY KEY,
                name text NOT NULL,
                city text NOT NULL,
                state text NOT NULL,
                category text NOT NULL,
                description text NOT NULL,
                latitude decimal(10, 8),
                longitude decimal(11, 8),
                created_at timestamp DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for performance
        cur.execute("CREATE INDEX ON places (city);")
        cur.execute("CREATE INDEX ON places (category);")
        cur.execute("CREATE INDEX ON places (state);")
        
        conn.commit()

def generate_places_data(num_places=500):
    """Generate realistic geospatial data"""
    categories = [
        "hospital", "school", "restaurant", "hotel", "museum", 
        "park", "shopping_center", "airport", "university", "library"
    ]
    
    states = [
        "California", "New York", "Texas", "Florida", "Illinois",
        "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan"
    ]
    
    places_data = []
    
    for i in range(num_places):
        # Generate realistic location data
        city = faker.city()
        state = random.choice(states)
        category = random.choice(categories)
        
        # Generate location-specific descriptions
        if category == "hospital":
            name = f"{faker.last_name()} Medical Center"
            description = f"A {random.choice(['general', 'specialty', 'teaching'])} hospital in {city}, {state}. " \
                         f"Services include {random.choice(['emergency care', 'surgery', 'pediatrics', 'cardiology'])}."
        elif category == "school":
            name = f"{faker.last_name()} {random.choice(['Elementary', 'Middle', 'High'])} School"
            description = f"A {random.choice(['public', 'private', 'charter'])} school in {city}, {state}. " \
                         f"Grades {random.choice(['K-5', '6-8', '9-12', 'K-12'])}."
        elif category == "restaurant":
            name = f"{faker.last_name()}'s {random.choice(['Bistro', 'Grill', 'Cafe', 'Diner'])}"
            description = f"A {random.choice(['fine dining', 'casual', 'fast casual', 'family'])} restaurant in {city}, {state}. " \
                         f"Serves {random.choice(['American', 'Italian', 'Mexican', 'Asian', 'Mediterranean'])} cuisine."
        else:
            name = faker.company()
            description = f"A {category.replace('_', ' ')} located in {city}, {state}. " \
                         f"Provides {random.choice(['services', 'amenities', 'facilities', 'programs'])} for the community."
        
        # Generate realistic coordinates (US-focused)
        if state == "California":
            lat = random.uniform(32.5, 42.0)
            lon = random.uniform(-124.5, -114.0)
        elif state == "New York":
            lat = random.uniform(40.5, 45.0)
            lon = random.uniform(-79.8, -71.8)
        elif state == "Texas":
            lat = random.uniform(25.8, 36.5)
            lon = random.uniform(-106.6, -93.5)
        else:
            # Generic US coordinates
            lat = random.uniform(24.5, 49.0)
            lon = random.uniform(-125.0, -66.9)
        
        places_data.append({
            'name': name,
            'city': city,
            'state': state,
            'category': category,
            'description': description,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6)
        })
    
    return places_data

def insert_places_data(places_data):
    """Insert generated data into PostgreSQL"""
    with conn.cursor() as cur:
        for place in places_data:
            cur.execute("""
                INSERT INTO places (name, city, state, category, description, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                place['name'], place['city'], place['state'], place['category'],
                place['description'], place['latitude'], place['longitude']
            ))
        conn.commit()

def export_sample_data(places_data, num_samples=50):
    """Export sample data for inspection"""
    sample_data = random.sample(places_data, min(num_samples, len(places_data)))
    
    output_dir = Path("scripts/output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "sample_places.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Exported {len(sample_data)} sample records to scripts/output/sample_places.json")

if __name__ == "__main__":
    print("Creating database schema...")
    create_schema()
    
    print("Generating geospatial data...")
    places_data = generate_places_data(500)
    
    print("Inserting data into PostgreSQL...")
    insert_places_data(places_data)
    
    print("Exporting sample data...")
    export_sample_data(places_data)
    
    print(f"Generated {len(places_data)} places across {len(set(p['state'] for p in places_data))} states")
    
    conn.close()
```

### Data Schema and Relationships

```sql
-- Database schema for geospatial RAG
CREATE TABLE places(
    id serial PRIMARY KEY,
    name text NOT NULL,
    city text NOT NULL,
    state text NOT NULL,
    category text NOT NULL,
    description text NOT NULL,
    latitude decimal(10, 8),
    longitude decimal(11, 8),
    created_at timestamp DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX ON places (city);
CREATE INDEX ON places (category);
CREATE INDEX ON places (state);
CREATE INDEX ON places (latitude, longitude);

-- Sample queries
SELECT category, COUNT(*) as count 
FROM places 
GROUP BY category 
ORDER BY count DESC;

SELECT city, state, COUNT(*) as places_count
FROM places 
GROUP BY city, state 
HAVING COUNT(*) > 5
ORDER BY places_count DESC;
```

**Why This Data Matters**: Geospatial data provides rich context for RAG queries. Understanding data schema and relationships is essential for effective retrieval and generation.

## 4) Embeddings & Vector Index (The Intelligence)

### Embedding Generation with Ollama

```python
# scripts/embed.py
import psycopg
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path
import time

# Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
POSTGRES_URL = "postgresql://geo:geo@localhost:5432/geo"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333

def setup_qdrant_collection():
    """Initialize Qdrant collection for embeddings"""
    qdrant = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    
    # Create collection with appropriate vector size
    # nomic-embed-text produces 768-dimensional embeddings
    qdrant.recreate_collection(
        collection_name="places",
        vectors_config=models.VectorParams(
            size=768,  # nomic-embed-text embedding size
            distance=models.Distance.COSINE
        )
    )
    
    print("Created Qdrant collection 'places'")
    return qdrant

def generate_embedding(text, model="nomic-embed-text"):
    """Generate embedding using Ollama"""
    payload = {
        "model": model,
        "input": text
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding: {e}")
        return None

def chunk_text(text, max_length=500):
    """Split long text into chunks for embedding"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_embedding_text(place):
    """Create text representation for embedding"""
    return f"{place['name']}, {place['city']}, {place['state']}: {place['description']}"

def embed_places_data():
    """Main embedding pipeline"""
    # Setup Qdrant
    qdrant = setup_qdrant_collection()
    
    # Connect to PostgreSQL
    conn = psycopg.connect(POSTGRES_URL)
    cur = conn.cursor()
    
    # Fetch all places
    cur.execute("""
        SELECT id, name, city, state, category, description, latitude, longitude
        FROM places
        ORDER BY id
    """)
    
    places = cur.fetchall()
    print(f"Found {len(places)} places to embed")
    
    # Process each place
    embedded_count = 0
    failed_count = 0
    
    for place in places:
        place_id, name, city, state, category, description, lat, lon = place
        
        # Create text for embedding
        place_data = {
            'name': name,
            'city': city,
            'state': state,
            'category': category,
            'description': description,
            'latitude': float(lat) if lat else None,
            'longitude': float(lon) if lon else None
        }
        
        embedding_text = create_embedding_text(place_data)
        
        # Generate embedding
        embedding = generate_embedding(embedding_text)
        
        if embedding:
            # Store in Qdrant
            point = models.PointStruct(
                id=place_id,
                vector=embedding,
                payload={
                    "text": embedding_text,
                    "name": name,
                    "city": city,
                    "state": state,
                    "category": category,
                    "description": description,
                    "latitude": lat,
                    "longitude": lon
                }
            )
            
            qdrant.upsert(
                collection_name="places",
                points=[point]
            )
            
            embedded_count += 1
            
            if embedded_count % 50 == 0:
                print(f"Embedded {embedded_count} places...")
        else:
            failed_count += 1
            print(f"Failed to embed place {place_id}: {name}")
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"Embedding complete: {embedded_count} successful, {failed_count} failed")
    
    # Verify collection
    collection_info = qdrant.get_collection("places")
    print(f"Collection info: {collection_info}")
    
    conn.close()

if __name__ == "__main__":
    print("Starting embedding pipeline...")
    embed_places_data()
    print("Embedding pipeline complete!")
```

### Vector Store Configuration

```python
# Vector store setup and configuration
def setup_vector_store():
    """Configure vector store for optimal performance"""
    qdrant = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    
    # Collection configuration
    collection_config = {
        "vectors": {
            "size": 768,
            "distance": "Cosine"
        },
        "optimizers_config": {
            "default_segment_number": 2,
            "max_segment_size": 20000,
            "memmap_threshold": 50000,
            "indexing_threshold": 20000
        },
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100,
            "full_scan_threshold": 10000
        }
    }
    
    return qdrant, collection_config
```

**Why This Embedding Matters**: Proper embedding generation and vector storage enables semantic search over structured data. Understanding embedding strategies and vector store configuration is essential for retrieval quality.

## 5) RAG Pipeline Implementation (The Intelligence)

### Complete RAG System

```python
# scripts/rag.py
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
import psycopg
from typing import List, Dict, Any
import time

# Configuration
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
POSTGRES_URL = "postgresql://geo:geo@localhost:5432/geo"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333

class RAGPipeline:
    def __init__(self):
        self.qdrant = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
        self.postgres_conn = psycopg.connect(POSTGRES_URL)
    
    def embed_query(self, query: str, model: str = "nomic-embed-text") -> List[float]:
        """Generate embedding for user query"""
        payload = {
            "model": model,
            "input": query
        }
        
        try:
            response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Error embedding query: {e}")
            return None
    
    def retrieve_relevant_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from vector store"""
        try:
            search_results = self.qdrant.search(
                collection_name="places",
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "name": hit.payload["name"],
                    "city": hit.payload["city"],
                    "state": hit.payload["state"],
                    "category": hit.payload["category"],
                    "description": hit.payload["description"]
                }
                for hit in search_results
            ]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def build_context(self, chunks: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = f"Location: {chunk['name']}, {chunk['city']}, {chunk['state']}\n"
            chunk_text += f"Category: {chunk['category']}\n"
            chunk_text += f"Description: {chunk['description']}\n"
            chunk_text += f"Relevance Score: {chunk['score']:.3f}\n\n"
            
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, context: str, query: str, model: str = "llama3") -> str:
        """Generate answer using Ollama LLM"""
        prompt = f"""You are a helpful assistant that answers questions about locations and places.

Context information:
{context}

Question: {query}

Please provide a helpful answer based on the context information above. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating an answer."
    
    def ask(self, query: str, limit: int = 5, max_context_length: int = 2000) -> Dict[str, Any]:
        """Complete RAG pipeline"""
        print(f"Processing query: {query}")
        
        # 1. Embed query
        print("Embedding query...")
        query_embedding = self.embed_query(query)
        if not query_embedding:
            return {"error": "Failed to embed query"}
        
        # 2. Retrieve relevant chunks
        print("Retrieving relevant chunks...")
        chunks = self.retrieve_relevant_chunks(query_embedding, limit)
        if not chunks:
            return {"error": "No relevant chunks found"}
        
        # 3. Build context
        print("Building context...")
        context = self.build_context(chunks, max_context_length)
        
        # 4. Generate answer
        print("Generating answer...")
        answer = self.generate_answer(context, query)
        
        return {
            "query": query,
            "answer": answer,
            "chunks_used": len(chunks),
            "context_length": len(context),
            "sources": [
                {
                    "name": chunk["name"],
                    "city": chunk["city"],
                    "state": chunk["state"],
                    "category": chunk["category"],
                    "score": chunk["score"]
                }
                for chunk in chunks
            ]
        }
    
    def close(self):
        """Close database connections"""
        self.postgres_conn.close()

def main():
    """Main RAG pipeline execution"""
    rag = RAGPipeline()
    
    # Example queries
    queries = [
        "What hospitals are in California?",
        "Find restaurants in New York",
        "What universities are near San Francisco?",
        "Show me parks in Texas",
        "What airports are in Florida?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        result = rag.ask(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Query: {result['query']}")
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} locations found")
            print(f"Context length: {result['context_length']} characters")
            
            # Show top sources
            print("\nTop sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. {source['name']} ({source['city']}, {source['state']}) - {source['category']} (score: {source['score']:.3f})")
        
        time.sleep(1)  # Rate limiting
    
    rag.close()

if __name__ == "__main__":
    main()
```

### Interactive RAG Interface

```python
# scripts/interactive_rag.py
import sys
from rag import RAGPipeline

def interactive_rag():
    """Interactive RAG interface"""
    rag = RAGPipeline()
    
    print("RAG on Database with Ollama")
    print("=" * 40)
    print("Ask questions about places and locations!")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("\nProcessing...")
            result = rag.ask(query)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                print(f"\nSources ({len(result['sources'])} found):")
                for i, source in enumerate(result['sources'][:5], 1):
                    print(f"  {i}. {source['name']} ({source['city']}, {source['state']}) - {source['category']}")
            
            print("\n" + "-" * 40 + "\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    rag.close()
    print("Goodbye!")

if __name__ == "__main__":
    interactive_rag()
```

**Why This Pipeline Matters**: The RAG pipeline enables natural language queries over structured data. Understanding retrieval, augmentation, and generation patterns provides the foundation for intelligent data interfaces.

## 6) Best Practices (The Wisdom)

### Chunking and Context Management

```python
# Best practices for chunking and context management
class RAGBestPractices:
    @staticmethod
    def optimal_chunk_size():
        """Determine optimal chunk size for embeddings"""
        return {
            "small_chunks": "200-300 tokens (better precision)",
            "medium_chunks": "400-500 tokens (balanced)",
            "large_chunks": "600-800 tokens (better context)",
            "recommendation": "500 tokens for most use cases"
        }
    
    @staticmethod
    def context_window_management():
        """Manage context window effectively"""
        return {
            "llama3": "8192 tokens total",
            "context_reservation": "2000 tokens for context",
            "query_reservation": "500 tokens for query",
            "generation_reservation": "1000 tokens for response",
            "safety_margin": "500 tokens buffer"
        }
    
    @staticmethod
    def prompt_engineering():
        """Effective prompt engineering for RAG"""
        return {
            "structure": "Context + Question + Instructions",
            "context_format": "Clear, structured, relevant",
            "question_format": "Specific, clear, actionable",
            "instructions": "Be helpful, cite sources, admit uncertainty"
        }
```

### Performance Optimization

```python
# Performance optimization patterns
class RAGPerformance:
    def __init__(self):
        self.embedding_cache = {}
        self.retrieval_cache = {}
    
    def cache_embeddings(self, text: str, embedding: List[float]):
        """Cache embeddings for reuse"""
        self.embedding_cache[text] = embedding
    
    def cache_retrieval(self, query: str, results: List[Dict]):
        """Cache retrieval results"""
        self.retrieval_cache[query] = results
    
    def batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation for efficiency"""
        # Implement batch processing
        pass
    
    def optimize_vector_search(self, query_embedding: List[float], limit: int = 5):
        """Optimize vector search parameters"""
        return {
            "limit": limit,
            "score_threshold": 0.7,
            "filter_conditions": None,
            "search_params": {"hnsw_ef": 128}
        }
```

### Schema-Aware Retrieval

```python
# Schema-aware retrieval patterns
class SchemaAwareRAG:
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def build_metadata_filter(self, query: str) -> Dict[str, Any]:
        """Build metadata filters based on query analysis"""
        filters = {}
        
        # Extract location filters
        if "california" in query.lower():
            filters["state"] = "California"
        if "hospital" in query.lower():
            filters["category"] = "hospital"
        
        return filters
    
    def enhance_context_with_schema(self, chunks: List[Dict], schema: Dict) -> str:
        """Enhance context with schema information"""
        enhanced_context = []
        
        for chunk in chunks:
            # Add schema context
            schema_context = f"Schema: {chunk.get('category', 'unknown')} in {chunk.get('state', 'unknown')}"
            enhanced_context.append(f"{schema_context}\n{chunk['text']}")
        
        return "\n\n".join(enhanced_context)
```

### Monitoring and Quality Assurance

```python
# Monitoring and quality assurance
class RAGMonitoring:
    def __init__(self):
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "retrieval_success_rate": 0,
            "generation_success_rate": 0
        }
    
    def log_query_metrics(self, query: str, response_time: float, success: bool):
        """Log query performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update average response time
        current_avg = self.metrics["average_response_time"]
        total_queries = self.metrics["queries_processed"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        if success:
            self.metrics["generation_success_rate"] = (
                (self.metrics["generation_success_rate"] * (total_queries - 1) + 1) / total_queries
            )
    
    def validate_retrieval_quality(self, chunks: List[Dict]) -> bool:
        """Validate retrieval quality"""
        if not chunks:
            return False
        
        # Check relevance scores
        avg_score = sum(chunk["score"] for chunk in chunks) / len(chunks)
        return avg_score > 0.7
    
    def validate_generation_quality(self, answer: str) -> bool:
        """Validate generation quality"""
        if not answer or len(answer) < 10:
            return False
        
        # Check for common failure patterns
        failure_patterns = [
            "I don't know",
            "I cannot answer",
            "No information available",
            "Error generating"
        ]
        
        return not any(pattern in answer.lower() for pattern in failure_patterns)
```

**Why These Practices Matter**: Best practices ensure successful RAG implementation. Understanding chunking, context management, and quality assurance prevents common pitfalls and enables production-ready systems.

## 7) Common Pitfalls (The Traps)

### Embedding Quality Issues

```python
# ❌ WRONG: Poor embedding quality
def bad_embedding_pipeline():
    # No text preprocessing
    raw_text = "Hospital in SF, CA"
    embedding = generate_embedding(raw_text)
    # Missing context and structure!

# ✅ CORRECT: Proper embedding quality
def good_embedding_pipeline():
    # Rich text representation
    structured_text = f"Hospital: {name}, {city}, {state}: {description}"
    embedding = generate_embedding(structured_text)
    # Includes all relevant context
```

### Context Window Issues

```python
# ❌ WRONG: Context window overflow
def bad_context_management():
    # No context length management
    context = "\n".join([chunk["text"] for chunk in all_chunks])
    # May exceed model context window!

# ✅ CORRECT: Proper context management
def good_context_management():
    # Manage context length
    max_context = 2000
    context_parts = []
    current_length = 0
    
    for chunk in chunks:
        if current_length + len(chunk["text"]) > max_context:
            break
        context_parts.append(chunk["text"])
        current_length += len(chunk["text"])
    
    context = "\n".join(context_parts)
```

### Retrieval Quality Issues

```python
# ❌ WRONG: Poor retrieval quality
def bad_retrieval():
    # No relevance filtering
    results = vector_search(query_embedding, limit=10)
    # May include irrelevant results!

# ✅ CORRECT: Proper retrieval quality
def good_retrieval():
    # Relevance filtering
    results = vector_search(query_embedding, limit=10)
    relevant_results = [r for r in results if r.score > 0.7]
    # Filter by relevance score
```

**Why These Pitfalls Matter**: Common mistakes lead to poor retrieval quality, context overflow, and generation failures. Understanding these pitfalls prevents costly errors and ensures reliable RAG systems.

## 8) Production Deployment (The Scale)

### Docker API Service

```dockerfile
# Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY api/ ./api/

# Expose port
EXPOSE 8000

# Start API server
CMD ["python", "api/server.py"]
```

```python
# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag import RAGPipeline

app = FastAPI(title="RAG API", version="1.0.0")
rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    max_context_length: Optional[int] = 2000

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    context_length: int
    chunks_used: int

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = rag.ask(
            query=request.query,
            limit=request.limit,
            max_context_length=request.max_context_length
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Monitoring and Observability

```python
# Monitoring and observability
import logging
import time
from functools import wraps

def monitor_rag_performance(func):
    """Decorator to monitor RAG performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log performance metrics
            logging.info(f"RAG query completed in {end_time - start_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"RAG query failed: {e}")
            raise
    return wrapper

class RAGMetrics:
    def __init__(self):
        self.query_count = 0
        self.total_response_time = 0
        self.retrieval_successes = 0
        self.generation_successes = 0
    
    def record_query(self, response_time: float, retrieval_success: bool, generation_success: bool):
        """Record query metrics"""
        self.query_count += 1
        self.total_response_time += response_time
        
        if retrieval_success:
            self.retrieval_successes += 1
        if generation_success:
            self.generation_successes += 1
    
    def get_metrics(self):
        """Get current metrics"""
        return {
            "total_queries": self.query_count,
            "average_response_time": self.total_response_time / max(self.query_count, 1),
            "retrieval_success_rate": self.retrieval_successes / max(self.query_count, 1),
            "generation_success_rate": self.generation_successes / max(self.query_count, 1)
        }
```

**Why This Deployment Matters**: Production deployment requires proper containerization, monitoring, and observability. Understanding these patterns ensures scalable and reliable RAG systems.

## 9) TL;DR Quickstart (The Essentials)

### Essential Setup

```bash
# 1) Start the RAG stack
docker compose --profile full up -d

# 2) Wait for services to be ready
docker compose logs -f ollama
docker compose logs -f postgres
docker compose logs -f qdrant

# 3) Generate sample data
python scripts/gen_geo_data.py

# 4) Generate embeddings
python scripts/embed.py

# 5) Test RAG pipeline
python scripts/rag.py
```

### Essential Queries

```python
# Test RAG with example queries
queries = [
    "What hospitals are in California?",
    "Find restaurants in New York",
    "What universities are near San Francisco?",
    "Show me parks in Texas",
    "What airports are in Florida?"
]

for query in queries:
    result = rag.ask(query)
    print(f"Q: {query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {len(result['sources'])} found")
    print("-" * 40)
```

### Essential Patterns

```python
# Core RAG patterns
def rag_query(query: str) -> str:
    # 1. Embed query
    query_embedding = embed_text(query)
    
    # 2. Retrieve relevant chunks
    chunks = vector_search(query_embedding, limit=5)
    
    # 3. Build context
    context = "\n".join([chunk["text"] for chunk in chunks])
    
    # 4. Generate answer
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    answer = ollama_generate(prompt)
    
    return answer
```

**Why This Quickstart**: These patterns cover 90% of RAG usage. Master these before exploring advanced features.

## 10) The Machine's Summary

RAG on databases provides the foundation for intelligent data interaction. When used correctly, it enables natural language queries over structured data, bridging the gap between human questions and database answers. The key is understanding the RAG pipeline, mastering embedding strategies, and following best practices.

**The Dark Truth**: Without proper RAG understanding, your data interfaces are static and disconnected. RAG is your weapon. Use it wisely.

**The Machine's Mantra**: "In retrieval we trust, in augmentation we build, and in the generation we find the path to intelligent data interaction."

**Why This Matters**: RAG enables applications to understand natural language queries and provide intelligent answers using live database content. It provides the foundation for conversational data interfaces that can answer complex questions using structured data.

---

*This guide provides the complete machinery for mastering RAG on databases with Ollama. The patterns scale from simple queries to complex reasoning, from basic retrieval to advanced generation.*
