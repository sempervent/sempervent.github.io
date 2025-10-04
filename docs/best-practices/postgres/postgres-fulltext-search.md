# PostgreSQL Full-Text Search Best Practices

**Objective**: Master senior-level PostgreSQL full-text search patterns for production systems. When you need to implement powerful text search, when you want to optimize search performance, when you need enterprise-grade search strategies—these best practices become your weapon of choice.

## Core Principles

- **Text Processing**: Proper text normalization and tokenization
- **Index Strategy**: GIN indexes for optimal search performance
- **Query Optimization**: Efficient full-text search queries
- **Ranking**: Relevance scoring and result ranking
- **Multilingual**: Support for multiple languages and character sets

## Text Search Configuration

### Custom Text Search Configuration

```sql
-- Create custom text search configuration
CREATE TEXT SEARCH CONFIGURATION english_custom (COPY = english);

-- Add custom dictionary for domain-specific terms
CREATE TEXT SEARCH DICTIONARY domain_terms (
    TEMPLATE = simple,
    STOPWORDS = domain_stopwords
);

-- Add custom stopwords
CREATE TEXT SEARCH DICTIONARY domain_stopwords (
    TEMPLATE = simple,
    STOPWORDS = 'the, a, an, and, or, but, in, on, at, to, for, of, with, by'
);

-- Configure text search with custom dictionary
ALTER TEXT SEARCH CONFIGURATION english_custom
    ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, word, hword, hword_part
    WITH domain_terms, english_stem;

-- Test the configuration
SELECT to_tsvector('english_custom', 'PostgreSQL is a powerful database system');
```

### Multilingual Text Search

```sql
-- Create multilingual text search configuration
CREATE TEXT SEARCH CONFIGURATION multilingual (COPY = simple);

-- Add language-specific mappings
ALTER TEXT SEARCH CONFIGURATION multilingual
    ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, word, hword, hword_part
    WITH unaccent, simple;

-- Create function to detect language and search
CREATE OR REPLACE FUNCTION search_multilingual(
    search_text TEXT,
    content_column TEXT
)
RETURNS TABLE (
    id INTEGER,
    rank REAL,
    snippet TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        ts_rank(t.search_vector, plainto_tsquery('multilingual', search_text)) as rank,
        ts_headline('multilingual', t.content, plainto_tsquery('multilingual', search_text)) as snippet
    FROM documents t
    WHERE t.search_vector @@ plainto_tsquery('multilingual', search_text)
    ORDER BY rank DESC;
END;
$$ LANGUAGE plpgsql;
```

## Full-Text Search Implementation

### Document Table with Search Vector

```sql
-- Create documents table with full-text search
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    author VARCHAR(100),
    category VARCHAR(50),
    tags TEXT[],
    search_vector TSVECTOR,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create GIN index for search vector
CREATE INDEX idx_documents_search_vector ON documents USING GIN (search_vector);

-- Create B-tree index for category filtering
CREATE INDEX idx_documents_category ON documents (category);

-- Create GIN index for tags
CREATE INDEX idx_documents_tags ON documents USING GIN (tags);

-- Function to update search vector
CREATE OR REPLACE FUNCTION update_document_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english_custom', 
        COALESCE(NEW.title, '') || ' ' || 
        COALESCE(NEW.content, '') || ' ' || 
        COALESCE(NEW.author, '') || ' ' || 
        COALESCE(array_to_string(NEW.tags, ' '), '')
    );
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update search vector
CREATE TRIGGER update_documents_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_document_search_vector();

-- Insert sample documents
INSERT INTO documents (title, content, author, category, tags) VALUES
    ('PostgreSQL Performance Tuning', 'Learn how to optimize PostgreSQL queries and improve database performance.', 'John Doe', 'Database', ARRAY['postgresql', 'performance', 'optimization']),
    ('Database Design Principles', 'Essential principles for designing efficient and scalable database schemas.', 'Jane Smith', 'Database', ARRAY['database', 'design', 'schema']),
    ('Full-Text Search Implementation', 'Implementing powerful search functionality using PostgreSQL full-text search.', 'Bob Johnson', 'Search', ARRAY['search', 'fulltext', 'postgresql']),
    ('Advanced SQL Techniques', 'Master advanced SQL patterns and techniques for complex queries.', 'Alice Brown', 'SQL', ARRAY['sql', 'advanced', 'queries']);
```

### Advanced Search Queries

```sql
-- Basic full-text search
SELECT 
    id,
    title,
    author,
    ts_rank(search_vector, plainto_tsquery('english_custom', 'postgresql performance')) as rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'postgresql performance')
ORDER BY rank DESC;

-- Search with highlighting
SELECT 
    id,
    title,
    ts_headline('english_custom', content, plainto_tsquery('english_custom', 'database design'), 
                'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=10') as highlighted_content
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'database design');

-- Search with category filtering
SELECT 
    id,
    title,
    author,
    category,
    ts_rank(search_vector, plainto_tsquery('english_custom', 'sql advanced')) as rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'sql advanced')
AND category = 'SQL'
ORDER BY rank DESC;

-- Search with tag filtering
SELECT 
    id,
    title,
    author,
    tags,
    ts_rank(search_vector, plainto_tsquery('english_custom', 'database')) as rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'database')
AND 'postgresql' = ANY(tags)
ORDER BY rank DESC;
```

## Search Ranking and Scoring

### Custom Ranking Functions

```sql
-- Create custom ranking function
CREATE OR REPLACE FUNCTION custom_rank(
    search_vector TSVECTOR,
    query TSQUERY,
    title_weight REAL DEFAULT 1.0,
    content_weight REAL DEFAULT 0.5,
    author_weight REAL DEFAULT 0.3
)
RETURNS REAL AS $$
DECLARE
    title_rank REAL;
    content_rank REAL;
    author_rank REAL;
BEGIN
    -- Calculate rank for title
    title_rank := ts_rank(search_vector, query) * title_weight;
    
    -- Calculate rank for content
    content_rank := ts_rank(search_vector, query) * content_weight;
    
    -- Calculate rank for author
    author_rank := ts_rank(search_vector, query) * author_weight;
    
    RETURN title_rank + content_rank + author_rank;
END;
$$ LANGUAGE plpgsql;

-- Use custom ranking in search
SELECT 
    id,
    title,
    author,
    custom_rank(search_vector, plainto_tsquery('english_custom', 'postgresql'), 2.0, 1.0, 0.5) as custom_rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'postgresql')
ORDER BY custom_rank DESC;
```

### Advanced Search Features

```sql
-- Search with phrase matching
SELECT 
    id,
    title,
    ts_rank(search_vector, phraseto_tsquery('english_custom', 'database design principles')) as rank
FROM documents
WHERE search_vector @@ phraseto_tsquery('english_custom', 'database design principles')
ORDER BY rank DESC;

-- Search with wildcard matching
SELECT 
    id,
    title,
    ts_rank(search_vector, to_tsquery('english_custom', 'postgresql:*')) as rank
FROM documents
WHERE search_vector @@ to_tsquery('english_custom', 'postgresql:*')
ORDER BY rank DESC;

-- Search with boolean operators
SELECT 
    id,
    title,
    ts_rank(search_vector, to_tsquery('english_custom', 'database & !mysql')) as rank
FROM documents
WHERE search_vector @@ to_tsquery('english_custom', 'database & !mysql')
ORDER BY rank DESC;

-- Search with proximity operators
SELECT 
    id,
    title,
    ts_rank(search_vector, to_tsquery('english_custom', 'postgresql <-> performance')) as rank
FROM documents
WHERE search_vector @@ to_tsquery('english_custom', 'postgresql <-> performance')
ORDER BY rank DESC;
```

## Search Performance Optimization

### Index Optimization

```sql
-- Create partial indexes for better performance
CREATE INDEX idx_documents_search_active ON documents USING GIN (search_vector)
WHERE created_at > CURRENT_DATE - INTERVAL '1 year';

-- Create composite indexes
CREATE INDEX idx_documents_category_search ON documents (category, search_vector);

-- Create expression indexes
CREATE INDEX idx_documents_title_search ON documents USING GIN (to_tsvector('english_custom', title));

-- Analyze index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexname LIKE '%search%'
ORDER BY idx_scan DESC;
```

### Query Performance Analysis

```sql
-- Analyze search query performance
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT 
    id,
    title,
    ts_rank(search_vector, plainto_tsquery('english_custom', 'postgresql performance')) as rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'postgresql performance')
ORDER BY rank DESC
LIMIT 10;

-- Check search vector statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup
FROM pg_stat_user_tables
WHERE tablename = 'documents';
```

## Search Analytics and Monitoring

### Search Analytics Implementation

```sql
-- Create search analytics table
CREATE TABLE search_analytics (
    id SERIAL PRIMARY KEY,
    search_query TEXT NOT NULL,
    user_id INTEGER,
    results_count INTEGER,
    execution_time_ms INTEGER,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to log search analytics
CREATE OR REPLACE FUNCTION log_search_analytics(
    p_search_query TEXT,
    p_user_id INTEGER,
    p_results_count INTEGER,
    p_execution_time_ms INTEGER
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO search_analytics (search_query, user_id, results_count, execution_time_ms)
    VALUES (p_search_query, p_user_id, p_results_count, p_execution_time_ms);
END;
$$ LANGUAGE plpgsql;

-- Create search function with analytics
CREATE OR REPLACE FUNCTION search_documents_with_analytics(
    search_text TEXT,
    user_id INTEGER DEFAULT NULL,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    id INTEGER,
    title VARCHAR(200),
    author VARCHAR(100),
    rank REAL,
    snippet TEXT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    execution_time INTEGER;
    result_count INTEGER;
BEGIN
    start_time := clock_timestamp();
    
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.author,
        ts_rank(d.search_vector, plainto_tsquery('english_custom', search_text)) as rank,
        ts_headline('english_custom', d.content, plainto_tsquery('english_custom', search_text)) as snippet
    FROM documents d
    WHERE d.search_vector @@ plainto_tsquery('english_custom', search_text)
    ORDER BY rank DESC
    LIMIT limit_count;
    
    end_time := clock_timestamp();
    execution_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    GET DIAGNOSTICS result_count = ROW_COUNT;
    
    -- Log analytics
    PERFORM log_search_analytics(search_text, user_id, result_count, execution_time);
END;
$$ LANGUAGE plpgsql;
```

### Search Performance Monitoring

```python
# monitoring/search_monitor.py
import psycopg2
import json
from datetime import datetime, timedelta
import logging

class SearchPerformanceMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_search_statistics(self, days=7):
        """Get search statistics for the last N days."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        DATE(timestamp) as search_date,
                        COUNT(*) as total_searches,
                        AVG(results_count) as avg_results,
                        AVG(execution_time_ms) as avg_execution_time,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM search_analytics
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY DATE(timestamp)
                    ORDER BY search_date DESC
                """, (days,))
                
                statistics = cur.fetchall()
                return statistics
                
        except Exception as e:
            self.logger.error(f"Error getting search statistics: {e}")
            return []
        finally:
            conn.close()
    
    def get_popular_searches(self, limit=10):
        """Get most popular search queries."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        search_query,
                        COUNT(*) as search_count,
                        AVG(results_count) as avg_results,
                        AVG(execution_time_ms) as avg_execution_time
                    FROM search_analytics
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY search_query
                    ORDER BY search_count DESC
                    LIMIT %s
                """, (limit,))
                
                popular_searches = cur.fetchall()
                return popular_searches
                
        except Exception as e:
            self.logger.error(f"Error getting popular searches: {e}")
            return []
        finally:
            conn.close()
    
    def get_search_performance_metrics(self):
        """Get search performance metrics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                # Get average execution time
                cur.execute("""
                    SELECT 
                        AVG(execution_time_ms) as avg_execution_time,
                        MAX(execution_time_ms) as max_execution_time,
                        MIN(execution_time_ms) as min_execution_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time
                    FROM search_analytics
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '1 day'
                """)
                
                performance_metrics = cur.fetchone()
                
                # Get search success rate
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        COUNT(*) FILTER (WHERE results_count > 0) as successful_searches,
                        COUNT(*) FILTER (WHERE results_count = 0) as failed_searches
                    FROM search_analytics
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '1 day'
                """)
                
                success_metrics = cur.fetchone()
                
                return {
                    'performance_metrics': performance_metrics,
                    'success_metrics': success_metrics
                }
                
        except Exception as e:
            self.logger.error(f"Error getting search performance metrics: {e}")
            return {}
        finally:
            conn.close()
    
    def generate_search_report(self):
        """Generate comprehensive search report."""
        statistics = self.get_search_statistics(7)
        popular_searches = self.get_popular_searches(10)
        performance_metrics = self.get_search_performance_metrics()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'statistics': statistics,
            'popular_searches': popular_searches,
            'performance_metrics': performance_metrics
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = SearchPerformanceMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_search_report()
    print(json.dumps(report, indent=2))
```

## Search Suggestions and Autocomplete

### Search Suggestions Implementation

```sql
-- Create search suggestions table
CREATE TABLE search_suggestions (
    id SERIAL PRIMARY KEY,
    suggestion TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_searched TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create index for suggestions
CREATE INDEX idx_search_suggestions_text ON search_suggestions (suggestion);
CREATE INDEX idx_search_suggestions_frequency ON search_suggestions (frequency DESC);

-- Function to get search suggestions
CREATE OR REPLACE FUNCTION get_search_suggestions(
    partial_query TEXT,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    suggestion TEXT,
    frequency INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.suggestion,
        s.frequency
    FROM search_suggestions s
    WHERE s.suggestion ILIKE partial_query || '%'
    ORDER BY s.frequency DESC, s.suggestion
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update search suggestions
CREATE OR REPLACE FUNCTION update_search_suggestions(
    search_query TEXT
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO search_suggestions (suggestion, frequency)
    VALUES (search_query, 1)
    ON CONFLICT (suggestion) DO UPDATE SET
        frequency = search_suggestions.frequency + 1,
        last_searched = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
```

### Autocomplete Implementation

```python
# search/autocomplete.py
import psycopg2
import json
from typing import List, Dict, Any

class SearchAutocomplete:
    def __init__(self, connection_params):
        self.conn_params = connection_params
    
    def get_suggestions(self, partial_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get search suggestions for autocomplete."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        suggestion,
                        frequency,
                        last_searched
                    FROM search_suggestions
                    WHERE suggestion ILIKE %s
                    ORDER BY frequency DESC, suggestion
                    LIMIT %s
                """, (f"{partial_query}%", limit))
                
                suggestions = []
                for row in cur.fetchall():
                    suggestions.append({
                        'suggestion': row[0],
                        'frequency': row[1],
                        'last_searched': row[2].isoformat() if row[2] else None
                    })
                
                return suggestions
                
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []
        finally:
            conn.close()
    
    def update_suggestion(self, search_query: str):
        """Update search suggestion frequency."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO search_suggestions (suggestion, frequency)
                    VALUES (%s, 1)
                    ON CONFLICT (suggestion) DO UPDATE SET
                        frequency = search_suggestions.frequency + 1,
                        last_searched = CURRENT_TIMESTAMP
                """, (search_query,))
                conn.commit()
                
        except Exception as e:
            print(f"Error updating suggestion: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_popular_suggestions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular search suggestions."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        suggestion,
                        frequency,
                        last_searched
                    FROM search_suggestions
                    ORDER BY frequency DESC, last_searched DESC
                    LIMIT %s
                """, (limit,))
                
                suggestions = []
                for row in cur.fetchall():
                    suggestions.append({
                        'suggestion': row[0],
                        'frequency': row[1],
                        'last_searched': row[2].isoformat() if row[2] else None
                    })
                
                return suggestions
                
        except Exception as e:
            print(f"Error getting popular suggestions: {e}")
            return []
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    autocomplete = SearchAutocomplete({
        'host': 'localhost',
        'database': 'production',
        'user': 'search_user',
        'password': 'search_password'
    })
    
    # Get suggestions for partial query
    suggestions = autocomplete.get_suggestions("postgresql")
    print(json.dumps(suggestions, indent=2))
    
    # Update suggestion
    autocomplete.update_suggestion("postgresql performance")
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create text search configuration
CREATE TEXT SEARCH CONFIGURATION english_custom (COPY = english);

-- 2. Create documents table with search vector
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    search_vector TSVECTOR
);

-- 3. Create GIN index
CREATE INDEX idx_documents_search ON documents USING GIN (search_vector);

-- 4. Insert and search
INSERT INTO documents (title, content) VALUES ('PostgreSQL Guide', 'Learn PostgreSQL');
UPDATE documents SET search_vector = to_tsvector('english_custom', title || ' ' || content);

SELECT title, ts_rank(search_vector, plainto_tsquery('english_custom', 'postgresql')) as rank
FROM documents
WHERE search_vector @@ plainto_tsquery('english_custom', 'postgresql')
ORDER BY rank DESC;
```

### Essential Patterns

```python
# Complete PostgreSQL full-text search setup
def setup_postgresql_fulltext_search():
    # 1. Text search configuration
    # 2. Search vector implementation
    # 3. Indexing strategies
    # 4. Query optimization
    # 5. Ranking and scoring
    # 6. Search analytics
    # 7. Autocomplete and suggestions
    # 8. Performance monitoring
    
    print("PostgreSQL full-text search setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL full-text search excellence. Each pattern includes implementation examples, optimization strategies, and real-world usage patterns for enterprise PostgreSQL search systems.*
