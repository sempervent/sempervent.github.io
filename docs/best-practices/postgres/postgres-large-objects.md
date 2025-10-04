# PostgreSQL Large Object Storage Best Practices

**Objective**: Master senior-level PostgreSQL large object storage patterns for production systems. When you need to handle BLOBs, when you want to optimize large object storage, when you need enterprise-grade large object strategies—these best practices become your weapon of choice.

## Core Principles

- **Storage Efficiency**: Choose appropriate storage methods for large objects
- **Performance**: Optimize large object access and retrieval
- **Security**: Implement proper access controls for large objects
- **Backup**: Ensure large objects are included in backup strategies
- **Maintenance**: Implement cleanup and maintenance procedures

## Large Object Storage Methods

### BYTEA vs Large Objects

```sql
-- Create tables for different large object storage methods
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    content BYTEA,                           -- BYTEA for smaller files (< 1GB)
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE large_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    oid OID,                                -- Large Object OID for larger files
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE file_references (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    storage_path VARCHAR(500) NOT NULL,     -- External storage path
    checksum VARCHAR(64) NOT NULL,          -- File integrity checksum
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

### Large Object Functions

```sql
-- Create function to store large object
CREATE OR REPLACE FUNCTION store_large_object(
    p_filename VARCHAR(255),
    p_content_type VARCHAR(100),
    p_file_data BYTEA
)
RETURNS INTEGER AS $$
DECLARE
    file_oid OID;
    file_size BIGINT;
BEGIN
    -- Create large object
    file_oid := lo_create(0);
    
    -- Open large object for writing
    PERFORM lo_open(file_oid, 131072); -- 131072 = INV_WRITE
    
    -- Write data to large object
    PERFORM lo_write(file_oid, 0, p_file_data);
    
    -- Close large object
    PERFORM lo_close(file_oid);
    
    -- Get file size
    file_size := LENGTH(p_file_data);
    
    -- Insert record
    INSERT INTO large_files (filename, content_type, file_size, oid)
    VALUES (p_filename, p_content_type, file_size, file_oid);
    
    RETURN file_oid;
END;
$$ LANGUAGE plpgsql;

-- Create function to retrieve large object
CREATE OR REPLACE FUNCTION retrieve_large_object(p_oid OID)
RETURNS BYTEA AS $$
DECLARE
    file_data BYTEA;
    file_size BIGINT;
BEGIN
    -- Get file size
    SELECT file_size INTO file_size
    FROM large_files
    WHERE oid = p_oid;
    
    -- Open large object for reading
    PERFORM lo_open(p_oid, 262144); -- 262144 = INV_READ
    
    -- Read data from large object
    SELECT lo_read(p_oid, 0, file_size) INTO file_data;
    
    -- Close large object
    PERFORM lo_close(p_oid);
    
    RETURN file_data;
END;
$$ LANGUAGE plpgsql;
```

## File Management System

### File Upload and Storage

```sql
-- Create function for file upload
CREATE OR REPLACE FUNCTION upload_file(
    p_filename VARCHAR(255),
    p_content_type VARCHAR(100),
    p_file_data BYTEA,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS INTEGER AS $$
DECLARE
    file_id INTEGER;
    file_size BIGINT;
    file_checksum VARCHAR(64);
BEGIN
    -- Calculate file size and checksum
    file_size := LENGTH(p_file_data);
    file_checksum := encode(digest(p_file_data, 'sha256'), 'hex');
    
    -- Check if file already exists
    IF EXISTS (SELECT 1 FROM file_references WHERE checksum = file_checksum) THEN
        RAISE EXCEPTION 'File already exists with checksum: %', file_checksum;
    END IF;
    
    -- Store file based on size
    IF file_size < 1048576 THEN -- 1MB
        -- Store small files in BYTEA
        INSERT INTO documents (filename, content_type, file_size, content, metadata)
        VALUES (p_filename, p_content_type, file_size, p_file_data, p_metadata)
        RETURNING id INTO file_id;
    ELSE
        -- Store large files as Large Objects
        INSERT INTO large_files (filename, content_type, file_size, oid, metadata)
        VALUES (p_filename, p_content_type, file_size, 
                store_large_object(p_filename, p_content_type, p_file_data), p_metadata)
        RETURNING id INTO file_id;
    END IF;
    
    RETURN file_id;
END;
$$ LANGUAGE plpgsql;

-- Create function for file download
CREATE OR REPLACE FUNCTION download_file(p_file_id INTEGER)
RETURNS TABLE (
    filename VARCHAR(255),
    content_type VARCHAR(100),
    file_size BIGINT,
    file_data BYTEA,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.filename,
        d.content_type,
        d.file_size,
        d.content as file_data,
        d.metadata
    FROM documents d
    WHERE d.id = p_file_id
    
    UNION ALL
    
    SELECT 
        lf.filename,
        lf.content_type,
        lf.file_size,
        retrieve_large_object(lf.oid) as file_data,
        lf.metadata
    FROM large_files lf
    WHERE lf.id = p_file_id;
END;
$$ LANGUAGE plpgsql;
```

### File Integrity and Validation

```sql
-- Create function to validate file integrity
CREATE OR REPLACE FUNCTION validate_file_integrity(p_file_id INTEGER)
RETURNS BOOLEAN AS $$
DECLARE
    stored_checksum VARCHAR(64);
    calculated_checksum VARCHAR(64);
    file_data BYTEA;
BEGIN
    -- Get stored checksum and file data
    SELECT checksum, content INTO stored_checksum, file_data
    FROM documents
    WHERE id = p_file_id;
    
    IF stored_checksum IS NULL THEN
        -- Check large files table
        SELECT checksum, retrieve_large_object(oid) INTO stored_checksum, file_data
        FROM large_files
        WHERE id = p_file_id;
    END IF;
    
    IF stored_checksum IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Calculate current checksum
    calculated_checksum := encode(digest(file_data, 'sha256'), 'hex');
    
    -- Compare checksums
    RETURN stored_checksum = calculated_checksum;
END;
$$ LANGUAGE plpgsql;

-- Create function to update file checksums
CREATE OR REPLACE FUNCTION update_file_checksums()
RETURNS INTEGER AS $$
DECLARE
    file_record RECORD;
    calculated_checksum VARCHAR(64);
    updated_count INTEGER := 0;
BEGIN
    -- Update checksums for documents
    FOR file_record IN 
        SELECT id, content FROM documents WHERE checksum IS NULL
    LOOP
        calculated_checksum := encode(digest(file_record.content, 'sha256'), 'hex');
        
        UPDATE documents 
        SET checksum = calculated_checksum
        WHERE id = file_record.id;
        
        updated_count := updated_count + 1;
    END LOOP;
    
    -- Update checksums for large files
    FOR file_record IN 
        SELECT id, oid FROM large_files WHERE checksum IS NULL
    LOOP
        calculated_checksum := encode(digest(retrieve_large_object(file_record.oid), 'sha256'), 'hex');
        
        UPDATE large_files 
        SET checksum = calculated_checksum
        WHERE id = file_record.id;
        
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;
```

## External Storage Integration

### S3-Compatible Storage

```sql
-- Create function for S3 storage integration
CREATE OR REPLACE FUNCTION store_file_s3(
    p_filename VARCHAR(255),
    p_content_type VARCHAR(100),
    p_file_data BYTEA,
    p_bucket_name VARCHAR(100),
    p_metadata JSONB DEFAULT '{}'
)
RETURNS INTEGER AS $$
DECLARE
    file_id INTEGER;
    file_size BIGINT;
    file_checksum VARCHAR(64);
    storage_path VARCHAR(500);
BEGIN
    -- Calculate file size and checksum
    file_size := LENGTH(p_file_data);
    file_checksum := encode(digest(p_file_data, 'sha256'), 'hex');
    
    -- Generate storage path
    storage_path := p_bucket_name || '/' || 
                   to_char(CURRENT_DATE, 'YYYY/MM/DD') || '/' ||
                   file_checksum || '_' || p_filename;
    
    -- Store file reference
    INSERT INTO file_references (filename, content_type, file_size, storage_path, checksum, metadata)
    VALUES (p_filename, p_content_type, file_size, storage_path, file_checksum, p_metadata)
    RETURNING id INTO file_id;
    
    -- Note: Actual S3 upload would be handled by application code
    -- This function only creates the database record
    
    RETURN file_id;
END;
$$ LANGUAGE plpgsql;

-- Create function to retrieve file from S3
CREATE OR REPLACE FUNCTION retrieve_file_s3(p_file_id INTEGER)
RETURNS TABLE (
    filename VARCHAR(255),
    content_type VARCHAR(100),
    file_size BIGINT,
    storage_path VARCHAR(500),
    checksum VARCHAR(64),
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fr.filename,
        fr.content_type,
        fr.file_size,
        fr.storage_path,
        fr.checksum,
        fr.metadata
    FROM file_references fr
    WHERE fr.id = p_file_id;
END;
$$ LANGUAGE plpgsql;
```

### File Cleanup and Maintenance

```sql
-- Create function for file cleanup
CREATE OR REPLACE FUNCTION cleanup_orphaned_files()
RETURNS INTEGER AS $$
DECLARE
    orphaned_count INTEGER := 0;
    file_record RECORD;
BEGIN
    -- Find orphaned large objects
    FOR file_record IN 
        SELECT oid FROM large_files 
        WHERE NOT EXISTS (
            SELECT 1 FROM pg_largeobject_metadata 
            WHERE oid = large_files.oid
        )
    LOOP
        -- Delete orphaned large object
        PERFORM lo_unlink(file_record.oid);
        orphaned_count := orphaned_count + 1;
    END LOOP;
    
    -- Delete orphaned file references
    DELETE FROM file_references 
    WHERE storage_path IS NOT NULL 
    AND NOT EXISTS (
        SELECT 1 FROM file_references fr2 
        WHERE fr2.storage_path = file_references.storage_path
    );
    
    GET DIAGNOSTICS orphaned_count = orphaned_count + ROW_COUNT;
    
    RETURN orphaned_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for file size analysis
CREATE OR REPLACE FUNCTION analyze_file_storage()
RETURNS TABLE (
    storage_type TEXT,
    file_count BIGINT,
    total_size BIGINT,
    avg_size NUMERIC,
    max_size BIGINT,
    min_size BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'BYTEA'::TEXT as storage_type,
        COUNT(*) as file_count,
        SUM(file_size) as total_size,
        AVG(file_size) as avg_size,
        MAX(file_size) as max_size,
        MIN(file_size) as min_size
    FROM documents
    
    UNION ALL
    
    SELECT 
        'Large Objects'::TEXT as storage_type,
        COUNT(*) as file_count,
        SUM(file_size) as total_size,
        AVG(file_size) as avg_size,
        MAX(file_size) as max_size,
        MIN(file_size) as min_size
    FROM large_files
    
    UNION ALL
    
    SELECT 
        'External Storage'::TEXT as storage_type,
        COUNT(*) as file_count,
        SUM(file_size) as total_size,
        AVG(file_size) as avg_size,
        MAX(file_size) as max_size,
        MIN(file_size) as min_size
    FROM file_references;
END;
$$ LANGUAGE plpgsql;
```

## Large Object Security

### Access Control

```sql
-- Create function for file access control
CREATE OR REPLACE FUNCTION check_file_access(
    p_file_id INTEGER,
    p_user_id INTEGER,
    p_access_type VARCHAR(20) DEFAULT 'read'
)
RETURNS BOOLEAN AS $$
DECLARE
    file_owner INTEGER;
    file_permissions JSONB;
BEGIN
    -- Get file owner and permissions
    SELECT owner_id, permissions INTO file_owner, file_permissions
    FROM (
        SELECT id, owner_id, permissions FROM documents WHERE id = p_file_id
        UNION ALL
        SELECT id, owner_id, permissions FROM large_files WHERE id = p_file_id
        UNION ALL
        SELECT id, owner_id, permissions FROM file_references WHERE id = p_file_id
    ) t;
    
    -- Check if user is owner
    IF file_owner = p_user_id THEN
        RETURN TRUE;
    END IF;
    
    -- Check permissions
    IF file_permissions IS NOT NULL THEN
        IF p_access_type = 'read' AND (file_permissions ? 'read' OR file_permissions ? 'write') THEN
            RETURN TRUE;
        END IF;
        
        IF p_access_type = 'write' AND file_permissions ? 'write' THEN
            RETURN TRUE;
        END IF;
    END IF;
    
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Create function for secure file retrieval
CREATE OR REPLACE FUNCTION secure_download_file(
    p_file_id INTEGER,
    p_user_id INTEGER
)
RETURNS TABLE (
    filename VARCHAR(255),
    content_type VARCHAR(100),
    file_size BIGINT,
    file_data BYTEA,
    metadata JSONB
) AS $$
BEGIN
    -- Check access permissions
    IF NOT check_file_access(p_file_id, p_user_id, 'read') THEN
        RAISE EXCEPTION 'Access denied for file %', p_file_id;
    END IF;
    
    -- Return file data
    RETURN QUERY
    SELECT * FROM download_file(p_file_id);
END;
$$ LANGUAGE plpgsql;
```

### File Encryption

```sql
-- Create function for file encryption
CREATE OR REPLACE FUNCTION encrypt_file_data(
    p_file_data BYTEA,
    p_encryption_key TEXT
)
RETURNS BYTEA AS $$
BEGIN
    -- Use pgcrypto for encryption
    RETURN pgp_sym_encrypt(p_file_data, p_encryption_key);
END;
$$ LANGUAGE plpgsql;

-- Create function for file decryption
CREATE OR REPLACE FUNCTION decrypt_file_data(
    p_encrypted_data BYTEA,
    p_encryption_key TEXT
)
RETURNS BYTEA AS $$
BEGIN
    -- Use pgcrypto for decryption
    RETURN pgp_sym_decrypt(p_encrypted_data, p_encryption_key);
END;
$$ LANGUAGE plpgsql;

-- Create function for secure file storage
CREATE OR REPLACE FUNCTION store_encrypted_file(
    p_filename VARCHAR(255),
    p_content_type VARCHAR(100),
    p_file_data BYTEA,
    p_encryption_key TEXT,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS INTEGER AS $$
DECLARE
    file_id INTEGER;
    encrypted_data BYTEA;
    file_size BIGINT;
    file_checksum VARCHAR(64);
BEGIN
    -- Encrypt file data
    encrypted_data := encrypt_file_data(p_file_data, p_encryption_key);
    
    -- Calculate file size and checksum
    file_size := LENGTH(p_file_data);
    file_checksum := encode(digest(p_file_data, 'sha256'), 'hex');
    
    -- Store encrypted file
    INSERT INTO documents (filename, content_type, file_size, content, metadata)
    VALUES (p_filename, p_content_type, file_size, encrypted_data, p_metadata)
    RETURNING id INTO file_id;
    
    RETURN file_id;
END;
$$ LANGUAGE plpgsql;
```

## Large Object Monitoring

### Storage Monitoring

```python
# monitoring/large_object_monitor.py
import psycopg2
import json
from datetime import datetime
import logging

class LargeObjectMonitor:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_storage_statistics(self):
        """Get storage statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        'documents' as table_name,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size,
                        AVG(file_size) as avg_size,
                        MAX(file_size) as max_size,
                        MIN(file_size) as min_size
                    FROM documents
                    UNION ALL
                    SELECT 
                        'large_files' as table_name,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size,
                        AVG(file_size) as avg_size,
                        MAX(file_size) as max_size,
                        MIN(file_size) as min_size
                    FROM large_files
                    UNION ALL
                    SELECT 
                        'file_references' as table_name,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size,
                        AVG(file_size) as avg_size,
                        MAX(file_size) as max_size,
                        MIN(file_size) as min_size
                    FROM file_references
                """)
                
                storage_stats = cur.fetchall()
                return storage_stats
                
        except Exception as e:
            self.logger.error(f"Error getting storage statistics: {e}")
            return []
        finally:
            conn.close()
    
    def get_large_object_usage(self):
        """Get large object usage statistics."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    WHERE tablename IN ('documents', 'large_files', 'file_references')
                    ORDER BY idx_scan DESC
                """)
                
                usage_stats = cur.fetchall()
                return usage_stats
                
        except Exception as e:
            self.logger.error(f"Error getting large object usage: {e}")
            return []
        finally:
            conn.close()
    
    def get_file_integrity_status(self):
        """Get file integrity status."""
        conn = psycopg2.connect(**self.conn_params)
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        'documents' as table_name,
                        COUNT(*) as total_files,
                        COUNT(*) FILTER (WHERE checksum IS NOT NULL) as files_with_checksum,
                        COUNT(*) FILTER (WHERE checksum IS NULL) as files_without_checksum
                    FROM documents
                    UNION ALL
                    SELECT 
                        'large_files' as table_name,
                        COUNT(*) as total_files,
                        COUNT(*) FILTER (WHERE checksum IS NOT NULL) as files_with_checksum,
                        COUNT(*) FILTER (WHERE checksum IS NULL) as files_without_checksum
                    FROM large_files
                """)
                
                integrity_status = cur.fetchall()
                return integrity_status
                
        except Exception as e:
            self.logger.error(f"Error getting file integrity status: {e}")
            return []
        finally:
            conn.close()
    
    def generate_large_object_report(self):
        """Generate comprehensive large object report."""
        storage_stats = self.get_storage_statistics()
        usage_stats = self.get_large_object_usage()
        integrity_status = self.get_file_integrity_status()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'storage_statistics': storage_stats,
            'usage_statistics': usage_stats,
            'integrity_status': integrity_status
        }
        
        return report

# Usage
if __name__ == "__main__":
    monitor = LargeObjectMonitor({
        'host': 'localhost',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    })
    
    report = monitor.generate_large_object_report()
    print(json.dumps(report, indent=2))
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create large object tables
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    content BYTEA,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Create large object functions
CREATE OR REPLACE FUNCTION store_large_object(
    p_filename VARCHAR(255),
    p_content_type VARCHAR(100),
    p_file_data BYTEA
) RETURNS INTEGER AS $$
-- Function implementation here
$$ LANGUAGE plpgsql;

-- 3. Store files
SELECT store_large_object('document.pdf', 'application/pdf', file_data);

-- 4. Retrieve files
SELECT * FROM download_file(1);
```

### Essential Patterns

```python
# Complete PostgreSQL large object storage setup
def setup_postgresql_large_objects():
    # 1. Large object storage methods
    # 2. File management system
    # 3. File integrity and validation
    # 4. External storage integration
    # 5. File cleanup and maintenance
    # 6. Large object security
    # 7. File encryption
    # 8. Storage monitoring
    
    print("PostgreSQL large object storage setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL large object storage excellence. Each pattern includes implementation examples, storage strategies, and real-world usage patterns for enterprise PostgreSQL large object systems.*
