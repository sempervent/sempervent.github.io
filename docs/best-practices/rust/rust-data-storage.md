# Rust Data Storage Best Practices

**Objective**: Master senior-level Rust data storage patterns for production systems. When you need to implement robust data persistence, when you want to optimize storage performance, when you need enterprise-grade data storage strategies—these best practices become your weapon of choice.

## Core Principles

- **Data Persistence**: Reliable data storage and retrieval
- **Performance Optimization**: Efficient storage operations
- **Data Integrity**: Ensure data consistency and reliability
- **Backup and Recovery**: Implement robust backup strategies
- **Scalability**: Design for horizontal and vertical scaling

## Data Storage Patterns

### File System Storage

```rust
// rust/01-file-system-storage.rs

/*
File system storage patterns and best practices
*/

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// File system storage manager.
pub struct FileSystemStorage {
    base_path: PathBuf,
    file_locks: Arc<RwLock<HashMap<String, Arc<RwLock<()>>>>>,
    max_file_size: u64,
    compression_enabled: bool,
    encryption_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub size: u64,
    pub created_at: Instant,
    pub modified_at: Instant,
    pub checksum: String,
    pub is_compressed: bool,
    pub is_encrypted: bool,
}

impl FileSystemStorage {
    pub fn new(base_path: PathBuf, max_file_size: u64) -> Self {
        Self {
            base_path,
            file_locks: Arc::new(RwLock::new(HashMap::new())),
            max_file_size,
            compression_enabled: false,
            encryption_enabled: false,
        }
    }
    
    /// Enable compression.
    pub fn enable_compression(&mut self) {
        self.compression_enabled = true;
    }
    
    /// Enable encryption.
    pub fn enable_encryption(&mut self) {
        self.encryption_enabled = true;
    }
    
    /// Write data to a file.
    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<FileMetadata, String> {
        let file_path = self.base_path.join(path);
        
        // Ensure directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        
        // Get file lock
        let lock = self.get_file_lock(path).await;
        let _guard = lock.write().await;
        
        // Check file size
        if data.len() as u64 > self.max_file_size {
            return Err("File size exceeds maximum allowed size".to_string());
        }
        
        // Process data
        let processed_data = self.process_data(data).await?;
        
        // Write to file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| format!("Failed to open file for writing: {}", e))?;
        
        file.write_all(&processed_data)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        
        file.sync_all()
            .map_err(|e| format!("Failed to sync file: {}", e))?;
        
        // Get file metadata
        let metadata = self.get_file_metadata(&file_path).await?;
        
        Ok(metadata)
    }
    
    /// Read data from a file.
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>, String> {
        let file_path = self.base_path.join(path);
        
        // Get file lock
        let lock = self.get_file_lock(path).await;
        let _guard = lock.read().await;
        
        // Read file
        let mut file = File::open(&file_path)
            .map_err(|e| format!("Failed to open file for reading: {}", e))?;
        
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        // Process data
        let processed_data = self.process_data_for_reading(&data).await?;
        
        Ok(processed_data)
    }
    
    /// Delete a file.
    pub async fn delete_file(&self, path: &str) -> Result<(), String> {
        let file_path = self.base_path.join(path);
        
        // Get file lock
        let lock = self.get_file_lock(path).await;
        let _guard = lock.write().await;
        
        std::fs::remove_file(&file_path)
            .map_err(|e| format!("Failed to delete file: {}", e))?;
        
        Ok(())
    }
    
    /// List files in a directory.
    pub async fn list_files(&self, path: &str) -> Result<Vec<FileMetadata>, String> {
        let dir_path = self.base_path.join(path);
        
        let mut files = Vec::new();
        
        let entries = std::fs::read_dir(&dir_path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.is_file() {
                let metadata = self.get_file_metadata(&path).await?;
                files.push(metadata);
            }
        }
        
        Ok(files)
    }
    
    /// Get file metadata.
    async fn get_file_metadata(&self, path: &Path) -> Result<FileMetadata, String> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| format!("Failed to get file metadata: {}", e))?;
        
        let size = metadata.len();
        let created_at = metadata.created()
            .map_err(|e| format!("Failed to get creation time: {}", e))?
            .into();
        let modified_at = metadata.modified()
            .map_err(|e| format!("Failed to get modification time: {}", e))?
            .into();
        
        // Calculate checksum
        let checksum = self.calculate_checksum(path).await?;
        
        Ok(FileMetadata {
            path: path.to_path_buf(),
            size,
            created_at,
            modified_at,
            checksum,
            is_compressed: self.compression_enabled,
            is_encrypted: self.encryption_enabled,
        })
    }
    
    /// Calculate file checksum.
    async fn calculate_checksum(&self, path: &Path) -> Result<String, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut file = File::open(path)
            .map_err(|e| format!("Failed to open file for checksum: {}", e))?;
        
        let mut hasher = DefaultHasher::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| format!("Failed to read file for checksum: {}", e))?;
            
            if bytes_read == 0 {
                break;
            }
            
            buffer[..bytes_read].hash(&mut hasher);
        }
        
        Ok(format!("{:x}", hasher.finish()))
    }
    
    /// Process data for writing.
    async fn process_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut processed = data.to_vec();
        
        // Apply compression if enabled
        if self.compression_enabled {
            processed = self.compress_data(&processed).await?;
        }
        
        // Apply encryption if enabled
        if self.encryption_enabled {
            processed = self.encrypt_data(&processed).await?;
        }
        
        Ok(processed)
    }
    
    /// Process data for reading.
    async fn process_data_for_reading(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut processed = data.to_vec();
        
        // Apply decryption if enabled
        if self.encryption_enabled {
            processed = self.decrypt_data(&processed).await?;
        }
        
        // Apply decompression if enabled
        if self.compression_enabled {
            processed = self.decompress_data(&processed).await?;
        }
        
        Ok(processed)
    }
    
    /// Compress data.
    async fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, you would use a compression library
        // like flate2, lz4, or zstd
        println!("Compressing {} bytes", data.len());
        
        // Simulate compression
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // For this example, just return the data as-is
        Ok(data.to_vec())
    }
    
    /// Decompress data.
    async fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, you would use a decompression library
        println!("Decompressing {} bytes", data.len());
        
        // Simulate decompression
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // For this example, just return the data as-is
        Ok(data.to_vec())
    }
    
    /// Encrypt data.
    async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, you would use an encryption library
        // like aes-gcm, chacha20-poly1305, or similar
        println!("Encrypting {} bytes", data.len());
        
        // Simulate encryption
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        // For this example, just return the data as-is
        Ok(data.to_vec())
    }
    
    /// Decrypt data.
    async fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, you would use a decryption library
        println!("Decrypting {} bytes", data.len());
        
        // Simulate decryption
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        // For this example, just return the data as-is
        Ok(data.to_vec())
    }
    
    /// Get file lock for concurrent access.
    async fn get_file_lock(&self, path: &str) -> Arc<RwLock<()>> {
        let mut locks = self.file_locks.write().await;
        
        if let Some(lock) = locks.get(path) {
            Arc::clone(lock)
        } else {
            let lock = Arc::new(RwLock::new(()));
            locks.insert(path.to_string(), Arc::clone(&lock));
            lock
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_file_system_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileSystemStorage::new(temp_dir.path().to_path_buf(), 1024 * 1024);
        
        let data = b"Hello, World!";
        let metadata = storage.write_file("test.txt", data).await.unwrap();
        
        assert_eq!(metadata.size, data.len() as u64);
        assert_eq!(metadata.path.file_name().unwrap(), "test.txt");
        
        let read_data = storage.read_file("test.txt").await.unwrap();
        assert_eq!(read_data, data);
    }
    
    #[tokio::test]
    async fn test_file_compression() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = FileSystemStorage::new(temp_dir.path().to_path_buf(), 1024 * 1024);
        storage.enable_compression();
        
        let data = b"Hello, World!";
        let metadata = storage.write_file("test.txt", data).await.unwrap();
        
        assert!(metadata.is_compressed);
    }
    
    #[tokio::test]
    async fn test_file_encryption() {
        let temp_dir = TempDir::new().unwrap();
        let mut storage = FileSystemStorage::new(temp_dir.path().to_path_buf(), 1024 * 1024);
        storage.enable_encryption();
        
        let data = b"Hello, World!";
        let metadata = storage.write_file("test.txt", data).await.unwrap();
        
        assert!(metadata.is_encrypted);
    }
}
```

### Object Storage

```rust
// rust/02-object-storage.rs

/*
Object storage patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Object storage client.
pub struct ObjectStorageClient {
    endpoint: String,
    access_key: String,
    secret_key: String,
    bucket: String,
    region: String,
    client: Arc<RwLock<HashMap<String, Object>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    pub key: String,
    pub size: u64,
    pub content_type: String,
    pub etag: String,
    pub last_modified: Instant,
    pub metadata: HashMap<String, String>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    pub key: String,
    pub size: u64,
    pub content_type: String,
    pub etag: String,
    pub last_modified: Instant,
    pub metadata: HashMap<String, String>,
}

impl ObjectStorageClient {
    pub fn new(endpoint: String, access_key: String, secret_key: String, bucket: String, region: String) -> Self {
        Self {
            endpoint,
            access_key,
            secret_key,
            bucket,
            region,
            client: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Upload an object.
    pub async fn upload_object(&self, key: &str, data: &[u8], content_type: &str) -> Result<ObjectMetadata, String> {
        // In a real implementation, you would use an S3-compatible client
        // like rusoto_s3 or aws-sdk-s3
        println!("Uploading object {} to bucket {}", key, self.bucket);
        
        // Simulate upload
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let etag = self.calculate_etag(data);
        let metadata = ObjectMetadata {
            key: key.to_string(),
            size: data.len() as u64,
            content_type: content_type.to_string(),
            etag,
            last_modified: Instant::now(),
            metadata: HashMap::new(),
        };
        
        // Store object in client
        let mut objects = self.client.write().await;
        objects.insert(key.to_string(), Object {
            key: key.to_string(),
            size: data.len() as u64,
            content_type: content_type.to_string(),
            etag: etag.clone(),
            last_modified: Instant::now(),
            metadata: HashMap::new(),
            data: data.to_vec(),
        });
        
        Ok(metadata)
    }
    
    /// Download an object.
    pub async fn download_object(&self, key: &str) -> Result<Vec<u8>, String> {
        // In a real implementation, you would download from S3-compatible storage
        println!("Downloading object {} from bucket {}", key, self.bucket);
        
        // Simulate download
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        let objects = self.client.read().await;
        if let Some(object) = objects.get(key) {
            Ok(object.data.clone())
        } else {
            Err("Object not found".to_string())
        }
    }
    
    /// Delete an object.
    pub async fn delete_object(&self, key: &str) -> Result<(), String> {
        // In a real implementation, you would delete from S3-compatible storage
        println!("Deleting object {} from bucket {}", key, self.bucket);
        
        // Simulate deletion
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        let mut objects = self.client.write().await;
        objects.remove(key);
        
        Ok(())
    }
    
    /// List objects in a bucket.
    pub async fn list_objects(&self, prefix: &str) -> Result<Vec<ObjectMetadata>, String> {
        // In a real implementation, you would list objects from S3-compatible storage
        println!("Listing objects with prefix {} in bucket {}", prefix, self.bucket);
        
        // Simulate listing
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        let objects = self.client.read().await;
        let mut result = Vec::new();
        
        for (key, object) in objects.iter() {
            if key.starts_with(prefix) {
                result.push(ObjectMetadata {
                    key: object.key.clone(),
                    size: object.size,
                    content_type: object.content_type.clone(),
                    etag: object.etag.clone(),
                    last_modified: object.last_modified,
                    metadata: object.metadata.clone(),
                });
            }
        }
        
        Ok(result)
    }
    
    /// Get object metadata.
    pub async fn get_object_metadata(&self, key: &str) -> Result<ObjectMetadata, String> {
        // In a real implementation, you would get metadata from S3-compatible storage
        println!("Getting metadata for object {} in bucket {}", key, self.bucket);
        
        // Simulate metadata retrieval
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let objects = self.client.read().await;
        if let Some(object) = objects.get(key) {
            Ok(ObjectMetadata {
                key: object.key.clone(),
                size: object.size,
                content_type: object.content_type.clone(),
                etag: object.etag.clone(),
                last_modified: object.last_modified,
                metadata: object.metadata.clone(),
            })
        } else {
            Err("Object not found".to_string())
        }
    }
    
    /// Copy an object.
    pub async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<ObjectMetadata, String> {
        // In a real implementation, you would copy objects in S3-compatible storage
        println!("Copying object {} to {} in bucket {}", source_key, dest_key, self.bucket);
        
        // Simulate copy
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        let objects = self.client.read().await;
        if let Some(source_object) = objects.get(source_key) {
            let mut objects = self.client.write().await;
            let dest_object = Object {
                key: dest_key.to_string(),
                size: source_object.size,
                content_type: source_object.content_type.clone(),
                etag: self.calculate_etag(&source_object.data),
                last_modified: Instant::now(),
                metadata: source_object.metadata.clone(),
                data: source_object.data.clone(),
            };
            
            objects.insert(dest_key.to_string(), dest_object);
            
            Ok(ObjectMetadata {
                key: dest_key.to_string(),
                size: source_object.size,
                content_type: source_object.content_type.clone(),
                etag: self.calculate_etag(&source_object.data),
                last_modified: Instant::now(),
                metadata: source_object.metadata.clone(),
            })
        } else {
            Err("Source object not found".to_string())
        }
    }
    
    /// Calculate ETag for data.
    fn calculate_etag(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("\"{:x}\"", hasher.finish())
    }
}

/// Object storage with versioning.
pub struct VersionedObjectStorage {
    client: ObjectStorageClient,
    versions: Arc<RwLock<HashMap<String, Vec<ObjectVersion>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectVersion {
    pub version_id: String,
    pub key: String,
    pub size: u64,
    pub content_type: String,
    pub etag: String,
    pub last_modified: Instant,
    pub is_latest: bool,
    pub is_delete_marker: bool,
}

impl VersionedObjectStorage {
    pub fn new(client: ObjectStorageClient) -> Self {
        Self {
            client,
            versions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Upload an object with versioning.
    pub async fn upload_object(&self, key: &str, data: &[u8], content_type: &str) -> Result<ObjectVersion, String> {
        let metadata = self.client.upload_object(key, data, content_type).await?;
        
        let version_id = format!("v{}", Instant::now().elapsed().as_nanos());
        let version = ObjectVersion {
            version_id: version_id.clone(),
            key: key.to_string(),
            size: metadata.size,
            content_type: metadata.content_type,
            etag: metadata.etag,
            last_modified: metadata.last_modified,
            is_latest: true,
            is_delete_marker: false,
        };
        
        // Update versions
        let mut versions = self.versions.write().await;
        let key_versions = versions.entry(key.to_string()).or_insert_with(Vec::new);
        
        // Mark previous versions as not latest
        for v in key_versions.iter_mut() {
            v.is_latest = false;
        }
        
        key_versions.push(version.clone());
        
        Ok(version)
    }
    
    /// Delete an object with versioning.
    pub async fn delete_object(&self, key: &str) -> Result<ObjectVersion, String> {
        let version_id = format!("v{}", Instant::now().elapsed().as_nanos());
        let version = ObjectVersion {
            version_id: version_id.clone(),
            key: key.to_string(),
            size: 0,
            content_type: String::new(),
            etag: String::new(),
            last_modified: Instant::now(),
            is_latest: true,
            is_delete_marker: true,
        };
        
        // Update versions
        let mut versions = self.versions.write().await;
        let key_versions = versions.entry(key.to_string()).or_insert_with(Vec::new);
        
        // Mark previous versions as not latest
        for v in key_versions.iter_mut() {
            v.is_latest = false;
        }
        
        key_versions.push(version.clone());
        
        Ok(version)
    }
    
    /// List object versions.
    pub async fn list_object_versions(&self, key: &str) -> Result<Vec<ObjectVersion>, String> {
        let versions = self.versions.read().await;
        if let Some(key_versions) = versions.get(key) {
            Ok(key_versions.clone())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get a specific version of an object.
    pub async fn get_object_version(&self, key: &str, version_id: &str) -> Result<Vec<u8>, String> {
        let versions = self.versions.read().await;
        if let Some(key_versions) = versions.get(key) {
            if let Some(version) = key_versions.iter().find(|v| v.version_id == version_id) {
                if version.is_delete_marker {
                    return Err("Object version is a delete marker".to_string());
                }
                
                return self.client.download_object(key).await;
            }
        }
        
        Err("Object version not found".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_object_storage() {
        let client = ObjectStorageClient::new(
            "https://s3.amazonaws.com".to_string(),
            "access_key".to_string(),
            "secret_key".to_string(),
            "test-bucket".to_string(),
            "us-east-1".to_string(),
        );
        
        let data = b"Hello, World!";
        let metadata = client.upload_object("test.txt", data, "text/plain").await.unwrap();
        
        assert_eq!(metadata.key, "test.txt");
        assert_eq!(metadata.size, data.len() as u64);
        
        let downloaded = client.download_object("test.txt").await.unwrap();
        assert_eq!(downloaded, data);
    }
    
    #[tokio::test]
    async fn test_versioned_object_storage() {
        let client = ObjectStorageClient::new(
            "https://s3.amazonaws.com".to_string(),
            "access_key".to_string(),
            "secret_key".to_string(),
            "test-bucket".to_string(),
            "us-east-1".to_string(),
        );
        
        let versioned_storage = VersionedObjectStorage::new(client);
        
        let data = b"Hello, World!";
        let version = versioned_storage.upload_object("test.txt", data, "text/plain").await.unwrap();
        
        assert_eq!(version.key, "test.txt");
        assert!(version.is_latest);
        assert!(!version.is_delete_marker);
        
        let versions = versioned_storage.list_object_versions("test.txt").await.unwrap();
        assert_eq!(versions.len(), 1);
    }
}
```

### Database Storage

```rust
// rust/03-database-storage.rs

/*
Database storage patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Database storage manager.
pub struct DatabaseStorage {
    connection_pool: Arc<ConnectionPool>,
    table_schema: HashMap<String, TableSchema>,
    indexes: HashMap<String, Vec<Index>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<Column>,
    pub primary_key: Vec<String>,
    pub foreign_keys: Vec<ForeignKey>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data_type: DataType,
    pub is_nullable: bool,
    pub default_value: Option<String>,
    pub is_unique: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    BigInt,
    Text,
    Varchar(usize),
    Boolean,
    Timestamp,
    Json,
    Blob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKey {
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
    pub on_delete: OnDeleteAction,
    pub on_update: OnUpdateAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnDeleteAction {
    Cascade,
    Restrict,
    SetNull,
    NoAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnUpdateAction {
    Cascade,
    Restrict,
    SetNull,
    NoAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub columns: Vec<String>,
    pub expression: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Check,
    Unique,
    NotNull,
    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub is_unique: bool,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    GIN,
    GiST,
    SPGiST,
}

impl DatabaseStorage {
    pub fn new(connection_pool: Arc<ConnectionPool>) -> Self {
        Self {
            connection_pool,
            table_schema: HashMap::new(),
            indexes: HashMap::new(),
        }
    }
    
    /// Create a table.
    pub async fn create_table(&mut self, schema: TableSchema) -> Result<(), String> {
        // In a real implementation, you would execute CREATE TABLE statement
        println!("Creating table: {}", schema.name);
        
        // Simulate table creation
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        self.table_schema.insert(schema.name.clone(), schema);
        
        Ok(())
    }
    
    /// Insert data into a table.
    pub async fn insert(&self, table: &str, data: HashMap<String, serde_json::Value>) -> Result<u64, String> {
        // In a real implementation, you would execute INSERT statement
        println!("Inserting data into table: {}", table);
        
        // Simulate insert
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(1) // Return number of affected rows
    }
    
    /// Update data in a table.
    pub async fn update(&self, table: &str, data: HashMap<String, serde_json::Value>, where_clause: &str) -> Result<u64, String> {
        // In a real implementation, you would execute UPDATE statement
        println!("Updating data in table: {} where {}", table, where_clause);
        
        // Simulate update
        tokio::time::sleep(Duration::from_millis(15)).await;
        
        Ok(1) // Return number of affected rows
    }
    
    /// Delete data from a table.
    pub async fn delete(&self, table: &str, where_clause: &str) -> Result<u64, String> {
        // In a real implementation, you would execute DELETE statement
        println!("Deleting data from table: {} where {}", table, where_clause);
        
        // Simulate delete
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(1) // Return number of affected rows
    }
    
    /// Select data from a table.
    pub async fn select(&self, table: &str, columns: &[String], where_clause: Option<&str>) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        // In a real implementation, you would execute SELECT statement
        println!("Selecting data from table: {}", table);
        
        // Simulate select
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        // Return mock data
        let mut result = Vec::new();
        let mut row = HashMap::new();
        for column in columns {
            row.insert(column.clone(), serde_json::Value::String("mock_value".to_string()));
        }
        result.push(row);
        
        Ok(result)
    }
    
    /// Create an index.
    pub async fn create_index(&mut self, index: Index) -> Result<(), String> {
        // In a real implementation, you would execute CREATE INDEX statement
        println!("Creating index: {} on table: {}", index.name, index.table);
        
        // Simulate index creation
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        self.indexes.entry(index.table.clone()).or_insert_with(Vec::new).push(index);
        
        Ok(())
    }
    
    /// Drop an index.
    pub async fn drop_index(&mut self, table: &str, index_name: &str) -> Result<(), String> {
        // In a real implementation, you would execute DROP INDEX statement
        println!("Dropping index: {} from table: {}", index_name, table);
        
        // Simulate index drop
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        if let Some(indexes) = self.indexes.get_mut(table) {
            indexes.retain(|index| index.name != index_name);
        }
        
        Ok(())
    }
    
    /// Get table schema.
    pub fn get_table_schema(&self, table: &str) -> Option<&TableSchema> {
        self.table_schema.get(table)
    }
    
    /// Get table indexes.
    pub fn get_table_indexes(&self, table: &str) -> Option<&[Index]> {
        self.indexes.get(table).map(|indexes| indexes.as_slice())
    }
    
    /// Get database statistics.
    pub async fn get_database_statistics(&self) -> DatabaseStatistics {
        let total_tables = self.table_schema.len();
        let total_indexes: usize = self.indexes.values().map(|indexes| indexes.len()).sum();
        
        DatabaseStatistics {
            total_tables,
            total_indexes,
            connection_pool_stats: self.connection_pool.get_statistics().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    pub total_tables: usize,
    pub total_indexes: usize,
    pub connection_pool_stats: PoolStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_database_storage() {
        let connection_pool = Arc::new(ConnectionPool::new(
            10,
            2,
            Duration::from_secs(30),
            Duration::from_secs(300),
            Duration::from_secs(60),
        ));
        
        let mut storage = DatabaseStorage::new(connection_pool);
        
        let schema = TableSchema {
            name: "users".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    is_nullable: false,
                    default_value: None,
                    is_unique: true,
                },
                Column {
                    name: "name".to_string(),
                    data_type: DataType::Varchar(255),
                    is_nullable: false,
                    default_value: None,
                    is_unique: false,
                },
            ],
            primary_key: vec!["id".to_string()],
            foreign_keys: Vec::new(),
            constraints: Vec::new(),
        };
        
        storage.create_table(schema).await.unwrap();
        
        let mut data = HashMap::new();
        data.insert("name".to_string(), serde_json::Value::String("John".to_string()));
        
        let result = storage.insert("users", data).await.unwrap();
        assert_eq!(result, 1);
    }
    
    #[tokio::test]
    async fn test_database_indexes() {
        let connection_pool = Arc::new(ConnectionPool::new(
            10,
            2,
            Duration::from_secs(30),
            Duration::from_secs(300),
            Duration::from_secs(60),
        ));
        
        let mut storage = DatabaseStorage::new(connection_pool);
        
        let index = Index {
            name: "idx_users_name".to_string(),
            table: "users".to_string(),
            columns: vec!["name".to_string()],
            is_unique: false,
            index_type: IndexType::BTree,
        };
        
        storage.create_index(index).await.unwrap();
        
        let indexes = storage.get_table_indexes("users").unwrap();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].name, "idx_users_name");
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. File system storage
let storage = FileSystemStorage::new(PathBuf::from("/data"), 1024 * 1024);
storage.write_file("test.txt", b"Hello, World!").await?;

// 2. Object storage
let client = ObjectStorageClient::new(endpoint, access_key, secret_key, bucket, region);
client.upload_object("test.txt", data, "text/plain").await?;

// 3. Database storage
let mut storage = DatabaseStorage::new(connection_pool);
storage.create_table(schema).await?;
storage.insert("users", data).await?;
```

### Essential Patterns

```rust
// Complete data storage setup
pub fn setup_rust_data_storage() {
    // 1. File system storage
    // 2. Object storage
    // 3. Database storage
    // 4. Data compression
    // 5. Data encryption
    // 6. Backup strategies
    // 7. Performance optimization
    // 8. Data integrity
    
    println!("Rust data storage setup complete!");
}
```

---

*This guide provides the complete machinery for Rust data storage. Each pattern includes implementation examples, storage strategies, and real-world usage patterns for enterprise data storage systems.*
