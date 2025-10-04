# Python Data Storage Best Practices

**Objective**: Master senior-level Python data storage patterns for production systems. When you need to implement robust data persistence, when you want to build scalable storage solutions, when you need enterprise-grade data storage strategies—these best practices become your weapon of choice.

## Core Principles

- **Reliability**: Ensure data durability and consistency
- **Performance**: Optimize for read/write operations
- **Scalability**: Design for horizontal and vertical scaling
- **Security**: Implement data encryption and access controls
- **Backup**: Implement comprehensive backup and recovery strategies

## File System Storage

### Advanced File Operations

```python
# python/01-file-storage.py

"""
File system storage patterns and data persistence strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import os
import shutil
import tempfile
import hashlib
import json
import pickle
import gzip
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles
from contextlib import asynccontextmanager, contextmanager
import mmap
import fcntl
import stat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageType(Enum):
    """Storage type enumeration"""
    LOCAL = "local"
    NETWORK = "network"
    CLOUD = "cloud"
    DISTRIBUTED = "distributed"

class FileOperation(Enum):
    """File operation enumeration"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"

@dataclass
class FileMetadata:
    """File metadata definition"""
    path: str
    size: int
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    checksum: str
    permissions: int
    owner: str
    group: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class FileStorageManager:
    """Advanced file storage manager"""
    
    def __init__(self, base_path: str, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        self.base_path = Path(base_path)
        self.max_file_size = max_file_size
        self.storage_lock = threading.Lock()
        self.file_metadata = {}
        self.initialize_storage()
    
    def initialize_storage(self) -> None:
        """Initialize storage directory"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            (self.base_path / "backup").mkdir(exist_ok=True)
            (self.base_path / "metadata").mkdir(exist_ok=True)
            
            logger.info(f"Storage initialized at {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
    
    def store_file(self, file_path: str, data: Union[str, bytes], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store file with metadata"""
        try:
            # Generate unique filename
            unique_filename = self._generate_unique_filename(file_path)
            full_path = self.base_path / "data" / unique_filename
            
            # Write file
            if isinstance(data, str):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(data)
            else:
                with open(full_path, 'wb') as f:
                    f.write(data)
            
            # Calculate checksum
            checksum = self._calculate_checksum(full_path)
            
            # Create metadata
            file_metadata = FileMetadata(
                path=str(full_path),
                size=full_path.stat().st_size,
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                checksum=checksum,
                permissions=full_path.stat().st_mode,
                owner=full_path.owner(),
                group=full_path.group()
            )
            
            # Store metadata
            self._store_metadata(unique_filename, file_metadata)
            
            logger.info(f"File stored: {unique_filename}")
            return unique_filename
        
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            raise
    
    def retrieve_file(self, filename: str) -> Optional[bytes]:
        """Retrieve file by filename"""
        try:
            file_path = self.base_path / "data" / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {filename}")
                return None
            
            # Verify checksum
            current_checksum = self._calculate_checksum(file_path)
            stored_metadata = self._get_metadata(filename)
            
            if stored_metadata and current_checksum != stored_metadata.checksum:
                logger.error(f"Checksum mismatch for file: {filename}")
                return None
            
            # Update access time
            self._update_access_time(filename)
            
            with open(file_path, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Failed to retrieve file {filename}: {e}")
            return None
    
    def delete_file(self, filename: str) -> bool:
        """Delete file and metadata"""
        try:
            file_path = self.base_path / "data" / filename
            metadata_path = self.base_path / "metadata" / f"{filename}.json"
            
            # Delete file
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from memory
            if filename in self.file_metadata:
                del self.file_metadata[filename]
            
            logger.info(f"File deleted: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {e}")
            return False
    
    def list_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files with optional pattern matching"""
        try:
            files = []
            data_dir = self.base_path / "data"
            
            for file_path in data_dir.glob(pattern):
                if file_path.is_file():
                    metadata = self._get_metadata(file_path.name)
                    if metadata:
                        files.append(metadata.to_dict())
            
            return files
        
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def backup_file(self, filename: str) -> bool:
        """Create backup of file"""
        try:
            source_path = self.base_path / "data" / filename
            backup_path = self.base_path / "backup" / f"{filename}.backup"
            
            if not source_path.exists():
                logger.warning(f"Source file not found: {filename}")
                return False
            
            # Copy file to backup
            shutil.copy2(source_path, backup_path)
            
            # Create backup metadata
            backup_metadata = {
                "original_file": filename,
                "backup_path": str(backup_path),
                "created_at": datetime.utcnow().isoformat(),
                "size": backup_path.stat().st_size
            }
            
            backup_metadata_path = self.base_path / "backup" / f"{filename}.backup.meta"
            with open(backup_metadata_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Backup created: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to backup file {filename}: {e}")
            return False
    
    def restore_file(self, filename: str) -> bool:
        """Restore file from backup"""
        try:
            backup_path = self.base_path / "backup" / f"{filename}.backup"
            target_path = self.base_path / "data" / filename
            
            if not backup_path.exists():
                logger.warning(f"Backup not found: {filename}")
                return False
            
            # Restore file
            shutil.copy2(backup_path, target_path)
            
            # Update metadata
            self._update_metadata(filename)
            
            logger.info(f"File restored: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore file {filename}: {e}")
            return False
    
    def _generate_unique_filename(self, original_path: str) -> str:
        """Generate unique filename"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(original_path.encode()).hexdigest()[:8]
        extension = Path(original_path).suffix
        return f"{timestamp}_{name_hash}{extension}"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _store_metadata(self, filename: str, metadata: FileMetadata) -> None:
        """Store file metadata"""
        metadata_path = self.base_path / "metadata" / f"{filename}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        self.file_metadata[filename] = metadata
    
    def _get_metadata(self, filename: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        if filename in self.file_metadata:
            return self.file_metadata[filename]
        
        metadata_path = self.base_path / "metadata" / f"{filename}.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                metadata = FileMetadata(**data)
                self.file_metadata[filename] = metadata
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata for {filename}: {e}")
        
        return None
    
    def _update_access_time(self, filename: str) -> None:
        """Update file access time"""
        metadata = self._get_metadata(filename)
        if metadata:
            metadata.accessed_at = datetime.utcnow()
            self._store_metadata(filename, metadata)
    
    def _update_metadata(self, filename: str) -> None:
        """Update file metadata"""
        file_path = self.base_path / "data" / filename
        
        if file_path.exists():
            metadata = FileMetadata(
                path=str(file_path),
                size=file_path.stat().st_size,
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                modified_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                accessed_at=datetime.fromtimestamp(file_path.stat().st_atime),
                checksum=self._calculate_checksum(file_path),
                permissions=file_path.stat().st_mode,
                owner=file_path.owner(),
                group=file_path.group()
            )
            
            self._store_metadata(filename, metadata)

class AsyncFileStorage:
    """Async file storage implementation"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.storage_lock = asyncio.Lock()
        self.initialize_storage()
    
    async def initialize_storage(self) -> None:
        """Initialize async storage directory"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            logger.info(f"Async storage initialized at {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize async storage: {e}")
            raise
    
    async def store_file(self, file_path: str, data: bytes) -> str:
        """Store file asynchronously"""
        try:
            unique_filename = self._generate_unique_filename(file_path)
            full_path = self.base_path / "data" / unique_filename
            
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(data)
            
            logger.info(f"Async file stored: {unique_filename}")
            return unique_filename
        
        except Exception as e:
            logger.error(f"Failed to store async file: {e}")
            raise
    
    async def retrieve_file(self, filename: str) -> Optional[bytes]:
        """Retrieve file asynchronously"""
        try:
            file_path = self.base_path / "data" / filename
            
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
        
        except Exception as e:
            logger.error(f"Failed to retrieve async file {filename}: {e}")
            return None
    
    def _generate_unique_filename(self, original_path: str) -> str:
        """Generate unique filename"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(original_path.encode()).hexdigest()[:8]
        extension = Path(original_path).suffix
        return f"{timestamp}_{name_hash}{extension}"

class CompressedFileStorage:
    """Compressed file storage implementation"""
    
    def __init__(self, base_path: str, compression_type: str = "gzip"):
        self.base_path = Path(base_path)
        self.compression_type = compression_type
        self.initialize_storage()
    
    def initialize_storage(self) -> None:
        """Initialize compressed storage"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "compressed").mkdir(exist_ok=True)
        (self.base_path / "archives").mkdir(exist_ok=True)
    
    def store_compressed(self, file_path: str, data: bytes) -> str:
        """Store file with compression"""
        try:
            unique_filename = self._generate_unique_filename(file_path)
            compressed_path = self.base_path / "compressed" / f"{unique_filename}.gz"
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(data)
            
            logger.info(f"Compressed file stored: {unique_filename}")
            return unique_filename
        
        except Exception as e:
            logger.error(f"Failed to store compressed file: {e}")
            raise
    
    def retrieve_compressed(self, filename: str) -> Optional[bytes]:
        """Retrieve compressed file"""
        try:
            compressed_path = self.base_path / "compressed" / f"{filename}.gz"
            
            if not compressed_path.exists():
                return None
            
            with gzip.open(compressed_path, 'rb') as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"Failed to retrieve compressed file {filename}: {e}")
            return None
    
    def create_archive(self, files: List[str], archive_name: str) -> str:
        """Create archive from multiple files"""
        try:
            archive_path = self.base_path / "archives" / f"{archive_name}.zip"
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if Path(file_path).exists():
                        zipf.write(file_path, Path(file_path).name)
            
            logger.info(f"Archive created: {archive_name}")
            return str(archive_path)
        
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            raise
    
    def extract_archive(self, archive_path: str, extract_to: str) -> List[str]:
        """Extract archive to directory"""
        try:
            extract_path = Path(extract_to)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            extracted_files = []
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                for file_info in zipf.infolist():
                    zipf.extract(file_info, extract_path)
                    extracted_files.append(str(extract_path / file_info.filename))
            
            logger.info(f"Archive extracted to: {extract_to}")
            return extracted_files
        
        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            raise
    
    def _generate_unique_filename(self, original_path: str) -> str:
        """Generate unique filename"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(original_path.encode()).hexdigest()[:8]
        extension = Path(original_path).suffix
        return f"{timestamp}_{name_hash}{extension}"

# Usage examples
def example_file_storage():
    """Example file storage usage"""
    # Create file storage manager
    storage_manager = FileStorageManager("/tmp/storage")
    
    # Store file
    file_id = storage_manager.store_file("test.txt", "Hello, World!")
    print(f"File stored with ID: {file_id}")
    
    # Retrieve file
    data = storage_manager.retrieve_file(file_id)
    print(f"Retrieved data: {data.decode() if data else None}")
    
    # List files
    files = storage_manager.list_files()
    print(f"Files in storage: {len(files)}")
    
    # Backup file
    backup_success = storage_manager.backup_file(file_id)
    print(f"Backup created: {backup_success}")
    
    # Compressed storage
    compressed_storage = CompressedFileStorage("/tmp/compressed")
    
    # Store compressed file
    compressed_id = compressed_storage.store_compressed("data.txt", b"Compressed data")
    print(f"Compressed file stored: {compressed_id}")
    
    # Retrieve compressed file
    compressed_data = compressed_storage.retrieve_compressed(compressed_id)
    print(f"Retrieved compressed data: {compressed_data}")
    
    # Create archive
    files_to_archive = ["/tmp/storage/data/test.txt"]
    archive_path = compressed_storage.create_archive(files_to_archive, "test_archive")
    print(f"Archive created: {archive_path}")
```

### Object Storage Integration

```python
# python/02-object-storage.py

"""
Object storage integration patterns and cloud storage strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import boto3
from botocore.exceptions import ClientError
import google.cloud.storage as gcs
from azure.storage.blob import BlobServiceClient
import os
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class StorageProvider(Enum):
    """Storage provider enumeration"""
    AWS_S3 = "aws_s3"
    GOOGLE_GCS = "google_gcs"
    AZURE_BLOB = "azure_blob"
    MINIO = "minio"

@dataclass
class ObjectMetadata:
    """Object metadata definition"""
    key: str
    size: int
    content_type: str
    last_modified: datetime
    etag: str
    metadata: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "last_modified": self.last_modified.isoformat(),
            "etag": self.etag,
            "metadata": self.metadata
        }

class S3StorageManager:
    """AWS S3 storage manager"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)
        self.bucket = self.s3_resource.Bucket(bucket_name)
    
    def upload_object(self, key: str, data: bytes, 
                     content_type: str = "application/octet-stream",
                     metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload object to S3"""
        try:
            extra_args = {
                'ContentType': content_type
            }
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                **extra_args
            )
            
            logger.info(f"Object uploaded to S3: {key}")
            return True
        
        except ClientError as e:
            logger.error(f"Failed to upload object to S3: {e}")
            return False
    
    def download_object(self, key: str) -> Optional[bytes]:
        """Download object from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        
        except ClientError as e:
            logger.error(f"Failed to download object from S3: {e}")
            return None
    
    def delete_object(self, key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Object deleted from S3: {key}")
            return True
        
        except ClientError as e:
            logger.error(f"Failed to delete object from S3: {e}")
            return False
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[ObjectMetadata]:
        """List objects in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                metadata = ObjectMetadata(
                    key=obj['Key'],
                    size=obj['Size'],
                    content_type=obj.get('ContentType', 'application/octet-stream'),
                    last_modified=obj['LastModified'],
                    etag=obj['ETag'],
                    metadata=obj.get('Metadata', {})
                )
                objects.append(metadata)
            
            return objects
        
        except ClientError as e:
            logger.error(f"Failed to list objects in S3: {e}")
            return []
    
    def generate_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """Generate presigned URL for object"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

class GCSStorageManager:
    """Google Cloud Storage manager"""
    
    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.client = gcs.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_object(self, key: str, data: bytes,
                     content_type: str = "application/octet-stream",
                     metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload object to GCS"""
        try:
            blob = self.bucket.blob(key)
            blob.content_type = content_type
            
            if metadata:
                blob.metadata = metadata
            
            blob.upload_from_string(data)
            logger.info(f"Object uploaded to GCS: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload object to GCS: {e}")
            return False
    
    def download_object(self, key: str) -> Optional[bytes]:
        """Download object from GCS"""
        try:
            blob = self.bucket.blob(key)
            return blob.download_as_bytes()
        
        except Exception as e:
            logger.error(f"Failed to download object from GCS: {e}")
            return None
    
    def delete_object(self, key: str) -> bool:
        """Delete object from GCS"""
        try:
            blob = self.bucket.blob(key)
            blob.delete()
            logger.info(f"Object deleted from GCS: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete object from GCS: {e}")
            return False
    
    def list_objects(self, prefix: str = "", max_results: int = 1000) -> List[ObjectMetadata]:
        """List objects in GCS bucket"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
            
            objects = []
            for blob in blobs:
                metadata = ObjectMetadata(
                    key=blob.name,
                    size=blob.size,
                    content_type=blob.content_type or 'application/octet-stream',
                    last_modified=blob.updated,
                    etag=blob.etag,
                    metadata=blob.metadata or {}
                )
                objects.append(metadata)
            
            return objects
        
        except Exception as e:
            logger.error(f"Failed to list objects in GCS: {e}")
            return []
    
    def generate_signed_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """Generate signed URL for object"""
        try:
            blob = self.bucket.blob(key)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration),
                method="GET"
            )
            return url
        
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None

class AzureBlobStorageManager:
    """Azure Blob Storage manager"""
    
    def __init__(self, account_name: str, account_key: str, container_name: str):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key
        )
        self.container_client = self.blob_service_client.get_container_client(container_name)
    
    def upload_object(self, key: str, data: bytes,
                     content_type: str = "application/octet-stream",
                     metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload object to Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(key)
            
            upload_options = {
                'content_type': content_type
            }
            
            if metadata:
                upload_options['metadata'] = metadata
            
            blob_client.upload_blob(data, **upload_options)
            logger.info(f"Object uploaded to Azure Blob: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload object to Azure Blob: {e}")
            return False
    
    def download_object(self, key: str) -> Optional[bytes]:
        """Download object from Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(key)
            return blob_client.download_blob().readall()
        
        except Exception as e:
            logger.error(f"Failed to download object from Azure Blob: {e}")
            return None
    
    def delete_object(self, key: str) -> bool:
        """Delete object from Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(key)
            blob_client.delete_blob()
            logger.info(f"Object deleted from Azure Blob: {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete object from Azure Blob: {e}")
            return False
    
    def list_objects(self, prefix: str = "", max_results: int = 1000) -> List[ObjectMetadata]:
        """List objects in Azure Blob container"""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            
            objects = []
            count = 0
            for blob in blobs:
                if count >= max_results:
                    break
                
                metadata = ObjectMetadata(
                    key=blob.name,
                    size=blob.size,
                    content_type=blob.content_settings.content_type or 'application/octet-stream',
                    last_modified=blob.last_modified,
                    etag=blob.etag,
                    metadata=blob.metadata or {}
                )
                objects.append(metadata)
                count += 1
            
            return objects
        
        except Exception as e:
            logger.error(f"Failed to list objects in Azure Blob: {e}")
            return []

class MultiCloudStorageManager:
    """Multi-cloud storage manager"""
    
    def __init__(self, providers: Dict[StorageProvider, Any]):
        self.providers = providers
        self.primary_provider = list(providers.keys())[0]
        self.replication_providers = list(providers.keys())[1:]
    
    def upload_object(self, key: str, data: bytes,
                     content_type: str = "application/octet-stream",
                     metadata: Optional[Dict[str, str]] = None,
                     replicate: bool = True) -> bool:
        """Upload object to primary provider and optionally replicate"""
        try:
            # Upload to primary provider
            primary_success = self._upload_to_provider(
                self.primary_provider, key, data, content_type, metadata
            )
            
            if not primary_success:
                return False
            
            # Replicate to other providers
            if replicate:
                self._replicate_object(key, data, content_type, metadata)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload object: {e}")
            return False
    
    def download_object(self, key: str, provider: Optional[StorageProvider] = None) -> Optional[bytes]:
        """Download object from specified provider or try all providers"""
        if provider:
            return self._download_from_provider(provider, key)
        
        # Try primary provider first
        data = self._download_from_provider(self.primary_provider, key)
        if data:
            return data
        
        # Try other providers
        for provider in self.replication_providers:
            data = self._download_from_provider(provider, key)
            if data:
                return data
        
        return None
    
    def delete_object(self, key: str) -> bool:
        """Delete object from all providers"""
        success = True
        
        for provider in self.providers.keys():
            if not self._delete_from_provider(provider, key):
                success = False
        
        return success
    
    def _upload_to_provider(self, provider: StorageProvider, key: str, data: bytes,
                           content_type: str, metadata: Optional[Dict[str, str]]) -> bool:
        """Upload object to specific provider"""
        try:
            if provider == StorageProvider.AWS_S3:
                return self.providers[provider].upload_object(key, data, content_type, metadata)
            elif provider == StorageProvider.GOOGLE_GCS:
                return self.providers[provider].upload_object(key, data, content_type, metadata)
            elif provider == StorageProvider.AZURE_BLOB:
                return self.providers[provider].upload_object(key, data, content_type, metadata)
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to upload to {provider}: {e}")
            return False
    
    def _download_from_provider(self, provider: StorageProvider, key: str) -> Optional[bytes]:
        """Download object from specific provider"""
        try:
            if provider == StorageProvider.AWS_S3:
                return self.providers[provider].download_object(key)
            elif provider == StorageProvider.GOOGLE_GCS:
                return self.providers[provider].download_object(key)
            elif provider == StorageProvider.AZURE_BLOB:
                return self.providers[provider].download_object(key)
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to download from {provider}: {e}")
            return None
    
    def _delete_from_provider(self, provider: StorageProvider, key: str) -> bool:
        """Delete object from specific provider"""
        try:
            if provider == StorageProvider.AWS_S3:
                return self.providers[provider].delete_object(key)
            elif provider == StorageProvider.GOOGLE_GCS:
                return self.providers[provider].delete_object(key)
            elif provider == StorageProvider.AZURE_BLOB:
                return self.providers[provider].delete_object(key)
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete from {provider}: {e}")
            return False
    
    def _replicate_object(self, key: str, data: bytes, content_type: str, metadata: Optional[Dict[str, str]]) -> None:
        """Replicate object to other providers"""
        def replicate_to_provider(provider: StorageProvider):
            try:
                self._upload_to_provider(provider, key, data, content_type, metadata)
            except Exception as e:
                logger.error(f"Failed to replicate to {provider}: {e}")
        
        # Replicate in parallel
        with ThreadPoolExecutor(max_workers=len(self.replication_providers)) as executor:
            futures = [executor.submit(replicate_to_provider, provider) for provider in self.replication_providers]
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Replication error: {e}")

# Usage examples
def example_object_storage():
    """Example object storage usage"""
    # S3 storage
    s3_manager = S3StorageManager("my-bucket", "us-east-1")
    
    # Upload object
    upload_success = s3_manager.upload_object("test.txt", b"Hello, S3!", "text/plain")
    print(f"S3 upload success: {upload_success}")
    
    # Download object
    data = s3_manager.download_object("test.txt")
    print(f"S3 download result: {data}")
    
    # List objects
    objects = s3_manager.list_objects()
    print(f"S3 objects: {len(objects)}")
    
    # Generate presigned URL
    url = s3_manager.generate_presigned_url("test.txt")
    print(f"S3 presigned URL: {url}")
    
    # GCS storage
    gcs_manager = GCSStorageManager("my-bucket", "my-project")
    
    # Upload object
    gcs_success = gcs_manager.upload_object("test.txt", b"Hello, GCS!", "text/plain")
    print(f"GCS upload success: {gcs_success}")
    
    # Multi-cloud storage
    providers = {
        StorageProvider.AWS_S3: s3_manager,
        StorageProvider.GOOGLE_GCS: gcs_manager
    }
    
    multi_cloud_manager = MultiCloudStorageManager(providers)
    
    # Upload with replication
    multi_success = multi_cloud_manager.upload_object("multi.txt", b"Hello, Multi-Cloud!", "text/plain")
    print(f"Multi-cloud upload success: {multi_success}")
    
    # Download from any provider
    multi_data = multi_cloud_manager.download_object("multi.txt")
    print(f"Multi-cloud download result: {multi_data}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. File storage
storage_manager = FileStorageManager("/tmp/storage")
file_id = storage_manager.store_file("test.txt", "Hello, World!")
data = storage_manager.retrieve_file(file_id)

# 2. Compressed storage
compressed_storage = CompressedFileStorage("/tmp/compressed")
compressed_id = compressed_storage.store_compressed("data.txt", b"Data")

# 3. S3 storage
s3_manager = S3StorageManager("my-bucket", "us-east-1")
s3_manager.upload_object("key", b"data", "text/plain")

# 4. Multi-cloud storage
providers = {StorageProvider.AWS_S3: s3_manager}
multi_cloud = MultiCloudStorageManager(providers)
multi_cloud.upload_object("key", b"data", replicate=True)

# 5. Async storage
async_storage = AsyncFileStorage("/tmp/async")
await async_storage.store_file("test.txt", b"Async data")
```

### Essential Patterns

```python
# Complete data storage setup
def setup_data_storage():
    """Setup complete data storage environment"""
    
    # File storage
    file_storage = FileStorageManager("/tmp/storage")
    
    # Compressed storage
    compressed_storage = CompressedFileStorage("/tmp/compressed")
    
    # Object storage
    s3_manager = S3StorageManager("my-bucket", "us-east-1")
    gcs_manager = GCSStorageManager("my-bucket", "my-project")
    
    # Multi-cloud storage
    providers = {
        StorageProvider.AWS_S3: s3_manager,
        StorageProvider.GOOGLE_GCS: gcs_manager
    }
    multi_cloud = MultiCloudStorageManager(providers)
    
    # Async storage
    async_storage = AsyncFileStorage("/tmp/async")
    
    print("Data storage setup complete!")
```

---

*This guide provides the complete machinery for Python data storage. Each pattern includes implementation examples, storage strategies, and real-world usage patterns for enterprise data management.*
