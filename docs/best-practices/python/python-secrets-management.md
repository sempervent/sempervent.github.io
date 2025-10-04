# Python Secrets Management Best Practices

**Objective**: Master senior-level Python secrets management patterns for production systems. When you need to implement secure credential handling, when you want to build robust secret rotation, when you need enterprise-grade secrets management—these best practices become your weapon of choice.

## Core Principles

- **Never Store Secrets in Code**: Use external secret stores
- **Encrypt at Rest and in Transit**: Protect secrets with encryption
- **Rotate Regularly**: Implement automatic secret rotation
- **Least Privilege**: Grant minimum necessary access
- **Audit Everything**: Log all secret access and changes

## Secret Storage Strategies

### Environment-Based Secrets

```python
# python/01-secret-storage.py

"""
Secret storage patterns and credential management
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
import json
import base64
import hashlib
import secrets
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import toml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import hvac
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import google.cloud.secretmanager as secretmanager
from google.cloud import secretmanager_v1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecretType(Enum):
    """Secret type enumeration"""
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    DATABASE_URL = "database_url"
    ENCRYPTION_KEY = "encryption_key"

class SecretSource(Enum):
    """Secret source enumeration"""
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"
    CLOUD = "cloud"
    KEYRING = "keyring"

@dataclass
class Secret:
    """Secret definition"""
    name: str
    value: str
    secret_type: SecretType
    source: SecretSource
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: Dict[str, str] = None
    version: str = "1"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class EnvironmentSecretManager:
    """Environment variable secret manager"""
    
    def __init__(self, prefix: str = "SECRET_"):
        self.prefix = prefix
        self.secrets = {}
        self.load_secrets()
    
    def load_secrets(self) -> None:
        """Load secrets from environment variables"""
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                secret_name = key[len(self.prefix):].lower()
                self.secrets[secret_name] = value
                logger.debug(f"Loaded secret from environment: {secret_name}")
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value"""
        return self.secrets.get(name, default)
    
    def set_secret(self, name: str, value: str) -> None:
        """Set secret value (for testing only)"""
        self.secrets[name] = value
        os.environ[f"{self.prefix}{name.upper()}"] = value
    
    def list_secrets(self) -> List[str]:
        """List all secret names"""
        return list(self.secrets.keys())
    
    def has_secret(self, name: str) -> bool:
        """Check if secret exists"""
        return name in self.secrets

class FileSecretManager:
    """File-based secret manager"""
    
    def __init__(self, secrets_file: str, encrypted: bool = True):
        self.secrets_file = Path(secrets_file)
        self.encrypted = encrypted
        self.encryption_key = None
        self.secrets = {}
        
        if encrypted:
            self.encryption_key = self._get_or_create_encryption_key()
        
        self.load_secrets()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = self.secrets_file.parent / ".secret_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            key_file.chmod(0o600)
            return key
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt secret value"""
        if not self.encrypted:
            return value
        
        fernet = Fernet(self.encryption_key)
        encrypted_value = fernet.encrypt(value.encode())
        return base64.b64encode(encrypted_value).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt secret value"""
        if not self.encrypted:
            return encrypted_value
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            fernet = Fernet(self.encryption_key)
            decrypted_value = fernet.decrypt(encrypted_bytes)
            return decrypted_value.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
    
    def load_secrets(self) -> None:
        """Load secrets from file"""
        if not self.secrets_file.exists():
            logger.warning(f"Secrets file not found: {self.secrets_file}")
            return
        
        try:
            with open(self.secrets_file, 'r') as f:
                if self.secrets_file.suffix == '.json':
                    data = json.load(f)
                elif self.secrets_file.suffix == '.yaml' or self.secrets_file.suffix == '.yml':
                    data = yaml.safe_load(f)
                elif self.secrets_file.suffix == '.toml':
                    data = toml.load(f)
                else:
                    # Assume plain text format
                    data = {}
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            data[key] = value
            
            for name, value in data.items():
                if self.encrypted:
                    decrypted_value = self._decrypt_value(value)
                else:
                    decrypted_value = value
                
                self.secrets[name] = decrypted_value
                logger.debug(f"Loaded secret from file: {name}")
        
        except Exception as e:
            logger.error(f"Failed to load secrets from file: {e}")
            raise
    
    def save_secrets(self) -> None:
        """Save secrets to file"""
        try:
            data = {}
            for name, value in self.secrets.items():
                if self.encrypted:
                    encrypted_value = self._encrypt_value(value)
                else:
                    encrypted_value = value
                data[name] = encrypted_value
            
            with open(self.secrets_file, 'w') as f:
                if self.secrets_file.suffix == '.json':
                    json.dump(data, f, indent=2)
                elif self.secrets_file.suffix == '.yaml' or self.secrets_file.suffix == '.yml':
                    yaml.dump(data, f, default_flow_style=False)
                elif self.secrets_file.suffix == '.toml':
                    toml.dump(data, f)
                else:
                    # Plain text format
                    for name, value in data.items():
                        f.write(f"{name}={value}\n")
            
            # Set restrictive permissions
            self.secrets_file.chmod(0o600)
            logger.info(f"Secrets saved to file: {self.secrets_file}")
        
        except Exception as e:
            logger.error(f"Failed to save secrets to file: {e}")
            raise
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value"""
        return self.secrets.get(name, default)
    
    def set_secret(self, name: str, value: str) -> None:
        """Set secret value"""
        self.secrets[name] = value
        self.save_secrets()
    
    def delete_secret(self, name: str) -> bool:
        """Delete secret"""
        if name in self.secrets:
            del self.secrets[name]
            self.save_secrets()
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """List all secret names"""
        return list(self.secrets.keys())

class KeyringSecretManager:
    """Keyring-based secret manager"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from keyring"""
        try:
            return keyring.get_password(self.service_name, name)
        except Exception as e:
            logger.error(f"Failed to get secret from keyring: {e}")
            return default
    
    def set_secret(self, name: str, value: str) -> None:
        """Set secret in keyring"""
        try:
            keyring.set_password(self.service_name, name, value)
            logger.info(f"Secret set in keyring: {name}")
        except Exception as e:
            logger.error(f"Failed to set secret in keyring: {e}")
            raise
    
    def delete_secret(self, name: str) -> bool:
        """Delete secret from keyring"""
        try:
            keyring.delete_password(self.service_name, name)
            logger.info(f"Secret deleted from keyring: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from keyring: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List secrets (keyring doesn't support listing)"""
        logger.warning("Keyring doesn't support listing secrets")
        return []

class VaultSecretManager:
    """HashiCorp Vault secret manager"""
    
    def __init__(self, vault_url: str, token: str, mount_point: str = "secret"):
        self.vault_url = vault_url
        self.token = token
        self.mount_point = mount_point
        self.client = hvac.Client(url=vault_url, token=token)
        self.client.is_authenticated()
    
    def get_secret(self, path: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            return response['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Failed to get secret from Vault: {e}")
            return default
    
    def set_secret(self, path: str, value: str, metadata: Optional[Dict[str, str]] = None) -> None:
        """Set secret in Vault"""
        try:
            secret_data = {'value': value}
            if metadata:
                secret_data.update(metadata)
            
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path, secret=secret_data, mount_point=self.mount_point
            )
            logger.info(f"Secret set in Vault: {path}")
        except Exception as e:
            logger.error(f"Failed to set secret in Vault: {e}")
            raise
    
    def delete_secret(self, path: str) -> bool:
        """Delete secret from Vault"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=path, mount_point=self.mount_point
            )
            logger.info(f"Secret deleted from Vault: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from Vault: {e}")
            return False
    
    def list_secrets(self, path: str = "") -> List[str]:
        """List secrets in Vault"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=path, mount_point=self.mount_point
            )
            return response['data']['keys']
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []

class AWSSecretsManager:
    """AWS Secrets Manager client"""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except Exception as e:
            logger.error(f"Failed to get secret from AWS: {e}")
            return default
    
    def set_secret(self, secret_name: str, value: str, description: str = "") -> None:
        """Set secret in AWS Secrets Manager"""
        try:
            self.client.create_secret(
                Name=secret_name,
                SecretString=value,
                Description=description
            )
            logger.info(f"Secret set in AWS: {secret_name}")
        except self.client.exceptions.ResourceExistsException:
            # Update existing secret
            self.client.update_secret(
                SecretId=secret_name,
                SecretString=value
            )
            logger.info(f"Secret updated in AWS: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to set secret in AWS: {e}")
            raise
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from AWS Secrets Manager"""
        try:
            self.client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True
            )
            logger.info(f"Secret deleted from AWS: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from AWS: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List secrets in AWS Secrets Manager"""
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []

class AzureKeyVaultManager:
    """Azure Key Vault secret manager"""
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=self.credential)
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from Azure Key Vault"""
        try:
            secret = self.client.get_secret(name)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get secret from Azure: {e}")
            return default
    
    def set_secret(self, name: str, value: str) -> None:
        """Set secret in Azure Key Vault"""
        try:
            self.client.set_secret(name, value)
            logger.info(f"Secret set in Azure: {name}")
        except Exception as e:
            logger.error(f"Failed to set secret in Azure: {e}")
            raise
    
    def delete_secret(self, name: str) -> bool:
        """Delete secret from Azure Key Vault"""
        try:
            self.client.begin_delete_secret(name)
            logger.info(f"Secret deleted from Azure: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from Azure: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List secrets in Azure Key Vault"""
        try:
            secrets = []
            for secret_properties in self.client.list_properties_of_secrets():
                secrets.append(secret_properties.name)
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets from Azure: {e}")
            return []

class GCPSecretManager:
    """Google Cloud Secret Manager client"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
    
    def get_secret(self, secret_id: str, version: str = "latest", default: Optional[str] = None) -> Optional[str]:
        """Get secret from GCP Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to get secret from GCP: {e}")
            return default
    
    def set_secret(self, secret_id: str, value: str) -> None:
        """Set secret in GCP Secret Manager"""
        try:
            parent = f"projects/{self.project_id}"
            
            # Create secret if it doesn't exist
            try:
                self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            except Exception:
                # Secret might already exist
                pass
            
            # Add secret version
            self.client.add_secret_version(
                request={
                    "parent": f"{parent}/secrets/{secret_id}",
                    "payload": {"data": value.encode("UTF-8")},
                }
            )
            logger.info(f"Secret set in GCP: {secret_id}")
        except Exception as e:
            logger.error(f"Failed to set secret in GCP: {e}")
            raise
    
    def delete_secret(self, secret_id: str) -> bool:
        """Delete secret from GCP Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}"
            self.client.delete_secret(request={"name": name})
            logger.info(f"Secret deleted from GCP: {secret_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from GCP: {e}")
            return False
    
    def list_secrets(self) -> List[str]:
        """List secrets in GCP Secret Manager"""
        try:
            parent = f"projects/{self.project_id}"
            secrets = []
            for secret in self.client.list_secrets(request={"parent": parent}):
                secret_id = secret.name.split("/")[-1]
                secrets.append(secret_id)
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets from GCP: {e}")
            return []

class UnifiedSecretManager:
    """Unified secret manager supporting multiple backends"""
    
    def __init__(self, primary_backend: Any, fallback_backends: List[Any] = None):
        self.primary_backend = primary_backend
        self.fallback_backends = fallback_backends or []
        self.all_backends = [primary_backend] + self.fallback_backends
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from primary backend with fallback"""
        # Try primary backend first
        value = self.primary_backend.get_secret(name)
        if value is not None:
            return value
        
        # Try fallback backends
        for backend in self.fallback_backends:
            value = backend.get_secret(name)
            if value is not None:
                return value
        
        return default
    
    def set_secret(self, name: str, value: str) -> None:
        """Set secret in primary backend"""
        self.primary_backend.set_secret(name, value)
    
    def delete_secret(self, name: str) -> bool:
        """Delete secret from all backends"""
        success = True
        for backend in self.all_backends:
            if hasattr(backend, 'delete_secret'):
                if not backend.delete_secret(name):
                    success = False
        return success
    
    def list_secrets(self) -> List[str]:
        """List secrets from primary backend"""
        return self.primary_backend.list_secrets()

# Usage examples
def example_secret_management():
    """Example secret management usage"""
    # Environment-based secrets
    env_manager = EnvironmentSecretManager("APP_")
    env_secret = env_manager.get_secret("database_password")
    print(f"Environment secret: {env_secret}")
    
    # File-based secrets
    file_manager = FileSecretManager("/tmp/secrets.json", encrypted=True)
    file_manager.set_secret("api_key", "secret-api-key")
    file_secret = file_manager.get_secret("api_key")
    print(f"File secret: {file_secret}")
    
    # Keyring secrets
    keyring_manager = KeyringSecretManager("my-app")
    keyring_manager.set_secret("database_url", "postgresql://user:pass@localhost/db")
    keyring_secret = keyring_manager.get_secret("database_url")
    print(f"Keyring secret: {keyring_secret}")
    
    # AWS Secrets Manager
    aws_manager = AWSSecretsManager("us-east-1")
    aws_secret = aws_manager.get_secret("my-secret")
    print(f"AWS secret: {aws_secret}")
    
    # Unified secret manager
    unified_manager = UnifiedSecretManager(
        primary_backend=env_manager,
        fallback_backends=[file_manager, keyring_manager]
    )
    
    unified_secret = unified_manager.get_secret("database_password")
    print(f"Unified secret: {unified_secret}")
```

### Secret Rotation

```python
# python/02-secret-rotation.py

"""
Secret rotation patterns and automated credential management
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import time
import threading
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import schedule
import croniter

logger = logging.getLogger(__name__)

class RotationStrategy(Enum):
    """Rotation strategy enumeration"""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    MANUAL = "manual"
    EVENT_BASED = "event_based"

@dataclass
class RotationPolicy:
    """Secret rotation policy"""
    secret_name: str
    strategy: RotationStrategy
    interval_days: int = 30
    max_usage_count: int = 1000
    cron_schedule: str = "0 0 * * *"  # Daily at midnight
    auto_rotate: bool = True
    notify_before_days: int = 7

class SecretRotator:
    """Secret rotation manager"""
    
    def __init__(self, secret_manager: Any):
        self.secret_manager = secret_manager
        self.rotation_policies = {}
        self.rotation_history = {}
        self.usage_counts = {}
        self.rotation_lock = threading.Lock()
        self.is_running = False
        self.rotation_thread = None
    
    def add_rotation_policy(self, policy: RotationPolicy) -> None:
        """Add rotation policy for secret"""
        self.rotation_policies[policy.secret_name] = policy
        logger.info(f"Added rotation policy for secret: {policy.secret_name}")
    
    def start_rotation_scheduler(self) -> None:
        """Start automatic rotation scheduler"""
        if self.is_running:
            logger.warning("Rotation scheduler is already running")
            return
        
        self.is_running = True
        self.rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self.rotation_thread.start()
        logger.info("Secret rotation scheduler started")
    
    def stop_rotation_scheduler(self) -> None:
        """Stop automatic rotation scheduler"""
        self.is_running = False
        if self.rotation_thread:
            self.rotation_thread.join()
        logger.info("Secret rotation scheduler stopped")
    
    def _rotation_loop(self) -> None:
        """Main rotation loop"""
        while self.is_running:
            try:
                self._check_rotation_requirements()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                time.sleep(60)
    
    def _check_rotation_requirements(self) -> None:
        """Check if any secrets need rotation"""
        for secret_name, policy in self.rotation_policies.items():
            if not policy.auto_rotate:
                continue
            
            try:
                if self._should_rotate_secret(secret_name, policy):
                    self._rotate_secret(secret_name, policy)
            except Exception as e:
                logger.error(f"Error checking rotation for {secret_name}: {e}")
    
    def _should_rotate_secret(self, secret_name: str, policy: RotationPolicy) -> bool:
        """Check if secret should be rotated"""
        if policy.strategy == RotationStrategy.TIME_BASED:
            return self._check_time_based_rotation(secret_name, policy)
        elif policy.strategy == RotationStrategy.USAGE_BASED:
            return self._check_usage_based_rotation(secret_name, policy)
        elif policy.strategy == RotationStrategy.EVENT_BASED:
            return self._check_event_based_rotation(secret_name, policy)
        
        return False
    
    def _check_time_based_rotation(self, secret_name: str, policy: RotationPolicy) -> bool:
        """Check time-based rotation requirements"""
        if secret_name not in self.rotation_history:
            return True  # First rotation
        
        last_rotation = self.rotation_history[secret_name].get('last_rotation')
        if not last_rotation:
            return True
        
        days_since_rotation = (datetime.utcnow() - last_rotation).days
        return days_since_rotation >= policy.interval_days
    
    def _check_usage_based_rotation(self, secret_name: str, policy: RotationPolicy) -> bool:
        """Check usage-based rotation requirements"""
        usage_count = self.usage_counts.get(secret_name, 0)
        return usage_count >= policy.max_usage_count
    
    def _check_event_based_rotation(self, secret_name: str, policy: RotationPolicy) -> bool:
        """Check event-based rotation requirements"""
        # This would be implemented based on specific events
        # For example, security breach, compromised credentials, etc.
        return False
    
    def _rotate_secret(self, secret_name: str, policy: RotationPolicy) -> None:
        """Rotate secret"""
        with self.rotation_lock:
            try:
                logger.info(f"Starting rotation for secret: {secret_name}")
                
                # Generate new secret
                new_secret = self._generate_new_secret(secret_name)
                
                # Update secret in all backends
                self._update_secret_in_backends(secret_name, new_secret)
                
                # Update rotation history
                self._update_rotation_history(secret_name, new_secret)
                
                # Reset usage count
                self.usage_counts[secret_name] = 0
                
                # Notify about rotation
                self._notify_rotation(secret_name, policy)
                
                logger.info(f"Successfully rotated secret: {secret_name}")
                
            except Exception as e:
                logger.error(f"Failed to rotate secret {secret_name}: {e}")
                raise
    
    def _generate_new_secret(self, secret_name: str) -> str:
        """Generate new secret value"""
        # This would be implemented based on secret type
        # For example, generate new password, API key, etc.
        return f"new_secret_{int(time.time())}"
    
    def _update_secret_in_backends(self, secret_name: str, new_secret: str) -> None:
        """Update secret in all backends"""
        if hasattr(self.secret_manager, 'set_secret'):
            self.secret_manager.set_secret(secret_name, new_secret)
        elif hasattr(self.secret_manager, 'all_backends'):
            for backend in self.secret_manager.all_backends:
                if hasattr(backend, 'set_secret'):
                    backend.set_secret(secret_name, new_secret)
    
    def _update_rotation_history(self, secret_name: str, new_secret: str) -> None:
        """Update rotation history"""
        if secret_name not in self.rotation_history:
            self.rotation_history[secret_name] = {}
        
        self.rotation_history[secret_name].update({
            'last_rotation': datetime.utcnow(),
            'rotation_count': self.rotation_history[secret_name].get('rotation_count', 0) + 1,
            'previous_secret': self.rotation_history[secret_name].get('current_secret'),
            'current_secret': new_secret
        })
    
    def _notify_rotation(self, secret_name: str, policy: RotationPolicy) -> None:
        """Notify about secret rotation"""
        # This would implement actual notification logic
        # For example, send email, Slack message, etc.
        logger.info(f"Notification: Secret {secret_name} has been rotated")
    
    def record_secret_usage(self, secret_name: str) -> None:
        """Record secret usage for usage-based rotation"""
        if secret_name in self.usage_counts:
            self.usage_counts[secret_name] += 1
    
    def get_rotation_status(self, secret_name: str) -> Dict[str, Any]:
        """Get rotation status for secret"""
        if secret_name not in self.rotation_policies:
            return {"error": "No rotation policy found"}
        
        policy = self.rotation_policies[secret_name]
        history = self.rotation_history.get(secret_name, {})
        
        return {
            "secret_name": secret_name,
            "policy": {
                "strategy": policy.strategy.value,
                "interval_days": policy.interval_days,
                "auto_rotate": policy.auto_rotate
            },
            "history": history,
            "usage_count": self.usage_counts.get(secret_name, 0),
            "next_rotation": self._calculate_next_rotation(secret_name, policy)
        }
    
    def _calculate_next_rotation(self, secret_name: str, policy: RotationPolicy) -> Optional[datetime]:
        """Calculate next rotation time"""
        if not policy.auto_rotate:
            return None
        
        if policy.strategy == RotationStrategy.TIME_BASED:
            last_rotation = self.rotation_history.get(secret_name, {}).get('last_rotation')
            if last_rotation:
                return last_rotation + timedelta(days=policy.interval_days)
            else:
                return datetime.utcnow() + timedelta(days=policy.interval_days)
        
        return None
    
    def manual_rotate(self, secret_name: str) -> bool:
        """Manually rotate secret"""
        if secret_name not in self.rotation_policies:
            logger.error(f"No rotation policy found for secret: {secret_name}")
            return False
        
        policy = self.rotation_policies[secret_name]
        try:
            self._rotate_secret(secret_name, policy)
            return True
        except Exception as e:
            logger.error(f"Manual rotation failed for {secret_name}: {e}")
            return False

class SecretAuditor:
    """Secret access auditor"""
    
    def __init__(self):
        self.access_logs = []
        self.audit_lock = threading.Lock()
    
    def log_secret_access(self, secret_name: str, user: str, action: str, 
                         success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log secret access"""
        with self.audit_lock:
            log_entry = {
                "timestamp": datetime.utcnow(),
                "secret_name": secret_name,
                "user": user,
                "action": action,
                "success": success,
                "metadata": metadata or {}
            }
            self.access_logs.append(log_entry)
            logger.info(f"Secret access logged: {secret_name} by {user}")
    
    def get_access_logs(self, secret_name: Optional[str] = None, 
                       user: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get access logs with optional filtering"""
        with self.audit_lock:
            logs = self.access_logs.copy()
        
        # Apply filters
        if secret_name:
            logs = [log for log in logs if log["secret_name"] == secret_name]
        
        if user:
            logs = [log for log in logs if log["user"] == user]
        
        if start_time:
            logs = [log for log in logs if log["timestamp"] >= start_time]
        
        if end_time:
            logs = [log for log in logs if log["timestamp"] <= end_time]
        
        return logs
    
    def get_secret_usage_stats(self, secret_name: str) -> Dict[str, Any]:
        """Get usage statistics for secret"""
        logs = self.get_access_logs(secret_name=secret_name)
        
        total_accesses = len(logs)
        successful_accesses = len([log for log in logs if log["success"]])
        failed_accesses = total_accesses - successful_accesses
        
        # Group by user
        user_accesses = {}
        for log in logs:
            user = log["user"]
            if user not in user_accesses:
                user_accesses[user] = 0
            user_accesses[user] += 1
        
        return {
            "secret_name": secret_name,
            "total_accesses": total_accesses,
            "successful_accesses": successful_accesses,
            "failed_accesses": failed_accesses,
            "success_rate": (successful_accesses / total_accesses * 100) if total_accesses > 0 else 0,
            "user_accesses": user_accesses,
            "last_access": logs[-1]["timestamp"] if logs else None
        }

# Usage examples
def example_secret_rotation():
    """Example secret rotation usage"""
    # Create secret manager
    secret_manager = EnvironmentSecretManager()
    
    # Create rotator
    rotator = SecretRotator(secret_manager)
    
    # Add rotation policies
    policy1 = RotationPolicy(
        secret_name="database_password",
        strategy=RotationStrategy.TIME_BASED,
        interval_days=30,
        auto_rotate=True
    )
    rotator.add_rotation_policy(policy1)
    
    policy2 = RotationPolicy(
        secret_name="api_key",
        strategy=RotationStrategy.USAGE_BASED,
        max_usage_count=1000,
        auto_rotate=True
    )
    rotator.add_rotation_policy(policy2)
    
    # Start rotation scheduler
    rotator.start_rotation_scheduler()
    
    # Record secret usage
    rotator.record_secret_usage("api_key")
    
    # Get rotation status
    status = rotator.get_rotation_status("database_password")
    print(f"Rotation status: {status}")
    
    # Manual rotation
    manual_success = rotator.manual_rotate("database_password")
    print(f"Manual rotation: {manual_success}")
    
    # Secret auditing
    auditor = SecretAuditor()
    auditor.log_secret_access("database_password", "user1", "read", True)
    auditor.log_secret_access("api_key", "user2", "write", False)
    
    # Get usage stats
    stats = auditor.get_secret_usage_stats("database_password")
    print(f"Usage stats: {stats}")
    
    # Stop rotation scheduler
    rotator.stop_rotation_scheduler()
```

## TL;DR Runbook

### Quick Start

```python
# 1. Environment secrets
env_manager = EnvironmentSecretManager("APP_")
secret = env_manager.get_secret("database_password")

# 2. File-based secrets
file_manager = FileSecretManager("/tmp/secrets.json", encrypted=True)
file_manager.set_secret("api_key", "secret-value")

# 3. Cloud secrets
aws_manager = AWSSecretsManager("us-east-1")
aws_secret = aws_manager.get_secret("my-secret")

# 4. Secret rotation
rotator = SecretRotator(secret_manager)
policy = RotationPolicy("database_password", RotationStrategy.TIME_BASED, interval_days=30)
rotator.add_rotation_policy(policy)
rotator.start_rotation_scheduler()

# 5. Secret auditing
auditor = SecretAuditor()
auditor.log_secret_access("secret_name", "user", "read", True)
```

### Essential Patterns

```python
# Complete secrets management setup
def setup_secrets_management():
    """Setup complete secrets management environment"""
    
    # Multiple secret backends
    env_manager = EnvironmentSecretManager("APP_")
    file_manager = FileSecretManager("/tmp/secrets.json", encrypted=True)
    aws_manager = AWSSecretsManager("us-east-1")
    
    # Unified secret manager
    unified_manager = UnifiedSecretManager(
        primary_backend=env_manager,
        fallback_backends=[file_manager, aws_manager]
    )
    
    # Secret rotation
    rotator = SecretRotator(unified_manager)
    
    # Secret auditing
    auditor = SecretAuditor()
    
    print("Secrets management setup complete!")
```

---

*This guide provides the complete machinery for Python secrets management. Each pattern includes implementation examples, rotation strategies, and real-world usage patterns for enterprise secrets management.*
