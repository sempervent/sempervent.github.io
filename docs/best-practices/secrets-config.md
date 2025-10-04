# Secrets & Configuration Management Best Practices (2025 Edition)

**Objective**: Master production-grade secrets and configuration management for modern data engineering and DevOps environments. When you need to secure sensitive data across environments, when you want to prevent credential exposure, when you're responsible for keeping secrets safe—proper secrets management becomes your weapon of choice.

Secrets are the sharpest knives in your system. If you don't sheath them properly, you'll cut yourself—or worse, everyone else.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Never hardcode secrets**
   - No secrets in source code
   - No secrets in Dockerfiles
   - No secrets in configuration files
   - No secrets in Git history

2. **Use proper secret stores**
   - Environment variables for containers
   - Secret management systems for production
   - Encrypted storage for sensitive data
   - Access control and audit logging

3. **Plan for secret rotation**
   - Regular credential rotation
   - Graceful handling of expired tokens
   - Automated rotation where possible
   - Emergency rotation procedures

4. **Validate everything**
   - Test secret injection in all environments
   - Verify access controls and permissions
   - Check for secret leakage in logs
   - Monitor for unauthorized access

5. **Plan for production**
   - Design for multiple environments
   - Enable audit logging and monitoring
   - Support emergency rotation procedures
   - Document operational procedures

**Why These Principles**: Production secrets management requires understanding security patterns, access control, and operational excellence. Understanding these patterns prevents security breaches and enables reliable system operations.

## 1) Core Principles

### The Golden Rules

```bash
# ❌ NEVER DO THIS
export DATABASE_PASSWORD="super_secret_password_123"
echo "password=super_secret_password_123" >> config.txt
git add config.txt  # This will expose your secret!

# ✅ ALWAYS DO THIS
export DATABASE_PASSWORD="${DB_PASSWORD}"  # From secure source
echo "password=${DB_PASSWORD}" >> config.txt
echo "config.txt" >> .gitignore  # Never commit secrets
```

**Why Core Principles Matter**: Proper secret handling prevents exposure and security breaches. Understanding these patterns prevents credential chaos and enables reliable security practices.

### Secret Classification

```yaml
# Secret classification levels
secrets:
  critical:
    - database_passwords
    - api_keys
    - private_keys
    - oauth_tokens
  
  sensitive:
    - connection_strings
    - encryption_keys
    - service_accounts
    - webhook_urls
  
  internal:
    - internal_apis
    - monitoring_tokens
    - debug_keys
    - test_credentials
```

**Why Secret Classification Matters**: Proper classification enables appropriate security controls. Understanding these patterns prevents over-engineering and enables efficient secret management.

## 2) .env vs Environment Variables

### .env Files for Local Development

```bash
# .env.local (NEVER commit this file)
DATABASE_URL=postgresql://user:password@localhost:5432/app
REDIS_URL=redis://localhost:6379/0
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# .env.example (ALWAYS commit this file)
DATABASE_URL=postgresql://user:password@localhost:5432/app
REDIS_URL=redis://localhost:6379/0
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

**Why .env Files Matter**: Local development requires convenient secret management. Understanding these patterns prevents development chaos and enables efficient local workflows.

### Environment Variables for Containers

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use environment variables, never hardcode
ENV DATABASE_URL=${DATABASE_URL}
ENV REDIS_URL=${REDIS_URL}
ENV API_KEY=${API_KEY}

CMD ["python", "app.py"]
```

**Why Environment Variables Matter**: Container security requires proper secret injection. Understanding these patterns prevents container vulnerabilities and enables reliable container operations.

### Python Configuration Loading

```python
# config.py
import os
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with secret management"""
    
    # Database settings
    database_url: str = Field(..., env="DATABASE_URL")
    database_password: str = Field(..., env="DATABASE_PASSWORD")
    
    # Redis settings
    redis_url: str = Field(..., env="REDIS_URL")
    redis_password: str = Field(..., env="REDIS_PASSWORD")
    
    # API settings
    api_key: str = Field(..., env="API_KEY")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # Optional settings with defaults
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Usage
settings = Settings()

# Access secrets
database_url = settings.database_url
api_key = settings.api_key
```

**Why Configuration Loading Matters**: Proper configuration management prevents hardcoded secrets. Understanding these patterns prevents configuration chaos and enables reliable application deployment.

## 3) Docker Compose Secrets & Profiles

### Docker Compose with Secrets

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    image: myapp:latest
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    secrets:
      - db_password
      - redis_password
    profiles:
      - production
      - staging
  
  db:
    image: postgres:17
    environment:
      - POSTGRES_DB=app
      - POSTGRES_USER=app_user
    secrets:
      - db_password
    profiles:
      - production
      - staging
  
  redis:
    image: redis:7-alpine
    secrets:
      - redis_password
    profiles:
      - production
      - staging

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
```

**Why Docker Compose Secrets Matter**: Container orchestration requires secure secret management. Understanding these patterns prevents container vulnerabilities and enables reliable container operations.

### Environment-Specific Profiles

```yaml
# docker-compose.dev.yml
version: "3.9"
services:
  app:
    image: myapp:dev
    environment:
      - DATABASE_URL=postgresql://user:dev_password@db:5432/app
      - REDIS_URL=redis://redis:6379/0
      - DEBUG=true
    profiles:
      - development
  
  db:
    image: postgres:17
    environment:
      - POSTGRES_DB=app
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=dev_password
    profiles:
      - development

# docker-compose.prod.yml
version: "3.9"
services:
  app:
    image: myapp:latest
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    secrets:
      - db_password
      - redis_password
    profiles:
      - production
  
  db:
    image: postgres:17
    environment:
      - POSTGRES_DB=app
      - POSTGRES_USER=app_user
    secrets:
      - db_password
    profiles:
      - production
```

**Why Environment Profiles Matter**: Environment separation prevents secret leakage. Understanding these patterns prevents environment chaos and enables reliable multi-environment deployment.

## 4) Kubernetes Secrets

### Basic Kubernetes Secrets

```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: default
type: Opaque
data:
  database-password: cG9zdGdyZXM=  # base64 encoded
  redis-password: cmVkaXM=        # base64 encoded
  api-key: YWJjMTIz              # base64 encoded
  secret-key: eHl6Nzg5            # base64 encoded
```

**Why Kubernetes Secrets Matter**: Container orchestration requires secure secret management. Understanding these patterns prevents Kubernetes vulnerabilities and enables reliable container operations.

### Sealed Secrets for GitOps

```yaml
# sealed-secret.yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: app-secrets
  namespace: default
spec:
  encryptedData:
    database-password: AgBy3i4OJSWK+PiTySYZZA8rO43cGDEQAx...
    redis-password: AgBy3i4OJSWK+PiTySYZZA8rO43cGDEQAx...
    api-key: AgBy3i4OJSWK+PiTySYZZA8rO43cGDEQAx...
    secret-key: AgBy3i4OJSWK+PiTySYZZA8rO43cGDEQAx...
```

**Why Sealed Secrets Matter**: GitOps requires encrypted secret storage. Understanding these patterns prevents Git vulnerabilities and enables reliable GitOps workflows.

### SOPS for Secret Encryption

```yaml
# secrets.yaml (encrypted with SOPS)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: default
type: Opaque
data:
  database-password: ENC[AES256_GCM,data:...]
  redis-password: ENC[AES256_GCM,data:...]
  api-key: ENC[AES256_GCM,data:...]
  secret-key: ENC[AES256_GCM,data:...]
sops:
  kms: []
  gcp_kms: []
  azure_kv: []
  hc_vault: []
  age:
    - recipient: age1...
      enc: |
        -----BEGIN AGE ENCRYPTED FILE-----
        ...
        -----END AGE ENCRYPTED FILE-----
```

**Why SOPS Matters**: Advanced encryption enables secure secret storage. Understanding these patterns prevents encryption vulnerabilities and enables reliable secret management.

## 5) Prefect & FastAPI Integration

### FastAPI Secret Loading

```python
# main.py
from fastapi import FastAPI, Depends
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    """FastAPI settings with secret management"""
    
    database_url: str = os.getenv("DATABASE_URL")
    redis_url: str = os.getenv("REDIS_URL")
    api_key: str = os.getenv("API_KEY")
    secret_key: str = os.getenv("SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

app = FastAPI()
settings = Settings()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Get configuration (without secrets)"""
    return {
        "database_url": settings.database_url,
        "redis_url": settings.redis_url,
        "api_key": "***",  # Never expose secrets
        "secret_key": "***"  # Never expose secrets
    }
```

**Why FastAPI Integration Matters**: Web applications require secure secret management. Understanding these patterns prevents web vulnerabilities and enables reliable web operations.

### Prefect Secret Blocks

```python
# prefect_secrets.py
from prefect import flow, task
from prefect.blocks.system import Secret
import os

@task
async def get_database_connection():
    """Get database connection with secret management"""
    
    # Use Prefect secret blocks
    db_password = await Secret.load("database-password")
    db_url = f"postgresql://user:{db_password.get()}@localhost:5432/app"
    
    return db_url

@task
async def get_api_credentials():
    """Get API credentials with secret management"""
    
    # Use Prefect secret blocks
    api_key = await Secret.load("api-key")
    secret_key = await Secret.load("secret-key")
    
    return {
        "api_key": api_key.get(),
        "secret_key": secret_key.get()
    }

@flow
async def data_processing_flow():
    """Main data processing flow with secret management"""
    
    # Get secrets
    db_url = await get_database_connection()
    credentials = await get_api_credentials()
    
    # Process data
    result = await process_data(db_url, credentials)
    
    return result
```

**Why Prefect Integration Matters**: Workflow orchestration requires secure secret management. Understanding these patterns prevents workflow vulnerabilities and enables reliable data processing.

## 6) Cloud-Native Secret Stores

### AWS Secrets Manager

```python
# aws_secrets.py
import boto3
import json
from typing import Dict, Any

class AWSSecretsManager:
    def __init__(self, region_name: str = "us-west-2"):
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            print(f"Error retrieving secret: {e}")
            raise
    
    def create_secret(self, secret_name: str, secret_value: Dict[str, Any]):
        """Create secret in AWS Secrets Manager"""
        try:
            self.client.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value)
            )
        except Exception as e:
            print(f"Error creating secret: {e}")
            raise

# Usage
secrets_manager = AWSSecretsManager()
database_creds = secrets_manager.get_secret("database-credentials")
api_creds = secrets_manager.get_secret("api-credentials")
```

**Why AWS Secrets Manager Matters**: Cloud-native secret storage enables secure secret management. Understanding these patterns prevents cloud vulnerabilities and enables reliable cloud operations.

### HashiCorp Vault Integration

```python
# vault_integration.py
import hvac
import os
from typing import Dict, Any

class VaultClient:
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)
    
    def get_secret(self, path: str) -> Dict[str, Any]:
        """Get secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            print(f"Error retrieving secret: {e}")
            raise
    
    def create_secret(self, path: str, secret_data: Dict[str, Any]):
        """Create secret in Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret_data
            )
        except Exception as e:
            print(f"Error creating secret: {e}")
            raise

# Usage
vault_client = VaultClient(
    vault_url=os.getenv("VAULT_URL"),
    vault_token=os.getenv("VAULT_TOKEN")
)

database_creds = vault_client.get_secret("database/credentials")
api_creds = vault_client.get_secret("api/credentials")
```

**Why Vault Integration Matters**: Enterprise secret management enables advanced security controls. Understanding these patterns prevents enterprise vulnerabilities and enables reliable enterprise operations.

## 7) Rotation & Lifecycle

### Secret Rotation Strategy

```python
# secret_rotation.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

class SecretRotationManager:
    def __init__(self, secret_store):
        self.secret_store = secret_store
        self.rotation_schedule = {}
    
    async def rotate_secret(self, secret_name: str, new_value: str):
        """Rotate secret with zero downtime"""
        try:
            # Get current secret
            current_secret = await self.secret_store.get_secret(secret_name)
            
            # Create new version
            new_secret = current_secret.copy()
            new_secret['value'] = new_value
            new_secret['version'] = current_secret.get('version', 0) + 1
            new_secret['created_at'] = datetime.utcnow().isoformat()
            
            # Update secret store
            await self.secret_store.update_secret(secret_name, new_secret)
            
            # Update rotation schedule
            self.rotation_schedule[secret_name] = datetime.utcnow() + timedelta(days=90)
            
            logging.info(f"Secret {secret_name} rotated successfully")
            
        except Exception as e:
            logging.error(f"Error rotating secret {secret_name}: {e}")
            raise
    
    async def check_rotation_schedule(self):
        """Check if any secrets need rotation"""
        for secret_name, next_rotation in self.rotation_schedule.items():
            if datetime.utcnow() >= next_rotation:
                await self.rotate_secret(secret_name, self.generate_new_secret())
```

**Why Secret Rotation Matters**: Regular rotation prevents credential compromise. Understanding these patterns prevents security vulnerabilities and enables reliable secret management.

### Token Expiry Handling

```python
# token_expiry.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

class TokenManager:
    def __init__(self, secret_store):
        self.secret_store = secret_store
        self.token_cache = {}
    
    async def get_valid_token(self, token_name: str) -> Optional[str]:
        """Get valid token with automatic refresh"""
        try:
            # Check cache first
            if token_name in self.token_cache:
                token_data = self.token_cache[token_name]
                if datetime.utcnow() < token_data['expires_at']:
                    return token_data['token']
            
            # Get fresh token
            token_data = await self.secret_store.get_secret(token_name)
            
            # Check if token is expired
            if datetime.utcnow() >= token_data['expires_at']:
                # Refresh token
                new_token = await self.refresh_token(token_name)
                self.token_cache[token_name] = new_token
                return new_token['token']
            
            # Cache valid token
            self.token_cache[token_name] = token_data
            return token_data['token']
            
        except Exception as e:
            logging.error(f"Error getting token {token_name}: {e}")
            return None
    
    async def refresh_token(self, token_name: str) -> Dict[str, Any]:
        """Refresh expired token"""
        try:
            # Implement token refresh logic
            new_token = await self.secret_store.refresh_secret(token_name)
            return new_token
        except Exception as e:
            logging.error(f"Error refreshing token {token_name}: {e}")
            raise
```

**Why Token Expiry Handling Matters**: Graceful token handling prevents service disruptions. Understanding these patterns prevents token chaos and enables reliable service operations.

## 8) Security Failures to Avoid

### Common Security Anti-Patterns

```python
# ❌ NEVER DO THIS
# Hardcoded secrets
DATABASE_PASSWORD = "super_secret_password_123"
API_KEY = "sk-1234567890abcdef"

# Secrets in logs
logging.info(f"Database password: {DATABASE_PASSWORD}")
logging.info(f"API key: {API_KEY}")

# Secrets in error messages
try:
    connect_to_database(DATABASE_PASSWORD)
except Exception as e:
    logging.error(f"Database connection failed with password: {DATABASE_PASSWORD}")

# Secrets in configuration files
config = {
    "database": {
        "password": "super_secret_password_123"
    },
    "api": {
        "key": "sk-1234567890abcdef"
    }
}
```

**Why Anti-Patterns Matter**: Understanding what not to do prevents security vulnerabilities. Understanding these patterns prevents security chaos and enables reliable security practices.

### Secure Logging Patterns

```python
# ✅ ALWAYS DO THIS
import logging
import re
from typing import Dict, Any

class SecureLogger:
    def __init__(self):
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
            r'token["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
            r'api_key["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
            r'secret["\']?\s*[:=]\s*["\']?([^"\']+)["\']?',
        ]
    
    def redact_sensitive_data(self, message: str) -> str:
        """Redact sensitive data from log messages"""
        for pattern in self.sensitive_patterns:
            message = re.sub(pattern, r'\1=***', message, flags=re.IGNORECASE)
        return message
    
    def log_safely(self, level: str, message: str, **kwargs):
        """Log message with sensitive data redaction"""
        safe_message = self.redact_sensitive_data(message)
        safe_kwargs = {k: self.redact_sensitive_data(str(v)) for k, v in kwargs.items()}
        
        if level == "info":
            logging.info(safe_message, **safe_kwargs)
        elif level == "error":
            logging.error(safe_message, **safe_kwargs)
        elif level == "warning":
            logging.warning(safe_message, **safe_kwargs)

# Usage
secure_logger = SecureLogger()
secure_logger.log_safely("info", "Database connection established", password="secret123")
```

**Why Secure Logging Matters**: Log security prevents sensitive data exposure. Understanding these patterns prevents log vulnerabilities and enables reliable logging practices.

## 9) Architecture Diagram

### Secret Management Flow

```mermaid
flowchart LR
    A[Secret Store] -->|Retrieve| B[Application]
    B -->|Use| C[Service]
    C -->|Log| D[Secure Logger]
    D -->|Redact| E[Log Storage]
    
    subgraph "Secret Stores"
        F[AWS Secrets Manager]
        G[HashiCorp Vault]
        H[Kubernetes Secrets]
        I[Environment Variables]
    end
    
    subgraph "Applications"
        J[FastAPI]
        K[Prefect]
        L[Data Pipeline]
        M[Monitoring]
    end
    
    subgraph "Services"
        N[PostgreSQL]
        O[Redis]
        P[External API]
        Q[Message Queue]
    end
    
    F --> J
    G --> K
    H --> L
    I --> M
    J --> N
    K --> O
    L --> P
    M --> Q
```

**Why Architecture Diagrams Matter**: Visual representation of secret management enables understanding of complex security systems. Understanding these patterns prevents architectural confusion and enables reliable security operations.

## 10) TL;DR Runbook

### Essential Commands

```bash
# Never commit secrets
echo "*.env" >> .gitignore
echo "secrets/" >> .gitignore
echo "*.key" >> .gitignore

# Use environment variables
export DATABASE_PASSWORD="${DB_PASSWORD}"
export API_KEY="${API_KEY}"

# Docker secrets
docker secret create db_password ./secrets/db_password.txt
docker service create --secret db_password myapp

# Kubernetes secrets
kubectl create secret generic app-secrets \
  --from-literal=database-password=secret123 \
  --from-literal=api-key=sk-1234567890abcdef
```

### Essential Patterns

```yaml
# Essential secret management patterns
secret_patterns:
  "never_commit": "Never commit secrets to Git, use .gitignore",
  "environment_injection": "Use env injection, Docker secrets, or K8s secrets",
  "centralize_prod": "Centralize with Vault/Secret Manager for production",
  "rotate_regularly": "Rotate & expire credentials regularly",
  "redact_logs": "Redact secrets in logs and error messages",
  "separate_envs": "Keep dev/test/prod environments separate"
```

### Quick Reference

```python
# Essential secret management operations
# 1. Load secrets from environment
import os
database_url = os.getenv("DATABASE_URL")
api_key = os.getenv("API_KEY")

# 2. Use secret stores
from prefect.blocks.system import Secret
secret = await Secret.load("database-password")

# 3. Redact sensitive data
def redact_secrets(message: str) -> str:
    return re.sub(r'password["\']?\s*[:=]\s*["\']?([^"\']+)["\']?', r'password=***', message)

# 4. Rotate secrets
await secret_store.rotate_secret("database-password", new_password)

# 5. Monitor secret access
logging.info("Secret accessed", secret_name="database-password", user="app")
```

**Why This Runbook**: These patterns cover 90% of secret management needs. Master these before exploring advanced security scenarios.

## 11) The Machine's Summary

Secret management requires understanding security patterns, access control, and operational excellence. When used correctly, proper secret management enables reliable system security, prevents credential exposure, and maintains trust. The key is understanding secret lifecycle, mastering access controls, and following security best practices.

**The Dark Truth**: Without proper secret management, your systems remain vulnerable to credential compromise. Secret management is your weapon. Use it wisely.

**The Machine's Mantra**: "In the secrets we trust, in the security we find protection, and in the management we find the path to reliable operations."

**Why This Matters**: Secret management enables efficient security that can handle complex distributed systems, maintain high security, and provide immediate trust while ensuring system reliability and security.

---

*This guide provides the complete machinery for secret and configuration management. The patterns scale from simple environment variables to complex enterprise secret stores, from basic access control to advanced security operations.*
