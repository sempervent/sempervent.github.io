# Secrets Management Best Practices (2025 Edition)

**Objective**: Secrets are dangerous. They leak, they rot, they get copied into Slack and Git commits. Here's how to lock them down without breaking your developer flow.

Secrets are dangerous. They leak, they rot, they get copied into Slack and Git commits. Here's how to lock them down without breaking your developer flow.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Never commit secrets to Git**
   - Use scanners like trufflehog or gitleaks
   - Implement pre-commit hooks
   - Scan repositories regularly
   - Rotate any exposed secrets immediately

2. **Use external secret stores**
   - HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager
   - SOPS for encrypted secret-as-code workflows
   - Doppler/1Password for developer-friendly workflows
   - Avoid environment variables for production secrets

3. **Rotate secrets regularly**
   - Set expiration dates on all tokens
   - Implement automated rotation where possible
   - Monitor for expired secrets
   - Have a rotation playbook ready

4. **Log redaction is mandatory**
   - Secrets should never appear in logs
   - Implement log sanitization
   - Use structured logging with redaction
   - Monitor for secret leakage

5. **Principle of least privilege**
   - Tokens should do one job, not all jobs
   - Use service-specific credentials
   - Implement proper RBAC
   - Regular access reviews

**Why These Principles**: Secrets management requires understanding security risks, access patterns, and lifecycle management. Understanding these patterns prevents security breaches and enables reliable secret handling.

## 1) Core Principles

### The Reality of Secrets

```yaml
# What secrets think they are
secrets_fantasy:
  "security": "I'm secure because I'm in a .env file"
  "access": "Only my team can see me"
  "lifecycle": "I'll live forever"
  "audit": "Nobody needs to know I exist"

# What secrets actually are
secrets_reality:
  "security": "I'm only as secure as my weakest link"
  "access": "Everyone with repo access can see me"
  "lifecycle": "I expire and break everything"
  "audit": "Every access is logged and monitored"
```

**Why Reality Checks Matter**: Understanding the true nature of secrets enables proper security practices and risk management. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Secret Classification

```markdown
## Secret Classification Levels

### Level 1: Public (Not Actually Secret)
- API endpoints
- Public keys
- Documentation URLs
- Configuration values

### Level 2: Internal (Team Access)
- Database connection strings
- Internal API keys
- Development credentials
- Test environment secrets

### Level 3: Sensitive (Limited Access)
- Production database passwords
- Third-party API keys
- Encryption keys
- Service account credentials

### Level 4: Critical (Minimal Access)
- Master encryption keys
- Root account credentials
- Security certificates
- Audit logs
```

**Why Classification Matters**: Proper secret classification enables appropriate security controls and access management. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 2) Secrets in Local Development

### Environment Variables (Development Only)

```python
# .env (NEVER commit this file)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
API_KEY=sk_test_1234567890abcdef
REDIS_URL=redis://localhost:6379/0
DEBUG=True

# .env.example (Commit this file)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
API_KEY=your_api_key_here
REDIS_URL=redis://localhost:6379/0
DEBUG=True
```

**Why Environment Variables Matter**: Local development requires easy secret management without compromising security. Understanding these patterns prevents development chaos and enables reliable secret handling.

### Python Secret Loading

```python
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings, Field

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    database_url: str = Field(..., env="DATABASE_URL")
    api_key: str = Field(..., env="API_KEY")
    redis_url: str = Field(..., env="REDIS_URL")
    debug: bool = Field(False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
settings = Settings()
```

**Why Python Secret Loading Matters**: Proper secret loading enables secure configuration management in Python applications. Understanding these patterns prevents configuration chaos and enables reliable secret handling.

### Rust Secret Loading

```rust
// Cargo.toml
[dependencies]
dotenvy = "0.15"
serde = { version = "1.0", features = ["derive"] }

// config.rs
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub api_key: String,
    pub redis_url: String,
    pub debug: bool,
}

impl Config {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        dotenvy::dotenv()?;
        
        Ok(Config {
            database_url: env::var("DATABASE_URL")?,
            api_key: env::var("API_KEY")?,
            redis_url: env::var("REDIS_URL")?,
            debug: env::var("DEBUG").unwrap_or_default().parse()?,
        })
    }
}
```

**Why Rust Secret Loading Matters**: Proper secret loading enables secure configuration management in Rust applications. Understanding these patterns prevents configuration chaos and enables reliable secret handling.

## 3) Secrets in Docker & Compose

### Bad Patterns (Don't Do This)

```yaml
# docker-compose.yml (BAD - Never do this)
version: "3.9"
services:
  api:
    image: my-api
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - API_KEY=sk_live_1234567890abcdef
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
```

**Why Bad Patterns Matter**: Understanding what not to do prevents security vulnerabilities and secret exposure. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Better Patterns

```yaml
# docker-compose.yml (Better)
version: "3.9"
services:
  api:
    image: my-api
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

**Why Better Patterns Matter**: Proper Docker secret management enables secure containerized applications. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Best Patterns (External Secret Stores)

```yaml
# docker-compose.yml (Best)
version: "3.9"
services:
  api:
    image: my-api
    environment:
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN_FILE=/run/secrets/vault_token
    secrets:
      - vault_token
    depends_on:
      - vault

  vault:
    image: vault:latest
    ports:
      - "8200:8200"
    environment:
      - VAULT_DEV_ROOT_TOKEN_ID=myroot
      - VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200
    command: ["vault", "server", "-dev"]

secrets:
  vault_token:
    file: ./secrets/vault_token.txt
```

**Why Best Patterns Matter**: External secret stores enable centralized secret management and rotation. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 4) Secrets in Kubernetes

### Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAZGI6NTQzMi9teWRi
  api-key: c2tfbGl2ZV8xMjM0NTY3ODkwYWJjZGVm
  redis-url: cmVkaXM6Ly9yZWRpczozNjc5LzA=

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: my-api:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: api-key
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
```

**Why Kubernetes Secrets Matter**: Proper Kubernetes secret management enables secure container orchestration. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Sealed Secrets

```yaml
# sealed-secret.yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: app-secrets
  namespace: default
spec:
  encryptedData:
    database-url: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
    api-key: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
    redis-url: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
```

**Why Sealed Secrets Matter**: Sealed Secrets enable GitOps workflows with encrypted secret storage. Understanding these patterns prevents security chaos and enables reliable secret handling.

### External Secrets Operator

```yaml
# external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "my-role"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: app-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: app/database
      property: url
  - secretKey: api-key
    remoteRef:
      key: app/api
      property: key
```

**Why External Secrets Matter**: External Secrets Operator enables centralized secret management with automatic synchronization. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 5) Application Integration

### Python with Vault

```python
# vault_client.py
import hvac
import os
from typing import Optional

class VaultClient:
    def __init__(self, vault_addr: str, vault_token: str):
        self.client = hvac.Client(url=vault_addr, token=vault_token)
    
    def get_secret(self, path: str, key: str) -> Optional[str]:
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            print(f"Failed to get secret: {e}")
            return None

# Usage
vault_client = VaultClient(
    vault_addr=os.getenv("VAULT_ADDR"),
    vault_token=os.getenv("VAULT_TOKEN")
)

database_url = vault_client.get_secret("app/database", "url")
api_key = vault_client.get_secret("app/api", "key")
```

**Why Vault Integration Matters**: Vault integration enables centralized secret management with audit trails. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Prefect Secrets

```python
# prefect_secrets.py
from prefect import flow, task
from prefect.blocks.system import Secret

@task
def get_database_url():
    return Secret.load("database-url").get()

@task
def get_api_key():
    return Secret.load("api-key").get()

@flow
def my_flow():
    db_url = get_database_url()
    api_key = get_api_key()
    
    # Use secrets in your flow
    print(f"Connecting to database: {db_url}")
    print(f"Using API key: {api_key[:10]}...")

if __name__ == "__main__":
    my_flow()
```

**Why Prefect Secrets Matter**: Prefect secrets enable secure workflow orchestration with centralized secret management. Understanding these patterns prevents security chaos and enables reliable secret handling.

### FastAPI with Dependency Injection

```python
# fastapi_secrets.py
from fastapi import FastAPI, Depends
from pydantic import BaseSettings
from typing import Optional

class SecretSettings(BaseSettings):
    database_url: str
    api_key: str
    redis_url: str
    
    class Config:
        env_file = ".env"

def get_secrets() -> SecretSettings:
    return SecretSettings()

app = FastAPI()

@app.get("/")
async def root(secrets: SecretSettings = Depends(get_secrets)):
    return {
        "database": secrets.database_url,
        "api_key": secrets.api_key[:10] + "...",
        "redis": secrets.redis_url
    }
```

**Why FastAPI Secrets Matter**: FastAPI dependency injection enables secure API development with proper secret management. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 6) Centralized Secret Stores

### HashiCorp Vault

```bash
# Vault setup
vault server -dev

# Create secrets
vault kv put secret/app/database url="postgresql://user:pass@db:5432/mydb"
vault kv put secret/app/api key="sk_live_1234567890abcdef"

# Read secrets
vault kv get secret/app/database
vault kv get secret/app/api
```

**Why Vault Matters**: Vault enables centralized secret management with audit trails and rotation. Understanding these patterns prevents security chaos and enables reliable secret handling.

### AWS Secrets Manager

```python
# aws_secrets.py
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret("app/database")
database_url = secrets['url']
api_key = secrets['key']
```

**Why AWS Secrets Manager Matters**: AWS Secrets Manager enables cloud-native secret management with automatic rotation. Understanding these patterns prevents security chaos and enables reliable secret handling.

### SOPS (Secrets OPerationS)

```yaml
# secrets.yaml (encrypted)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAZGI6NTQzMi9teWRi
  api-key: c2tfbGl2ZV8xMjM0NTY3ODkwYWJjZGVm
  redis-url: cmVkaXM6Ly9yZWRpczozNjc5LzA=
```

```bash
# Encrypt with SOPS
sops -e -i secrets.yaml

# Decrypt with SOPS
sops -d secrets.yaml
```

**Why SOPS Matters**: SOPS enables encrypted secret storage in Git with proper access control. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 7) Rotation, Auditing & Monitoring

### Secret Rotation

```python
# rotation.py
import boto3
import time
from datetime import datetime, timedelta

class SecretRotator:
    def __init__(self, secret_name: str):
        self.secret_name = secret_name
        self.client = boto3.client('secretsmanager')
    
    def rotate_secret(self):
        # Generate new secret
        new_secret = self.generate_new_secret()
        
        # Update secret in store
        self.client.update_secret(
            SecretId=self.secret_name,
            SecretString=new_secret
        )
        
        # Update application configuration
        self.update_application_config(new_secret)
        
        # Verify new secret works
        self.verify_secret(new_secret)
    
    def generate_new_secret(self) -> str:
        # Generate new secret logic
        pass
    
    def update_application_config(self, new_secret: str):
        # Update application configuration
        pass
    
    def verify_secret(self, secret: str) -> bool:
        # Verify secret works
        return True
```

**Why Secret Rotation Matters**: Regular secret rotation prevents long-term exposure and reduces security risks. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Auditing Secret Access

```python
# audit.py
import boto3
from datetime import datetime, timedelta

class SecretAuditor:
    def __init__(self):
        self.client = boto3.client('cloudtrail')
    
    def audit_secret_access(self, secret_name: str, days: int = 30):
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        response = self.client.lookup_events(
            StartTime=start_time,
            EndTime=end_time,
            LookupAttributes=[
                {
                    'AttributeKey': 'ResourceName',
                    'AttributeValue': secret_name
                }
            ]
        )
        
        return response['Events']
    
    def generate_access_report(self, secret_name: str):
        events = self.audit_secret_access(secret_name)
        
        report = {
            'secret_name': secret_name,
            'total_accesses': len(events),
            'unique_users': len(set(event['Username'] for event in events)),
            'access_times': [event['EventTime'] for event in events]
        }
        
        return report
```

**Why Auditing Matters**: Secret access auditing enables security monitoring and compliance. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Monitoring Secret Leakage

```python
# monitoring.py
import requests
import re
from typing import List

class SecretLeakMonitor:
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.headers = {'Authorization': f'token {github_token}'}
    
    def scan_repository(self, owner: str, repo: str) -> List[dict]:
        # Search for potential secrets
        secrets = []
        
        # Search for API keys
        api_key_pattern = r'[a-zA-Z0-9]{32,}'
        response = requests.get(
            f'https://api.github.com/search/code?q={api_key_pattern}+repo:{owner}/{repo}',
            headers=self.headers
        )
        
        if response.status_code == 200:
            results = response.json()
            for item in results['items']:
                secrets.append({
                    'type': 'api_key',
                    'file': item['path'],
                    'url': item['html_url']
                })
        
        return secrets
    
    def alert_on_leak(self, secrets: List[dict]):
        if secrets:
            # Send alert to security team
            print(f"ALERT: {len(secrets)} potential secrets found!")
            for secret in secrets:
                print(f"  - {secret['type']} in {secret['file']}")
```

**Why Monitoring Matters**: Secret leakage monitoring enables early detection of security breaches. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 8) Anti-Patterns

### Common Security Mistakes

```yaml
# What NOT to do
anti_patterns:
  "slack_secrets": "Never share secrets in Slack/Teams"
  "git_commits": "Never commit secrets to Git"
  "hardcoded_secrets": "Never hardcode secrets in code"
  "shared_secrets": "Never use the same secret for multiple services"
  "long_lived_tokens": "Never use tokens without expiration"
  "copy_paste_secrets": "Never copy production secrets to staging"
  "docker_images": "Never bake secrets into Docker images"
  "log_secrets": "Never log secrets in application logs"
  "env_files": "Never commit .env files to Git"
  "default_secrets": "Never use default passwords or keys"
```

**Why Anti-Patterns Matter**: Understanding common mistakes prevents security vulnerabilities and secret exposure. Understanding these patterns prevents security chaos and enables reliable secret handling.

### Security Checklist

```markdown
## Security Checklist

### Development
- [ ] Never commit secrets to Git
- [ ] Use .env files for local development only
- [ ] Implement pre-commit hooks
- [ ] Use dummy secrets in tests
- [ ] Scan repositories regularly

### Production
- [ ] Use external secret stores
- [ ] Implement secret rotation
- [ ] Monitor secret access
- [ ] Use least privilege access
- [ ] Audit secret usage

### Monitoring
- [ ] Set up secret leakage alerts
- [ ] Monitor for exposed secrets
- [ ] Track secret access patterns
- [ ] Implement anomaly detection
- [ ] Regular security reviews
```

**Why Security Checklists Matter**: Systematic security practices enable comprehensive secret protection and risk management. Understanding these patterns prevents security chaos and enables reliable secret handling.

## 9) TL;DR Runbook

### Essential Commands

```bash
# Scan for secrets in Git
trufflehog git file://. --no-verification

# Check for secrets in Docker images
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  trufflesecurity/trufflehog:latest docker --image my-app:latest

# Rotate secrets
vault kv put secret/app/database url="new_url"

# Monitor secret access
vault audit list
```

### Essential Patterns

```yaml
# Essential secrets management patterns
secrets_patterns:
  "never_commit": "Never commit secrets to Git",
  "external_stores": "Use external secret stores (Vault, AWS SM, etc.)",
  "rotate_regularly": "Rotate and audit regularly",
  "sealed_secrets": "Use sealed/encrypted secrets in GitOps",
  "treat_radioactive": "Treat secrets like radioactive waste: short-lived, contained, and monitored",
  "least_privilege": "Use principle of least privilege",
  "log_redaction": "Implement log redaction",
  "access_monitoring": "Monitor secret access patterns",
  "automated_rotation": "Implement automated secret rotation",
  "security_scanning": "Regular security scanning and monitoring"
```

### Quick Reference

```markdown
## Emergency Secret Response

### If Secret is Exposed
1. **Immediately rotate the secret**
2. **Revoke access for the exposed secret**
3. **Scan for other exposures**
4. **Notify security team**
5. **Update incident response plan**

### If Secret is Compromised
1. **Rotate all related secrets**
2. **Review access logs**
3. **Check for unauthorized access**
4. **Implement additional monitoring**
5. **Conduct security review**
```

**Why This Runbook**: These patterns cover 90% of secrets management needs. Master these before exploring advanced security scenarios.

## 10) The Machine's Summary

Secrets management requires understanding security risks, access patterns, and lifecycle management. When used correctly, proper secrets management enables secure applications, prevents security breaches, and provides insights into access patterns. The key is understanding secret classification, external stores, rotation, and monitoring.

**The Dark Truth**: Without proper secrets management, your applications remain vulnerable and your data exposed. Secrets management is your weapon. Use it wisely.

**The Machine's Mantra**: "In the external stores we trust, in the rotation we find security, and in the monitoring we find the path to bulletproof secret handling."

**Why This Matters**: Secrets management enables secure applications that can handle sensitive data, prevent security breaches, and provide insights into access patterns while ensuring technical accuracy and reliability.

---

*This guide provides the complete machinery for secrets management. The patterns scale from simple environment variables to complex enterprise secret stores, from basic security to advanced threat protection.*
