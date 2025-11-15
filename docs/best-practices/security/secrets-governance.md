# End-to-End Secrets Management & Key Rotation Governance: Best Practices for Infrastructure, Applications, and CI/CD

**Objective**: Master production-grade secrets management across hybrid multi-cloud and air-gapped environments. When you need to secure credentials, rotate keys safely, inject secrets into workloads, and maintain compliance—this guide provides complete patterns and automation.

## Introduction

Secrets management is the foundation of infrastructure security. This guide covers the complete lifecycle: creation, storage, distribution, injection, rotation, revocation, and auditing. It addresses both connected and air-gapped environments, supporting Kubernetes (RKE2), Docker Compose, bare-metal services, and CI/CD pipelines.

**What This Guide Covers**:
- Secret storage options (Vault, SOPS, KMS, Kubernetes Secrets)
- Injection patterns for all workload types
- Automated key rotation strategies
- CI/CD integration (GitHub Actions, GitLab CI)
- Air-gapped environment patterns
- Logging, auditing, and incident response
- Reference implementations

**Prerequisites**:
- Understanding of Kubernetes, Docker, and CI/CD basics
- Familiarity with infrastructure as code (Terraform, Ansible)
- Access to infrastructure for implementation

## Scope & Goals

### Complete Lifecycle Coverage

This guide addresses every phase of secrets management:

1. **Creation**: How secrets are generated (random passwords, PKI certs, API keys)
2. **Storage**: Where secrets are stored (Vault, SOPS, KMS, encrypted at rest)
3. **Distribution**: How secrets reach workloads (injection, sidecars, env vars)
4. **Injection**: How applications consume secrets (files, env vars, API calls)
5. **Expiration**: How secrets expire and trigger rotation
6. **Rotation**: How secrets are rotated without downtime
7. **Revocation**: How compromised secrets are invalidated
8. **Auditing**: How secret access is logged and monitored

### Integration Requirements

**Must Support**:
- **RKE2 Kubernetes clusters** with Rancher management
- **Docker Compose** for local and edge deployments
- **Terraform/Ansible** for infrastructure as code
- **Air-gapped environments** with no external API access
- **Machine identities** (service-to-service auth, not just human users)

**Workload Types**:
- PostgreSQL databases (primary/replica clusters)
- Redis clusters
- MLflow tracking servers
- Grafana/Prometheus/Loki observability stack
- NGINX reverse proxies
- Internal APIs and microservices
- CI/CD pipelines (GitHub Actions, GitLab CI)

### Non-Goals

- **Not a basic "don't commit secrets" tutorial**: Assumes you already know this
- **Not vendor-specific**: Covers multiple approaches (Vault, SOPS, KMS, native K8s)
- **Not theoretical**: Every pattern includes working examples

## Core Architecture

### System Overview

```mermaid
graph TB
    subgraph Sources["Secret Sources"]
        Vault["HashiCorp Vault"]
        SOPS["SOPS + Git"]
        KMS["Cloud KMS<br/>(AWS/GCP/Azure)"]
        K8S["Kubernetes Secrets<br/>(encrypted at rest)"]
        ESO["External Secrets<br/>Operator"]
    end
    
    subgraph Rotation["Rotation Controllers"]
        Cron["Cron Jobs<br/>(pg_cron, systemd)"]
        Ansible["Ansible Playbooks"]
        GitOps["GitOps Operators<br/>(Flux, ArgoCD)"]
        CertMgr["cert-manager<br/>(ACME, CA)"]
    end
    
    subgraph Platforms["Workload Platforms"]
        RKE2["RKE2 Cluster"]
        Rancher["Rancher UI"]
        Compose["Docker Compose"]
        BareMetal["Bare Metal Services"]
        CI["CI/CD Pipelines"]
    end
    
    subgraph Workloads["Workloads"]
        PG["PostgreSQL"]
        Redis["Redis"]
        MLflow["MLflow"]
        Grafana["Grafana"]
        NGINX["NGINX"]
        APIs["Internal APIs"]
    end
    
    subgraph Audit["Audit & Monitoring"]
        Loki["Loki Logs"]
        Prom["Prometheus Metrics"]
        GrafanaDash["Grafana Dashboards"]
        Alerts["Alertmanager"]
    end
    
    Vault --> ESO
    SOPS --> GitOps
    KMS --> ESO
    K8S --> RKE2
    
    ESO --> RKE2
    GitOps --> RKE2
    CertMgr --> RKE2
    
    Cron --> PG
    Cron --> Redis
    Ansible --> RKE2
    Ansible --> BareMetal
    
    RKE2 --> PG
    RKE2 --> Redis
    RKE2 --> MLflow
    RKE2 --> Grafana
    RKE2 --> NGINX
    RKE2 --> APIs
    
    Compose --> PG
    Compose --> Redis
    Compose --> APIs
    
    BareMetal --> PG
    BareMetal --> Redis
    
    CI --> Vault
    CI --> SOPS
    CI --> KMS
    
    PG --> Loki
    Redis --> Loki
    RKE2 --> Prom
    ESO --> Prom
    CertMgr --> Prom
    
    Loki --> GrafanaDash
    Prom --> GrafanaDash
    Prom --> Alerts
```

### Data Flow

1. **Secret Creation**: Secrets generated in Vault, SOPS, or KMS
2. **Storage**: Encrypted storage (Vault transit, SOPS encryption, KMS encryption)
3. **Distribution**: External Secrets Operator or GitOps syncs to Kubernetes
4. **Injection**: Pods mount secrets as volumes or env vars
5. **Rotation**: Cron jobs or operators trigger rotation workflows
6. **Audit**: All access logged to Loki, metrics to Prometheus

## Storage Options

### HashiCorp Vault

**Use Case**: Enterprise-grade secrets management with dynamic secrets, PKI, and advanced features.

**Architecture**:

```mermaid
graph LR
    subgraph Vault["Vault Cluster"]
        Transit["Transit Engine<br/>(Encryption)"]
        PKI["PKI Engine<br/>(Certificates)"]
        Database["Database Engine<br/>(Dynamic Secrets)"]
        KV["KV v2<br/>(Static Secrets)"]
    end
    
    Apps["Applications"] --> Transit
    Apps --> PKI
    Apps --> Database
    Apps --> KV
    
    Transit --> KMS["KMS Backend"]
    PKI --> CA["CA Backend"]
    Database --> PG["Postgres"]
```

**Configuration Example**:

```hcl
# vault.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}

api_addr = "https://vault.example.com:8200"
ui = true

# Enable audit logging
audit_device "file" {
  path = "/var/log/vault/audit.log"
  format = "json"
}
```

**Enable Engines**:

```bash
# Enable transit engine for encryption
vault secrets enable transit

# Enable PKI engine
vault secrets enable pki
vault secrets tune -max-lease-ttl=87600h pki

# Enable database secrets engine
vault secrets enable database

# Configure Postgres dynamic secrets
vault write database/config/postgres \
  plugin_name=postgresql-database-plugin \
  allowed_roles="readonly,readwrite" \
  connection_url="postgresql://{{username}}:{{password}}@postgres:5432/mydb" \
  username="vault" \
  password="vault-password"
```

**Pros**:
- Dynamic secrets (auto-rotating database credentials)
- PKI engine for certificate management
- Transit engine for encryption as a service
- Fine-grained access policies
- Audit logging built-in

**Cons**:
- Operational complexity
- Requires HA setup for production
- Learning curve for teams

### SOPS + Git (Encrypted GitOps)

**Use Case**: GitOps workflows where secrets are committed to Git in encrypted form.

**Architecture**:

```mermaid
graph LR
    Dev["Developer"] --> SOPS["SOPS<br/>(encrypt)"]
    SOPS --> Git["Git Repo<br/>(encrypted secrets)"]
    Git --> GitOps["GitOps Operator<br/>(Flux/ArgoCD)"]
    GitOps --> K8S["Kubernetes<br/>(decrypted)"]
    
    SOPS -.-> AgeKey["Age Key<br/>(private key)"]
    GitOps -.-> AgeKey
```

**Setup**:

```bash
# Install SOPS
brew install sops  # macOS
# or
wget https://github.com/mozilla/sops/releases/download/v3.8.1/sops-v3.8.1.linux
sudo mv sops-v3.8.1.linux /usr/local/bin/sops
sudo chmod +x /usr/local/bin/sops

# Generate Age key pair
age-keygen -o age-key.txt
# Public key: age1abc123...
# Private key: age-secret-key-1abc123...

# Create .sops.yaml
cat > .sops.yaml <<EOF
creation_rules:
  - path_regex: secrets/.*\.yaml$
    age: age1abc123...  # Public key
  - path_regex: secrets/.*\.env$
    age: age1abc123...
EOF
```

**Encrypt Secret**:

```bash
# Create secret file
cat > secrets/postgres-credentials.yaml <<EOF
postgres:
  username: myapp
  password: super-secret-password
  host: postgres.example.com
  port: 5432
EOF

# Encrypt with SOPS
sops -e -i secrets/postgres-credentials.yaml

# File is now encrypted
cat secrets/postgres-credentials.yaml
# Output: encrypted binary/JSON
```

**Decrypt in CI/CD**:

```yaml
# .github/workflows/deploy.yml
name: Deploy Secrets

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install SOPS
        run: |
          wget https://github.com/mozilla/sops/releases/download/v3.8.1/sops-v3.8.1.linux
          sudo mv sops-v3.8.1.linux /usr/local/bin/sops
          sudo chmod +x /usr/local/bin/sops
      
      - name: Decrypt secrets
        env:
          SOPS_AGE_KEY: ${{ secrets.SOPS_AGE_KEY }}
        run: |
          sops -d secrets/postgres-credentials.yaml > decrypted.yaml
          kubectl apply -f decrypted.yaml
```

**Pros**:
- GitOps-friendly (secrets in Git)
- No external service dependency
- Works in air-gapped environments
- Simple key management (Age keys)

**Cons**:
- Manual rotation of encryption keys
- No dynamic secrets
- Requires key distribution to CI/CD systems

### Kubernetes Secrets (with Encryption at Rest)

**Use Case**: Native Kubernetes secrets with encryption at rest enabled.

**Enable Encryption at Rest**:

```yaml
# EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}  # Fallback
```

**Apply Configuration**:

```bash
# Generate encryption key
ENCRYPTION_KEY=$(head -c 32 /dev/urandom | base64)

# Update EncryptionConfiguration
kubectl create secret generic encryption-config \
  --from-file=encryption.yaml=encryption-config.yaml \
  -n kube-system

# Update kube-apiserver manifest
# Add: --encryption-provider-config=/etc/kubernetes/encryption/encryption.yaml
```

**Create Secret**:

```yaml
# postgres-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
  namespace: production
type: Opaque
stringData:
  username: myapp
  password: super-secret-password
  host: postgres.example.com
  port: "5432"
```

**Pros**:
- Native Kubernetes integration
- No external dependencies
- Encryption at rest available

**Cons**:
- Base64 encoding (not encryption) by default
- Requires encryption at rest for security
- No automatic rotation
- Secrets visible to anyone with cluster access

### Docker Secrets

**Use Case**: Docker Swarm or Docker Compose with secrets.

**Create Secret**:

```bash
# Create secret
echo "super-secret-password" | docker secret create postgres_password -

# Or from file
docker secret create postgres_password ./password.txt
```

**Use in Docker Compose**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    image: myapp:latest
    secrets:
      - postgres_password
      - postgres_username
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
      - POSTGRES_USERNAME_FILE=/run/secrets/postgres_username

secrets:
  postgres_password:
    external: true
  postgres_username:
    file: ./username.txt  # Plain text (not recommended for production)
```

**Access in Container**:

```bash
# Secret mounted at /run/secrets/<secret-name>
cat /run/secrets/postgres_password
```

**Pros**:
- Native Docker integration
- Secrets not in image layers
- Simple for Docker Compose

**Cons**:
- Docker Swarm only (not standard Docker)
- Limited rotation capabilities
- No encryption at rest by default

### External Secrets Operator (ESO)

**Use Case**: Sync secrets from external sources (Vault, AWS Secrets Manager, GCP Secret Manager) to Kubernetes.

**Architecture**:

```mermaid
graph LR
    Vault["Vault"] --> ESO["External Secrets<br/>Operator"]
    AWS["AWS Secrets<br/>Manager"] --> ESO
    GCP["GCP Secret<br/>Manager"] --> ESO
    
    ESO --> SecretStore["SecretStore<br/>(auth config)"]
    SecretStore --> ExternalSecret["ExternalSecret<br/>(sync config)"]
    ExternalSecret --> K8SSecret["Kubernetes<br/>Secret"]
    K8SSecret --> Pod["Pod"]
```

**Install ESO**:

```bash
# Install via Helm
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace
```

**Configure SecretStore**:

```yaml
# vault-secretstore.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: production
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"
          serviceAccountRef:
            name: external-secrets
```

**Create ExternalSecret**:

```yaml
# postgres-external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: postgres-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: postgres-credentials
    creationPolicy: Owner
  data:
    - secretKey: username
      remoteRef:
        key: database/postgres
        property: username
    - secretKey: password
      remoteRef:
        key: database/postgres
        property: password
```

**Pros**:
- Automatic sync from external sources
- Supports multiple backends (Vault, AWS, GCP, Azure)
- Kubernetes-native (CRDs)
- Automatic refresh

**Cons**:
- Requires external secret source
- Additional operator to manage
- Learning curve for CRDs

## Secret Formats

### PKI Certificates

**Use Case**: TLS certificates for services, mTLS for service-to-service auth.

**Certificate Structure**:

```
Certificate Chain:
  - Root CA (long-lived, offline)
  - Intermediate CA (medium-lived, online)
  - Leaf Certificate (short-lived, auto-rotating)
```

**Vault PKI Setup**:

```bash
# Generate root CA
vault secrets enable -path=pki pki
vault secrets tune -max-lease-ttl=87600h pki

vault write -field=certificate pki/root/generate/internal \
  common_name="example.com Root CA" \
  ttl=87600h > ca_cert.pem

# Configure CA and CRL URLs
vault write pki/config/urls \
  issuing_certificates="https://vault.example.com/v1/pki/ca" \
  crl_distribution_points="https://vault.example.com/v1/pki/crl"

# Create intermediate CA
vault secrets enable -path=pki_int pki
vault secrets tune -max-lease-ttl=43800h pki_int

# Generate CSR
vault write -field=csr pki_int/intermediate/generate/internal \
  common_name="example.com Intermediate CA" \
  ttl=43800h > pki_intermediate.csr

# Sign with root CA
vault write -field=certificate pki/root/sign-intermediate \
  csr=@pki_intermediate.csr \
  common_name="example.com Intermediate CA" \
  ttl=43800h > intermediate_cert.pem

# Set intermediate certificate
vault write pki_int/intermediate/set-signed certificate=@intermediate_cert.pem

# Create role for leaf certificates
vault write pki_int/roles/example-dot-com \
  allowed_domains="example.com" \
  allow_subdomains=true \
  max_ttl="720h" \
  key_type="ec" \
  key_bits=256
```

**Issue Leaf Certificate**:

```bash
# Issue certificate
vault write -format=json pki_int/issue/example-dot-com \
  common_name="api.example.com" \
  ttl="24h" | jq -r '.data.certificate' > api.crt

# Get private key
vault write -format=json pki_int/issue/example-dot-com \
  common_name="api.example.com" \
  ttl="24h" | jq -r '.data.private_key' > api.key
```

### JWT Signing Keys

**Use Case**: Service authentication, API tokens.

**Key Rotation Pattern**:

```mermaid
graph LR
    Key1["Key v1<br/>(kid: key-1)"] --> Active["Active Keys<br/>(v1, v2)"]
    Key2["Key v2<br/>(kid: key-2)"] --> Active
    
    Active --> Verify["JWT Verification<br/>(try v2, fallback v1)"]
    
    Key3["Key v3<br/>(kid: key-3)"] --> Rotate["Rotation<br/>(add v3, keep v2)"]
    Rotate --> Active
```

**Key Generation**:

```bash
# Generate RSA key pair
openssl genrsa -out jwt-signing-key-v1.pem 2048
openssl rsa -in jwt-signing-key-v1.pem -pubout -out jwt-signing-key-v1.pub

# Generate EC key pair (smaller, faster)
openssl ecparam -genkey -name secp256r1 -noout -out jwt-signing-key-v2.pem
openssl ec -in jwt-signing-key-v2.pem -pubout -out jwt-signing-key-v2.pub
```

**JWT with Key ID**:

```python
# Python example
import jwt
from cryptography.hazmat.primitives import serialization

# Load keys
with open("jwt-signing-key-v2.pem", "rb") as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

# Sign JWT with key ID
payload = {"sub": "user123", "exp": 1234567890}
token = jwt.encode(
    payload,
    private_key,
    algorithm="ES256",
    headers={"kid": "key-2"}  # Key ID for rotation
)

# Verify (try multiple keys)
keys = {
    "key-1": load_public_key("jwt-signing-key-v1.pub"),
    "key-2": load_public_key("jwt-signing-key-v2.pub"),
}

kid = jwt.get_unverified_header(token)["kid"]
public_key = keys.get(kid)
if public_key:
    payload = jwt.decode(token, public_key, algorithms=["ES256"])
```

### Database Credentials

**PostgreSQL Password**:

```sql
-- Create user with password
CREATE USER myapp WITH PASSWORD 'initial-password';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO myapp;
GRANT USAGE ON SCHEMA public TO myapp;

-- Store in Vault (dynamic secret)
-- Vault will auto-rotate this password
```

**Redis AUTH Token**:

```redis
# redis.conf
requirepass super-secret-token

# Or via command
CONFIG SET requirepass "super-secret-token"
```

### API Keys

**Dual-Key Rotation Pattern**:

```mermaid
graph LR
    App["Application"] --> Key1["API Key v1<br/>(active)"]
    App --> Key2["API Key v2<br/>(active)"]
    
    Rotate["Rotation"] --> Key3["API Key v3<br/>(new)"]
    Rotate --> Revoke["Revoke Key v1"]
    
    Key3 --> App
```

**Implementation**:

```python
# api_key_manager.py
class APIKeyManager:
    def __init__(self):
        self.keys = {
            "v1": "sk_live_abc123...",
            "v2": "sk_live_def456...",
        }
        self.active_keys = ["v1", "v2"]
    
    def validate_key(self, api_key: str) -> bool:
        # Try all active keys
        for key_id, key_value in self.keys.items():
            if key_id in self.active_keys and api_key == key_value:
                return True
        return False
    
    def rotate(self):
        # Generate new key
        new_key_id = f"v{len(self.keys) + 1}"
        new_key = f"sk_live_{generate_random_token()}"
        self.keys[new_key_id] = new_key
        
        # Add to active keys
        self.active_keys.append(new_key_id)
        
        # Remove oldest key after grace period
        if len(self.active_keys) > 2:
            oldest = self.active_keys.pop(0)
            # Revoke after 24h grace period
            schedule_revocation(oldest, delay_hours=24)
```

## Secret Injection Patterns

### Kubernetes Pod-Level Secret Injection

**Volume Mount**:

```yaml
# postgres-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
  namespace: production
spec:
  containers:
    - name: app
      image: myapp:latest
      volumeMounts:
        - name: postgres-secret
          mountPath: /etc/secrets/postgres
          readOnly: true
  volumes:
    - name: postgres-secret
      secret:
        secretName: postgres-credentials
        items:
          - key: username
            path: username
          - key: password
            path: password
```

**Environment Variables**:

```yaml
# postgres-pod-env.yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
  namespace: production
spec:
  containers:
    - name: app
      image: myapp:latest
      env:
        - name: POSTGRES_USERNAME
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
```

**Deployment with Secrets**:

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: app
          image: myapp:latest
          envFrom:
            - secretRef:
                name: postgres-credentials
          # Or use env with secretKeyRef
          env:
            - name: POSTGRES_HOST
              value: postgres.example.com
```

### Sidecar-Based Dynamic Secret Rotation

**Vault Agent Sidecar**:

```yaml
# vault-sidecar-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp-with-vault
  namespace: production
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "myapp"
    vault.hashicorp.com/agent-inject-secret-database: "database/creds/myapp"
    vault.hashicorp.com/agent-inject-template-database: |
      {{- with secret "database/creds/myapp" -}}
      POSTGRES_USERNAME={{ .Data.username }}
      POSTGRES_PASSWORD={{ .Data.password }}
      {{- end }}
spec:
  serviceAccountName: myapp
  containers:
    - name: app
      image: myapp:latest
      env:
        - name: POSTGRES_USERNAME
          valueFrom:
            secretKeyRef:
              name: vault-env
              key: POSTGRES_USERNAME
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: vault-env
              key: POSTGRES_PASSWORD
    - name: vault-agent
      image: hashicorp/vault:latest
      args:
        - "agent"
        - "-config=/vault/config/agent.hcl"
      volumeMounts:
        - name: vault-config
          mountPath: /vault/config
```

### Docker Compose Environment Variable Injection

**Environment File**:

```bash
# .env (DO NOT COMMIT)
POSTGRES_USERNAME=myapp
POSTGRES_PASSWORD=super-secret-password
POSTGRES_HOST=postgres.example.com
POSTGRES_PORT=5432
```

**Docker Compose**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    image: myapp:latest
    env_file:
      - .env
    environment:
      - POSTGRES_HOST=${POSTGRES_HOST}
      - POSTGRES_USERNAME=${POSTGRES_USERNAME}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=${POSTGRES_USERNAME}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

**Docker Secrets (Swarm)**:

```yaml
# docker-compose.secrets.yml
version: '3.8'

services:
  app:
    image: myapp:latest
    secrets:
      - postgres_username
      - postgres_password
    environment:
      - POSTGRES_USERNAME_FILE=/run/secrets/postgres_username
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

secrets:
  postgres_username:
    external: true
  postgres_password:
    external: true
```

### File-Based Credential Injection

**Systemd Service**:

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp
EnvironmentFile=/etc/myapp/secrets.env
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Secrets File**:

```bash
# /etc/myapp/secrets.env (permissions: 600, owner: myapp)
POSTGRES_USERNAME=myapp
POSTGRES_PASSWORD=super-secret-password
POSTGRES_HOST=postgres.example.com
```

### Sealed Secrets (GitOps)

**Install Sealed Secrets Controller**:

```bash
# Install via Helm
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets \
  -n kube-system
```

**Create Sealed Secret**:

```bash
# Create Kubernetes secret
kubectl create secret generic postgres-credentials \
  --from-literal=username=myapp \
  --from-literal=password=super-secret-password \
  --dry-run=client -o yaml > secret.yaml

# Seal it
kubeseal < secret.yaml > sealed-secret.yaml

# Now safe to commit
git add sealed-secret.yaml
git commit -m "Add sealed secret for postgres"
```

**Sealed Secret YAML**:

```yaml
# sealed-secret.yaml (safe to commit)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: postgres-credentials
  namespace: production
spec:
  encryptedData:
    username: AgBx...
    password: AgBy...
  template:
    metadata:
      name: postgres-credentials
      namespace: production
    type: Opaque
```

### Rancher Secret Store Integration

**Rancher Secret**:

```yaml
# Via Rancher UI or API
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
  namespace: production
  labels:
    cattle.io/creator: norman
type: Opaque
stringData:
  username: myapp
  password: super-secret-password
```

**Use in Rancher Workload**:

```yaml
# Rancher workload YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  template:
    spec:
      containers:
        - name: app
          image: myapp:latest
          envFrom:
            - secretRef:
                name: postgres-credentials
```

### Prefect Agent Secrets

**Prefect Secret**:

```python
# Create secret in Prefect
from prefect import flow, task
from prefect.blocks.system import Secret

@task
def get_db_credentials():
    username = Secret.load("postgres-username").get()
    password = Secret.load("postgres-password").get()
    return username, password

@flow
def my_flow():
    username, password = get_db_credentials()
    # Use credentials
    pass
```

**Prefect Deployment with Secrets**:

```yaml
# prefect-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prefect-deployment
data:
  deployment.yaml: |
    name: my-flow
    work_queue_name: k8s-default
    work_pool_name: k8s-work-pool
    job_variables:
      env:
        - name: POSTGRES_USERNAME
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
```

## Key Rotation Strategies

### Postgres Password Rotation (Zero-Downtime)

**Dual-Secret Pattern**:

```mermaid
sequenceDiagram
    participant Admin as Admin/Ansible
    participant PG as PostgreSQL
    participant App1 as App Instance 1
    participant App2 as App Instance 2
    
    Note over Admin,PG: Phase 1: Create New Credential
    Admin->>PG: CREATE USER myapp_v2 WITH PASSWORD 'new-password'
    Admin->>PG: GRANT permissions TO myapp_v2
    
    Note over Admin,App1: Phase 2: Rolling Update (App 1)
    Admin->>App1: Update secret to v2
    App1->>PG: Connect with v2 credentials
    App1->>PG: Verify connection works
    
    Note over Admin,App2: Phase 3: Rolling Update (App 2)
    Admin->>App2: Update secret to v2
    App2->>PG: Connect with v2 credentials
    
    Note over Admin,PG: Phase 4: Cleanup
    Admin->>PG: DROP USER myapp_v1
```

**Ansible Playbook**:

```yaml
# rotate-postgres-password.yml
---
- name: Rotate Postgres Password
  hosts: postgres_servers
  vars:
    app_user: myapp
    new_password: "{{ vault_postgres_new_password }}"
    old_user: "{{ app_user }}_v1"
    new_user: "{{ app_user }}_v2"
  
  tasks:
    - name: Generate new password
      set_fact:
        generated_password: "{{ lookup('password', '/dev/null length=32 chars=ascii_letters,digits') }}"
      when: new_password is not defined
    
    - name: Create new user with new password
      postgresql_user:
        name: "{{ new_user }}"
        password: "{{ new_password | default(generated_password) }}"
        priv: "ALL:public"
        state: present
      become: yes
      become_user: postgres
    
    - name: Grant permissions to new user
      postgresql_privs:
        db: mydb
        privs: ALL
        objs: ALL_IN_SCHEMA
        schema: public
        roles: "{{ new_user }}"
      become: yes
      become_user: postgres
    
    - name: Update Kubernetes secret (v2)
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: v1
          kind: Secret
          metadata:
            name: postgres-credentials-v2
            namespace: production
          type: Opaque
          stringData:
            username: "{{ new_user }}"
            password: "{{ new_password | default(generated_password) }}"
    
    - name: Rolling update deployment (canary)
      kubernetes.core.k8s_scale:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: myapp
            namespace: production
          spec:
            replicas: 1
        wait: true
        wait_condition:
          type: Available
          status: "True"
    
    - name: Update deployment to use v2 secret
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: myapp
            namespace: production
          spec:
            template:
              spec:
                containers:
                  - name: app
                    envFrom:
                      - secretRef:
                          name: postgres-credentials-v2
    
    - name: Scale up deployment
      kubernetes.core.k8s_scale:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: myapp
            namespace: production
          spec:
            replicas: 3
        wait: true
    
    - name: Wait for all pods healthy
      kubernetes.core.k8s_info:
        api_version: v1
        kind: Pod
        namespace: production
        label_selectors:
          - app=myapp
      register: pods
      until: pods.resources | selectattr('status.phase', 'equalto', 'Running') | list | length == 3
      retries: 30
      delay: 10
    
    - name: Drop old user (after grace period)
      postgresql_user:
        name: "{{ old_user }}"
        state: absent
      become: yes
      become_user: postgres
      when: cleanup_old_user | default(false)
```

**PgCron Automated Rotation**:

```sql
-- Function to rotate Postgres password
CREATE OR REPLACE FUNCTION rotate_app_password()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    new_password text;
    new_username text;
    old_username text;
BEGIN
    -- Generate new password
    new_password := encode(gen_random_bytes(32), 'base64');
    new_username := 'myapp_v' || extract(epoch from now())::bigint;
    old_username := (SELECT usename FROM pg_user WHERE usename LIKE 'myapp_v%' ORDER BY usename DESC LIMIT 1);
    
    -- Create new user
    EXECUTE format('CREATE USER %I WITH PASSWORD %L', new_username, new_password);
    EXECUTE format('GRANT ALL PRIVILEGES ON DATABASE mydb TO %I', new_username);
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO %I', new_username);
    
    -- Update Kubernetes secret (via API call or external script)
    -- This would be done by an external service that watches for new users
    
    -- Drop old user after 24h grace period
    -- Scheduled separately via pg_cron
END;
$$;

-- Schedule rotation (monthly)
SELECT cron.schedule(
    'rotate-postgres-password',
    '0 2 1 * *',  -- First day of month at 02:00
    $$SELECT rotate_app_password();$$
);
```

### Redis AUTH Token Rotation

**Rotation Script**:

```bash
#!/bin/bash
# rotate-redis-auth.sh

set -euo pipefail

OLD_TOKEN="${REDIS_OLD_TOKEN}"
NEW_TOKEN=$(openssl rand -base64 32)
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Connect with old token and set new token
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$OLD_TOKEN" \
  CONFIG SET requirepass "$NEW_TOKEN"

# Update Kubernetes secret
kubectl create secret generic redis-credentials \
  --from-literal=password="$NEW_TOKEN" \
  --dry-run=client -o yaml | \
  kubectl apply -f -

# Rolling restart of Redis clients
kubectl rollout restart deployment/myapp -n production

# Wait for rollout
kubectl rollout status deployment/myapp -n production

# After grace period, old token will be invalid
echo "Rotation complete. Old token will expire after grace period."
```

**Dual-Token Support**:

```python
# redis_client.py
import redis
from redis.connection import ConnectionPool

class RotatingRedisClient:
    def __init__(self, primary_token: str, secondary_token: str = None):
        self.primary_token = primary_token
        self.secondary_token = secondary_token
        self.pool = ConnectionPool(
            host='redis.example.com',
            port=6379,
            password=primary_token,
            decode_responses=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
    
    def execute_command(self, *args, **kwargs):
        try:
            return self.client.execute_command(*args, **kwargs)
        except redis.AuthenticationError:
            # Try secondary token if primary fails
            if self.secondary_token:
                self.pool.connection_kwargs['password'] = self.secondary_token
                self.client = redis.Redis(connection_pool=self.pool)
                return self.client.execute_command(*args, **kwargs)
            raise
```

### TLS Certificate Rotation

**Cert-Manager (ACME)**:

```yaml
# cert-manager-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

**Certificate Resource**:

```yaml
# tls-certificate.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: api-example-com
  namespace: production
spec:
  secretName: api-example-com-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - api.example.com
    - www.api.example.com
  renewBefore: 720h  # Renew 30 days before expiry
```

**Manual Certificate Rotation**:

```bash
#!/bin/bash
# rotate-tls-cert.sh

CERT_NAME="api-example-com"
NAMESPACE="production"
SECRET_NAME="${CERT_NAME}-tls"

# Generate new certificate
certbot certonly --standalone -d api.example.com

# Create new secret
kubectl create secret tls "$SECRET_NAME" \
  --cert=/etc/letsencrypt/live/api.example.com/fullchain.pem \
  --key=/etc/letsencrypt/live/api.example.com/privkey.pem \
  --dry-run=client -o yaml | \
  kubectl apply -f -

# Update ingress to use new secret
kubectl patch ingress api-ingress -n "$NAMESPACE" \
  -p '{"spec":{"tls":[{"secretName":"'$SECRET_NAME'"}]}}'

# Rolling restart NGINX
kubectl rollout restart deployment/nginx -n "$NAMESPACE"
kubectl rollout status deployment/nginx -n "$NAMESPACE"
```

**NGINX Certificate Reload**:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/nginx/ssl/api-example-com.crt;
    ssl_certificate_key /etc/nginx/ssl/api-example-com.key;
    
    # Reload certificates without restart
    # Use: nginx -s reload
}
```

```bash
# Reload NGINX after cert update
kubectl exec -n production deployment/nginx -- nginx -s reload
```

### JWT Signing Key Rotation (Kid-Based)

**Key Rotation Function**:

```python
# jwt_key_rotation.py
import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

class JWTKeyManager:
    def __init__(self, key_storage_path: str):
        self.key_storage_path = key_storage_path
        self.keys = self._load_keys()
    
    def _load_keys(self) -> dict:
        """Load all active keys from storage"""
        keys = {}
        for key_file in Path(self.key_storage_path).glob("jwt-key-*.pem"):
            kid = key_file.stem.split("-")[-1]  # Extract kid from filename
            with open(key_file, "rb") as f:
                keys[kid] = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        return keys
    
    def rotate_key(self) -> str:
        """Generate new key and return kid"""
        # Generate new EC key
        private_key = ec.generate_private_key(ec.SECP256R1())
        kid = f"key-{int(time.time())}"
        
        # Save private key
        key_path = Path(self.key_storage_path) / f"jwt-key-{kid}.pem"
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Save public key
        pub_key_path = Path(self.key_storage_path) / f"jwt-key-{kid}.pub"
        public_key = private_key.public_key()
        with open(pub_key_path, "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        
        # Add to active keys
        self.keys[kid] = private_key
        
        return kid
    
    def sign_token(self, payload: dict, kid: str = None) -> str:
        """Sign JWT with specified or latest key"""
        if kid is None:
            kid = max(self.keys.keys())  # Use latest key
        
        private_key = self.keys[kid]
        headers = {"kid": kid, "alg": "ES256"}
        
        return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT, trying all active keys"""
        # Get kid from token
        unverified = jwt.decode(token, options={"verify_signature": False})
        kid = jwt.get_unverified_header(token).get("kid")
        
        # Try kid first, then all keys
        keys_to_try = [kid] if kid else []
        keys_to_try.extend(self.keys.keys())
        
        for key_id in keys_to_try:
            if key_id not in self.keys:
                continue
            
            try:
                # Load public key
                pub_key_path = Path(self.key_storage_path) / f"jwt-key-{key_id}.pub"
                with open(pub_key_path, "rb") as f:
                    public_key = serialization.load_pem_public_key(f.read())
                
                # Verify
                payload = jwt.decode(token, public_key, algorithms=["ES256"])
                return payload
            except jwt.InvalidSignatureError:
                continue
        
        raise jwt.InvalidTokenError("No valid key found")
```

**Automated Rotation**:

```bash
#!/bin/bash
# rotate-jwt-keys.sh

KEY_STORAGE="/etc/jwt/keys"
GRACE_PERIOD_DAYS=7

# Generate new key
python3 -c "
from jwt_key_rotation import JWTKeyManager
manager = JWTKeyManager('$KEY_STORAGE')
new_kid = manager.rotate_key()
print(f'New key created: {new_kid}')
"

# After grace period, remove old keys
find "$KEY_STORAGE" -name "jwt-key-*.pem" -mtime +$GRACE_PERIOD_DAYS \
  -exec rm {} \;
```

### SSH Key Rotation

**Host Key Rotation**:

```bash
#!/bin/bash
# rotate-ssh-host-keys.sh

BACKUP_DIR="/etc/ssh/keys.backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup old keys
mkdir -p "$BACKUP_DIR"
cp /etc/ssh/ssh_host_* "$BACKUP_DIR/"

# Generate new keys
ssh-keygen -t rsa -b 4096 -f /etc/ssh/ssh_host_rsa_key -N "" -C "rotated-$TIMESTAMP"
ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -N "" -C "rotated-$TIMESTAMP"
ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -N "" -C "rotated-$TIMESTAMP"

# Restart SSH (or reload if supported)
systemctl restart sshd

# Update known_hosts on all clients (via Ansible)
ansible-playbook update-known-hosts.yml
```

**User Key Rotation**:

```bash
#!/bin/bash
# rotate-user-ssh-key.sh

USER="${1:-myuser}"
AUTHORIZED_KEYS="/home/$USER/.ssh/authorized_keys"
BACKUP="$AUTHORIZED_KEYS.backup.$(date +%Y%m%d)"

# Backup
cp "$AUTHORIZED_KEYS" "$BACKUP"

# Generate new key pair
ssh-keygen -t ed25519 -f "/home/$USER/.ssh/id_ed25519_new" -N "" -C "$USER-$(date +%Y%m%d)"

# Add new public key
cat "/home/$USER/.ssh/id_ed25519_new.pub" >> "$AUTHORIZED_KEYS"

# After grace period, remove old keys
# (Manual or scheduled cleanup)
```

### SOPS Encryption Key Rotation

**Rotate Age Key**:

```bash
#!/bin/bash
# rotate-sops-key.sh

OLD_KEY_ID="age1abc123..."
NEW_KEY_ID="age1def456..."

# Generate new Age key
age-keygen -o age-key-new.txt
NEW_PUBLIC_KEY=$(grep "public key" age-key-new.txt | cut -d: -f2 | tr -d ' ')

# Update .sops.yaml to include both keys (dual-key mode)
cat > .sops.yaml <<EOF
creation_rules:
  - path_regex: secrets/.*\.yaml$
    age: >-
      ${OLD_KEY_ID},
      ${NEW_PUBLIC_KEY}
EOF

# Re-encrypt all secrets with new key
find secrets/ -name "*.yaml" -type f | while read file; do
    sops -r -i "$file"
done

# After grace period, remove old key from .sops.yaml
# Update to use only new key
```

### Docker Registry Credential Rotation

**Rotation Script**:

```bash
#!/bin/bash
# rotate-docker-registry-creds.sh

REGISTRY="${DOCKER_REGISTRY:-registry.example.com}"
NEW_USERNAME="${NEW_USERNAME:-deployer}"
NEW_PASSWORD=$(openssl rand -base64 32)

# Login with new credentials
echo "$NEW_PASSWORD" | docker login "$REGISTRY" -u "$NEW_USERNAME" --password-stdin

# Update Kubernetes image pull secret
kubectl create secret docker-registry registry-credentials \
  --docker-server="$REGISTRY" \
  --docker-username="$NEW_USERNAME" \
  --docker-password="$NEW_PASSWORD" \
  --dry-run=client -o yaml | \
  kubectl apply -f -

# Update all deployments to use new secret
kubectl get deployments -A -o json | \
  jq -r '.items[] | select(.spec.template.spec.imagePullSecrets) | "\(.metadata.namespace) \(.metadata.name)"' | \
  while read namespace name; do
    kubectl patch deployment "$name" -n "$namespace" \
      -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"registry-credentials"}]}}}}'
  done

# Rolling restart to pick up new credentials
kubectl rollout restart deployment --all -A
```

## CI/CD Integration

### GitHub Actions with OpenID Connect

**OIDC Setup**:

```yaml
# .github/workflows/deploy.yml
name: Deploy with OIDC

on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: us-east-1
      
      - name: Get secrets from AWS Secrets Manager
        run: |
          POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value \
            --secret-id production/postgres/password \
            --query SecretString --output text)
          echo "::add-mask::$POSTGRES_PASSWORD"
          echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> $GITHUB_ENV
      
      - name: Deploy to Kubernetes
        env:
          KUBECONFIG: ${{ secrets.KUBECONFIG }}
        run: |
          kubectl set env deployment/myapp \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
            -n production
```

**Vault Authentication**:

```yaml
# .github/workflows/vault-auth.yml
name: Deploy with Vault

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Authenticate with Vault
        uses: hashicorp/vault-action@v2
        with:
          url: https://vault.example.com
          method: jwt
          jwtGithubAudience: 'https://github.com/myorg'
          jwtMountPath: 'jwt'
          role: 'github-actions'
          secrets: |
            database/creds/myapp postgres_username POSTGRES_USERNAME
            database/creds/myapp postgres_password POSTGRES_PASSWORD
      
      - name: Deploy
        env:
          POSTGRES_USERNAME: ${{ env.POSTGRES_USERNAME }}
          POSTGRES_PASSWORD: ${{ env.POSTGRES_PASSWORD }}
        run: |
          kubectl set env deployment/myapp \
            POSTGRES_USERNAME="$POSTGRES_USERNAME" \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
            -n production
```

**SOPS Decryption**:

```yaml
# .github/workflows/sops-deploy.yml
name: Deploy with SOPS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install SOPS
        uses: mozilla/sops@v3
        with:
          sops_version: '3.8.1'
      
      - name: Decrypt secrets
        env:
          SOPS_AGE_KEY: ${{ secrets.SOPS_AGE_KEY }}
        run: |
          sops -d secrets/postgres-credentials.yaml > decrypted.yaml
          kubectl apply -f decrypted.yaml
          rm decrypted.yaml
```

### GitLab CI with Protected Variables

**GitLab CI Configuration**:

```yaml
# .gitlab-ci.yml
stages:
  - deploy

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
  before_script:
    - kubectl config use-context production
  script:
    - |
      kubectl create secret generic postgres-credentials \
        --from-literal=username="$POSTGRES_USERNAME" \
        --from-literal=password="$POSTGRES_PASSWORD" \
        --dry-run=client -o yaml | \
        kubectl apply -f -
    - kubectl rollout restart deployment/myapp -n production
    - kubectl rollout status deployment/myapp -n production
  only:
    - main
  when: manual
```

**Protected Variables** (set in GitLab UI):
- `POSTGRES_USERNAME` (protected, masked)
- `POSTGRES_PASSWORD` (protected, masked)

**Vault Integration**:

```yaml
# .gitlab-ci.yml with Vault
deploy_production:
  stage: deploy
  image: 
    name: vault:latest
    entrypoint: [""]
  before_script:
    - apk add --no-cache curl jq
    - |
      VAULT_TOKEN=$(vault write -field=token auth/jwt/login \
        role=gitlab-ci \
        jwt=$CI_JOB_JWT)
      export VAULT_TOKEN
  script:
    - |
      POSTGRES_CREDS=$(vault read -format=json database/creds/myapp)
      POSTGRES_USERNAME=$(echo $POSTGRES_CREDS | jq -r '.data.username')
      POSTGRES_PASSWORD=$(echo $POSTGRES_CREDS | jq -r '.data.password')
      
      kubectl create secret generic postgres-credentials \
        --from-literal=username="$POSTGRES_USERNAME" \
        --from-literal=password="$POSTGRES_PASSWORD" \
        --dry-run=client -o yaml | \
        kubectl apply -f -
```

### Pipeline for Secret Rotation

**Rotation Pipeline**:

```yaml
# .github/workflows/rotate-secrets.yml
name: Rotate Secrets

on:
  schedule:
    - cron: '0 2 1 * *'  # First day of month at 02:00
  workflow_dispatch:

jobs:
  rotate-postgres:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Authenticate with Vault
        uses: hashicorp/vault-action@v2
        with:
          url: https://vault.example.com
          method: jwt
          role: 'github-actions'
      
      - name: Rotate Postgres password
        run: |
          # Generate new password
          NEW_PASSWORD=$(openssl rand -base64 32)
          
          # Update in Vault
          vault kv put database/postgres password="$NEW_PASSWORD"
          
          # Update Kubernetes secret
          kubectl create secret generic postgres-credentials \
            --from-literal=password="$NEW_PASSWORD" \
            --dry-run=client -o yaml | \
            kubectl apply -f -
          
          # Rolling restart
          kubectl rollout restart deployment/myapp -n production
          kubectl rollout status deployment/myapp -n production
```

## Air-Gapped Secrets Management

### Offline CA Structure

**CA Hierarchy**:

```
Root CA (offline, 10-year lifetime)
  └── Intermediate CA 1 (online, 2-year lifetime)
      ├── Server Certificates
      └── Client Certificates
  └── Intermediate CA 2 (online, 2-year lifetime)
      ├── Server Certificates
      └── Client Certificates
```

**Generate Root CA (Offline)**:

```bash
#!/bin/bash
# generate-root-ca.sh (run on air-gapped machine)

CA_DIR="/secure/ca/root"
mkdir -p "$CA_DIR"
cd "$CA_DIR"

# Generate private key
openssl genrsa -aes256 -out root-ca.key 4096

# Generate self-signed certificate
openssl req -new -x509 -days 3650 -key root-ca.key -out root-ca.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=Root CA"

# Create certificate database
touch index.txt
echo 1000 > serial

# Secure permissions
chmod 600 root-ca.key
chmod 644 root-ca.crt
```

**Generate Intermediate CA (Online)**:

```bash
#!/bin/bash
# generate-intermediate-ca.sh

CA_DIR="/secure/ca/intermediate"
ROOT_CA_DIR="/secure/ca/root"
mkdir -p "$CA_DIR"
cd "$CA_DIR"

# Generate private key
openssl genrsa -aes256 -out intermediate-ca.key 4096

# Generate CSR
openssl req -new -key intermediate-ca.key -out intermediate-ca.csr \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=Intermediate CA"

# Transfer CSR to offline machine for signing
# (via secure USB or secure network)

# On offline machine, sign CSR:
cd "$ROOT_CA_DIR"
openssl ca -config openssl.cnf -extensions v3_intermediate_ca \
  -days 730 -notext -md sha256 \
  -in "$CA_DIR/intermediate-ca.csr" \
  -out "$CA_DIR/intermediate-ca.crt"

# Transfer signed certificate back to online machine
```

**Issue Server Certificate**:

```bash
#!/bin/bash
# issue-server-cert.sh

INTERMEDIATE_CA_DIR="/secure/ca/intermediate"
SERVER_NAME="api.example.com"

cd "$INTERMEDIATE_CA_DIR"

# Generate private key
openssl genrsa -out "${SERVER_NAME}.key" 2048

# Generate CSR
openssl req -new -key "${SERVER_NAME}.key" -out "${SERVER_NAME}.csr" \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=${SERVER_NAME}"

# Sign with intermediate CA
openssl ca -config openssl-intermediate.cnf -extensions server_cert \
  -days 90 -notext -md sha256 \
  -in "${SERVER_NAME}.csr" \
  -out "${SERVER_NAME}.crt"

# Create certificate chain
cat "${SERVER_NAME}.crt" intermediate-ca.crt > "${SERVER_NAME}-chain.crt"
```

### Offline Secret Distribution

**Secure USB Transfer**:

```bash
#!/bin/bash
# transfer-secrets.sh

USB_MOUNT="/mnt/secure-usb"
SECRETS_DIR="/secure/secrets"

# On source machine (connected)
# Encrypt secrets
tar czf - "$SECRETS_DIR" | \
  gpg --encrypt --recipient recipient@example.com > secrets.tar.gz.gpg

# Copy to USB
cp secrets.tar.gz.gpg "$USB_MOUNT/"

# On destination machine (air-gapped)
# Decrypt secrets
gpg --decrypt "$USB_MOUNT/secrets.tar.gz.gpg" | tar xzf -

# Verify integrity
sha256sum "$SECRETS_DIR"/* > "$USB_MOUNT/secrets.sha256"
```

**SOPS with Age Keys (Offline)**:

```bash
# Generate Age key on air-gapped machine
age-keygen -o age-key.txt

# Extract public key
PUBLIC_KEY=$(grep "public key" age-key.txt | cut -d: -f2 | tr -d ' ')

# Transfer public key to connected machine (safe to share)
echo "$PUBLIC_KEY" > age-key.pub

# On connected machine, encrypt with public key
sops -e -age "$PUBLIC_KEY" secrets.yaml > secrets.encrypted.yaml

# Transfer encrypted file to air-gapped machine
# On air-gapped machine, decrypt with private key
sops -d -age-key-file age-key.txt secrets.encrypted.yaml > secrets.yaml
```

### Offline Vault Setup

**Vault in Air-Gapped Environment**:

```hcl
# vault-offline.hcl
storage "file" {
  path = "/vault/data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/vault.crt"
  tls_key_file  = "/vault/tls/vault.key"
  tls_client_ca_file = "/vault/tls/ca.crt"  # mTLS
}

api_addr = "https://vault.internal:8200"
cluster_addr = "https://vault.internal:8201"
ui = true

# Disable external services
disable_mlock = false
```

**Initialize Vault (Offline)**:

```bash
# Initialize Vault
vault operator init -key-shares=5 -key-threshold=3

# Save unseal keys and root token securely
# Store in secure enclave or hardware security module
```

## Logging & Auditing

### What Must Be Logged

**Secret Access Events**:
- Secret retrieval (who, when, which secret)
- Secret creation/modification
- Secret deletion
- Failed authentication attempts
- Rotation events

**Vault Audit Log**:

```hcl
# vault.hcl
audit_device "file" {
  path = "/var/log/vault/audit.log"
  format = "json"
  log_raw = false
  hmac_accessor = true
}

audit_device "syslog" {
  facility = "AUTH"
  tag = "vault"
}
```

**Audit Log Format**:

```json
{
  "time": "2024-01-15T10:30:45Z",
  "type": "response",
  "auth": {
    "client_token": "hmac-sha256:abc123...",
    "accessor": "hmac-sha256:def456...",
    "display_name": "github-actions",
    "policies": ["default", "github-actions"]
  },
  "request": {
    "id": "req-789",
    "operation": "read",
    "path": "database/creds/myapp"
  },
  "response": {
    "data": {
      "username": "v-token-myapp-abc123",
      "password": "hmac-sha256:ghi789..."
    }
  }
}
```

### Loki Log Queries

**Secret Access Queries**:

```logql
# All secret access in last hour
{job="vault"} |= "response" | json | request_path=~"secret.*" | line_format "{{.auth.display_name}} accessed {{.request.path}} at {{.time}}"

# Failed authentication attempts
{job="vault"} |= "error" | json | error =~ ".*authentication.*" | line_format "{{.auth.display_name}} failed auth: {{.error}}"

# Secret rotation events
{job="vault"} |= "rotation" | json | line_format "Secret {{.secret_name}} rotated by {{.user}}"
```

**Prometheus Metrics**:

```yaml
# vault-exporter metrics
vault_secret_access_total{secret_path="database/creds/myapp", status="success"} 1250
vault_secret_access_total{secret_path="database/creds/myapp", status="error"} 5
vault_secret_rotation_total{secret_path="database/creds/myapp"} 12
vault_secret_expiry_seconds{secret_path="database/creds/myapp"} 86400
```

**Alert Rules**:

```yaml
# prometheus-alerts.yaml
groups:
  - name: secrets
    rules:
      - alert: SecretAccessFailureRate
        expr: rate(vault_secret_access_total{status="error"}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High secret access failure rate"
          description: "{{ $labels.secret_path }} has {{ $value }} failures/sec"
      
      - alert: SecretExpiringSoon
        expr: vault_secret_expiry_seconds < 86400
        for: 1h
        annotations:
          summary: "Secret expiring soon"
          description: "{{ $labels.secret_path }} expires in {{ $value }} seconds"
      
      - alert: UnauthorizedSecretAccess
        expr: increase(vault_audit_log_unauthorized_total[5m]) > 0
        annotations:
          summary: "Unauthorized secret access attempt"
          description: "Unauthorized access to {{ $labels.secret_path }}"
```

### System-Wide Secret Usage Audit

**Kubernetes Secret Audit**:

```bash
#!/bin/bash
# audit-k8s-secrets.sh

# List all secrets
kubectl get secrets -A -o json | \
  jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name) \(.type)"'

# Find secrets with old creation dates
kubectl get secrets -A -o json | \
  jq -r '.items[] | select(.metadata.creationTimestamp < "2023-01-01") | "\(.metadata.namespace)/\(.metadata.name) created \(.metadata.creationTimestamp)"'

# Find secrets not referenced by any pod
kubectl get secrets -A -o json > /tmp/all-secrets.json
kubectl get pods -A -o json | \
  jq -r '.items[].spec.containers[].envFrom[]?.secretRef.name // empty' | \
  sort -u > /tmp/used-secrets.txt

# Compare
comm -23 <(jq -r '.items[].metadata.name' /tmp/all-secrets.json | sort) \
         <(sort /tmp/used-secrets.txt)
```

## Incident Response

### Immediate Revocation Procedures

**Revocation Playbook**:

```bash
#!/bin/bash
# revoke-secret.sh

SECRET_NAME="${1}"
SECRET_TYPE="${2:-generic}"

case "$SECRET_TYPE" in
  "postgres")
    # Revoke Postgres user
    psql -U postgres -c "REVOKE ALL PRIVILEGES ON DATABASE mydb FROM ${SECRET_NAME};"
    psql -U postgres -c "DROP USER IF EXISTS ${SECRET_NAME};"
    ;;
  
  "redis")
    # Rotate Redis password immediately
    NEW_PASSWORD=$(openssl rand -base64 32)
    redis-cli CONFIG SET requirepass "$NEW_PASSWORD"
    kubectl create secret generic redis-credentials \
      --from-literal=password="$NEW_PASSWORD" \
      --dry-run=client -o yaml | \
      kubectl apply -f -
    ;;
  
  "jwt")
    # Remove JWT key from active keys
    kubectl delete secret jwt-signing-key-"${SECRET_NAME}"
    # Update key manager to exclude key
    ;;
  
  "api-key")
    # Revoke API key in database
    psql -U postgres -d mydb -c \
      "UPDATE api_keys SET revoked=true WHERE key_id='${SECRET_NAME}';"
    ;;
esac

# Force pod restarts
kubectl rollout restart deployment --all -A

# Log revocation
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SECRET_REVOKED: ${SECRET_NAME} (${SECRET_TYPE})" \
  >> /var/log/security/secret-revocations.log
```

### Blast Radius Analysis

**Analysis Script**:

```bash
#!/bin/bash
# analyze-blast-radius.sh

SECRET_NAME="${1}"

echo "Analyzing blast radius for secret: ${SECRET_NAME}"

# Find all pods using this secret
kubectl get pods -A -o json | \
  jq -r --arg secret "$SECRET_NAME" \
    '.items[] | select(.spec.containers[].envFrom[]?.secretRef.name == $secret or .spec.volumes[]?.secret.secretName == $secret) | "\(.metadata.namespace)/\(.metadata.name)"'

# Find all deployments
kubectl get deployments -A -o json | \
  jq -r --arg secret "$SECRET_NAME" \
    '.items[] | select(.spec.template.spec.containers[].envFrom[]?.secretRef.name == $secret) | "\(.metadata.namespace)/\(.metadata.name)"'

# Find all services
kubectl get services -A -o json | \
  jq -r --arg secret "$SECRET_NAME" \
    '.items[] | select(.metadata.annotations."secret" == $secret) | "\(.metadata.namespace)/\(.metadata.name)"'

# Check CI/CD pipelines
grep -r "$SECRET_NAME" .github/workflows/ .gitlab-ci.yml 2>/dev/null || echo "No CI/CD references found"
```

### Rapid Rotation Automation

**Emergency Rotation Script**:

```bash
#!/bin/bash
# emergency-rotate.sh

set -euo pipefail

SECRET_TYPE="${1}"
NAMESPACE="${2:-production}"

case "$SECRET_TYPE" in
  "all")
    ./rotate-postgres-password.sh
    ./rotate-redis-auth.sh
    ./rotate-jwt-keys.sh
    ./rotate-tls-cert.sh
    ;;
  
  "postgres")
    NEW_PASSWORD=$(openssl rand -base64 32)
    kubectl create secret generic postgres-credentials \
      --from-literal=password="$NEW_PASSWORD" \
      --namespace="$NAMESPACE" \
      --dry-run=client -o yaml | \
      kubectl apply -f -
    kubectl rollout restart deployment -n "$NAMESPACE"
    ;;
  
  "redis")
    NEW_TOKEN=$(openssl rand -base64 32)
    redis-cli CONFIG SET requirepass "$NEW_TOKEN"
    kubectl create secret generic redis-credentials \
      --from-literal=password="$NEW_TOKEN" \
      --namespace="$NAMESPACE" \
      --dry-run=client -o yaml | \
      kubectl apply -f -
    kubectl rollout restart deployment -n "$NAMESPACE"
    ;;
esac

# Notify team
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"🚨 Emergency secret rotation completed for ${SECRET_TYPE}\"}"
```

### Developer Notification Template

**Notification Script**:

```bash
#!/bin/bash
# notify-secret-rotation.sh

SECRET_NAME="${1}"
ROTATION_REASON="${2:-scheduled}"

# Email template
cat > /tmp/notification.txt <<EOF
Subject: Secret Rotation Notification: ${SECRET_NAME}

A secret has been rotated:

Secret: ${SECRET_NAME}
Reason: ${ROTATION_REASON}
Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Action Required:
- Update local .env files if applicable
- Restart local development services
- Update any cached credentials

If you experience authentication issues, contact the infrastructure team.

This is an automated notification from the secrets management system.
EOF

# Send notification
mail -s "Secret Rotation: ${SECRET_NAME}" team@example.com < /tmp/notification.txt

# Slack notification
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"Secret Rotation: ${SECRET_NAME}\",
    \"blocks\": [
      {
        \"type\": \"section\",
        \"text\": {
          \"type\": \"mrkdwn\",
          \"text\": \"*Secret Rotation Notification*\n\n*Secret:* ${SECRET_NAME}\n*Reason:* ${ROTATION_REASON}\n*Time:* $(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }
      }
    ]
  }"
```

### Forced Session Invalidation

**Postgres Session Invalidation**:

```sql
-- Terminate all connections for a user
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE usename = 'compromised_user';

-- Revoke and recreate user
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM compromised_user;
DROP USER compromised_user;
```

**Redis Session Invalidation**:

```bash
# Rotate password (invalidates all connections)
redis-cli CONFIG SET requirepass "$NEW_PASSWORD"

# Kill all existing connections
redis-cli CLIENT KILL TYPE normal
```

**JWT Session Invalidation**:

```python
# Add to JWT blacklist
import redis

redis_client = redis.Redis(host='redis.example.com')

def invalidate_jwt(token: str, expiry_seconds: int = 3600):
    """Add JWT to blacklist"""
    jti = jwt.decode(token, options={"verify_signature": False})["jti"]
    redis_client.setex(f"jwt:blacklist:{jti}", expiry_seconds, "1")

def is_jwt_valid(token: str) -> bool:
    """Check if JWT is blacklisted"""
    jti = jwt.decode(token, options={"verify_signature": False})["jti"]
    return not redis_client.exists(f"jwt:blacklist:{jti}")
```

## Reference Implementation

### Complete Setup: RKE2 + Rancher + ESO + SOPS

**Directory Structure**:

```
secrets-governance/
├── vault/
│   ├── vault.hcl
│   ├── policies/
│   │   ├── github-actions.hcl
│   │   └── external-secrets.hcl
│   └── init.sh
├── kubernetes/
│   ├── external-secrets/
│   │   ├── secretstore.yaml
│   │   └── externalsecret.yaml
│   ├── sealed-secrets/
│   │   └── sealed-secret.yaml
│   └── cert-manager/
│       ├── cluster-issuer.yaml
│       └── certificate.yaml
├── sops/
│   ├── .sops.yaml
│   └── secrets/
│       ├── postgres-credentials.yaml
│       └── redis-credentials.yaml
├── ansible/
│   ├── rotate-postgres-password.yml
│   └── rotate-redis-auth.yml
├── scripts/
│   ├── rotate-all-secrets.sh
│   ├── emergency-revoke.sh
│   └── audit-secrets.sh
└── monitoring/
    ├── prometheus-alerts.yaml
    └── grafana-dashboard.json
```

**Vault Configuration**:

```hcl
# vault/vault.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}

api_addr = "https://vault.example.com:8200"
ui = true

# Enable audit logging
audit_device "file" {
  path = "/var/log/vault/audit.log"
  format = "json"
}
```

**External Secrets Operator Setup**:

```yaml
# kubernetes/external-secrets/secretstore.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: production
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"
          serviceAccountRef:
            name: external-secrets
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: postgres-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: postgres-credentials
    creationPolicy: Owner
  data:
    - secretKey: username
      remoteRef:
        key: database/postgres
        property: username
    - secretKey: password
      remoteRef:
        key: database/postgres
        property: password
```

**Rotation Automation**:

```yaml
# kubernetes/cronjobs/rotate-secrets.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-postgres-password
  namespace: production
spec:
  schedule: "0 2 1 * *"  # First day of month
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
            - name: rotate
              image: postgres:15
              command:
                - /bin/bash
                - -c
                - |
                  # Generate new password
                  NEW_PASSWORD=$(openssl rand -base64 32)
                  
                  # Update in Vault
                  vault kv put secret/database/postgres password="$NEW_PASSWORD"
                  
                  # Update Kubernetes secret
                  kubectl create secret generic postgres-credentials \
                    --from-literal=password="$NEW_PASSWORD" \
                    --dry-run=client -o yaml | \
                    kubectl apply -f -
                  
                  # Rolling restart
                  kubectl rollout restart deployment/myapp -n production
          restartPolicy: OnFailure
```

**Monitoring Dashboard**:

```json
{
  "dashboard": {
    "title": "Secrets Management",
    "panels": [
      {
        "title": "Secret Access Rate",
        "targets": [
          {
            "expr": "rate(vault_secret_access_total[5m])",
            "legendFormat": "{{secret_path}}"
          }
        ]
      },
      {
        "title": "Secret Expiry",
        "targets": [
          {
            "expr": "vault_secret_expiry_seconds",
            "legendFormat": "{{secret_path}}"
          }
        ]
      },
      {
        "title": "Rotation Events",
        "targets": [
          {
            "expr": "increase(vault_secret_rotation_total[1h])",
            "legendFormat": "{{secret_path}}"
          }
        ]
      }
    ]
  }
}
```

## Common Pitfalls & Anti-Patterns

### Hardcoded Secrets in Code

**Symptom**: Secrets appear in source code, Git history, or container images.

**Root Cause**: Developers commit secrets accidentally or use secrets in code instead of environment variables.

**Corrective Action**:
- Use pre-commit hooks to scan for secrets:
  ```bash
  # .pre-commit-config.yaml
  repos:
    - repo: https://github.com/Yelp/detect-secrets
      rev: v1.4.0
      hooks:
        - id: detect-secrets
  ```
- Use environment variables or secret injection patterns
- Rotate any exposed secrets immediately
- Use Git secrets scanning (GitHub Advanced Security, GitLab Secret Detection)

### Long-Lived API Keys with No Rotation Plan

**Symptom**: API keys created years ago, never rotated, no expiration.

**Root Cause**: No rotation policy or automation.

**Corrective Action**:
- Implement dual-key rotation pattern
- Set expiration dates on all API keys
- Automate rotation via cron jobs or operators
- Monitor key age and alert on old keys

### Shared Credentials Between Services

**Symptom**: Multiple services use the same database user, API key, or token.

**Root Cause**: Convenience over security, lack of service isolation.

**Corrective Action**:
- Use dynamic secrets (Vault database engine)
- Create per-service credentials
- Use service accounts with least privilege
- Implement service-to-service authentication (mTLS, JWT)

### "Temporary" Secrets That Live 3 Years

**Symptom**: Secrets created for testing that are still in production.

**Root Cause**: No expiration or cleanup process.

**Corrective Action**:
- Set TTL on all secrets
- Implement automated cleanup of expired secrets
- Regular audit to identify stale secrets
- Use naming conventions (e.g., `temp-*` prefix with auto-expiry)

### Storing Secrets Unencrypted on Persistent Storage

**Symptom**: Secrets in plain text files, databases, or object storage.

**Root Cause**: Lack of encryption at rest.

**Corrective Action**:
- Encrypt all secrets at rest (Vault transit, SOPS, KMS)
- Use encrypted volumes for secret storage
- Enable encryption at rest for Kubernetes Secrets
- Never store secrets in plain text files

### Using Kubernetes Secrets Without Encryption at Rest

**Symptom**: Kubernetes Secrets are base64-encoded but not encrypted.

**Root Cause**: Encryption at rest not enabled on Kubernetes cluster.

**Corrective Action**:
- Enable EncryptionConfiguration in kube-apiserver
- Use external secret management (Vault, ESO)
- Consider Sealed Secrets for GitOps workflows
- Never rely on base64 encoding as security

### Forgetting to Rotate Service Tokens

**Symptom**: Grafana, MLflow, RabbitMQ service tokens never rotated.

**Root Cause**: Service tokens treated as "set and forget."

**Corrective Action**:
- Document all service tokens in inventory
- Implement rotation schedule for all tokens
- Use automated rotation where possible
- Monitor token age and alert on old tokens

### Circular Dependencies Between Secret Providers and Workloads

**Symptom**: Vault needs database credentials, but database needs Vault credentials.

**Root Cause**: Poor architecture, no bootstrap process.

**Corrective Action**:
- Design bootstrap process for secret providers
- Use different authentication methods for providers (mTLS, OIDC)
- Avoid circular dependencies in architecture
- Use initial secrets for provider setup, then rotate

## Summary & Checklist

### Secrets Governance Checklist

**Storage**:
- [ ] All secrets stored in encrypted form (Vault, SOPS, KMS)
- [ ] Kubernetes Secrets have encryption at rest enabled
- [ ] No secrets in Git (or encrypted with SOPS)
- [ ] Secrets backed up securely

**Access Control**:
- [ ] Least privilege access policies
- [ ] Service accounts for machine-to-machine auth
- [ ] Audit logging enabled for all secret access
- [ ] Regular access reviews

**Rotation**:
- [ ] Rotation schedule defined for all secret types
- [ ] Automated rotation implemented where possible
- [ ] Dual-key/zero-downtime rotation for critical secrets
- [ ] Rotation tested in non-production first

**Monitoring**:
- [ ] Secret access logged and monitored
- [ ] Alerts for failed authentication
- [ ] Alerts for expiring secrets
- [ ] Dashboard for secret health

**Incident Response**:
- [ ] Revocation procedures documented
- [ ] Emergency rotation scripts tested
- [ ] Blast radius analysis tools available
- [ ] Notification templates prepared

### Rotation Cadence Table

| Secret Type | Rotation Frequency | Method | Zero-Downtime |
|-------------|-------------------|--------|---------------|
| **Postgres Passwords** | Monthly | Ansible/PgCron | Yes (dual-user) |
| **Redis AUTH** | Quarterly | Script + K8s | Yes (grace period) |
| **TLS Certificates** | 90 days | cert-manager/ACME | Yes (reload) |
| **JWT Signing Keys** | Quarterly | Kid-based rotation | Yes (multi-key) |
| **API Keys** | Monthly | Dual-key pattern | Yes |
| **SSH Host Keys** | Annually | Keygen + restart | No (planned downtime) |
| **SSH User Keys** | Quarterly | Keygen + authorized_keys | Yes |
| **Docker Registry** | Quarterly | Script + K8s | Yes (rolling restart) |
| **SOPS Age Keys** | Annually | Re-encrypt all secrets | Yes (dual-key) |
| **Vault Root Token** | Never (if possible) | Manual process | N/A |
| **Vault Unseal Keys** | Annually | Rekey operation | Yes |

### Directory Layout for Secrets Governance

```
docs/best-practices/security/
├── secrets-governance.md (this document)
├── vault-setup.md
├── sops-workflow.md
└── incident-response.md

infrastructure/secrets/
├── vault/
│   ├── config/
│   ├── policies/
│   └── scripts/
├── kubernetes/
│   ├── external-secrets/
│   ├── sealed-secrets/
│   └── cert-manager/
├── sops/
│   ├── .sops.yaml
│   └── secrets/
├── ansible/
│   └── rotation-playbooks/
└── scripts/
    ├── rotation/
    ├── audit/
    └── emergency/
```

## See Also

- **[Secrets Management Best Practices](../architecture-design/secrets-management.md)** - High-level secrets management patterns
- **[PostgreSQL Security Best Practices](../../best-practices/postgres/postgres-security-best-practices.md)** - Database security
- **[CI/CD Pipeline Best Practices](../architecture-design/ci-cd-pipelines.md)** - Secure CI/CD patterns
- **[Monitoring & Observability](../operations-monitoring/grafana-prometheus-loki-observability.md)** - Security monitoring

---

*This guide provides a complete framework for secrets management and key rotation. Start with storage and injection patterns, then implement rotation automation. The goal is security without operational burden.*

