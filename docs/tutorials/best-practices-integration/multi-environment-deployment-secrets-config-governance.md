# Multi-Environment Deployment with Secrets and Configuration Governance: A Complete Integration Tutorial

**Objective**: Build a production-ready multi-environment deployment pipeline that integrates environment configuration governance, secrets management, CI/CD pipelines, and configuration drift prevention. This tutorial demonstrates how to safely promote configurations from dev → staging → production while maintaining security and consistency.

This tutorial combines:
- **[Cross-Environment Configuration Strategy](../../best-practices/architecture-design/environment-config-governance.md)** - Configuration governance and drift prevention
- **[End-to-End Secrets Management](../../best-practices/security/secrets-governance.md)** - Secrets lifecycle and rotation
- **[CI/CD Pipelines](../../best-practices/architecture-design/ci-cd-pipelines.md)** - Automated deployment workflows
- **[Cross-Environment Configuration Drift Prevention](../../best-practices/operations-monitoring/environment-promotion-drift-governance.md)** - Drift detection and enforcement

## 1) Prerequisites

```bash
# Required tools
docker --version          # >= 20.10
docker compose --version  # >= 2.0
kubectl --version         # >= 1.28
helm --version            # >= 3.12
ansible --version         # >= 2.14
sops --version            # For secrets encryption
age --version             # For SOPS encryption keys
git --version             # >= 2.30

# System requirements
# - Kubernetes cluster (local or remote)
# - 8+ GB RAM recommended
# - 20+ GB free disk space
```

**Why**: Multi-environment deployments require orchestration tools (Kubernetes, Ansible), secrets management (SOPS), and configuration management to ensure consistency and security.

## 2) Architecture Overview

We'll deploy a **Web Application** across three environments with configuration governance:

```mermaid
graph TB
    subgraph "CI/CD Pipeline"
        GIT[Git Repository]
        CI[GitHub Actions CI]
        CD[Deployment Pipeline]
    end

    subgraph "Configuration Management"
        CONFIG_REPO[Config Repository]
        SOPS[SOPS Encrypted Secrets]
        HELM[Helm Charts]
        ANSIBLE[Ansible Playbooks]
    end

    subgraph "Environments"
        DEV[Development<br/>Kubernetes]
        STAGING[Staging<br/>Kubernetes]
        PROD[Production<br/>Kubernetes]
    end

    subgraph "Secrets Management"
        VAULT[Hashicorp Vault]
        KMS[KMS/age keys]
        SECRETS[Kubernetes Secrets]
    end

    subgraph "Monitoring"
        DRIFT[Drift Detection]
        AUDIT[Audit Logs]
        ALERTS[Alerts]
    end

    GIT --> CI
    CI --> CD
    CD --> CONFIG_REPO
    
    CONFIG_REPO --> SOPS
    CONFIG_REPO --> HELM
    CONFIG_REPO --> ANSIBLE
    
    HELM --> DEV
    HELM --> STAGING
    HELM --> PROD
    
    ANSIBLE --> DEV
    ANSIBLE --> STAGING
    ANSIBLE --> PROD
    
    VAULT --> SECRETS
    KMS --> SOPS
    SECRETS --> DEV
    SECRETS --> STAGING
    SECRETS --> PROD
    
    DEV --> DRIFT
    STAGING --> DRIFT
    PROD --> DRIFT
    
    DRIFT --> AUDIT
    DRIFT --> ALERTS
```

**Environment Promotion Flow**:
1. **Development**: Manual deployments, relaxed validation
2. **Staging**: Automated promotion from dev, full validation
3. **Production**: Automated promotion from staging, strict validation, approval gates

## 3) Repository Layout

```
multi-env-deployment/
├── .github/
│   └── workflows/
│       ├── deploy-dev.yml
│       ├── deploy-staging.yml
│       └── deploy-prod.yml
├── config/
│   ├── base/
│   │   ├── values.yaml
│   │   └── kustomization.yaml
│   ├── environments/
│   │   ├── dev/
│   │   │   ├── values.yaml
│   │   │   ├── secrets.yaml.enc  # SOPS encrypted
│   │   │   └── kustomization.yaml
│   │   ├── staging/
│   │   │   ├── values.yaml
│   │   │   ├── secrets.yaml.enc
│   │   │   └── kustomization.yaml
│   │   └── prod/
│   │       ├── values.yaml
│   │       ├── secrets.yaml.enc
│   │       └── kustomization.yaml
│   └── drift-detection/
│       └── drift-check.sh
├── helm/
│   └── webapp/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── configmap.yaml
│           └── secret.yaml
├── ansible/
│   ├── inventory/
│   │   ├── dev.yml
│   │   ├── staging.yml
│   │   └── prod.yml
│   ├── playbooks/
│   │   ├── deploy.yml
│   │   ├── validate-config.yml
│   │   └── rotate-secrets.yml
│   └── roles/
│       └── webapp/
├── scripts/
│   ├── setup-sops.sh
│   ├── promote-config.sh
│   └── validate-drift.sh
└── .sops.yaml
```

**Why**: This structure separates base configuration from environment-specific overrides, enables secrets encryption, and supports both Helm and Ansible deployment methods.

## 4) SOPS Configuration for Secrets Management

### 4.1) Initialize SOPS

Create `.sops.yaml`:

```yaml
creation_rules:
  - path_regex: config/environments/dev/.*\.enc$
    age: >age1devkey123456789012345678901234567890123456789012345678901234567890
    encrypted_regex: ^(data|stringData|password|secret|token|key)$
  
  - path_regex: config/environments/staging/.*\.enc$
    age: >age1stagingkey123456789012345678901234567890123456789012345678901234567890
    encrypted_regex: ^(data|stringData|password|secret|token|key)$
  
  - path_regex: config/environments/prod/.*\.enc$
    age: >age1prodkey123456789012345678901234567890123456789012345678901234567890
    encrypted_regex: ^(data|stringData|password|secret|token|key)$
```

**Why**: SOPS provides encryption at rest for secrets in Git, with different keys per environment for security isolation.

### 4.2) Generate Age Keys

```bash
#!/bin/bash
# scripts/setup-sops.sh

# Generate age keys for each environment
age-keygen -o dev-key.txt
age-keygen -o staging-key.txt
age-keygen -o prod-key.txt

# Extract public keys
age-keygen -y dev-key.txt > dev-key.pub
age-keygen -y staging-key.txt > staging-key.pub
age-keygen -y prod-key.txt > prod-key.pub

echo "Age keys generated. Store private keys securely!"
echo "Add public keys to .sops.yaml"
```

### 4.3) Environment Secrets

Create `config/environments/dev/secrets.yaml` (unencrypted template):

```yaml
# Database credentials
database:
  host: dev-db.example.com
  port: 5432
  username: dev_user
  password: dev_password_123  # Will be encrypted
  database: dev_db

# API keys
api_keys:
  stripe: sk_test_1234567890abcdef
  sendgrid: SG.test_key_1234567890

# TLS certificates
tls:
  cert: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  key: |
    -----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----
```

Encrypt with SOPS:

```bash
sops -e -i config/environments/dev/secrets.yaml
# Creates secrets.yaml.enc
```

## 5) Base Configuration

### 5.1) Base Values

Create `config/base/values.yaml`:

```yaml
# Base configuration shared across all environments
app:
  name: webapp
  version: "1.0.0"
  replicas: 2
  image:
    repository: myregistry/webapp
    tag: latest
    pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: false
  className: nginx
  annotations: {}

# Observability
observability:
  enabled: true
  prometheus:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    jaeger:
      endpoint: http://jaeger:14268/api/traces
```

### 5.2) Environment-Specific Overrides

Create `config/environments/dev/values.yaml`:

```yaml
# Development overrides
app:
  replicas: 1
  image:
    tag: dev-latest

resources:
  requests:
    cpu: 50m
    memory: 128Mi
  limits:
    cpu: 200m
    memory: 256Mi

ingress:
  enabled: true
  host: dev.example.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-staging

# Development-specific config
config:
  logLevel: debug
  featureFlags:
    experimental: true
    beta: true
```

Create `config/environments/staging/values.yaml`:

```yaml
# Staging overrides
app:
  replicas: 2
  image:
    tag: staging-latest

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

ingress:
  enabled: true
  host: staging.example.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod

config:
  logLevel: info
  featureFlags:
    experimental: true
    beta: false
```

Create `config/environments/prod/values.yaml`:

```yaml
# Production overrides
app:
  replicas: 5
  image:
    tag: v1.0.0  # Pinned version

resources:
  requests:
    cpu: 200m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi

ingress:
  enabled: true
  host: example.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"

config:
  logLevel: warn
  featureFlags:
    experimental: false
    beta: false

# Production-specific
autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

## 6) Helm Chart

### 6.1) Chart Structure

Create `helm/webapp/Chart.yaml`:

```yaml
apiVersion: v2
name: webapp
description: Web application Helm chart
type: application
version: 1.0.0
appVersion: "1.0.0"
```

### 6.2) Deployment Template

Create `helm/webapp/templates/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Values.app.name }}
  namespace: {{ .Values.namespace | default "default" }}
  labels:
    app: {{ .Values.app.name }}
    version: {{ .Values.app.version }}
    environment: {{ .Values.environment }}
spec:
  replicas: {{ .Values.app.replicas }}
  selector:
    matchLabels:
      app: {{ .Values.app.name }}
  template:
    metadata:
      labels:
        app: {{ .Values.app.name }}
        version: {{ .Values.app.version }}
        environment: {{ .Values.environment }}
      annotations:
        prometheus.io/scrape: "{{ .Values.observability.prometheus.enabled }}"
        prometheus.io/port: "{{ .Values.observability.prometheus.port }}"
    spec:
      containers:
      - name: {{ .Values.app.name }}
        image: "{{ .Values.app.image.repository }}:{{ .Values.app.image.tag }}"
        imagePullPolicy: {{ .Values.app.image.pullPolicy }}
        ports:
        - containerPort: {{ .Values.service.targetPort }}
          name: http
        {{- if .Values.observability.prometheus.enabled }}
        - containerPort: {{ .Values.observability.prometheus.port }}
          name: metrics
        {{- end }}
        env:
        - name: ENVIRONMENT
          value: {{ .Values.environment }}
        - name: LOG_LEVEL
          value: {{ .Values.config.logLevel }}
        - name: DATABASE_HOST
          valueFrom:
            secretKeyRef:
              name: {{ .Values.app.name }}-secrets
              key: database.host
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: {{ .Values.app.name }}-secrets
              key: database.password
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 6.3) Secret Template

Create `helm/webapp/templates/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Values.app.name }}-secrets
  namespace: {{ .Values.namespace | default "default" }}
type: Opaque
data:
  {{- $secrets := .Files.Get "secrets.yaml" | fromYaml }}
  {{- range $key, $value := $secrets }}
  {{ $key }}: {{ $value | b64enc }}
  {{- end }}
```

## 7) Configuration Promotion Script

Create `scripts/promote-config.sh`:

```bash
#!/bin/bash
# Promote configuration from source to target environment

set -euo pipefail

SOURCE_ENV="${1:-}"
TARGET_ENV="${2:-}"

if [[ -z "$SOURCE_ENV" || -z "$TARGET_ENV" ]]; then
    echo "Usage: $0 <source-env> <target-env>"
    echo "Example: $0 dev staging"
    exit 1
fi

SOURCE_DIR="config/environments/$SOURCE_ENV"
TARGET_DIR="config/environments/$TARGET_ENV"

# Validate source exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source environment $SOURCE_ENV not found"
    exit 1
fi

# Validate promotion rules
case "$SOURCE_ENV->$TARGET_ENV" in
    "dev->staging"|"staging->prod")
        echo "Promoting from $SOURCE_ENV to $TARGET_ENV..."
        ;;
    *)
        echo "Error: Invalid promotion path: $SOURCE_ENV -> $TARGET_ENV"
        echo "Allowed: dev -> staging, staging -> prod"
        exit 1
        ;;
esac

# Copy and merge values
echo "Merging values.yaml..."
yq eval-all '. as $item ireduce ({}; . * $item)' \
    "config/base/values.yaml" \
    "$SOURCE_DIR/values.yaml" \
    > "$TARGET_DIR/values.yaml.tmp"

# Validate merged configuration
echo "Validating configuration..."
if ! yq eval '.app.replicas > 0' "$TARGET_DIR/values.yaml.tmp" > /dev/null; then
    echo "Error: Invalid configuration after merge"
    exit 1
fi

# Move temp file
mv "$TARGET_DIR/values.yaml.tmp" "$TARGET_DIR/values.yaml"

# Secrets must be manually updated (never auto-promote secrets)
echo "Warning: Secrets must be manually updated for $TARGET_ENV"
echo "Run: sops config/environments/$TARGET_ENV/secrets.yaml"

echo "Configuration promoted successfully!"
```

## 8) Drift Detection

Create `scripts/validate-drift.sh`:

```bash
#!/bin/bash
# Detect configuration drift between Git and cluster

set -euo pipefail

ENVIRONMENT="${1:-dev}"
NAMESPACE="${2:-default}"

echo "Checking drift for environment: $ENVIRONMENT"

# Get live configuration from cluster
kubectl get deployment webapp -n "$NAMESPACE" -o yaml > /tmp/live-config.yaml

# Get expected configuration from Helm
helm template webapp ./helm/webapp \
    -f "config/base/values.yaml" \
    -f "config/environments/$ENVIRONMENT/values.yaml" \
    > /tmp/expected-config.yaml

# Compare
if diff -u /tmp/expected-config.yaml /tmp/live-config.yaml; then
    echo "✓ No drift detected"
    exit 0
else
    echo "✗ Configuration drift detected!"
    echo "Run: helm upgrade webapp ./helm/webapp -f config/environments/$ENVIRONMENT/values.yaml"
    exit 1
fi
```

## 9) CI/CD Pipeline

### 9.1) Development Deployment

Create `.github/workflows/deploy-dev.yml`:

```yaml
name: Deploy to Development

on:
  push:
    branches: [develop]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: development
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup tools
      run: |
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        kubectl version --client
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG_DEV }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Decrypt secrets
      run: |
        echo "${{ secrets.SOPS_AGE_KEY_DEV }}" > age-key.txt
        export SOPS_AGE_KEY_FILE=age-key.txt
        sops -d config/environments/dev/secrets.yaml.enc > config/environments/dev/secrets.yaml
    
    - name: Deploy to cluster
      run: |
        helm upgrade --install webapp ./helm/webapp \
          -f config/base/values.yaml \
          -f config/environments/dev/values.yaml \
          --set environment=dev \
          --namespace dev \
          --create-namespace \
          --wait
    
    - name: Validate deployment
      run: |
        kubectl rollout status deployment/webapp -n dev --timeout=5m
        kubectl get pods -n dev
```

### 9.2) Staging Deployment

Create `.github/workflows/deploy-staging.yml`:

```yaml
name: Deploy to Staging

on:
  push:
    branches: [staging]
  workflow_dispatch:

jobs:
  promote-and-deploy:
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Promote configuration
      run: |
        ./scripts/promote-config.sh dev staging
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add config/environments/staging/
        git commit -m "Promote config: dev -> staging" || exit 0
        git push
    
    - name: Setup tools
      run: |
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG_STAGING }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Decrypt secrets
      run: |
        echo "${{ secrets.SOPS_AGE_KEY_STAGING }}" > age-key.txt
        export SOPS_AGE_KEY_FILE=age-key.txt
        sops -d config/environments/staging/secrets.yaml.enc > config/environments/staging/secrets.yaml
    
    - name: Validate configuration
      run: |
        helm lint ./helm/webapp \
          -f config/base/values.yaml \
          -f config/environments/staging/values.yaml
    
    - name: Deploy to cluster
      run: |
        helm upgrade --install webapp ./helm/webapp \
          -f config/base/values.yaml \
          -f config/environments/staging/values.yaml \
          --set environment=staging \
          --namespace staging \
          --create-namespace \
          --wait
    
    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod -l app=webapp -n staging --timeout=5m
        # Add actual smoke tests here
    
    - name: Check for drift
      run: |
        ./scripts/validate-drift.sh staging staging
```

### 9.3) Production Deployment

Create `.github/workflows/deploy-prod.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  promote-and-deploy:
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Promote configuration
      run: |
        ./scripts/promote-config.sh staging prod
    
    - name: Validate production config
      run: |
        # Strict validation for production
        if ! yq eval '.app.image.tag | test("^v[0-9]+\\.[0-9]+\\.[0-9]+$")' \
          config/environments/prod/values.yaml; then
          echo "Error: Production must use semantic version tag"
          exit 1
        fi
        
        if ! yq eval '.app.replicas >= 3' \
          config/environments/prod/values.yaml; then
          echo "Error: Production must have at least 3 replicas"
          exit 1
        fi
    
    - name: Setup tools
      run: |
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG_PROD }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Decrypt secrets
      run: |
        echo "${{ secrets.SOPS_AGE_KEY_PROD }}" > age-key.txt
        export SOPS_AGE_KEY_FILE=age-key.txt
        sops -d config/environments/prod/secrets.yaml.enc > config/environments/prod/secrets.yaml
    
    - name: Create backup
      run: |
        helm get values webapp -n prod > backup-$(date +%Y%m%d-%H%M%S).yaml || true
    
    - name: Deploy to cluster
      run: |
        helm upgrade --install webapp ./helm/webapp \
          -f config/base/values.yaml \
          -f config/environments/prod/values.yaml \
          --set environment=prod \
          --namespace prod \
          --create-namespace \
          --wait \
          --timeout 10m
    
    - name: Validate deployment
      run: |
        kubectl rollout status deployment/webapp -n prod --timeout=10m
        kubectl get pods -n prod
    
    - name: Check for drift
      run: |
        ./scripts/validate-drift.sh prod prod
```

## 10) Ansible Playbook for Configuration Validation

Create `ansible/playbooks/validate-config.yml`:

```yaml
---
- name: Validate Configuration Across Environments
  hosts: localhost
  gather_facts: false
  
  vars:
    environments:
      - dev
      - staging
      - prod
  
  tasks:
    - name: Load base configuration
      set_fact:
        base_config: "{{ lookup('file', 'config/base/values.yaml') | from_yaml }}"
    
    - name: Validate each environment
      include_tasks: validate-environment.yml
      loop: "{{ environments }}"
      loop_control:
        loop_var: env
```

Create `ansible/playbooks/validate-environment.yml`:

```yaml
---
- name: Load environment configuration
  set_fact:
    env_config: "{{ lookup('file', 'config/environments/{{ env }}/values.yaml') | from_yaml }}"
  
- name: Merge configurations
  set_fact:
    merged_config: "{{ base_config | combine(env_config, recursive=True) }}"
  
- name: Validate required fields
  assert:
    that:
      - merged_config.app.name is defined
      - merged_config.app.replicas is defined
      - merged_config.app.replicas > 0
      - merged_config.resources.requests.cpu is defined
      - merged_config.resources.requests.memory is defined
    fail_msg: "Missing required configuration fields for {{ env }}"
    success_msg: "Configuration valid for {{ env }}"
  
- name: Validate environment-specific rules
  assert:
    that:
      - env != 'prod' or merged_config.app.replicas >= 3
      - env != 'prod' or merged_config.app.image.tag | regex_search('^v[0-9]+\\.[0-9]+\\.[0-9]+$')
      - env != 'prod' or merged_config.config.logLevel != 'debug'
    fail_msg: "Environment-specific validation failed for {{ env }}"
    success_msg: "Environment-specific rules validated for {{ env }}"
```

## 11) Testing the System

### 11.1) Setup SOPS

```bash
# Generate age keys
./scripts/setup-sops.sh

# Encrypt secrets
sops -e -i config/environments/dev/secrets.yaml
```

### 11.2) Validate Configuration

```bash
# Validate all environments
ansible-playbook ansible/playbooks/validate-config.yml

# Check for drift
./scripts/validate-drift.sh dev dev
```

### 11.3) Deploy to Development

```bash
# Manual deployment
helm upgrade --install webapp ./helm/webapp \
  -f config/base/values.yaml \
  -f config/environments/dev/values.yaml \
  --set environment=dev \
  --namespace dev \
  --create-namespace
```

### 11.4) Promote to Staging

```bash
# Promote configuration
./scripts/promote-config.sh dev staging

# Deploy to staging
helm upgrade --install webapp ./helm/webapp \
  -f config/base/values.yaml \
  -f config/environments/staging/values.yaml \
  --set environment=staging \
  --namespace staging \
  --create-namespace
```

## 12) Best Practices Integration Summary

This tutorial demonstrates:

1. **Configuration Governance**: Base configuration with environment-specific overrides, preventing drift
2. **Secrets Management**: SOPS encryption for secrets at rest, different keys per environment
3. **CI/CD Integration**: Automated promotion pipelines with validation gates
4. **Drift Prevention**: Automated detection of configuration drift between Git and cluster

**Key Integration Points**:
- Configuration promotion enforces environment promotion rules
- Secrets are never auto-promoted (manual review required)
- Drift detection runs in CI/CD pipelines
- Production deployments require manual approval

## 13) Next Steps

- Integrate Hashicorp Vault for dynamic secrets
- Add automated secret rotation
- Implement configuration versioning
- Add rollback capabilities
- Integrate with monitoring for configuration change alerts

---

*This tutorial demonstrates how multiple best practices integrate to create a secure, consistent multi-environment deployment system.*

