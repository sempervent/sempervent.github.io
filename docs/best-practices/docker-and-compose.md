# Docker & Compose Best Practices

This document establishes the definitive approach to Docker and Compose workflows that refuse to break. Every command is copy-paste runnable, every configuration is auditable, and every practice eliminates "works on my machine" entropy. We enforce speed, determinism, and repeatability through Bake for orchestration, profiles for intent, and security by default.

## 1. Multi-Stage Dockerfiles That Don't Lie

```dockerfile
# docker/Dockerfile
# syntax=docker/dockerfile:1.9

ARG RUNTIME=python:3.12-slim
ARG BUILDER=python:3.12

FROM ${BUILDER} AS build
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build -o /dist

FROM ${RUNTIME} AS run
WORKDIR /app
# Non-root by default
RUN useradd -r -u 10001 app && chown -R app:app /app
USER app
# Provenance-friendly: wheel install, no network at runtime
COPY --from=build /dist/*.whl /tmp/pkg.whl
RUN python -m pip install --no-cache-dir /tmp/pkg.whl && rm /tmp/pkg.whl
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import yourpkg; print('ok')"
ENTRYPOINT ["yourpkg"]
```

**Why:** Multi-stage keeps runtime lean, reproducible, and non-root. Healthcheck makes orchestration honest about container state.

## 2. Buildx: Multi-Arch, Cache, SBOM, Provenance

```bash
docker buildx create --use --name blade
docker buildx inspect --bootstrap
docker buildx bake -f docker-bake.hcl --print
```

**Why:** Buildx unlocks multi-arch, cache exports, and attestations. We'll centralize strategy in Bake for consistency.

## 3. Bake HCL: Matrices, Caching, Attestations

```hcl
# docker-bake.hcl
group "default" {
  targets = ["app"]
}

variable "REGISTRY" { default = "ghcr.io/yourorg" }
variable "NAME"     { default = "yourpkg" }
variable "VERSION"  { default = "0.1.0" }

# Matrix of platforms
variable "PLATFORMS" {
  default = "linux/amd64,linux/arm64"
}

target "base" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = split(",", PLATFORMS)
  pull       = true

  # Enable SBOM + provenance
  attest = ["type=sbom", "type=provenance,mode=max"]
  # Cache: registry or local
  cache-from = [
    "type=registry,ref=${REGISTRY}/${NAME}:cache",
  ]
  cache-to = [
    "type=registry,ref=${REGISTRY}/${NAME}:cache,mode=max,compression=zstd",
  ]
}

# Production image
target "app" {
  inherits = ["base"]
  tags = [
    "${REGISTRY}/${NAME}:latest",
    "${REGISTRY}/${NAME}:${VERSION}",
    "${REGISTRY}/${NAME}:${VERSION}-{{.platform}}",
  ]
  # Build args -> Dockerfile ARGs
  args = {
    RUNTIME = "python:3.12-slim"
    BUILDER = "python:3.12"
  }
  # Push by default in CI; override locally with --set
  output = ["type=registry"]
}

# Local dev (no push, local cache)
target "dev" {
  inherits = ["base"]
  tags     = ["${NAME}:dev"]
  output   = ["type=docker"]
}
```

**Why:** Bake makes multi-arch, cache, and attestations declarative and repeatable. CI can flip one flag to push.

**Usage:**
```bash
# Local debug
docker buildx bake dev

# CI publish (push to registry)
docker buildx bake app --set *.output=type=registry
```

## 4. Compose With Profiles: Dev / Test / Prod Without Forking YAML

```yaml
# docker-compose.yaml
name: yourstack

services:
  app:
    image: ghcr.io/yourorg/yourpkg:latest
    depends_on: [db]
    environment:
      YOURPKG_LOG: "info"
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
    profiles: ["prod","dev"]

  app-dev:
    image: yourpkg:dev
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: run
    volumes:
      - ./src:/app/src:ro
    environment:
      YOURPKG_LOG: "debug"
    command: ["yourpkg", "--reload"]
    profiles: ["dev"]

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: "postgres"
    volumes:
      - db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 3s
      retries: 10
    profiles: ["dev","test","prod"]

  test:
    image: yourpkg:dev
    depends_on: [db]
    command: ["bash","-lc","pytest -q --maxfail=1"]
    profiles: ["test"]

volumes:
  db: {}
```

**Why:** One file, many realities. Profiles expose intent: dev, test, prod—no YAML hydra.

**Usage:**
```bash
# Dev stack with hot reload
docker compose --profile dev up --build

# Test pipeline (ephemeral)
docker compose --profile test up --abort-on-container-exit --exit-code-from test

# Prod-like (no dev services)
docker compose --profile prod up -d
```

## 5. Secrets, SSH, and Build-Time Inputs (Safely)

### Build-time secrets via BuildKit

```bash
docker buildx build \
  --secret id=pypi_token,env=PYPI_TOKEN \
  --ssh default \
  -f docker/Dockerfile .
```

### Dockerfile usage

```dockerfile
RUN --mount=type=secret,id=pypi_token \
    --mount=type=ssh \
    bash -lc 'echo "token loaded" >/dev/null'
```

**Why:** Keep credentials out of layers and logs. Use ephemeral mounts for build-time secrets.

## 6. Image Hygiene: Labels, User, Minimal Surface

```dockerfile
LABEL org.opencontainers.image.title="yourpkg" \
      org.opencontainers.image.source="https://github.com/yourorg/yourpkg" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"

# Already shown: USER app
# Also consider dropping Linux capabilities if you must escalate:
# RUN setcap cap_net_bind_service=+ep /usr/local/bin/yourpkg
```

**Why:** Metadata aids provenance and supply chain tracking. Non-root and fewer caps reduce blast radius.

## 7. Caching Strategy That Actually Works

### Order matters: copy only manifests first

```dockerfile
# Copy dependency manifests first (cache-friendly)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Then copy source code (changes frequently)
COPY src ./src
```

### Bake cache: publish cache in CI and pull locally

```hcl
cache-from = [
  "type=registry,ref=${REGISTRY}/${NAME}:cache",
]
cache-to = [
  "type=registry,ref=${REGISTRY}/${NAME}:cache,mode=max,compression=zstd",
]
```

### Layer busting: avoid --no-cache

```bash
# Bump arguments or pins intentionally
docker buildx bake app --set base.args.BUILD_DATE=$(date -Iseconds)
```

**Why:** Cache is a weapon; use it consciously. Registry cache enables team-wide build acceleration.

## 8. Resource Limits, Health & Restart Policies

```yaml
# compose.prod.yaml (optional)
services:
  app:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.5"
          memory: 768M
        reservations:
          cpus: "0.5"
          memory: 256M
    ulimits:
      nofile: 65536
```

**Why:** Put the container on a leash. Health and ulimits prevent slow-motion failures and resource exhaustion.

## 9. CI: Build, Test, Bake, Push, Attest

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Bake (multi-arch, sbom, provenance)
        run: |
          docker buildx bake -f docker-bake.hcl app \
            --set *.output=type=registry \
            --set base.attest=type=sbom \
            --set base.attest=type=provenance,mode=max
```

**Why:** CI mirrors local Bake. Provenance and SBOM emitted on every push ensures supply-chain sanity.

## 10. Local Dev Loop With Compose + Bake

```bash
# Build dev image with cache
docker buildx bake dev
docker compose --profile dev up --build
```

**Why:** The loop should be quick and merciless. Slow loops breed apathy and context switching.

## 11. Troubleshooting Playbook (Short, Surgical)

### Multi-arch fails on native arm64
→ Ensure QEMU installed and builder bootstrapped:
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

### Cache not used
→ Check cache-from ref exists; identical Dockerfile path and args required.

### Healthcheck never healthy
→ Test command inside container; lower interval only after it passes manually:
```bash
docker run --rm yourpkg:latest python -c "import yourpkg; print('ok')"
```

### Compose profiles ignored
→ Ensure docker compose (v2) not docker-compose (v1); pass --profile:
```bash
docker compose --profile dev up
```

**Why:** These solutions address 90% of Docker issues. The key is understanding that reproducible builds require consistent tooling and explicit configuration.

## 12. TL;DR (Zero → Green)

```bash
# 1) Buildx & QEMU (once)
docker buildx create --use
docker run --privileged --rm tonistiigi/binfmt --install all

# 2) Dev image + stack
docker buildx bake dev
docker compose --profile dev up --build

# 3) Release (push with attestations)
docker buildx bake app --set *.output=type=registry
```

**Why:** This sequence establishes a working Docker environment with multi-arch support, caching, and attestations in under 5 minutes. Each command builds on the previous, ensuring a deterministic setup that matches production CI environments.
