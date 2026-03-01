---
tags:
  - docker
  - security
  - performance
---

# Go Auth Backend: Postgres + JWT + bcrypt → scratch + Compose

**Objective**: Build a minimal but production-shaped authentication backend in Go — local email/password auth, bcrypt hashing, JWT sessions, Postgres — packaged into a `FROM scratch` Docker image and orchestrated with Compose.

No SaaS. No OAuth provider. No 1 GB base image. Just a static binary and a database.

---

## Table of Contents

1. [What You're Building](#1-what-youre-building)
2. [Threat Model](#2-threat-model)
3. [API Overview](#3-api-overview)
4. [DB Schema and Migrations](#4-db-schema-and-migrations)
5. [JWT and Password Hashing](#5-jwt-and-password-hashing)
6. [Multi-Stage Docker Build to scratch](#6-multi-stage-docker-build-to-scratch)
7. [Compose Stack](#7-compose-stack)
8. [Runbook](#8-runbook)
9. [Testing with curl](#9-testing-with-curl)
10. [Troubleshooting](#10-troubleshooting)
11. [Hardening Ideas](#11-hardening-ideas)
12. [See Also](#12-see-also)

---

## 1. What You're Building

```
  ┌─────────────────────────────────────────────────────────┐
  │  Client (curl / browser / frontend)                     │
  └───────────────────┬─────────────────────────────────────┘
                      │ HTTP
                      ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Go API  (FROM scratch, port 8080)                      │
  │  POST /auth/register  → bcrypt hash → INSERT users      │
  │  POST /auth/login     → bcrypt verify → issue JWT       │
  │  GET  /me             → verify JWT → return profile      │
  │  GET  /healthz        → 200 OK                          │
  └───────────────────┬─────────────────────────────────────┘
                      │ pgx/v5 connection pool
                      ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Postgres 16  (docker volume for persistence)           │
  │  users: id UUID, email, password_hash, created_at       │
  └─────────────────────────────────────────────────────────┘
```

Final image size: **~12–15 MB** (static Go binary + CA certs + healthcheck binary).
Compare to: `python:3.11-slim` (~150 MB), `node:20-alpine` (~170 MB).

---

## 2. Threat Model

This backend defends against:

| Threat | Mitigation |
|---|---|
| Password theft from DB breach | bcrypt cost 12 — brute-force impractical |
| Timing attacks on login (email enumeration) | Always run bcrypt check even on not-found |
| Brute-force login | In-memory rate limiter: 10 attempts / IP / minute |
| Token forgery | HMAC-SHA256 JWT with `JWT_SECRET`; short TTL |
| Injection | Parameterized queries via pgx — no string SQL interpolation |
| Privilege escalation | Non-root UID 65532 inside scratch; no shell |

**Out of scope for this tutorial**: CSRF, refresh tokens, account lockout, email verification, MFA.

---

## 3. API Overview

| Method | Path | Body / Headers | Response |
|---|---|---|---|
| GET | `/healthz` | — | `{"status":"ok"}` |
| POST | `/auth/register` | `{"email","password"}` | `{"id":"<uuid>"}` |
| POST | `/auth/login` | `{"email","password"}` | `{"token":"<jwt>"}` |
| GET | `/me` | `Authorization: Bearer <jwt>` | `{"id","email"}` |

All responses are `application/json`. Errors follow `{"error":"message"}`.

---

## 4. DB Schema and Migrations

```sql
-- migrations/0001_init.sql
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
```

**Migration strategy**: on startup, the server runs embedded SQL files via `embed.FS`. All statements use `IF NOT EXISTS` — safe to re-run on every boot. For production, replace with [golang-migrate](https://github.com/golang-migrate/migrate) or [Atlas](https://atlasgo.io/).

---

## 5. JWT and Password Hashing

### bcrypt

```go
// Hash a plaintext password (cost 12 = ~250ms on modern hardware)
hash, err := bcrypt.GenerateFromPassword([]byte(password), 12)

// Verify at login
err = bcrypt.CompareHashAndPassword([]byte(storedHash), []byte(inputPassword))
```

bcrypt is slow by design. Cost 12 means an attacker needs ~250ms per guess — even with the full DB, a brute-force campaign is expensive. Don't lower the cost in production.

### JWT claims structure

```json
{
  "sub":   "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "iat":   1709000000,
  "exp":   1709003600
}
```

Signed with HMAC-SHA256. The `JWT_SECRET` must be ≥ 32 random bytes. Never commit it. Rotate it if leaked (all existing tokens become invalid immediately).

### Timing-attack defense on login

```go
// Always run bcrypt, even when the email doesn't exist in DB.
// Otherwise a fast failure reveals "email not found."
checkHash := storedHash
if dbErr != nil {
    checkHash = "$2a$12$invalidhashfortiming..."  // fake hash — bcrypt still runs
}
checkErr := bcrypt.CompareHashAndPassword([]byte(checkHash), []byte(inputPassword))
if dbErr != nil || checkErr != nil {
    // Always return the same error message
    writeJSON(w, 401, map[string]string{"error": "invalid credentials"})
}
```

---

## 6. Multi-Stage Docker Build to scratch

```
  golang:1.22      alpine:3.20
      │                │
      │ build          │ apk add ca-certificates
      ▼                ▼
  /out/server     /etc/ssl/certs/ca-certificates.crt
  /out/healthcheck
      │                │
      └────────┬────────┘
               ▼
          FROM scratch
          /server         (static Go binary, ~9 MB)
          /healthcheck    (static helper, ~3 MB)
          /etc/ssl/certs/ (CA bundle for outbound TLS)
          USER 65532
          ENTRYPOINT ["/server"]
```

**Why `CGO_ENABLED=0`?** CGO links against glibc. Without it, the binary is fully self-contained — no dynamic linker needed. `scratch` has no dynamic linker.

**Why a separate `healthcheck` binary?** `scratch` has no `curl`, `wget`, or shell. The Compose `healthcheck` directive needs an executable. A tiny Go binary that does `http.Get("http://127.0.0.1:8080/healthz")` fills that role.

### Dockerfile (condensed view)

```dockerfile
FROM alpine:3.20 AS certs
RUN apk add --no-cache ca-certificates

FROM golang:1.22-bookworm AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/server ./cmd/server
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /out/healthcheck ./cmd/healthcheck

FROM scratch AS final
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /out/server      /server
COPY --from=builder /out/healthcheck /healthcheck
USER 65532
EXPOSE 8080
ENTRYPOINT ["/server"]
```

See the full file: `docs/assets/examples/go-auth-backend/Dockerfile`.

---

## 7. Compose Stack

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U authuser -d authdb"]
      interval: 5s
      retries: 10

  api:
    build: .
    environment:
      DATABASE_URL: postgres://authuser:${POSTGRES_PASSWORD}@db:5432/authdb?sslmode=disable
      JWT_SECRET:   ${JWT_SECRET}
    ports: ["8080:8080"]
    depends_on:
      db: { condition: service_healthy }
    healthcheck:
      test: ["/healthcheck"]   # static binary in scratch image
      interval: 10s
      retries: 5

  adminer:                     # optional admin UI
    image: adminer:latest
    ports: ["8081:8080"]
    profiles: [admin]          # only starts with --profile admin

volumes:
  db_data:
```

The Compose file uses `depends_on: condition: service_healthy` — the API won't start until Postgres is accepting connections. No `sleep 5` hacks.

---

## 8. Runbook

```bash
# 1. Clone the example
cp docs/assets/examples/go-auth-backend/ /tmp/go-auth && cd /tmp/go-auth

# 2. Configure
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD (≥16 chars) and JWT_SECRET (≥32 chars)

# 3. Start
docker compose up --build

# 4. Verify
curl http://localhost:8080/healthz
# {"status":"ok"}

# 5. Optional: start with Adminer
docker compose --profile admin up --build
# Open http://localhost:8081
# Server: db | User: authuser | Password: (from .env) | Database: authdb

# 6. Tear down (preserve data)
docker compose down

# 7. Tear down (destroy data)
docker compose down -v
```

---

## 9. Testing with curl

```bash
BASE="http://localhost:8080"

# Register
curl -s -X POST "$BASE/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"supersecret1"}' | jq
# {"id":"550e8400-..."}

# Login
TOKEN=$(curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"supersecret1"}' \
  | jq -r .token)
echo $TOKEN

# /me — authenticated
curl -s "$BASE/me" -H "Authorization: Bearer $TOKEN" | jq
# {"id":"550e8400-...","email":"alice@example.com"}

# /me — no token
curl -s "$BASE/me" | jq
# {"error":"missing token"}

# Login with wrong password
curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"wrong"}' | jq
# {"error":"invalid credentials"}

# Duplicate registration
curl -s -X POST "$BASE/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"supersecret1"}' | jq
# {"error":"registration failed"}  (409 Conflict)
```

---

## 10. Troubleshooting

**`exec /server: no such file or directory`** — The binary was compiled for the wrong OS/arch. Check `GOOS=linux GOARCH=amd64` in the Dockerfile and that you're running on amd64. For ARM (Raspberry Pi), change to `GOARCH=arm64`.

**`connection refused` on api startup** — Postgres isn't healthy yet. The `depends_on: condition: service_healthy` in Compose should handle this. If you're running the binary directly, wait for Postgres to accept connections before starting.

**`pq: password authentication failed`** — `POSTGRES_PASSWORD` in `.env` doesn't match what Postgres was initialized with. If you changed the password after first run, destroy the volume: `docker compose down -v`.

**`JWT_SECRET is required`** — Missing environment variable. Check `.env` is in the same directory as `docker-compose.yaml` and is not empty.

**`bcrypt: password length exceeds 72 bytes`** — bcrypt silently truncates at 72 bytes. For passwords longer than 72 chars, pre-hash with SHA-256 before bcrypt. This backend rejects passwords < 8 chars; add a ≤ 72-char validation if needed.

**Image size larger than expected** — Run `docker history go-auth-backend-api` to inspect layers. The `COPY go.mod go.sum` + `go mod download` layer caches dependencies; ensure it runs before `COPY . .` to preserve cache on code changes.

---

## 11. Hardening Ideas

**Refresh tokens**: issue a short-lived access token (15 min) and a long-lived refresh token stored in DB. The refresh endpoint issues a new access token without re-entering the password.

**Account lockout**: after N failed logins, mark the account locked in DB for M minutes. The in-memory rate limiter is per-IP; a distributed attacker with many IPs bypasses it.

**Email verification**: on register, insert with `verified=false`; send a signed link; flip `verified=true` on click. Block login until verified.

**HTTPS termination**: run Nginx or Caddy in front of the Go server. Never expose plain HTTP to the internet. The CA cert bundle in the scratch image is for outbound calls from Go, not inbound TLS.

**Structured logging**: replace `log.Printf` with `slog` (stdlib since Go 1.21). Log request ID, method, path, status, duration — never log passwords or tokens.

**Metrics**: add a `/metrics` endpoint with Prometheus counters for `register_total`, `login_success_total`, `login_failure_total`. Expose via a separate internal port.

**Secret rotation**: `JWT_SECRET` rotation invalidates all live tokens. Implement a `kid` (key ID) header in JWTs and keep two active signing keys during the rotation window.

---

## 12. See Also

!!! tip "See also"
    - [Multi-Stage Docker: Conda Build → scratch](multistage-conda-to-scratch.md) — the same scratch pattern applied to a Python/Conda stack; explains dynamic linker and CA cert mechanics in depth
    - [Docker & Compose Best Practices](../../best-practices/docker-infrastructure/docker-and-compose.md) — Bake, profiles, security defaults, SBOM scanning
    - [End-to-End Secrets Management](../../best-practices/security/secrets-governance.md) — how to handle `JWT_SECRET` and `POSTGRES_PASSWORD` in production (Vault, SOPS, etc.)
    - [IAM & RBAC Governance](../../best-practices/security/iam-rbac-abac-governance.md) — extend this backend with role-based access control
    - [RKE2 on Raspberry Pi](../docker-infrastructure/rke2-raspberry-pi.md) — deploy this Compose stack to a real Kubernetes cluster

---

*The final image is 12 MB, has no shell, and runs as UID 65532. Ship it.*
