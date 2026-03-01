# go-auth-backend

Minimal Go authentication backend — email/password, bcrypt, JWT, Postgres — packaged in a `FROM scratch` Docker image.

## Quick start

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and JWT_SECRET
docker compose up --build
```

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /healthz | — | Health check |
| POST | /auth/register | — | Create user (`{"email","password"}`) |
| POST | /auth/login | — | Return JWT (`{"email","password"}`) |
| GET | /me | Bearer JWT | Return current user |

## Optional: Adminer UI

```bash
docker compose --profile admin up
# Open http://localhost:8081
```

## Tutorial

See the full tutorial at the site documentation.
