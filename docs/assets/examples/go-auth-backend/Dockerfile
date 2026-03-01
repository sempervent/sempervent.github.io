# syntax=docker/dockerfile:1.6
# ─── Stage 1: CA certificates ────────────────────────────────────────────────
FROM alpine:3.20 AS certs
RUN apk add --no-cache ca-certificates

# ─── Stage 2: Build Go binaries ───────────────────────────────────────────────
FROM golang:1.22-bookworm AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build the server — fully static, no CGO, no glibc dependency.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath \
    -o /out/server ./cmd/server

# Build the healthcheck helper — also static.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath \
    -o /out/healthcheck ./cmd/healthcheck

# ─── Stage 3: Minimal scratch runtime ────────────────────────────────────────
FROM scratch AS final

# CA bundle (for outbound TLS if the app ever needs it)
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Binaries
COPY --from=builder /out/server      /server
COPY --from=builder /out/healthcheck /healthcheck

# No shell in scratch — non-root via numeric UID only.
USER 65532

EXPOSE 8080

ENTRYPOINT ["/server"]
