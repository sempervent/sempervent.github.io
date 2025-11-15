# NGINX Best Practices: Patterns, Hardening, and Multi-API Replication

**Objective**: Master production-grade NGINX configuration for reverse proxying, load balancing, SSL termination, and multi-API gateways. When you need to front multiple services, replicate traffic for testing, and harden your edge layer—these best practices become your foundation.

## Introduction

NGINX is the Swiss Army knife of modern infrastructure. It's a reverse proxy, API gateway-lite, SSL terminator, load balancer, and static asset server rolled into one. Most applications don't need a full API gateway—NGINX handles 90% of use cases with better performance and simpler operations.

**What This Guide Covers**:

- **Practical Patterns**: Reverse proxy, load balancing, path/host-based routing
- **Hardened Defaults**: Security headers, rate limiting, timeouts
- **Multi-API Setup**: Fronting multiple backend services from a single NGINX instance
- **Traffic Mirroring**: Replicating requests to shadow APIs for canary testing

**What This Guide Assumes**:

- You understand HTTP basics (methods, headers, status codes)
- You know what a reverse proxy does
- You have basic Linux and Docker experience
- You're deploying NGINX standalone or in containers

We're not here to explain HTTP. We're here to show you how to configure NGINX correctly.

## Core NGINX Concepts

### Configuration Structure

NGINX config is hierarchical:

```
http {
    # Global settings
    
    upstream backend {
        # Backend server definitions
    }
    
    server {
        # Virtual host / server block
        
        location / {
            # Request routing rules
        }
    }
}
```

**Key Blocks**:

- **`http {}`**: Top-level HTTP context, contains all HTTP configuration
- **`server {}`**: Virtual host definition, matches requests by `server_name` and `listen`
- **`location {}`**: Request routing within a server, matches by URI pattern
- **`upstream {}`**: Backend server pool definition, used with `proxy_pass`

### Essential Directives

**Request Routing**:
- `listen`: Port and protocol (e.g., `listen 443 ssl http2`)
- `server_name`: Hostname matching (e.g., `server_name api.example.com`)
- `location`: URI pattern matching (e.g., `location /api/`)
- `proxy_pass`: Forward requests to upstream (e.g., `proxy_pass http://backend`)

**Upstream Configuration**:
- `upstream`: Define backend server pool
- `server`: Individual backend server in pool
- `keepalive`: Connection pool size for upstream

**Proxy Headers**:
- `proxy_set_header`: Set headers sent to upstream
- `proxy_pass_request_headers`: Control header forwarding

**Performance & Reliability**:
- `proxy_read_timeout`: Time to wait for upstream response
- `proxy_connect_timeout`: Time to establish upstream connection
- `client_max_body_size`: Maximum request body size
- `keepalive_timeout`: HTTP keepalive duration

**Includes**:
- `include`: Include other config files (e.g., `include /etc/nginx/conf.d/*.conf`)

## Baseline Best Practices

### File Layout

**Recommended Structure**:

```
/etc/nginx/
├── nginx.conf              # Main config
├── conf.d/                 # Site configs
│   ├── api-gateway.conf
│   └── static.conf
├── upstreams/              # Upstream definitions
│   ├── user-api.conf
│   └── orders-api.conf
└── snippets/              # Reusable config snippets
    ├── ssl-params.conf
    └── security-headers.conf
```

**Main nginx.conf**:

```nginx
# /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    log_format api '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent $request_time '
                   '$upstream_response_time "$http_x_request_id"';
    
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;
    
    # Security
    server_tokens off;
    client_max_body_size 10m;
    
    # Timeouts
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
    
    # Buffers
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;
    
    # Include site configs
    include /etc/nginx/conf.d/*.conf;
}
```

### Logging

**Custom API Log Format**:

```nginx
log_format api_detailed '$remote_addr - $remote_user [$time_local] '
                        '"$request" $status $body_bytes_sent '
                        '$request_time $upstream_response_time '
                        '"$http_x_request_id" "$http_user_agent" '
                        '$upstream_addr $upstream_status';
```

**Per-Service Logging**:

```nginx
server {
    server_name api.example.com;
    access_log /var/log/nginx/api-access.log api_detailed;
    error_log /var/log/nginx/api-error.log warn;
    
    location / {
        # ...
    }
}
```

### Security Hardening

**Hide Version**:

```nginx
http {
    server_tokens off;
}
```

**Limit HTTP Methods**:

```nginx
location /api/ {
    limit_except GET POST PUT DELETE {
        deny all;
    }
    proxy_pass http://backend;
}
```

**Rate Limiting**:

```nginx
# Define rate limit zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=1r/s;

# Apply to locations
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    proxy_pass http://backend;
}

location /api/login {
    limit_req zone=login_limit burst=3 nodelay;
    proxy_pass http://backend;
}
```

**Connection Limiting**:

```nginx
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    limit_conn conn_limit 10;
    # ...
}
```

### Client Limits

**Global and Per-Location**:

```nginx
http {
    client_max_body_size 10m;  # Default
}

server {
    location /api/upload {
        client_max_body_size 100m;  # Override for uploads
        proxy_pass http://backend;
    }
}
```

### Compression

**Gzip Configuration**:

```nginx
http {
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 1000;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/rss+xml
        font/truetype
        font/opentype
        application/vnd.ms-fontobject
        image/svg+xml;
}
```

**Note**: If upstream compresses responses, NGINX won't recompress. Use `gzip_proxied any` to compress even if upstream sets `Content-Encoding`.

## Common NGINX Patterns

### 1. Reverse Proxy to Single Upstream

**Basic Pattern**:

```nginx
upstream backend {
    server backend.example.com:8080;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Header Explanation**:

- `Host`: Original host header (upstream needs this for virtual hosting)
- `X-Real-IP`: Client's real IP (single value)
- `X-Forwarded-For`: Client IP chain (for multiple proxies)
- `X-Forwarded-Proto`: Original protocol (http/https)

**Complete Example**:

```nginx
upstream app_backend {
    server app:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;
    
    access_log /var/log/nginx/api.log api_detailed;
    
    location / {
        proxy_pass http://app_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 2. Load Balancing Across Multiple Upstreams

**Round-Robin (Default)**:

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;
}
```

**Least Connections**:

```nginx
upstream backend {
    least_conn;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;
}
```

**IP Hash (Session Persistence)**:

```nginx
upstream backend {
    ip_hash;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;
}
```

**With Health Checks**:

```nginx
upstream backend {
    server backend1.example.com:8080 max_fails=3 fail_timeout=30s;
    server backend2.example.com:8080 max_fails=3 fail_timeout=30s;
    server backend3.example.com:8080 max_fails=3 fail_timeout=30s backup;
}
```

**Complete Example**:

```nginx
upstream api_backend {
    least_conn;
    server api1:8080 max_fails=3 fail_timeout=30s;
    server api2:8080 max_fails=3 fail_timeout=30s;
    server api3:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. SSL/TLS Termination

**Basic SSL Configuration**:

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/nginx/ssl/api.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/api.example.com.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    location / {
        proxy_pass http://backend;
        # ... proxy headers ...
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

**With Let's Encrypt**:

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    
    location / {
        proxy_pass http://backend;
        # ... proxy headers ...
    }
}
```

**Modern TLS Configuration**:

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;
```

### 4. Path-Based Routing to Multiple Services

**Location Matching Order**:

NGINX evaluates locations in this order:
1. Exact match (`=`)
2. Longest prefix match (`^~`)
3. Regular expression (`~`, `~*`)
4. Prefix match (default)

**Example: Multiple APIs**:

```nginx
upstream user_api {
    server user-api:8080;
}

upstream order_api {
    server order-api:8080;
}

upstream analytics_api {
    server analytics-api:8080;
}

server {
    listen 80;
    server_name api.example.com;
    
    # Health check endpoint
    location = /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # User API
    location /api/users {
        proxy_pass http://user_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Order API
    location /api/orders {
        proxy_pass http://order_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Analytics API
    location /api/analytics {
        proxy_pass http://analytics_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static frontend
    location / {
        root /var/www/frontend;
        try_files $uri $uri/ /index.html;
    }
}
```

**Important**: When using `proxy_pass` with a URI path, NGINX replaces the matched location path. Use trailing slashes consistently:

```nginx
# Correct: /api/users/ -> http://backend/api/users/
location /api/users/ {
    proxy_pass http://backend/;
}

# Correct: /api/users -> http://backend/api/users
location /api/users {
    proxy_pass http://backend;
}
```

### 5. Host-Based Routing (Multiple Domains)

**Multiple Virtual Hosts**:

```nginx
# API subdomain
server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://api_backend;
        # ... proxy headers ...
    }
}

# Admin subdomain
server {
    listen 80;
    server_name admin.example.com;
    
    location / {
        proxy_pass http://admin_backend;
        # ... proxy headers ...
    }
}

# Default catch-all
server {
    listen 80 default_server;
    server_name _;
    return 444;  # Close connection
}
```

**When to Use Host-Based vs Path-Based**:

- **Host-based**: Different services, different domains, different SSL certs
- **Path-based**: Same domain, different services, unified SSL cert

### 6. Caching

**Basic Proxy Cache**:

```nginx
http {
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;
    
    server {
        location /api/ {
            proxy_pass http://backend;
            proxy_cache api_cache;
            proxy_cache_valid 200 302 10m;
            proxy_cache_valid 404 1m;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            add_header X-Cache-Status $upstream_cache_status;
        }
    }
}
```

**Cache Bypass**:

```nginx
location /api/ {
    proxy_pass http://backend;
    proxy_cache api_cache;
    
    # Bypass cache for POST/PUT/DELETE
    set $no_cache 0;
    if ($request_method !~ ^(GET|HEAD)$) {
        set $no_cache 1;
    }
    proxy_cache_bypass $no_cache;
    proxy_no_cache $no_cache;
}
```

## Multi-API Gateway Pattern

NGINX excels at fronting multiple APIs from a single instance. Here's how to structure it properly.

### Upstream Definitions

**Separate Upstream Files**:

```nginx
# /etc/nginx/upstreams/user-api.conf
upstream user_api {
    least_conn;
    server user-api-1:8080 max_fails=3 fail_timeout=30s;
    server user-api-2:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# /etc/nginx/upstreams/orders-api.conf
upstream orders_api {
    least_conn;
    server orders-api-1:8080 max_fails=3 fail_timeout=30s;
    server orders-api-2:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# /etc/nginx/upstreams/analytics-api.conf
upstream analytics_api {
    least_conn;
    server analytics-api-1:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}
```

**Include in Main Config**:

```nginx
http {
    include /etc/nginx/upstreams/*.conf;
    include /etc/nginx/conf.d/*.conf;
}
```

### Path-Based Multi-API Routing

**Complete Server Block**:

```nginx
# /etc/nginx/conf.d/api-gateway.conf
server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    ssl_certificate /etc/nginx/ssl/api.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/api.example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers off;
    
    # Global proxy settings
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Request-ID $request_id;
    
    # Health check
    location = /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # User API
    location /api/users {
        access_log /var/log/nginx/user-api.log api_detailed;
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://user_api;
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        
        # Health check endpoint
        location = /api/users/healthz {
            proxy_pass http://user_api/healthz;
            access_log off;
        }
    }
    
    # Orders API
    location /api/orders {
        access_log /var/log/nginx/orders-api.log api_detailed;
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://orders_api;
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;  # Longer timeout for order processing
        
        location = /api/orders/healthz {
            proxy_pass http://orders_api/healthz;
            access_log off;
        }
    }
    
    # Analytics API
    location /api/analytics {
        access_log /var/log/nginx/analytics-api.log api_detailed;
        limit_req zone=api_limit burst=50 nodelay;  # Higher limit for analytics
        
        proxy_pass http://analytics_api;
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        
        location = /api/analytics/healthz {
            proxy_pass http://analytics_api/healthz;
            access_log off;
        }
    }
    
    # Default: 404
    location / {
        return 404;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

### Host-Based Multi-API Routing

**Alternative: Separate Domains**:

```nginx
# User API
server {
    listen 443 ssl http2;
    server_name users.api.example.com;
    
    ssl_certificate /etc/nginx/ssl/users.api.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/users.api.example.com.key;
    
    location / {
        proxy_pass http://user_api;
        # ... proxy headers ...
    }
}

# Orders API
server {
    listen 443 ssl http2;
    server_name orders.api.example.com;
    
    ssl_certificate /etc/nginx/ssl/orders.api.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/orders.api.example.com.key;
    
    location / {
        proxy_pass http://orders_api;
        # ... proxy headers ...
    }
}
```

## Replicating / Mirroring Multiple APIs

Traffic mirroring (shadowing) sends a copy of requests to a secondary upstream while the primary upstream handles the actual response. This is invaluable for testing new API versions with real production traffic.

### Concept

**How It Works**:

1. Client sends request to NGINX
2. NGINX forwards to primary upstream (normal flow)
3. NGINX mirrors request to shadow upstream (background, response ignored)
4. Client receives response from primary upstream only

**Use Cases**:

- **Canary Testing**: Test new API version with real traffic
- **Load Testing**: Stress test candidate service with production load patterns
- **A/B Testing**: Compare behavior between versions
- **Debugging**: Replay production requests in staging environment

### Basic Mirroring Pattern

**Single API Mirror**:

```nginx
upstream user_api_v1 {
    server user-api-v1:8080;
}

upstream user_api_v2 {
    server user-api-v2:8080;
}

server {
    listen 80;
    server_name api.example.com;
    
    location /api/users {
        # Mirror to v2
        mirror /mirror_users;
        mirror_request_body on;
        
        # Primary: v1
        proxy_pass http://user_api_v1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Internal location for mirroring
    location = /mirror_users {
        internal;
        proxy_pass http://user_api_v2$request_uri;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass_request_body on;
        proxy_set_header X-Mirrored "true";  # Mark as mirrored request
    }
}
```

**Key Points**:

- `mirror` directive specifies internal location for mirroring
- `mirror_request_body on` includes request body in mirrored request
- Internal location uses `internal` directive (not accessible externally)
- Mark mirrored requests with custom header for logging/filtering

### Multiple APIs with Mirroring

**Extend to Multiple Services**:

```nginx
# Upstreams
upstream user_api_v1 {
    server user-api-v1:8080;
}

upstream user_api_v2 {
    server user-api-v2:8080;
}

upstream orders_api_v1 {
    server orders-api-v1:8080;
}

upstream orders_api_v2 {
    server orders-api-v2:8080;
}

upstream analytics_api_v1 {
    server analytics-api-v1:8080;
}

# Analytics doesn't have v2 yet, so no mirroring

server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    # ... SSL config ...
    
    # User API with mirroring
    location /api/users {
        mirror /mirror_users;
        mirror_request_body on;
        
        proxy_pass http://user_api_v1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        
        access_log /var/log/nginx/user-api.log api_detailed;
    }
    
    location = /mirror_users {
        internal;
        proxy_pass http://user_api_v2$request_uri;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Mirrored "true";
        proxy_pass_request_body on;
        proxy_read_timeout 300s;  # Longer timeout for shadow
        access_log /var/log/nginx/user-api-v2-shadow.log api_detailed;
    }
    
    # Orders API with mirroring
    location /api/orders {
        mirror /mirror_orders;
        mirror_request_body on;
        
        proxy_pass http://orders_api_v1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        
        access_log /var/log/nginx/orders-api.log api_detailed;
    }
    
    location = /mirror_orders {
        internal;
        proxy_pass http://orders_api_v2$request_uri;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Mirrored "true";
        proxy_pass_request_body on;
        proxy_read_timeout 300s;
        access_log /var/log/nginx/orders-api-v2-shadow.log api_detailed;
    }
    
    # Analytics API (no mirroring)
    location /api/analytics {
        proxy_pass http://analytics_api_v1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        
        access_log /var/log/nginx/analytics-api.log api_detailed;
    }
}
```

### Metrics and Logging

**Separate Logs for Shadow Traffic**:

```nginx
log_format shadow '$remote_addr [$time_local] "$request" '
                  '$status $body_bytes_sent $request_time '
                  '$upstream_response_time "$http_x_request_id" '
                  'shadow="true"';

location = /mirror_users {
    internal;
    access_log /var/log/nginx/user-api-v2-shadow.log shadow;
    # ... rest of config ...
}
```

**Compare Metrics**:

```bash
# Compare response times
awk '{print $7}' /var/log/nginx/user-api.log | sort -n | tail -1  # Primary p95
awk '{print $7}' /var/log/nginx/user-api-v2-shadow.log | sort -n | tail -1  # Shadow p95

# Compare error rates
grep -c " 5[0-9][0-9] " /var/log/nginx/user-api.log  # Primary errors
grep -c " 5[0-9][0-9] " /var/log/nginx/user-api-v2-shadow.log  # Shadow errors
```

### Preventing Side Effects in Shadow APIs

**High-Level Strategies**:

1. **Read-Only Mode**: Configure shadow API to reject writes (return 405 for POST/PUT/DELETE)
2. **Separate Database**: Shadow API uses test/staging database
3. **Request Filtering**: Filter out write operations in mirror location
4. **Idempotency Keys**: Use idempotency to prevent duplicate processing

**Example: Filter Writes**:

```nginx
location = /mirror_users {
    internal;
    
    # Only mirror GET requests
    if ($request_method !~ ^(GET|HEAD)$) {
        return 200;  # Skip mirroring for writes
    }
    
    proxy_pass http://user_api_v2$request_uri;
    # ... rest of config ...
}
```

## Complete Multi-API, Mirror-Enabled Config

Here's a production-ready configuration that brings everything together:

```nginx
# /etc/nginx/nginx.conf (main)
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 2048;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging formats
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    log_format api '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent $request_time '
                   '$upstream_response_time "$http_x_request_id" '
                   'upstream="$upstream_addr"';
    
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10m;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss;
    
    # Security
    server_tokens off;
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login_limit:10m rate=1r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    
    # Proxy defaults
    proxy_connect_timeout 5s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;
    
    # Upstream definitions
    upstream user_api_v1 {
        least_conn;
        server user-api-v1:8080 max_fails=3 fail_timeout=30s;
        server user-api-v1-replica:8080 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream user_api_v2 {
        server user-api-v2:8080;
        keepalive 32;
    }
    
    upstream orders_api_v1 {
        least_conn;
        server orders-api-v1:8080 max_fails=3 fail_timeout=30s;
        server orders-api-v1-replica:8080 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream orders_api_v2 {
        server orders-api-v2:8080;
        keepalive 32;
    }
    
    upstream analytics_api {
        least_conn;
        server analytics-api-1:8080 max_fails=3 fail_timeout=30s;
        server analytics-api-2:8080 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    # Main API gateway server
    server {
        listen 443 ssl http2;
        server_name api.example.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/api.example.com.crt;
        ssl_certificate_key /etc/nginx/ssl/api.example.com.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 1d;
        
        # Connection limits
        limit_conn conn_limit 20;
        
        # Global proxy headers
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Generate request ID if not present
        set $request_id $request_id;
        if ($http_x_request_id = "") {
            set $request_id $request_id_generated;
        }
        proxy_set_header X-Request-ID $request_id;
        
        # Health check endpoint
        location = /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # User API (with mirroring to v2)
        location /api/users {
            mirror /mirror_users;
            mirror_request_body on;
            
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://user_api_v1;
            proxy_connect_timeout 5s;
            proxy_read_timeout 30s;
            
            access_log /var/log/nginx/user-api.log api;
        }
        
        location = /mirror_users {
            internal;
            proxy_pass http://user_api_v2$request_uri;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Mirrored "true";
            proxy_set_header X-Request-ID $request_id;
            proxy_pass_request_body on;
            proxy_read_timeout 300s;  # Longer timeout for shadow
            access_log /var/log/nginx/user-api-v2-shadow.log api;
        }
        
        # User API health check
        location = /api/users/healthz {
            proxy_pass http://user_api_v1/healthz;
            access_log off;
        }
        
        # Orders API (with mirroring to v2)
        location /api/orders {
            mirror /mirror_orders;
            mirror_request_body on;
            
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://orders_api_v1;
            proxy_connect_timeout 5s;
            proxy_read_timeout 60s;  # Longer for order processing
            
            access_log /var/log/nginx/orders-api.log api;
        }
        
        location = /mirror_orders {
            internal;
            proxy_pass http://orders_api_v2$request_uri;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Mirrored "true";
            proxy_set_header X-Request-ID $request_id;
            proxy_pass_request_body on;
            proxy_read_timeout 300s;
            access_log /var/log/nginx/orders-api-v2-shadow.log api;
        }
        
        # Orders API health check
        location = /api/orders/healthz {
            proxy_pass http://orders_api_v1/healthz;
            access_log off;
        }
        
        # Analytics API (no mirroring)
        location /api/analytics {
            limit_req zone=api_limit burst=50 nodelay;  # Higher limit
            
            proxy_pass http://analytics_api;
            proxy_connect_timeout 5s;
            proxy_read_timeout 30s;
            
            access_log /var/log/nginx/analytics-api.log api;
        }
        
        # Analytics API health check
        location = /api/analytics/healthz {
            proxy_pass http://analytics_api/healthz;
            access_log off;
        }
        
        # Default: 404
        location / {
            return 404;
        }
    }
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name api.example.com;
        return 301 https://$server_name$request_uri;
    }
}
```

## Operational & Performance Best Practices

### Keepalive Connections

**Upstream Keepalive**:

```nginx
upstream backend {
    server backend1:8080;
    server backend2:8080;
    keepalive 32;  # Maintain 32 idle connections per worker
}

server {
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # Clear Connection header for keepalive
    }
}
```

**Benefits**: Reduces connection overhead by 10-100x. Essential for high-throughput APIs.

### Worker Tuning

**Process and Connection Sizing**:

```nginx
worker_processes auto;  # Match CPU cores
worker_rlimit_nofile 65535;  # Increase file descriptor limit

events {
    worker_connections 2048;  # Connections per worker
    use epoll;  # Linux: use epoll for better performance
}
```

**Calculation**: `worker_processes × worker_connections = max concurrent connections`

Example: 4 workers × 2048 connections = 8,192 max concurrent connections.

### Rate Limiting Patterns

**Per-IP Rate Limiting**:

```nginx
# Define zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;

# Apply to locations
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    proxy_pass http://backend;
}

location /api/login {
    limit_req zone=login_limit burst=3 nodelay;
    proxy_pass http://backend;
}

location /api/upload {
    limit_req zone=upload_limit burst=5 nodelay;
    client_max_body_size 100m;
    proxy_pass http://backend;
}
```

**Per-Service Rate Limiting**:

```nginx
# Rate limit by API key or user ID
map $http_x_api_key $api_key_zone {
    default "api_limit";
    "key1" "api_limit_premium";
    "key2" "api_limit_premium";
}

limit_req_zone $api_key_zone zone=api_limit_premium:10m rate=100r/s;

location /api/ {
    limit_req zone=$api_key_zone burst=50 nodelay;
    proxy_pass http://backend;
}
```

### Protection Against Abuse

**Timeouts**:

```nginx
http {
    client_body_timeout 12;      # Time to read client body
    client_header_timeout 12;     # Time to read client headers
    send_timeout 10;              # Time between send operations
    keepalive_timeout 65;         # Keepalive connection timeout
}
```

**Connection Limits**:

```nginx
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    limit_conn conn_limit 10;  # Max 10 connections per IP
    # ...
}
```

### Zero-Downtime Reloads

**Process**:

```bash
# 1. Test configuration
nginx -t

# 2. Reload (graceful, doesn't drop connections)
nginx -s reload

# Or send HUP signal
kill -HUP $(cat /var/run/nginx.pid)
```

**What Happens**:
- Master process reads new config
- Starts new worker processes with new config
- Old workers finish current requests, then exit
- Zero dropped connections (if config is valid)

**Validation Script**:

```bash
#!/bin/bash
# validate-and-reload.sh

if nginx -t; then
    echo "Configuration valid, reloading..."
    nginx -s reload
    echo "Reload complete"
else
    echo "Configuration invalid, not reloading"
    exit 1
fi
```

### Observability

**Enhanced Logging Format**:

```nginx
log_format detailed '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '$request_time $upstream_response_time '
                    '$upstream_connect_time '
                    '"$http_x_request_id" "$http_user_agent" '
                    'upstream="$upstream_addr" '
                    'cache="$upstream_cache_status"';

access_log /var/log/nginx/access.log detailed;
```

**Request ID Generation**:

```nginx
# Generate UUID if not present
map $http_x_request_id $request_id {
    default $http_x_request_id;
    "" $request_id_generated;
}

# Or use Lua (if nginx-lua module available)
# set $request_id $request_id;
```

**Key Metrics to Track**:

- `$request_time`: Total request time (client → NGINX → client)
- `$upstream_response_time`: Time to get response from upstream
- `$upstream_connect_time`: Time to connect to upstream
- `$upstream_addr`: Which backend served the request
- `$upstream_status`: Upstream response status

**Centralized Logging**:

```nginx
# Send logs to syslog (forward to ELK, Loki, etc.)
access_log syslog:server=log-aggregator:514,facility=local7,tag=nginx,severity=info detailed;
error_log syslog:server=log-aggregator:514,facility=local7,tag=nginx,severity=warn;
```

## NGINX with Docker / Compose

### Custom NGINX Dockerfile

```dockerfile
# Dockerfile
FROM nginx:1.25-alpine

# Copy custom configs
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/
COPY upstreams/ /etc/nginx/upstreams/
COPY ssl/ /etc/nginx/ssl/

# Create log directories
RUN mkdir -p /var/log/nginx

# Expose ports
EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose Example

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./conf.d:/etc/nginx/conf.d:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx-logs:/var/log/nginx
    depends_on:
      - user-api-v1
      - user-api-v2
      - orders-api-v1
      - orders-api-v2
      - analytics-api
    networks:
      - api-network
    restart: unless-stopped

  user-api-v1:
    image: user-api:v1
    networks:
      - api-network
    # ... other config ...

  user-api-v2:
    image: user-api:v2
    networks:
      - api-network
    # ... other config ...

  orders-api-v1:
    image: orders-api:v1
    networks:
      - api-network
    # ... other config ...

  orders-api-v2:
    image: orders-api:v2
    networks:
      - api-network
    # ... other config ...

  analytics-api:
    image: analytics-api:latest
    networks:
      - api-network
    # ... other config ...

volumes:
  nginx-logs:

networks:
  api-network:
    driver: bridge
```

### Hot Reload in Containers

**Option 1: Volume Mount + Reload**:

```bash
# Edit config on host
vim nginx.conf

# Reload inside container
docker exec nginx nginx -s reload
```

**Option 2: ConfigMap (Kubernetes)**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    # ... config ...
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      containers:
      - name: nginx
        image: nginx:1.25-alpine
        volumeMounts:
        - name: config
          mountPath: /etc/nginx
        lifecycle:
          postStart:
            exec:
              command: ["/bin/sh", "-c", "nginx -s reload"]
      volumes:
      - name: config
        configMap:
          name: nginx-config
```

**Trade-offs**:

- **Volume mounts**: Easy development, but requires container exec for reload
- **ConfigMaps**: Kubernetes-native, but reload requires pod restart or lifecycle hooks
- **Rebuild image**: Most secure, but slower iteration

## Summary & Maturity Path

### Phase 1: Start Simple

**Single Reverse Proxy**:

```nginx
upstream app {
    server app:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Goals**: Get basic proxying working, understand headers.

### Phase 2: Add Load Balancing

**Multiple Backends**:

```nginx
upstream app {
    least_conn;
    server app1:8000;
    server app2:8000;
    server app3:8000;
}
```

**Goals**: Distribute load, handle backend failures.

### Phase 3: Add SSL Termination

**HTTPS + Redirect**:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ... SSL config ...
}

server {
    listen 80;
    return 301 https://$server_name$request_uri;
}
```

**Goals**: Encrypt traffic, meet compliance requirements.

### Phase 4: Add Path/Host-Based Routing

**Multiple APIs**:

```nginx
location /api/users {
    proxy_pass http://user_api;
}

location /api/orders {
    proxy_pass http://orders_api;
}
```

**Goals**: Front multiple services from single NGINX instance.

### Phase 5: Add Mirrored/Replicated APIs

**Traffic Shadowing**:

```nginx
location /api/users {
    mirror /mirror_users;
    proxy_pass http://user_api_v1;
}

location = /mirror_users {
    internal;
    proxy_pass http://user_api_v2$request_uri;
}
```

**Goals**: Test new versions with production traffic, canary deployments.

### Phase 6: Add Hardening & Observability

**Complete Production Config**:

- Rate limiting
- Connection limiting
- Security headers
- Enhanced logging
- Caching where appropriate
- Health checks
- Zero-downtime reloads

**Goals**: Production-ready, secure, observable.

### Phase 7: Document & Standardize

**Team Reuse**:

- Template configs for common patterns
- Documentation for each service
- Runbooks for common operations
- Monitoring dashboards

**Goals**: Scale knowledge, reduce toil.

**Key Principles**:

1. **Start simple, add complexity gradually**: Don't over-engineer from day one
2. **Test configs before deploying**: Always run `nginx -t`
3. **Use includes for organization**: Keep configs maintainable
4. **Log everything**: You'll need it for debugging
5. **Monitor upstream health**: Use health checks and failover
6. **Mirror carefully**: Prevent side effects in shadow APIs
7. **Document decisions**: Future you will thank present you

This maturity path takes you from a basic reverse proxy to a production-grade API gateway that can handle multiple services, traffic mirroring, and enterprise-scale traffic.

## See Also

- **[Nginx Production](nginx-production.md)** - Production-grade web server configuration
- **[Docker & Compose](docker-and-compose.md)** - Containerization patterns
- **[Ansible Security Hardening](ansible-security-hardening.md)** - Infrastructure security practices

---

*This guide provides the complete machinery for production-grade NGINX configuration. The patterns scale from simple reverse proxies to complex multi-API gateways with traffic mirroring, from single instances to load-balanced clusters.*

