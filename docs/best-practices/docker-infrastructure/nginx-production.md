# Nginx Production Best Practices

**Objective**: Build bulletproof web infrastructure with Nginx. Handle traffic spikes, secure your applications, and optimize for performance while maintaining operational sanity.

When your web application needs to handle millions of requests, serve static content at lightning speed, and protect against every attack vector imaginable, Nginx becomes your first line of defense. This guide shows you how to configure Nginx for production workloads that demand reliability, security, and performance.

## 0) Principles (Read Once, Live by Them)

### The Five Commandments

1. **Security first, always**
   - HTTPS everywhere, HSTS headers, security headers
   - Rate limiting, request validation, input sanitization
   - Regular security updates, minimal attack surface

2. **Performance is not optional**
   - Gzip compression, HTTP/2, keepalive connections
   - Caching strategies, static file optimization
   - Connection pooling, worker process tuning

3. **Monitoring and observability**
   - Structured logging, metrics collection
   - Health checks, error tracking, performance monitoring
   - Alerting on anomalies and failures

4. **Configuration management**
   - Version control, environment-specific configs
   - Automated deployment, configuration validation
   - Documentation and change tracking

5. **Graceful degradation**
   - Custom error pages, fallback mechanisms
   - Circuit breakers, timeout handling
   - Graceful shutdowns and restarts

**Why These Principles**: Nginx sits between your users and your applications. When it fails, everything fails. These principles ensure your web infrastructure is bulletproof.

## 1) Core Configuration: The Foundation

### Main Configuration Structure

```nginx
# /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging format
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    access_log /var/log/nginx/access.log main;
    
    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    
    # Include site configurations
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

**Why This Structure**: Proper worker configuration, security headers, compression, and rate limiting form the foundation of production Nginx. Every setting has a purpose.

### Security Headers Configuration

```nginx
# /etc/nginx/conf.d/security.conf
# Security headers for all sites
map $sent_http_content_type $csp_header {
    ~*text/html "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none';";
    default "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none';";
}

# HSTS (HTTP Strict Transport Security)
map $https $hsts_header {
    on "max-age=31536000; includeSubDomains; preload";
    off "";
}

# Security headers
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy $csp_header always;
add_header Strict-Transport-Security $hsts_header always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

# Hide Nginx version
server_tokens off;
```

**Why Security Headers**: These headers protect against XSS, clickjacking, MIME sniffing, and other common web vulnerabilities. They're your first line of defense.

## 2) Custom Error Pages: The User Experience

### Error Page Configuration

```nginx
# /etc/nginx/conf.d/error-pages.conf
# Custom error pages with proper HTTP status codes
error_page 400 /errors/400.html;
error_page 401 /errors/401.html;
error_page 403 /errors/403.html;
error_page 404 /errors/404.html;
error_page 404 /errors/404.html;
error_page 405 /errors/405.html;
error_page 408 /errors/408.html;
error_page 410 /errors/410.html;
error_page 413 /errors/413.html;
error_page 414 /errors/414.html;
error_page 415 /errors/415.html;
error_page 429 /errors/429.html;
error_page 500 /errors/500.html;
error_page 502 /errors/502.html;
error_page 503 /errors/503.html;
error_page 504 /errors/504.html;

# Error page locations
location /errors/ {
    internal;
    root /var/www/html;
    expires 1h;
    add_header Cache-Control "public, immutable";
}

# Specific error page for 404s
location = /errors/404.html {
    internal;
    root /var/www/html;
    expires 1h;
    add_header Cache-Control "public, immutable";
}

# Specific error page for 500s
location = /errors/500.html {
    internal;
    root /var/www/html;
    expires 1h;
    add_header Cache-Control "public, immutable";
}
```

### Custom Error Page HTML

```html
<!-- /var/www/html/errors/404.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Page Not Found</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background: #f8f9fa; }
        .container { max-width: 600px; margin: 100px auto; text-align: center; padding: 20px; }
        .error-code { font-size: 120px; font-weight: 300; color: #6c757d; margin: 0; }
        .error-message { font-size: 24px; color: #495057; margin: 20px 0; }
        .error-description { font-size: 16px; color: #6c757d; margin: 20px 0; }
        .btn { display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="error-code">404</h1>
        <h2 class="error-message">Page Not Found</h2>
        <p class="error-description">The page you're looking for doesn't exist or has been moved.</p>
        <a href="/" class="btn">Go Home</a>
    </div>
</body>
</html>
```

### Advanced Error Handling

```nginx
# /etc/nginx/conf.d/advanced-errors.conf
# Advanced error handling with logging
map $status $loggable {
    ~^[23] 0;
    ~^[45] 1;
    default 1;
}

# Custom error pages with dynamic content
location /errors/ {
    internal;
    root /var/www/html;
    
    # Add error context to headers
    add_header X-Error-Code $status always;
    add_header X-Error-Time $time_iso8601 always;
    
    # Log errors to separate file
    access_log /var/log/nginx/error-requests.log main if=$loggable;
}

# Error page for API endpoints
location ~ ^/api/ {
    error_page 404 = @api_404;
    error_page 500 = @api_500;
    
    # API error responses
    location @api_404 {
        return 404 '{"error": "Not Found", "code": 404, "message": "The requested resource was not found"}';
        add_header Content-Type application/json always;
    }
    
    location @api_500 {
        return 500 '{"error": "Internal Server Error", "code": 500, "message": "An internal error occurred"}';
        add_header Content-Type application/json always;
    }
}
```

**Why Custom Error Pages**: Professional error pages improve user experience and provide context. They also help with debugging and monitoring.

## 3) HTTP Status Codes: The Protocol Mastery

### Proper Status Code Usage

```nginx
# /etc/nginx/conf.d/status-codes.conf
# Proper HTTP status code handling

# 200 OK - Successful requests
location /health {
    return 200 '{"status": "healthy", "timestamp": "$time_iso8601"}';
    add_header Content-Type application/json always;
}

# 201 Created - Resource creation
location /api/users {
    if ($request_method = POST) {
        return 201 '{"id": "123", "message": "User created successfully"}';
        add_header Content-Type application/json always;
    }
}

# 204 No Content - Successful deletion
location /api/users/ {
    if ($request_method = DELETE) {
        return 204;
    }
}

# 301 Moved Permanently - SEO-friendly redirects
location /old-page {
    return 301 /new-page;
}

# 302 Found - Temporary redirects
location /temporary {
    return 302 /temp-page;
}

# 304 Not Modified - Caching
location /static/ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header ETag $upstream_http_etag;
    
    # Handle If-Modified-Since
    if ($http_if_modified_since) {
        return 304;
    }
}

# 400 Bad Request - Client errors
location /api/validate {
    if ($request_method != POST) {
        return 400 '{"error": "Method not allowed", "code": 400}';
        add_header Content-Type application/json always;
    }
}

# 401 Unauthorized - Authentication required
location /api/protected {
    auth_basic "Restricted Area";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    if ($remote_user = "") {
        return 401 '{"error": "Authentication required", "code": 401}';
        add_header Content-Type application/json always;
        add_header WWW-Authenticate 'Basic realm="Restricted Area"' always;
    }
}

# 403 Forbidden - Access denied
location /admin {
    allow 192.168.1.0/24;
    deny all;
    
    return 403 '{"error": "Access denied", "code": 403}';
    add_header Content-Type application/json always;
}

# 404 Not Found - Resource not found
location /api/not-found {
    return 404 '{"error": "Resource not found", "code": 404}';
    add_header Content-Type application/json always;
}

# 405 Method Not Allowed - HTTP method not supported
location /api/method-test {
    if ($request_method !~ ^(GET|POST)$) {
        return 405 '{"error": "Method not allowed", "code": 405}';
        add_header Content-Type application/json always;
        add_header Allow "GET, POST" always;
    }
}

# 408 Request Timeout - Client timeout
location /api/slow {
    proxy_read_timeout 5s;
    proxy_connect_timeout 5s;
    proxy_send_timeout 5s;
    
    # Custom timeout handling
    error_page 408 = @timeout;
    location @timeout {
        return 408 '{"error": "Request timeout", "code": 408}';
        add_header Content-Type application/json always;
    }
}

# 409 Conflict - Resource conflict
location /api/conflict {
    if ($http_content_type != "application/json") {
        return 409 '{"error": "Content type conflict", "code": 409}';
        add_header Content-Type application/json always;
    }
}

# 410 Gone - Resource permanently removed
location /api/removed {
    return 410 '{"error": "Resource permanently removed", "code": 410}';
    add_header Content-Type application/json always;
}

# 413 Payload Too Large - Request too large
location /api/upload {
    client_max_body_size 10M;
    
    if ($content_length > 10485760) {
        return 413 '{"error": "Payload too large", "code": 413}';
        add_header Content-Type application/json always;
    }
}

# 414 URI Too Long - URL too long
location /api/long-url {
    if ($request_uri ~ "^/api/long-url/(.{1000,})") {
        return 414 '{"error": "URI too long", "code": 414}';
        add_header Content-Type application/json always;
    }
}

# 415 Unsupported Media Type - Content type not supported
location /api/upload {
    if ($http_content_type !~ "^(multipart/form-data|application/json)$") {
        return 415 '{"error": "Unsupported media type", "code": 415}';
        add_header Content-Type application/json always;
    }
}

# 429 Too Many Requests - Rate limiting
location /api/rate-limited {
    limit_req zone=api burst=10 nodelay;
    
    # Custom rate limit response
    error_page 429 = @rate_limit;
    location @rate_limit {
        return 429 '{"error": "Too many requests", "code": 429, "retry_after": 60}';
        add_header Content-Type application/json always;
        add_header Retry-After 60 always;
    }
}

# 500 Internal Server Error - Server errors
location /api/error {
    # Simulate server error
    return 500 '{"error": "Internal server error", "code": 500}';
    add_header Content-Type application/json always;
}

# 502 Bad Gateway - Upstream server error
location /api/proxy {
    proxy_pass http://backend;
    proxy_intercept_errors on;
    
    error_page 502 = @bad_gateway;
    location @bad_gateway {
        return 502 '{"error": "Bad gateway", "code": 502}';
        add_header Content-Type application/json always;
    }
}

# 503 Service Unavailable - Server overloaded
location /api/overloaded {
    # Simulate service unavailable
    return 503 '{"error": "Service unavailable", "code": 503}';
    add_header Content-Type application/json always;
    add_header Retry-After 60 always;
}

# 504 Gateway Timeout - Upstream timeout
location /api/timeout {
    proxy_pass http://backend;
    proxy_read_timeout 5s;
    
    error_page 504 = @gateway_timeout;
    location @gateway_timeout {
        return 504 '{"error": "Gateway timeout", "code": 504}';
        add_header Content-Type application/json always;
    }
}
```

**Why Proper Status Codes**: HTTP status codes communicate intent and enable proper error handling. They're essential for API design and client integration.

## 4) Advanced Nginx Features: The Power User

### Load Balancing and Upstream

```nginx
# /etc/nginx/conf.d/upstream.conf
# Upstream server configuration
upstream backend {
    # Load balancing methods
    least_conn;
    
    # Health checks
    server 192.168.1.10:8080 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:8080 max_fails=3 fail_timeout=30s;
    server 192.168.1.12:8080 max_fails=3 fail_timeout=30s backup;
    
    # Keepalive connections
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# Advanced load balancing
upstream api {
    # Consistent hashing
    hash $remote_addr consistent;
    
    server 192.168.1.20:8080 weight=3;
    server 192.168.1.21:8080 weight=2;
    server 192.168.1.22:8080 weight=1;
    
    # Health checks
    health_check interval=10s fails=3 passes=2;
}

# WebSocket support
upstream websocket {
    server 192.168.1.30:8080;
    server 192.168.1.31:8080;
    
    # WebSocket keepalive
    keepalive 16;
}
```

### Caching Strategies

```nginx
# /etc/nginx/conf.d/caching.conf
# Advanced caching configuration

# Proxy cache
proxy_cache_path /var/cache/nginx/proxy levels=1:2 keys_zone=proxy_cache:10m max_size=1g inactive=60m use_temp_path=off;
proxy_cache_key "$scheme$request_method$host$request_uri";
proxy_cache_valid 200 302 10m;
proxy_cache_valid 404 1m;
proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
proxy_cache_background_update on;
proxy_cache_lock on;

# FastCGI cache
fastcgi_cache_path /var/cache/nginx/fastcgi levels=1:2 keys_zone=fastcgi_cache:10m max_size=1g inactive=60m use_temp_path=off;
fastcgi_cache_key "$scheme$request_method$host$request_uri";
fastcgi_cache_valid 200 302 10m;
fastcgi_cache_valid 404 1m;
fastcgi_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;

# Cache bypass
map $http_pragma $cache_bypass {
    "no-cache" 1;
    default 0;
}

map $http_cache_control $cache_bypass {
    "no-cache" 1;
    default 0;
}

# Cache headers
add_header X-Cache-Status $upstream_cache_status always;
add_header X-Cache-Key $proxy_cache_key always;
```

### SSL/TLS Configuration

```nginx
# /etc/nginx/conf.d/ssl.conf
# Modern SSL/TLS configuration

# SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;

# HSTS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Certificate transparency
add_header Expect-CT "max-age=86400, enforce" always;
```

### Rate Limiting and DDoS Protection

```nginx
# /etc/nginx/conf.d/rate-limiting.conf
# Advanced rate limiting and DDoS protection

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=burst:10m rate=1r/s;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
limit_conn_zone $server_name zone=conn_limit_per_server:10m;

# DDoS protection
limit_req_zone $binary_remote_addr zone=ddos:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=ddos_burst:10m rate=5r/s;

# Advanced rate limiting
location /api/ {
    # Multiple rate limits
    limit_req zone=api burst=20 nodelay;
    limit_req zone=general burst=50 nodelay;
    limit_conn conn_limit_per_ip 10;
    
    # Custom rate limit headers
    add_header X-RateLimit-Limit "100" always;
    add_header X-RateLimit-Remaining "$limit_req_status" always;
    add_header X-RateLimit-Reset "$time_iso8601" always;
}

# DDoS protection
location / {
    limit_req zone=ddos burst=5 nodelay;
    limit_req zone=ddos_burst burst=10 nodelay;
    
    # Block suspicious requests
    if ($http_user_agent ~* (bot|crawler|spider)) {
        return 403;
    }
    
    # Block requests with no user agent
    if ($http_user_agent = "") {
        return 403;
    }
}
```

**Why Advanced Features**: Production environments need sophisticated load balancing, caching, and security features. These configurations provide enterprise-grade capabilities.

## 5) Docker Compose Example: The Containerized Setup

### Complete Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:1.25-alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/sites-enabled:/etc/nginx/sites-enabled:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/html:/var/www/html:ro
      - ./nginx/logs:/var/log/nginx
      - ./nginx/cache:/var/cache/nginx
    environment:
      - NGINX_ENVSUBST_TEMPLATE_DIR=/etc/nginx/templates
      - NGINX_ENVSUBST_OUTPUT_DIR=/etc/nginx/conf.d
    depends_on:
      - app1
      - app2
      - app3
    networks:
      - frontend
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  app1:
    image: nginx:1.25-alpine
    container_name: app1
    ports:
      - "8081:80"
    volumes:
      - ./apps/app1:/usr/share/nginx/html:ro
    networks:
      - backend
    restart: unless-stopped

  app2:
    image: nginx:1.25-alpine
    container_name: app2
    ports:
      - "8082:80"
    volumes:
      - ./apps/app2:/usr/share/nginx/html:ro
    networks:
      - backend
    restart: unless-stopped

  app3:
    image: nginx:1.25-alpine
    container_name: app3
    ports:
      - "8083:80"
    volumes:
      - ./apps/app3:/usr/share/nginx/html:ro
    networks:
      - backend
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - backend
    restart: unless-stopped
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring
    restart: unless-stopped

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
  monitoring:
    driver: bridge

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Nginx Configuration for Docker

```nginx
# nginx/nginx.conf
user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    
    # Include configurations
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

### Site Configuration for Docker

```nginx
# nginx/sites-enabled/default.conf
# Main site configuration

# Upstream servers
upstream backend {
    least_conn;
    server app1:80 max_fails=3 fail_timeout=30s;
    server app2:80 max_fails=3 fail_timeout=30s;
    server app3:80 max_fails=3 fail_timeout=30s backup;
    
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# Main server block
server {
    listen 80;
    server_name localhost;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Rate limiting
    limit_req zone=general burst=50 nodelay;
    limit_conn conn_limit_per_ip 10;
    
    # Health check
    location /health {
        access_log off;
        return 200 '{"status": "healthy", "timestamp": "$time_iso8601"}';
        add_header Content-Type application/json always;
    }
    
    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
        
        # Error handling
        proxy_intercept_errors on;
        error_page 502 503 504 = @api_error;
    }
    
    # Static files
    location /static/ {
        alias /var/www/html/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # Compression
        gzip_static on;
    }
    
    # Error pages
    location /errors/ {
        internal;
        root /var/www/html;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }
    
    # Custom error pages
    error_page 400 /errors/400.html;
    error_page 401 /errors/401.html;
    error_page 403 /errors/403.html;
    error_page 404 /errors/404.html;
    error_page 500 /errors/500.html;
    error_page 502 /errors/502.html;
    error_page 503 /errors/503.html;
    error_page 504 /errors/504.html;
    
    # API error handling
    location @api_error {
        return 502 '{"error": "Service temporarily unavailable", "code": 502}';
        add_header Content-Type application/json always;
    }
    
    # Default location
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### CNAME Configuration

```nginx
# nginx/sites-enabled/cnames.conf
# CNAME and subdomain configuration

# Main domain
server {
    listen 80;
    server_name example.com www.example.com;
    
    # Redirect www to non-www
    if ($host = www.example.com) {
        return 301 http://example.com$request_uri;
    }
    
    # Main site
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# API subdomain
server {
    listen 80;
    server_name api.example.com;
    
    # API-specific configuration
    location / {
        limit_req zone=api burst=100 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # API-specific headers
        add_header X-API-Version "1.0" always;
        add_header X-API-Status "active" always;
    }
}

# Admin subdomain
server {
    listen 80;
    server_name admin.example.com;
    
    # Admin access control
    allow 192.168.1.0/24;
    deny all;
    
    # Admin-specific configuration
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Admin headers
        add_header X-Admin-Access "restricted" always;
    }
}

# CDN subdomain
server {
    listen 80;
    server_name cdn.example.com;
    
    # CDN configuration
    location / {
        root /var/www/html/cdn;
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # Compression
        gzip_static on;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept" always;
    }
}
```

**Why Docker Compose**: Containerized Nginx provides consistency, scalability, and easy deployment. The configuration demonstrates production-ready patterns.

## 6) Monitoring and Observability: The Operations

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/nginx_status'
    scrape_interval: 5s

  - job_name: 'apps'
    static_configs:
      - targets: ['app1:80', 'app2:80', 'app3:80']
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 10s
```

### Nginx Status Module

```nginx
# nginx/conf.d/status.conf
# Nginx status and metrics
server {
    listen 80;
    server_name status.example.com;
    
    # Status endpoint
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 192.168.1.0/24;
        deny all;
    }
    
    # Metrics endpoint
    location /metrics {
        stub_status on;
        access_log off;
        allow 192.168.1.0/24;
        deny all;
    }
}
```

### Log Analysis

```bash
# nginx/logs/analyze.sh
#!/bin/bash

# Log analysis script
LOG_FILE="/var/log/nginx/access.log"

echo "Nginx Log Analysis"
echo "=================="

# Top IPs
echo "Top 10 IPs:"
awk '{print $1}' $LOG_FILE | sort | uniq -c | sort -nr | head -10

# Top pages
echo "Top 10 Pages:"
awk '{print $7}' $LOG_FILE | sort | uniq -c | sort -nr | head -10

# Status codes
echo "Status Code Distribution:"
awk '{print $9}' $LOG_FILE | sort | uniq -c | sort -nr

# Response times
echo "Average Response Time:"
awk '{sum+=$NF} END {print sum/NR}' $LOG_FILE

# Error rate
echo "Error Rate:"
ERRORS=$(awk '$9 >= 400 {count++} END {print count+0}' $LOG_FILE)
TOTAL=$(wc -l < $LOG_FILE)
echo "scale=2; $ERRORS/$TOTAL*100" | bc -l
```

**Why Monitoring**: Production systems need visibility into performance, errors, and usage patterns. Monitoring enables proactive maintenance and optimization.

## 7) TL;DR Quickstart

### Essential Commands

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs nginx

# Test configuration
docker-compose exec nginx nginx -t

# Reload configuration
docker-compose exec nginx nginx -s reload

# Stop services
docker-compose down
```

### Quick Verification

```bash
# Test HTTP endpoints
curl -I http://localhost/health
curl -I http://localhost/api/
curl -I http://localhost/static/

# Test error pages
curl -I http://localhost/nonexistent

# Check status
curl http://localhost/nginx_status
```

### Performance Testing

```bash
# Load testing
ab -n 1000 -c 10 http://localhost/

# Stress testing
wrk -t12 -c400 -d30s http://localhost/

# Memory usage
docker stats nginx-proxy
```

## 8) The Machine's Summary

Nginx is the foundation of modern web infrastructure. When configured properly, it handles traffic spikes, secures your applications, and optimizes performance while maintaining operational sanity. The key is understanding the advanced features and configuring them correctly.

**The Dark Truth**: Nginx sits between your users and your applications. When it fails, everything fails. These configurations ensure your web infrastructure is bulletproof.

**The Machine's Mantra**: "In performance we trust, in security we build, and in the single server we find the path to efficient web infrastructure."

**Why This Matters**: Production web infrastructure needs reliability, security, and performance. These configurations provide enterprise-grade capabilities that scale from development to production.

---

*This tutorial provides the complete machinery for building production-ready Nginx infrastructure. The patterns scale from development to production, from single machines to enterprise deployments.*
