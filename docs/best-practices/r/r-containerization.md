# R Containerization Best Practices

**Objective**: Master senior-level R containerization patterns for production systems. When you need to package R applications for deployment, when you want to ensure reproducibility across environments, when you need enterprise-grade containerization patterns—these best practices become your weapon of choice.

## Core Principles

- **Reproducibility**: Ensure consistent environments across development and production
- **Security**: Implement proper security practices in containers
- **Performance**: Optimize container size and runtime performance
- **Scalability**: Design containers for horizontal scaling
- **Maintainability**: Keep containers maintainable and updatable

## Docker Fundamentals

### Basic Docker Setup

```r
# R/01-docker-fundamentals.R

#' Create comprehensive Docker setup
#'
#' @param app_config Application configuration
#' @return Docker setup
create_docker_setup <- function(app_config) {
  # Create Dockerfile
  dockerfile <- create_dockerfile(app_config)
  
  # Create docker-compose.yml
  docker_compose <- create_docker_compose(app_config)
  
  # Create .dockerignore
  dockerignore <- create_dockerignore(app_config)
  
  # Create build script
  build_script <- create_build_script(app_config)
  
  setup <- list(
    dockerfile = dockerfile,
    docker_compose = docker_compose,
    dockerignore = dockerignore,
    build_script = build_script
  )
  
  return(setup)
}

#' Create Dockerfile
#'
#' @param app_config Application configuration
#' @return Dockerfile content
create_dockerfile <- function(app_config) {
  dockerfile_content <- c(
    "# Use Rocker R base image",
    paste("FROM", app_config$base_image %||% "rocker/r-ver:4.3.0"),
    "",
    "# Set working directory",
    "WORKDIR /app",
    "",
    "# Install system dependencies",
    "RUN apt-get update && apt-get install -y \\",
    "    libcurl4-openssl-dev \\",
    "    libssl-dev \\",
    "    libxml2-dev \\",
    "    libmariadb-dev \\",
    "    libpq-dev \\",
    "    libsodium-dev \\",
    "    && rm -rf /var/lib/apt/lists/*",
    "",
    "# Copy package files",
    "COPY requirements.txt .",
    "",
    "# Install R packages",
    "RUN Rscript -e 'install.packages(c(\"renv\", \"remotes\"), repos = \"https://cloud.r-project.org\")'",
    "RUN Rscript -e 'renv::restore()'",
    "",
    "# Copy application code",
    "COPY . .",
    "",
    "# Expose port",
    paste("EXPOSE", app_config$port %||% 8000),
    "",
    "# Set environment variables",
    "ENV R_ENVIRON_USER=/app/.Renviron",
    "",
    "# Run application",
    paste("CMD [\"Rscript\", \"", app_config$entry_point %||% "app.R", "\"]")
  )
  
  return(dockerfile_content)
}

#' Create docker-compose.yml
#'
#' @param app_config Application configuration
#' @return Docker Compose content
create_docker_compose <- function(app_config) {
  docker_compose_content <- c(
    "version: '3.8'",
    "",
    "services:",
    "  r-app:",
    "    build: .",
    "    ports:",
    paste("      - \"", app_config$port %||% 8000, ":", app_config$port %||% 8000, "\""),
    "    volumes:",
    "      - ./data:/app/data",
    "      - ./output:/app/output",
    "    environment:",
    "      - R_ENVIRON_USER=/app/.Renviron",
    "    depends_on:",
    "      - database",
    "",
    "  database:",
    "    image: postgres:15",
    "    environment:",
    "      POSTGRES_DB: ${DB_NAME}",
    "      POSTGRES_USER: ${DB_USER}",
    "      POSTGRES_PASSWORD: ${DB_PASSWORD}",
    "    volumes:",
    "      - postgres_data:/var/lib/postgresql/data",
    "",
    "volumes:",
    "  postgres_data:"
  )
  
  return(docker_compose_content)
}

#' Create .dockerignore
#'
#' @param app_config Application configuration
#' @return .dockerignore content
create_dockerignore <- function(app_config) {
  dockerignore_content <- c(
    "# R files",
    "*.Rproj",
    ".Rhistory",
    ".RData",
    ".Ruserdata",
    "",
    "# Data files",
    "*.csv",
    "*.xlsx",
    "*.rds",
    "",
    "# Output files",
    "output/",
    "results/",
    "",
    "# Git",
    ".git/",
    ".gitignore",
    "",
    "# Documentation",
    "README.md",
    "docs/",
    "",
    "# Environment files",
    ".env",
    ".Renviron"
  )
  
  return(dockerignore_content)
}

#' Create build script
#'
#' @param app_config Application configuration
#' @return Build script content
create_build_script <- function(app_config) {
  build_script_content <- c(
    "#!/bin/bash",
    "",
    "# Build Docker image",
    "docker build -t r-app .",
    "",
    "# Run container",
    "docker run -p 8000:8000 r-app",
    "",
    "# Or use docker-compose",
    "docker-compose up --build"
  )
  
  return(build_script_content)
}
```

### Multi-stage Docker Builds

```r
# R/01-docker-fundamentals.R (continued)

#' Create multi-stage Dockerfile
#'
#' @param app_config Application configuration
#' @return Multi-stage Dockerfile
create_multi_stage_dockerfile <- function(app_config) {
  dockerfile_content <- c(
    "# Build stage",
    "FROM rocker/r-ver:4.3.0 AS builder",
    "",
    "# Install build dependencies",
    "RUN apt-get update && apt-get install -y \\",
    "    build-essential \\",
    "    libcurl4-openssl-dev \\",
    "    libssl-dev \\",
    "    libxml2-dev \\",
    "    && rm -rf /var/lib/apt/lists/*",
    "",
    "# Set working directory",
    "WORKDIR /app",
    "",
    "# Copy package files",
    "COPY requirements.txt .",
    "",
    "# Install R packages",
    "RUN Rscript -e 'install.packages(c(\"renv\", \"remotes\"), repos = \"https://cloud.r-project.org\")'",
    "RUN Rscript -e 'renv::restore()'",
    "",
    "# Copy source code",
    "COPY . .",
    "",
    "# Build application",
    "RUN Rscript -e 'source(\"build.R\")'",
    "",
    "# Production stage",
    "FROM rocker/r-ver:4.3.0",
    "",
    "# Install runtime dependencies",
    "RUN apt-get update && apt-get install -y \\",
    "    libcurl4-openssl-dev \\",
    "    libssl-dev \\",
    "    libxml2-dev \\",
    "    && rm -rf /var/lib/apt/lists/*",
    "",
    "# Copy built application from builder stage",
    "COPY --from=builder /app /app",
    "",
    "# Set working directory",
    "WORKDIR /app",
    "",
    "# Create non-root user",
    "RUN useradd -m -u 1000 ruser && chown -R ruser:ruser /app",
    "USER ruser",
    "",
    "# Expose port",
    paste("EXPOSE", app_config$port %||% 8000),
    "",
    "# Run application",
    paste("CMD [\"Rscript\", \"", app_config$entry_point %||% "app.R", "\"]")
  )
  
  return(dockerfile_content)
}
```

## R-Specific Containerization

### Rocker Images

```r
# R/02-r-specific-containerization.R

#' Create R-specific container setup
#'
#' @param app_config Application configuration
#' @return R container setup
create_r_container_setup <- function(app_config) {
  setup <- list(
    base_image = choose_rocker_image(app_config),
    r_packages = install_r_packages(app_config),
    system_dependencies = install_system_dependencies(app_config),
    r_configuration = configure_r_environment(app_config)
  )
  
  return(setup)
}

#' Choose Rocker base image
#'
#' @param app_config Application configuration
#' @return Rocker base image
choose_rocker_image <- function(app_config) {
  base_images <- list(
    "base" = "rocker/r-ver:4.3.0",
    "tidyverse" = "rocker/tidyverse:4.3.0",
    "geospatial" = "rocker/geospatial:4.3.0",
    "ml" = "rocker/ml:4.3.0",
    "shiny" = "rocker/shiny:4.3.0",
    "rstudio" = "rocker/rstudio:4.3.0"
  )
  
  return(base_images[[app_config$rocker_image %||% "base"]])
}

#' Install R packages
#'
#' @param app_config Application configuration
#' @return R package installation commands
install_r_packages <- function(app_config) {
  packages <- app_config$r_packages %||% c("renv", "remotes")
  
  install_commands <- c(
    "# Install R packages",
    "RUN Rscript -e 'install.packages(c(\"renv\", \"remotes\"), repos = \"https://cloud.r-project.org\")'",
    "",
    "# Install additional packages",
    paste("RUN Rscript -e 'install.packages(c(", 
          paste0("\"", packages, "\"", collapse = ", "), 
          "), repos = \"https://cloud.r-project.org\")'")
  )
  
  return(install_commands)
}

#' Install system dependencies
#'
#' @param app_config Application configuration
#' @return System dependency installation commands
install_system_dependencies <- function(app_config) {
  dependencies <- app_config$system_dependencies %||% c(
    "libcurl4-openssl-dev",
    "libssl-dev",
    "libxml2-dev"
  )
  
  install_commands <- c(
    "# Install system dependencies",
    paste("RUN apt-get update && apt-get install -y \\"),
    paste("    ", paste(dependencies, collapse = " \\\n    ")),
    "    && rm -rf /var/lib/apt/lists/*"
  )
  
  return(install_commands)
}

#' Configure R environment
#'
#' @param app_config Application configuration
#' @return R configuration commands
configure_r_environment <- function(app_config) {
  config_commands <- c(
    "# Configure R environment",
    "ENV R_ENVIRON_USER=/app/.Renviron",
    "ENV R_LIBS_USER=/app/R/library",
    "",
    "# Set R options",
    "ENV R_OPTIONS='--no-restore --no-save'"
  )
  
  return(config_commands)
}
```

### Package Management

```r
# R/02-r-specific-containerization.R (continued)

#' Create package management setup
#'
#' @param app_config Application configuration
#' @return Package management setup
create_package_management_setup <- function(app_config) {
  setup <- list(
    renv_setup = create_renv_setup(app_config),
    package_installation = create_package_installation(app_config),
    dependency_management = create_dependency_management(app_config)
  )
  
  return(setup)
}

#' Create renv setup
#'
#' @param app_config Application configuration
#' @return renv setup
create_renv_setup <- function(app_config) {
  renv_setup <- c(
    "# Initialize renv",
    "RUN Rscript -e 'renv::init()'",
    "",
    "# Restore packages from lock file",
    "COPY renv.lock .",
    "RUN Rscript -e 'renv::restore()'",
    "",
    "# Activate renv",
    "RUN Rscript -e 'renv::activate()'"
  )
  
  return(renv_setup)
}

#' Create package installation
#'
#' @param app_config Application configuration
#' @return Package installation commands
create_package_installation <- function(app_config) {
  packages <- app_config$packages %||% list()
  
  install_commands <- c()
  
  # CRAN packages
  if (!is.null(packages$cran)) {
    install_commands <- c(install_commands,
      "# Install CRAN packages",
      paste("RUN Rscript -e 'install.packages(c(", 
            paste0("\"", packages$cran, "\"", collapse = ", "), 
            "), repos = \"https://cloud.r-project.org\")'")
    )
  }
  
  # GitHub packages
  if (!is.null(packages$github)) {
    install_commands <- c(install_commands,
      "# Install GitHub packages",
      paste("RUN Rscript -e 'remotes::install_github(c(", 
            paste0("\"", packages$github, "\"", collapse = ", "), 
            "))'")
    )
  }
  
  # Bioconductor packages
  if (!is.null(packages$bioc)) {
    install_commands <- c(install_commands,
      "# Install Bioconductor packages",
      "RUN Rscript -e 'BiocManager::install(c(", 
      paste0("\"", packages$bioc, "\"", collapse = ", "), 
      "))'"
    )
  }
  
  return(install_commands)
}

#' Create dependency management
#'
#' @param app_config Application configuration
#' @return Dependency management setup
create_dependency_management <- function(app_config) {
  dependency_management <- c(
    "# Create dependency management script",
    "COPY install_dependencies.R .",
    "RUN Rscript install_dependencies.R",
    "",
    "# Verify package installation",
    "RUN Rscript -e 'installed.packages()'"
  )
  
  return(dependency_management)
}
```

## Container Orchestration

### Docker Compose

```r
# R/03-container-orchestration.R

#' Create comprehensive Docker Compose setup
#'
#' @param app_config Application configuration
#' @return Docker Compose setup
create_docker_compose_setup <- function(app_config) {
  setup <- list(
    services = create_services(app_config),
    networks = create_networks(app_config),
    volumes = create_volumes(app_config),
    environment = create_environment(app_config)
  )
  
  return(setup)
}

#' Create services
#'
#' @param app_config Application configuration
#' @return Services configuration
create_services <- function(app_config) {
  services <- list(
    r_app = create_r_app_service(app_config),
    database = create_database_service(app_config),
    redis = create_redis_service(app_config),
    nginx = create_nginx_service(app_config)
  )
  
  return(services)
}

#' Create R app service
#'
#' @param app_config Application configuration
#' @return R app service configuration
create_r_app_service <- function(app_config) {
  r_app_service <- c(
    "  r-app:",
    "    build: .",
    "    ports:",
    paste("      - \"", app_config$port %||% 8000, ":", app_config$port %||% 8000, "\""),
    "    volumes:",
    "      - ./data:/app/data",
    "      - ./output:/app/output",
    "    environment:",
    "      - R_ENVIRON_USER=/app/.Renviron",
    "      - DB_HOST=database",
    "      - DB_PORT=5432",
    "      - REDIS_HOST=redis",
    "      - REDIS_PORT=6379",
    "    depends_on:",
    "      - database",
    "      - redis",
    "    networks:",
    "      - app-network"
  )
  
  return(r_app_service)
}

#' Create database service
#'
#' @param app_config Application configuration
#' @return Database service configuration
create_database_service <- function(app_config) {
  database_service <- c(
    "  database:",
    "    image: postgres:15",
    "    environment:",
    "      POSTGRES_DB: ${DB_NAME}",
    "      POSTGRES_USER: ${DB_USER}",
    "      POSTGRES_PASSWORD: ${DB_PASSWORD}",
    "    volumes:",
    "      - postgres_data:/var/lib/postgresql/data",
    "    networks:",
    "      - app-network"
  )
  
  return(database_service)
}

#' Create Redis service
#'
#' @param app_config Application configuration
#' @return Redis service configuration
create_redis_service <- function(app_config) {
  redis_service <- c(
    "  redis:",
    "    image: redis:7-alpine",
    "    volumes:",
    "      - redis_data:/data",
    "    networks:",
    "      - app-network"
  )
  
  return(redis_service)
}

#' Create Nginx service
#'
#' @param app_config Application configuration
#' @return Nginx service configuration
create_nginx_service <- function(app_config) {
  nginx_service <- c(
    "  nginx:",
    "    image: nginx:alpine",
    "    ports:",
    "      - \"80:80\"",
    "      - \"443:443\"",
    "    volumes:",
    "      - ./nginx.conf:/etc/nginx/nginx.conf",
    "      - ./ssl:/etc/nginx/ssl",
    "    depends_on:",
    "      - r-app",
    "    networks:",
    "      - app-network"
  )
  
  return(nginx_service)
}
```

### Kubernetes Deployment

```r
# R/03-container-orchestration.R (continued)

#' Create Kubernetes deployment
#'
#' @param app_config Application configuration
#' @return Kubernetes deployment
create_kubernetes_deployment <- function(app_config) {
  deployment <- list(
    deployment = create_k8s_deployment(app_config),
    service = create_k8s_service(app_config),
    ingress = create_k8s_ingress(app_config),
    configmap = create_k8s_configmap(app_config),
    secret = create_k8s_secret(app_config)
  )
  
  return(deployment)
}

#' Create Kubernetes deployment
#'
#' @param app_config Application configuration
#' @return Kubernetes deployment YAML
create_k8s_deployment <- function(app_config) {
  deployment_yaml <- c(
    "apiVersion: apps/v1",
    "kind: Deployment",
    "metadata:",
    paste("  name:", app_config$app_name %||% "r-app"),
    "spec:",
    "  replicas: 3",
    "  selector:",
    "    matchLabels:",
    paste("      app:", app_config$app_name %||% "r-app"),
    "  template:",
    "    metadata:",
    "      labels:",
    paste("        app:", app_config$app_name %||% "r-app"),
    "    spec:",
    "      containers:",
    "      - name: r-app",
    paste("        image:", app_config$image_name %||% "r-app:latest"),
    "        ports:",
    "        - containerPort: 8000",
    "        env:",
    "        - name: DB_HOST",
    "          valueFrom:",
    "            configMapKeyRef:",
    "              name: app-config",
    "              key: db-host",
    "        - name: DB_PASSWORD",
    "          valueFrom:",
    "            secretKeyRef:",
    "              name: app-secrets",
    "              key: db-password"
  )
  
  return(deployment_yaml)
}

#' Create Kubernetes service
#'
#' @param app_config Application configuration
#' @return Kubernetes service YAML
create_k8s_service <- function(app_config) {
  service_yaml <- c(
    "apiVersion: v1",
    "kind: Service",
    "metadata:",
    paste("  name:", app_config$app_name %||% "r-app", "-service"),
    "spec:",
    "  selector:",
    paste("    app:", app_config$app_name %||% "r-app"),
    "  ports:",
    "  - port: 80",
    "    targetPort: 8000",
    "  type: ClusterIP"
  )
  
  return(service_yaml)
}

#' Create Kubernetes ingress
#'
#' @param app_config Application configuration
#' @return Kubernetes ingress YAML
create_k8s_ingress <- function(app_config) {
  ingress_yaml <- c(
    "apiVersion: networking.k8s.io/v1",
    "kind: Ingress",
    "metadata:",
    paste("  name:", app_config$app_name %||% "r-app", "-ingress"),
    "  annotations:",
    "    nginx.ingress.kubernetes.io/rewrite-target: /",
    "spec:",
    "  rules:",
    "  - host: example.com",
    "    http:",
    "      paths:",
    "      - path: /",
    "        pathType: Prefix",
    "        backend:",
    "          service:",
    paste("            name:", app_config$app_name %||% "r-app", "-service"),
    "            port:",
    "              number: 80"
  )
  
  return(ingress_yaml)
}
```

## Security Best Practices

### Container Security

```r
# R/04-security-best-practices.R

#' Implement container security
#'
#' @param app_config Application configuration
#' @return Security configuration
implement_container_security <- function(app_config) {
  security_config <- list(
    user_security = implement_user_security(app_config),
    network_security = implement_network_security(app_config),
    image_security = implement_image_security(app_config),
    runtime_security = implement_runtime_security(app_config)
  )
  
  return(security_config)
}

#' Implement user security
#'
#' @param app_config Application configuration
#' @return User security configuration
implement_user_security <- function(app_config) {
  user_security <- c(
    "# Create non-root user",
    "RUN useradd -m -u 1000 ruser && chown -R ruser:ruser /app",
    "USER ruser",
    "",
    "# Set proper permissions",
    "RUN chmod 755 /app",
    "RUN chmod 644 /app/*.R"
  )
  
  return(user_security)
}

#' Implement network security
#'
#' @param app_config Application configuration
#' @return Network security configuration
implement_network_security <- function(app_config) {
  network_security <- c(
    "# Use specific network",
    "networks:",
    "  - app-network",
    "",
    "# Limit exposed ports",
    "expose:",
    "  - \"8000\"",
    "",
    "# Use internal networks for database communication"
  )
  
  return(network_security)
}

#' Implement image security
#'
#' @param app_config Application configuration
#' @return Image security configuration
implement_image_security <- function(app_config) {
  image_security <- c(
    "# Use specific image tags",
    "FROM rocker/r-ver:4.3.0",
    "",
    "# Scan for vulnerabilities",
    "RUN apt-get update && apt-get upgrade -y",
    "",
    "# Remove unnecessary packages",
    "RUN apt-get autoremove -y && apt-get clean"
  )
  
  return(image_security)
}

#' Implement runtime security
#'
#' @param app_config Application configuration
#' @return Runtime security configuration
implement_runtime_security <- function(app_config) {
  runtime_security <- c(
    "# Set security options",
    "security_opt:",
    "  - no-new-privileges:true",
    "",
    "# Limit resources",
    "deploy:",
    "  resources:",
    "    limits:",
    "      memory: 1G",
    "      cpus: '0.5'",
    "    reservations:",
    "      memory: 512M",
    "      cpus: '0.25'"
  )
  
  return(runtime_security)
}
```

### Secrets Management

```r
# R/04-security-best-practices.R (continued)

#' Implement secrets management
#'
#' @param app_config Application configuration
#' @return Secrets management configuration
implement_secrets_management <- function(app_config) {
  secrets_config <- list(
    docker_secrets = implement_docker_secrets(app_config),
    kubernetes_secrets = implement_k8s_secrets(app_config),
    environment_secrets = implement_environment_secrets(app_config)
  )
  
  return(secrets_config)
}

#' Implement Docker secrets
#'
#' @param app_config Application configuration
#' @return Docker secrets configuration
implement_docker_secrets <- function(app_config) {
  docker_secrets <- c(
    "# Use Docker secrets",
    "secrets:",
    "  - db_password",
    "  - api_key",
    "",
    "# Mount secrets",
    "volumes:",
    "  - db_password:/run/secrets/db_password",
    "  - api_key:/run/secrets/api_key"
  )
  
  return(docker_secrets)
}

#' Implement Kubernetes secrets
#'
#' @param app_config Application configuration
#' @return Kubernetes secrets configuration
implement_k8s_secrets <- function(app_config) {
  k8s_secrets <- c(
    "apiVersion: v1",
    "kind: Secret",
    "metadata:",
    "  name: app-secrets",
    "type: Opaque",
    "data:",
    "  db-password: <base64-encoded-password>",
    "  api-key: <base64-encoded-api-key>"
  )
  
  return(k8s_secrets)
}

#' Implement environment secrets
#'
#' @param app_config Application configuration
#' @return Environment secrets configuration
implement_environment_secrets <- function(app_config) {
  env_secrets <- c(
    "# Use environment variables for secrets",
    "environment:",
    "  - DB_PASSWORD_FILE=/run/secrets/db_password",
    "  - API_KEY_FILE=/run/secrets/api_key",
    "",
    "# Read secrets from files",
    "RUN chmod 600 /run/secrets/*"
  )
  
  return(env_secrets)
}
```

## Performance Optimization

### Container Optimization

```r
# R/05-performance-optimization.R

#' Optimize container performance
#'
#' @param app_config Application configuration
#' @return Performance optimization configuration
optimize_container_performance <- function(app_config) {
  optimization_config <- list(
    image_optimization = optimize_image_size(app_config),
    runtime_optimization = optimize_runtime_performance(app_config),
    resource_optimization = optimize_resource_usage(app_config)
  )
  
  return(optimization_config)
}

#' Optimize image size
#'
#' @param app_config Application configuration
#' @return Image optimization configuration
optimize_image_size <- function(app_config) {
  image_optimization <- c(
    "# Use multi-stage build",
    "FROM rocker/r-ver:4.3.0 AS builder",
    "",
    "# Install only necessary packages",
    "RUN apt-get update && apt-get install -y \\",
    "    --no-install-recommends \\",
    "    libcurl4-openssl-dev \\",
    "    libssl-dev \\",
    "    && rm -rf /var/lib/apt/lists/*",
    "",
    "# Clean up package cache",
    "RUN apt-get clean && rm -rf /var/lib/apt/lists/*",
    "",
    "# Use Alpine-based image for production",
    "FROM alpine:latest",
    "RUN apk add --no-cache r-base"
  )
  
  return(image_optimization)
}

#' Optimize runtime performance
#'
#' @param app_config Application configuration
#' @return Runtime optimization configuration
optimize_runtime_performance <- function(app_config) {
  runtime_optimization <- c(
    "# Set R options for performance",
    "ENV R_OPTIONS='--no-restore --no-save --max-mem-size=1G'",
    "",
    "# Use parallel processing",
    "ENV R_PARALLEL=TRUE",
    "ENV R_PARALLEL_CORES=4",
    "",
    "# Optimize garbage collection",
    "ENV R_GC_OPTIONS='--max-mem-size=1G'"
  )
  
  return(runtime_optimization)
}

#' Optimize resource usage
#'
#' @param app_config Application configuration
#' @return Resource optimization configuration
optimize_resource_usage <- function(app_config) {
  resource_optimization <- c(
    "# Limit memory usage",
    "ENV R_MAX_MEM_SIZE=1G",
    "",
    "# Set CPU limits",
    "ENV R_CPU_LIMIT=4",
    "",
    "# Optimize for container environment",
    "ENV R_CONTAINER=TRUE"
  )
  
  return(resource_optimization)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Create Docker setup
docker_setup <- create_docker_setup(app_config)

# 2. Create R-specific container
r_container <- create_r_container_setup(app_config)

# 3. Create Docker Compose setup
compose_setup <- create_docker_compose_setup(app_config)

# 4. Implement security
security_config <- implement_container_security(app_config)

# 5. Optimize performance
optimization_config <- optimize_container_performance(app_config)
```

### Essential Patterns

```r
# Complete containerization pipeline
create_containerization_pipeline <- function(app_config) {
  # Create Docker setup
  docker_setup <- create_docker_setup(app_config)
  
  # Create R-specific container
  r_container <- create_r_container_setup(app_config)
  
  # Create orchestration setup
  orchestration <- create_docker_compose_setup(app_config)
  
  # Implement security
  security_config <- implement_container_security(app_config)
  
  # Optimize performance
  optimization_config <- optimize_container_performance(app_config)
  
  return(list(
    docker = docker_setup,
    r_container = r_container,
    orchestration = orchestration,
    security = security_config,
    optimization = optimization_config
  ))
}
```

---

*This guide provides the complete machinery for containerizing R applications. Each pattern includes implementation examples, security strategies, and real-world usage patterns for enterprise deployment.*
