# R Production Deployment Best Practices

**Objective**: Master senior-level R production deployment patterns for enterprise systems. When you need to deploy R applications to production, when you want to ensure reliability and scalability, when you need enterprise-grade deployment strategies—these best practices become your weapon of choice.

## Core Principles

- **Reliability**: Ensure consistent and reliable deployments
- **Scalability**: Design for horizontal and vertical scaling
- **Security**: Implement security best practices
- **Monitoring**: Monitor application health and performance
- **Automation**: Automate deployment processes

## Shiny Server Deployment

### Shiny Server Configuration

```r
# R/01-shiny-server.R

#' Create Shiny Server configuration
#'
#' @param server_config Server configuration
#' @return Shiny Server configuration
create_shiny_server_config <- function(server_config) {
  config <- list(
    server = create_server_config(server_config),
    applications = create_applications_config(server_config),
    security = create_security_config(server_config)
  )
  
  return(config)
}

#' Create server configuration
#'
#' @param server_config Server configuration
#' @return Server configuration
create_server_config <- function(server_config) {
  server_config <- c(
    "# Shiny Server Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "}"
  )
  
  return(server_config)
}

#' Create applications configuration
#'
#' @param server_config Server configuration
#' @return Applications configuration
create_applications_config <- function(server_config) {
  applications_config <- c(
    "# Applications Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(applications_config)
}

#' Create security configuration
#'
#' @param server_config Server configuration
#' @return Security configuration
create_security_config <- function(server_config) {
  security_config <- c(
    "# Security Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(security_config)
}
```

### Shiny Server Pro Configuration

```r
# R/01-shiny-server.R (continued)

#' Create Shiny Server Pro configuration
#'
#' @param server_config Server configuration
#' @return Shiny Server Pro configuration
create_shiny_server_pro_config <- function(server_config) {
  config <- list(
    server = create_pro_server_config(server_config),
    applications = create_pro_applications_config(server_config),
    security = create_pro_security_config(server_config),
    monitoring = create_pro_monitoring_config(server_config)
  )
  
  return(config)
}

#' Create Pro server configuration
#'
#' @param server_config Server configuration
#' @return Pro server configuration
create_pro_server_config <- function(server_config) {
  pro_server_config <- c(
    "# Shiny Server Pro Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(pro_server_config)
}

#' Create Pro applications configuration
#'
#' @param server_config Server configuration
#' @return Pro applications configuration
create_pro_applications_config <- function(server_config) {
  pro_applications_config <- c(
    "# Pro Applications Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(pro_applications_config)
}

#' Create Pro security configuration
#'
#' @param server_config Server configuration
#' @return Pro security configuration
create_pro_security_config <- function(server_config) {
  pro_security_config <- c(
    "# Pro Security Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(pro_security_config)
}

#' Create Pro monitoring configuration
#'
#' @param server_config Server configuration
#' @return Pro monitoring configuration
create_pro_monitoring_config <- function(server_config) {
  pro_monitoring_config <- c(
    "# Pro Monitoring Configuration",
    "server {",
    "  listen 3838;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as shiny;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/shiny-server;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/shiny-server;",
    "  }",
    "}"
  )
  
  return(pro_monitoring_config)
}
```

## RStudio Connect Deployment

### RStudio Connect Configuration

```r
# R/02-rstudio-connect.R

#' Create RStudio Connect configuration
#'
#' @param connect_config Connect configuration
#' @return RStudio Connect configuration
create_rstudio_connect_config <- function(connect_config) {
  config <- list(
    server = create_connect_server_config(connect_config),
    applications = create_connect_applications_config(connect_config),
    security = create_connect_security_config(connect_config),
    monitoring = create_connect_monitoring_config(connect_config)
  )
  
  return(config)
}

#' Create Connect server configuration
#'
#' @param connect_config Connect configuration
#' @return Connect server configuration
create_connect_server_config <- function(connect_config) {
  connect_server_config <- c(
    "# RStudio Connect Configuration",
    "server {",
    "  listen 3939;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as rstudio-connect;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "  }",
    "}"
  )
  
  return(connect_server_config)
}

#' Create Connect applications configuration
#'
#' @param connect_config Connect configuration
#' @return Connect applications configuration
create_connect_applications_config <- function(connect_config) {
  connect_applications_config <- c(
    "# Connect Applications Configuration",
    "server {",
    "  listen 3939;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as rstudio-connect;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "  }",
    "}"
  )
  
  return(connect_applications_config)
}

#' Create Connect security configuration
#'
#' @param connect_config Connect configuration
#' @return Connect security configuration
create_connect_security_config <- function(connect_config) {
  connect_security_config <- c(
    "# Connect Security Configuration",
    "server {",
    "  listen 3939;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as rstudio-connect;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "  }",
    "}"
  )
  
  return(connect_security_config)
}

#' Create Connect monitoring configuration
#'
#' @param connect_config Connect configuration
#' @return Connect monitoring configuration
create_connect_monitoring_config <- function(connect_config) {
  connect_monitoring_config <- c(
    "# Connect Monitoring Configuration",
    "server {",
    "  listen 3939;",
    "  ",
    "  # Define the user we should use to spawn R processes",
    "  run_as rstudio-connect;",
    "  ",
    "  # Define a location that can be used to run R scripts",
    "  location / {",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "    ",
    "    # When a user visits the base URL, make sure they see the index page",
    "    directory_index on;",
    "  }",
    "  ",
    "  # Define a location for the admin interface",
    "  location /admin {",
    "    # Only allow access from localhost",
    "    allow 127.0.0.1;",
    "    deny all;",
    "    ",
    "    # Host the directory of Shiny Apps stored in this directory",
    "    site_dir /srv/rstudio-connect;",
    "    ",
    "    # Log all Shiny output to files in this directory",
    "    log_dir /var/log/rstudio-connect;",
    "  }",
    "}"
  )
  
  return(connect_monitoring_config)
}
```

## Cloud Deployment

### AWS Deployment

```r
# R/03-cloud-deployment.R

#' Create AWS deployment configuration
#'
#' @param aws_config AWS configuration
#' @return AWS deployment configuration
create_aws_deployment_config <- function(aws_config) {
  config <- list(
    ec2 = create_ec2_config(aws_config),
    rds = create_rds_config(aws_config),
    s3 = create_s3_config(aws_config),
    cloudfront = create_cloudfront_config(aws_config)
  )
  
  return(config)
}

#' Create EC2 configuration
#'
#' @param aws_config AWS configuration
#' @return EC2 configuration
create_ec2_config <- function(aws_config) {
  ec2_config <- c(
    "# EC2 Configuration",
    "resource \"aws_instance\" \"r_server\" {",
    "  ami           = \"ami-0c02fb55956c7d316\"",
    "  instance_type = \"t3.medium\"",
    "  ",
    "  vpc_security_group_ids = [aws_security_group.r_server.id]",
    "  ",
    "  user_data = <<-EOF",
    "    #!/bin/bash",
    "    yum update -y",
    "    yum install -y R",
    "    yum install -y shiny-server",
    "    systemctl start shiny-server",
    "    systemctl enable shiny-server",
    "  EOF",
    "  ",
    "  tags = {",
    "    Name = \"R Server\"",
    "  }",
    "}"
  )
  
  return(ec2_config)
}

#' Create RDS configuration
#'
#' @param aws_config AWS configuration
#' @return RDS configuration
create_rds_config <- function(aws_config) {
  rds_config <- c(
    "# RDS Configuration",
    "resource \"aws_db_instance\" \"r_database\" {",
    "  identifier = \"r-database\"",
    "  ",
    "  engine         = \"postgres\"",
    "  engine_version = \"13.7\"",
    "  instance_class = \"db.t3.micro\"",
    "  ",
    "  allocated_storage     = 20",
    "  max_allocated_storage = 100",
    "  storage_type          = \"gp2\"",
    "  ",
    "  db_name  = \"rdatabase\"",
    "  username = \"ruser\"",
    "  password = \"rpassword\"",
    "  ",
    "  vpc_security_group_ids = [aws_security_group.r_database.id]",
    "  ",
    "  backup_retention_period = 7",
    "  backup_window          = \"03:00-04:00\"",
    "  maintenance_window     = \"sun:04:00-sun:05:00\"",
    "  ",
    "  skip_final_snapshot = true",
    "  ",
    "  tags = {",
    "    Name = \"R Database\"",
    "  }",
    "}"
  )
  
  return(rds_config)
}

#' Create S3 configuration
#'
#' @param aws_config AWS configuration
#' @return S3 configuration
create_s3_config <- function(aws_config) {
  s3_config <- c(
    "# S3 Configuration",
    "resource \"aws_s3_bucket\" \"r_data\" {",
    "  bucket = \"r-data-bucket\"",
    "  ",
    "  tags = {",
    "    Name = \"R Data Bucket\"",
    "  }",
    "}",
    "",
    "resource \"aws_s3_bucket_versioning\" \"r_data_versioning\" {",
    "  bucket = aws_s3_bucket.r_data.id",
    "  versioning_configuration {",
    "    status = \"Enabled\"",
    "  }",
    "}",
    "",
    "resource \"aws_s3_bucket_server_side_encryption_configuration\" \"r_data_encryption\" {",
    "  bucket = aws_s3_bucket.r_data.id",
    "  ",
    "  rule {",
    "    "apply_server_side_encryption_by_default {",
    "      sse_algorithm = \"AES256\"",
    "    }",
    "  }",
    "}"
  )
  
  return(s3_config)
}

#' Create CloudFront configuration
#'
#' @param aws_config AWS configuration
#' @return CloudFront configuration
create_cloudfront_config <- function(aws_config) {
  cloudfront_config <- c(
    "# CloudFront Configuration",
    "resource \"aws_cloudfront_distribution\" \"r_distribution\" {",
    "  origin {",
    "    domain_name = aws_instance.r_server.public_dns",
    "    origin_id   = \"r-server\"",
    "    ",
    "    custom_origin_config {",
    "      http_port              = 80",
    "      https_port             = 443",
    "      origin_protocol_policy = \"http-only\"",
    "      origin_ssl_protocols   = [\"TLSv1.2\"]",
    "    }",
    "  }",
    "  ",
    "  enabled             = true",
    "  is_ipv6_enabled     = true",
    "  default_root_object = \"index.html\"",
    "  ",
    "  default_cache_behavior {",
    "    allowed_methods  = [\"DELETE\", \"GET\", \"HEAD\", \"OPTIONS\", \"PATCH\", \"POST\", \"PUT\"]",
    "    cached_methods   = [\"GET\", \"HEAD\"]",
    "    target_origin_id = \"r-server\"",
    "    ",
    "    forwarded_values {",
    "      query_string = false",
    "      ",
    "      cookies {",
    "        forward = \"none\"",
    "      }",
    "    }",
    "    ",
    "    viewer_protocol_policy = \"redirect-to-https\"",
    "    min_ttl                = 0",
    "    default_ttl            = 3600",
    "    max_ttl                = 86400",
    "  }",
    "  ",
    "  price_class = \"PriceClass_All\"",
    "  ",
    "  restrictions {",
    "    geo_restriction {",
    "      restriction_type = \"none\"",
    "    }",
    "  }",
    "  ",
    "  viewer_certificate {",
    "    cloudfront_default_certificate = true",
    "  }",
    "}"
  )
  
  return(cloudfront_config)
}
```

### Azure Deployment

```r
# R/03-cloud-deployment.R (continued)

#' Create Azure deployment configuration
#'
#' @param azure_config Azure configuration
#' @return Azure deployment configuration
create_azure_deployment_config <- function(azure_config) {
  config <- list(
    vm = create_azure_vm_config(azure_config),
    database = create_azure_database_config(azure_config),
    storage = create_azure_storage_config(azure_config),
    cdn = create_azure_cdn_config(azure_config)
  )
  
  return(config)
}

#' Create Azure VM configuration
#'
#' @param azure_config Azure configuration
#' @return Azure VM configuration
create_azure_vm_config <- function(azure_config) {
  azure_vm_config <- c(
    "# Azure VM Configuration",
    "resource \"azurerm_virtual_machine\" \"r_server\" {",
    "  name                  = \"r-server\"",
    "  location              = azurerm_resource_group.r_group.location",
    "  resource_group_name   = azurerm_resource_group.r_group.name",
    "  network_interface_ids = [azurerm_network_interface.r_nic.id]",
    "  vm_size               = \"Standard_B2s\"",
    "  ",
    "  storage_image_reference {",
    "    publisher = \"Canonical\"",
    "    offer     = \"UbuntuServer\"",
    "    sku       = \"18.04-LTS\"",
    "    version   = \"latest\"",
    "  }",
    "  ",
    "  storage_os_disk {",
    "    name              = \"r-server-os\"",
    "    caching           = \"ReadWrite\"",
    "    create_option     = \"FromImage\"",
    "    managed_disk_type = \"Standard_LRS\"",
    "  }",
    "  ",
    "  os_profile {",
    "    computer_name  = \"r-server\"",
    "    admin_username = \"azureuser\"",
    "    admin_password = \"AzurePassword123!\"",
    "  }",
    "  ",
    "  os_profile_linux_config {",
    "    disable_password_authentication = false",
    "  }",
    "  ",
    "  tags = {",
    "    environment = \"production\"",
    "  }",
    "}"
  )
  
  return(azure_vm_config)
}

#' Create Azure database configuration
#'
#' @param azure_config Azure configuration
#' @return Azure database configuration
create_azure_database_config <- function(azure_config) {
  azure_database_config <- c(
    "# Azure Database Configuration",
    "resource \"azurerm_postgresql_server\" \"r_database\" {",
    "  name                = \"r-database-server\"",
    "  location            = azurerm_resource_group.r_group.location",
    "  resource_group_name = azurerm_resource_group.r_group.name",
    "  ",
    "  administrator_login          = \"ruser\"",
    "  administrator_login_password = \"rpassword\"",
    "  ",
    "  sku_name   = \"B_Gen5_2\"",
    "  version    = \"11\"",
    "  storage_mb = 51200",
    "  ",
    "  backup_retention_days        = 7",
    "  geo_redundant_backup_enabled = false",
    "  ",
    "  ssl_enforcement_enabled = true",
    "  ",
    "  tags = {",
    "    environment = \"production\"",
    "  }",
    "}",
    "",
    "resource \"azurerm_postgresql_database\" \"r_database\" {",
    "  name                = \"rdatabase\"",
    "  resource_group_name = azurerm_resource_group.r_group.name",
    "  server_name         = azurerm_postgresql_server.r_database.name",
    "  charset             = \"UTF8\"",
    "  collation           = \"English_United States.utf8\"",
    "}"
  )
  
  return(azure_database_config)
}

#' Create Azure storage configuration
#'
#' @param azure_config Azure configuration
#' @return Azure storage configuration
create_azure_storage_config <- function(azure_config) {
  azure_storage_config <- c(
    "# Azure Storage Configuration",
    "resource \"azurerm_storage_account\" \"r_storage\" {",
    "  name                     = \"rstorageaccount\"",
    "  resource_group_name      = azurerm_resource_group.r_group.name",
    "  location                 = azurerm_resource_group.r_group.location",
    "  account_tier             = \"Standard\"",
    "  account_replication_type = \"LRS\"",
    "  ",
    "  tags = {",
    "    environment = \"production\"",
    "  }",
    "}",
    "",
    "resource \"azurerm_storage_container\" \"r_data\" {",
    "  name                  = \"r-data\"",
    "  storage_account_name  = azurerm_storage_account.r_storage.name",
    "  container_access_type = \"private\"",
    "}"
  )
  
  return(azure_storage_config)
}

#' Create Azure CDN configuration
#'
#' @param azure_config Azure configuration
#' @return Azure CDN configuration
create_azure_cdn_config <- function(azure_config) {
  azure_cdn_config <- c(
    "# Azure CDN Configuration",
    "resource \"azurerm_cdn_profile\" \"r_cdn\" {",
    "  name                = \"r-cdn-profile\"",
    "  location            = azurerm_resource_group.r_group.location",
    "  resource_group_name = azurerm_resource_group.r_group.name",
    "  sku                 = \"Standard_Microsoft\"",
    "  ",
    "  tags = {",
    "    environment = \"production\"",
    "  }",
    "}",
    "",
    "resource \"azurerm_cdn_endpoint\" \"r_cdn_endpoint\" {",
    "  name                = \"r-cdn-endpoint\"",
    "  profile_name        = azurerm_cdn_profile.r_cdn.name",
    "  location            = azurerm_resource_group.r_group.location",
    "  resource_group_name = azurerm_resource_group.r_group.name",
    "  ",
    "  origin {",
    "    name      = \"r-server\"",
    "    host_name = azurerm_public_ip.r_public_ip.fqdn",
    "  }",
    "  ",
    "  tags = {",
    "    environment = \"production\"",
    "  }",
    "}"
  )
  
  return(azure_cdn_config)
}
```

## Docker Deployment

### Docker Compose Configuration

```r
# R/04-docker-deployment.R

#' Create Docker deployment configuration
#'
#' @param docker_config Docker configuration
#' @return Docker deployment configuration
create_docker_deployment_config <- function(docker_config) {
  config <- list(
    compose = create_docker_compose_config(docker_config),
    dockerfile = create_dockerfile_config(docker_config),
    nginx = create_nginx_config(docker_config)
  )
  
  return(config)
}

#' Create Docker Compose configuration
#'
#' @param docker_config Docker configuration
#' @return Docker Compose configuration
create_docker_compose_config <- function(docker_config) {
  docker_compose_config <- c(
    "# Docker Compose Configuration",
    "version: '3.8'",
    "",
    "services:",
    "  r-app:",
    "    build: .",
    "    ports:",
    "      - \"3838:3838\"",
    "    environment:",
    "      - R_ENV=production",
    "    volumes:",
    "      - ./app:/srv/shiny-server",
    "      - ./logs:/var/log/shiny-server",
    "    depends_on:",
    "      - r-database",
    "    restart: unless-stopped",
    "",
    "  r-database:",
    "    image: postgres:13",
    "    environment:",
    "      - POSTGRES_DB=rdatabase",
    "      - POSTGRES_USER=ruser",
    "      - POSTGRES_PASSWORD=rpassword",
    "    volumes:",
    "      - postgres_data:/var/lib/postgresql/data",
    "    ports:",
    "      - \"5432:5432\"",
    "    restart: unless-stopped",
    "",
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
    "    restart: unless-stopped",
    "",
    "volumes:",
    "  postgres_data:"
  )
  
  return(docker_compose_config)
}

#' Create Dockerfile configuration
#'
#' @param docker_config Docker configuration
#' @return Dockerfile configuration
create_dockerfile_config <- function(docker_config) {
  dockerfile_config <- c(
    "# Dockerfile Configuration",
    "FROM rocker/shiny:4.3.0",
    "",
    "# Install system dependencies",
    "RUN apt-get update && apt-get install -y \\",
    "    libcurl4-openssl-dev \\",
    "    libssl-dev \\",
    "    libxml2-dev \\",
    "    libpq-dev \\",
    "    && rm -rf /var/lib/apt/lists/*",
    "",
    "# Install R packages",
    "RUN R -e \"install.packages(c('shiny', 'DT', 'plotly', 'leaflet', 'dplyr', 'ggplot2', 'RPostgreSQL'))\"",
    "",
    "# Copy application files",
    "COPY app/ /srv/shiny-server/",
    "",
    "# Set working directory",
    "WORKDIR /srv/shiny-server",
    "",
    "# Expose port",
    "EXPOSE 3838",
    "",
    "# Start Shiny Server",
    "CMD [\"shiny-server\"]"
  )
  
  return(dockerfile_config)
}

#' Create Nginx configuration
#'
#' @param docker_config Docker configuration
#' @return Nginx configuration
create_nginx_config <- function(docker_config) {
  nginx_config <- c(
    "# Nginx Configuration",
    "events {",
    "    worker_connections 1024;",
    "}",
    "",
    "http {",
    "    upstream r_app {",
    "        server r-app:3838;",
    "    }",
    "",
    "    server {",
    "        listen 80;",
    "        server_name localhost;",
    "",
    "        location / {",
    "            proxy_pass http://r_app;",
    "            proxy_set_header Host $host;",
    "            proxy_set_header X-Real-IP $remote_addr;",
    "            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
    "            proxy_set_header X-Forwarded-Proto $scheme;",
    "        }",
    "    }",
    "}"
  )
  
  return(nginx_config)
}
```

## Monitoring and Health Checks

### Application Monitoring

```r
# R/05-monitoring-health.R

#' Create application monitoring
#'
#' @param monitoring_config Monitoring configuration
#' @return Application monitoring
create_application_monitoring <- function(monitoring_config) {
  monitoring <- list(
    health_checks = create_health_checks(monitoring_config),
    metrics = create_application_metrics(monitoring_config),
    alerts = create_application_alerts(monitoring_config)
  )
  
  return(monitoring)
}

#' Create health checks
#'
#' @param monitoring_config Monitoring configuration
#' @return Health checks
create_health_checks <- function(monitoring_config) {
  health_checks <- c(
    "# Health Checks",
    "health_check:",
    "  stage: health",
    "  image: rocker/r-ver:4.3.0",
    "  script:",
    "    - echo 'Running health checks...'",
    "    - Rscript -e 'source(\"health_check.R\")'",
    "  artifacts:",
    "    reports:",
    "      junit: health-results.xml",
    "    paths:",
    "      - health-results/"
  )
  
  return(health_checks)
}

#' Create application metrics
#'
#' @param monitoring_config Monitoring configuration
#' @return Application metrics
create_application_metrics <- function(monitoring_config) {
  application_metrics <- c(
    "# Application Metrics",
    "application_metrics:",
    "  stage: metrics",
    "  image: rocker/r-ver:4.3.0",
    "  script:",
    "    - echo 'Collecting application metrics...'",
    "    - Rscript -e 'source(\"metrics.R\")'",
    "  artifacts:",
    "    reports:",
    "      junit: metrics.xml",
    "    paths:",
    "      - metrics/"
  )
  
  return(application_metrics)
}

#' Create application alerts
#'
#' @param monitoring_config Monitoring configuration
#' @return Application alerts
create_application_alerts <- function(monitoring_config) {
  application_alerts <- c(
    "# Application Alerts",
    "application_alerts:",
    "  stage: alerts",
    "  image: rocker/r-ver:4.3.0",
    "  script:",
    "    - echo 'Setting up application alerts...'",
    "    - Rscript -e 'source(\"alerts.R\")'",
    "  rules:",
    "    - if: '$CI_PIPELINE_STATUS == \"failed\"'",
    "    - if: '$CI_PIPELINE_STATUS == \"success\"'"
  )
  
  return(application_alerts)
}
```

## TL;DR Runbook

### Quick Start

```r
# 1. Create Shiny Server configuration
shiny_config <- create_shiny_server_config(server_config)

# 2. Create RStudio Connect configuration
connect_config <- create_rstudio_connect_config(connect_config)

# 3. Create cloud deployment configuration
aws_config <- create_aws_deployment_config(aws_config)
azure_config <- create_azure_deployment_config(azure_config)

# 4. Create Docker deployment configuration
docker_config <- create_docker_deployment_config(docker_config)

# 5. Create monitoring configuration
monitoring_config <- create_application_monitoring(monitoring_config)
```

### Essential Patterns

```r
# Complete production deployment
create_production_deployment <- function(deployment_config) {
  # Create server configuration
  server_config <- create_shiny_server_config(deployment_config$server_config)
  
  # Create cloud deployment
  cloud_config <- create_aws_deployment_config(deployment_config$cloud_config)
  
  # Create Docker deployment
  docker_config <- create_docker_deployment_config(deployment_config$docker_config)
  
  # Create monitoring
  monitoring_config <- create_application_monitoring(deployment_config$monitoring_config)
  
  return(list(
    server = server_config,
    cloud = cloud_config,
    docker = docker_config,
    monitoring = monitoring_config
  ))
}
```

---

*This guide provides the complete machinery for implementing production deployment for R applications. Each pattern includes implementation examples, deployment strategies, and real-world usage patterns for enterprise deployment.*
