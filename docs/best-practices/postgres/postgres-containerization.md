# PostgreSQL Containerization Best Practices

**Objective**: Master senior-level PostgreSQL containerization patterns for production systems. When you need to containerize PostgreSQL, when you want to implement Docker best practices, when you need enterprise-grade container strategies—these best practices become your weapon of choice.

## Core Principles

- **Immutable Infrastructure**: Use containers for consistent deployments
- **Security**: Implement container security best practices
- **Performance**: Optimize container resource usage
- **Scalability**: Design for container orchestration
- **Monitoring**: Track container health and performance

## Docker Configuration

### Multi-Stage Dockerfile

```dockerfile
# Dockerfile for PostgreSQL with extensions
FROM postgres:15-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    postgresql-dev \
    postgresql-contrib \
    postgis \
    && rm -rf /var/cache/apk/*

# Install PostgreSQL extensions
RUN echo "CREATE EXTENSION IF NOT EXISTS postgis;" > /docker-entrypoint-initdb.d/01-postgis.sql
RUN echo "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;" > /docker-entrypoint-initdb.d/02-pg_stat_statements.sql
RUN echo "CREATE EXTENSION IF NOT EXISTS pgcrypto;" > /docker-entrypoint-initdb.d/03-pgcrypto.sql

# Production stage
FROM postgres:15-alpine AS production

# Copy extensions from base stage
COPY --from=base /usr/local/lib/postgresql/* /usr/local/lib/postgresql/
COPY --from=base /usr/local/share/postgresql/extension/* /usr/local/share/postgresql/extension/

# Set environment variables
ENV POSTGRES_DB=production
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=secure_password
ENV POSTGRES_INITDB_ARGS="--auth-host=scram-sha-256"

# Create custom configuration
COPY postgresql.conf /etc/postgresql/postgresql.conf
COPY pg_hba.conf /etc/postgresql/pg_hba.conf

# Set proper permissions
RUN chown -R postgres:postgres /etc/postgresql
RUN chmod 600 /etc/postgresql/postgresql.conf
RUN chmod 600 /etc/postgresql/pg_hba.conf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres -d production || exit 1

# Expose port
EXPOSE 5432

# Use custom configuration
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: postgres-production
    restart: unless-stopped
    environment:
      POSTGRES_DB: production
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./config/pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
      - ./scripts:/docker-entrypoint-initdb.d:ro
      - ./backups:/backups
    ports:
      - "5432:5432"
    networks:
      - postgres_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d production"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run/postgresql

  postgres-replica:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: postgres-replica
    restart: unless-stopped
    environment:
      POSTGRES_DB: production
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      PGUSER: postgres
      PGPASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
      - ./config/postgresql-replica.conf:/etc/postgresql/postgresql.conf:ro
      - ./config/pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
    ports:
      - "5433:5432"
    networks:
      - postgres_network
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d production"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD}
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    ports:
      - "8080:80"
    networks:
      - postgres_network
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
    driver: local
  postgres_replica_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  postgres_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Container Security

### Security Configuration

```sql
-- Create container security configuration table
CREATE TABLE container_security_config (
    id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) UNIQUE NOT NULL,
    security_level VARCHAR(20) NOT NULL,
    config_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to configure container security
CREATE OR REPLACE FUNCTION configure_container_security(
    p_config_name VARCHAR(100),
    p_security_level VARCHAR(20),
    p_config_data JSONB
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO container_security_config (
        config_name, security_level, config_data
    ) VALUES (
        p_config_name, p_security_level, p_config_data
    ) ON CONFLICT (config_name) 
    DO UPDATE SET 
        security_level = EXCLUDED.security_level,
        config_data = EXCLUDED.config_data;
END;
$$ LANGUAGE plpgsql;

-- Create function to get security configuration
CREATE OR REPLACE FUNCTION get_security_config(p_config_name VARCHAR(100))
RETURNS TABLE (
    config_name VARCHAR(100),
    security_level VARCHAR(20),
    config_data JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        csc.config_name, csc.security_level, csc.config_data
    FROM container_security_config csc
    WHERE csc.config_name = p_config_name AND csc.is_active = TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Network Security

```sql
-- Create network security configuration table
CREATE TABLE network_security_config (
    id SERIAL PRIMARY KEY,
    network_name VARCHAR(100) NOT NULL,
    security_policy JSONB NOT NULL,
    allowed_ports INTEGER[],
    denied_ports INTEGER[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to configure network security
CREATE OR REPLACE FUNCTION configure_network_security(
    p_network_name VARCHAR(100),
    p_security_policy JSONB,
    p_allowed_ports INTEGER[] DEFAULT NULL,
    p_denied_ports INTEGER[] DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO network_security_config (
        network_name, security_policy, allowed_ports, denied_ports
    ) VALUES (
        p_network_name, p_security_policy, p_allowed_ports, p_denied_ports
    ) ON CONFLICT (network_name) 
    DO UPDATE SET 
        security_policy = EXCLUDED.security_policy,
        allowed_ports = EXCLUDED.allowed_ports,
        denied_ports = EXCLUDED.denied_ports;
END;
$$ LANGUAGE plpgsql;
```

## Container Orchestration

### Kubernetes Configuration

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  labels:
    app: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: production
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
            - -d
            - production
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - postgres
            - -d
            - production
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
      - name: postgres-config
        configMap:
          name: postgres-config
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
type: Opaque
data:
  password: c2VjdXJlX3Bhc3N3b3Jk  # base64 encoded
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  postgresql.conf: |
    # Connection settings
    listen_addresses = '*'
    port = 5432
    max_connections = 100
    
    # Memory settings
    shared_buffers = 256MB
    effective_cache_size = 1GB
    work_mem = 4MB
    
    # Logging settings
    log_destination = 'stderr'
    logging_collector = on
    log_directory = '/var/log/postgresql'
    log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
    log_rotation_age = 1d
    log_rotation_size = 100MB
    
    # Performance settings
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
```

### Docker Swarm Configuration

```yaml
# docker-stack.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
    environment:
      POSTGRES_DB: production
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    networks:
      - postgres_network
    secrets:
      - postgres_password
    configs:
      - postgres_config

  postgres-replica:
    image: postgres:15-alpine
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == worker
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    environment:
      POSTGRES_DB: production
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_replica_data:/var/lib/postgresql/data
    networks:
      - postgres_network
    depends_on:
      - postgres
    secrets:
      - postgres_password

volumes:
  postgres_data:
    driver: local
  postgres_replica_data:
    driver: local

networks:
  postgres_network:
    driver: overlay
    attachable: true

secrets:
  postgres_password:
    external: true

configs:
  postgres_config:
    external: true
```

## Container Monitoring

### Health Checks

```sql
-- Create container health monitoring table
CREATE TABLE container_health_monitoring (
    id SERIAL PRIMARY KEY,
    container_id VARCHAR(100) NOT NULL,
    container_name VARCHAR(100) NOT NULL,
    health_status VARCHAR(20) NOT NULL,
    cpu_usage NUMERIC,
    memory_usage NUMERIC,
    disk_usage NUMERIC,
    network_io NUMERIC,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create function to record container health
CREATE OR REPLACE FUNCTION record_container_health(
    p_container_id VARCHAR(100),
    p_container_name VARCHAR(100),
    p_health_status VARCHAR(20),
    p_cpu_usage NUMERIC DEFAULT NULL,
    p_memory_usage NUMERIC DEFAULT NULL,
    p_disk_usage NUMERIC DEFAULT NULL,
    p_network_io NUMERIC DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO container_health_monitoring (
        container_id, container_name, health_status,
        cpu_usage, memory_usage, disk_usage, network_io
    ) VALUES (
        p_container_id, p_container_name, p_health_status,
        p_cpu_usage, p_memory_usage, p_disk_usage, p_network_io
    );
END;
$$ LANGUAGE plpgsql;

-- Create function to get container health report
CREATE OR REPLACE FUNCTION get_container_health_report(
    p_container_name VARCHAR(100),
    p_start_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP - INTERVAL '1 hour',
    p_end_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TABLE (
    container_name VARCHAR(100),
    avg_cpu_usage NUMERIC,
    avg_memory_usage NUMERIC,
    avg_disk_usage NUMERIC,
    avg_network_io NUMERIC,
    health_status_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        chm.container_name,
        AVG(chm.cpu_usage) as avg_cpu_usage,
        AVG(chm.memory_usage) as avg_memory_usage,
        AVG(chm.disk_usage) as avg_disk_usage,
        AVG(chm.network_io) as avg_network_io,
        COUNT(*) as health_status_count
    FROM container_health_monitoring chm
    WHERE chm.container_name = p_container_name
    AND chm.recorded_at BETWEEN p_start_date AND p_end_date
    GROUP BY chm.container_name;
END;
$$ LANGUAGE plpgsql;
```

## Container Implementation

### Python Container Manager

```python
# containerization/postgres_container_manager.py
import psycopg2
import json
import docker
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import logging

class PostgreSQLContainerManager:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.docker_client = docker.from_env()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)
    
    def create_postgres_container(self, container_name: str, 
                                 postgres_password: str, 
                                 postgres_db: str = 'production',
                                 postgres_user: str = 'postgres',
                                 port: int = 5432,
                                 volumes: Dict[str, str] = None,
                                 environment: Dict[str, str] = None):
        """Create PostgreSQL container."""
        try:
            # Default volumes
            if volumes is None:
                volumes = {
                    'postgres_data': '/var/lib/postgresql/data',
                    './config/postgresql.conf': '/etc/postgresql/postgresql.conf:ro'
                }
            
            # Default environment
            if environment is None:
                environment = {
                    'POSTGRES_DB': postgres_db,
                    'POSTGRES_USER': postgres_user,
                    'POSTGRES_PASSWORD': postgres_password
                }
            
            # Create container
            container = self.docker_client.containers.run(
                'postgres:15-alpine',
                name=container_name,
                environment=environment,
                ports={5432: port},
                volumes=volumes,
                detach=True,
                restart_policy={'Name': 'unless-stopped'},
                healthcheck={
                    'test': ['CMD-SHELL', 'pg_isready -U postgres -d production'],
                    'interval': 30000000000,  # 30 seconds in nanoseconds
                    'timeout': 10000000000,   # 10 seconds in nanoseconds
                    'retries': 3,
                    'start_period': 40000000000  # 40 seconds in nanoseconds
                }
            )
            
            self.logger.info(f"PostgreSQL container {container_name} created successfully")
            return container
            
        except Exception as e:
            self.logger.error(f"Error creating PostgreSQL container: {e}")
            raise
    
    def start_container(self, container_name: str):
        """Start container."""
        try:
            container = self.docker_client.containers.get(container_name)
            container.start()
            self.logger.info(f"Container {container_name} started")
            
        except Exception as e:
            self.logger.error(f"Error starting container {container_name}: {e}")
            raise
    
    def stop_container(self, container_name: str):
        """Stop container."""
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop()
            self.logger.info(f"Container {container_name} stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping container {container_name}: {e}")
            raise
    
    def restart_container(self, container_name: str):
        """Restart container."""
        try:
            container = self.docker_client.containers.get(container_name)
            container.restart()
            self.logger.info(f"Container {container_name} restarted")
            
        except Exception as e:
            self.logger.error(f"Error restarting container {container_name}: {e}")
            raise
    
    def get_container_status(self, container_name: str):
        """Get container status."""
        try:
            container = self.docker_client.containers.get(container_name)
            status = container.status
            health = container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
            
            return {
                'status': status,
                'health': health,
                'created': container.attrs['Created'],
                'started_at': container.attrs['State'].get('StartedAt', None)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting container status: {e}")
            return None
    
    def get_container_metrics(self, container_name: str):
        """Get container metrics."""
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_usage = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percentage = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0
            
            # Calculate disk usage
            disk_usage = 0
            for device in stats['blkio_stats']['io_service_bytes_recursive']:
                if device['op'] == 'Read':
                    disk_usage += device['value']
                elif device['op'] == 'Write':
                    disk_usage += device['value']
            
            # Calculate network I/O
            network_io = 0
            for network in stats['networks'].values():
                network_io += network['rx_bytes'] + network['tx_bytes']
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_percentage,
                'disk_usage': disk_usage,
                'network_io': network_io
            }
            
        except Exception as e:
            self.logger.error(f"Error getting container metrics: {e}")
            return None
    
    def record_container_health(self, container_name: str, health_status: str, 
                               metrics: Dict[str, float]):
        """Record container health in database."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT record_container_health(%s, %s, %s, %s, %s, %s, %s)
                """, (
                    container_name, container_name, health_status,
                    metrics.get('cpu_usage'), metrics.get('memory_usage'),
                    metrics.get('disk_usage'), metrics.get('network_io')
                ))
                
                conn.commit()
                self.logger.info(f"Container health recorded for {container_name}")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error recording container health: {e}")
            raise
        finally:
            conn.close()
    
    def monitor_containers(self, container_names: List[str]):
        """Monitor container health and metrics."""
        for container_name in container_names:
            try:
                # Get container status
                status = self.get_container_status(container_name)
                if not status:
                    continue
                
                # Get container metrics
                metrics = self.get_container_metrics(container_name)
                if not metrics:
                    continue
                
                # Record health
                self.record_container_health(container_name, status['health'], metrics)
                
            except Exception as e:
                self.logger.error(f"Error monitoring container {container_name}: {e}")
    
    def get_container_health_report(self, container_name: str, 
                                   start_date: datetime = None, 
                                   end_date: datetime = None):
        """Get container health report."""
        conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM get_container_health_report(%s, %s, %s)
                """, (container_name, start_date, end_date))
                
                report = cur.fetchone()
                if report:
                    return {
                        'container_name': report[0],
                        'avg_cpu_usage': report[1],
                        'avg_memory_usage': report[2],
                        'avg_disk_usage': report[3],
                        'avg_network_io': report[4],
                        'health_status_count': report[5]
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting container health report: {e}")
            return None
        finally:
            conn.close()

# Usage
if __name__ == "__main__":
    manager = PostgreSQLContainerManager({
        'host': 'localhost',
        'database': 'production',
        'user': 'container_manager_user',
        'password': 'container_manager_password'
    })
    
    # Create PostgreSQL container
    container = manager.create_postgres_container(
        'postgres-production',
        'secure_password',
        'production',
        'postgres',
        5432
    )
    
    # Monitor containers
    manager.monitor_containers(['postgres-production'])
    
    print("PostgreSQL containerization setup complete")
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Build PostgreSQL container
docker build -t postgres-custom:latest .

# 2. Run PostgreSQL container
docker run -d --name postgres-production \
  -e POSTGRES_DB=production \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres-custom:latest

# 3. Check container health
docker ps
docker logs postgres-production

# 4. Monitor container metrics
docker stats postgres-production
```

### Essential Patterns

```python
# Complete PostgreSQL containerization setup
def setup_postgresql_containerization():
    # 1. Docker configuration
    # 2. Container security
    # 3. Container orchestration
    # 4. Container monitoring
    # 5. Health checks
    # 6. Resource management
    # 7. Network security
    # 8. Performance optimization
    
    print("PostgreSQL containerization setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL containerization excellence. Each pattern includes implementation examples, container strategies, and real-world usage patterns for enterprise PostgreSQL containerized systems.*
