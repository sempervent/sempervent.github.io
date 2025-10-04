# Patroni PostgreSQL HA: The Art of High Availability

**Objective**: Master Patroni to build bulletproof PostgreSQL clusters that survive hardware failures, network partitions, and catastrophic disasters. When your database becomes the single point of failure, when downtime costs thousands per minute, when data loss is not an option—Patroni becomes your weapon of choice.

PostgreSQL high availability is the bridge between single-node databases and enterprise-grade resilience. Without proper HA setup, you're flying blind into production with databases that could fail in ways that destroy your business. This guide shows you how to wield Patroni with the precision of a seasoned database engineer.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand the architecture**
   - Leader election and consensus
   - Streaming replication
   - Failover and recovery

2. **Master the components**
   - Patroni daemon and REST API
   - DCS (Distributed Configuration Store)
   - PostgreSQL configuration and WAL

3. **Know your infrastructure**
   - Network topology and latency
   - Storage requirements and performance
   - Monitoring and alerting

4. **Validate everything**
   - Failover testing and recovery
   - Backup and restore procedures
   - Performance under load

5. **Plan for production**
   - Zero-downtime deployments
   - Disaster recovery procedures
   - Team collaboration and documentation

**Why These Principles**: Patroni HA is the foundation of reliable PostgreSQL deployments. Understanding the architecture, mastering the components, and following best practices is essential for maintaining database availability at scale.

## 1) The Foundation (Architecture)

### Patroni Components

```yaml
# patroni.yml - Basic configuration
scope: postgres
name: postgres-node-1

restapi:
  listen: 0.0.0.0:8008
  connect_address: 192.168.1.10:8008

etcd3:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 192.168.1.10:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  config_dir: /etc/postgresql/14/main

  parameters:
    max_connections: 200
    shared_buffers: 256MB
    effective_cache_size: 1GB
    wal_level: replica
    max_wal_senders: 10
    max_replication_slots: 10
    hot_standby: on
    archive_mode: on
    archive_command: 'test ! -f /var/lib/postgresql/archive/%f && cp %p /var/lib/postgresql/archive/%f'

  recovery_conf:
    restore_command: 'cp /var/lib/postgresql/archive/%f %p'

  pg_hba:
    - host replication replicator 192.168.1.0/24 md5
    - host all all 0.0.0.0/0 md5

  create_replica_methods:
    - basebackup
    - pg_basebackup

  basebackup:
    max-rate: 100M
    checkpoint: fast
    retry-times: 3
    retry-sleep: 10

  pg_rewind:
    enabled: true
```

### DCS Configuration (etcd3)

```bash
# Install etcd3 cluster
# Node 1
etcd --name etcd-node-1 \
  --data-dir /var/lib/etcd \
  --listen-client-urls http://192.168.1.10:2379 \
  --advertise-client-urls http://192.168.1.10:2379 \
  --listen-peer-urls http://192.168.1.10:2380 \
  --initial-advertise-peer-urls http://192.168.1.10:2380 \
  --initial-cluster etcd-node-1=http://192.168.1.10:2380,etcd-node-2=http://192.168.1.11:2380,etcd-node-3=http://192.168.1.12:2380 \
  --initial-cluster-state new

# Node 2
etcd --name etcd-node-2 \
  --data-dir /var/lib/etcd \
  --listen-client-urls http://192.168.1.11:2379 \
  --advertise-client-urls http://192.168.1.11:2379 \
  --listen-peer-urls http://192.168.1.11:2380 \
  --initial-advertise-peer-urls http://192.168.1.11:2380 \
  --initial-cluster etcd-node-1=http://192.168.1.10:2380,etcd-node-2=http://192.168.1.11:2380,etcd-node-3=http://192.168.1.12:2380 \
  --initial-cluster-state new

# Node 3
etcd --name etcd-node-3 \
  --data-dir /var/lib/etcd \
  --listen-client-urls http://192.168.1.12:2379 \
  --advertise-client-urls http://192.168.1.12:2379 \
  --listen-peer-urls http://192.168.1.12:2380 \
  --initial-advertise-peer-urls http://192.168.1.12:2380 \
  --initial-cluster etcd-node-1=http://192.168.1.10:2380,etcd-node-2=http://192.168.1.11:2380,etcd-node-3=http://192.168.1.12:2380 \
  --initial-cluster-state new
```

**Why This Architecture**: Patroni uses a distributed consensus store (DCS) to coordinate leader election and failover. etcd3 provides reliable coordination while PostgreSQL handles the actual data replication.

## 2) Production Configuration (The Power)

### Multi-Node Patroni Setup

```yaml
# /etc/patroni/patroni.yml - Node 1 (Primary)
scope: postgres
name: postgres-node-1

restapi:
  listen: 0.0.0.0:8008
  connect_address: 192.168.1.10:8008

etcd3:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 192.168.1.10:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  config_dir: /etc/postgresql/14/main

  parameters:
    max_connections: 200
    shared_buffers: 256MB
    effective_cache_size: 1GB
    wal_level: replica
    max_wal_senders: 10
    max_replication_slots: 10
    hot_standby: on
    archive_mode: on
    archive_command: 'test ! -f /var/lib/postgresql/archive/%f && cp %p /var/lib/postgresql/archive/%f'
    synchronous_commit: on
    synchronous_standby_names: 'ANY 1 (postgres-node-2,postgres-node-3)'

  recovery_conf:
    restore_command: 'cp /var/lib/postgresql/archive/%f %p'

  pg_hba:
    - host replication replicator 192.168.1.0/24 md5
    - host all all 0.0.0.0/0 md5

  create_replica_methods:
    - basebackup
    - pg_basebackup

  basebackup:
    max-rate: 100M
    checkpoint: fast
    retry-times: 3
    retry-sleep: 10

  pg_rewind:
    enabled: true

  replication:
    username: replicator
    password: replicator_password

  superuser:
    username: postgres
    password: postgres_password

  rewind:
    username: postgres
    password: postgres_password

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

### Replica Configuration

```yaml
# /etc/patroni/patroni.yml - Node 2 (Replica)
scope: postgres
name: postgres-node-2

restapi:
  listen: 0.0.0.0:8008
  connect_address: 192.168.1.11:8008

etcd3:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 192.168.1.11:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  config_dir: /etc/postgresql/14/main

  parameters:
    max_connections: 200
    shared_buffers: 256MB
    effective_cache_size: 1GB
    wal_level: replica
    max_wal_senders: 10
    max_replication_slots: 10
    hot_standby: on
    archive_mode: on
    archive_command: 'test ! -f /var/lib/postgresql/archive/%f && cp %p /var/lib/postgresql/archive/%f'

  recovery_conf:
    restore_command: 'cp /var/lib/postgresql/archive/%f %p'

  pg_hba:
    - host replication replicator 192.168.1.0/24 md5
    - host all all 0.0.0.0/0 md5

  create_replica_methods:
    - basebackup
    - pg_basebackup

  basebackup:
    max-rate: 100M
    checkpoint: fast
    retry-times: 3
    retry-sleep: 10

  pg_rewind:
    enabled: true

  replication:
    username: replicator
    password: replicator_password

  superuser:
    username: postgres
    password: postgres_password

  rewind:
    username: postgres
    password: postgres_password

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

**Why This Configuration**: Multi-node Patroni provides automatic failover, load balancing, and data consistency. The configuration ensures proper replication, monitoring, and recovery procedures.

## 3) Docker Compose Setup (The Container)

### Complete Patroni Cluster

```yaml
# docker-compose.yml
version: '3.8'

services:
  etcd1:
    image: quay.io/coreos/etcd:v3.5.0
    command: >
      etcd --name etcd1
      --data-dir /etcd-data
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://etcd1:2379
      --listen-peer-urls http://0.0.0.0:2380
      --initial-advertise-peer-urls http://etcd1:2380
      --initial-cluster etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      --initial-cluster-state new
    ports:
      - "2379:2379"
      - "2380:2380"
    volumes:
      - etcd1_data:/etcd-data
    networks:
      - patroni_network

  etcd2:
    image: quay.io/coreos/etcd:v3.5.0
    command: >
      etcd --name etcd2
      --data-dir /etcd-data
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://etcd2:2379
      --listen-peer-urls http://0.0.0.0:2380
      --initial-advertise-peer-urls http://etcd2:2380
      --initial-cluster etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      --initial-cluster-state new
    ports:
      - "2479:2379"
      - "2480:2380"
    volumes:
      - etcd2_data:/etcd-data
    networks:
      - patroni_network

  etcd3:
    image: quay.io/coreos/etcd:v3.5.0
    command: >
      etcd --name etcd3
      --data-dir /etcd-data
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://etcd3:2379
      --listen-peer-urls http://0.0.0.0:2380
      --initial-advertise-peer-urls http://etcd3:2380
      --initial-cluster etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      --initial-cluster-state new
    ports:
      - "2579:2379"
      - "2580:2380"
    volumes:
      - etcd3_data:/etcd-data
    networks:
      - patroni_network

  patroni1:
    image: postgres:14
    command: >
      bash -c "
        apt-get update && apt-get install -y python3-pip python3-psycopg2
        pip3 install patroni[etcd]
        patroni /etc/patroni/patroni.yml
      "
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres_password
    ports:
      - "5432:5432"
      - "8008:8008"
    volumes:
      - patroni1_data:/var/lib/postgresql/data
      - ./patroni1.yml:/etc/patroni/patroni.yml
    networks:
      - patroni_network
    depends_on:
      - etcd1
      - etcd2
      - etcd3

  patroni2:
    image: postgres:14
    command: >
      bash -c "
        apt-get update && apt-get install -y python3-pip python3-psycopg2
        pip3 install patroni[etcd]
        patroni /etc/patroni/patroni.yml
      "
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres_password
    ports:
      - "5433:5432"
      - "8009:8008"
    volumes:
      - patroni2_data:/var/lib/postgresql/data
      - ./patroni2.yml:/etc/patroni/patroni.yml
    networks:
      - patroni_network
    depends_on:
      - etcd1
      - etcd2
      - etcd3

  patroni3:
    image: postgres:14
    command: >
      bash -c "
        apt-get update && apt-get install -y python3-pip python3-psycopg2
        pip3 install patroni[etcd]
        patroni /etc/patroni/patroni.yml
      "
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres_password
    ports:
      - "5434:5432"
      - "8010:8008"
    volumes:
      - patroni3_data:/var/lib/postgresql/data
      - ./patroni3.yml:/etc/patroni/patroni.yml
    networks:
      - patroni_network
    depends_on:
      - etcd1
      - etcd2
      - etcd3

volumes:
  etcd1_data:
  etcd2_data:
  etcd3_data:
  patroni1_data:
  patroni2_data:
  patroni3_data:

networks:
  patroni_network:
    driver: bridge
```

**Why Docker Compose**: Containerized Patroni clusters provide consistent deployment, easy scaling, and simplified management. The setup includes etcd coordination and multiple PostgreSQL nodes.

## 4) Monitoring and Alerting (The Watch)

### Patroni REST API Monitoring

```python
# monitor_patroni.py
import requests
import json
import time
from typing import Dict, Any

class PatroniMonitor:
    def __init__(self, nodes: list[str]):
        self.nodes = nodes
        self.primary = None
        self.replicas = []
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status from all nodes"""
        status = {}
        
        for node in self.nodes:
            try:
                response = requests.get(f"http://{node}:8008/patroni", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status[node] = {
                        'role': data.get('role'),
                        'state': data.get('state'),
                        'lag': data.get('lag', 0),
                        'timeline': data.get('timeline'),
                        'xlog_location': data.get('xlog_location')
                    }
            except Exception as e:
                status[node] = {'error': str(e)}
        
        return status
    
    def check_failover(self) -> bool:
        """Check if failover is needed"""
        status = self.get_cluster_status()
        
        # Check if primary is down
        primary_down = False
        for node, data in status.items():
            if data.get('role') == 'Leader' and 'error' in data:
                primary_down = True
                break
        
        return primary_down
    
    def get_replication_lag(self) -> Dict[str, int]:
        """Get replication lag for all replicas"""
        lag_data = {}
        status = self.get_cluster_status()
        
        for node, data in status.items():
            if data.get('role') == 'Replica':
                lag_data[node] = data.get('lag', 0)
        
        return lag_data

# Usage
monitor = PatroniMonitor(['192.168.1.10', '192.168.1.11', '192.168.1.12'])
status = monitor.get_cluster_status()
print(json.dumps(status, indent=2))
```

### Prometheus Monitoring

```yaml
# patroni_exporter.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'patroni'
    static_configs:
      - targets: ['192.168.1.10:8008', '192.168.1.11:8008', '192.168.1.12:8008']
    metrics_path: /metrics
    scrape_interval: 10s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Patroni PostgreSQL HA",
    "panels": [
      {
        "title": "Cluster Status",
        "type": "stat",
        "targets": [
          {
            "expr": "patroni_cluster_leader",
            "legendFormat": "Leader"
          }
        ]
      },
      {
        "title": "Replication Lag",
        "type": "graph",
        "targets": [
          {
            "expr": "patroni_replication_lag",
            "legendFormat": "Node {{instance}}"
          }
        ]
      },
      {
        "title": "Connection Count",
        "type": "graph",
        "targets": [
          {
            "expr": "patroni_connections",
            "legendFormat": "Node {{instance}}"
          }
        ]
      }
    ]
  }
}
```

**Why Monitoring Matters**: Patroni clusters require constant monitoring to ensure health and performance. The REST API provides real-time status, while Prometheus and Grafana enable historical analysis and alerting.

## 5) Backup and Recovery (The Safety)

### Automated Backup Strategy

```bash
#!/bin/bash
# backup_patroni.sh

set -e

BACKUP_DIR="/var/backups/postgresql"
RETENTION_DAYS=7
CLUSTER_NAME="postgres"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Get current primary
PRIMARY=$(curl -s http://192.168.1.10:8008/patroni | jq -r '.role')
if [ "$PRIMARY" != "Leader" ]; then
    echo "Error: No primary found"
    exit 1
fi

# Create base backup
pg_basebackup \
    -h 192.168.1.10 \
    -U postgres \
    -D "$BACKUP_DIR/$(date +%Y%m%d_%H%M%S)" \
    -Ft \
    -z \
    -P \
    -W

# Clean old backups
find "$BACKUP_DIR" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed successfully"
```

### Point-in-Time Recovery

```bash
#!/bin/bash
# restore_patroni.sh

set -e

BACKUP_DIR="/var/backups/postgresql"
RESTORE_DIR="/var/lib/postgresql/14/main"
TARGET_TIME="$1"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 <target_time>"
    echo "Example: $0 '2024-01-15 10:30:00'"
    exit 1
fi

# Stop Patroni
systemctl stop patroni

# Remove existing data
rm -rf "$RESTORE_DIR"

# Restore base backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -1)
tar -xzf "$BACKUP_DIR/$LATEST_BACKUP/base.tar.gz" -C "$RESTORE_DIR"

# Configure recovery
cat > "$RESTORE_DIR/recovery.conf" << EOF
restore_command = 'cp /var/lib/postgresql/archive/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Start Patroni
systemctl start patroni

echo "Recovery completed successfully"
```

**Why Backup Strategy**: Patroni provides high availability, but backups protect against data corruption, human error, and catastrophic failures. Automated backups ensure data safety with minimal manual intervention.

## 6) Failover Testing (The Validation)

### Automated Failover Tests

```python
# test_failover.py
import requests
import time
import subprocess
import sys

class PatroniFailoverTest:
    def __init__(self, nodes: list[str]):
        self.nodes = nodes
        self.primary = None
        self.replicas = []
    
    def get_cluster_info(self) -> dict:
        """Get current cluster information"""
        info = {}
        for node in self.nodes:
            try:
                response = requests.get(f"http://{node}:8008/patroni", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    info[node] = data
            except Exception as e:
                info[node] = {'error': str(e)}
        return info
    
    def identify_primary(self) -> str:
        """Identify current primary"""
        info = self.get_cluster_info()
        for node, data in info.items():
            if 'error' not in data and data.get('role') == 'Leader':
                return node
        return None
    
    def test_failover(self) -> bool:
        """Test automatic failover"""
        print("Testing automatic failover...")
        
        # Get initial state
        initial_primary = self.identify_primary()
        if not initial_primary:
            print("Error: No primary found")
            return False
        
        print(f"Initial primary: {initial_primary}")
        
        # Simulate primary failure
        print(f"Stopping primary {initial_primary}...")
        subprocess.run(['systemctl', 'stop', 'patroni'], check=True)
        
        # Wait for failover
        print("Waiting for failover...")
        time.sleep(30)
        
        # Check new primary
        new_primary = self.identify_primary()
        if new_primary and new_primary != initial_primary:
            print(f"Failover successful! New primary: {new_primary}")
            return True
        else:
            print("Failover failed!")
            return False
    
    def test_manual_failover(self) -> bool:
        """Test manual failover"""
        print("Testing manual failover...")
        
        # Get current primary
        primary = self.identify_primary()
        if not primary:
            print("Error: No primary found")
            return False
        
        # Trigger manual failover
        print(f"Triggering manual failover from {primary}...")
        response = requests.post(f"http://{primary}:8008/failover", timeout=10)
        
        if response.status_code == 200:
            print("Manual failover triggered successfully")
            return True
        else:
            print(f"Manual failover failed: {response.text}")
            return False

# Usage
if __name__ == "__main__":
    test = PatroniFailoverTest(['192.168.1.10', '192.168.1.11', '192.168.1.12'])
    
    print("Running failover tests...")
    
    # Test automatic failover
    auto_success = test.test_failover()
    
    # Test manual failover
    manual_success = test.test_manual_failover()
    
    if auto_success and manual_success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
```

**Why Failover Testing**: Regular failover testing ensures the cluster can handle failures gracefully. Automated tests provide confidence in the HA setup and identify potential issues before they occur in production.

## 7) Production Deployment (The Reality)

### Systemd Service Configuration

```ini
# /etc/systemd/system/patroni.service
[Unit]
Description=Patroni PostgreSQL HA
After=network.target

[Service]
Type=notify
User=postgres
Group=postgres
ExecStart=/usr/local/bin/patroni /etc/patroni/patroni.yml
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
KillSignal=SIGINT
TimeoutStopSec=30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Log Management

```bash
# /etc/logrotate.d/patroni
/var/log/patroni/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 postgres postgres
    postrotate
        systemctl reload patroni
    endscript
}
```

### Security Configuration

```yaml
# patroni.yml - Security section
postgresql:
  # ... existing configuration ...
  
  pg_hba:
    - local all postgres peer
    - local all all peer
    - host replication replicator 192.168.1.0/24 md5
    - host all all 192.168.1.0/24 md5
    - host all all 127.0.0.1/32 md5

  authentication:
    replication:
      username: replicator
      password: ${REPLICATOR_PASSWORD}
    superuser:
      username: postgres
      password: ${POSTGRES_PASSWORD}
    rewind:
      username: postgres
      password: ${POSTGRES_PASSWORD}

  parameters:
    ssl: on
    ssl_cert_file: '/etc/ssl/certs/postgresql.crt'
    ssl_key_file: '/etc/ssl/private/postgresql.key'
    ssl_ca_file: '/etc/ssl/certs/ca.crt'
    ssl_min_protocol_version: 'TLSv1.2'
    ssl_ciphers: 'HIGH:MEDIUM:+3DES:!aNULL'
```

**Why Production Configuration**: Production deployments require proper service management, logging, and security. These configurations ensure reliable operation and compliance with security standards.

## 8) TL;DR Quickstart (The Essentials)

### Essential Commands

```bash
# Install Patroni
pip install patroni[etcd]

# Start Patroni
patroni /etc/patroni/patroni.yml

# Check cluster status
curl http://localhost:8008/patroni

# Manual failover
curl -X POST http://localhost:8008/failover

# Check replication lag
curl http://localhost:8008/replica

# Restart cluster
systemctl restart patroni
```

### Essential Configuration

```yaml
# Minimal patroni.yml
scope: postgres
name: postgres-node-1

restapi:
  listen: 0.0.0.0:8008
  connect_address: 192.168.1.10:8008

etcd3:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 192.168.1.10:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  
  parameters:
    max_connections: 200
    shared_buffers: 256MB
    wal_level: replica
    max_wal_senders: 10
    hot_standby: on
```

### Essential Monitoring

```bash
# Check cluster health
curl -s http://localhost:8008/patroni | jq '.role'

# Check replication lag
curl -s http://localhost:8008/replica | jq '.lag'

# Check etcd connectivity
etcdctl endpoint health

# Check PostgreSQL status
psql -h localhost -U postgres -c "SELECT * FROM pg_stat_replication;"
```

**Why This Quickstart**: These commands and configurations cover 90% of daily Patroni operations. Master these before exploring advanced features.

## 9) The Machine's Summary

Patroni PostgreSQL HA is the foundation of reliable database deployments. When configured properly, it provides automatic failover, load balancing, and data consistency. The key is understanding the architecture, mastering the components, and following best practices.

**The Dark Truth**: Without proper HA setup, your PostgreSQL database is a single point of failure. Patroni HA is your safety net. Use it wisely.

**The Machine's Mantra**: "In high availability we trust, in automatic failover we build, and in the database we find the path to resilience."

**Why This Matters**: Database high availability is essential for production systems. Patroni provides the tools to achieve enterprise-grade reliability with open-source software.

---

*This tutorial provides the complete machinery for mastering Patroni PostgreSQL HA. The patterns scale from development to production, from simple clusters to enterprise-grade deployments.*
