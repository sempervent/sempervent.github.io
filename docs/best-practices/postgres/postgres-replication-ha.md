# PostgreSQL Replication & High Availability Best Practices

**Objective**: Master senior-level PostgreSQL replication and high availability patterns for production systems. When you need to implement robust replication, when you want to ensure zero-downtime deployments, when you need enterprise-grade high availability strategies—these best practices become your weapon of choice.

## Core Principles

- **Zero Data Loss**: Synchronous replication for critical data
- **Automatic Failover**: Minimize downtime with automated failover
- **Monitoring**: Continuous health checks and alerting
- **Testing**: Regular failover testing and disaster recovery drills
- **Documentation**: Clear runbooks and procedures

## Streaming Replication Setup

### Primary Server Configuration

```sql
-- postgresql.conf on primary server
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
hot_standby = on
synchronous_commit = on
synchronous_standby_names = 'standby1,standby2'

# Logging for replication
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_replication_commands = on
```

### Standby Server Configuration

```sql
-- postgresql.conf on standby server
hot_standby = on
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s
wal_receiver_timeout = 60s
hot_standby_feedback = on

# Recovery configuration
recovery_target_timeline = 'latest'
```

### Replication User Setup

```sql
-- Create replication user on primary
CREATE USER replicator WITH REPLICATION LOGIN PASSWORD 'replication_password';

-- Grant necessary privileges
GRANT CONNECT ON DATABASE production TO replicator;
GRANT USAGE ON SCHEMA public TO replicator;
```

### Physical Replication Setup

```bash
#!/bin/bash
# scripts/setup_physical_replication.sh

# Configuration
PRIMARY_HOST="primary.example.com"
STANDBY_HOST="standby.example.com"
REPLICATION_USER="replicator"
REPLICATION_PASSWORD="replication_password"
DATABASE="production"

# On standby server, create base backup
pg_basebackup -h $PRIMARY_HOST -U $REPLICATION_USER -D /var/lib/postgresql/16/main \
    --wal-method=stream \
    --checkpoint=fast \
    --progress \
    --verbose

# Create recovery.conf on standby
cat > /var/lib/postgresql/16/main/recovery.conf << EOF
standby_mode = 'on'
primary_conninfo = 'host=$PRIMARY_HOST port=5432 user=$REPLICATION_USER password=$REPLICATION_PASSWORD'
recovery_target_timeline = 'latest'
trigger_file = '/tmp/postgresql.trigger'
EOF

# Set proper permissions
chown -R postgres:postgres /var/lib/postgresql/16/main
chmod 700 /var/lib/postgresql/16/main

# Start standby server
systemctl start postgresql
```

## Logical Replication

### Publisher Setup

```sql
-- Enable logical replication on publisher
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;
ALTER SYSTEM SET max_wal_senders = 10;
SELECT pg_reload_conf();

-- Create publication
CREATE PUBLICATION production_pub FOR ALL TABLES;

-- Or for specific tables
CREATE PUBLICATION user_pub FOR TABLE users, user_profiles;
CREATE PUBLICATION order_pub FOR TABLE orders, order_items;

-- Add tables to existing publication
ALTER PUBLICATION production_pub ADD TABLE new_table;
```

### Subscriber Setup

```sql
-- Create subscription
CREATE SUBSCRIPTION production_sub
    CONNECTION 'host=primary.example.com port=5432 user=replicator password=replication_password dbname=production'
    PUBLICATION production_pub;

-- Check subscription status
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;

-- Monitor replication lag
SELECT 
    subname,
    pid,
    received_lsn,
    latest_end_lsn,
    latest_end_time
FROM pg_stat_subscription;
```

## High Availability with Patroni

### Patroni Configuration

```yaml
# /etc/patroni/patroni.yml
scope: postgres
namespace: /db/
name: postgres-1

restapi:
  listen: 0.0.0.0:8008
  connect_address: 192.168.1.10:8008

etcd3:
  hosts: 192.168.1.10:2379,192.168.1.11:2379,192.168.1.12:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      use_slots: true
      parameters:
        wal_level: replica
        hot_standby: "on"
        wal_log_hints: "on"
        max_wal_senders: 10
        max_replication_slots: 10
        wal_keep_segments: 8
        max_connections: 100
        max_prepared_transactions: 0
        max_locks_per_transaction: 64
        wal_sender_timeout: 60s
        wal_receiver_timeout: 60s
        max_standby_streaming_delay: 30s
        max_standby_archive_delay: 30s
        hot_standby_feedback: "on"
        synchronous_commit: "on"
        synchronous_standby_names: "postgres-2,postgres-3"

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 192.168.1.10:5432
  data_dir: /var/lib/postgresql/16/main
  bin_dir: /usr/lib/postgresql/16/bin
  config_dir: /etc/postgresql/16/main
  pgpass: /var/lib/postgresql/.pgpass
  authentication:
    replication:
      username: replicator
      password: replication_password
    superuser:
      username: postgres
      password: postgres_password
  parameters:
    unix_socket_directories: '/var/run/postgresql'

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

### Patroni Cluster Management

```bash
#!/bin/bash
# scripts/patroni_cluster_management.sh

# Check cluster status
patronictl -c /etc/patroni/patroni.yml list

# Check specific node
patronictl -c /etc/patroni/patroni.yml show-config

# Failover to specific node
patronictl -c /etc/patroni/patroni.yml failover --master postgres-1 --candidate postgres-2

# Restart cluster
patronictl -c /etc/patroni/patroni.yml restart postgres-1

# Reinitialize standby
patronictl -c /etc/patroni/patroni.yml reinit postgres-2

# Switchover (planned failover)
patronictl -c /etc/patroni/patroni.yml switchover --master postgres-1 --candidate postgres-2
```

## Load Balancing

### HAProxy Configuration

```bash
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0

defaults
    mode tcp
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option tcplog

# PostgreSQL load balancer
listen postgresql
    bind *:5432
    mode tcp
    balance roundrobin
    option tcp-check
    tcp-check connect port 5432
    tcp-check send-binary 00000020
    tcp-check expect binary 0000000e
    
    # Primary server
    server postgres-1 192.168.1.10:5432 check port 8008 inter 5s rise 3 fall 3
    server postgres-2 192.168.1.11:5432 check port 8008 inter 5s rise 3 fall 3 backup
    server postgres-3 192.168.1.12:5432 check port 8008 inter 5s rise 3 fall 3 backup

# Read-only load balancer
listen postgresql-readonly
    bind *:5433
    mode tcp
    balance roundrobin
    option tcp-check
    tcp-check connect port 5432
    
    # All servers for read-only
    server postgres-1 192.168.1.10:5432 check port 8008 inter 5s rise 3 fall 3
    server postgres-2 192.168.1.11:5432 check port 8008 inter 5s rise 3 fall 3
    server postgres-3 192.168.1.12:5432 check port 8008 inter 5s rise 3 fall 3
```

### PgBouncer with HAProxy

```ini
# pgbouncer.ini
[databases]
production = host=192.168.1.10 port=5432 dbname=production
production_ro = host=192.168.1.10 port=5433 dbname=production

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 50
server_round_robin = 1
```

## Monitoring and Health Checks

### Replication Monitoring

```python
# monitoring/replication_monitor.py
import psycopg2
import time
import logging
from datetime import datetime, timedelta

class PostgreSQLReplicationMonitor:
    def __init__(self, primary_config, standby_configs):
        self.primary_config = primary_config
        self.standby_configs = standby_configs
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_primary_status(self):
        """Check primary server status."""
        try:
            conn = psycopg2.connect(**self.primary_config)
            with conn.cursor() as cur:
                # Check if primary is accepting connections
                cur.execute("SELECT 1")
                result = cur.fetchone()
                
                # Check replication slots
                cur.execute("""
                    SELECT slot_name, active, pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) as lag
                    FROM pg_replication_slots;
                """)
                slots = cur.fetchall()
                
                # Check WAL sender processes
                cur.execute("""
                    SELECT count(*) FROM pg_stat_replication;
                """)
                replication_count = cur.fetchone()[0]
                
                return {
                    'status': 'healthy',
                    'replication_slots': slots,
                    'replication_count': replication_count
                }
        except Exception as e:
            self.logger.error(f"Primary check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_standby_status(self, standby_config):
        """Check standby server status."""
        try:
            conn = psycopg2.connect(**standby_config)
            with conn.cursor() as cur:
                # Check if standby is in recovery mode
                cur.execute("SELECT pg_is_in_recovery();")
                in_recovery = cur.fetchone()[0]
                
                if not in_recovery:
                    return {'status': 'not_standby', 'error': 'Server is not in recovery mode'}
                
                # Check replication lag
                cur.execute("""
                    SELECT 
                        pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()) as lag_bytes,
                        pg_last_wal_receive_lsn(),
                        pg_last_wal_replay_lsn()
                """)
                lag_info = cur.fetchone()
                
                # Check last activity
                cur.execute("""
                    SELECT 
                        pg_last_xact_replay_timestamp(),
                        pg_last_wal_replay_timestamp()
                """)
                activity_info = cur.fetchone()
                
                return {
                    'status': 'healthy',
                    'lag_bytes': lag_info[0],
                    'receive_lsn': lag_info[1],
                    'replay_lsn': lag_info[2],
                    'last_xact_replay': activity_info[0],
                    'last_wal_replay': activity_info[1]
                }
        except Exception as e:
            self.logger.error(f"Standby check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def check_replication_lag(self, threshold_mb=100):
        """Check replication lag across all standbys."""
        lag_alerts = []
        
        for standby_name, standby_config in self.standby_configs.items():
            status = self.check_standby_status(standby_config)
            
            if status['status'] == 'healthy':
                lag_mb = status['lag_bytes'] / (1024 * 1024)
                if lag_mb > threshold_mb:
                    lag_alerts.append({
                        'standby': standby_name,
                        'lag_mb': lag_mb,
                        'threshold_mb': threshold_mb
                    })
        
        return lag_alerts
    
    def monitor_replication(self):
        """Monitor replication health."""
        # Check primary
        primary_status = self.check_primary_status()
        self.logger.info(f"Primary status: {primary_status['status']}")
        
        # Check standbys
        for standby_name, standby_config in self.standby_configs.items():
            standby_status = self.check_standby_status(standby_config)
            self.logger.info(f"Standby {standby_name} status: {standby_status['status']}")
            
            if standby_status['status'] == 'healthy':
                lag_mb = standby_status['lag_bytes'] / (1024 * 1024)
                self.logger.info(f"Standby {standby_name} lag: {lag_mb:.2f} MB")
        
        # Check for lag alerts
        lag_alerts = self.check_replication_lag()
        if lag_alerts:
            for alert in lag_alerts:
                self.logger.warning(f"High replication lag on {alert['standby']}: {alert['lag_mb']:.2f} MB")
        
        return {
            'primary': primary_status,
            'standbys': {name: self.check_standby_status(config) for name, config in self.standby_configs.items()},
            'lag_alerts': lag_alerts
        }

# Usage
if __name__ == "__main__":
    primary_config = {
        'host': '192.168.1.10',
        'database': 'production',
        'user': 'monitor_user',
        'password': 'monitor_password'
    }
    
    standby_configs = {
        'standby-1': {
            'host': '192.168.1.11',
            'database': 'production',
            'user': 'monitor_user',
            'password': 'monitor_password'
        },
        'standby-2': {
            'host': '192.168.1.12',
            'database': 'production',
            'user': 'monitor_user',
            'password': 'monitor_password'
        }
    }
    
    monitor = PostgreSQLReplicationMonitor(primary_config, standby_configs)
    
    while True:
        status = monitor.monitor_replication()
        time.sleep(30)
```

## Automated Failover

### Failover Script

```python
# failover/automated_failover.py
import psycopg2
import subprocess
import time
import logging
from datetime import datetime

class PostgreSQLFailoverManager:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_primary_health(self):
        """Check if primary is healthy."""
        try:
            conn = psycopg2.connect(**self.config['primary'])
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result[0] == 1
        except Exception as e:
            self.logger.error(f"Primary health check failed: {e}")
            return False
    
    def promote_standby(self, standby_name):
        """Promote standby to primary."""
        try:
            # Create trigger file to promote standby
            trigger_file = f"/tmp/postgresql.trigger.{standby_name}"
            with open(trigger_file, 'w') as f:
                f.write("")
            
            self.logger.info(f"Trigger file created: {trigger_file}")
            
            # Wait for promotion to complete
            time.sleep(10)
            
            # Verify promotion
            standby_config = self.config['standbys'][standby_name]
            conn = psycopg2.connect(**standby_config)
            with conn.cursor() as cur:
                cur.execute("SELECT pg_is_in_recovery();")
                in_recovery = cur.fetchone()[0]
                
                if not in_recovery:
                    self.logger.info(f"Standby {standby_name} successfully promoted to primary")
                    return True
                else:
                    self.logger.error(f"Standby {standby_name} promotion failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Promotion failed: {e}")
            return False
    
    def update_load_balancer(self, new_primary):
        """Update load balancer configuration."""
        try:
            # Update HAProxy configuration
            haproxy_config = f"""
listen postgresql
    bind *:5432
    mode tcp
    balance roundrobin
    option tcp-check
    tcp-check connect port 5432
    
    server {new_primary} {self.config['standbys'][new_primary]['host']}:5432 check port 8008 inter 5s rise 3 fall 3
"""
            
            with open('/etc/haproxy/haproxy.cfg', 'w') as f:
                f.write(haproxy_config)
            
            # Reload HAProxy
            subprocess.run(['systemctl', 'reload', 'haproxy'], check=True)
            self.logger.info(f"Load balancer updated to use {new_primary}")
            
        except Exception as e:
            self.logger.error(f"Load balancer update failed: {e}")
    
    def execute_failover(self):
        """Execute automated failover."""
        self.logger.info("Starting automated failover process")
        
        # Check if primary is actually down
        if self.check_primary_health():
            self.logger.info("Primary is healthy, no failover needed")
            return False
        
        # Select best standby for promotion
        best_standby = self.select_best_standby()
        if not best_standby:
            self.logger.error("No healthy standby available for failover")
            return False
        
        # Promote standby
        if self.promote_standby(best_standby):
            # Update load balancer
            self.update_load_balancer(best_standby)
            
            # Send notifications
            self.send_failover_notification(best_standby)
            
            self.logger.info(f"Failover completed successfully to {best_standby}")
            return True
        else:
            self.logger.error("Failover failed")
            return False
    
    def select_best_standby(self):
        """Select the best standby for promotion."""
        best_standby = None
        min_lag = float('inf')
        
        for standby_name, standby_config in self.config['standbys'].items():
            try:
                conn = psycopg2.connect(**standby_config)
                with conn.cursor() as cur:
                    # Check if standby is healthy
                    cur.execute("SELECT pg_is_in_recovery();")
                    in_recovery = cur.fetchone()[0]
                    
                    if in_recovery:
                        # Check replication lag
                        cur.execute("""
                            SELECT pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn())
                        """)
                        lag = cur.fetchone()[0]
                        
                        if lag < min_lag:
                            min_lag = lag
                            best_standby = standby_name
                            
            except Exception as e:
                self.logger.error(f"Error checking standby {standby_name}: {e}")
                continue
        
        return best_standby
    
    def send_failover_notification(self, new_primary):
        """Send failover notification."""
        message = f"""
PostgreSQL Failover Alert

Time: {datetime.now().isoformat()}
Action: Automated failover executed
New Primary: {new_primary}
Previous Primary: {self.config['primary']['host']}

Please verify the new primary is functioning correctly.
"""
        
        # Send email notification
        # Implementation depends on your notification system
        self.logger.info(f"Failover notification sent: {message}")

# Usage
if __name__ == "__main__":
    config = {
        'primary': {
            'host': '192.168.1.10',
            'database': 'production',
            'user': 'monitor_user',
            'password': 'monitor_password'
        },
        'standbys': {
            'standby-1': {
                'host': '192.168.1.11',
                'database': 'production',
                'user': 'monitor_user',
                'password': 'monitor_password'
            },
            'standby-2': {
                'host': '192.168.1.12',
                'database': 'production',
                'user': 'monitor_user',
                'password': 'monitor_password'
            }
        }
    }
    
    failover_manager = PostgreSQLFailoverManager(config)
    
    # Check primary health every 30 seconds
    while True:
        if not failover_manager.check_primary_health():
            failover_manager.execute_failover()
        time.sleep(30)
```

## Disaster Recovery

### Backup and Restore Procedures

```bash
#!/bin/bash
# scripts/disaster_recovery.sh

# Configuration
BACKUP_DIR="/var/backups/postgresql"
WAL_ARCHIVE_DIR="/var/backups/postgresql/wal_archive"
NEW_PRIMARY_HOST="new-primary.example.com"
RECOVERY_TIME="$1"  # Format: '2024-01-15 14:30:00'

if [ -z "$RECOVERY_TIME" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS'"
    exit 1
fi

echo "Starting disaster recovery process..."

# 1. Stop all PostgreSQL instances
systemctl stop postgresql

# 2. Create new primary from backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR/base_backup_* | head -1)
if [ -z "$LATEST_BACKUP" ]; then
    echo "No base backup found!"
    exit 1
fi

echo "Using backup: $LATEST_BACKUP"
tar -xzf $LATEST_BACKUP -C /var/lib/postgresql/16/main

# 3. Configure for recovery
cat > /var/lib/postgresql/16/main/recovery.conf << EOF
restore_command = 'cp $WAL_ARCHIVE_DIR/%f %p'
recovery_target_time = '$RECOVERY_TIME'
recovery_target_action = 'promote'
EOF

# 4. Start PostgreSQL in recovery mode
pg_ctl -D /var/lib/postgresql/16/main start

# 5. Wait for recovery to complete
echo "Waiting for recovery to complete..."
while [ -f /var/lib/postgresql/16/main/recovery.conf ]; do
    sleep 5
done

echo "Recovery completed. New primary is ready."

# 6. Update configuration for normal operation
cat > /var/lib/postgresql/16/main/postgresql.conf << EOF
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
hot_standby = on
synchronous_commit = on
EOF

# 7. Restart PostgreSQL
systemctl restart postgresql

echo "Disaster recovery completed successfully."
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Setup streaming replication
pg_basebackup -h primary.example.com -U replicator -D /var/lib/postgresql/16/main --wal-method=stream

# 2. Configure standby
echo "standby_mode = 'on'" > /var/lib/postgresql/16/main/recovery.conf
echo "primary_conninfo = 'host=primary.example.com port=5432 user=replicator password=replication_password'" >> /var/lib/postgresql/16/main/recovery.conf

# 3. Start standby
systemctl start postgresql

# 4. Check replication status
psql -c "SELECT * FROM pg_stat_replication;"
```

### Essential Patterns

```python
# Complete PostgreSQL replication and HA setup
def setup_postgresql_replication_ha():
    # 1. Streaming replication
    # 2. Logical replication
    # 3. High availability with Patroni
    # 4. Load balancing
    # 5. Monitoring and health checks
    # 6. Automated failover
    # 7. Disaster recovery
    # 8. Testing procedures
    
    print("PostgreSQL replication and HA setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL replication and high availability excellence. Each pattern includes implementation examples, HA strategies, and real-world usage patterns for enterprise PostgreSQL HA systems.*
