# PostgreSQL Backup & Recovery Best Practices

**Objective**: Master senior-level PostgreSQL backup and recovery strategies for production systems. When you need to implement robust backup solutions, when you want to ensure data protection, when you need enterprise-grade recovery strategies—these best practices become your weapon of choice.

## Core Principles

- **3-2-1 Rule**: 3 copies, 2 different media, 1 offsite
- **Test Regularly**: Verify backups with restore tests
- **Automate Everything**: Consistent, reliable backup processes
- **Monitor Continuously**: Backup success/failure monitoring
- **Document Procedures**: Clear recovery runbooks

## Backup Strategies

### Full Database Backups

```bash
#!/bin/bash
# scripts/full_backup.sh

# Configuration
DB_NAME="production"
DB_USER="backup_user"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="/var/backups/postgresql"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Full database backup
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
    --verbose \
    --no-password \
    --format=custom \
    --compress=9 \
    --file="$BACKUP_DIR/${DB_NAME}_full_${DATE}.dump"

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: ${DB_NAME}_full_${DATE}.dump"
    
    # Compress backup
    gzip "$BACKUP_DIR/${DB_NAME}_full_${DATE}.dump"
    
    # Clean old backups
    find $BACKUP_DIR -name "${DB_NAME}_full_*.dump.gz" -mtime +$RETENTION_DAYS -delete
    
    # Log backup completion
    echo "$(date): Full backup completed - ${DB_NAME}_full_${DATE}.dump.gz" >> /var/log/postgresql/backup.log
else
    echo "Backup failed!" >&2
    exit 1
fi
```

### Incremental Backups with WAL Archiving

```bash
# postgresql.conf WAL archiving configuration
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/backups/postgresql/wal_archive/%f'
archive_timeout = 300
max_wal_senders = 3
max_replication_slots = 3

# Create WAL archive directory
mkdir -p /var/backups/postgresql/wal_archive
chown postgres:postgres /var/backups/postgresql/wal_archive
chmod 700 /var/backups/postgresql/wal_archive
```

### Continuous Archiving Setup

```bash
#!/bin/bash
# scripts/setup_continuous_archiving.sh

# Create base backup
pg_basebackup -h localhost -U backup_user -D /var/backups/postgresql/base_backup_$(date +%Y%m%d) \
    --format=tar \
    --gzip \
    --progress \
    --verbose

# Create recovery.conf for point-in-time recovery
cat > /var/backups/postgresql/recovery.conf << EOF
restore_command = 'cp /var/backups/postgresql/wal_archive/%f %p'
recovery_target_time = '2024-01-15 14:30:00'
recovery_target_action = 'promote'
EOF
```

## Automated Backup System

### Python Backup Manager

```python
# backup/backup_manager.py
import psycopg2
import subprocess
import os
import gzip
import shutil
from datetime import datetime, timedelta
import logging
from pathlib import Path

class PostgreSQLBackupManager:
    def __init__(self, config):
        self.config = config
        self.backup_dir = Path(config['backup_dir'])
        self.wal_archive_dir = Path(config['wal_archive_dir'])
        self.setup_logging()
        self.setup_directories()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/postgresql/backup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create backup directories."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.wal_archive_dir.mkdir(parents=True, exist_ok=True)
    
    def test_connection(self):
        """Test database connection."""
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def full_backup(self):
        """Perform full database backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"{self.config['database']}_full_{timestamp}.dump"
        
        try:
            cmd = [
                'pg_dump',
                '-h', self.config['host'],
                '-p', str(self.config['port']),
                '-U', self.config['user'],
                '-d', self.config['database'],
                '--verbose',
                '--no-password',
                '--format=custom',
                '--compress=9',
                '--file', str(backup_file)
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config['password']
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Compress backup
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(f"{backup_file}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                backup_file.unlink()
                
                self.logger.info(f"Full backup completed: {backup_file}.gz")
                return str(backup_file) + '.gz'
            else:
                self.logger.error(f"Backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Backup error: {e}")
            return None
    
    def base_backup(self):
        """Perform base backup for continuous archiving."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_backup_dir = self.backup_dir / f"base_backup_{timestamp}"
        
        try:
            cmd = [
                'pg_basebackup',
                '-h', self.config['host'],
                '-U', self.config['user'],
                '-D', str(base_backup_dir),
                '--format=tar',
                '--gzip',
                '--progress',
                '--verbose'
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config['password']
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Base backup completed: {base_backup_dir}")
                return str(base_backup_dir)
            else:
                self.logger.error(f"Base backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Base backup error: {e}")
            return None
    
    def cleanup_old_backups(self, retention_days):
        """Remove old backup files."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for backup_file in self.backup_dir.glob("*.dump.gz"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                self.logger.info(f"Deleted old backup: {backup_file}")
        
        for base_backup_dir in self.backup_dir.glob("base_backup_*"):
            if base_backup_dir.is_dir() and base_backup_dir.stat().st_mtime < cutoff_date.timestamp():
                shutil.rmtree(base_backup_dir)
                self.logger.info(f"Deleted old base backup: {base_backup_dir}")
    
    def verify_backup(self, backup_file):
        """Verify backup integrity."""
        try:
            cmd = ['pg_restore', '--list', backup_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Backup verification successful: {backup_file}")
                return True
            else:
                self.logger.error(f"Backup verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup verification error: {e}")
            return False
    
    def schedule_backups(self):
        """Schedule different types of backups."""
        current_hour = datetime.now().hour
        
        if current_hour == 2:  # 2 AM - Full backup
            self.logger.info("Starting scheduled full backup")
            backup_file = self.full_backup()
            if backup_file:
                self.verify_backup(backup_file)
                self.cleanup_old_backups(self.config['retention_days'])
        
        elif current_hour == 14:  # 2 PM - Base backup
            self.logger.info("Starting scheduled base backup")
            self.base_backup()
        
        else:  # Other hours - WAL archiving check
            self.logger.info("Checking WAL archiving status")
            self.check_wal_archiving()

# Usage
if __name__ == "__main__":
    config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'production',
        'user': 'backup_user',
        'password': 'backup_password',
        'backup_dir': '/var/backups/postgresql',
        'wal_archive_dir': '/var/backups/postgresql/wal_archive',
        'retention_days': 30
    }
    
    backup_manager = PostgreSQLBackupManager(config)
    backup_manager.schedule_backups()
```

## Point-in-Time Recovery

### PITR Setup

```bash
#!/bin/bash
# scripts/pitr_recovery.sh

# Configuration
BACKUP_DIR="/var/backups/postgresql"
WAL_ARCHIVE_DIR="/var/backups/postgresql/wal_archive"
RECOVERY_DIR="/var/lib/postgresql/16/recovery"
TARGET_TIME="$1"  # Format: '2024-01-15 14:30:00'

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS'"
    exit 1
fi

# Stop PostgreSQL
systemctl stop postgresql

# Create recovery directory
mkdir -p $RECOVERY_DIR

# Copy base backup
LATEST_BASE_BACKUP=$(ls -t $BACKUP_DIR/base_backup_* | head -1)
if [ -z "$LATEST_BASE_BACKUP" ]; then
    echo "No base backup found!"
    exit 1
fi

echo "Using base backup: $LATEST_BASE_BACKUP"
tar -xzf $LATEST_BASE_BACKUP -C $RECOVERY_DIR

# Create recovery.conf
cat > $RECOVERY_DIR/recovery.conf << EOF
restore_command = 'cp $WAL_ARCHIVE_DIR/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Update postgresql.conf for recovery
cat >> $RECOVERY_DIR/postgresql.conf << EOF
port = 5433
data_directory = '$RECOVERY_DIR'
EOF

# Start PostgreSQL in recovery mode
pg_ctl -D $RECOVERY_DIR start

echo "Recovery started. Check logs for progress."
echo "Recovery will complete at: $TARGET_TIME"
```

### Automated PITR Testing

```python
# recovery/pitr_tester.py
import psycopg2
import subprocess
import time
from datetime import datetime, timedelta
import logging

class PITRTester:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def test_pitr_recovery(self, target_time):
        """Test point-in-time recovery."""
        try:
            # Create test data
            self.create_test_data()
            
            # Perform recovery
            recovery_result = self.perform_recovery(target_time)
            
            if recovery_result:
                # Verify recovery
                verification_result = self.verify_recovery(target_time)
                return verification_result
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"PITR test failed: {e}")
            return False
    
    def create_test_data(self):
        """Create test data for recovery testing."""
        conn = psycopg2.connect(**self.config)
        
        with conn.cursor() as cur:
            # Create test table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recovery_test (
                    id SERIAL PRIMARY KEY,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Insert test data
            for i in range(10):
                cur.execute("""
                    INSERT INTO recovery_test (data) VALUES (%s)
                """, (f"test_data_{i}",))
            
            conn.commit()
        conn.close()
    
    def perform_recovery(self, target_time):
        """Perform point-in-time recovery."""
        try:
            # Stop PostgreSQL
            subprocess.run(['systemctl', 'stop', 'postgresql'], check=True)
            
            # Run recovery script
            cmd = ['./scripts/pitr_recovery.sh', target_time]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Recovery completed successfully")
                return True
            else:
                self.logger.error(f"Recovery failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery error: {e}")
            return False
    
    def verify_recovery(self, target_time):
        """Verify recovery results."""
        try:
            # Connect to recovered database
            conn = psycopg2.connect(**self.config)
            
            with conn.cursor() as cur:
                # Check if data exists at target time
                cur.execute("""
                    SELECT COUNT(*) FROM recovery_test 
                    WHERE created_at <= %s
                """, (target_time,))
                
                count = cur.fetchone()[0]
                
                if count > 0:
                    self.logger.info(f"Recovery verification successful: {count} records found")
                    return True
                else:
                    self.logger.error("Recovery verification failed: No data found")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Recovery verification error: {e}")
            return False

# Usage
if __name__ == "__main__":
    config = {
        'host': 'localhost',
        'database': 'production',
        'user': 'test_user',
        'password': 'test_password'
    }
    
    tester = PITRTester(config)
    target_time = '2024-01-15 14:30:00'
    result = tester.test_pitr_recovery(target_time)
    print(f"PITR test result: {result}")
```

## Cloud Backup Integration

### AWS S3 Backup

```python
# backup/s3_backup.py
import boto3
import os
from pathlib import Path
from datetime import datetime
import logging

class S3BackupManager:
    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def upload_backup(self, backup_file, s3_key):
        """Upload backup to S3."""
        try:
            self.s3_client.upload_file(
                backup_file,
                self.config['bucket'],
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA'
                }
            )
            self.logger.info(f"Backup uploaded to S3: {s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_backup(self, s3_key, local_file):
        """Download backup from S3."""
        try:
            self.s3_client.download_file(
                self.config['bucket'],
                s3_key,
                local_file
            )
            self.logger.info(f"Backup downloaded from S3: {s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"S3 download failed: {e}")
            return False
    
    def list_backups(self, prefix=""):
        """List available backups in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config['bucket'],
                Prefix=prefix
            )
            
            backups = []
            for obj in response.get('Contents', []):
                backups.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
            
            return sorted(backups, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"S3 list failed: {e}")
            return []
    
    def cleanup_old_backups(self, retention_days):
        """Remove old backups from S3."""
        try:
            backups = self.list_backups()
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for backup in backups:
                if backup['last_modified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.config['bucket'],
                        Key=backup['key']
                    )
                    self.logger.info(f"Deleted old backup: {backup['key']}")
            
        except Exception as e:
            self.logger.error(f"S3 cleanup failed: {e}")

# Usage
if __name__ == "__main__":
    config = {
        'bucket': 'postgresql-backups',
        'region': 'us-west-2'
    }
    
    s3_manager = S3BackupManager(config)
    
    # Upload backup
    backup_file = '/var/backups/postgresql/production_full_20240115.dump.gz'
    s3_key = f"backups/{datetime.now().strftime('%Y/%m/%d')}/production_full.dump.gz"
    s3_manager.upload_backup(backup_file, s3_key)
    
    # List backups
    backups = s3_manager.list_backups()
    for backup in backups:
        print(f"{backup['key']} - {backup['last_modified']}")
```

## Monitoring and Alerting

### Backup Monitoring

```python
# monitoring/backup_monitor.py
import psycopg2
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import logging

class BackupMonitor:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_backup_status(self):
        """Check backup status and send alerts if needed."""
        try:
            # Check if backup completed today
            backup_status = self.get_backup_status()
            
            if not backup_status['completed']:
                self.send_alert("Backup Failed", "Daily backup did not complete")
                return False
            
            # Check backup size
            if backup_status['size'] < self.config['min_backup_size']:
                self.send_alert("Backup Size Warning", 
                              f"Backup size {backup_status['size']} is below minimum")
                return False
            
            # Check backup age
            if backup_status['age_hours'] > 24:
                self.send_alert("Backup Age Warning", 
                              f"Backup is {backup_status['age_hours']} hours old")
                return False
            
            self.logger.info("Backup status check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup status check failed: {e}")
            return False
    
    def get_backup_status(self):
        """Get current backup status."""
        backup_dir = Path(self.config['backup_dir'])
        today = datetime.now().date()
        
        # Find today's backup
        backup_files = list(backup_dir.glob(f"*_full_{today.strftime('%Y%m%d')}*.dump.gz"))
        
        if not backup_files:
            return {
                'completed': False,
                'size': 0,
                'age_hours': 24
            }
        
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
        
        return {
            'completed': True,
            'size': latest_backup.stat().st_size,
            'age_hours': (datetime.now() - datetime.fromtimestamp(latest_backup.stat().st_mtime)).total_seconds() / 3600
        }
    
    def send_alert(self, subject, message):
        """Send alert email."""
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['username'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

# Usage
if __name__ == "__main__":
    config = {
        'backup_dir': '/var/backups/postgresql',
        'min_backup_size': 100 * 1024 * 1024,  # 100MB
        'from_email': 'alerts@company.com',
        'to_email': 'dba@company.com',
        'smtp_server': 'smtp.company.com',
        'smtp_port': 587,
        'username': 'alerts@company.com',
        'password': 'smtp_password'
    }
    
    monitor = BackupMonitor(config)
    monitor.check_backup_status()
```

## TL;DR Runbook

### Quick Start

```bash
# 1. Full backup
pg_dump -h localhost -U backup_user -d production --format=custom --compress=9 --file=production_full.dump

# 2. Base backup for PITR
pg_basebackup -h localhost -U backup_user -D /var/backups/base_backup --format=tar --gzip

# 3. Restore from backup
pg_restore -h localhost -U restore_user -d restored_db --clean --if-exists production_full.dump
```

### Essential Patterns

```python
# Complete PostgreSQL backup and recovery setup
def setup_postgresql_backup_recovery():
    # 1. Full database backups
    # 2. Incremental backups
    # 3. WAL archiving
    # 4. Point-in-time recovery
    # 5. Cloud integration
    # 6. Monitoring and alerting
    # 7. Testing procedures
    # 8. Documentation
    
    print("PostgreSQL backup and recovery setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL backup and recovery excellence. Each pattern includes implementation examples, recovery strategies, and real-world usage patterns for enterprise PostgreSQL backup systems.*
