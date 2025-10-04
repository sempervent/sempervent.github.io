# PostgreSQL Security Best Practices

**Objective**: Master senior-level PostgreSQL security patterns for production systems. When you need to implement robust security measures, when you want to protect sensitive data, when you need enterprise-grade security strategies—these best practices become your weapon of choice.

## Core Principles

- **Defense in Depth**: Multiple layers of security
- **Least Privilege**: Minimal access and permissions
- **Encryption**: Data at rest and in transit
- **Audit Trail**: Comprehensive logging and monitoring
- **Regular Updates**: Keep PostgreSQL and extensions current

## Authentication & Authorization

### User Management

```sql
-- Create application users with minimal privileges
CREATE USER app_user WITH PASSWORD 'strong_password_here';
CREATE USER readonly_user WITH PASSWORD 'readonly_password_here';
CREATE USER backup_user WITH PASSWORD 'backup_password_here';

-- Grant specific privileges
GRANT CONNECT ON DATABASE production TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Read-only user
GRANT CONNECT ON DATABASE production TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- Backup user
GRANT CONNECT ON DATABASE production TO backup_user;
GRANT USAGE ON SCHEMA public TO backup_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO backup_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
```

### Role-Based Access Control

```sql
-- Create roles for different access levels
CREATE ROLE app_developer;
CREATE ROLE app_readonly;
CREATE ROLE app_admin;
CREATE ROLE db_maintenance;

-- Grant role privileges
GRANT CONNECT ON DATABASE production TO app_developer;
GRANT USAGE ON SCHEMA public TO app_developer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_developer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_developer;

GRANT CONNECT ON DATABASE production TO app_readonly;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;

GRANT ALL PRIVILEGES ON DATABASE production TO app_admin;
GRANT ALL PRIVILEGES ON SCHEMA public TO app_admin;

-- Grant maintenance privileges
GRANT CONNECT ON DATABASE production TO db_maintenance;
GRANT USAGE ON SCHEMA public TO db_maintenance;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO db_maintenance;

-- Assign users to roles
GRANT app_developer TO app_user;
GRANT app_readonly TO readonly_user;
GRANT app_admin TO admin_user;
GRANT db_maintenance TO backup_user;
```

### Row-Level Security

```sql
-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial_data ENABLE ROW LEVEL SECURITY;

-- Create policies for user data access
CREATE POLICY user_own_data ON users
    FOR ALL TO app_user
    USING (id = current_setting('app.current_user_id')::integer);

CREATE POLICY user_own_posts ON posts
    FOR ALL TO app_user
    USING (user_id = current_setting('app.current_user_id')::integer);

-- Policy for financial data (admin only)
CREATE POLICY admin_financial_access ON financial_data
    FOR ALL TO app_admin
    USING (true);

-- Policy for read-only access
CREATE POLICY readonly_access ON posts
    FOR SELECT TO app_readonly
    USING (published = true);
```

## Network Security

### Connection Security

```bash
# postgresql.conf network security settings
listen_addresses = 'localhost'  # Only listen on localhost for development
# listen_addresses = '10.0.0.0/8'  # Specific network for production

# SSL configuration
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.2'

# Connection limits
max_connections = 100
superuser_reserved_connections = 3

# Connection timeouts
tcp_keepalives_idle = 600
tcp_keepalives_interval = 30
tcp_keepalives_count = 3
```

### pg_hba.conf Configuration

```bash
# pg_hba.conf - Host-based authentication
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     md5

# IPv4 local connections
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5

# Application connections (specific IP ranges)
host    production      app_user        10.0.0.0/8              md5
host    production      readonly_user   10.0.0.0/8              md5

# Admin connections (restricted)
host    all             admin_user      10.0.1.0/24            md5

# SSL connections only
hostssl production      app_user        0.0.0.0/0               md5
hostssl production      readonly_user   0.0.0.0/0               md5

# Reject all other connections
host    all             all             0.0.0.0/0               reject
```

## Data Encryption

### Transparent Data Encryption

```sql
-- Create encrypted tablespace
CREATE TABLESPACE encrypted_tablespace
LOCATION '/var/lib/postgresql/encrypted_data'
WITH (encryption = 'on');

-- Create table in encrypted tablespace
CREATE TABLE sensitive_data (
    id SERIAL PRIMARY KEY,
    credit_card_number TEXT,
    ssn TEXT,
    personal_info JSONB
) TABLESPACE encrypted_tablespace;

-- Create encrypted columns
CREATE TABLE users_encrypted (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    password_hash TEXT,
    encrypted_notes TEXT
);

-- Encrypt sensitive data
UPDATE users_encrypted 
SET encrypted_notes = pgp_sym_encrypt(notes, 'encryption_key_here')
WHERE notes IS NOT NULL;
```

### Application-Level Encryption

```python
# encryption/field_encryption.py
import psycopg2
from cryptography.fernet import Fernet
import base64
import os

class PostgreSQLFieldEncryption:
    def __init__(self, encryption_key=None):
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            print(f"Generated encryption key: {key.decode()}")
    
    def encrypt_field(self, value):
        """Encrypt a field value."""
        if value is None:
            return None
        encrypted = self.cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_field(self, encrypted_value):
        """Decrypt a field value."""
        if encrypted_value is None:
            return None
        try:
            encrypted = base64.b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            print(f"Decryption error: {e}")
            return None
    
    def create_encrypted_table(self, connection):
        """Create table with encrypted fields."""
        with connection.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50),
                    email VARCHAR(100),
                    encrypted_ssn TEXT,
                    encrypted_credit_card TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            connection.commit()
    
    def insert_encrypted_data(self, connection, username, email, ssn, credit_card):
        """Insert data with encrypted fields."""
        encrypted_ssn = self.encrypt_field(ssn)
        encrypted_credit_card = self.encrypt_field(credit_card)
        
        with connection.cursor() as cur:
            cur.execute("""
                INSERT INTO encrypted_users (username, email, encrypted_ssn, encrypted_credit_card)
                VALUES (%s, %s, %s, %s)
            """, (username, email, encrypted_ssn, encrypted_credit_card))
            connection.commit()
    
    def retrieve_encrypted_data(self, connection, user_id):
        """Retrieve and decrypt data."""
        with connection.cursor() as cur:
            cur.execute("""
                SELECT id, username, email, encrypted_ssn, encrypted_credit_card
                FROM encrypted_users WHERE id = %s
            """, (user_id,))
            
            row = cur.fetchone()
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'ssn': self.decrypt_field(row[3]),
                    'credit_card': self.decrypt_field(row[4])
                }
            return None

# Usage
encryption = PostgreSQLFieldEncryption()
conn = psycopg2.connect(
    host='localhost',
    database='production',
    user='app_user',
    password='app_password'
)

encryption.create_encrypted_table(conn)
encryption.insert_encrypted_data(conn, 'user1', 'user1@example.com', '123-45-6789', '4111-1111-1111-1111')
user_data = encryption.retrieve_encrypted_data(conn, 1)
print(user_data)
```

## Audit Logging

### Comprehensive Audit Setup

```sql
-- Create audit schema
CREATE SCHEMA IF NOT EXISTS audit;

-- Create audit log table
CREATE TABLE audit.audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_values JSONB,
    new_values JSONB,
    changed_by TEXT NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    client_ip INET,
    application_name TEXT
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit.audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, new_values, changed_by, client_ip, application_name
        ) VALUES (
            TG_TABLE_NAME, TG_OP, to_jsonb(NEW), current_user, 
            inet_client_addr(), current_setting('application_name', true)
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, old_values, new_values, changed_by, client_ip, application_name
        ) VALUES (
            TG_TABLE_NAME, TG_OP, to_jsonb(OLD), to_jsonb(NEW), current_user,
            inet_client_addr(), current_setting('application_name', true)
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, old_values, changed_by, client_ip, application_name
        ) VALUES (
            TG_TABLE_NAME, TG_OP, to_jsonb(OLD), current_user,
            inet_client_addr(), current_setting('application_name', true)
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for sensitive tables
CREATE TRIGGER users_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit.audit_trigger_function();

CREATE TRIGGER posts_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON posts
    FOR EACH ROW EXECUTE FUNCTION audit.audit_trigger_function();
```

### Security Event Monitoring

```python
# monitoring/security_monitor.py
import psycopg2
import json
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

class PostgreSQLSecurityMonitor:
    def __init__(self, connection_params, alert_config):
        self.conn_params = connection_params
        self.alert_config = alert_config
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(**self.conn_params)
    
    def check_failed_logins(self, hours=1):
        """Check for failed login attempts."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    usename,
                    client_addr,
                    application_name,
                    count(*) as failed_attempts
                FROM pg_stat_activity 
                WHERE state = 'idle in transaction (aborted)'
                AND query_start > NOW() - INTERVAL '%s hours'
                GROUP BY usename, client_addr, application_name
                HAVING count(*) > 5
                ORDER BY failed_attempts DESC;
            """, (hours,))
            return cur.fetchall()
    
    def check_privilege_escalation(self):
        """Check for privilege escalation attempts."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    usename,
                    client_addr,
                    application_name,
                    query
                FROM pg_stat_activity 
                WHERE query ILIKE '%GRANT%' 
                OR query ILIKE '%REVOKE%'
                OR query ILIKE '%ALTER USER%'
                OR query ILIKE '%CREATE USER%'
                ORDER BY query_start DESC;
            """)
            return cur.fetchall()
    
    def check_suspicious_queries(self):
        """Check for suspicious query patterns."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    usename,
                    client_addr,
                    application_name,
                    query,
                    query_start
                FROM pg_stat_activity 
                WHERE query ILIKE '%DROP%'
                OR query ILIKE '%TRUNCATE%'
                OR query ILIKE '%DELETE FROM%'
                OR query ILIKE '%UPDATE%'
                ORDER BY query_start DESC;
            """)
            return cur.fetchall()
    
    def check_audit_log_anomalies(self, hours=24):
        """Check audit log for anomalies."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    changed_by,
                    client_ip,
                    application_name,
                    count(*) as operations
                FROM audit.audit_log 
                WHERE changed_at > NOW() - INTERVAL '%s hours'
                GROUP BY changed_by, client_ip, application_name
                HAVING count(*) > 100
                ORDER BY operations DESC;
            """, (hours,))
            return cur.fetchall()
    
    def send_security_alert(self, subject, message):
        """Send security alert via email."""
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.alert_config['from_email']
            msg['To'] = self.alert_config['to_email']
            
            server = smtplib.SMTP(self.alert_config['smtp_server'], self.alert_config['smtp_port'])
            server.starttls()
            server.login(self.alert_config['username'], self.alert_config['password'])
            server.send_message(msg)
            server.quit()
            print(f"Security alert sent: {subject}")
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def run_security_checks(self):
        """Run all security checks."""
        alerts = []
        
        # Check failed logins
        failed_logins = self.check_failed_logins()
        if failed_logins:
            alerts.append({
                'type': 'failed_logins',
                'message': f"Multiple failed login attempts detected: {failed_logins}",
                'severity': 'high'
            })
        
        # Check privilege escalation
        privilege_attempts = self.check_privilege_escalation()
        if privilege_attempts:
            alerts.append({
                'type': 'privilege_escalation',
                'message': f"Privilege escalation attempts detected: {privilege_attempts}",
                'severity': 'critical'
            })
        
        # Check suspicious queries
        suspicious_queries = self.check_suspicious_queries()
        if suspicious_queries:
            alerts.append({
                'type': 'suspicious_queries',
                'message': f"Suspicious queries detected: {suspicious_queries}",
                'severity': 'medium'
            })
        
        # Send alerts
        for alert in alerts:
            self.send_security_alert(
                f"PostgreSQL Security Alert: {alert['type']}",
                alert['message']
            )
        
        return alerts

# Usage
if __name__ == "__main__":
    monitor = PostgreSQLSecurityMonitor(
        connection_params={
            'host': 'localhost',
            'database': 'production',
            'user': 'security_monitor',
            'password': 'monitor_password'
        },
        alert_config={
            'from_email': 'alerts@company.com',
            'to_email': 'security@company.com',
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'username': 'alerts@company.com',
            'password': 'smtp_password'
        }
    )
    
    monitor.connect()
    alerts = monitor.run_security_checks()
    print(f"Security check completed. {len(alerts)} alerts generated.")
```

## Data Masking and Anonymization

### Data Masking Functions

```sql
-- Create data masking functions
CREATE OR REPLACE FUNCTION mask_email(email TEXT)
RETURNS TEXT AS $$
BEGIN
    IF email IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN regexp_replace(email, '^(.{1,3}).*@(.*)$', '\1***@\2');
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mask_ssn(ssn TEXT)
RETURNS TEXT AS $$
BEGIN
    IF ssn IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN regexp_replace(ssn, '^(\d{3})-(\d{2})-(\d{4})$', '\1-**-****');
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mask_credit_card(cc TEXT)
RETURNS TEXT AS $$
BEGIN
    IF cc IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN regexp_replace(cc, '^(\d{4})-(\d{4})-(\d{4})-(\d{4})$', '\1-****-****-\4');
END;
$$ LANGUAGE plpgsql;

-- Create masked views for non-production environments
CREATE VIEW users_masked AS
SELECT 
    id,
    username,
    mask_email(email) as email,
    mask_ssn(ssn) as ssn,
    mask_credit_card(credit_card) as credit_card,
    created_at
FROM users;
```

### Data Anonymization

```python
# anonymization/data_anonymizer.py
import psycopg2
import random
import string
from faker import Faker

class PostgreSQLDataAnonymizer:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.fake = Faker()
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(**self.conn_params)
    
    def anonymize_users(self):
        """Anonymize user data."""
        with self.conn.cursor() as cur:
            # Get all users
            cur.execute("SELECT id, username, email FROM users")
            users = cur.fetchall()
            
            for user_id, username, email in users:
                # Generate fake data
                fake_username = self.fake.user_name()
                fake_email = self.fake.email()
                
                # Update user data
                cur.execute("""
                    UPDATE users 
                    SET username = %s, email = %s
                    WHERE id = %s
                """, (fake_username, fake_email, user_id))
            
            self.conn.commit()
            print(f"Anonymized {len(users)} users")
    
    def anonymize_posts(self):
        """Anonymize post content."""
        with self.conn.cursor() as cur:
            # Get all posts
            cur.execute("SELECT id, title, content FROM posts")
            posts = cur.fetchall()
            
            for post_id, title, content in posts:
                # Generate fake content
                fake_title = self.fake.sentence(nb_words=6)
                fake_content = self.fake.text(max_nb_chars=500)
                
                # Update post data
                cur.execute("""
                    UPDATE posts 
                    SET title = %s, content = %s
                    WHERE id = %s
                """, (fake_title, fake_content, post_id))
            
            self.conn.commit()
            print(f"Anonymized {len(posts)} posts")
    
    def create_anonymized_copy(self, source_db, target_db):
        """Create anonymized copy of database."""
        with self.conn.cursor() as cur:
            # Create target database
            cur.execute(f"CREATE DATABASE {target_db}")
            self.conn.commit()
        
        # Connect to target database
        target_conn = psycopg2.connect(
            host=self.conn_params['host'],
            database=target_db,
            user=self.conn_params['user'],
            password=self.conn_params['password']
        )
        
        with target_conn.cursor() as cur:
            # Copy schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS public")
            
            # Copy tables structure
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            
            for table_name, in tables:
                # Create table in target database
                cur.execute(f"CREATE TABLE {table_name} (LIKE {source_db}.{table_name})")
            
            target_conn.commit()
        
        target_conn.close()
        print(f"Created anonymized copy: {target_db}")

# Usage
if __name__ == "__main__":
    anonymizer = PostgreSQLDataAnonymizer({
        'host': 'localhost',
        'database': 'production',
        'user': 'anonymizer',
        'password': 'anonymizer_password'
    })
    
    anonymizer.connect()
    anonymizer.anonymize_users()
    anonymizer.anonymize_posts()
    anonymizer.create_anonymized_copy('production', 'production_anonymized')
```

## TL;DR Runbook

### Quick Start

```sql
-- 1. Create secure users
CREATE USER app_user WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE production TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;

-- 2. Enable RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_own_data ON users FOR ALL TO app_user USING (id = current_setting('app.current_user_id')::integer);

-- 3. Enable audit logging
CREATE SCHEMA audit;
CREATE TABLE audit.audit_log (id BIGSERIAL PRIMARY KEY, table_name TEXT, operation TEXT, old_values JSONB, new_values JSONB, changed_by TEXT, changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
```

### Essential Patterns

```python
# Complete PostgreSQL security setup
def setup_postgresql_security():
    # 1. Authentication & authorization
    # 2. Network security
    # 3. Data encryption
    # 4. Audit logging
    # 5. Data masking
    # 6. Security monitoring
    # 7. Compliance
    # 8. Incident response
    
    print("PostgreSQL security setup complete!")
```

---

*This guide provides the complete machinery for PostgreSQL security excellence. Each pattern includes implementation examples, security strategies, and real-world usage patterns for enterprise PostgreSQL security systems.*
