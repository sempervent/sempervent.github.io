# Ansible Security Hardening Best Practices

**Objective**: Master Ansible security patterns for enterprise-grade automation. When you need to secure automation workflows, when you want to protect sensitive data and credentials, when you're building compliant infrastructure—security hardening becomes your weapon of choice.

Ansible security is critical for enterprise automation. Proper security practices prevent credential exposure, ensure compliance, and protect sensitive infrastructure. This guide shows you how to wield security hardening with the precision of a security engineer.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand security threats**
   - Credential exposure and data breaches
   - Privilege escalation and unauthorized access
   - Network security and communication encryption
   - Compliance and audit requirements

2. **Master secret management**
   - Ansible Vault encryption and key management
   - External secret management integration
   - Credential rotation and lifecycle management
   - Access control and audit logging

3. **Know your authentication patterns**
   - SSH key management and rotation
   - Certificate-based authentication
   - Multi-factor authentication integration
   - Service account management

4. **Validate everything**
   - Test security configurations and access controls
   - Verify encryption and key management
   - Check compliance and audit requirements
   - Validate network security and communication

5. **Plan for production**
   - Security monitoring and alerting
   - Incident response and forensics
   - Compliance reporting and auditing
   - Security training and awareness

**Why These Principles**: Security hardening requires understanding both Ansible mechanics and security best practices. Understanding these patterns prevents security breaches and enables compliant automation.

## 1) Credential Security

### Ansible Vault Best Practices

```bash
# Create encrypted files
ansible-vault create group_vars/all/secrets.yml
ansible-vault create host_vars/web-prod-01/secrets.yml

# Use strong passwords
ansible-vault create --vault-password-file .vault_pass group_vars/all/secrets.yml

# Rotate vault passwords
ansible-vault rekey group_vars/all/secrets.yml
```

### Vault Organization

```yaml
# group_vars/all/secrets.yml (encrypted)
# Database credentials
database:
  root_password: "{{ vault_db_root_password }}"
  user_password: "{{ vault_db_user_password }}"
  replication_password: "{{ vault_db_replication_password }}"

# SSL certificates
ssl:
  private_key: "{{ vault_ssl_private_key }}"
  certificate: "{{ vault_ssl_certificate }}"
  ca_certificate: "{{ vault_ssl_ca_certificate }}"

# API keys
api_keys:
  monitoring: "{{ vault_monitoring_api_key }}"
  backup: "{{ vault_backup_api_key }}"
  notification: "{{ vault_notification_api_key }}"

# Service accounts
service_accounts:
  ansible_user: "{{ vault_ansible_user }}"
  ansible_password: "{{ vault_ansible_password }}"
```

### External Secret Management

```yaml
# HashiCorp Vault integration
vault:
  url: "https://vault.example.com"
  auth_method: "token"
  token: "{{ vault_token }}"
  secrets_path: "secret/ansible"

# AWS Secrets Manager
aws_secrets:
  region: "us-west-2"
  secrets:
    database_password: "prod/database/password"
    api_key: "prod/api/key"

# Azure Key Vault
azure_keyvault:
  vault_url: "https://vault.vault.azure.net"
  client_id: "{{ azure_client_id }}"
  client_secret: "{{ azure_client_secret }}"
```

**Why Credential Security Matters**: Proper credential management prevents unauthorized access and data breaches. Understanding these patterns prevents security incidents and enables compliant automation.

## 2) Network Security

### SSH Security Configuration

```yaml
# SSH security settings
ssh_security:
  # Disable password authentication
  password_authentication: "no"
  
  # Use key-based authentication
  pubkey_authentication: "yes"
  
  # Disable root login
  permit_root_login: "no"
  
  # Use specific users
  allowed_users: "ansible,admin"
  
  # Disable X11 forwarding
  x11_forwarding: "no"
  
  # Use strong ciphers
  ciphers: "chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com"
  
  # Use strong MACs
  macs: "hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com"
  
  # Use strong key exchange
  kex_algorithms: "curve25519-sha256,curve25519-sha256@libssh.org"
```

### SSL/TLS Configuration

```yaml
# SSL/TLS security settings
ssl_security:
  # Use strong protocols
  ssl_protocols: "TLSv1.2 TLSv1.3"
  
  # Use strong ciphers
  ssl_ciphers: "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256"
  
  # Use strong curves
  ssl_curves: "X25519:P-256:P-384:P-521"
  
  # Enable HSTS
  hsts_enabled: true
  hsts_max_age: 31536000
  
  # Enable OCSP stapling
  ocsp_stapling: true
```

### Firewall Configuration

```yaml
# Firewall security settings
firewall_security:
  # Default policy
  default_policy: "DROP"
  
  # Allow SSH
  ssh_port: 22
  ssh_source: "10.0.0.0/8"
  
  # Allow HTTP/HTTPS
  http_port: 80
  https_port: 443
  
  # Allow specific services
  allowed_services:
    - port: 22
      protocol: "tcp"
      source: "10.0.0.0/8"
    - port: 80
      protocol: "tcp"
      source: "0.0.0.0/0"
    - port: 443
      protocol: "tcp"
      source: "0.0.0.0/0"
```

**Why Network Security Matters**: Proper network security prevents unauthorized access and protects communication. Understanding these patterns prevents network attacks and enables secure automation.

## 3) Access Control

### User Management

```yaml
# User security settings
user_security:
  # Create ansible user
  ansible_user:
    name: "ansible"
    shell: "/bin/bash"
    home: "/home/ansible"
    create_home: true
    groups: ["sudo", "ansible"]
    ssh_key: "{{ ansible_ssh_public_key }}"
  
  # Disable root login
  root_login: false
  
  # Remove default users
  remove_users:
    - "ubuntu"
    - "ec2-user"
    - "centos"
  
  # Set password policies
  password_policies:
    min_length: 12
    complexity: true
    history: 5
    max_age: 90
```

### Sudo Configuration

```yaml
# Sudo security settings
sudo_security:
  # Require password for sudo
  require_password: true
  
  # Use timestamp timeout
  timestamp_timeout: 5
  
  # Log sudo commands
  log_commands: true
  
  # Restrict sudo access
  allowed_commands:
    - "/usr/bin/systemctl"
    - "/usr/bin/docker"
    - "/usr/bin/kubectl"
  
  # Disable dangerous commands
  forbidden_commands:
    - "/bin/su"
    - "/usr/bin/sudo su"
```

### File Permissions

```yaml
# File permission security
file_security:
  # Secure sensitive files
  sensitive_files:
    - path: "/etc/shadow"
      mode: "0640"
      owner: "root"
      group: "shadow"
    - path: "/etc/passwd"
      mode: "0644"
      owner: "root"
      group: "root"
    - path: "/etc/ssh/sshd_config"
      mode: "0600"
      owner: "root"
      group: "root"
  
  # Secure Ansible files
  ansible_files:
    - path: "{{ ansible_config_dir }}"
      mode: "0755"
      owner: "ansible"
      group: "ansible"
    - path: "{{ ansible_playbook_dir }}"
      mode: "0755"
      owner: "ansible"
      group: "ansible"
```

**Why Access Control Matters**: Proper access control prevents unauthorized access and privilege escalation. Understanding these patterns prevents security breaches and enables compliant automation.

## 4) Audit and Compliance

### Audit Logging

```yaml
# Audit logging configuration
audit_logging:
  # Enable auditd
  auditd_enabled: true
  
  # Audit rules
  audit_rules:
    - "-w /etc/passwd -p wa -k identity"
    - "-w /etc/group -p wa -k identity"
    - "-w /etc/shadow -p wa -k identity"
    - "-w /etc/sudoers -p wa -k privilege"
    - "-w /var/log/auth.log -p wa -k authentication"
    - "-w /etc/ssh/sshd_config -p wa -k ssh_config"
  
  # Log rotation
  log_rotation:
    max_size: "100M"
    max_files: 5
    compress: true
```

### Compliance Monitoring

```yaml
# Compliance monitoring
compliance_monitoring:
  # CIS benchmarks
  cis_benchmarks:
    enabled: true
    level: 2
  
  # NIST guidelines
  nist_guidelines:
    enabled: true
    framework: "NIST CSF"
  
  # PCI DSS
  pci_dss:
    enabled: false
    level: 1
  
  # SOX compliance
  sox_compliance:
    enabled: false
    controls: ["access_control", "change_management"]
```

### Security Scanning

```yaml
# Security scanning
security_scanning:
  # Vulnerability scanning
  vulnerability_scanning:
    enabled: true
    tools: ["nessus", "openvas", "trivy"]
    schedule: "0 2 * * *"
  
  # Configuration scanning
  configuration_scanning:
    enabled: true
    tools: ["ansible-lint", "ansible-security"]
    schedule: "0 1 * * *"
  
  # Compliance scanning
  compliance_scanning:
    enabled: true
    tools: ["inspec", "oscap"]
    schedule: "0 3 * * 0"
```

**Why Audit and Compliance Matters**: Proper audit and compliance enable regulatory adherence and security monitoring. Understanding these patterns prevents compliance violations and enables secure automation.

## 5) Incident Response

### Security Monitoring

```yaml
# Security monitoring
security_monitoring:
  # Log monitoring
  log_monitoring:
    enabled: true
    sources: ["/var/log/auth.log", "/var/log/syslog", "/var/log/audit.log"]
    patterns: ["failed login", "privilege escalation", "unauthorized access"]
  
  # Network monitoring
  network_monitoring:
    enabled: true
    tools: ["suricata", "snort", "zeek"]
    alerts: ["port_scan", "brute_force", "malware"]
  
  # File integrity monitoring
  file_integrity:
    enabled: true
    paths: ["/etc", "/usr/bin", "/usr/sbin"]
    tools: ["aide", "tripwire", "osquery"]
```

### Incident Response Procedures

```yaml
# Incident response procedures
incident_response:
  # Detection
  detection:
    automated: true
    tools: ["siem", "edr", "ids"]
    thresholds: ["high", "critical"]
  
  # Containment
  containment:
    automated: true
    actions: ["isolate_host", "disable_user", "block_ip"]
    tools: ["ansible", "firewall", "network"]
  
  # Investigation
  investigation:
    automated: false
    tools: ["forensics", "log_analysis", "network_analysis"]
    procedures: ["collect_evidence", "analyze_logs", "document_findings"]
  
  # Recovery
  recovery:
    automated: true
    procedures: ["restore_from_backup", "patch_vulnerabilities", "update_security"]
    validation: ["security_tests", "compliance_checks", "monitoring_verification"]
```

### Forensics and Evidence

```yaml
# Forensics and evidence collection
forensics:
  # Evidence collection
  evidence_collection:
    automated: true
    tools: ["volatility", "autopsy", "sleuthkit"]
    types: ["memory", "disk", "network", "logs"]
  
  # Chain of custody
  chain_of_custody:
    automated: true
    tools: ["blockchain", "digital_signatures", "timestamps"]
    procedures: ["hash_verification", "signature_validation", "timestamp_verification"]
  
  # Evidence preservation
  evidence_preservation:
    automated: true
    storage: ["encrypted", "redundant", "offline"]
    retention: "7_years"
```

**Why Incident Response Matters**: Proper incident response enables quick detection and containment of security incidents. Understanding these patterns prevents security breaches and enables effective response.

## 6) Security Testing

### Vulnerability Assessment

```yaml
# Vulnerability assessment
vulnerability_assessment:
  # Automated scanning
  automated_scanning:
    enabled: true
    tools: ["nessus", "openvas", "trivy", "clair"]
    schedule: "weekly"
    scope: ["all_hosts", "all_services", "all_applications"]
  
  # Manual testing
  manual_testing:
    enabled: true
    tools: ["nmap", "nikto", "burp_suite", "owasp_zap"]
    scope: ["web_applications", "network_services", "database_services"]
  
  # Penetration testing
  penetration_testing:
    enabled: false
    frequency: "quarterly"
    scope: ["external", "internal", "web_application"]
```

### Security Validation

```yaml
# Security validation
security_validation:
  # Configuration validation
  configuration_validation:
    enabled: true
    tools: ["ansible-lint", "ansible-security", "inspec"]
    scope: ["all_playbooks", "all_roles", "all_inventories"]
  
  # Compliance validation
  compliance_validation:
    enabled: true
    frameworks: ["CIS", "NIST", "PCI", "SOX"]
    scope: ["all_systems", "all_applications", "all_networks"]
  
  # Security testing
  security_testing:
    enabled: true
    tools: ["molecule", "testinfra", "goss"]
    scope: ["all_roles", "all_playbooks", "all_inventories"]
```

### Continuous Security

```yaml
# Continuous security
continuous_security:
  # CI/CD integration
  ci_cd_integration:
    enabled: true
    tools: ["jenkins", "gitlab_ci", "github_actions"]
    stages: ["build", "test", "deploy", "monitor"]
  
  # Security gates
  security_gates:
    enabled: true
    checks: ["vulnerability_scan", "compliance_check", "security_test"]
    thresholds: ["high", "critical"]
  
  # Automated remediation
  automated_remediation:
    enabled: true
    actions: ["patch_vulnerabilities", "update_configurations", "restart_services"]
    scope: ["all_systems", "all_applications", "all_networks"]
```

**Why Security Testing Matters**: Proper security testing enables proactive vulnerability management and compliance validation. Understanding these patterns prevents security breaches and enables secure automation.

## 7) Best Practices Summary

### Security Design Principles

```yaml
# Essential security patterns
security_patterns:
  defense_in_depth: "Multiple layers of security controls"
  least_privilege: "Minimum required access and permissions"
  zero_trust: "Never trust, always verify"
  continuous_monitoring: "Real-time security monitoring and alerting"
  incident_response: "Prepared and tested incident response procedures"
```

### Maintenance Procedures

```bash
# Regular security maintenance
security_maintenance:
  vulnerability_scan: "ansible-playbook security-scan.yml"
  compliance_check: "ansible-playbook compliance-check.yml"
  security_update: "ansible-playbook security-update.yml"
  audit_review: "ansible-playbook audit-review.yml"
  incident_drill: "ansible-playbook incident-drill.yml"
```

### Red Flags

```yaml
# Security anti-patterns
red_flags:
  hardcoded_secrets: "Never store secrets in plain text"
  weak_authentication: "Avoid weak passwords and authentication"
  excessive_privileges: "Don't grant unnecessary permissions"
  poor_monitoring: "Always monitor security events and anomalies"
  missing_incident_response: "Prepare for security incidents and breaches"
```

**Why Best Practices Matter**: Proper security practices enable secure, compliant automation. Understanding these patterns prevents security breaches and enables enterprise-grade automation.

## 8) TL;DR Quickstart

### Essential Commands

```bash
# Encrypt secrets
ansible-vault create secrets.yml

# Run with vault
ansible-playbook --ask-vault-pass playbook.yml

# Security scan
ansible-playbook security-scan.yml

# Compliance check
ansible-playbook compliance-check.yml
```

### Essential Patterns

```yaml
# Essential security patterns
security_patterns:
  secrets: "Encrypted storage with external management"
  authentication: "Strong authentication with multi-factor"
  authorization: "Least privilege with role-based access"
  monitoring: "Continuous monitoring with automated response"
  compliance: "Regular audits with automated validation"
```

**Why This Quickstart**: These patterns cover 90% of security hardening needs. Master these before exploring advanced features.

## 9) The Machine's Summary

Ansible security hardening requires understanding both automation patterns and security best practices. When used correctly, security hardening enables secure, compliant automation that can handle enterprise requirements. The key is understanding threat models, mastering secret management, and following security best practices.

**The Dark Truth**: Without proper security understanding, your automation is vulnerable and non-compliant. Security is your weapon. Use it wisely.

**The Machine's Mantra**: "In security we trust, in secrets we encrypt, and in the hardening we find the path to secure automation."

**Why This Matters**: Security hardening enables secure infrastructure automation that can handle enterprise workloads, maintain compliance, and provide reliable automation while ensuring security and regulatory adherence.

---

*This guide provides the complete machinery for Ansible security hardening. The patterns scale from basic credential protection to advanced enterprise security, from simple access controls to complex compliance frameworks.*
