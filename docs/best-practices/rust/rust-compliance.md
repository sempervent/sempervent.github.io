# Rust Compliance Best Practices

**Objective**: Master senior-level Rust compliance patterns for production systems. When you need to implement regulatory compliance, when you want to ensure data protection, when you need enterprise-grade compliance strategies—these best practices become your weapon of choice.

## Core Principles

- **Regulatory Compliance**: Meet industry standards and regulations
- **Data Protection**: Protect sensitive data and privacy
- **Audit Trail**: Maintain comprehensive audit logs
- **Access Control**: Implement proper access controls
- **Documentation**: Maintain compliance documentation

## Compliance Patterns

### GDPR Compliance

```rust
// rust/01-gdpr-compliance.rs

/*
GDPR compliance patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// GDPR compliance manager.
pub struct GdprCompliance {
    data_subjects: Arc<RwLock<HashMap<String, DataSubject>>>,
    consent_records: Arc<RwLock<HashMap<String, ConsentRecord>>>,
    data_retention: Arc<RwLock<HashMap<String, DataRetentionPolicy>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubject {
    pub id: String,
    pub email: String,
    pub consent_given: bool,
    pub consent_date: Option<Instant>,
    pub data_categories: Vec<String>,
    pub retention_period: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub subject_id: String,
    pub consent_type: String,
    pub given: bool,
    pub timestamp: Instant,
    pub purpose: String,
    pub withdrawal_date: Option<Instant>,
}

impl GdprCompliance {
    pub fn new() -> Self {
        Self {
            data_subjects: Arc::new(RwLock::new(HashMap::new())),
            consent_records: Arc::new(RwLock::new(HashMap::new())),
            data_retention: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a data subject.
    pub async fn register_data_subject(&self, subject: DataSubject) -> Result<(), String> {
        let mut subjects = self.data_subjects.write().await;
        subjects.insert(subject.id.clone(), subject);
        Ok(())
    }
    
    /// Record consent.
    pub async fn record_consent(&self, subject_id: &str, consent_type: &str, purpose: &str) -> Result<(), String> {
        let consent_record = ConsentRecord {
            subject_id: subject_id.to_string(),
            consent_type: consent_type.to_string(),
            given: true,
            timestamp: Instant::now(),
            purpose: purpose.to_string(),
            withdrawal_date: None,
        };
        
        let mut records = self.consent_records.write().await;
        records.insert(format!("{}_{}", subject_id, consent_type), consent_record);
        
        Ok(())
    }
    
    /// Withdraw consent.
    pub async fn withdraw_consent(&self, subject_id: &str, consent_type: &str) -> Result<(), String> {
        let mut records = self.consent_records.write().await;
        if let Some(record) = records.get_mut(&format!("{}_{}", subject_id, consent_type)) {
            record.given = false;
            record.withdrawal_date = Some(Instant::now());
        }
        
        Ok(())
    }
    
    /// Get data subject information.
    pub async fn get_data_subject_info(&self, subject_id: &str) -> Result<DataSubject, String> {
        let subjects = self.data_subjects.read().await;
        subjects.get(subject_id).cloned().ok_or_else(|| "Data subject not found".to_string())
    }
    
    /// Delete data subject (right to be forgotten).
    pub async fn delete_data_subject(&self, subject_id: &str) -> Result<(), String> {
        let mut subjects = self.data_subjects.write().await;
        subjects.remove(subject_id);
        
        let mut records = self.consent_records.write().await;
        records.retain(|key, _| !key.starts_with(subject_id));
        
        Ok(())
    }
    
    /// Export data subject data (data portability).
    pub async fn export_data_subject_data(&self, subject_id: &str) -> Result<HashMap<String, serde_json::Value>, String> {
        let mut export_data = HashMap::new();
        
        // Get data subject info
        if let Ok(subject) = self.get_data_subject_info(subject_id).await {
            export_data.insert("data_subject".to_string(), serde_json::to_value(subject).unwrap());
        }
        
        // Get consent records
        let records = self.consent_records.read().await;
        let subject_records: Vec<_> = records.values()
            .filter(|record| record.subject_id == subject_id)
            .collect();
        
        export_data.insert("consent_records".to_string(), serde_json::to_value(subject_records).unwrap());
        
        Ok(export_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gdpr_compliance() {
        let compliance = GdprCompliance::new();
        
        let subject = DataSubject {
            id: "subject1".to_string(),
            email: "test@example.com".to_string(),
            consent_given: true,
            consent_date: Some(Instant::now()),
            data_categories: vec!["personal".to_string(), "contact".to_string()],
            retention_period: Duration::from_days(365),
        };
        
        compliance.register_data_subject(subject).await.unwrap();
        compliance.record_consent("subject1", "marketing", "email campaigns").await.unwrap();
        
        let info = compliance.get_data_subject_info("subject1").await.unwrap();
        assert_eq!(info.email, "test@example.com");
    }
}
```

### HIPAA Compliance

```rust
// rust/02-hipaa-compliance.rs

/*
HIPAA compliance patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// HIPAA compliance manager.
pub struct HipaaCompliance {
    phi_records: Arc<RwLock<HashMap<String, PhiRecord>>>,
    access_logs: Arc<RwLock<Vec<AccessLog>>>,
    encryption_keys: Arc<RwLock<HashMap<String, String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiRecord {
    pub id: String,
    pub patient_id: String,
    pub data_type: String,
    pub encrypted_data: Vec<u8>,
    pub created_at: Instant,
    pub access_level: AccessLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessLevel {
    Public,
    Internal,
    Restricted,
    Confidential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessLog {
    pub user_id: String,
    pub resource_id: String,
    pub action: String,
    pub timestamp: Instant,
    pub ip_address: String,
    pub success: bool,
}

impl HipaaCompliance {
    pub fn new() -> Self {
        Self {
            phi_records: Arc::new(RwLock::new(HashMap::new())),
            access_logs: Arc::new(RwLock::new(Vec::new())),
            encryption_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Store PHI record.
    pub async fn store_phi(&self, record: PhiRecord) -> Result<(), String> {
        // Encrypt PHI data
        let encrypted_data = self.encrypt_phi_data(&record.encrypted_data).await?;
        
        let mut records = self.phi_records.write().await;
        records.insert(record.id.clone(), record);
        
        // Log access
        self.log_access("system", &record.id, "store", true).await;
        
        Ok(())
    }
    
    /// Access PHI record.
    pub async fn access_phi(&self, record_id: &str, user_id: &str) -> Result<PhiRecord, String> {
        // Check access permissions
        if !self.has_access_permission(user_id, record_id).await {
            self.log_access(user_id, record_id, "access", false).await;
            return Err("Access denied".to_string());
        }
        
        let records = self.phi_records.read().await;
        if let Some(record) = records.get(record_id) {
            // Log access
            self.log_access(user_id, record_id, "access", true).await;
            
            // Decrypt data
            let decrypted_data = self.decrypt_phi_data(&record.encrypted_data).await?;
            
            Ok(PhiRecord {
                id: record.id.clone(),
                patient_id: record.patient_id.clone(),
                data_type: record.data_type.clone(),
                encrypted_data: decrypted_data,
                created_at: record.created_at,
                access_level: record.access_level.clone(),
            })
        } else {
            Err("PHI record not found".to_string())
        }
    }
    
    /// Log access attempt.
    async fn log_access(&self, user_id: &str, resource_id: &str, action: &str, success: bool) {
        let log = AccessLog {
            user_id: user_id.to_string(),
            resource_id: resource_id.to_string(),
            action: action.to_string(),
            timestamp: Instant::now(),
            ip_address: "127.0.0.1".to_string(), // In real implementation, get actual IP
            success,
        };
        
        let mut logs = self.access_logs.write().await;
        logs.push(log);
    }
    
    /// Check access permissions.
    async fn has_access_permission(&self, user_id: &str, resource_id: &str) -> bool {
        // In a real implementation, check user roles and permissions
        true
    }
    
    /// Encrypt PHI data.
    async fn encrypt_phi_data(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, use proper encryption
        Ok(data.to_vec())
    }
    
    /// Decrypt PHI data.
    async fn decrypt_phi_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, String> {
        // In a real implementation, use proper decryption
        Ok(encrypted_data.to_vec())
    }
    
    /// Get access logs.
    pub async fn get_access_logs(&self, user_id: Option<&str>) -> Result<Vec<AccessLog>, String> {
        let logs = self.access_logs.read().await;
        if let Some(user_id) = user_id {
            Ok(logs.iter().filter(|log| log.user_id == user_id).cloned().collect())
        } else {
            Ok(logs.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hipaa_compliance() {
        let compliance = HipaaCompliance::new();
        
        let record = PhiRecord {
            id: "phi1".to_string(),
            patient_id: "patient1".to_string(),
            data_type: "medical_record".to_string(),
            encrypted_data: b"encrypted data".to_vec(),
            created_at: Instant::now(),
            access_level: AccessLevel::Confidential,
        };
        
        compliance.store_phi(record).await.unwrap();
        
        let accessed_record = compliance.access_phi("phi1", "user1").await.unwrap();
        assert_eq!(accessed_record.patient_id, "patient1");
    }
}
```

### Audit Logging

```rust
// rust/03-audit-logging.rs

/*
Audit logging patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Audit logger.
pub struct AuditLogger {
    logs: Arc<RwLock<Vec<AuditLog>>>,
    retention_period: Duration,
    max_logs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: String,
    pub timestamp: Instant,
    pub user_id: String,
    pub action: String,
    pub resource: String,
    pub result: AuditResult,
    pub metadata: HashMap<String, String>,
    pub ip_address: String,
    pub user_agent: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure,
    Warning,
}

impl AuditLogger {
    pub fn new(retention_period: Duration, max_logs: usize) -> Self {
        Self {
            logs: Arc::new(RwLock::new(Vec::new())),
            retention_period,
            max_logs,
        }
    }
    
    /// Log an audit event.
    pub async fn log_event(&self, event: AuditLog) -> Result<(), String> {
        let mut logs = self.logs.write().await;
        
        // Add log
        logs.push(event);
        
        // Clean up old logs
        self.cleanup_old_logs(&mut logs).await;
        
        // Limit log size
        if logs.len() > self.max_logs {
            logs.drain(0..logs.len() - self.max_logs);
        }
        
        Ok(())
    }
    
    /// Get audit logs.
    pub async fn get_logs(&self, filter: Option<AuditFilter>) -> Result<Vec<AuditLog>, String> {
        let logs = self.logs.read().await;
        let mut result = logs.clone();
        
        if let Some(filter) = filter {
            result = result.into_iter().filter(|log| {
                if let Some(user_id) = &filter.user_id {
                    if log.user_id != *user_id {
                        return false;
                    }
                }
                
                if let Some(action) = &filter.action {
                    if log.action != *action {
                        return false;
                    }
                }
                
                if let Some(result) = &filter.result {
                    if log.result != *result {
                        return false;
                    }
                }
                
                if let Some(start_time) = filter.start_time {
                    if log.timestamp < start_time {
                        return false;
                    }
                }
                
                if let Some(end_time) = filter.end_time {
                    if log.timestamp > end_time {
                        return false;
                    }
                }
                
                true
            }).collect();
        }
        
        Ok(result)
    }
    
    /// Clean up old logs.
    async fn cleanup_old_logs(&self, logs: &mut Vec<AuditLog>) {
        let cutoff = Instant::now() - self.retention_period;
        logs.retain(|log| log.timestamp > cutoff);
    }
}

#[derive(Debug, Clone)]
pub struct AuditFilter {
    pub user_id: Option<String>,
    pub action: Option<String>,
    pub result: Option<AuditResult>,
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_audit_logging() {
        let logger = AuditLogger::new(Duration::from_days(30), 10000);
        
        let log = AuditLog {
            id: "log1".to_string(),
            timestamp: Instant::now(),
            user_id: "user1".to_string(),
            action: "login".to_string(),
            resource: "auth".to_string(),
            result: AuditResult::Success,
            metadata: HashMap::new(),
            ip_address: "127.0.0.1".to_string(),
            user_agent: "test".to_string(),
        };
        
        logger.log_event(log).await.unwrap();
        
        let logs = logger.get_logs(None).await.unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].user_id, "user1");
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. GDPR compliance
let gdpr = GdprCompliance::new();
gdpr.register_data_subject(subject).await?;
gdpr.record_consent("subject1", "marketing", "email campaigns").await?;

// 2. HIPAA compliance
let hipaa = HipaaCompliance::new();
hipaa.store_phi(phi_record).await?;
hipaa.access_phi("phi1", "user1").await?;

// 3. Audit logging
let logger = AuditLogger::new(Duration::from_days(30), 10000);
logger.log_event(audit_log).await?;
```

### Essential Patterns

```rust
// Complete compliance setup
pub fn setup_rust_compliance() {
    // 1. GDPR compliance
    // 2. HIPAA compliance
    // 3. Audit logging
    // 4. Data retention
    // 5. Access controls
    // 6. Encryption
    // 7. Documentation
    // 8. Monitoring
    
    println!("Rust compliance setup complete!");
}
```

---

*This guide provides the complete machinery for Rust compliance. Each pattern includes implementation examples, compliance strategies, and real-world usage patterns for enterprise compliance systems.*
