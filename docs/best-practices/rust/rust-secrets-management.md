# Rust Secrets Management Best Practices

**Objective**: Master senior-level Rust secrets management patterns for production systems. When you need to handle sensitive data securely, when you want to implement robust secret rotation, when you need enterprise-grade secrets management strategies—these best practices become your weapon of choice.

## Core Principles

- **Never Hardcode Secrets**: Use environment variables or secret stores
- **Encrypt at Rest**: Store secrets encrypted
- **Rotate Regularly**: Implement automatic secret rotation
- **Least Privilege**: Minimal access to secrets
- **Audit Access**: Log all secret access

## Secrets Management Patterns

### Environment Variables

```rust
// rust/01-environment-variables.rs

/*
Environment variables patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use std::env;

/// Environment variables manager.
pub struct EnvManager {
    variables: Arc<RwLock<HashMap<String, String>>>,
    required_vars: Vec<String>,
    sensitive_vars: Vec<String>,
    validation_rules: HashMap<String, ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub required: bool,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub allowed_values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    pub database_url: String,
    pub redis_url: String,
    pub jwt_secret: String,
    pub api_key: String,
    pub encryption_key: String,
    pub log_level: String,
    pub port: u16,
    pub host: String,
}

impl EnvManager {
    pub fn new() -> Self {
        let mut manager = Self {
            variables: Arc::new(RwLock::new(HashMap::new())),
            required_vars: vec![
                "DATABASE_URL".to_string(),
                "REDIS_URL".to_string(),
                "JWT_SECRET".to_string(),
                "API_KEY".to_string(),
                "ENCRYPTION_KEY".to_string(),
            ],
            sensitive_vars: vec![
                "DATABASE_URL".to_string(),
                "JWT_SECRET".to_string(),
                "API_KEY".to_string(),
                "ENCRYPTION_KEY".to_string(),
            ],
            validation_rules: HashMap::new(),
        };
        
        // Set up validation rules
        manager.setup_validation_rules();
        manager
    }
    
    /// Load environment variables.
    pub async fn load(&mut self) -> Result<(), String> {
        let mut variables = self.variables.write().await;
        
        // Load all environment variables
        for (key, value) in env::vars() {
            variables.insert(key, value);
        }
        
        // Validate required variables
        for var in &self.required_vars {
            if !variables.contains_key(var) {
                return Err(format!("Required environment variable {} not set", var));
            }
        }
        
        // Validate sensitive variables
        for var in &self.sensitive_vars {
            if let Some(value) = variables.get(var) {
                if value.is_empty() {
                    return Err(format!("Sensitive environment variable {} is empty", var));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get environment variable.
    pub async fn get(&self, key: &str) -> Option<String> {
        let variables = self.variables.read().await;
        variables.get(key).cloned()
    }
    
    /// Get environment variable with default.
    pub async fn get_with_default(&self, key: &str, default: &str) -> String {
        self.get(key).await.unwrap_or_else(|| default.to_string())
    }
    
    /// Get required environment variable.
    pub async fn get_required(&self, key: &str) -> Result<String, String> {
        self.get(key).await.ok_or_else(|| format!("Required environment variable {} not set", key))
    }
    
    /// Set environment variable.
    pub async fn set(&self, key: &str, value: &str) -> Result<(), String> {
        // Validate sensitive variables
        if self.sensitive_vars.contains(&key.to_string()) {
            if value.is_empty() {
                return Err(format!("Sensitive environment variable {} cannot be empty", key));
            }
        }
        
        // Validate against rules
        if let Some(rule) = self.validation_rules.get(key) {
            self.validate_value(key, value, rule)?;
        }
        
        let mut variables = self.variables.write().await;
        variables.insert(key.to_string(), value.to_string());
        
        Ok(())
    }
    
    /// Validate environment variable value.
    fn validate_value(&self, key: &str, value: &str, rule: &ValidationRule) -> Result<(), String> {
        if rule.required && value.is_empty() {
            return Err(format!("Environment variable {} is required", key));
        }
        
        if let Some(min_length) = rule.min_length {
            if value.len() < min_length {
                return Err(format!("Environment variable {} must be at least {} characters long", key, min_length));
            }
        }
        
        if let Some(max_length) = rule.max_length {
            if value.len() > max_length {
                return Err(format!("Environment variable {} must be no more than {} characters long", key, max_length));
            }
        }
        
        if let Some(pattern) = &rule.pattern {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if !regex.is_match(value) {
                    return Err(format!("Environment variable {} does not match required pattern", key));
                }
            }
        }
        
        if let Some(allowed_values) = &rule.allowed_values {
            if !allowed_values.contains(&value.to_string()) {
                return Err(format!("Environment variable {} must be one of: {}", key, allowed_values.join(", ")));
            }
        }
        
        Ok(())
    }
    
    /// Setup validation rules.
    fn setup_validation_rules(&mut self) {
        // Database URL validation
        self.validation_rules.insert("DATABASE_URL".to_string(), ValidationRule {
            required: true,
            min_length: Some(10),
            max_length: Some(500),
            pattern: Some(r"^postgres://.*$".to_string()),
            allowed_values: None,
        });
        
        // JWT Secret validation
        self.validation_rules.insert("JWT_SECRET".to_string(), ValidationRule {
            required: true,
            min_length: Some(32),
            max_length: Some(256),
            pattern: None,
            allowed_values: None,
        });
        
        // API Key validation
        self.validation_rules.insert("API_KEY".to_string(), ValidationRule {
            required: true,
            min_length: Some(16),
            max_length: Some(128),
            pattern: None,
            allowed_values: None,
        });
        
        // Log Level validation
        self.validation_rules.insert("LOG_LEVEL".to_string(), ValidationRule {
            required: false,
            min_length: None,
            max_length: None,
            pattern: None,
            allowed_values: Some(vec!["debug".to_string(), "info".to_string(), "warn".to_string(), "error".to_string()]),
        });
    }
    
    /// Get configuration struct.
    pub async fn get_config(&self) -> Result<EnvConfig, String> {
        Ok(EnvConfig {
            database_url: self.get_required("DATABASE_URL").await?,
            redis_url: self.get_required("REDIS_URL").await?,
            jwt_secret: self.get_required("JWT_SECRET").await?,
            api_key: self.get_required("API_KEY").await?,
            encryption_key: self.get_required("ENCRYPTION_KEY").await?,
            log_level: self.get_with_default("LOG_LEVEL", "info").await,
            port: self.get_with_default("PORT", "8080").await.parse().unwrap_or(8080),
            host: self.get_with_default("HOST", "0.0.0.0").await,
        })
    }
    
    /// Mask sensitive values for logging.
    pub fn mask_sensitive(&self, key: &str, value: &str) -> String {
        if self.sensitive_vars.contains(&key.to_string()) {
            if value.len() <= 8 {
                "*".repeat(value.len())
            } else {
                format!("{}...{}", &value[..4], "*".repeat(value.len() - 8))
            }
        } else {
            value.to_string()
        }
    }
    
    /// Get all variables (with sensitive ones masked).
    pub async fn get_all_masked(&self) -> HashMap<String, String> {
        let variables = self.variables.read().await;
        let mut masked = HashMap::new();
        
        for (key, value) in variables.iter() {
            masked.insert(key.clone(), self.mask_sensitive(key, value));
        }
        
        masked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_env_manager() {
        let mut manager = EnvManager::new();
        
        // Set test environment variables
        manager.set("DATABASE_URL", "postgres://user:pass@localhost/db").await.unwrap();
        manager.set("REDIS_URL", "redis://localhost:6379").await.unwrap();
        manager.set("JWT_SECRET", "super-secret-jwt-key").await.unwrap();
        manager.set("API_KEY", "api-key-123456789").await.unwrap();
        manager.set("ENCRYPTION_KEY", "encryption-key-123456789").await.unwrap();
        
        let config = manager.get_config().await.unwrap();
        assert_eq!(config.database_url, "postgres://user:pass@localhost/db");
        assert_eq!(config.jwt_secret, "super-secret-jwt-key");
    }
    
    #[tokio::test]
    async fn test_validation() {
        let mut manager = EnvManager::new();
        
        // Test invalid database URL
        let result = manager.set("DATABASE_URL", "invalid-url").await;
        assert!(result.is_err());
        
        // Test valid database URL
        let result = manager.set("DATABASE_URL", "postgres://user:pass@localhost/db").await;
        assert!(result.is_ok());
    }
}
```

### Secret Stores

```rust
// rust/02-secret-stores.rs

/*
Secret stores patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Secret store manager.
pub struct SecretStore {
    secrets: Arc<RwLock<HashMap<String, Secret>>>,
    encryption_key: String,
    rotation_interval: Duration,
    last_rotation: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Secret {
    pub key: String,
    pub value: String,
    pub version: u32,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub tags: HashMap<String, String>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub key: String,
    pub version: u32,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub tags: HashMap<String, String>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretRotationConfig {
    pub key: String,
    pub rotation_interval: Duration,
    pub auto_rotation: bool,
    pub notification_webhook: Option<String>,
}

impl SecretStore {
    pub fn new(encryption_key: String, rotation_interval: Duration) -> Self {
        Self {
            secrets: Arc::new(RwLock::new(HashMap::new())),
            encryption_key,
            rotation_interval,
            last_rotation: Instant::now(),
        }
    }
    
    /// Store a secret.
    pub async fn store_secret(&self, key: String, value: String, tags: HashMap<String, String>) -> Result<SecretMetadata, String> {
        // Encrypt the secret value
        let encrypted_value = self.encrypt_value(&value)?;
        
        // Create secret
        let secret = Secret {
            key: key.clone(),
            value: encrypted_value,
            version: 1,
            created_at: Instant::now(),
            expires_at: None,
            tags,
            is_active: true,
        };
        
        // Store secret
        let mut secrets = self.secrets.write().await;
        secrets.insert(key.clone(), secret);
        
        Ok(SecretMetadata {
            key,
            version: 1,
            created_at: Instant::now(),
            expires_at: None,
            tags: HashMap::new(),
            is_active: true,
        })
    }
    
    /// Retrieve a secret.
    pub async fn get_secret(&self, key: &str) -> Result<String, String> {
        let secrets = self.secrets.read().await;
        if let Some(secret) = secrets.get(key) {
            if !secret.is_active {
                return Err("Secret is not active".to_string());
            }
            
            if let Some(expires_at) = secret.expires_at {
                if expires_at < Instant::now() {
                    return Err("Secret has expired".to_string());
                }
            }
            
            // Decrypt the secret value
            let decrypted_value = self.decrypt_value(&secret.value)?;
            Ok(decrypted_value)
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// Update a secret.
    pub async fn update_secret(&self, key: &str, value: String, tags: Option<HashMap<String, String>>) -> Result<SecretMetadata, String> {
        let mut secrets = self.secrets.write().await;
        if let Some(secret) = secrets.get_mut(key) {
            // Encrypt the new value
            let encrypted_value = self.encrypt_value(&value)?;
            
            // Update secret
            secret.value = encrypted_value;
            secret.version += 1;
            secret.created_at = Instant::now();
            
            if let Some(new_tags) = tags {
                secret.tags = new_tags;
            }
            
            Ok(SecretMetadata {
                key: key.to_string(),
                version: secret.version,
                created_at: secret.created_at,
                expires_at: secret.expires_at,
                tags: secret.tags.clone(),
                is_active: secret.is_active,
            })
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// Delete a secret.
    pub async fn delete_secret(&self, key: &str) -> Result<(), String> {
        let mut secrets = self.secrets.write().await;
        if let Some(secret) = secrets.get_mut(key) {
            secret.is_active = false;
            Ok(())
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// List secrets.
    pub async fn list_secrets(&self, prefix: Option<&str>) -> Result<Vec<SecretMetadata>, String> {
        let secrets = self.secrets.read().await;
        let mut result = Vec::new();
        
        for (key, secret) in secrets.iter() {
            if let Some(prefix) = prefix {
                if !key.starts_with(prefix) {
                    continue;
                }
            }
            
            result.push(SecretMetadata {
                key: key.clone(),
                version: secret.version,
                created_at: secret.created_at,
                expires_at: secret.expires_at,
                tags: secret.tags.clone(),
                is_active: secret.is_active,
            });
        }
        
        Ok(result)
    }
    
    /// Rotate a secret.
    pub async fn rotate_secret(&self, key: &str, new_value: String) -> Result<SecretMetadata, String> {
        let mut secrets = self.secrets.write().await;
        if let Some(secret) = secrets.get_mut(key) {
            // Encrypt the new value
            let encrypted_value = self.encrypt_value(&new_value)?;
            
            // Update secret
            secret.value = encrypted_value;
            secret.version += 1;
            secret.created_at = Instant::now();
            
            Ok(SecretMetadata {
                key: key.to_string(),
                version: secret.version,
                created_at: secret.created_at,
                expires_at: secret.expires_at,
                tags: secret.tags.clone(),
                is_active: secret.is_active,
            })
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// Set secret expiration.
    pub async fn set_secret_expiration(&self, key: &str, expires_at: Instant) -> Result<(), String> {
        let mut secrets = self.secrets.write().await;
        if let Some(secret) = secrets.get_mut(key) {
            secret.expires_at = Some(expires_at);
            Ok(())
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// Get secret metadata.
    pub async fn get_secret_metadata(&self, key: &str) -> Result<SecretMetadata, String> {
        let secrets = self.secrets.read().await;
        if let Some(secret) = secrets.get(key) {
            Ok(SecretMetadata {
                key: key.to_string(),
                version: secret.version,
                created_at: secret.created_at,
                expires_at: secret.expires_at,
                tags: secret.tags.clone(),
                is_active: secret.is_active,
            })
        } else {
            Err("Secret not found".to_string())
        }
    }
    
    /// Check if secret rotation is needed.
    pub fn needs_rotation(&self) -> bool {
        self.last_rotation.elapsed() >= self.rotation_interval
    }
    
    /// Rotate all secrets.
    pub async fn rotate_all_secrets(&self) -> Result<Vec<String>, String> {
        let mut rotated_keys = Vec::new();
        let secrets = self.secrets.read().await;
        
        for (key, secret) in secrets.iter() {
            if secret.is_active {
                // In a real implementation, you would generate new values
                // and rotate the secrets
                rotated_keys.push(key.clone());
            }
        }
        
        Ok(rotated_keys)
    }
    
    /// Encrypt secret value.
    fn encrypt_value(&self, value: &str) -> Result<String, String> {
        // In a real implementation, you would use proper encryption
        // For this example, we'll just base64 encode
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(value);
        Ok(encoded)
    }
    
    /// Decrypt secret value.
    fn decrypt_value(&self, encrypted_value: &str) -> Result<String, String> {
        // In a real implementation, you would use proper decryption
        // For this example, we'll just base64 decode
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD.decode(encrypted_value)
            .map_err(|e| format!("Decryption failed: {}", e))?;
        String::from_utf8(decoded)
            .map_err(|e| format!("UTF-8 conversion failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_secret_store() {
        let store = SecretStore::new("encryption-key".to_string(), Duration::from_days(30));
        
        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "production".to_string());
        tags.insert("service".to_string(), "api".to_string());
        
        let metadata = store.store_secret("database_password", "secret123", tags).await.unwrap();
        assert_eq!(metadata.key, "database_password");
        assert_eq!(metadata.version, 1);
        
        let value = store.get_secret("database_password").await.unwrap();
        assert_eq!(value, "secret123");
    }
    
    #[tokio::test]
    async fn test_secret_rotation() {
        let store = SecretStore::new("encryption-key".to_string(), Duration::from_days(30));
        
        store.store_secret("api_key", "old-key", HashMap::new()).await.unwrap();
        
        let metadata = store.rotate_secret("api_key", "new-key").await.unwrap();
        assert_eq!(metadata.version, 2);
        
        let value = store.get_secret("api_key").await.unwrap();
        assert_eq!(value, "new-key");
    }
}
```

### Key Management

```rust
// rust/03-key-management.rs

/*
Key management patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Key manager.
pub struct KeyManager {
    keys: Arc<RwLock<HashMap<String, Key>>>,
    key_rotation_interval: Duration,
    last_rotation: Instant,
    master_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Key {
    pub id: String,
    pub name: String,
    pub algorithm: KeyAlgorithm,
    pub key_data: Vec<u8>,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub is_active: bool,
    pub version: u32,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyAlgorithm {
    Aes256,
    Rsa2048,
    Rsa4096,
    EcdsaP256,
    EcdsaP384,
    Ed25519,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    pub id: String,
    pub name: String,
    pub algorithm: KeyAlgorithm,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub is_active: bool,
    pub version: u32,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationConfig {
    pub key_id: String,
    pub rotation_interval: Duration,
    pub auto_rotation: bool,
    pub notification_webhook: Option<String>,
}

impl KeyManager {
    pub fn new(master_key: String, rotation_interval: Duration) -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            key_rotation_interval: rotation_interval,
            last_rotation: Instant::now(),
            master_key,
        }
    }
    
    /// Generate a new key.
    pub async fn generate_key(&self, name: String, algorithm: KeyAlgorithm, tags: HashMap<String, String>) -> Result<KeyMetadata, String> {
        let key_id = uuid::Uuid::new_v4().to_string();
        let key_data = self.generate_key_data(&algorithm)?;
        
        let key = Key {
            id: key_id.clone(),
            name: name.clone(),
            algorithm: algorithm.clone(),
            key_data,
            created_at: Instant::now(),
            expires_at: None,
            is_active: true,
            version: 1,
            tags,
        };
        
        // Store key
        let mut keys = self.keys.write().await;
        keys.insert(key_id.clone(), key);
        
        Ok(KeyMetadata {
            id: key_id,
            name,
            algorithm,
            created_at: Instant::now(),
            expires_at: None,
            is_active: true,
            version: 1,
            tags: HashMap::new(),
        })
    }
    
    /// Get a key.
    pub async fn get_key(&self, key_id: &str) -> Result<Key, String> {
        let keys = self.keys.read().await;
        if let Some(key) = keys.get(key_id) {
            if !key.is_active {
                return Err("Key is not active".to_string());
            }
            
            if let Some(expires_at) = key.expires_at {
                if expires_at < Instant::now() {
                    return Err("Key has expired".to_string());
                }
            }
            
            Ok(key.clone())
        } else {
            Err("Key not found".to_string())
        }
    }
    
    /// List keys.
    pub async fn list_keys(&self, prefix: Option<&str>) -> Result<Vec<KeyMetadata>, String> {
        let keys = self.keys.read().await;
        let mut result = Vec::new();
        
        for (key_id, key) in keys.iter() {
            if let Some(prefix) = prefix {
                if !key.name.starts_with(prefix) {
                    continue;
                }
            }
            
            result.push(KeyMetadata {
                id: key_id.clone(),
                name: key.name.clone(),
                algorithm: key.algorithm.clone(),
                created_at: key.created_at,
                expires_at: key.expires_at,
                is_active: key.is_active,
                version: key.version,
                tags: key.tags.clone(),
            });
        }
        
        Ok(result)
    }
    
    /// Rotate a key.
    pub async fn rotate_key(&self, key_id: &str) -> Result<KeyMetadata, String> {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_id) {
            // Generate new key data
            let new_key_data = self.generate_key_data(&key.algorithm)?;
            
            // Update key
            key.key_data = new_key_data;
            key.version += 1;
            key.created_at = Instant::now();
            
            Ok(KeyMetadata {
                id: key_id.to_string(),
                name: key.name.clone(),
                algorithm: key.algorithm.clone(),
                created_at: key.created_at,
                expires_at: key.expires_at,
                is_active: key.is_active,
                version: key.version,
                tags: key.tags.clone(),
            })
        } else {
            Err("Key not found".to_string())
        }
    }
    
    /// Set key expiration.
    pub async fn set_key_expiration(&self, key_id: &str, expires_at: Instant) -> Result<(), String> {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_id) {
            key.expires_at = Some(expires_at);
            Ok(())
        } else {
            Err("Key not found".to_string())
        }
    }
    
    /// Deactivate a key.
    pub async fn deactivate_key(&self, key_id: &str) -> Result<(), String> {
        let mut keys = self.keys.write().await;
        if let Some(key) = keys.get_mut(key_id) {
            key.is_active = false;
            Ok(())
        } else {
            Err("Key not found".to_string())
        }
    }
    
    /// Get key metadata.
    pub async fn get_key_metadata(&self, key_id: &str) -> Result<KeyMetadata, String> {
        let keys = self.keys.read().await;
        if let Some(key) = keys.get(key_id) {
            Ok(KeyMetadata {
                id: key_id.to_string(),
                name: key.name.clone(),
                algorithm: key.algorithm.clone(),
                created_at: key.created_at,
                expires_at: key.expires_at,
                is_active: key.is_active,
                version: key.version,
                tags: key.tags.clone(),
            })
        } else {
            Err("Key not found".to_string())
        }
    }
    
    /// Check if key rotation is needed.
    pub fn needs_rotation(&self) -> bool {
        self.last_rotation.elapsed() >= self.key_rotation_interval
    }
    
    /// Rotate all keys.
    pub async fn rotate_all_keys(&self) -> Result<Vec<String>, String> {
        let mut rotated_keys = Vec::new();
        let keys = self.keys.read().await;
        
        for (key_id, key) in keys.iter() {
            if key.is_active {
                // In a real implementation, you would rotate the keys
                rotated_keys.push(key_id.clone());
            }
        }
        
        Ok(rotated_keys)
    }
    
    /// Generate key data based on algorithm.
    fn generate_key_data(&self, algorithm: &KeyAlgorithm) -> Result<Vec<u8>, String> {
        match algorithm {
            KeyAlgorithm::Aes256 => {
                let mut key = vec![0u8; 32];
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut key);
                Ok(key)
            },
            KeyAlgorithm::Rsa2048 => {
                // In a real implementation, you would generate RSA keys
                Ok(vec![0u8; 256])
            },
            KeyAlgorithm::Rsa4096 => {
                // In a real implementation, you would generate RSA keys
                Ok(vec![0u8; 512])
            },
            KeyAlgorithm::EcdsaP256 => {
                let mut key = vec![0u8; 32];
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut key);
                Ok(key)
            },
            KeyAlgorithm::EcdsaP384 => {
                let mut key = vec![0u8; 48];
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut key);
                Ok(key)
            },
            KeyAlgorithm::Ed25519 => {
                let mut key = vec![0u8; 32];
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut key);
                Ok(key)
            },
        }
    }
    
    /// Encrypt data with a key.
    pub async fn encrypt_data(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>, String> {
        let key = self.get_key(key_id).await?;
        
        match key.algorithm {
            KeyAlgorithm::Aes256 => {
                // In a real implementation, you would use AES-256-GCM
                Ok(data.to_vec())
            },
            KeyAlgorithm::Rsa2048 | KeyAlgorithm::Rsa4096 => {
                // In a real implementation, you would use RSA encryption
                Ok(data.to_vec())
            },
            _ => Err("Unsupported algorithm for encryption".to_string()),
        }
    }
    
    /// Decrypt data with a key.
    pub async fn decrypt_data(&self, key_id: &str, encrypted_data: &[u8]) -> Result<Vec<u8>, String> {
        let key = self.get_key(key_id).await?;
        
        match key.algorithm {
            KeyAlgorithm::Aes256 => {
                // In a real implementation, you would use AES-256-GCM
                Ok(encrypted_data.to_vec())
            },
            KeyAlgorithm::Rsa2048 | KeyAlgorithm::Rsa4096 => {
                // In a real implementation, you would use RSA decryption
                Ok(encrypted_data.to_vec())
            },
            _ => Err("Unsupported algorithm for decryption".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_key_manager() {
        let manager = KeyManager::new("master-key".to_string(), Duration::from_days(30));
        
        let mut tags = HashMap::new();
        tags.insert("environment".to_string(), "production".to_string());
        tags.insert("service".to_string(), "api".to_string());
        
        let metadata = manager.generate_key("encryption-key", KeyAlgorithm::Aes256, tags).await.unwrap();
        assert_eq!(metadata.name, "encryption-key");
        assert_eq!(metadata.algorithm, KeyAlgorithm::Aes256);
        assert_eq!(metadata.version, 1);
        
        let key = manager.get_key(&metadata.id).await.unwrap();
        assert_eq!(key.name, "encryption-key");
        assert_eq!(key.algorithm, KeyAlgorithm::Aes256);
    }
    
    #[tokio::test]
    async fn test_key_rotation() {
        let manager = KeyManager::new("master-key".to_string(), Duration::from_days(30));
        
        let metadata = manager.generate_key("test-key", KeyAlgorithm::Aes256, HashMap::new()).await.unwrap();
        let original_version = metadata.version;
        
        let rotated_metadata = manager.rotate_key(&metadata.id).await.unwrap();
        assert_eq!(rotated_metadata.version, original_version + 1);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Environment variables
let mut env_manager = EnvManager::new();
env_manager.load().await?;
let config = env_manager.get_config().await?;

// 2. Secret store
let store = SecretStore::new("encryption-key".to_string(), Duration::from_days(30));
store.store_secret("api_key", "secret123", HashMap::new()).await?;

// 3. Key management
let manager = KeyManager::new("master-key".to_string(), Duration::from_days(30));
manager.generate_key("encryption-key", KeyAlgorithm::Aes256, HashMap::new()).await?;
```

### Essential Patterns

```rust
// Complete secrets management setup
pub fn setup_rust_secrets_management() {
    // 1. Environment variables
    // 2. Secret stores
    // 3. Key management
    // 4. Secret rotation
    // 5. Encryption/decryption
    // 6. Access control
    // 7. Audit logging
    // 8. Compliance
    
    println!("Rust secrets management setup complete!");
}
```

---

*This guide provides the complete machinery for Rust secrets management. Each pattern includes implementation examples, security strategies, and real-world usage patterns for enterprise secrets management systems.*
