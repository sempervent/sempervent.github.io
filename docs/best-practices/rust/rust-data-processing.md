# Rust Data Processing Best Practices

**Objective**: Master senior-level Rust data processing patterns for production systems. When you need to build high-performance data pipelines, when you want to process large datasets efficiently, when you need enterprise-grade data processing—these best practices become your weapon of choice.

## Core Principles

- **Stream Processing**: Process data in streams for memory efficiency
- **Parallel Processing**: Leverage multiple cores for data processing
- **Memory Management**: Optimize memory usage for large datasets
- **Error Handling**: Robust error handling for data processing
- **Backpressure**: Handle data flow control in streaming scenarios

## Data Processing Patterns

### Stream Processing

```rust
// rust/01-stream-processing.rs

/*
Stream processing patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Data stream processor.
pub struct StreamProcessor {
    input_channel: mpsc::Receiver<DataRecord>,
    output_channel: mpsc::Sender<ProcessedRecord>,
    buffer_size: usize,
    processing_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    pub id: String,
    pub timestamp: Instant,
    pub data: HashMap<String, serde_json::Value>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedRecord {
    pub id: String,
    pub timestamp: Instant,
    pub processed_data: HashMap<String, serde_json::Value>,
    pub processing_time: Duration,
    pub source: String,
}

impl StreamProcessor {
    pub fn new(
        input_channel: mpsc::Receiver<DataRecord>,
        output_channel: mpsc::Sender<ProcessedRecord>,
        buffer_size: usize,
        processing_timeout: Duration,
    ) -> Self {
        Self {
            input_channel,
            output_channel,
            buffer_size,
            processing_timeout,
        }
    }
    
    /// Process data stream.
    pub async fn process_stream(&mut self) -> Result<(), String> {
        let mut buffer = Vec::with_capacity(self.buffer_size);
        
        loop {
            // Try to receive data with timeout
            match tokio::time::timeout(self.processing_timeout, self.input_channel.recv()).await {
                Ok(Some(record)) => {
                    buffer.push(record);
                    
                    // Process buffer when it's full
                    if buffer.len() >= self.buffer_size {
                        self.process_batch(&mut buffer).await?;
                    }
                }
                Ok(None) => {
                    // Input channel closed, process remaining buffer
                    if !buffer.is_empty() {
                        self.process_batch(&mut buffer).await?;
                    }
                    break;
                }
                Err(_) => {
                    // Timeout, process current buffer
                    if !buffer.is_empty() {
                        self.process_batch(&mut buffer).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Process a batch of records.
    async fn process_batch(&mut self, buffer: &mut Vec<DataRecord>) -> Result<(), String> {
        let start_time = Instant::now();
        
        // Process each record in the batch
        for record in buffer.drain(..) {
            let processed = self.process_record(record).await?;
            self.output_channel.send(processed).await
                .map_err(|e| format!("Failed to send processed record: {}", e))?;
        }
        
        let processing_time = start_time.elapsed();
        println!("Processed batch in {:?}", processing_time);
        
        Ok(())
    }
    
    /// Process a single record.
    async fn process_record(&self, record: DataRecord) -> Result<ProcessedRecord, String> {
        let start_time = Instant::now();
        
        // Simulate data processing
        let mut processed_data = HashMap::new();
        for (key, value) in record.data {
            // Transform the data
            let transformed_value = self.transform_value(&key, &value)?;
            processed_data.insert(key, transformed_value);
        }
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessedRecord {
            id: record.id,
            timestamp: record.timestamp,
            processed_data,
            processing_time,
            source: record.source,
        })
    }
    
    /// Transform a value based on its key.
    fn transform_value(&self, key: &str, value: &serde_json::Value) -> Result<serde_json::Value, String> {
        match key {
            "temperature" => {
                if let Some(temp) = value.as_f64() {
                    // Convert Celsius to Fahrenheit
                    let fahrenheit = (temp * 9.0 / 5.0) + 32.0;
                    Ok(serde_json::Value::Number(serde_json::Number::from_f64(fahrenheit).unwrap()))
                } else {
                    Err("Invalid temperature value".to_string())
                }
            }
            "pressure" => {
                if let Some(pressure) = value.as_f64() {
                    // Convert Pa to PSI
                    let psi = pressure * 0.000145038;
                    Ok(serde_json::Value::Number(serde_json::Number::from_f64(psi).unwrap()))
                } else {
                    Err("Invalid pressure value".to_string())
                }
            }
            _ => Ok(value.clone()),
        }
    }
}

/// Data pipeline orchestrator.
pub struct DataPipeline {
    processors: Vec<Arc<StreamProcessor>>,
    input_channels: Vec<mpsc::Sender<DataRecord>>,
    output_channels: Vec<mpsc::Receiver<ProcessedRecord>>,
}

impl DataPipeline {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
            input_channels: Vec::new(),
            output_channels: Vec::new(),
        }
    }
    
    /// Add a processor to the pipeline.
    pub fn add_processor(&mut self, buffer_size: usize, processing_timeout: Duration) -> (mpsc::Sender<DataRecord>, mpsc::Receiver<ProcessedRecord>) {
        let (input_tx, input_rx) = mpsc::channel(buffer_size);
        let (output_tx, output_rx) = mpsc::channel(buffer_size);
        
        let processor = Arc::new(StreamProcessor::new(
            input_rx,
            output_tx,
            buffer_size,
            processing_timeout,
        ));
        
        self.processors.push(processor);
        self.input_channels.push(input_tx);
        self.output_channels.push(output_rx);
        
        (self.input_channels.last().unwrap().clone(), output_rx)
    }
    
    /// Start all processors.
    pub async fn start(&self) -> Result<(), String> {
        let mut handles = Vec::new();
        
        for processor in &self.processors {
            let processor = Arc::clone(processor);
            let handle = tokio::spawn(async move {
                let mut processor = (*processor).clone();
                processor.process_stream().await
            });
            handles.push(handle);
        }
        
        // Wait for all processors to complete
        for handle in handles {
            handle.await.map_err(|e| format!("Processor error: {}", e))??;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stream_processor() {
        let (input_tx, input_rx) = mpsc::channel(10);
        let (output_tx, output_rx) = mpsc::channel(10);
        
        let mut processor = StreamProcessor::new(
            input_rx,
            output_tx,
            5,
            Duration::from_secs(1),
        );
        
        // Send test data
        let mut data = HashMap::new();
        data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(25)));
        data.insert("pressure".to_string(), serde_json::Value::Number(serde_json::Number::from(101325)));
        
        let record = DataRecord {
            id: "test-1".to_string(),
            timestamp: Instant::now(),
            data,
            source: "sensor-1".to_string(),
        };
        
        input_tx.send(record).await.unwrap();
        drop(input_tx);
        
        // Start processing
        let handle = tokio::spawn(async move {
            processor.process_stream().await
        });
        
        // Wait for processing to complete
        handle.await.unwrap().unwrap();
        
        // Check output
        let processed = output_rx.recv().await.unwrap();
        assert_eq!(processed.id, "test-1");
    }
    
    #[tokio::test]
    async fn test_data_pipeline() {
        let mut pipeline = DataPipeline::new();
        
        let (input_tx, output_rx) = pipeline.add_processor(10, Duration::from_secs(1));
        
        // Send test data
        let mut data = HashMap::new();
        data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(25)));
        
        let record = DataRecord {
            id: "test-1".to_string(),
            timestamp: Instant::now(),
            data,
            source: "sensor-1".to_string(),
        };
        
        input_tx.send(record).await.unwrap();
        drop(input_tx);
        
        // Start pipeline
        let handle = tokio::spawn(async move {
            pipeline.start().await
        });
        
        // Wait for processing to complete
        handle.await.unwrap().unwrap();
        
        // Check output
        let processed = output_rx.recv().await.unwrap();
        assert_eq!(processed.id, "test-1");
    }
}
```

### Batch Processing

```rust
// rust/02-batch-processing.rs

/*
Batch processing patterns and best practices
*/

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Batch processor for large datasets.
pub struct BatchProcessor {
    batch_size: usize,
    max_workers: usize,
    data: Arc<RwLock<Vec<DataRecord>>>,
    results: Arc<RwLock<Vec<ProcessedRecord>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    pub id: String,
    pub data: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedRecord {
    pub id: String,
    pub processed_data: HashMap<String, serde_json::Value>,
    pub processing_time: std::time::Duration,
    pub worker_id: usize,
}

impl BatchProcessor {
    pub fn new(batch_size: usize, max_workers: usize) -> Self {
        Self {
            batch_size,
            max_workers,
            data: Arc::new(RwLock::new(Vec::new())),
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Add data to the processor.
    pub async fn add_data(&self, record: DataRecord) {
        let mut data = self.data.write().await;
        data.push(record);
    }
    
    /// Process all data in batches.
    pub async fn process_all(&self) -> Result<Vec<ProcessedRecord>, String> {
        let data = self.data.read().await;
        let total_records = data.len();
        let total_batches = (total_records + self.batch_size - 1) / self.batch_size;
        
        println!("Processing {} records in {} batches", total_records, total_batches);
        
        // Process batches in parallel
        let mut handles = Vec::new();
        
        for batch_index in 0..total_batches {
            let start = batch_index * self.batch_size;
            let end = std::cmp::min(start + self.batch_size, total_records);
            let batch_data = data[start..end].to_vec();
            
            let results = Arc::clone(&self.results);
            let handle = tokio::spawn(async move {
                Self::process_batch(batch_data, batch_index).await
            });
            handles.push(handle);
        }
        
        // Wait for all batches to complete
        for handle in handles {
            let batch_results = handle.await.map_err(|e| format!("Batch processing error: {}", e))?;
            let mut results = self.results.write().await;
            results.extend(batch_results);
        }
        
        let results = self.results.read().await;
        Ok(results.clone())
    }
    
    /// Process a single batch.
    async fn process_batch(batch_data: Vec<DataRecord>, batch_index: usize) -> Vec<ProcessedRecord> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        for (index, record) in batch_data.into_iter().enumerate() {
            let worker_id = batch_index * 1000 + index;
            let processed = Self::process_record(record, worker_id).await;
            results.push(processed);
        }
        
        let processing_time = start_time.elapsed();
        println!("Batch {} processed in {:?}", batch_index, processing_time);
        
        results
    }
    
    /// Process a single record.
    async fn process_record(record: DataRecord, worker_id: usize) -> ProcessedRecord {
        let start_time = std::time::Instant::now();
        
        // Simulate data processing
        let mut processed_data = HashMap::new();
        for (key, value) in record.data {
            // Transform the data
            let transformed_value = Self::transform_value(&key, &value);
            processed_data.insert(key, transformed_value);
        }
        
        let processing_time = start_time.elapsed();
        
        ProcessedRecord {
            id: record.id,
            processed_data,
            processing_time,
            worker_id,
        }
    }
    
    /// Transform a value based on its key.
    fn transform_value(key: &str, value: &serde_json::Value) -> serde_json::Value {
        match key {
            "temperature" => {
                if let Some(temp) = value.as_f64() {
                    // Convert Celsius to Fahrenheit
                    let fahrenheit = (temp * 9.0 / 5.0) + 32.0;
                    serde_json::Value::Number(serde_json::Number::from_f64(fahrenheit).unwrap())
                } else {
                    value.clone()
                }
            }
            "pressure" => {
                if let Some(pressure) = value.as_f64() {
                    // Convert Pa to PSI
                    let psi = pressure * 0.000145038;
                    serde_json::Value::Number(serde_json::Number::from_f64(psi).unwrap())
                } else {
                    value.clone()
                }
            }
            _ => value.clone(),
        }
    }
}

/// Data aggregator for batch results.
pub struct DataAggregator {
    results: Arc<RwLock<Vec<ProcessedRecord>>>,
}

impl DataAggregator {
    pub fn new() -> Self {
        Self {
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Add processed results.
    pub async fn add_results(&self, results: Vec<ProcessedRecord>) {
        let mut all_results = self.results.write().await;
        all_results.extend(results);
    }
    
    /// Get aggregated statistics.
    pub async fn get_statistics(&self) -> ProcessingStatistics {
        let results = self.results.read().await;
        
        let total_records = results.len();
        let total_processing_time: std::time::Duration = results
            .iter()
            .map(|r| r.processing_time)
            .sum();
        
        let avg_processing_time = if total_records > 0 {
            total_processing_time / total_records as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let worker_counts: HashMap<usize, usize> = results
            .iter()
            .map(|r| r.worker_id)
            .fold(HashMap::new(), |mut acc, id| {
                *acc.entry(id).or_insert(0) += 1;
                acc
            });
        
        ProcessingStatistics {
            total_records,
            total_processing_time,
            avg_processing_time,
            worker_counts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    pub total_records: usize,
    pub total_processing_time: std::time::Duration,
    pub avg_processing_time: std::time::Duration,
    pub worker_counts: HashMap<usize, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_batch_processor() {
        let processor = BatchProcessor::new(10, 4);
        
        // Add test data
        for i in 0..25 {
            let mut data = HashMap::new();
            data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(20 + i)));
            data.insert("pressure".to_string(), serde_json::Value::Number(serde_json::Number::from(100000 + i * 1000)));
            
            let record = DataRecord {
                id: format!("test-{}", i),
                data,
                metadata: HashMap::new(),
            };
            
            processor.add_data(record).await;
        }
        
        // Process all data
        let results = processor.process_all().await.unwrap();
        assert_eq!(results.len(), 25);
    }
    
    #[tokio::test]
    async fn test_data_aggregator() {
        let aggregator = DataAggregator::new();
        
        let mut results = Vec::new();
        for i in 0..10 {
            let processed = ProcessedRecord {
                id: format!("test-{}", i),
                processed_data: HashMap::new(),
                processing_time: std::time::Duration::from_millis(100),
                worker_id: i % 3,
            };
            results.push(processed);
        }
        
        aggregator.add_results(results).await;
        
        let stats = aggregator.get_statistics().await;
        assert_eq!(stats.total_records, 10);
    }
}
```

### Data Validation

```rust
// rust/03-data-validation.rs

/*
Data validation patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Data validation error types.
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Field '{field}' is required")]
    RequiredField { field: String },
    
    #[error("Field '{field}' has invalid format: {message}")]
    InvalidFormat { field: String, message: String },
    
    #[error("Field '{field}' is out of range: {min} <= {value} <= {max}")]
    OutOfRange { field: String, value: f64, min: f64, max: f64 },
    
    #[error("Field '{field}' has invalid type: expected {expected}, got {actual}")]
    InvalidType { field: String, expected: String, actual: String },
}

/// Data validator trait.
pub trait DataValidator {
    fn validate(&self, data: &HashMap<String, serde_json::Value>) -> Result<(), Vec<ValidationError>>;
}

/// Temperature validator.
pub struct TemperatureValidator {
    min_temp: f64,
    max_temp: f64,
}

impl TemperatureValidator {
    pub fn new(min_temp: f64, max_temp: f64) -> Self {
        Self { min_temp, max_temp }
    }
}

impl DataValidator for TemperatureValidator {
    fn validate(&self, data: &HashMap<String, serde_json::Value>) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(temp_value) = data.get("temperature") {
            if let Some(temp) = temp_value.as_f64() {
                if temp < self.min_temp || temp > self.max_temp {
                    errors.push(ValidationError::OutOfRange {
                        field: "temperature".to_string(),
                        value: temp,
                        min: self.min_temp,
                        max: self.max_temp,
                    });
                }
            } else {
                errors.push(ValidationError::InvalidType {
                    field: "temperature".to_string(),
                    expected: "number".to_string(),
                    actual: temp_value.type_str().to_string(),
                });
            }
        } else {
            errors.push(ValidationError::RequiredField {
                field: "temperature".to_string(),
            });
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Pressure validator.
pub struct PressureValidator {
    min_pressure: f64,
    max_pressure: f64,
}

impl PressureValidator {
    pub fn new(min_pressure: f64, max_pressure: f64) -> Self {
        Self { min_pressure, max_pressure }
    }
}

impl DataValidator for PressureValidator {
    fn validate(&self, data: &HashMap<String, serde_json::Value>) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        if let Some(pressure_value) = data.get("pressure") {
            if let Some(pressure) = pressure_value.as_f64() {
                if pressure < self.min_pressure || pressure > self.max_pressure {
                    errors.push(ValidationError::OutOfRange {
                        field: "pressure".to_string(),
                        value: pressure,
                        min: self.min_pressure,
                        max: self.max_pressure,
                    });
                }
            } else {
                errors.push(ValidationError::InvalidType {
                    field: "pressure".to_string(),
                    expected: "number".to_string(),
                    actual: pressure_value.type_str().to_string(),
                });
            }
        } else {
            errors.push(ValidationError::RequiredField {
                field: "pressure".to_string(),
            });
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Composite validator.
pub struct CompositeValidator {
    validators: Vec<Box<dyn DataValidator>>,
}

impl CompositeValidator {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }
    
    pub fn add_validator(&mut self, validator: Box<dyn DataValidator>) {
        self.validators.push(validator);
    }
}

impl DataValidator for CompositeValidator {
    fn validate(&self, data: &HashMap<String, serde_json::Value>) -> Result<(), Vec<ValidationError>> {
        let mut all_errors = Vec::new();
        
        for validator in &self.validators {
            if let Err(errors) = validator.validate(data) {
                all_errors.extend(errors);
            }
        }
        
        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }
}

/// Data validation service.
pub struct ValidationService {
    validator: CompositeValidator,
}

impl ValidationService {
    pub fn new() -> Self {
        Self {
            validator: CompositeValidator::new(),
        }
    }
    
    pub fn add_validator(&mut self, validator: Box<dyn DataValidator>) {
        self.validator.add_validator(validator);
    }
    
    pub fn validate_data(&self, data: &HashMap<String, serde_json::Value>) -> Result<(), Vec<ValidationError>> {
        self.validator.validate(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temperature_validator() {
        let validator = TemperatureValidator::new(-50.0, 50.0);
        
        let mut data = HashMap::new();
        data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(25)));
        
        let result = validator.validate(&data);
        assert!(result.is_ok());
        
        data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(100)));
        let result = validator.validate(&data);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_pressure_validator() {
        let validator = PressureValidator::new(0.0, 200000.0);
        
        let mut data = HashMap::new();
        data.insert("pressure".to_string(), serde_json::Value::Number(serde_json::Number::from(101325)));
        
        let result = validator.validate(&data);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_composite_validator() {
        let mut validator = CompositeValidator::new();
        validator.add_validator(Box::new(TemperatureValidator::new(-50.0, 50.0)));
        validator.add_validator(Box::new(PressureValidator::new(0.0, 200000.0)));
        
        let mut data = HashMap::new();
        data.insert("temperature".to_string(), serde_json::Value::Number(serde_json::Number::from(25)));
        data.insert("pressure".to_string(), serde_json::Value::Number(serde_json::Number::from(101325)));
        
        let result = validator.validate(&data);
        assert!(result.is_ok());
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Stream processing
let (input_tx, input_rx) = mpsc::channel(10);
let (output_tx, output_rx) = mpsc::channel(10);
let processor = StreamProcessor::new(input_rx, output_tx, 100, Duration::from_secs(1));

// 2. Batch processing
let processor = BatchProcessor::new(1000, 4);
processor.add_data(record).await;
let results = processor.process_all().await?;

// 3. Data validation
let mut validator = CompositeValidator::new();
validator.add_validator(Box::new(TemperatureValidator::new(-50.0, 50.0)));
validator.validate(&data)?;
```

### Essential Patterns

```rust
// Complete data processing setup
pub fn setup_rust_data_processing() {
    // 1. Stream processing
    // 2. Batch processing
    // 3. Data validation
    // 4. Error handling
    // 5. Memory management
    // 6. Parallel processing
    // 7. Data transformation
    // 8. Performance optimization
    
    println!("Rust data processing setup complete!");
}
```

---

*This guide provides the complete machinery for Rust data processing. Each pattern includes implementation examples, data processing strategies, and real-world usage patterns for enterprise data pipelines.*
