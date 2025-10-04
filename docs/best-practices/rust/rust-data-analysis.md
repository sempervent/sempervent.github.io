# Rust Data Analysis Best Practices

**Objective**: Master senior-level Rust data analysis patterns for production systems. When you need to analyze large datasets efficiently, when you want to leverage Rust's performance for data analysis, when you need enterprise-grade data analysis—these best practices become your weapon of choice.

## Core Principles

- **Performance**: Leverage Rust's speed for data analysis
- **Memory Efficiency**: Optimize memory usage for large datasets
- **Parallel Processing**: Use multiple cores for data analysis
- **Data Structures**: Choose appropriate data structures for analysis
- **Statistical Computing**: Implement statistical analysis patterns

## Data Analysis Patterns

### Data Loading and Processing

```rust
// rust/01-data-loading.rs

/*
Data loading and processing patterns
*/

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Data loading error types.
#[derive(Error, Debug)]
pub enum DataLoadingError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Invalid CSV format: {0}")]
    InvalidCsv(String),
    
    #[error("Parsing error: {0}")]
    ParsingError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Data record structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub values: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

/// CSV data loader.
pub struct CsvDataLoader {
    file_path: String,
    delimiter: char,
    has_header: bool,
}

impl CsvDataLoader {
    pub fn new(file_path: String, delimiter: char, has_header: bool) -> Self {
        Self {
            file_path,
            delimiter,
            has_header,
        }
    }
    
    /// Load data from CSV file.
    pub async fn load_data(&self) -> Result<Vec<DataRecord>, DataLoadingError> {
        let file = File::open(&self.file_path)
            .map_err(|_| DataLoadingError::FileNotFound(self.file_path.clone()))?;
        
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header if present
        if self.has_header {
            lines.next();
        }
        
        let mut records = Vec::new();
        let mut record_id = 0;
        
        for line in lines {
            let line = line.map_err(DataLoadingError::IoError)?;
            let record = self.parse_line(&line, record_id)?;
            records.push(record);
            record_id += 1;
        }
        
        Ok(records)
    }
    
    /// Parse a single line of CSV data.
    fn parse_line(&self, line: &str, record_id: usize) -> Result<DataRecord, DataLoadingError> {
        let fields: Vec<&str> = line.split(self.delimiter).collect();
        
        if fields.len() < 3 {
            return Err(DataLoadingError::InvalidCsv(
                format!("Expected at least 3 fields, got {}", fields.len())
            ));
        }
        
        let mut values = HashMap::new();
        let mut metadata = HashMap::new();
        
        // Parse timestamp (assuming first field is timestamp)
        let timestamp = chrono::DateTime::parse_from_rfc3339(fields[0])
            .map_err(|e| DataLoadingError::ParsingError(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&chrono::Utc);
        
        // Parse numeric values
        for (i, field) in fields.iter().enumerate().skip(1) {
            if let Ok(value) = field.parse::<f64>() {
                values.insert(format!("field_{}", i), value);
            } else {
                metadata.insert(format!("field_{}", i), field.to_string());
            }
        }
        
        Ok(DataRecord {
            id: format!("record_{}", record_id),
            timestamp,
            values,
            metadata,
        })
    }
}

/// Data aggregator for analysis.
pub struct DataAggregator {
    data: Vec<DataRecord>,
    statistics: HashMap<String, StatisticalSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub quartiles: (f64, f64, f64), // Q1, Q2, Q3
}

impl DataAggregator {
    pub fn new(data: Vec<DataRecord>) -> Self {
        Self {
            data,
            statistics: HashMap::new(),
        }
    }
    
    /// Calculate statistics for all numeric fields.
    pub async fn calculate_statistics(&mut self) -> Result<(), String> {
        if self.data.is_empty() {
            return Err("No data to analyze".to_string());
        }
        
        // Get all numeric field names
        let field_names: Vec<String> = self.data[0]
            .values
            .keys()
            .cloned()
            .collect();
        
        // Calculate statistics for each field
        for field_name in field_names {
            let values: Vec<f64> = self.data
                .iter()
                .filter_map(|record| record.values.get(&field_name))
                .cloned()
                .collect();
            
            if !values.is_empty() {
                let summary = self.calculate_field_statistics(&values);
                self.statistics.insert(field_name, summary);
            }
        }
        
        Ok(())
    }
    
    /// Calculate statistics for a specific field.
    fn calculate_field_statistics(&self, values: &[f64]) -> StatisticalSummary {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        
        let median = if count % 2 == 0 {
            (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
        } else {
            sorted_values[count / 2]
        };
        
        let variance: f64 = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        let min = sorted_values[0];
        let max = sorted_values[count - 1];
        
        let q1 = if count >= 4 {
            sorted_values[count / 4]
        } else {
            min
        };
        
        let q3 = if count >= 4 {
            sorted_values[3 * count / 4]
        } else {
            max
        };
        
        StatisticalSummary {
            count,
            mean,
            median,
            std_dev,
            min,
            max,
            quartiles: (q1, median, q3),
        }
    }
    
    /// Get statistics for a specific field.
    pub fn get_field_statistics(&self, field_name: &str) -> Option<&StatisticalSummary> {
        self.statistics.get(field_name)
    }
    
    /// Get all statistics.
    pub fn get_all_statistics(&self) -> &HashMap<String, StatisticalSummary> {
        &self.statistics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_aggregator() {
        let mut data = Vec::new();
        
        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("temperature".to_string(), 20.0 + i as f64);
            values.insert("pressure".to_string(), 101325.0 + i as f64 * 1000.0);
            
            let record = DataRecord {
                id: format!("record_{}", i),
                timestamp: chrono::Utc::now(),
                values,
                metadata: HashMap::new(),
            };
            
            data.push(record);
        }
        
        let mut aggregator = DataAggregator::new(data);
        aggregator.calculate_statistics().await.unwrap();
        
        let stats = aggregator.get_field_statistics("temperature");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 10);
    }
}
```

### Statistical Analysis

```rust
// rust/02-statistical-analysis.rs

/*
Statistical analysis patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Statistical analyzer for data analysis.
pub struct StatisticalAnalyzer {
    data: Vec<f64>,
    statistics: Option<StatisticalSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub quartiles: (f64, f64, f64), // Q1, Q2, Q3
    pub iqr: f64, // Interquartile range
    pub skewness: f64,
    pub kurtosis: f64,
}

impl StatisticalAnalyzer {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            statistics: None,
        }
    }
    
    /// Calculate comprehensive statistics.
    pub fn calculate_statistics(&mut self) -> Result<&StatisticalSummary, String> {
        if self.data.is_empty() {
            return Err("No data to analyze".to_string());
        }
        
        let mut sorted_data = self.data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let count = self.data.len();
        let sum: f64 = self.data.iter().sum();
        let mean = sum / count as f64;
        
        let median = if count % 2 == 0 {
            (sorted_data[count / 2 - 1] + sorted_data[count / 2]) / 2.0
        } else {
            sorted_data[count / 2]
        };
        
        let mode = self.calculate_mode(&sorted_data);
        
        let variance: f64 = self.data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        let min = sorted_data[0];
        let max = sorted_data[count - 1];
        let range = max - min;
        
        let q1 = if count >= 4 {
            sorted_data[count / 4]
        } else {
            min
        };
        
        let q3 = if count >= 4 {
            sorted_data[3 * count / 4]
        } else {
            max
        };
        
        let iqr = q3 - q1;
        
        let skewness = self.calculate_skewness(&self.data, mean, std_dev);
        let kurtosis = self.calculate_kurtosis(&self.data, mean, std_dev);
        
        self.statistics = Some(StatisticalSummary {
            count,
            mean,
            median,
            mode,
            std_dev,
            variance,
            min,
            max,
            range,
            quartiles: (q1, median, q3),
            iqr,
            skewness,
            kurtosis,
        });
        
        Ok(self.statistics.as_ref().unwrap())
    }
    
    /// Calculate mode (most frequent value).
    fn calculate_mode(&self, sorted_data: &[f64]) -> Option<f64> {
        let mut frequency_map = HashMap::new();
        
        for value in sorted_data {
            *frequency_map.entry(*value).or_insert(0) += 1;
        }
        
        let max_frequency = frequency_map.values().max()?;
        let mode = frequency_map
            .iter()
            .find(|(_, &freq)| freq == *max_frequency)
            .map(|(value, _)| *value);
        
        mode
    }
    
    /// Calculate skewness.
    fn calculate_skewness(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let sum_cubed_deviations: f64 = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum();
        
        sum_cubed_deviations / n
    }
    
    /// Calculate kurtosis.
    fn calculate_kurtosis(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let sum_fourth_deviations: f64 = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum();
        
        (sum_fourth_deviations / n) - 3.0
    }
    
    /// Get statistics.
    pub fn get_statistics(&self) -> Option<&StatisticalSummary> {
        self.statistics.as_ref()
    }
    
    /// Get data summary.
    pub fn get_summary(&self) -> String {
        if let Some(stats) = &self.statistics {
            format!(
                "Count: {}, Mean: {:.2}, Median: {:.2}, Std Dev: {:.2}, Min: {:.2}, Max: {:.2}",
                stats.count, stats.mean, stats.median, stats.std_dev, stats.min, stats.max
            )
        } else {
            "No statistics calculated".to_string()
        }
    }
}

/// Correlation analyzer.
pub struct CorrelationAnalyzer {
    data: Vec<(f64, f64)>,
    correlation: Option<f64>,
}

impl CorrelationAnalyzer {
    pub fn new(data: Vec<(f64, f64)>) -> Self {
        Self {
            data,
            correlation: None,
        }
    }
    
    /// Calculate Pearson correlation coefficient.
    pub fn calculate_correlation(&mut self) -> Result<f64, String> {
        if self.data.len() < 2 {
            return Err("Need at least 2 data points for correlation".to_string());
        }
        
        let n = self.data.len() as f64;
        
        // Calculate means
        let x_mean: f64 = self.data.iter().map(|(x, _)| x).sum::<f64>() / n;
        let y_mean: f64 = self.data.iter().map(|(_, y)| y).sum::<f64>() / n;
        
        // Calculate correlation coefficient
        let numerator: f64 = self.data
            .iter()
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        let x_variance: f64 = self.data
            .iter()
            .map(|(x, _)| (x - x_mean).powi(2))
            .sum::<f64>();
        
        let y_variance: f64 = self.data
            .iter()
            .map(|(_, y)| (y - y_mean).powi(2))
            .sum::<f64>();
        
        let denominator = (x_variance * y_variance).sqrt();
        
        if denominator == 0.0 {
            return Err("Cannot calculate correlation: zero variance".to_string());
        }
        
        let correlation = numerator / denominator;
        self.correlation = Some(correlation);
        
        Ok(correlation)
    }
    
    /// Get correlation coefficient.
    pub fn get_correlation(&self) -> Option<f64> {
        self.correlation
    }
    
    /// Interpret correlation strength.
    pub fn interpret_correlation(&self) -> String {
        if let Some(corr) = self.correlation {
            let abs_corr = corr.abs();
            if abs_corr >= 0.9 {
                "Very strong correlation".to_string()
            } else if abs_corr >= 0.7 {
                "Strong correlation".to_string()
            } else if abs_corr >= 0.5 {
                "Moderate correlation".to_string()
            } else if abs_corr >= 0.3 {
                "Weak correlation".to_string()
            } else {
                "Very weak correlation".to_string()
            }
        } else {
            "No correlation calculated".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistical_analyzer() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut analyzer = StatisticalAnalyzer::new(data);
        
        let stats = analyzer.calculate_statistics().unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
    }
    
    #[test]
    fn test_correlation_analyzer() {
        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];
        let mut analyzer = CorrelationAnalyzer::new(data);
        
        let correlation = analyzer.calculate_correlation().unwrap();
        assert!((correlation - 1.0).abs() < 0.001); // Perfect positive correlation
    }
}
```

### Data Visualization

```rust
// rust/03-data-visualization.rs

/*
Data visualization patterns and best practices
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Data visualization service.
pub struct DataVisualizationService {
    data: Vec<DataPoint>,
    charts: Vec<Chart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
    pub color: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    pub id: String,
    pub title: String,
    pub chart_type: ChartType,
    pub data: Vec<DataPoint>,
    pub x_label: String,
    pub y_label: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Scatter,
    Histogram,
    BoxPlot,
}

impl DataVisualizationService {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            charts: Vec::new(),
        }
    }
    
    /// Add data points.
    pub fn add_data(&mut self, data: Vec<DataPoint>) {
        self.data.extend(data);
    }
    
    /// Create a line chart.
    pub fn create_line_chart(&mut self, id: String, title: String, data: Vec<DataPoint>) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot create chart with empty data".to_string());
        }
        
        let chart = Chart {
            id,
            title,
            chart_type: ChartType::Line,
            data,
            x_label: "X Axis".to_string(),
            y_label: "Y Axis".to_string(),
            width: 800,
            height: 600,
        };
        
        self.charts.push(chart);
        Ok(())
    }
    
    /// Create a bar chart.
    pub fn create_bar_chart(&mut self, id: String, title: String, data: Vec<DataPoint>) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot create chart with empty data".to_string());
        }
        
        let chart = Chart {
            id,
            title,
            chart_type: ChartType::Bar,
            data,
            x_label: "Categories".to_string(),
            y_label: "Values".to_string(),
            width: 800,
            height: 600,
        };
        
        self.charts.push(chart);
        Ok(())
    }
    
    /// Create a scatter plot.
    pub fn create_scatter_plot(&mut self, id: String, title: String, data: Vec<DataPoint>) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot create chart with empty data".to_string());
        }
        
        let chart = Chart {
            id,
            title,
            chart_type: ChartType::Scatter,
            data,
            x_label: "X Values".to_string(),
            y_label: "Y Values".to_string(),
            width: 800,
            height: 600,
        };
        
        self.charts.push(chart);
        Ok(())
    }
    
    /// Create a histogram.
    pub fn create_histogram(&mut self, id: String, title: String, data: Vec<f64>, bins: usize) -> Result<(), String> {
        if data.is_empty() {
            return Err("Cannot create histogram with empty data".to_string());
        }
        
        let histogram_data = self.create_histogram_data(data, bins);
        
        let chart = Chart {
            id,
            title,
            chart_type: ChartType::Histogram,
            data: histogram_data,
            x_label: "Values".to_string(),
            y_label: "Frequency".to_string(),
            width: 800,
            height: 600,
        };
        
        self.charts.push(chart);
        Ok(())
    }
    
    /// Create histogram data.
    fn create_histogram_data(&self, data: Vec<f64>, bins: usize) -> Vec<DataPoint> {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max - min) / bins as f64;
        
        let mut histogram = vec![0; bins];
        
        for value in data {
            let bin_index = ((value - min) / bin_width) as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }
        
        let mut histogram_data = Vec::new();
        for (i, count) in histogram.iter().enumerate() {
            let x = min + (i as f64 + 0.5) * bin_width;
            histogram_data.push(DataPoint {
                x,
                y: *count as f64,
                label: None,
                color: None,
            });
        }
        
        histogram_data
    }
    
    /// Get all charts.
    pub fn get_charts(&self) -> &[Chart] {
        &self.charts
    }
    
    /// Get a specific chart.
    pub fn get_chart(&self, id: &str) -> Option<&Chart> {
        self.charts.iter().find(|chart| chart.id == id)
    }
    
    /// Export chart data as JSON.
    pub fn export_chart_json(&self, id: &str) -> Result<String, String> {
        let chart = self.get_chart(id)
            .ok_or_else(|| "Chart not found".to_string())?;
        
        serde_json::to_string_pretty(chart)
            .map_err(|e| format!("Failed to serialize chart: {}", e))
    }
    
    /// Export all charts as JSON.
    pub fn export_all_charts_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.charts)
            .map_err(|e| format!("Failed to serialize charts: {}", e))
    }
}

/// Data analysis report generator.
pub struct DataAnalysisReport {
    title: String,
    sections: Vec<ReportSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub content: String,
    pub charts: Vec<String>, // Chart IDs
}

impl DataAnalysisReport {
    pub fn new(title: String) -> Self {
        Self {
            title,
            sections: Vec::new(),
        }
    }
    
    /// Add a section to the report.
    pub fn add_section(&mut self, title: String, content: String, charts: Vec<String>) {
        self.sections.push(ReportSection {
            title,
            content,
            charts,
        });
    }
    
    /// Generate HTML report.
    pub fn generate_html(&self) -> String {
        let mut html = String::new();
        
        html.push_str(&format!("<html><head><title>{}</title></head><body>", self.title));
        html.push_str(&format!("<h1>{}</h1>", self.title));
        
        for section in &self.sections {
            html.push_str(&format!("<h2>{}</h2>", section.title));
            html.push_str(&format!("<p>{}</p>", section.content));
            
            if !section.charts.is_empty() {
                html.push_str("<div class=\"charts\">");
                for chart_id in &section.charts {
                    html.push_str(&format!("<div class=\"chart\" id=\"{}\"></div>", chart_id));
                }
                html.push_str("</div>");
            }
        }
        
        html.push_str("</body></html>");
        html
    }
    
    /// Generate Markdown report.
    pub fn generate_markdown(&self) -> String {
        let mut markdown = String::new();
        
        markdown.push_str(&format!("# {}\n\n", self.title));
        
        for section in &self.sections {
            markdown.push_str(&format!("## {}\n\n", section.title));
            markdown.push_str(&format!("{}\n\n", section.content));
            
            if !section.charts.is_empty() {
                markdown.push_str("### Charts\n\n");
                for chart_id in &section.charts {
                    markdown.push_str(&format!("- Chart: {}\n", chart_id));
                }
                markdown.push_str("\n");
            }
        }
        
        markdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_visualization_service() {
        let mut service = DataVisualizationService::new();
        
        let data = vec![
            DataPoint { x: 1.0, y: 2.0, label: None, color: None },
            DataPoint { x: 2.0, y: 4.0, label: None, color: None },
            DataPoint { x: 3.0, y: 6.0, label: None, color: None },
        ];
        
        service.create_line_chart("chart1".to_string(), "Test Chart".to_string(), data).unwrap();
        
        assert_eq!(service.get_charts().len(), 1);
    }
    
    #[test]
    fn test_data_analysis_report() {
        let mut report = DataAnalysisReport::new("Test Report".to_string());
        
        report.add_section(
            "Introduction".to_string(),
            "This is a test report.".to_string(),
            vec!["chart1".to_string()],
        );
        
        let html = report.generate_html();
        assert!(html.contains("Test Report"));
        assert!(html.contains("Introduction"));
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Data loading
let loader = CsvDataLoader::new("data.csv".to_string(), ',', true);
let data = loader.load_data().await?;

// 2. Statistical analysis
let mut analyzer = StatisticalAnalyzer::new(values);
let stats = analyzer.calculate_statistics()?;

// 3. Data visualization
let mut service = DataVisualizationService::new();
service.create_line_chart("chart1".to_string(), "Title".to_string(), data)?;
```

### Essential Patterns

```rust
// Complete data analysis setup
pub fn setup_rust_data_analysis() {
    // 1. Data loading
    // 2. Statistical analysis
    // 3. Data visualization
    // 4. Report generation
    // 5. Performance optimization
    // 6. Memory management
    // 7. Parallel processing
    // 8. Error handling
    
    println!("Rust data analysis setup complete!");
}
```

---

*This guide provides the complete machinery for Rust data analysis. Each pattern includes implementation examples, analysis strategies, and real-world usage patterns for enterprise data analysis.*
