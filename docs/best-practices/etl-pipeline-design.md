# ETL Pipeline Design

**Objective**: Build robust, scalable ETL pipelines for geospatial data processing using modern orchestration tools.

ETL pipelines are the backbone of data engineering. This guide covers designing production-ready ETL systems that handle geospatial data at scale with reliability and performance.

## 1) Modern ETL Architecture

### Airflow-Based Orchestration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.s3 import S3FileTransformOperator
from datetime import datetime, timedelta
import geopandas as gpd

# Define DAG
dag = DAG(
    'geospatial_etl_pipeline',
    default_args={
        'owner': 'data_team',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'retry_exponential_backoff': True,
        'max_retry_delay': timedelta(hours=1)
    },
    schedule_interval=timedelta(hours=6),
    catchup=False,
    max_active_runs=1,
    tags=['geospatial', 'etl']
)

# Extract task
def extract_geospatial_data(**context):
    """Extract geospatial data from various sources"""
    sources = [
        's3://raw-data/geojson/',
        's3://raw-data/shapefiles/',
        'https://api.example.com/geospatial'
    ]
    
    extracted_data = []
    for source in sources:
        if source.startswith('s3://'):
            # Extract from S3
            data = extract_from_s3(source)
        else:
            # Extract from API
            data = extract_from_api(source)
        
        extracted_data.append(data)
    
    return extracted_data

extract_task = PythonOperator(
    task_id='extract_geospatial_data',
    python_callable=extract_geospatial_data,
    dag=dag
)
```

**Why**: Airflow provides reliable orchestration with built-in retry logic, monitoring, and dependency management for complex ETL workflows.

### Data Quality Validation

```python
def validate_geospatial_data(**context):
    """Validate geospatial data quality"""
    data = context['task_instance'].xcom_pull(task_ids='extract_geospatial_data')
    
    validation_results = []
    for dataset in data:
        # Check geometry validity
        invalid_geometries = (~dataset['geometry'].is_valid).sum()
        
        # Check for null geometries
        null_geometries = dataset['geometry'].isna().sum()
        
        # Check coordinate reference system
        crs_valid = dataset.crs is not None
        
        validation_results.append({
            'dataset': dataset.name,
            'invalid_geometries': invalid_geometries,
            'null_geometries': null_geometries,
            'crs_valid': crs_valid,
            'total_records': len(dataset)
        })
    
    # Fail if quality thresholds not met
    for result in validation_results:
        if result['invalid_geometries'] > result['total_records'] * 0.05:  # 5% threshold
            raise ValueError(f"Too many invalid geometries: {result['invalid_geometries']}")
    
    return validation_results

validate_task = PythonOperator(
    task_id='validate_geospatial_data',
    python_callable=validate_geospatial_data,
    dag=dag
)
```

**Why**: Data quality validation prevents downstream issues and ensures reliable analytics results.

## 2) Transformation Patterns

### Spatial Transformations

```python
def transform_geospatial_data(**context):
    """Transform geospatial data"""
    data = context['task_instance'].xcom_pull(task_ids='extract_geospatial_data')
    
    transformed_datasets = []
    for dataset in data:
        # Standardize coordinate reference system
        dataset = dataset.to_crs('EPSG:4326')
        
        # Add calculated fields
        dataset['area_sqkm'] = dataset.geometry.to_crs('EPSG:3857').area / 1e6
        dataset['centroid_lat'] = dataset.geometry.centroid.y
        dataset['centroid_lon'] = dataset.geometry.centroid.x
        
        # Spatial joins with reference data
        reference_data = load_reference_data()
        dataset = dataset.sjoin(reference_data, how='left', predicate='intersects')
        
        # Clean and standardize attributes
        dataset = clean_attributes(dataset)
        
        transformed_datasets.append(dataset)
    
    return transformed_datasets

def clean_attributes(df):
    """Clean and standardize attributes"""
    # Remove leading/trailing whitespace
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].str.strip()
    
    # Standardize case
    df['name'] = df['name'].str.title()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['geometry'])
    
    return df
```

**Why**: Consistent transformations ensure data quality and enable reliable analytics across different data sources.

### Performance Optimization

```python
def optimize_geospatial_processing(df):
    """Optimize geospatial data processing"""
    # Use spatial index for joins
    df = df.set_geometry('geometry')
    
    # Simplify geometries for performance
    df['geometry'] = df['geometry'].simplify(tolerance=0.001)
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Partition by spatial regions
    df['spatial_partition'] = create_spatial_partitions(df)
    
    return df

def optimize_dtypes(df):
    """Optimize data types for memory efficiency"""
    # Convert to categorical for repeated values
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Convert to appropriate numeric types
    numeric_columns = df.select_dtypes(include=['int64']).columns
    for col in numeric_columns:
        if df[col].min() >= 0 and df[col].max() < 255:
            df[col] = df[col].astype('uint8')
        elif df[col].min() >= -128 and df[col].max() < 127:
            df[col] = df[col].astype('int8')
    
    return df
```

**Why**: Optimization reduces processing time, memory usage, and storage costs while maintaining data quality.

## 3) Loading Strategies

### Incremental Loading

```python
def incremental_load(**context):
    """Implement incremental loading strategy"""
    # Get last successful run
    last_run = context['task_instance'].xcom_pull(
        task_ids='get_last_successful_run'
    )
    
    # Extract only new or modified data
    new_data = extract_incremental_data(
        since_timestamp=last_run,
        source_systems=['api', 's3', 'database']
    )
    
    # Process incremental data
    processed_data = transform_incremental_data(new_data)
    
    # Load to target with upsert logic
    load_incremental_data(processed_data)
    
    return len(processed_data)

def extract_incremental_data(since_timestamp, source_systems):
    """Extract only new or modified data"""
    incremental_data = []
    
    for system in source_systems:
        if system == 'api':
            # API with timestamp filtering
            data = extract_from_api(
                endpoint='https://api.example.com/geospatial',
                params={'modified_since': since_timestamp}
            )
        elif system == 's3':
            # S3 with object metadata filtering
            data = extract_from_s3_incremental(
                bucket='raw-data',
                prefix='geospatial/',
                since_timestamp=since_timestamp
            )
        elif system == 'database':
            # Database with change data capture
            data = extract_from_database_cdc(
                table='geospatial_data',
                since_timestamp=since_timestamp
            )
        
        incremental_data.append(data)
    
    return incremental_data
```

**Why**: Incremental loading reduces processing time and resource usage while ensuring data freshness.

### Batch vs Stream Processing

```python
# Batch processing for large datasets
def batch_process_large_dataset(data):
    """Process large datasets in batches"""
    batch_size = 10000
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        processed_batch = process_batch(batch)
        results.append(processed_batch)
    
    return pd.concat(results, ignore_index=True)

# Stream processing for real-time data
def stream_process_realtime_data(data_stream):
    """Process real-time data streams"""
    for record in data_stream:
        # Process individual record
        processed_record = process_record(record)
        
        # Send to real-time analytics
        send_to_analytics(processed_record)
        
        # Update real-time dashboard
        update_dashboard(processed_record)
```

**Why**: Choosing the right processing pattern optimizes performance and resource usage for different data characteristics.

## 4) Error Handling and Recovery

### Retry and Backoff Strategies

```python
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
import time
import random

def robust_geospatial_processing(**context):
    """Robust geospatial processing with retry logic"""
    max_retries = 3
    base_delay = 60  # seconds
    
    for attempt in range(max_retries):
        try:
            # Attempt processing
            result = process_geospatial_data(context)
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                raise e
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 10)
            time.sleep(delay)
            
            # Log retry attempt
            print(f"Retry attempt {attempt + 1} after {delay} seconds")

# Dead letter queue for failed records
def handle_failed_records(**context):
    """Handle records that failed processing"""
    failed_records = context['task_instance'].xcom_pull(
        task_ids='process_geospatial_data',
        key='failed_records'
    )
    
    # Send to dead letter queue for manual review
    send_to_dead_letter_queue(failed_records)
    
    # Alert data team
    send_alert_to_data_team(failed_records)
```

**Why**: Robust error handling ensures data pipeline reliability and provides visibility into processing issues.

### Data Lineage Tracking

```python
def track_data_lineage(**context):
    """Track data lineage for ETL pipeline"""
    lineage_info = {
        'pipeline_id': context['dag'].dag_id,
        'run_id': context['run_id'],
        'start_time': context['task_instance'].start_date,
        'source_systems': context['task_instance'].xcom_pull(
            task_ids='extract_geospatial_data'
        ),
        'transformations_applied': [
            'coordinate_system_standardization',
            'geometry_validation',
            'attribute_cleaning',
            'spatial_joins'
        ],
        'target_systems': ['data_warehouse', 'analytics_platform'],
        'data_quality_metrics': context['task_instance'].xcom_pull(
            task_ids='validate_geospatial_data'
        )
    }
    
    # Store lineage information
    store_lineage_info(lineage_info)
    
    return lineage_info
```

**Why**: Data lineage provides audit trails, debugging capabilities, and compliance documentation for data processing.

## 5) Monitoring and Observability

### Custom Metrics

```python
def track_etl_metrics(**context):
    """Track ETL pipeline metrics"""
    metrics = {
        'records_processed': context['task_instance'].xcom_pull(
            task_ids='transform_geospatial_data'
        ),
        'processing_duration': context['task_instance'].duration,
        'data_quality_score': calculate_data_quality_score(context),
        'error_rate': calculate_error_rate(context)
    }
    
    # Send metrics to monitoring system
    send_metrics_to_monitoring(metrics)
    
    return metrics

def calculate_data_quality_score(context):
    """Calculate data quality score"""
    validation_results = context['task_instance'].xcom_pull(
        task_ids='validate_geospatial_data'
    )
    
    total_records = sum(r['total_records'] for r in validation_results)
    invalid_records = sum(r['invalid_geometries'] for r in validation_results)
    
    quality_score = (total_records - invalid_records) / total_records
    return quality_score
```

**Why**: Comprehensive monitoring enables proactive issue detection and performance optimization.

### Alerting Strategies

```python
# Airflow alerting configuration
alerting_config = {
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['data-team@company.com'],
    'slack_webhook': 'https://hooks.slack.com/services/...',
    'pagerduty_integration': True
}

# Custom alerting logic
def send_custom_alerts(**context):
    """Send custom alerts based on ETL results"""
    metrics = context['task_instance'].xcom_pull(
        task_ids='track_etl_metrics'
    )
    
    # Alert on data quality issues
    if metrics['data_quality_score'] < 0.95:
        send_alert(
            'Data Quality Alert',
            f"Data quality score: {metrics['data_quality_score']}"
        )
    
    # Alert on performance issues
    if metrics['processing_duration'] > 3600:  # 1 hour
        send_alert(
            'Performance Alert',
            f"Processing duration: {metrics['processing_duration']} seconds"
        )
```

**Why**: Proactive alerting ensures rapid response to ETL issues and maintains data pipeline reliability.

## 6) Performance Optimization

### Parallel Processing

```python
from airflow.operators.subdag import SubDagOperator
from airflow.executors.local_executor import LocalExecutor

def create_parallel_processing_subdag(parent_dag_name, child_dag_name, args):
    """Create parallel processing subdag"""
    dag = DAG(
        f"{parent_dag_name}.{child_dag_name}",
        default_args=args,
        schedule_interval=None
    )
    
    # Create parallel tasks for different data sources
    sources = ['api', 's3', 'database', 'ftp']
    
    for source in sources:
        PythonOperator(
            task_id=f'process_{source}',
            python_callable=process_data_source,
            op_args=[source],
            dag=dag
        )
    
    return dag

# Use parallel processing
parallel_processing = SubDagOperator(
    task_id='parallel_processing',
    subdag=create_parallel_processing_subdag(
        'geospatial_etl_pipeline',
        'parallel_processing',
        dag.default_args
    ),
    executor=LocalExecutor(),
    dag=dag
)
```

**Why**: Parallel processing reduces overall pipeline execution time and improves resource utilization.

### Caching Strategies

```python
def implement_etl_caching(**context):
    """Implement caching for ETL pipeline"""
    cache_key = f"etl_cache_{context['dag'].dag_id}_{context['ds']}"
    
    # Check cache first
    cached_result = get_from_cache(cache_key)
    if cached_result:
        return cached_result
    
    # Process data if not cached
    result = process_geospatial_data(context)
    
    # Cache result
    set_cache(cache_key, result, ttl=3600)  # 1 hour TTL
    
    return result
```

**Why**: Caching reduces redundant processing and improves pipeline performance for repeated operations.

## 7) TL;DR Quickstart

```python
# 1. Define ETL pipeline
dag = DAG('geospatial_etl', default_args=default_args)

# 2. Extract data
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_geospatial_data,
    dag=dag
)

# 3. Transform data
transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_geospatial_data,
    dag=dag
)

# 4. Load data
load_task = PythonOperator(
    task_id='load',
    python_callable=load_geospatial_data,
    dag=dag
)

# 5. Set dependencies
extract_task >> transform_task >> load_task
```

## 8) Anti-Patterns to Avoid

- **Don't ignore data quality validation**—implement comprehensive quality checks
- **Don't skip error handling**—robust error handling is essential for reliability
- **Don't ignore monitoring**—comprehensive observability enables proactive management
- **Don't skip incremental loading**—full reloads are inefficient and costly
- **Don't ignore performance optimization**—poor performance leads to resource waste and delays

**Why**: These anti-patterns lead to unreliable pipelines, poor performance, and increased operational overhead.

---

*This guide provides the foundation for building production-ready ETL pipelines that handle geospatial data at scale with reliability and performance.*
