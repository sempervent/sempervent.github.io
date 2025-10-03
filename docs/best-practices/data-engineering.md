# Data Engineering Best Practices

This document establishes production-ready data engineering patterns for geospatial systems, covering ETL pipeline design, real-time processing, and data quality assurance.

## ETL Pipeline Design

### Modern Airflow DAGs

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'geospatial_etl_pipeline',
    default_args=default_args,
    description='Geospatial ETL pipeline with data quality checks',
    schedule_interval=timedelta(hours=6),
    max_active_runs=1,
    tags=['geospatial', 'etl', 'data-quality']
)

def extract_geospatial_data(**context):
    """
    Extract geospatial data from multiple sources
    """
    # Extract from various sources
    sources = [
        's3://data-lake/raw/geospatial/',
        'postgresql://source-db/geospatial_data',
        'https://api.example.com/geospatial'
    ]
    
    extracted_data = []
    for source in sources:
        # Implementation for each source
        data = extract_from_source(source)
        extracted_data.append(data)
    
    return extracted_data

def transform_geospatial_data(**context):
    """
    Transform and clean geospatial data
    """
    extracted_data = context['task_instance'].xcom_pull(task_ids='extract_data')
    
    transformed_data = []
    for data in extracted_data:
        # Apply transformations
        cleaned_data = clean_geospatial_data(data)
        standardized_data = standardize_crs(cleaned_data)
        transformed_data.append(standardized_data)
    
    return transformed_data

def load_to_warehouse(**context):
    """
    Load transformed data to data warehouse
    """
    transformed_data = context['task_instance'].xcom_pull(task_ids='transform_data')
    
    # Load to data warehouse
    for data in transformed_data:
        load_to_postgres(data)
        load_to_s3(data)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_geospatial_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_geospatial_data,
    dag=dag
)

quality_check = DockerOperator(
    task_id='data_quality_check',
    image='geospatial-quality-checker:latest',
    command=['python', 'quality_check.py'],
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_to_warehouse,
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> quality_check >> load_task
```

**Why:** Airflow provides workflow orchestration with dependency management, retry logic, and monitoring. Docker operators enable consistent execution environments across different data processing tasks.

### Data Quality Framework

```python
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Any
import logging

class DataQualityFramework:
    """
    Comprehensive data quality framework for geospatial data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_rules = {}
    
    def add_quality_rule(self, rule_name: str, rule_function, severity: str = 'error'):
        """
        Add a data quality rule
        """
        self.quality_rules[rule_name] = {
            'function': rule_function,
            'severity': severity
        }
    
    def validate_data(self, data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate data against all quality rules
        """
        validation_results = {
            'total_records': len(data),
            'passed_rules': 0,
            'failed_rules': 0,
            'errors': [],
            'warnings': []
        }
        
        for rule_name, rule_config in self.quality_rules.items():
            try:
                result = rule_config['function'](data)
                
                if result['passed']:
                    validation_results['passed_rules'] += 1
                else:
                    validation_results['failed_rules'] += 1
                    
                    if rule_config['severity'] == 'error':
                        validation_results['errors'].append({
                            'rule': rule_name,
                            'message': result['message']
                        })
                    else:
                        validation_results['warnings'].append({
                            'rule': rule_name,
                            'message': result['message']
                        })
                        
            except Exception as e:
                self.logger.error(f"Error executing rule {rule_name}: {str(e)}")
                validation_results['errors'].append({
                    'rule': rule_name,
                    'message': f"Rule execution failed: {str(e)}"
                })
        
        return validation_results

# Define quality rules
def check_geometry_validity(data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Check if all geometries are valid"""
    invalid_count = (~data.geometry.is_valid).sum()
    
    return {
        'passed': invalid_count == 0,
        'message': f"Found {invalid_count} invalid geometries" if invalid_count > 0 else "All geometries are valid"
    }

def check_crs_consistency(data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Check if CRS is defined and consistent"""
    crs_defined = data.crs is not None
    
    return {
        'passed': crs_defined,
        'message': "CRS is not defined" if not crs_defined else "CRS is properly defined"
    }

def check_spatial_bounds(data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Check if spatial bounds are reasonable"""
    bounds = data.total_bounds
    reasonable_bounds = (
        -180 <= bounds[0] <= 180 and  # minx
        -90 <= bounds[1] <= 90 and    # miny
        -180 <= bounds[2] <= 180 and # maxx
        -90 <= bounds[3] <= 90       # maxy
    )
    
    return {
        'passed': reasonable_bounds,
        'message': f"Spatial bounds are unreasonable: {bounds}" if not reasonable_bounds else "Spatial bounds are reasonable"
    }

# Usage example
quality_framework = DataQualityFramework()
quality_framework.add_quality_rule('geometry_validity', check_geometry_validity, 'error')
quality_framework.add_quality_rule('crs_consistency', check_crs_consistency, 'error')
quality_framework.add_quality_rule('spatial_bounds', check_spatial_bounds, 'warning')

# Validate data
validation_results = quality_framework.validate_data(your_geodataframe)
```

**Why:** Automated data quality checks prevent downstream errors and ensure data reliability. Configurable rules enable domain-specific validation requirements.

## Real-time Data Processing

### Kafka Stream Processing

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

class GeospatialStreamProcessor:
    def __init__(self, kafka_config, postgres_config):
        self.consumer = KafkaConsumer(
            'geospatial-events',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='geospatial-processor'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.postgres_config = postgres_config
    
    def process_stream(self):
        """
        Process geospatial data stream in real-time
        """
        for message in self.consumer:
            try:
                # Parse incoming data
                event_data = message.value
                
                # Process geospatial event
                processed_data = self.process_geospatial_event(event_data)
                
                # Store in database
                self.store_processed_data(processed_data)
                
                # Send to output topic
                self.producer.send('processed-geospatial', processed_data)
                
            except Exception as e:
                print(f"Error processing message: {e}")
                # Send to error topic
                self.producer.send('geospatial-errors', {
                    'original_message': event_data,
                    'error': str(e)
                })
    
    def process_geospatial_event(self, event_data):
        """
        Process individual geospatial event
        """
        # Extract coordinates
        lat = event_data.get('latitude')
        lon = event_data.get('longitude')
        
        if lat is None or lon is None:
            raise ValueError("Missing coordinates")
        
        # Create point geometry
        point = Point(lon, lat)
        
        # Perform spatial analysis
        spatial_analysis = self.perform_spatial_analysis(point, event_data)
        
        # Combine with original data
        processed_data = {
            **event_data,
            'geometry': point.wkt,
            'spatial_analysis': spatial_analysis,
            'processed_at': pd.Timestamp.now().isoformat()
        }
        
        return processed_data
    
    def perform_spatial_analysis(self, point, event_data):
        """
        Perform spatial analysis on point
        """
        # Implementation for spatial analysis
        return {
            'buffer_analysis': self.buffer_analysis(point),
            'proximity_analysis': self.proximity_analysis(point),
            'spatial_join': self.spatial_join_analysis(point)
        }
    
    def store_processed_data(self, data):
        """
        Store processed data in database
        """
        # Implementation for database storage
        pass
```

**Why:** Stream processing enables real-time geospatial analysis with low latency. Kafka provides reliable message delivery and horizontal scaling for high-throughput data streams.

### Apache Spark for Large-scale Processing

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import geopandas as gpd
from shapely.geometry import Point

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GeospatialDataProcessing") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

def process_large_geospatial_dataset(spark, input_path, output_path):
    """
    Process large geospatial dataset using Spark
    """
    # Read data
    df = spark.read.parquet(input_path)
    
    # Define UDF for spatial operations
    def calculate_distance(lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt
        
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in kilometers
        return c * r
    
    distance_udf = udf(calculate_distance, DoubleType())
    
    # Process data
    processed_df = df.withColumn(
        "distance_to_center",
        distance_udf(col("latitude"), col("longitude"), 0.0, 0.0)
    ).withColumn(
        "region",
        when(col("distance_to_center") < 100, "central")
        .when(col("distance_to_center") < 500, "regional")
        .otherwise("remote")
    )
    
    # Write results
    processed_df.write.mode("overwrite").parquet(output_path)
    
    return processed_df

# Usage
spark_session = SparkSession.builder.appName("GeospatialProcessing").getOrCreate()
result = process_large_geospatial_dataset(
    spark_session,
    "s3://data-lake/raw/geospatial/",
    "s3://data-lake/processed/geospatial/"
)
```

**Why:** Apache Spark provides distributed processing for large geospatial datasets. Adaptive query execution optimizes performance based on data characteristics and cluster resources.

## Data Pipeline Monitoring

### Pipeline Health Monitoring

```python
import logging
import time
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    start_time: datetime
    end_time: datetime
    records_processed: int
    records_failed: int
    processing_time: float
    data_quality_score: float

class PipelineMonitor:
    """
    Monitor data pipeline execution and health
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
    
    def track_pipeline_execution(self, pipeline_name: str):
        """
        Decorator to track pipeline execution
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_datetime = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Calculate metrics
                    end_time = time.time()
                    end_datetime = datetime.now()
                    
                    metrics = PipelineMetrics(
                        start_time=start_datetime,
                        end_time=end_datetime,
                        records_processed=result.get('records_processed', 0),
                        records_failed=result.get('records_failed', 0),
                        processing_time=end_time - start_time,
                        data_quality_score=result.get('quality_score', 0.0)
                    )
                    
                    self.record_metrics(pipeline_name, metrics)
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Pipeline {pipeline_name} failed: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    def record_metrics(self, pipeline_name: str, metrics: PipelineMetrics):
        """
        Record pipeline metrics
        """
        self.metrics_history.append({
            'pipeline': pipeline_name,
            'timestamp': metrics.start_time,
            'metrics': metrics
        })
        
        self.logger.info(f"Pipeline {pipeline_name} completed: {metrics.records_processed} records in {metrics.processing_time:.2f}s")
    
    def get_pipeline_health(self, pipeline_name: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get pipeline health metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if m['pipeline'] == pipeline_name and m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'status': 'no_data', 'message': 'No recent executions'}
        
        # Calculate health metrics
        total_executions = len(recent_metrics)
        successful_executions = len([m for m in recent_metrics if m['metrics'].records_failed == 0])
        success_rate = successful_executions / total_executions
        
        avg_processing_time = sum(m['metrics'].processing_time for m in recent_metrics) / total_executions
        avg_quality_score = sum(m['metrics'].data_quality_score for m in recent_metrics) / total_executions
        
        return {
            'status': 'healthy' if success_rate > 0.95 else 'degraded',
            'success_rate': success_rate,
            'avg_processing_time': avg_processing_time,
            'avg_quality_score': avg_quality_score,
            'total_executions': total_executions
        }

# Usage example
monitor = PipelineMonitor()

@monitor.track_pipeline_execution('geospatial_etl')
def run_geospatial_etl():
    """
    Example ETL pipeline with monitoring
    """
    # Simulate ETL processing
    time.sleep(2)
    
    return {
        'records_processed': 1000,
        'records_failed': 5,
        'quality_score': 0.95
    }
```

**Why:** Pipeline monitoring enables proactive issue detection and performance optimization. Historical metrics provide insights into system behavior and capacity planning.

## Data Lineage and Governance

### Data Lineage Tracking

```python
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class DataLineageNode:
    """Represents a node in the data lineage graph"""
    node_id: str
    node_type: str  # 'source', 'transform', 'sink'
    name: str
    description: str
    metadata: Dict[str, Any]

@dataclass
class DataLineageEdge:
    """Represents an edge in the data lineage graph"""
    source_node: str
    target_node: str
    transformation_type: str
    metadata: Dict[str, Any]

class DataLineageTracker:
    """
    Track data lineage for geospatial data pipelines
    """
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.execution_log = []
    
    def register_source(self, source_id: str, name: str, source_type: str, metadata: Dict[str, Any] = None):
        """
        Register a data source
        """
        self.nodes[source_id] = DataLineageNode(
            node_id=source_id,
            node_type='source',
            name=name,
            description=f"Data source: {name}",
            metadata=metadata or {}
        )
    
    def register_transform(self, transform_id: str, name: str, transform_type: str, metadata: Dict[str, Any] = None):
        """
        Register a data transformation
        """
        self.nodes[transform_id] = DataLineageNode(
            node_id=transform_id,
            node_type='transform',
            name=name,
            description=f"Transformation: {name}",
            metadata=metadata or {}
        )
    
    def register_sink(self, sink_id: str, name: str, sink_type: str, metadata: Dict[str, Any] = None):
        """
        Register a data sink
        """
        self.nodes[sink_id] = DataLineageNode(
            node_id=sink_id,
            node_type='sink',
            name=name,
            description=f"Data sink: {name}",
            metadata=metadata or {}
        )
    
    def add_lineage_edge(self, source_node: str, target_node: str, transformation_type: str, metadata: Dict[str, Any] = None):
        """
        Add a lineage edge between nodes
        """
        self.edges.append(DataLineageEdge(
            source_node=source_node,
            target_node=target_node,
            transformation_type=transformation_type,
            metadata=metadata or {}
        ))
    
    def log_execution(self, node_id: str, execution_time: datetime, status: str, metadata: Dict[str, Any] = None):
        """
        Log node execution
        """
        self.execution_log.append({
            'node_id': node_id,
            'execution_time': execution_time,
            'status': status,
            'metadata': metadata or {}
        })
    
    def get_lineage_graph(self) -> Dict[str, Any]:
        """
        Get the complete lineage graph
        """
        return {
            'nodes': {node_id: {
                'id': node.node_id,
                'type': node.node_type,
                'name': node.name,
                'description': node.description,
                'metadata': node.metadata
            } for node_id, node in self.nodes.items()},
            'edges': [{
                'source': edge.source_node,
                'target': edge.target_node,
                'transformation_type': edge.transformation_type,
                'metadata': edge.metadata
            } for edge in self.edges]
        }
    
    def get_node_lineage(self, node_id: str) -> Dict[str, List[str]]:
        """
        Get lineage for a specific node
        """
        upstream = []
        downstream = []
        
        for edge in self.edges:
            if edge.target_node == node_id:
                upstream.append(edge.source_node)
            elif edge.source_node == node_id:
                downstream.append(edge.target_node)
        
        return {
            'upstream': upstream,
            'downstream': downstream
        }

# Usage example
lineage_tracker = DataLineageTracker()

# Register data sources
lineage_tracker.register_source('s3_raw_data', 'S3 Raw Geospatial Data', 's3', {'bucket': 'data-lake', 'prefix': 'raw/geospatial/'})
lineage_tracker.register_source('postgres_source', 'PostgreSQL Source', 'postgres', {'host': 'source-db', 'database': 'geospatial'})

# Register transformations
lineage_tracker.register_transform('spatial_clean', 'Spatial Data Cleaning', 'data_quality', {'rules': ['geometry_validity', 'crs_consistency']})
lineage_tracker.register_transform('crs_transform', 'CRS Transformation', 'spatial_transform', {'target_crs': 'EPSG:4326'})

# Register sinks
lineage_tracker.register_sink('data_warehouse', 'Data Warehouse', 'postgres', {'host': 'warehouse-db', 'table': 'geospatial_processed'})

# Add lineage edges
lineage_tracker.add_lineage_edge('s3_raw_data', 'spatial_clean', 'extract')
lineage_tracker.add_lineage_edge('postgres_source', 'spatial_clean', 'extract')
lineage_tracker.add_lineage_edge('spatial_clean', 'crs_transform', 'transform')
lineage_tracker.add_lineage_edge('crs_transform', 'data_warehouse', 'load')

# Get lineage graph
lineage_graph = lineage_tracker.get_lineage_graph()
```

**Why:** Data lineage tracking enables impact analysis, compliance reporting, and debugging. Understanding data flow helps optimize pipelines and ensure data quality.
