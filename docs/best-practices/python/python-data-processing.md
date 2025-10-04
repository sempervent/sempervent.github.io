# Python Data Processing Best Practices

**Objective**: Master senior-level Python data processing patterns for production systems. When you need to build scalable data pipelines, when you want to implement efficient ETL workflows, when you need enterprise-grade data processing strategies—these best practices become your weapon of choice.

## Core Principles

- **Scalability**: Process large datasets efficiently
- **Reliability**: Ensure data integrity and fault tolerance
- **Performance**: Optimize for speed and resource usage
- **Monitoring**: Track processing metrics and errors
- **Flexibility**: Support various data formats and sources

## ETL Pipeline Design

### Data Pipeline Architecture

```python
# python/01-etl-pipeline-design.py

"""
ETL pipeline design patterns and data processing architecture
"""

from typing import List, Dict, Any, Optional, Callable, Iterator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    CLEAN = "clean"

class DataSource(Enum):
    """Data source enumeration"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    CACHE = "cache"

@dataclass
class DataRecord:
    """Data record definition"""
    id: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

class DataExtractor(ABC):
    """Abstract data extractor"""
    
    @abstractmethod
    async def extract(self, source: str, **kwargs) -> Iterator[DataRecord]:
        """Extract data from source"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get data schema"""
        pass

class FileExtractor(DataExtractor):
    """File-based data extractor"""
    
    def __init__(self, file_format: str = "json"):
        self.file_format = file_format
        self.schema = {}
    
    async def extract(self, source: str, **kwargs) -> Iterator[DataRecord]:
        """Extract data from file"""
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        if self.file_format == "json":
            async for record in self._extract_json(source):
                yield record
        elif self.file_format == "csv":
            async for record in self._extract_csv(source):
                yield record
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    async def _extract_json(self, source: str) -> Iterator[DataRecord]:
        """Extract from JSON file"""
        with open(source, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    record = DataRecord(
                        id=f"json_{i}",
                        data=item,
                        source=source,
                        timestamp=datetime.utcnow()
                    )
                    yield record
            else:
                record = DataRecord(
                    id="json_0",
                    data=data,
                    source=source,
                    timestamp=datetime.utcnow()
                )
                yield record
    
    async def _extract_csv(self, source: str) -> Iterator[DataRecord]:
        """Extract from CSV file"""
        with open(source, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                record = DataRecord(
                    id=f"csv_{i}",
                    data=row,
                    source=source,
                    timestamp=datetime.utcnow()
                )
                yield record
    
    def get_schema(self) -> Dict[str, Any]:
        """Get file schema"""
        return {
            "type": "file",
            "format": self.file_format,
            "schema": self.schema
        }

class DatabaseExtractor(DataExtractor):
    """Database data extractor"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.schema = {}
    
    async def extract(self, source: str, **kwargs) -> Iterator[DataRecord]:
        """Extract data from database"""
        query = kwargs.get("query", f"SELECT * FROM {source}")
        batch_size = kwargs.get("batch_size", 1000)
        
        # Simulate database extraction
        for i in range(batch_size):
            record = DataRecord(
                id=f"db_{i}",
                data={"id": i, "name": f"Record {i}", "value": i * 10},
                source=source,
                timestamp=datetime.utcnow()
            )
            yield record
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema"""
        return {
            "type": "database",
            "connection": self.connection_string,
            "schema": self.schema
        }

class DataTransformer(ABC):
    """Abstract data transformer"""
    
    @abstractmethod
    async def transform(self, record: DataRecord) -> DataRecord:
        """Transform data record"""
        pass
    
    @abstractmethod
    def get_transformation_rules(self) -> Dict[str, Any]:
        """Get transformation rules"""
        pass

class FieldMapper(DataTransformer):
    """Field mapping transformer"""
    
    def __init__(self, field_mappings: Dict[str, str]):
        self.field_mappings = field_mappings
    
    async def transform(self, record: DataRecord) -> DataRecord:
        """Transform record by mapping fields"""
        transformed_data = {}
        
        for old_field, new_field in self.field_mappings.items():
            if old_field in record.data:
                transformed_data[new_field] = record.data[old_field]
        
        # Keep unmapped fields
        for field, value in record.data.items():
            if field not in self.field_mappings:
                transformed_data[field] = value
        
        return DataRecord(
            id=record.id,
            data=transformed_data,
            source=record.source,
            timestamp=record.timestamp,
            metadata=record.metadata
        )
    
    def get_transformation_rules(self) -> Dict[str, Any]:
        """Get transformation rules"""
        return {
            "type": "field_mapping",
            "mappings": self.field_mappings
        }

class DataValidator(DataTransformer):
    """Data validation transformer"""
    
    def __init__(self, validation_rules: Dict[str, Callable]):
        self.validation_rules = validation_rules
    
    async def transform(self, record: DataRecord) -> DataRecord:
        """Validate and transform record"""
        errors = []
        
        for field, validator in self.validation_rules.items():
            if field in record.data:
                try:
                    if not validator(record.data[field]):
                        errors.append(f"Validation failed for field {field}")
                except Exception as e:
                    errors.append(f"Validation error for field {field}: {e}")
        
        if errors:
            record.metadata = record.metadata or {}
            record.metadata["validation_errors"] = errors
        
        return record
    
    def get_transformation_rules(self) -> Dict[str, Any]:
        """Get transformation rules"""
        return {
            "type": "validation",
            "rules": list(self.validation_rules.keys())
        }

class DataLoader(ABC):
    """Abstract data loader"""
    
    @abstractmethod
    async def load(self, records: List[DataRecord], destination: str) -> bool:
        """Load records to destination"""
        pass
    
    @abstractmethod
    def get_loader_info(self) -> Dict[str, Any]:
        """Get loader information"""
        pass

class FileLoader(DataLoader):
    """File-based data loader"""
    
    def __init__(self, file_format: str = "json"):
        self.file_format = file_format
    
    async def load(self, records: List[DataRecord], destination: str) -> bool:
        """Load records to file"""
        try:
            if self.file_format == "json":
                await self._load_json(records, destination)
            elif self.file_format == "csv":
                await self._load_csv(records, destination)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    async def _load_json(self, records: List[DataRecord], destination: str) -> None:
        """Load records to JSON file"""
        data = [record.to_dict() for record in records]
        with open(destination, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _load_csv(self, records: List[DataRecord], destination: str) -> None:
        """Load records to CSV file"""
        if not records:
            return
        
        fieldnames = set()
        for record in records:
            fieldnames.update(record.data.keys())
        
        with open(destination, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
            for record in records:
                writer.writerow(record.data)
    
    def get_loader_info(self) -> Dict[str, Any]:
        """Get loader information"""
        return {
            "type": "file",
            "format": self.file_format
        }

class DatabaseLoader(DataLoader):
    """Database data loader"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def load(self, records: List[DataRecord], destination: str) -> bool:
        """Load records to database"""
        try:
            # Simulate database loading
            for record in records:
                # In real implementation, this would insert into database
                await asyncio.sleep(0.001)
            
            logger.info(f"Loaded {len(records)} records to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to load to database: {e}")
            return False
    
    def get_loader_info(self) -> Dict[str, Any]:
        """Get loader information"""
        return {
            "type": "database",
            "connection": self.connection_string
        }

class ETLPipeline:
    """ETL pipeline orchestrator"""
    
    def __init__(self, name: str):
        self.name = name
        self.extractors: List[DataExtractor] = []
        self.transformers: List[DataTransformer] = []
        self.loaders: List[DataLoader] = []
        self.metrics: Dict[str, Any] = {}
        self.errors: List[str] = []
    
    def add_extractor(self, extractor: DataExtractor) -> None:
        """Add data extractor"""
        self.extractors.append(extractor)
    
    def add_transformer(self, transformer: DataTransformer) -> None:
        """Add data transformer"""
        self.transformers.append(transformer)
    
    def add_loader(self, loader: DataLoader) -> None:
        """Add data loader"""
        self.loaders.append(loader)
    
    async def run(self, source: str, destination: str, **kwargs) -> bool:
        """Run ETL pipeline"""
        start_time = time.time()
        self.metrics = {
            "start_time": start_time,
            "records_processed": 0,
            "records_failed": 0,
            "extraction_time": 0,
            "transformation_time": 0,
            "loading_time": 0
        }
        
        try:
            # Extract data
            extraction_start = time.time()
            records = []
            for extractor in self.extractors:
                async for record in extractor.extract(source, **kwargs):
                    records.append(record)
            
            self.metrics["extraction_time"] = time.time() - extraction_start
            self.metrics["records_processed"] = len(records)
            
            # Transform data
            transformation_start = time.time()
            transformed_records = []
            for record in records:
                transformed_record = record
                for transformer in self.transformers:
                    transformed_record = await transformer.transform(transformed_record)
                transformed_records.append(transformed_record)
            
            self.metrics["transformation_time"] = time.time() - transformation_start
            
            # Load data
            loading_start = time.time()
            for loader in self.loaders:
                success = await loader.load(transformed_records, destination)
                if not success:
                    self.errors.append(f"Failed to load data with {loader.__class__.__name__}")
            
            self.metrics["loading_time"] = time.time() - loading_start
            self.metrics["total_time"] = time.time() - start_time
            
            return len(self.errors) == 0
        
        except Exception as e:
            self.errors.append(f"Pipeline error: {e}")
            logger.error(f"Pipeline {self.name} failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics.copy()
    
    def get_errors(self) -> List[str]:
        """Get pipeline errors"""
        return self.errors.copy()

# Usage examples
async def example_etl_pipeline():
    """Example ETL pipeline usage"""
    # Create pipeline
    pipeline = ETLPipeline("user_data_pipeline")
    
    # Add extractor
    file_extractor = FileExtractor(file_format="json")
    pipeline.add_extractor(file_extractor)
    
    # Add transformers
    field_mapper = FieldMapper({
        "user_id": "id",
        "user_name": "name",
        "user_email": "email"
    })
    pipeline.add_transformer(field_mapper)
    
    # Add validators
    def validate_email(email: str) -> bool:
        return "@" in email and "." in email
    
    def validate_name(name: str) -> bool:
        return len(name) > 0
    
    validator = DataValidator({
        "email": validate_email,
        "name": validate_name
    })
    pipeline.add_transformer(validator)
    
    # Add loader
    file_loader = FileLoader(file_format="json")
    pipeline.add_loader(file_loader)
    
    # Run pipeline
    success = await pipeline.run("input.json", "output.json")
    print(f"Pipeline success: {success}")
    
    # Get metrics
    metrics = pipeline.get_metrics()
    print(f"Pipeline metrics: {metrics}")
    
    # Get errors
    errors = pipeline.get_errors()
    if errors:
        print(f"Pipeline errors: {errors}")
```

### Stream Processing

```python
# python/02-stream-processing.py

"""
Stream processing patterns for real-time data processing
"""

from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import json
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

class StreamEvent:
    """Stream event definition"""
    
    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: datetime = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.id = f"{event_type}_{int(self.timestamp.timestamp() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }

class StreamProcessor:
    """Stream processor for real-time data"""
    
    def __init__(self, name: str, buffer_size: int = 1000):
        self.name = name
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.processors: List[Callable] = []
        self.metrics: Dict[str, Any] = {
            "events_processed": 0,
            "events_failed": 0,
            "processing_time": 0,
            "buffer_utilization": 0
        }
        self.running = False
    
    def add_processor(self, processor: Callable) -> None:
        """Add event processor"""
        self.processors.append(processor)
    
    async def process_event(self, event: StreamEvent) -> bool:
        """Process single event"""
        try:
            start_time = time.time()
            
            # Add to buffer
            self.buffer.append(event)
            
            # Process event
            for processor in self.processors:
                if asyncio.iscoroutinefunction(processor):
                    await processor(event)
                else:
                    processor(event)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["events_processed"] += 1
            self.metrics["processing_time"] += processing_time
            self.metrics["buffer_utilization"] = len(self.buffer) / self.buffer_size
            
            return True
        
        except Exception as e:
            self.metrics["events_failed"] += 1
            logger.error(f"Failed to process event {event.id}: {e}")
            return False
    
    async def process_stream(self, stream: AsyncIterator[StreamEvent]) -> None:
        """Process stream of events"""
        self.running = True
        
        async for event in stream:
            if not self.running:
                break
            
            await self.process_event(event)
    
    def stop(self) -> None:
        """Stop stream processing"""
        self.running = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return self.metrics.copy()
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status"""
        return {
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.buffer_size,
            "utilization": len(self.buffer) / self.buffer_size,
            "oldest_event": self.buffer[0].timestamp if self.buffer else None,
            "newest_event": self.buffer[-1].timestamp if self.buffer else None
        }

class WindowProcessor:
    """Window-based stream processor"""
    
    def __init__(self, window_size: int, window_type: str = "time"):
        self.window_size = window_size
        self.window_type = window_type
        self.windows: List[Dict[str, Any]] = []
        self.current_window: Dict[str, Any] = None
        self.window_start = None
    
    def add_event(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Add event to window"""
        if self.window_type == "time":
            return self._add_to_time_window(event)
        elif self.window_type == "count":
            return self._add_to_count_window(event)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
    
    def _add_to_time_window(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Add event to time window"""
        if self.window_start is None:
            self.window_start = event.timestamp
            self.current_window = {
                "start_time": self.window_start,
                "end_time": self.window_start + timedelta(seconds=self.window_size),
                "events": []
            }
        
        # Check if event is within current window
        if event.timestamp < self.current_window["end_time"]:
            self.current_window["events"].append(event)
            return None
        
        # Window is complete, start new window
        completed_window = self.current_window.copy()
        self.windows.append(completed_window)
        
        self.window_start = event.timestamp
        self.current_window = {
            "start_time": self.window_start,
            "end_time": self.window_start + timedelta(seconds=self.window_size),
            "events": [event]
        }
        
        return completed_window
    
    def _add_to_count_window(self, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Add event to count window"""
        if self.current_window is None:
            self.current_window = {
                "start_time": event.timestamp,
                "events": []
            }
        
        self.current_window["events"].append(event)
        
        if len(self.current_window["events"]) >= self.window_size:
            completed_window = self.current_window.copy()
            self.windows.append(completed_window)
            self.current_window = None
            return completed_window
        
        return None
    
    def get_window_stats(self) -> Dict[str, Any]:
        """Get window statistics"""
        if not self.windows:
            return {"total_windows": 0, "avg_events_per_window": 0}
        
        total_events = sum(len(window["events"]) for window in self.windows)
        return {
            "total_windows": len(self.windows),
            "total_events": total_events,
            "avg_events_per_window": total_events / len(self.windows),
            "window_size": self.window_size,
            "window_type": self.window_type
        }

class StreamAggregator:
    """Stream aggregator for real-time analytics"""
    
    def __init__(self):
        self.aggregations: Dict[str, Any] = {}
        self.counts: Dict[str, int] = {}
        self.sums: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}
        self.avgs: Dict[str, List[float]] = {}
    
    def add_event(self, event: StreamEvent) -> None:
        """Add event to aggregations"""
        event_type = event.event_type
        
        # Update counts
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        
        # Update numeric aggregations
        for field, value in event.data.items():
            if isinstance(value, (int, float)):
                key = f"{event_type}.{field}"
                
                # Sum
                self.sums[key] = self.sums.get(key, 0) + value
                
                # Min/Max
                if key not in self.mins:
                    self.mins[key] = value
                    self.maxs[key] = value
                else:
                    self.mins[key] = min(self.mins[key], value)
                    self.maxs[key] = max(self.maxs[key], value)
                
                # Average
                if key not in self.avgs:
                    self.avgs[key] = []
                self.avgs[key].append(value)
    
    def get_aggregations(self) -> Dict[str, Any]:
        """Get current aggregations"""
        aggregations = {}
        
        for event_type in self.counts.keys():
            aggregations[event_type] = {
                "count": self.counts[event_type],
                "fields": {}
            }
            
            for field in self.avgs.keys():
                if field.startswith(f"{event_type}."):
                    field_name = field.split(".", 1)[1]
                    aggregations[event_type]["fields"][field_name] = {
                        "sum": self.sums.get(field, 0),
                        "min": self.mins.get(field, 0),
                        "max": self.maxs.get(field, 0),
                        "avg": sum(self.avgs[field]) / len(self.avgs[field]) if self.avgs[field] else 0
                    }
        
        return aggregations
    
    def reset(self) -> None:
        """Reset all aggregations"""
        self.aggregations.clear()
        self.counts.clear()
        self.sums.clear()
        self.mins.clear()
        self.maxs.clear()
        self.avgs.clear()

# Usage examples
async def example_stream_processing():
    """Example stream processing usage"""
    # Create stream processor
    processor = StreamProcessor("event_processor", buffer_size=100)
    
    # Add processors
    def log_processor(event: StreamEvent):
        print(f"Processing event: {event.event_type}")
    
    def metrics_processor(event: StreamEvent):
        print(f"Event metrics: {event.data}")
    
    processor.add_processor(log_processor)
    processor.add_processor(metrics_processor)
    
    # Create window processor
    window_processor = WindowProcessor(window_size=5, window_type="count")
    
    # Create aggregator
    aggregator = StreamAggregator()
    
    # Simulate stream processing
    async def generate_events():
        """Generate sample events"""
        for i in range(20):
            event = StreamEvent(
                event_type="user_action",
                data={"user_id": f"user_{i % 5}", "action": "click", "value": i * 10}
            )
            yield event
            await asyncio.sleep(0.1)
    
    # Process events
    async for event in generate_events():
        await processor.process_event(event)
        
        # Add to window processor
        completed_window = window_processor.add_event(event)
        if completed_window:
            print(f"Completed window: {len(completed_window['events'])} events")
        
        # Add to aggregator
        aggregator.add_event(event)
    
    # Get metrics
    metrics = processor.get_metrics()
    print(f"Processor metrics: {metrics}")
    
    # Get window stats
    window_stats = window_processor.get_window_stats()
    print(f"Window stats: {window_stats}")
    
    # Get aggregations
    aggregations = aggregator.get_aggregations()
    print(f"Aggregations: {aggregations}")
```

### Batch Processing

```python
# python/03-batch-processing.py

"""
Batch processing patterns for large-scale data processing
"""

from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processor for large-scale data processing"""
    
    def __init__(self, batch_size: int = 1000, max_workers: int = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.metrics: Dict[str, Any] = {
            "batches_processed": 0,
            "records_processed": 0,
            "processing_time": 0,
            "errors": 0
        }
    
    async def process_batches(self, data_source: Iterator[Dict[str, Any]], 
                            processor: Callable) -> List[Any]:
        """Process data in batches"""
        results = []
        batch = []
        start_time = time.time()
        
        for record in data_source:
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                try:
                    result = await self._process_batch(batch, processor)
                    results.extend(result)
                    self.metrics["batches_processed"] += 1
                    self.metrics["records_processed"] += len(batch)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    self.metrics["errors"] += 1
                
                batch = []
        
        # Process remaining records
        if batch:
            try:
                result = await self._process_batch(batch, processor)
                results.extend(result)
                self.metrics["batches_processed"] += 1
                self.metrics["records_processed"] += len(batch)
            except Exception as e:
                logger.error(f"Final batch processing error: {e}")
                self.metrics["errors"] += 1
        
        self.metrics["processing_time"] = time.time() - start_time
        return results
    
    async def _process_batch(self, batch: List[Dict[str, Any]], 
                           processor: Callable) -> List[Any]:
        """Process single batch"""
        if asyncio.iscoroutinefunction(processor):
            return await processor(batch)
        else:
            return processor(batch)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return self.metrics.copy()

class ParallelProcessor:
    """Parallel processor for CPU-intensive tasks"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    async def process_parallel(self, data: List[Any], processor: Callable, 
                             use_processes: bool = False) -> List[Any]:
        """Process data in parallel"""
        if use_processes:
            # Use process pool for CPU-intensive tasks
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.process_pool, self._process_batch, data, processor
            )
        else:
            # Use thread pool for I/O-intensive tasks
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.thread_pool, self._process_batch, data, processor
            )
        
        return results
    
    def _process_batch(self, data: List[Any], processor: Callable) -> List[Any]:
        """Process batch of data"""
        return [processor(item) for item in data]
    
    def close(self) -> None:
        """Close thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class DataPartitioner:
    """Data partitioner for distributed processing"""
    
    def __init__(self, num_partitions: int = 4):
        self.num_partitions = num_partitions
    
    def partition_data(self, data: List[Any], partition_key: Callable = None) -> List[List[Any]]:
        """Partition data into multiple chunks"""
        if partition_key is None:
            # Simple round-robin partitioning
            partitions = [[] for _ in range(self.num_partitions)]
            for i, item in enumerate(data):
                partitions[i % self.num_partitions].append(item)
        else:
            # Key-based partitioning
            partitions = [[] for _ in range(self.num_partitions)]
            for item in data:
                key = partition_key(item)
                partition_index = hash(key) % self.num_partitions
                partitions[partition_index].append(item)
        
        return partitions
    
    def get_partition_info(self, partitions: List[List[Any]]) -> Dict[str, Any]:
        """Get partition information"""
        return {
            "num_partitions": len(partitions),
            "partition_sizes": [len(partition) for partition in partitions],
            "total_items": sum(len(partition) for partition in partitions),
            "avg_partition_size": sum(len(partition) for partition in partitions) / len(partitions)
        }

class DataMerger:
    """Data merger for combining processed results"""
    
    def __init__(self, merge_strategy: str = "append"):
        self.merge_strategy = merge_strategy
    
    def merge_results(self, results: List[List[Any]]) -> List[Any]:
        """Merge results from multiple partitions"""
        if self.merge_strategy == "append":
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        elif self.merge_strategy == "concat":
            return [item for result in results for item in result]
        else:
            raise ValueError(f"Unsupported merge strategy: {self.merge_strategy}")
    
    def merge_with_deduplication(self, results: List[List[Any]], 
                                key_func: Callable = None) -> List[Any]:
        """Merge results with deduplication"""
        if key_func is None:
            # Simple deduplication
            seen = set()
            merged = []
            for result in results:
                for item in result:
                    if item not in seen:
                        seen.add(item)
                        merged.append(item)
            return merged
        else:
            # Key-based deduplication
            seen_keys = set()
            merged = []
            for result in results:
                for item in result:
                    key = key_func(item)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        merged.append(item)
            return merged

# Usage examples
async def example_batch_processing():
    """Example batch processing usage"""
    # Create batch processor
    batch_processor = BatchProcessor(batch_size=100)
    
    # Create parallel processor
    parallel_processor = ParallelProcessor(max_workers=4)
    
    # Create data partitioner
    partitioner = DataPartitioner(num_partitions=4)
    
    # Create data merger
    merger = DataMerger(merge_strategy="append")
    
    # Sample data
    data = [{"id": i, "value": i * 10} for i in range(1000)]
    
    # Partition data
    partitions = partitioner.partition_data(data)
    partition_info = partitioner.get_partition_info(partitions)
    print(f"Partition info: {partition_info}")
    
    # Process partitions in parallel
    async def process_partition(partition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process single partition"""
        results = []
        for item in partition:
            # Simulate processing
            processed_item = {
                "id": item["id"],
                "value": item["value"],
                "processed": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(processed_item)
        return results
    
    # Process all partitions
    partition_results = []
    for partition in partitions:
        result = await parallel_processor.process_parallel(
            partition, process_partition, use_processes=False
        )
        partition_results.append(result)
    
    # Merge results
    merged_results = merger.merge_results(partition_results)
    print(f"Merged {len(merged_results)} results")
    
    # Get batch processor metrics
    metrics = batch_processor.get_metrics()
    print(f"Batch processor metrics: {metrics}")
    
    # Close parallel processor
    parallel_processor.close()
```

## TL;DR Runbook

### Quick Start

```python
# 1. ETL Pipeline
pipeline = ETLPipeline("data_pipeline")
pipeline.add_extractor(FileExtractor())
pipeline.add_transformer(FieldMapper({"old": "new"}))
pipeline.add_loader(FileLoader())
await pipeline.run("input.json", "output.json")

# 2. Stream Processing
processor = StreamProcessor("event_processor")
await processor.process_stream(event_stream)

# 3. Batch Processing
batch_processor = BatchProcessor(batch_size=1000)
results = await batch_processor.process_batches(data_source, processor)

# 4. Parallel Processing
parallel_processor = ParallelProcessor(max_workers=4)
results = await parallel_processor.process_parallel(data, processor)

# 5. Data Partitioning
partitioner = DataPartitioner(num_partitions=4)
partitions = partitioner.partition_data(data)
```

### Essential Patterns

```python
# Complete data processing setup
def setup_data_processing():
    """Setup complete data processing environment"""
    
    # ETL Pipeline
    pipeline = ETLPipeline("main_pipeline")
    
    # Stream Processor
    stream_processor = StreamProcessor("real_time_processor")
    
    # Batch Processor
    batch_processor = BatchProcessor(batch_size=1000)
    
    # Parallel Processor
    parallel_processor = ParallelProcessor(max_workers=4)
    
    # Data Partitioner
    partitioner = DataPartitioner(num_partitions=4)
    
    # Data Merger
    merger = DataMerger(merge_strategy="append")
    
    print("Data processing setup complete!")
```

---

*This guide provides the complete machinery for Python data processing. Each pattern includes implementation examples, processing strategies, and real-world usage patterns for enterprise data processing management.*
