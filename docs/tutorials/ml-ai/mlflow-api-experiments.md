# MLflow API Experiments: Models, Inference, and Database Integration

**Objective**: Master MLflow's REST API to run experiments, manage models, and output inference to databases. Build production-ready ML pipelines that track experiments, serve models, and persist predictions.

When your ML models need to scale beyond notebooks and into production, MLflow's APIs become your bridge between experimentation and deployment. This guide shows you how to leverage MLflow's REST API for experiment tracking, model management, and inference serving with database integration.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **MLflow server running**
   - MLflow tracking server with database backend
   - Model registry and artifact storage
   - REST API accessible and authenticated

2. **Database connectivity**
   - PostgreSQL, MySQL, or SQLite for MLflow metadata
   - Target database for inference storage
   - Connection pooling and transaction management

3. **Model compatibility**
   - Models logged in MLflow format
   - Compatible with MLflow's model flavors
   - Versioned and tagged appropriately

4. **API authentication**
   - MLflow authentication configured
   - API keys or token-based auth
   - Proper permissions for model access

5. **Production considerations**
   - Error handling and retries
   - Rate limiting and throttling
   - Monitoring and logging

**Why These Requirements**: MLflow APIs are the foundation of production ML workflows. Getting the infrastructure right prevents hours of debugging later.

## 1) MLflow Server Setup: The Foundation

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    image: python:3.11-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
    volumes:
      - ./mlflow:/app
    command: >
      bash -c "
        pip install mlflow[extras] psycopg2-binary boto3 &&
        mlflow server 
          --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
          --default-artifact-root s3://mlflow-artifacts
          --host 0.0.0.0
          --port 5000
      "
    depends_on:
      - postgres
    networks:
      - mlflow-network

  postgres:
    image: postgres:15-alpine
    container_name: mlflow-postgres
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - mlflow-network

  inference-db:
    image: postgres:15-alpine
    container_name: inference-postgres
    environment:
      - POSTGRES_DB=inference
      - POSTGRES_USER=inference
      - POSTGRES_PASSWORD=inference
    volumes:
      - inference_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5433:5432"
    networks:
      - mlflow-network

volumes:
  postgres_data:
  inference_data:

networks:
  mlflow-network:
    driver: bridge
```

### Database Schema for Inference

```sql
-- sql/init.sql
-- Inference database schema
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_id UUID DEFAULT gen_random_uuid(),
    input_data JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    experiment_id VARCHAR(50),
    run_id VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_stage VARCHAR(50) DEFAULT 'None',
    model_uri TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_predictions_model_name ON model_predictions(model_name);
CREATE INDEX idx_model_predictions_created_at ON model_predictions(created_at);
CREATE INDEX idx_model_metadata_model_name ON model_metadata(model_name);
```

**Why This Setup**: MLflow needs a database backend for metadata and artifact storage. The inference database stores predictions for analysis and monitoring.

## 2) MLflow API Client: The Interface

### Python Client Setup

```python
# mlflow_client.py
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowAPIClient:
    """MLflow API client for experiment management and inference"""
    
    def __init__(self, tracking_uri: str, database_uri: str):
        self.tracking_uri = tracking_uri
        self.database_uri = database_uri
        mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize database connection
        self.db_conn = psycopg2.connect(database_uri)
        self.db_conn.autocommit = True
        
        # API base URL
        self.api_base = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow"
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db_conn:
            self.db_conn.close()
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated request to MLflow API"""
        url = f"{self.api_base}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        try:
            response = self._make_request("GET", "experiments/list")
            return response.get("experiments", [])
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def get_experiment(self, experiment_id: str) -> Dict:
        """Get experiment details"""
        try:
            response = self._make_request("GET", f"experiments/get?experiment_id={experiment_id}")
            return response.get("experiment", {})
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return {}
    
    def create_experiment(self, name: str, tags: Dict = None) -> str:
        """Create new experiment"""
        try:
            data = {"name": name}
            if tags:
                data["tags"] = [{"key": k, "value": v} for k, v in tags.items()]
            
            response = self._make_request("POST", "experiments/create", data)
            return response.get("experiment_id")
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        try:
            response = self._make_request("GET", "registered-models/list")
            return response.get("registered_models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a model"""
        try:
            response = self._make_request("GET", f"registered-models/get?name={model_name}")
            return response.get("registered_model", {}).get("latest_versions", [])
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []
    
    def get_model_stage(self, model_name: str, stage: str = "Production") -> Dict:
        """Get model by stage"""
        try:
            response = self._make_request("GET", f"registered-models/get?name={model_name}")
            model = response.get("registered_model", {})
            
            for version in model.get("latest_versions", []):
                if version.get("current_stage") == stage:
                    return version
            
            return {}
        except Exception as e:
            logger.error(f"Failed to get model {model_name} in stage {stage}: {e}")
            return {}
    
    def run_experiment(self, experiment_id: str, model_name: str, 
                      input_data: Dict, tags: Dict = None) -> Dict:
        """Run experiment with given model"""
        try:
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log parameters
                mlflow.log_params(input_data.get("parameters", {}))
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Load model
                model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
                
                # Make prediction
                prediction = model.predict(input_data.get("features", []))
                
                # Log metrics
                mlflow.log_metric("prediction_count", len(prediction))
                mlflow.log_metric("model_version", model.metadata.get("version", "unknown"))
                
                # Log prediction
                mlflow.log_text(str(prediction), "prediction.txt")
                
                return {
                    "run_id": run.info.run_id,
                    "experiment_id": experiment_id,
                    "prediction": prediction.tolist(),
                    "model_name": model_name,
                    "model_version": model.metadata.get("version", "unknown")
                }
                
        except Exception as e:
            logger.error(f"Failed to run experiment: {e}")
            raise
    
    def save_inference_to_db(self, prediction_result: Dict, input_data: Dict) -> str:
        """Save inference result to database"""
        try:
            cursor = self.db_conn.cursor()
            
            # Insert prediction
            insert_query = """
                INSERT INTO model_predictions 
                (model_name, model_version, input_data, prediction, 
                 experiment_id, run_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING prediction_id
            """
            
            cursor.execute(insert_query, (
                prediction_result.get("model_name"),
                prediction_result.get("model_version"),
                json.dumps(input_data),
                json.dumps(prediction_result.get("prediction")),
                prediction_result.get("experiment_id"),
                prediction_result.get("run_id"),
                datetime.now()
            ))
            
            prediction_id = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"Saved inference to database: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to save inference to database: {e}")
            raise
    
    def get_inference_history(self, model_name: str = None, 
                             limit: int = 100) -> List[Dict]:
        """Get inference history from database"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM model_predictions"
            params = []
            
            if model_name:
                query += " WHERE model_name = %s"
                params.append(model_name)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get inference history: {e}")
            return []
    
    def update_model_metadata(self, model_name: str, model_version: str, 
                             model_uri: str, stage: str = "None") -> None:
        """Update model metadata in database"""
        try:
            cursor = self.db_conn.cursor()
            
            # Upsert model metadata
            upsert_query = """
                INSERT INTO model_metadata 
                (model_name, model_version, model_stage, model_uri, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_name, model_version) 
                DO UPDATE SET 
                    model_stage = EXCLUDED.model_stage,
                    model_uri = EXCLUDED.model_uri,
                    updated_at = EXCLUDED.updated_at
            """
            
            cursor.execute(upsert_query, (
                model_name, model_version, stage, model_uri,
                datetime.now(), datetime.now()
            ))
            
            cursor.close()
            logger.info(f"Updated model metadata: {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to update model metadata: {e}")
            raise
```

**Why This Client**: MLflow's REST API provides programmatic access to experiments and models. This client abstracts the complexity and provides a clean interface for production use.

## 3) Experiment Management: The Workflow

### Running Experiments

```python
# experiment_runner.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow_client import MLflowAPIClient
import logging

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Run MLflow experiments with model training and inference"""
    
    def __init__(self, client: MLflowAPIClient):
        self.client = client
    
    def train_and_log_model(self, experiment_name: str, 
                           data: pd.DataFrame, target_column: str,
                           model_params: Dict = None) -> str:
        """Train model and log to MLflow"""
        try:
            # Create experiment
            experiment_id = self.client.create_experiment(
                name=experiment_name,
                tags={"type": "classification", "framework": "sklearn"}
            )
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 10),
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log to MLflow
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log parameters
                mlflow.log_params(model_params or {})
                mlflow.log_params({
                    "n_samples": len(data),
                    "n_features": X.shape[1],
                    "target_column": target_column
                })
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("n_estimators", model.n_estimators)
                mlflow.log_metric("max_depth", model.max_depth)
                
                # Log model
                mlflow.sklearn.log_model(
                    model, "model",
                    registered_model_name=experiment_name
                )
                
                # Log classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")
                
                logger.info(f"Model trained and logged: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to train and log model: {e}")
            raise
    
    def run_inference_experiment(self, experiment_name: str, 
                                model_name: str, input_data: Dict) -> Dict:
        """Run inference experiment"""
        try:
            # Get experiment
            experiments = self.client.list_experiments()
            experiment_id = None
            
            for exp in experiments:
                if exp.get("name") == experiment_name:
                    experiment_id = exp.get("experiment_id")
                    break
            
            if not experiment_id:
                raise ValueError(f"Experiment {experiment_name} not found")
            
            # Run experiment
            result = self.client.run_experiment(
                experiment_id=experiment_id,
                model_name=model_name,
                input_data=input_data,
                tags={"type": "inference", "model": model_name}
            )
            
            # Save to database
            prediction_id = self.client.save_inference_to_db(result, input_data)
            result["prediction_id"] = prediction_id
            
            logger.info(f"Inference experiment completed: {result['run_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run inference experiment: {e}")
            raise
    
    def batch_inference(self, model_name: str, input_data_list: List[Dict]) -> List[Dict]:
        """Run batch inference"""
        try:
            results = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    # Run inference
                    result = self.client.run_experiment(
                        experiment_id="0",  # Default experiment
                        model_name=model_name,
                        input_data=input_data,
                        tags={"type": "batch_inference", "batch_id": i}
                    )
                    
                    # Save to database
                    prediction_id = self.client.save_inference_to_db(result, input_data)
                    result["prediction_id"] = prediction_id
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process batch item {i}: {e}")
                    continue
            
            logger.info(f"Batch inference completed: {len(results)} predictions")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run batch inference: {e}")
            raise
```

### Model Management

```python
# model_manager.py
from mlflow_client import MLflowAPIClient
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage MLflow models and versions"""
    
    def __init__(self, client: MLflowAPIClient):
        self.client = client
    
    def list_available_models(self) -> List[Dict]:
        """List all available models"""
        try:
            models = self.client.list_models()
            
            # Get detailed information for each model
            detailed_models = []
            for model in models:
                model_name = model.get("name")
                versions = self.client.get_model_versions(model_name)
                
                detailed_models.append({
                    "name": model_name,
                    "description": model.get("description", ""),
                    "tags": model.get("tags", []),
                    "versions": versions,
                    "latest_version": versions[0] if versions else None
                })
            
            return detailed_models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed model information"""
        try:
            versions = self.client.get_model_versions(model_name)
            
            return {
                "name": model_name,
                "versions": versions,
                "total_versions": len(versions),
                "latest_version": versions[0] if versions else None,
                "stages": list(set(v.get("current_stage") for v in versions))
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}
    
    def promote_model_stage(self, model_name: str, version: str, 
                           stage: str) -> bool:
        """Promote model to new stage"""
        try:
            # This would require MLflow's model registry API
            # For now, we'll update our database metadata
            self.client.update_model_metadata(
                model_name=model_name,
                model_version=version,
                model_uri=f"models:/{model_name}/{version}",
                stage=stage
            )
            
            logger.info(f"Promoted {model_name}:{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Dict:
        """Get production model"""
        try:
            production_model = self.client.get_model_stage(
                model_name=model_name,
                stage="Production"
            )
            
            if not production_model:
                # Fallback to latest version
                versions = self.client.get_model_versions(model_name)
                if versions:
                    production_model = versions[0]
            
            return production_model
            
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            return {}
```

**Why Experiment Management**: MLflow experiments track model training and inference. This workflow provides a complete pipeline from training to production deployment.

## 4) Database Integration: The Persistence

### Inference Storage

```python
# inference_storage.py
import pandas as pd
from mlflow_client import MLflowAPIClient
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class InferenceStorage:
    """Store and retrieve inference results"""
    
    def __init__(self, client: MLflowAPIClient):
        self.client = client
    
    def store_inference(self, model_name: str, model_version: str,
                       input_data: Dict, prediction: List, 
                       confidence: float = None, metadata: Dict = None) -> str:
        """Store inference result"""
        try:
            # Prepare prediction result
            prediction_result = {
                "model_name": model_name,
                "model_version": model_version,
                "prediction": prediction,
                "experiment_id": metadata.get("experiment_id") if metadata else None,
                "run_id": metadata.get("run_id") if metadata else None
            }
            
            # Save to database
            prediction_id = self.client.save_inference_to_db(
                prediction_result, input_data
            )
            
            logger.info(f"Stored inference: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to store inference: {e}")
            raise
    
    def get_inference_history(self, model_name: str = None, 
                             start_date: datetime = None,
                             end_date: datetime = None,
                             limit: int = 1000) -> pd.DataFrame:
        """Get inference history as DataFrame"""
        try:
            # Get raw data
            history = self.client.get_inference_history(model_name, limit)
            
            if not history:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Filter by date range
            if start_date:
                df = df[df['created_at'] >= start_date]
            if end_date:
                df = df[df['created_at'] <= end_date]
            
            # Parse JSON columns
            if 'input_data' in df.columns:
                df['input_data'] = df['input_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            if 'prediction' in df.columns:
                df['prediction'] = df['prediction'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get inference history: {e}")
            return pd.DataFrame()
    
    def get_model_performance(self, model_name: str, 
                             days: int = 30) -> Dict:
        """Get model performance metrics"""
        try:
            # Get recent predictions
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.get_inference_history(
                model_name=model_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return {"error": "No data found"}
            
            # Calculate metrics
            total_predictions = len(df)
            unique_days = df['created_at'].dt.date.nunique()
            avg_predictions_per_day = total_predictions / unique_days if unique_days > 0 else 0
            
            # Confidence analysis
            confidence_stats = {}
            if 'confidence_score' in df.columns:
                confidence_stats = {
                    "mean_confidence": df['confidence_score'].mean(),
                    "min_confidence": df['confidence_score'].min(),
                    "max_confidence": df['confidence_score'].max(),
                    "std_confidence": df['confidence_score'].std()
                }
            
            return {
                "model_name": model_name,
                "period_days": days,
                "total_predictions": total_predictions,
                "unique_days": unique_days,
                "avg_predictions_per_day": avg_predictions_per_day,
                "confidence_stats": confidence_stats,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return {"error": str(e)}
    
    def export_inference_data(self, model_name: str, 
                            output_format: str = "csv",
                            output_path: str = None) -> str:
        """Export inference data"""
        try:
            # Get all data
            df = self.get_inference_history(model_name, limit=10000)
            
            if df.empty:
                raise ValueError("No data to export")
            
            # Generate output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"inference_export_{model_name}_{timestamp}.{output_format}"
            
            # Export based on format
            if output_format.lower() == "csv":
                df.to_csv(output_path, index=False)
            elif output_format.lower() == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif output_format.lower() == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            logger.info(f"Exported inference data to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export inference data: {e}")
            raise
```

### Database Analytics

```python
# database_analytics.py
import pandas as pd
import numpy as np
from mlflow_client import MLflowAPIClient
import logging
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseAnalytics:
    """Analyze inference data and model performance"""
    
    def __init__(self, client: MLflowAPIClient):
        self.client = client
    
    def analyze_model_usage(self, days: int = 30) -> Dict:
        """Analyze model usage patterns"""
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get inference history
            history = self.client.get_inference_history(limit=10000)
            
            if not history:
                return {"error": "No data found"}
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Filter by date range
            df = df[df['created_at'] >= start_date]
            
            if df.empty:
                return {"error": "No data in date range"}
            
            # Analyze by model
            model_analysis = df.groupby('model_name').agg({
                'id': 'count',
                'created_at': ['min', 'max'],
                'confidence_score': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            # Analyze by hour
            df['hour'] = df['created_at'].dt.hour
            hourly_usage = df.groupby('hour').size()
            
            # Analyze by day of week
            df['day_of_week'] = df['created_at'].dt.day_name()
            daily_usage = df.groupby('day_of_week').size()
            
            return {
                "period_days": days,
                "total_predictions": len(df),
                "unique_models": df['model_name'].nunique(),
                "model_analysis": model_analysis.to_dict(),
                "hourly_usage": hourly_usage.to_dict(),
                "daily_usage": daily_usage.to_dict(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze model usage: {e}")
            return {"error": str(e)}
    
    def detect_anomalies(self, model_name: str, 
                        threshold: float = 2.0) -> List[Dict]:
        """Detect anomalous predictions"""
        try:
            # Get model data
            df = self.get_inference_history(model_name, limit=1000)
            
            if df.empty:
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(df)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Analyze confidence scores
            if 'confidence_score' in df.columns:
                confidence = df['confidence_score'].dropna()
                
                if len(confidence) > 10:  # Need enough data
                    mean_conf = confidence.mean()
                    std_conf = confidence.std()
                    
                    # Find anomalies
                    anomalies = df[
                        (df['confidence_score'] < mean_conf - threshold * std_conf) |
                        (df['confidence_score'] > mean_conf + threshold * std_conf)
                    ]
                    
                    return anomalies.to_dict('records')
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return []
    
    def generate_report(self, model_name: str = None, 
                       days: int = 30) -> Dict:
        """Generate comprehensive report"""
        try:
            # Get basic stats
            usage_analysis = self.analyze_model_usage(days)
            
            # Get model-specific analysis
            if model_name:
                model_performance = self.client.get_model_performance(model_name, days)
                anomalies = self.detect_anomalies(model_name)
            else:
                model_performance = {}
                anomalies = []
            
            # Generate report
            report = {
                "report_generated": datetime.now().isoformat(),
                "period_days": days,
                "model_name": model_name,
                "usage_analysis": usage_analysis,
                "model_performance": model_performance,
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies[:10],  # Limit to first 10
                "summary": {
                    "total_predictions": usage_analysis.get("total_predictions", 0),
                    "unique_models": usage_analysis.get("unique_models", 0),
                    "anomalies": len(anomalies)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {"error": str(e)}
```

**Why Database Integration**: Inference data needs to be stored for analysis, monitoring, and compliance. This integration provides a complete data pipeline.

## 5) Complete Example: The Production Pipeline

### Main Application

```python
# main.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mlflow_client import MLflowAPIClient
from experiment_runner import ExperimentRunner
from model_manager import ModelManager
from inference_storage import InferenceStorage
from database_analytics import DatabaseAnalytics
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application demonstrating MLflow API usage"""
    
    # Configuration
    TRACKING_URI = "http://localhost:5000"
    DATABASE_URI = "postgresql://inference:inference@localhost:5433/inference"
    
    try:
        # Initialize MLflow client
        with MLflowAPIClient(TRACKING_URI, DATABASE_URI) as client:
            
            # Initialize components
            experiment_runner = ExperimentRunner(client)
            model_manager = ModelManager(client)
            inference_storage = InferenceStorage(client)
            analytics = DatabaseAnalytics(client)
            
            logger.info("MLflow API client initialized successfully")
            
            # 1. List available models
            logger.info("Listing available models...")
            models = model_manager.list_available_models()
            print(f"Found {len(models)} models:")
            for model in models:
                print(f"  - {model['name']}: {len(model['versions'])} versions")
            
            # 2. Create sample data and train model
            logger.info("Creating sample data...")
            X, y = make_classification(
                n_samples=1000, n_features=20, n_classes=2, 
                random_state=42
            )
            data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
            data['target'] = y
            
            # Train model
            logger.info("Training model...")
            run_id = experiment_runner.train_and_log_model(
                experiment_name="sample_classification",
                data=data,
                target_column="target",
                model_params={"n_estimators": 100, "max_depth": 10}
            )
            print(f"Model trained with run ID: {run_id}")
            
            # 3. Run inference experiments
            logger.info("Running inference experiments...")
            
            # Generate test data
            test_data = {
                "features": X[:10].tolist(),
                "parameters": {"batch_size": 10}
            }
            
            # Run inference
            inference_result = experiment_runner.run_inference_experiment(
                experiment_name="sample_classification",
                model_name="sample_classification",
                input_data=test_data
            )
            
            print(f"Inference completed:")
            print(f"  - Run ID: {inference_result['run_id']}")
            print(f"  - Predictions: {inference_result['prediction']}")
            print(f"  - Prediction ID: {inference_result['prediction_id']}")
            
            # 4. Batch inference
            logger.info("Running batch inference...")
            batch_data = [
                {"features": [X[i].tolist()], "parameters": {"batch_size": 1}}
                for i in range(5)
            ]
            
            batch_results = experiment_runner.batch_inference(
                model_name="sample_classification",
                input_data_list=batch_data
            )
            
            print(f"Batch inference completed: {len(batch_results)} predictions")
            
            # 5. Analyze results
            logger.info("Analyzing results...")
            
            # Get inference history
            history = inference_storage.get_inference_history(limit=100)
            print(f"Inference history: {len(history)} records")
            
            # Generate report
            report = analytics.generate_report(days=1)
            print(f"Report generated: {report['summary']}")
            
            # 6. Model management
            logger.info("Model management...")
            
            # Get model info
            model_info = model_manager.get_model_info("sample_classification")
            print(f"Model info: {model_info['total_versions']} versions")
            
            # Get production model
            production_model = model_manager.get_production_model("sample_classification")
            if production_model:
                print(f"Production model: {production_model.get('version')}")
            
            logger.info("Application completed successfully")
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Docker Compose for Complete Setup

```yaml
# docker-compose-complete.yml
version: '3.8'

services:
  mlflow:
    image: python:3.11-slim
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///mlflow/artifacts
    volumes:
      - ./mlflow:/app
      - mlflow_artifacts:/mlflow/artifacts
    command: >
      bash -c "
        pip install mlflow[extras] psycopg2-binary &&
        mlflow server 
          --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
          --default-artifact-root file:///mlflow/artifacts
          --host 0.0.0.0
          --port 5000
      "
    depends_on:
      - postgres
    networks:
      - mlflow-network

  postgres:
    image: postgres:15-alpine
    container_name: mlflow-postgres
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - mlflow-network

  inference-db:
    image: postgres:15-alpine
    container_name: inference-postgres
    environment:
      - POSTGRES_DB=inference
      - POSTGRES_USER=inference
      - POSTGRES_PASSWORD=inference
    volumes:
      - inference_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5433:5432"
    networks:
      - mlflow-network

  app:
    build: .
    container_name: mlflow-app
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URI=postgresql://inference:inference@inference-db:5432/inference
    volumes:
      - ./:/app
    depends_on:
      - mlflow
      - inference-db
    networks:
      - mlflow-network
    command: python main.py

volumes:
  postgres_data:
  inference_data:
  mlflow_artifacts:

networks:
  mlflow-network:
    driver: bridge
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Run application
CMD ["python", "main.py"]
```

### Requirements

```txt
# requirements.txt
mlflow[extras]==2.8.1
psycopg2-binary==2.9.7
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
requests==2.31.0
python-dotenv==1.0.0
```

**Why Complete Example**: Production ML workflows need end-to-end integration. This example demonstrates the complete pipeline from training to inference to database storage.

## 6) TL;DR Quickstart

### Essential Commands

```bash
# Start services
docker-compose up -d

# Run application
python main.py

# Check MLflow UI
open http://localhost:5000

# Check database
psql -h localhost -p 5433 -U inference -d inference
```

### Quick Verification

```bash
# Test MLflow API
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Test database
psql -h localhost -p 5433 -U inference -d inference -c "SELECT COUNT(*) FROM model_predictions;"

# Check logs
docker-compose logs mlflow
```

### Performance Testing

```bash
# Load testing
ab -n 100 -c 10 http://localhost:5000/api/2.0/mlflow/experiments/list

# Database performance
psql -h localhost -p 5433 -U inference -d inference -c "EXPLAIN ANALYZE SELECT * FROM model_predictions WHERE model_name = 'sample_classification';"
```

## 7) The Machine's Summary

MLflow APIs provide the foundation for production ML workflows. When configured properly, they enable experiment tracking, model management, and inference serving with database integration. The key is understanding the API structure and building robust error handling.

**The Dark Truth**: MLflow APIs are powerful but complex. Getting the configuration right prevents hours of debugging later.

**The Machine's Mantra**: "In experiment tracking we trust, in model management we build, and in the database we find the path to production ML."

**Why This Matters**: Production ML workflows need reliability, scalability, and observability. These APIs provide enterprise-grade capabilities that scale from experimentation to production.

---

*This tutorial provides the complete machinery for building production-ready ML workflows with MLflow APIs. The patterns scale from development to production, from single experiments to enterprise deployments.*
