# MCP ↔ MLflow Toolchain: Data-Agnostic LLM Experimentation

**Objective**: Build a complete MCP toolchain that enables LLMs to run MLflow experiments on any data source—files, SQL databases, or inline context. When you need to give LLMs safe access to machine learning workflows, when you want to track experiments across different data sources, when you're building AI-driven data science platforms—MCP becomes your weapon of choice.

MCP provides the foundation for secure, controlled access to MLflow experiments. Without proper understanding of tool design, data source abstraction, and experiment tracking, you're building vulnerable systems that miss the power of controlled ML experimentation. This guide shows you how to wield MCP with the precision of a senior data engineer.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand MCP fundamentals**
   - Tool server design and JSON-RPC protocol
   - Security boundaries and data source abstraction
   - Schema validation and error handling
   - Transport mechanisms (stdio vs HTTP)

2. **Master MLflow integration**
   - Experiment tracking and run management
   - Artifact storage and model logging
   - Parameter and metric logging
   - Model registry and deployment

3. **Know your data sources**
   - File-based data (CSV, Parquet)
   - SQL database integration
   - Inline context data
   - Data validation and preprocessing

4. **Validate everything**
   - Tool security and sandboxing
   - Data source validation
   - Experiment reproducibility
   - Performance under load

5. **Plan for production**
   - Scalable experiment tracking
   - Monitoring and alerting
   - Security and access control
   - Integration with existing ML workflows

**Why These Principles**: MCP mastery is the foundation of secure AI-driven ML experimentation. Understanding tool design, mastering MLflow integration, and following best practices is essential for building production-ready AI ML systems.

## 1) What You'll Build (The Vision)

### System Architecture

```mermaid
flowchart LR
    U[LLM / App] -- JSON tool call --> R[FastAPI MCP Router]
    R -- JSON-RPC over stdio --> S[mcp-mlflow-server.py]
    S -->|Tracks to| T[(MLflow Tracking Server)]
    S -->|Reads| D1[(Files: CSV/Parquet)]
    S -->|Queries| D2[(SQL DB)]
    S -->|Consumes| D3[(Inline context)]
    T --> A[Artifacts & Metrics] --> U
```

### Key Design Principles

```yaml
# MCP-MLflow toolchain principles
design_principles:
  data_agnostic:
    - "Single schema for all data sources"
    - "Unified experiment interface"
    - "Consistent parameter logging"
    - "Standardized artifact storage"
  
  security:
    - "Data source validation and jailing"
    - "SQL injection protection"
    - "File path restrictions"
    - "Resource limits and timeouts"
  
  reproducibility:
    - "Deterministic experiment runs"
    - "Version tracking for data and code"
    - "Parameter and metric logging"
    - "Artifact preservation"
  
  observability:
    - "Experiment tracking and monitoring"
    - "Performance metrics and logging"
    - "Error handling and recovery"
    - "Audit trails and compliance"
```

**Why This Architecture Matters**: The MCP-MLflow toolchain provides a unified interface for LLMs to run experiments across different data sources while maintaining security, reproducibility, and observability.

## 2) MLflow Infrastructure Setup (The Foundation)

### Docker Compose Configuration

```yaml
# docker-compose.yml - MLflow tracking server with optional Postgres
version: "3.9"
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.15.1
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlruns/mlruns.db
      --default-artifact-root /mlruns/artifacts
    ports: ["5000:5000"]
    volumes:
      - mlruns:/mlruns
    environment:
      MLFLOW_TRACKING_URI: http://0.0.0.0:5000

  # Optional: Postgres for SQL data sources
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: demo
      POSTGRES_USER: demo
      POSTGRES_PASSWORD: demo
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U demo -d demo"]
      interval: 5s
      timeout: 3s
      retries: 30

volumes:
  mlruns: {}
```

### Startup Commands

```bash
# Start MLflow infrastructure
docker compose up -d

# Verify MLflow is running
curl http://localhost:5000/health

# Access MLflow UI
open http://localhost:5000
```

**Why This Setup**: MLflow provides centralized experiment tracking, artifact storage, and model management. The Docker setup ensures consistent, reproducible MLflow infrastructure across environments.

## 3) Data Source Specification (The Schema)

### Unified Data Source Schema

```python
# Data source specifications for MCP-MLflow toolchain
data_source_specs = {
    "file_source": {
        "kind": "file",
        "path": "/abs/or/relative.csv",
        "format": "csv|parquet",
        "features": ["col1", "col2", "col3"],
        "target": "label"
    },
    "sql_source": {
        "kind": "sql",
        "conn": "postgresql+psycopg://demo:demo@localhost:5432/demo",
        "query": "SELECT * FROM training_table",
        "features": ["f1", "f2", "f3"],
        "target": "y"
    },
    "inline_source": {
        "kind": "inline",
        "task": "text-classification",
        "records": [
            {"text": "storm incoming near bay", "label": 1},
            {"text": "clear skies and sunshine", "label": 0}
        ]
    }
}
```

### Data Source Validation

```python
# Data source validation and preprocessing
def validate_data_source(source: dict) -> dict:
    """Validate and normalize data source specification"""
    kind = source.get("kind")
    
    if kind == "file":
        # Validate file path and format
        path = source.get("path")
        if not path or not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        
        format_type = source.get("format", "csv")
        if format_type not in ["csv", "parquet"]:
            raise ValueError(f"Unsupported format: {format_type}")
    
    elif kind == "sql":
        # Validate SQL connection and query
        conn = source.get("conn")
        if not conn or not conn.startswith(("postgresql", "sqlite", "mysql")):
            raise ValueError(f"Invalid connection string: {conn}")
        
        query = source.get("query", "")
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
    
    elif kind == "inline":
        # Validate inline data structure
        records = source.get("records", [])
        if not records or not isinstance(records, list):
            raise ValueError("Inline records must be a non-empty list")
    
    else:
        raise ValueError(f"Unsupported data source kind: {kind}")
    
    return source
```

**Why This Schema**: The unified data source specification enables consistent experiment tracking across different data types while maintaining security and validation boundaries.

## 4) MCP Server Implementation (The Core)

### Complete MCP-MLflow Server

```python
# mcp-mlflow-server.py - Complete MCP server for MLflow experiments
import os
import sys
import json
import time
import tempfile
import uuid
import io
import contextlib
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine

# Configuration
HARD_TIMEOUT_MS = 60_000
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Tool schema definitions
SCHEMA = {
    "tools": {
        "mlflow.run_experiment": {
            "description": "Run a supervised ML experiment and log to MLflow",
            "params": {
                "type": "object",
                "required": ["experiment_name", "source"],
                "properties": {
                    "experiment_name": {"type": "string"},
                    "model": {
                        "type": "string",
                        "enum": ["logreg_tabular", "logreg_text"],
                        "default": "logreg_tabular"
                    },
                    "tracking_uri": {
                        "type": "string",
                        "description": "MLflow server URI",
                        "default": MLFLOW_TRACKING_URI
                    },
                    "source": {"type": "object"}
                }
            },
            "returns": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "metrics": {"type": "object"}
                }
            }
        },
        "mlflow.get_run": {
            "description": "Fetch MLflow run information and metrics",
            "params": {
                "type": "object",
                "required": ["run_id"],
                "properties": {
                    "run_id": {"type": "string"}
                }
            },
            "returns": {"type": "object"}
        }
    }
}

def _load_dataframe(source: Dict[str, Any]) -> Tuple[pd.DataFrame, str, list]:
    """Load data from various sources into a pandas DataFrame"""
    kind = source.get("kind")
    
    if kind == "file":
        path = source["path"]
        format_type = source.get("format", "csv")
        
        if format_type == "csv":
            df = pd.read_csv(path)
        elif format_type == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")
        
        return df, source["target"], source.get("features")
    
    elif kind == "sql":
        engine = create_engine(source["conn"])
        df = pd.read_sql(source["query"], engine)
        return df, source["target"], source.get("features")
    
    elif kind == "inline":
        task = source.get("task", "text-classification")
        if task != "text-classification":
            raise ValueError("Only text-classification inline is implemented")
        
        records = source["records"]
        df = pd.DataFrame(records)
        return df, "label", ["text"]
    
    else:
        raise ValueError(f"Unsupported source kind: {kind}")

def _create_tabular_pipeline():
    """Create pipeline for tabular data"""
    return Pipeline([
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

def _create_text_pipeline():
    """Create pipeline for text data"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

def run_experiment(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run MLflow experiment with data source abstraction"""
    mlflow.set_tracking_uri(args.get("tracking_uri", MLFLOW_TRACKING_URI))
    
    experiment_name = args["experiment_name"]
    source = args["source"]
    model_kind = args.get("model", "logreg_tabular")
    
    # Load data
    df, target, features = _load_dataframe(source)
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    # Prepare features
    if source["kind"] in ("file", "sql") and features:
        X = df[features]
    elif source["kind"] in ("file", "sql") and not features:
        X = df.drop(columns=[target])
    else:  # inline text
        X = df["text"]
    
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )
    
    # Create model pipeline
    if model_kind == "logreg_text" or source["kind"] == "inline":
        pipeline = _create_text_pipeline()
    else:
        pipeline = _create_tabular_pipeline()
    
    # Setup experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Run experiment
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        
        # Log parameters
        mlflow.log_params({
            "model": model_kind,
            "n_rows": len(df),
            "n_features": X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 1,
            "source_kind": source["kind"],
            "target": target,
            "features": features if features else "all_except_target"
        })
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_weighted": f1_weighted
        })
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model")
            mlflow.sklearn.save_model(pipeline, model_path)
            mlflow.log_artifacts(model_path, artifact_path="model")
        
        # Save prediction preview
        preview_df = pd.DataFrame({
            "y_true": y_test[:20].tolist(),
            "y_pred": y_pred[:20].tolist()
        })
        
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            preview_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_path="preview")
        
        return {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "metrics": {"accuracy": accuracy, "f1_weighted": f1_weighted}
        }

def get_run_info(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get MLflow run information and artifacts"""
    mlflow.set_tracking_uri(args.get("tracking_uri", MLFLOW_TRACKING_URI))
    
    run_id = args["run_id"]
    run = mlflow.get_run(run_id)
    
    # Get artifacts
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "artifacts": [artifact.path for artifact in artifacts]
    }

def handle(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP method calls"""
    if method == "schema":
        return SCHEMA
    elif method == "mlflow.run_experiment":
        return run_experiment(params)
    elif method == "mlflow.get_run":
        return get_run_info(params)
    else:
        raise ValueError(f"Unknown method: {method}")

def rpc_loop():
    """Main RPC loop for stdio communication"""
    for line in sys.stdin:
        try:
            req = json.loads(line)
            
            # Check request age
            if (time.time() * 1000) - req.get("ts_ms", 0) > HARD_TIMEOUT_MS:
                raise TimeoutError("Request too old")
            
            result = handle(req["method"], req.get("params", {}))
            out = {"id": req.get("id"), "ok": True, "result": result}
            
        except Exception as e:
            out = {
                "id": req.get("id"),
                "ok": False,
                "error": type(e).__name__,
                "message": str(e)
            }
        
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    # Set MLflow tracking URI
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    rpc_loop()
```

### Installation and Dependencies

```bash
# Install required packages
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install mlflow==2.15.1 \
    scikit-learn==1.5.2 \
    pandas \
    pyarrow \
    sqlalchemy \
    psycopg2-binary \
    numpy \
    joblib
```

**Why This Server Works**: The MCP server provides a unified interface for running MLflow experiments across different data sources while maintaining security, reproducibility, and observability.

## 5) FastAPI Router (The Orchestration)

### Complete Router Implementation

```python
# router.py - FastAPI router for MCP-MLflow toolchain
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class InvokeReq(BaseModel):
    tool: str
    params: Dict[str, Any]

class InvokeRes(BaseModel):
    request_id: str
    tool: str
    duration_ms: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StdioServer:
    """Manages stdio-based MCP server processes"""
    
    def __init__(self, cmd: list[str], name: str):
        self.cmd = cmd
        self.name = name
        self.proc: Optional[asyncio.Process] = None
        self.lock = asyncio.Lock()

    async def start(self):
        """Start the MCP server process"""
        if self.proc and self.proc.returncode is None:
            return
        
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def call(self, method: str, params: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        """Call a method on the MCP server"""
        await self.start()
        assert self.proc and self.proc.stdin and self.proc.stdout
        
        req = {
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
            "ts_ms": int(time.time() * 1000)
        }
        
        line = (json.dumps(req) + "\n").encode()
        
        async with self.lock:
            self.proc.stdin.write(line)
            await self.proc.stdin.drain()
            fut = asyncio.create_task(self.proc.stdout.readline())
        
        try:
            raw = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"{self.name} timeout")
        
        resp = json.loads(raw.decode())
        if not resp.get("ok"):
            raise RuntimeError(f"{self.name} error: {resp.get('message')}")
        
        return resp["result"]

# FastAPI application
app = FastAPI(title="MCP-MLflow Router", version="1.0.0")

# Register MCP server
SERVER = StdioServer(["python", "mcp-mlflow-server.py"], "mlflow")

@app.post("/invoke", response_model=InvokeRes)
async def invoke(req: InvokeReq):
    """Invoke a tool through the MCP router"""
    t0 = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        result = await SERVER.call(req.tool, req.params, timeout=180.0)
        return InvokeRes(
            request_id=request_id,
            tool=req.tool,
            duration_ms=int((time.time() - t0) * 1000),
            result=result
        )
    except Exception as e:
        return InvokeRes(
            request_id=request_id,
            tool=req.tool,
            duration_ms=int((time.time() - t0) * 1000),
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server": "mcp-mlflow"}

@app.get("/tools")
async def list_tools():
    """List available tools and their schemas"""
    try:
        schema = await SERVER.call("schema", {})
        return {"tools": schema["tools"]}
    except Exception as e:
        return {"error": str(e)}
```

### Router Startup

```bash
# Start the FastAPI router
uvicorn router:app --port 8000 --reload

# Verify router is running
curl http://localhost:8000/health
```

**Why This Router Works**: The FastAPI router provides HTTP access to MCP tools while maintaining security, observability, and error handling.

## 6) Example Data Generation (The Setup)

### CSV Data Generation

```python
# generate_demo_data.py - Generate example datasets
import pandas as pd
import numpy as np

def generate_tabular_data():
    """Generate synthetic tabular data for classification"""
    n_samples = 400
    
    # Generate features
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    
    # Generate target with some signal
    y = ((X[:, 0] * 0.8 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + 
          np.random.randn(n_samples) * 0.2) > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "f1": X[:, 0],
        "f2": X[:, 1], 
        "f3": X[:, 2],
        "label": y
    })
    
    df.to_csv("train.csv", index=False)
    print(f"Generated train.csv with {len(df)} samples")
    return df

def generate_text_data():
    """Generate synthetic text classification data"""
    texts = [
        "storm incoming near bay",
        "clear skies and sunshine", 
        "bridge closure due to high winds",
        "picnic schedules released",
        "flooding expected tomorrow",
        "beautiful weather for hiking",
        "road construction delays",
        "festival planning underway"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = weather hazard, 0 = normal
    
    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv("text_data.csv", index=False)
    print(f"Generated text_data.csv with {len(df)} samples")
    return df

if __name__ == "__main__":
    generate_tabular_data()
    generate_text_data()
```

### SQL Database Setup

```sql
-- setup_demo_db.sql - Create demo PostgreSQL database
-- Connect to PostgreSQL: psql "postgresql://demo:demo@localhost:5432/demo"

DROP TABLE IF EXISTS training_table;

CREATE TABLE training_table (
    f1 DOUBLE PRECISION,
    f2 DOUBLE PRECISION, 
    f3 DOUBLE PRECISION,
    y INTEGER
);

-- Insert synthetic data
INSERT INTO training_table
SELECT 
    random() * 2 - 1,
    random() * 2 - 1,
    random() * 2 - 1,
    CASE 
        WHEN (random() * 2 - 1) * 0.8 + 
             (random() * 2 - 1) * 0.3 - 
             (random() * 2 - 1) * 0.2 + 
             (random() * 0.4 - 0.2) > 0 
        THEN 1 
        ELSE 0 
    END
FROM generate_series(1, 600);

-- Verify data
SELECT COUNT(*), AVG(y) FROM training_table;
```

## 7) Tool Invocation Examples (The Usage)

### File-Based Experiment

```bash
# Run experiment on CSV file
curl -s localhost:8000/invoke \
  -H 'content-type: application/json' \
  -d '{
    "tool": "mlflow.run_experiment",
    "params": {
      "experiment_name": "demo_tabular",
      "model": "logreg_tabular",
      "tracking_uri": "http://localhost:5000",
      "source": {
        "kind": "file",
        "path": "train.csv",
        "format": "csv",
        "features": ["f1", "f2", "f3"],
        "target": "label"
      }
    }
  }' | jq
```

### SQL-Based Experiment

```bash
# Run experiment on SQL database
curl -s localhost:8000/invoke \
  -H 'content-type: application/json' \
  -d '{
    "tool": "mlflow.run_experiment",
    "params": {
      "experiment_name": "demo_sql",
      "model": "logreg_tabular",
      "tracking_uri": "http://localhost:5000",
      "source": {
        "kind": "sql",
        "conn": "postgresql+psycopg://demo:demo@localhost:5432/demo",
        "query": "SELECT * FROM training_table",
        "features": ["f1", "f2", "f3"],
        "target": "y"
      }
    }
  }' | jq
```

### Inline Context Experiment

```bash
# Run experiment on inline text data
curl -s localhost:8000/invoke \
  -H 'content-type: application/json' \
  -d '{
    "tool": "mlflow.run_experiment",
    "params": {
      "experiment_name": "demo_inline_text",
      "model": "logreg_text",
      "tracking_uri": "http://localhost:5000",
      "source": {
        "kind": "inline",
        "task": "text-classification",
        "records": [
          {"text": "hazard: flooding likely tomorrow", "label": 1},
          {"text": "sunny day for the parade", "label": 0},
          {"text": "bridge closure due to high winds", "label": 1},
          {"text": "picnic schedules released", "label": 0}
        ]
      }
    }
  }' | jq
```

### Fetch Run Information

```bash
# Get run details (replace RUN_ID with actual run ID)
curl -s localhost:8000/invoke \
  -H 'content-type: application/json' \
  -d '{
    "tool": "mlflow.get_run",
    "params": {
      "run_id": "RUN_ID"
    }
  }' | jq
```

## 8) LLM Integration (The Intelligence)

### OpenAI Function Calling

```python
# llm_integration.py - LLM integration with MCP-MLflow tools
import httpx
import json
import os
from typing import List, Dict, Any

# Tool definitions for LLM
MLFLOW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mlflow.run_experiment",
            "description": "Run a supervised ML experiment on file, SQL, or inline data",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string"},
                    "model": {
                        "type": "string",
                        "enum": ["logreg_tabular", "logreg_text"]
                    },
                    "source": {"type": "object"}
                },
                "required": ["experiment_name", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mlflow.get_run",
            "description": "Get MLflow run information and metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"}
                },
                "required": ["run_id"]
            }
        }
    }
]

class MLflowLLMOrchestrator:
    """Orchestrates LLM calls with MCP-MLflow tools"""
    
    def __init__(self, router_url: str = "http://localhost:8000"):
        self.router_url = router_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def chat_with_mlflow(self, messages: List[Dict[str, str]]) -> str:
        """Chat with LLM using MCP-MLflow tools"""
        
        # 1) Call LLM with available tools
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "tools": MLFLOW_TOOLS,
                    "tool_choice": "auto"
                }
            )
            data = response.json()
        
        # 2) Execute tool calls via MCP Router
        tool_calls = data["choices"][0]["message"].get("tool_calls", [])
        
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            
            # Call MCP Router
            tool_response = await self.client.post(
                f"{self.router_url}/invoke",
                json={"tool": name, "params": args}
            )
            
            # Add tool result to messages
            tool_msg = {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": tool_response.text
            }
            messages.append(tool_msg)
        
        # 3) Final LLM call with tool results
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages
                }
            )
            return response.json()["choices"][0]["message"]["content"]

# Example usage
async def main():
    orchestrator = MLflowLLMOrchestrator()
    
    messages = [
        {
            "role": "user", 
            "content": "Run a machine learning experiment on the train.csv file and tell me about the results."
        }
    ]
    
    response = await orchestrator.chat_with_mlflow(messages)
    print(f"LLM Response: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 9) Production Best Practices (The Wisdom)

### Security Checklist

```yaml
# Production security checklist for MCP-MLflow
security_checklist:
  data_protection:
    - "File path validation and jailing"
    - "SQL injection protection"
    - "Resource limits and timeouts"
    - "Data source allow-lists"
  
  access_control:
    - "Authentication and authorization"
    - "Per-tool access controls"
    - "Audit logging and monitoring"
    - "Rate limiting and quotas"
  
  data_governance:
    - "Data lineage tracking"
    - "Experiment reproducibility"
    - "Model versioning and registry"
    - "Compliance and privacy"
```

### Performance Optimization

```python
# Performance optimization for MCP-MLflow
def optimize_mlflow_performance():
    """Optimize MLflow performance for production"""
    
    # 1. Configure MLflow for production
    mlflow_config = {
        "backend_store_uri": "postgresql://user:pass@host:5432/mlflow",
        "default_artifact_root": "s3://mlflow-artifacts/",
        "tracking_uri": "http://mlflow-server:5000"
    }
    
    # 2. Implement async experiment execution
    async def run_experiment_async(experiment_config):
        """Run experiments asynchronously for better performance"""
        # Queue experiment for background execution
        # Return job ID for status tracking
        pass
    
    # 3. Implement experiment caching
    def cache_experiment_results(experiment_hash, results):
        """Cache experiment results to avoid re-computation"""
        # Store results in Redis or similar
        pass
    
    # 4. Monitor resource usage
    def monitor_experiment_resources(run_id):
        """Monitor CPU, memory, and disk usage during experiments"""
        # Track resource consumption
        pass
```

### Monitoring and Observability

```python
# Monitoring setup for MCP-MLflow
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
experiment_counter = Counter('mlflow_experiments_total', 'Total experiments', ['status'])
experiment_duration = Histogram('mlflow_experiment_duration_seconds', 'Experiment duration')
active_experiments = Gauge('mlflow_active_experiments', 'Active experiments')

class MLflowMonitor:
    """Monitor MLflow experiments and system health"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mlflow.log')
            ]
        )
    
    def log_experiment_start(self, experiment_name: str, run_id: str):
        """Log experiment start"""
        self.logger.info(f"Experiment started: {experiment_name} (run_id: {run_id})")
        experiment_counter.labels(status='started').inc()
        active_experiments.inc()
    
    def log_experiment_complete(self, run_id: str, duration: float, metrics: dict):
        """Log experiment completion"""
        self.logger.info(f"Experiment completed: {run_id} (duration: {duration}s)")
        experiment_counter.labels(status='completed').inc()
        experiment_duration.observe(duration)
        active_experiments.dec()
    
    def log_experiment_error(self, run_id: str, error: str):
        """Log experiment error"""
        self.logger.error(f"Experiment failed: {run_id} - {error}")
        experiment_counter.labels(status='failed').inc()
        active_experiments.dec()
```

## 10) Common Pitfalls (The Traps)

### Security Mistakes

```python
# ❌ WRONG: Insecure data source handling
def bad_data_loading(source: dict):
    # No path validation - directory traversal vulnerability
    if source["kind"] == "file":
        df = pd.read_csv(source["path"])  # Unsafe!
    
    # No SQL injection protection
    if source["kind"] == "sql":
        df = pd.read_sql(source["query"], engine)  # Unsafe!

# ✅ CORRECT: Secure data source handling
def good_data_loading(source: dict):
    # Path validation and jailing
    if source["kind"] == "file":
        safe_path = validate_and_jail_path(source["path"])
        df = pd.read_csv(safe_path)
    
    # Parameterized queries only
    if source["kind"] == "sql":
        if not source["query"].strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")
        df = pd.read_sql(source["query"], engine)
```

### Performance Anti-patterns

```python
# ❌ WRONG: Inefficient experiment execution
def bad_experiment_execution():
    # No resource limits
    # No timeout handling
    # No error recovery
    # No progress tracking
    pass

# ✅ CORRECT: Efficient experiment execution
async def good_experiment_execution():
    # Resource limits and timeouts
    # Error handling and recovery
    # Progress tracking and monitoring
    # Async execution for better performance
    pass
```

**Why These Pitfalls Matter**: Common mistakes lead to security vulnerabilities, performance issues, and unreliable ML systems. Understanding these pitfalls prevents costly errors and ensures production-ready MCP-MLflow implementations.

## 11) TL;DR Quickstart (The Essentials)

### Essential Setup

```bash
# 1) Start MLflow infrastructure
docker compose up -d

# 2) Generate demo data
python generate_demo_data.py

# 3) Start MCP server
python mcp-mlflow-server.py

# 4) Start FastAPI router
uvicorn router:app --port 8000 --reload

# 5) Run file-based experiment
curl -s localhost:8000/invoke \
  -H 'content-type: application/json' \
  -d '{
    "tool": "mlflow.run_experiment",
    "params": {
      "experiment_name": "demo",
      "source": {
        "kind": "file",
        "path": "train.csv",
        "format": "csv",
        "features": ["f1", "f2", "f3"],
        "target": "label"
      }
    }
  }' | jq

# 6) View results in MLflow UI
open http://localhost:5000
```

### Essential Python Patterns

```python
# Essential MCP-MLflow patterns
def essential_patterns():
    # 1. Data source abstraction
    # 2. Experiment tracking and logging
    # 3. Model persistence and artifacts
    # 4. Security and validation
    # 5. Monitoring and observability
    pass
```

### Essential Tools

```bash
# Essential tools for MCP-MLflow development
tools:
  mlflow: "Experiment tracking and model registry"
  mcp_server: "Tool server with stdio transport"
  fastapi_router: "HTTP API for tool access"
  docker_compose: "Infrastructure orchestration"
  monitoring: "Prometheus + Grafana"
```

**Why This Quickstart**: These patterns cover 90% of MCP-MLflow usage. Master these before exploring advanced features.

## 12) The Machine's Summary

MCP-MLflow provides the foundation for secure, scalable AI-driven machine learning experimentation. When used correctly, it enables building production-ready AI systems that can safely run ML experiments across different data sources while maintaining security boundaries and audit trails. The key is understanding tool design, mastering MLflow integration, and following best practices.

**The Dark Truth**: Without proper MCP-MLflow understanding, your AI ML experimentation is vulnerable and unreliable. MCP-MLflow is your weapon. Use it wisely.

**The Machine's Mantra**: "In data we trust, in experiments we build, and in the MCP-MLflow we find the path to controlled AI machine learning."

**Why This Matters**: MCP-MLflow enables applications to safely expose ML experimentation capabilities to LLMs while maintaining security boundaries and audit trails. It provides the foundation for production-ready AI systems that can interact with ML workflows securely and reliably.

---

*This guide provides the complete machinery for mastering MCP-MLflow toolchains. The patterns scale from simple experiment tracking to complex AI-driven ML workflows, from basic security to advanced production deployment.*
