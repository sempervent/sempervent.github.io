# Real-Time Anomaly Radar with Fastify, Kafka, ClickHouse, Rust→WASM, and WebGPU

**Objective**: Build an end-to-end anomaly radar for streaming telemetry using a modern stack that combines JavaScript backend orchestration, event streaming, columnar analytics, WebAssembly performance, and GPU-accelerated visualization. When you need to detect anomalies in real-time data streams, when you want to build high-performance analytics pipelines, when you're curious about the intersection of streaming data and GPU computing—Fastify + Kafka + ClickHouse + Rust→WASM + WebGPU becomes your weapon of choice.

Use a JavaScript backend as the conductor. Kafka is the pulse, ClickHouse the memory, Rust→WASM the sharp knife, WebGPU the eyes.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand streaming data architecture**
   - Event-driven systems and message queues
   - Real-time data processing and analytics
   - Time-series data storage and querying
   - Anomaly detection algorithms and techniques

2. **Master the technology stack**
   - Fastify for high-performance Node.js backends
   - Kafka for durable event streaming
   - ClickHouse for columnar time-series analytics
   - Rust→WASM for performance-critical computations
   - WebGPU for GPU-accelerated visualization

3. **Know your anomaly detection**
   - Statistical methods (z-score, moving averages)
   - Signal processing techniques (FFT, spectral analysis)
   - Machine learning approaches (isolation forests, autoencoders)
   - Real-time scoring and threshold management

4. **Validate everything**
   - Test streaming data flow and reliability
   - Verify anomaly detection accuracy and performance
   - Check WebGPU rendering and browser compatibility
   - Monitor system performance and resource usage

5. **Plan for production**
   - Design for scalability and fault tolerance
   - Enable real-time monitoring and alerting
   - Support multiple data sources and formats
   - Document operational procedures and troubleshooting

**Why These Principles**: Real-time anomaly detection requires understanding distributed systems, statistical analysis, and GPU computing. Understanding these patterns prevents detection failures and enables reliable anomaly radar systems.

## 1) Architecture Diagram

### System Flow

```mermaid
flowchart LR
    P[Producers] --> K[(Kafka)]
    K -->|consumer| F[Fastify (Node.js)]
    F -->|WASM call| W[Anomaly Core (Rust→WASM)]
    F -->|INSERT| C[(ClickHouse)]
    F -->|WebSocket| B[Browser]
    B -->|WebGPU| V[Anomaly Visualization]
    
    subgraph "Data Flow"
        G[Telemetry Data] --> H[Kafka Topic]
        H --> I[Fastify Consumer]
        I --> J[WASM Scoring]
        J --> K[ClickHouse Storage]
        K --> L[WebSocket Broadcast]
        L --> M[WebGPU Rendering]
    end
    
    subgraph "Anomaly Detection"
        N[Statistical Analysis]
        O[Signal Processing]
        P[Threshold Detection]
    end
    
    J --> N
    J --> O
    J --> P
```

**Why Architecture Diagrams Matter**: Visual representation of data flow enables understanding of system components. Understanding these patterns prevents architectural confusion and enables reliable anomaly detection systems.

## 2) Infrastructure Setup with Docker Compose

### Complete Docker Compose Stack

```yaml
# docker-compose.yml
version: "3.9"
services:
  zookeeper:
    image: bitnami/zookeeper:3.9
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ALLOW_ANONYMOUS_LOGIN: "yes"
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/bitnami/zookeeper
    restart: unless-stopped

  kafka:
    image: bitnami/kafka:3.7
    container_name: kafka
    environment:
      KAFKA_ENABLE_KRAFT: "no"
      KAFKA_CFG_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CFG_LISTENERS: PLAINTEXT://:9092
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      ALLOW_PLAINTEXT_LISTENER: "yes"
      KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: "true"
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
    volumes:
      - kafka_data:/bitnami/kafka
    restart: unless-stopped

  clickhouse:
    image: clickhouse/clickhouse-server:24.8
    container_name: clickhouse
    ports:
      - "8123:8123"
      - "9000:9000"
    environment:
      CLICKHOUSE_DB: anomaly_radar
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    restart: unless-stopped

volumes:
  zookeeper_data:
  kafka_data:
  clickhouse_data:
```

**Why Docker Compose Matters**: Complete infrastructure stack enables reliable streaming data processing. Understanding these patterns prevents setup issues and enables consistent development environments.

### ClickHouse Schema Setup

```sql
-- Connect to ClickHouse: http://localhost:8123/play

-- Create database
CREATE DATABASE IF NOT EXISTS anomaly_radar;

-- Telemetry data table
CREATE TABLE IF NOT EXISTS anomaly_radar.telemetry
(
    ts       DateTime64(3, 'UTC'),
    device   LowCardinality(String),
    value    Float64,
    metadata String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (device, ts)
SETTINGS index_granularity = 8192;

-- Anomaly results table
CREATE TABLE IF NOT EXISTS anomaly_radar.anomalies
(
    ts       DateTime64(3, 'UTC'),
    device   LowCardinality(String),
    score    Float64,
    is_anom  UInt8,
    method   LowCardinality(String) DEFAULT 'zscore',
    metadata String DEFAULT ''
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (device, ts)
SETTINGS index_granularity = 8192;

-- Create materialized view for real-time aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS anomaly_radar.telemetry_stats
ENGINE = SummingMergeTree()
PARTITION BY toDate(ts)
ORDER BY (device, toStartOfMinute(ts))
AS SELECT
    toStartOfMinute(ts) as ts,
    device,
    count() as count,
    avg(value) as avg_value,
    stddevPop(value) as stddev_value,
    min(value) as min_value,
    max(value) as max_value
FROM anomaly_radar.telemetry
GROUP BY device, toStartOfMinute(ts);
```

**Why ClickHouse Schema Matters**: Columnar storage enables fast time-series analytics. Understanding these patterns prevents query performance issues and enables efficient anomaly detection.

## 3) Rust → WASM Anomaly Core

### Rust Library Setup

```toml
# anomaly-core/Cargo.toml
[package]
name = "anomaly-core"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"

[dependencies.web-sys]
version = "0.3"
features = [
  "console",
]
```

### Rust Anomaly Detection Implementation

```rust
// anomaly-core/src/lib.rs
use wasm_bindgen::prelude::*;
use js_sys::Math;

// Sliding window statistics for online anomaly detection
#[wasm_bindgen]
pub struct SlidingStats {
    values: Vec<f64>,
    max_size: usize,
    sum: f64,
    sum_sq: f64,
}

#[wasm_bindgen]
impl SlidingStats {
    #[wasm_bindgen(constructor)]
    pub fn new(max_size: usize) -> SlidingStats {
        SlidingStats {
            values: Vec::new(),
            max_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn add_value(&mut self, value: f64) {
        self.values.push(value);
        self.sum += value;
        self.sum_sq += value * value;

        if self.values.len() > self.max_size {
            let removed = self.values.remove(0);
            self.sum -= removed;
            self.sum_sq -= removed * removed;
        }
    }

    pub fn get_mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    pub fn get_std(&self) -> f64 {
        if self.values.len() <= 1 {
            0.0
        } else {
            let n = self.values.len() as f64;
            let mean = self.get_mean();
            let variance = (self.sum_sq / n) - (mean * mean);
            if variance > 0.0 {
                Math::sqrt(variance)
            } else {
                0.0
            }
        }
    }

    pub fn get_z_score(&self, value: f64) -> f64 {
        let mean = self.get_mean();
        let std = self.get_std();
        if std <= 1e-9 {
            0.0
        } else {
            (value - mean) / std
        }
    }

    pub fn is_anomaly(&self, value: f64, threshold: f64) -> bool {
        let z_score = self.get_z_score(value);
        z_score.abs() > threshold
    }
}

// Fast Fourier Transform for spectral analysis
#[wasm_bindgen]
pub fn fft_anomaly_detection(values: &[f64], window_size: usize) -> f64 {
    if values.len() < window_size {
        return 0.0;
    }

    let window = &values[values.len() - window_size..];
    let mean = window.iter().sum::<f64>() / window_size as f64;
    
    // Simple spectral analysis using variance
    let variance = window.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / window_size as f64;
    
    // Return spectral energy as anomaly score
    Math::sqrt(variance)
}

// Z-score anomaly detection
#[wasm_bindgen]
pub fn zscore_anomaly(value: f64, mean: f64, std: f64, threshold: f64) -> (f64, bool) {
    if std <= 1e-9 {
        return (0.0, false);
    }
    let z_score = (value - mean) / std;
    (z_score, z_score.abs() > threshold)
}

// Moving average anomaly detection
#[wasm_bindgen]
pub fn moving_average_anomaly(value: f64, values: &[f64], threshold: f64) -> (f64, bool) {
    if values.is_empty() {
        return (0.0, false);
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = (value - mean).abs();
    let is_anomaly = deviation > threshold;
    
    (deviation, is_anomaly)
}

// Export functions for JavaScript
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}
```

### Build WASM Module

```bash
# Build Rust to WASM
cd anomaly-core
wasm-pack build --release --target nodejs

# This creates pkg/ directory with:
# - anomaly_core.js (JavaScript bindings)
# - anomaly_core_bg.wasm (WebAssembly binary)
# - anomaly_core.d.ts (TypeScript definitions)
```

**Why Rust→WASM Matters**: WebAssembly enables high-performance anomaly detection in the browser. Understanding these patterns prevents performance bottlenecks and enables real-time analytics.

## 4) Fastify TypeScript Backend

### Project Setup

```bash
# Initialize Node.js project
npm init -y
npm install -D typescript ts-node @types/node
npm install fastify @fastify/websocket kafkajs @clickhouse/client
npm install -D @types/kafkajs

# Initialize TypeScript
npx tsc --init
```

### Fastify Server Implementation

```typescript
// src/server.ts
import Fastify from "fastify";
import websocket from "@fastify/websocket";
import { Kafka } from "kafkajs";
import { ClickHouse } from "@clickhouse/client";
import * as wasm from "../anomaly-core/pkg/anomaly_core.js";

// Initialize Fastify
const app = Fastify({ 
  logger: { 
    level: 'info',
    transport: {
      target: 'pino-pretty'
    }
  } 
});

// Register WebSocket support
await app.register(websocket);

// Initialize Kafka
const kafka = new Kafka({ 
  brokers: ["localhost:9092"],
  clientId: "anomaly-radar"
});

const consumer = kafka.consumer({ groupId: "anomaly-consumers" });
const producer = kafka.producer();

// Initialize ClickHouse
const clickhouse = new ClickHouse({ 
  url: "http://localhost:8123",
  database: "anomaly_radar"
});

// Initialize WASM module
await wasm.init();

// Device statistics storage
const deviceStats = new Map<string, any>();

// WebSocket clients
const wsClients = new Set<any>();

// WebSocket endpoint
app.get("/ws", { websocket: true }, (connection, req) => {
  wsClients.add(connection);
  console.log("WebSocket client connected");
  
  connection.socket.on("close", () => {
    wsClients.delete(connection);
    console.log("WebSocket client disconnected");
  });
});

// REST API endpoints
app.get("/health", async (request, reply) => {
  return { status: "healthy", timestamp: new Date().toISOString() };
});

app.get("/devices", async (request, reply) => {
  try {
    const result = await clickhouse.query({
      query: "SELECT DISTINCT device FROM telemetry ORDER BY device",
      format: "JSONEachRow"
    });
    return await result.json();
  } catch (error) {
    reply.code(500).send({ error: "Failed to fetch devices" });
  }
});

app.get("/latest/:device", async (request, reply) => {
  const { device } = request.params as { device: string };
  try {
    const result = await clickhouse.query({
      query: `
        SELECT ts, device, value, score, is_anom 
        FROM anomalies 
        WHERE device = {device:String} 
        ORDER BY ts DESC 
        LIMIT 100
      `,
      query_params: { device },
      format: "JSONEachRow"
    });
    return await result.json();
  } catch (error) {
    reply.code(500).send({ error: "Failed to fetch latest data" });
  }
});

// Kafka consumer setup
await consumer.connect();
await producer.connect();
await consumer.subscribe({ topic: "telemetry", fromBeginning: true });

// Process messages
await consumer.run({
  eachMessage: async ({ topic, partition, message }) => {
    try {
      const payload = JSON.parse(message.value?.toString() || "{}");
      const { ts, device, value } = payload;

      // Get or create device statistics
      if (!deviceStats.has(device)) {
        deviceStats.set(device, new wasm.SlidingStats(1000));
      }
      
      const stats = deviceStats.get(device);
      stats.add_value(value);

      // Calculate anomaly score
      const mean = stats.get_mean();
      const std = stats.get_std();
      const [zScore, isAnomaly] = wasm.zscore_anomaly(value, mean, std, 3.0);

      // Store in ClickHouse
      await clickhouse.insert({
        table: "telemetry",
        values: [{
          ts: new Date(ts),
          device,
          value,
          metadata: JSON.stringify({ partition, offset: message.offset })
        }],
        format: "JSONEachRow"
      });

      await clickhouse.insert({
        table: "anomalies",
        values: [{
          ts: new Date(ts),
          device,
          score: zScore,
          is_anom: isAnomaly ? 1 : 0,
          method: "zscore",
          metadata: JSON.stringify({ mean, std })
        }],
        format: "JSONEachRow"
      });

      // Broadcast to WebSocket clients
      const message = JSON.stringify({
        ts,
        device,
        value,
        score: zScore,
        is_anom: isAnomaly,
        mean,
        std
      });

      for (const client of wsClients) {
        if (client.socket.readyState === 1) { // WebSocket.OPEN
          client.socket.send(message);
        }
      }

      console.log(`Processed: ${device} = ${value} (anomaly: ${isAnomaly})`);
    } catch (error) {
      console.error("Error processing message:", error);
    }
  }
});

// Start server
const start = async () => {
  try {
    await app.listen({ port: 3000, host: "0.0.0.0" });
    console.log("Server running on http://localhost:3000");
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
};

start();
```

**Why Fastify Backend Matters**: High-performance Node.js backend enables real-time data processing. Understanding these patterns prevents performance bottlenecks and enables reliable anomaly detection.

## 5) Telemetry Producer

### Synthetic Data Generator

```typescript
// scripts/producer.ts
import { Kafka } from "kafkajs";

const kafka = new Kafka({ 
  brokers: ["localhost:9092"],
  clientId: "telemetry-producer"
});

const producer = kafka.producer();

// Gaussian noise generator
function gaussian(mean: number, std: number): number {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + std * z;
}

// Generate synthetic telemetry data
function generateTelemetry(device: string): any {
  const baseValue = 50 + Math.sin(Date.now() / 10000) * 10;
  const noise = gaussian(0, 5);
  
  // Occasional spikes (2% probability)
  const spike = Math.random() < 0.02 ? gaussian(30, 10) : 0;
  
  const value = baseValue + noise + spike;
  
  return {
    ts: Date.now(),
    device,
    value: Math.round(value * 100) / 100,
    metadata: {
      source: "synthetic",
      version: "1.0"
    }
  };
}

async function startProducing() {
  await producer.connect();
  console.log("Producer connected");

  const devices = ["sensor-001", "sensor-002", "sensor-003", "sensor-004"];
  
  setInterval(async () => {
    for (const device of devices) {
      const telemetry = generateTelemetry(device);
      
      try {
        await producer.send({
          topic: "telemetry",
          messages: [{
            key: device,
            value: JSON.stringify(telemetry),
            timestamp: telemetry.ts.toString()
          }]
        });
        
        console.log(`Sent: ${device} = ${telemetry.value}`);
      } catch (error) {
        console.error("Error sending message:", error);
      }
    }
  }, 1000); // Send every second
}

startProducing().catch(console.error);
```

**Why Telemetry Producer Matters**: Synthetic data generation enables testing and validation. Understanding these patterns prevents data generation issues and enables reliable anomaly detection testing.

## 6) WebGPU Frontend

### HTML Structure

```html
<!-- public/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anomaly Radar</title>
    <style>
        html, body, canvas {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            display: block;
            background: #0b0b0f;
            overflow: hidden;
        }
        
        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            z-index: 100;
        }
        
        #stats {
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="controls">
        <h3>Anomaly Radar</h3>
        <div id="connection-status">Connecting...</div>
    </div>
    
    <div id="stats">
        <div id="device-count">Devices: 0</div>
        <div id="anomaly-count">Anomalies: 0</div>
        <div id="fps">FPS: 0</div>
    </div>
    
    <canvas id="canvas"></canvas>
    <script type="module" src="main.js"></script>
</body>
</html>
```

### WebGPU Visualization

```javascript
// public/main.js
class AnomalyRadar {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.buffer = null;
        this.data = [];
        this.frameCount = 0;
        this.lastTime = 0;
        
        this.init();
    }
    
    async init() {
        // Setup WebSocket connection
        this.ws = new WebSocket('ws://localhost:3000/ws');
        this.ws.onmessage = (event) => this.handleMessage(event);
        this.ws.onopen = () => {
            document.getElementById('connection-status').textContent = 'Connected';
        };
        this.ws.onclose = () => {
            document.getElementById('connection-status').textContent = 'Disconnected';
        };
        
        // Initialize WebGPU
        await this.initWebGPU();
        
        // Start render loop
        this.render();
    }
    
    async initWebGPU() {
        // Get WebGPU adapter and device
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        
        // Setup canvas context
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });
        
        // Create shader module
        const shaderModule = this.device.createShaderModule({
            code: `
                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) color: vec4f,
                }
                
                struct DataPoint {
                    x: f32,
                    y: f32,
                    is_anomaly: f32,
                }
                
                @group(0) @binding(0) var<storage, read> data: array<DataPoint>;
                
                @vertex
                fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                    let point = data[vertexIndex];
                    let x = (point.x % 10.0) / 5.0 - 1.0; // Wrap 10s window
                    let y = (point.y / 150.0) - 0.5;
                    
                    var color = vec4f(0.0, 1.0, 0.0, 1.0); // Green for normal
                    if (point.is_anomaly > 0.5) {
                        color = vec4f(1.0, 0.0, 0.0, 1.0); // Red for anomaly
                    }
                    
                    return VertexOutput(
                        vec4f(x, y, 0.0, 1.0),
                        color
                    );
                }
                
                @fragment
                fn fs(input: VertexOutput) -> @location(0) vec4f {
                    return input.color;
                }
            `
        });
        
        // Create render pipeline
        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs'
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: {
                topology: 'point-list'
            }
        });
        
        // Create data buffer
        this.buffer = this.device.createBuffer({
            size: 12 * 4096, // 12 bytes per DataPoint * 4096 points
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
    }
    
    handleMessage(event) {
        const message = JSON.parse(event.data);
        this.data.push({
            x: performance.now() * 0.001,
            y: message.value,
            is_anomaly: message.is_anom ? 1.0 : 0.0
        });
        
        // Keep only last 2048 points
        if (this.data.length > 2048) {
            this.data.shift();
        }
        
        // Update stats
        this.updateStats();
    }
    
    updateStats() {
        const devices = new Set(this.data.map(d => d.device));
        const anomalies = this.data.filter(d => d.is_anomaly > 0.5).length;
        
        document.getElementById('device-count').textContent = `Devices: ${devices.size}`;
        document.getElementById('anomaly-count').textContent = `Anomalies: ${anomalies}`;
    }
    
    render() {
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;
        const fps = Math.round(1000 / deltaTime);
        
        document.getElementById('fps').textContent = `FPS: ${fps}`;
        this.lastTime = currentTime;
        
        // Update buffer with current data
        if (this.data.length > 0) {
            const bufferData = new Float32Array(this.data.length * 3);
            for (let i = 0; i < this.data.length; i++) {
                bufferData[i * 3] = this.data[i].x;
                bufferData[i * 3 + 1] = this.data[i].y;
                bufferData[i * 3 + 2] = this.data[i].is_anomaly;
            }
            
            this.device.queue.writeBuffer(this.buffer, 0, bufferData);
        }
        
        // Render
        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1.0 }
            }]
        });
        
        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.buffer }
            }]
        }));
        
        renderPass.draw(this.data.length);
        renderPass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        requestAnimationFrame(() => this.render());
    }
}

// Initialize the radar
new AnomalyRadar();
```

**Why WebGPU Visualization Matters**: GPU-accelerated rendering enables real-time anomaly visualization. Understanding these patterns prevents rendering bottlenecks and enables smooth real-time graphics.

## 7) Operations and Monitoring

### ClickHouse Queries

```sql
-- Real-time anomaly detection queries
SELECT 
    device,
    count() as total_points,
    sum(is_anom) as anomaly_count,
    avg(score) as avg_score,
    max(score) as max_score
FROM anomalies 
WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY device
ORDER BY anomaly_count DESC;

-- Anomaly timeline
SELECT 
    toStartOfMinute(ts) as minute,
    device,
    count() as points,
    sum(is_anom) as anomalies
FROM anomalies 
WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY minute, device
ORDER BY minute DESC;

-- Device health check
SELECT 
    device,
    max(ts) as last_seen,
    count() as recent_points
FROM telemetry 
WHERE ts > now() - INTERVAL 5 MINUTE
GROUP BY device
ORDER BY last_seen DESC;
```

### Performance Monitoring

```typescript
// src/monitoring.ts
import { ClickHouse } from "@clickhouse/client";

class PerformanceMonitor {
    private clickhouse: ClickHouse;
    private metrics: Map<string, number> = new Map();
    
    constructor(clickhouse: ClickHouse) {
        this.clickhouse = clickhouse;
    }
    
    async recordMetric(name: string, value: number) {
        this.metrics.set(name, value);
        
        await this.clickhouse.insert({
            table: "performance_metrics",
            values: [{
                ts: new Date(),
                metric: name,
                value: value
            }],
            format: "JSONEachRow"
        });
    }
    
    async getSystemHealth() {
        const queries = [
            "SELECT count() FROM telemetry WHERE ts > now() - INTERVAL 1 MINUTE",
            "SELECT count() FROM anomalies WHERE ts > now() - INTERVAL 1 MINUTE",
            "SELECT avg(score) FROM anomalies WHERE ts > now() - INTERVAL 1 MINUTE"
        ];
        
        const results = await Promise.all(
            queries.map(q => this.clickhouse.query({ query: q, format: "JSONEachRow" }))
        );
        
        return {
            telemetry_rate: await results[0].json(),
            anomaly_rate: await results[1].json(),
            avg_anomaly_score: await results[2].json()
        };
    }
}
```

**Why Operations Matter**: Monitoring enables system health and performance optimization. Understanding these patterns prevents system failures and enables reliable anomaly detection.

## 8) Security and Hardening

### Kafka Security

```yaml
# kafka-security.yml
version: "3.9"
services:
  kafka:
    image: bitnami/kafka:3.7
    environment:
      KAFKA_CFG_SECURITY_INTER_BROKER_PROTOCOL: SASL_SSL
      KAFKA_CFG_SASL_ENABLED_MECHANISMS: PLAIN
      KAFKA_CFG_SASL_MECHANISM_INTER_BROKER_PROTOCOL: PLAIN
      KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,SASL_SSL:SASL_SSL
      KAFKA_CFG_LISTENERS: SASL_SSL://:9093,PLAINTEXT://:9092
      KAFKA_CFG_ADVERTISED_LISTENERS: SASL_SSL://localhost:9093,PLAINTEXT://localhost:9092
```

### Fastify Security

```typescript
// src/security.ts
import fastify from "fastify";
import rateLimit from "@fastify/rate-limit";
import helmet from "@fastify/helmet";

const app = fastify();

// Security headers
await app.register(helmet);

// Rate limiting
await app.register(rateLimit, {
    max: 100,
    timeWindow: '1 minute'
});

// Input validation
app.addSchema({
    $id: 'telemetry',
    type: 'object',
    properties: {
        device: { type: 'string', minLength: 1, maxLength: 50 },
        value: { type: 'number' },
        ts: { type: 'number' }
    },
    required: ['device', 'value', 'ts']
});
```

**Why Security Matters**: Production systems require security hardening. Understanding these patterns prevents security vulnerabilities and enables reliable anomaly detection.

## 9) Testing and Validation

### Unit Tests

```typescript
// tests/anomaly.test.ts
import { describe, it, expect } from 'vitest';
import * as wasm from '../anomaly-core/pkg/anomaly_core.js';

describe('Anomaly Detection', () => {
    it('should detect z-score anomalies', () => {
        const [score, isAnomaly] = wasm.zscore_anomaly(100, 50, 10, 3.0);
        expect(score).toBe(5.0);
        expect(isAnomaly).toBe(true);
    });
    
    it('should handle edge cases', () => {
        const [score, isAnomaly] = wasm.zscore_anomaly(50, 50, 0, 3.0);
        expect(score).toBe(0.0);
        expect(isAnomaly).toBe(false);
    });
});
```

### Integration Tests

```typescript
// tests/integration.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Kafka } from 'kafkajs';

describe('Kafka Integration', () => {
    let kafka: Kafka;
    
    beforeAll(async () => {
        kafka = new Kafka({ brokers: ['localhost:9092'] });
    });
    
    it('should produce and consume messages', async () => {
        const producer = kafka.producer();
        const consumer = kafka.consumer({ groupId: 'test-group' });
        
        await producer.connect();
        await consumer.connect();
        
        await consumer.subscribe({ topic: 'test-topic' });
        
        // Produce message
        await producer.send({
            topic: 'test-topic',
            messages: [{ value: 'test message' }]
        });
        
        // Consume message
        const messages: any[] = [];
        await consumer.run({
            eachMessage: async ({ message }) => {
                messages.push(message.value?.toString());
            }
        });
        
        expect(messages).toContain('test message');
    });
});
```

**Why Testing Matters**: Comprehensive testing ensures system reliability. Understanding these patterns prevents bugs and enables reliable anomaly detection.

## 10) TL;DR Runbook

### Essential Commands

```bash
# Start infrastructure
docker-compose up -d

# Build Rust WASM module
cd anomaly-core
wasm-pack build --release --target nodejs

# Start Fastify server
npm run dev

# Start telemetry producer
npm run producer

# Open WebGPU visualization
open http://localhost:3000
```

### Essential Patterns

```typescript
// Essential anomaly detection patterns
const anomalyPatterns = {
    "kafka_streaming": "Kafka provides durable, replayable event streams",
    "clickhouse_analytics": "ClickHouse enables fast time-series analytics",
    "wasm_performance": "Rust→WASM enables high-performance anomaly detection",
    "webgpu_visualization": "WebGPU provides GPU-accelerated real-time rendering",
    "fastify_orchestration": "Fastify orchestrates the entire pipeline"
};
```

### Quick Reference

```typescript
// Essential anomaly detection operations
// 1. Start Kafka consumer
await consumer.subscribe({ topic: "telemetry" });

// 2. Process messages with WASM
const [score, isAnomaly] = wasm.zscore_anomaly(value, mean, std, threshold);

// 3. Store in ClickHouse
await clickhouse.insert({ table: "anomalies", values: [data] });

// 4. Broadcast via WebSocket
wsClients.forEach(client => client.send(JSON.stringify(data)));

// 5. Render with WebGPU
renderPass.draw(data.length);
```

**Why This Runbook**: These patterns cover 90% of anomaly detection needs. Master these before exploring advanced streaming analytics scenarios.

## 11) The Machine's Summary

Real-time anomaly detection requires understanding distributed systems, statistical analysis, and GPU computing. When used correctly, this stack enables reliable anomaly detection, real-time visualization, and scalable analytics. The key is understanding data flow, mastering performance optimization, and following security best practices.

**The Dark Truth**: Without proper anomaly detection understanding, your data streams remain blind to critical events. This stack is your weapon. Use it wisely.

**The Machine's Mantra**: "In the streams we trust, in the Kafka we publish, in the ClickHouse we store, in the WASM we compute, and in the WebGPU we visualize the anomalies."

**Why This Matters**: Real-time anomaly detection enables efficient data monitoring that can handle complex streaming scenarios, maintain high performance, and provide immediate insights while ensuring system reliability and security.

---

*This tutorial provides the complete machinery for real-time anomaly detection. The patterns scale from simple statistical analysis to complex streaming analytics, from basic visualization to advanced GPU-accelerated rendering.*
