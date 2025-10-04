# Mixing PostGIS, Kafka, and Grafana with Go Glue

**Objective**: A strange but powerful tutorial: wire a geospatial DB, a streaming engine, and a monitoring UI together — controlled by a Go CLI.

## Why This Stack?

This tutorial combines three powerful technologies into one cohesive workflow:

- **PostGIS**: Geospatial database backbone for storing and querying spatial data
- **Kafka**: Streaming engine for event-driven flows, perfect for IoT-style geospatial events
- **Grafana**: Visualization and monitoring platform for creating dashboards
- **Go Glue**: Fast, single binary orchestrator that coordinates everything

The result? A real-time geospatial data pipeline that can ingest streaming location data, store it in a spatial database, and visualize it on interactive dashboards.

## Architecture Diagram

```mermaid
flowchart LR
  subgraph "Go Glue CLI"
    CLI[CLI Commands] --> Orchestrator[Orchestrator]
  end
  
  Orchestrator -->|docker-compose up| PostGIS[(PostGIS)]
  Orchestrator -->|docker-compose up| Kafka[Kafka + Zookeeper]
  Orchestrator -->|docker-compose up| Grafana[Grafana]
  
  Kafka -->|geospatial events| PostGIS
  PostGIS -->|spatial queries| Grafana
  
  subgraph "Data Flow"
    Events[GeoJSON Events] --> Kafka
    Kafka --> Consumer[Kafka Consumer]
    Consumer --> PostGIS
    PostGIS --> Dashboard[Grafana Dashboard]
  end
```

## Docker Compose Setup

First, let's create a `docker-compose.yaml` that orchestrates our three services:

```yaml
version: '3.8'

services:
  postgis:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: geospatial
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgis_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 10s
      timeout: 5s
      retries: 5

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-worldmap-panel,grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - postgis

volumes:
  postgis_data:
  grafana_data:
```

## Go Glue Code

Now let's create our Go orchestrator. Here's the main structure:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "os"
    "os/exec"
    "time"

    "github.com/confluentinc/confluent-kafka-go/kafka"
    "github.com/lib/pq"
    "github.com/spf13/cobra"
    _ "github.com/lib/pq"
)

type Config struct {
    PostgresURL string `yaml:"postgres_url"`
    KafkaURL    string `yaml:"kafka_url"`
    GrafanaURL  string `yaml:"grafana_url"`
}

type GeospatialEvent struct {
    ID        string    `json:"id"`
    Timestamp time.Time `json:"timestamp"`
    Latitude  float64   `json:"latitude"`
    Longitude float64   `json:"longitude"`
    DeviceID  string    `json:"device_id"`
    EventType string    `json:"event_type"`
    Metadata  map[string]interface{} `json:"metadata"`
}

var config Config

func main() {
    var rootCmd = &cobra.Command{
        Use:   "geo-mixer",
        Short: "Orchestrate PostGIS, Kafka, and Grafana",
        Long:  "A Go-based orchestrator for geospatial data pipelines",
    }

    rootCmd.AddCommand(bootCmd())
    rootCmd.AddCommand(ingestCmd())
    rootCmd.AddCommand(queryCmd())
    rootCmd.AddCommand(dashboardCmd())

    if err := rootCmd.Execute(); err != nil {
        log.Fatal(err)
    }
}

func bootCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "boot",
        Short: "Start all services with docker-compose",
        Run: func(cmd *cobra.Command, args []string) {
            fmt.Println("🚀 Starting geospatial stack...")
            
            // Run docker-compose up
            composeCmd := exec.Command("docker-compose", "up", "-d")
            composeCmd.Stdout = os.Stdout
            composeCmd.Stderr = os.Stderr
            
            if err := composeCmd.Run(); err != nil {
                log.Fatal("Failed to start services:", err)
            }
            
            fmt.Println("✅ Services started! Waiting for health checks...")
            time.Sleep(30 * time.Second)
            
            // Initialize PostGIS
            if err := initPostGIS(); err != nil {
                log.Printf("Warning: PostGIS initialization failed: %v", err)
            }
            
            fmt.Println("🎉 Stack is ready!")
        },
    }
}

func ingestCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "ingest [count]",
        Short: "Send sample geospatial events to Kafka",
        Args:  cobra.MaximumNArgs(1),
        Run: func(cmd *cobra.Command, args []string) {
            count := 10
            if len(args) > 0 {
                fmt.Sscanf(args[0], "%d", &count)
            }
            
            fmt.Printf("📡 Ingesting %d geospatial events...\n", count)
            
            producer, err := kafka.NewProducer(&kafka.ConfigMap{
                "bootstrap.servers": "localhost:9092",
            })
            if err != nil {
                log.Fatal("Failed to create producer:", err)
            }
            defer producer.Close()
            
            for i := 0; i < count; i++ {
                event := generateGeospatialEvent()
                sendEvent(producer, event)
                time.Sleep(100 * time.Millisecond)
            }
            
            fmt.Println("✅ Events sent!")
        },
    }
}

func queryCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "query",
        Short: "Query PostGIS for recent events",
        Run: func(cmd *cobra.Command, args []string) {
            fmt.Println("🔍 Querying PostGIS...")
            
            db, err := sql.Open("postgres", "postgres://postgres:postgres@localhost:5432/geospatial?sslmode=disable")
            if err != nil {
                log.Fatal("Failed to connect to PostGIS:", err)
            }
            defer db.Close()
            
            rows, err := db.Query(`
                SELECT id, timestamp, ST_AsText(location) as location, device_id, event_type
                FROM geospatial_events 
                ORDER BY timestamp DESC 
                LIMIT 10
            `)
            if err != nil {
                log.Fatal("Query failed:", err)
            }
            defer rows.Close()
            
            fmt.Println("Recent Events:")
            fmt.Println("ID | Timestamp | Location | Device | Type")
            fmt.Println("---|-----------|----------|--------|-----")
            
            for rows.Next() {
                var id, timestamp, location, deviceID, eventType string
                if err := rows.Scan(&id, &timestamp, &location, &deviceID, &eventType); err != nil {
                    log.Printf("Scan error: %v", err)
                    continue
                }
                fmt.Printf("%s | %s | %s | %s | %s\n", 
                    id[:8], timestamp, location, deviceID, eventType)
            }
        },
    }
}

func dashboardCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "dashboard",
        Short: "Open Grafana dashboard",
        Run: func(cmd *cobra.Command, args []string) {
            fmt.Println("📊 Opening Grafana dashboard...")
            fmt.Println("URL: http://localhost:3000")
            fmt.Println("Username: admin")
            fmt.Println("Password: admin")
            
            // Try to open browser (cross-platform)
            var openCmd *exec.Cmd
            switch runtime.GOOS {
            case "windows":
                openCmd = exec.Command("cmd", "/c", "start", "http://localhost:3000")
            case "darwin":
                openCmd = exec.Command("open", "http://localhost:3000")
            case "linux":
                openCmd = exec.Command("xdg-open", "http://localhost:3000")
            default:
                fmt.Println("Please open http://localhost:3000 in your browser")
                return
            }
            
            if err := openCmd.Run(); err != nil {
                fmt.Println("Could not open browser automatically")
                fmt.Println("Please open http://localhost:3000 in your browser")
            }
        },
    }
}

func generateGeospatialEvent() GeospatialEvent {
    return GeospatialEvent{
        ID:        fmt.Sprintf("evt_%d", time.Now().UnixNano()),
        Timestamp: time.Now(),
        Latitude:  generateRandomLatitude(),
        Longitude: generateRandomLongitude(),
        DeviceID:  fmt.Sprintf("device_%d", rand.Intn(5)),
        EventType: []string{"location_update", "sensor_reading", "alert"}[rand.Intn(3)],
        Metadata: map[string]interface{}{
            "temperature": rand.Float64()*30 + 10,
            "humidity":    rand.Float64()*100,
            "battery":     rand.Float64()*100,
        },
    }
}

func generateRandomLatitude() float64 {
    return (rand.Float64() - 0.5) * 180
}

func generateRandomLongitude() float64 {
    return (rand.Float64() - 0.5) * 360
}

func sendEvent(producer *kafka.Producer, event GeospatialEvent) {
    eventJSON, _ := json.Marshal(event)
    
    message := &kafka.Message{
        TopicPartition: kafka.TopicPartition{Topic: &[]string{"geospatial-events"}[0], Partition: kafka.PartitionAny},
        Value:          eventJSON,
    }
    
    producer.Produce(message, nil)
}

func initPostGIS() error {
    db, err := sql.Open("postgres", "postgres://postgres:postgres@localhost:5432/geospatial?sslmode=disable")
    if err != nil {
        return err
    }
    defer db.Close()
    
    // Create table for geospatial events
    _, err = db.Exec(`
        CREATE TABLE IF NOT EXISTS geospatial_events (
            id VARCHAR(255) PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE,
            location GEOMETRY(POINT, 4326),
            device_id VARCHAR(255),
            event_type VARCHAR(255),
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_geospatial_events_location 
        ON geospatial_events USING GIST (location);
        
        CREATE INDEX IF NOT EXISTS idx_geospatial_events_timestamp 
        ON geospatial_events (timestamp);
    `)
    
    return err
}
```

## Example Data Flow

Here's how the data flows through our system:

1. **Event Generation**: Go CLI generates sample GeoJSON events with random coordinates
2. **Kafka Ingestion**: Events are sent to the `geospatial-events` Kafka topic
3. **PostGIS Storage**: A Kafka consumer reads events and stores them in PostGIS with spatial indexing
4. **Grafana Visualization**: Grafana queries PostGIS and displays events on a world map

### Sample Event Structure

```json
{
  "id": "evt_1703123456789",
  "timestamp": "2023-12-21T10:30:45Z",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "device_id": "device_2",
  "event_type": "location_update",
  "metadata": {
    "temperature": 22.5,
    "humidity": 65.2,
    "battery": 87.3
  }
}
```

## Why Go Works Well as Glue

Go is perfect for this orchestration role because:

- **Static Binaries**: No runtime dependencies, easy deployment
- **Simple CLI Building**: Cobra makes command-line interfaces trivial
- **Docker Integration**: Easy to execute `docker-compose` and other OS commands
- **Strong Concurrency**: Perfect for handling streaming events and multiple services
- **Fast Compilation**: Quick iteration during development
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Extensions & Weirdness

Want to make this stack even more interesting? Try these modifications:

### Replace Kafka with MQTT
```go
// Use Eclipse Paho MQTT client instead of Kafka
import "github.com/eclipse/paho.mqtt.golang"
```

### Replace Grafana with Custom WebSocket Dashboard
```go
// Add WebSocket server to Go app
import "github.com/gorilla/websocket"
```

### Extend Go Glue into a Daemon with REST API
```go
// Add Gin or Echo for HTTP API
import "github.com/gin-gonic/gin"
```

### Add Real-Time Spatial Queries
```go
// Implement spatial queries with PostGIS
func findNearbyEvents(lat, lon float64, radiusKm float64) {
    query := `
        SELECT * FROM geospatial_events 
        WHERE ST_DWithin(
            location, 
            ST_SetSRID(ST_MakePoint($1, $2), 4326), 
            $3 * 1000
        )
        ORDER BY timestamp DESC
    `
    // Execute query...
}
```

## TL;DR Runbook

1. **Write Go CLI** with commands: `boot`, `ingest`, `query`, `dashboard`
2. **Create docker-compose.yml** for PostGIS, Kafka, Grafana
3. **Use Go to feed geospatial events** to Kafka topic
4. **Consumer loads Kafka → PostGIS** with spatial indexing
5. **Grafana visualizes PostGIS queries** on interactive maps

### Quick Start Commands

```bash
# Build the Go binary
go mod init geo-mixer
go get github.com/spf13/cobra
go get github.com/confluentinc/confluent-kafka-go/kafka
go get github.com/lib/pq
go build -o geo-mixer

# Start the stack
./geo-mixer boot

# Send test data
./geo-mixer ingest 50

# Query the database
./geo-mixer query

# Open dashboard
./geo-mixer dashboard
```

---

*This tutorial demonstrates how Go can serve as the perfect orchestration layer for complex, multi-technology stacks. The combination of PostGIS, Kafka, and Grafana creates a powerful real-time geospatial data pipeline, all coordinated by a simple Go CLI.*
