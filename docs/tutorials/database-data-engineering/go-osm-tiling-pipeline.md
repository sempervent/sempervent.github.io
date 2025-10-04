# Gluing the Real World: Go CLI for OSM → PostGIS → Tiles → Object Storage

**Objective**: A single Go binary conducts PBF fetch, checksum, PostGIS load, view materialization, tile generation/serving, and publishing. Concrete, durable, and debuggable.

## Architecture Diagram

```mermaid
flowchart LR
  A[OSM PBF Source] -->|fetch+sha256| B[(Staging FS)]
  B -->|osm2pgsql| C[(PostGIS)]
  C -->|SQL views + indexes| C2[(Generalized Layers)]
  C2 -->|tippecanoe| D[(MBTiles/PMTiles)]
  C -->|live MVT| M[Martin Tileserver]
  D -->|rclone/s3 cp| S[(S3/MinIO)]
  
  subgraph "Go CLI"
    F[fetch] --> L[load] --> I[index] --> T[tile] --> P[publish] --> X[purge]
  end
  
  F -->|metrics| G[Pushgateway]
  L -->|metrics| G
  T -->|metrics| G
```

## Docker Compose (PostGIS + Martin + Pushgateway)

```yaml
version: "3.9"

services:
  db:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: osm
    ports:
      - "5432:5432"
    volumes:
      - ./volumes/pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  martin:
    image: ghcr.io/maplibre/martin:latest
    command: ["--listen-addr", "0.0.0.0:3000", "postgresql://postgres:postgres@db:5432/osm"]
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "3000:3000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  pushgateway:
    image: prom/pushgateway:latest
    ports:
      - "9091:9091"
    volumes:
      - ./volumes/pushgateway:/pushgateway

volumes:
  pgdata:
  pushgateway:
```

**Why**: PostGIS is your compute nucleus; Martin for "live tiles" mode; Pushgateway shows state-of-practice instrumentation.

## Data Model & SQL (Indexes, Generalized Views)

### Base Tables via osm2pgsql

```sql
-- osm2pgsql creates these tables automatically:
-- planet_osm_point, planet_osm_line, planet_osm_polygon, planet_osm_roads

-- Add spatial indexes for performance
CREATE INDEX IF NOT EXISTS idx_roads_geom ON planet_osm_roads USING gist(way);
CREATE INDEX IF NOT EXISTS idx_line_geom ON planet_osm_line USING gist(way);
CREATE INDEX IF NOT EXISTS idx_polygon_geom ON planet_osm_polygon USING gist(way);

-- Analyze tables for query planner
ANALYZE planet_osm_roads;
ANALYZE planet_osm_line;
ANALYZE planet_osm_polygon;
```

### Generalized Views by Zoom Level

```sql
-- Roads for zoom 8 (simplified)
CREATE MATERIALIZED VIEW IF NOT EXISTS roads_z8 AS
SELECT 
    osm_id,
    highway,
    name,
    ST_SimplifyVW(ST_Transform(way, 3857), 30) AS geom
FROM planet_osm_roads
WHERE highway IS NOT NULL 
  AND highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary');

CREATE INDEX IF NOT EXISTS idx_roads_z8_geom ON roads_z8 USING gist(geom);

-- Roads for zoom 10 (less simplified)
CREATE MATERIALIZED VIEW IF NOT EXISTS roads_z10 AS
SELECT 
    osm_id,
    highway,
    name,
    ST_SimplifyVW(ST_Transform(way, 3857), 10) AS geom
FROM planet_osm_roads
WHERE highway IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_roads_z10_geom ON roads_z10 USING gist(geom);

-- Buildings for zoom 12
CREATE MATERIALIZED VIEW IF NOT EXISTS buildings_z12 AS
SELECT 
    osm_id,
    building,
    name,
    ST_Transform(way, 3857) AS geom
FROM planet_osm_polygon
WHERE building IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_buildings_z12_geom ON buildings_z12 USING gist(geom);

-- Refresh materialized views
REFRESH MATERIALIZED VIEW roads_z8;
REFRESH MATERIALIZED VIEW roads_z10;
REFRESH MATERIALIZED VIEW buildings_z12;
```

**Why**: Predictable performance, deterministic tile size, and clear separation per zoom tier.

## The Go CLI — Structure and Subcommands

### Main CLI Structure

```go
package main

import (
    "context"
    "flag"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/spf13/cobra"
    "github.com/spf13/viper"
)

type Config struct {
    OSM struct {
        URL    string `mapstructure:"url"`
        SHA256 string `mapstructure:"sha256"`
    } `mapstructure:"osm"`
    
    DB struct {
        DSN string `mapstructure:"dsn"`
    } `mapstructure:"db"`
    
    Tiling struct {
        MinZoom int `mapstructure:"min_zoom"`
        MaxZoom int `mapstructure:"max_zoom"`
        Layers  []Layer `mapstructure:"layers"`
    } `mapstructure:"tiling"`
    
    Publish struct {
        S3Bucket string `mapstructure:"s3_bucket"`
        Prefix   string `mapstructure:"prefix"`
    } `mapstructure:"publish"`
    
    Metrics struct {
        Pushgateway string `mapstructure:"pushgateway"`
    } `mapstructure:"metrics"`
}

type Layer struct {
    Name string `mapstructure:"name"`
    SQL  string `mapstructure:"sql"`
}

var (
    cfg Config
    runID string
)

func main() {
    runID = fmt.Sprintf("osmglue-%d", time.Now().Unix())
    
    var rootCmd = &cobra.Command{
        Use:   "osmglue",
        Short: "OSM → PostGIS → Tiles pipeline orchestrator",
        Long:  "A Go CLI that orchestrates the complete OSM data pipeline",
    }
    
    rootCmd.AddCommand(bootCmd())
    rootCmd.AddCommand(fetchCmd())
    rootCmd.AddCommand(loadCmd())
    rootCmd.AddCommand(indexCmd())
    rootCmd.AddCommand(tileCmd())
    rootCmd.AddCommand(serveCmd())
    rootCmd.AddCommand(publishCmd())
    rootCmd.AddCommand(statusCmd())
    
    // Load config
    viper.SetConfigName("pipeline")
    viper.SetConfigType("yaml")
    viper.AddConfigPath(".")
    viper.AutomaticEnv()
    
    if err := viper.ReadInConfig(); err != nil {
        log.Fatal("Failed to read config:", err)
    }
    
    if err := viper.Unmarshal(&cfg); err != nil {
        log.Fatal("Failed to unmarshal config:", err)
    }
    
    if err := rootCmd.Execute(); err != nil {
        log.Fatal(err)
    }
}
```

### Subcommands

```go
func bootCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "boot",
        Short: "Ensure docker services are running",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("boot_start", map[string]any{
                "run_id": runID,
                "services": []string{"postgis", "martin", "pushgateway"},
            })
            
            // Start docker compose
            if err := execWithRetry(ctx, Opts{
                Cmd:  "docker",
                Args: []string{"compose", "up", "-d"},
                Retries: 3,
            }); err != nil {
                log.Fatal("Failed to start services:", err)
            }
            
            // Wait for health checks
            time.Sleep(30 * time.Second)
            
            logJSON("boot_complete", map[string]any{
                "run_id": runID,
                "status": "healthy",
            })
        },
    }
}

func fetchCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "fetch",
        Short: "Download OSM PBF and verify checksum",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("fetch_start", map[string]any{
                "run_id": runID,
                "url": cfg.OSM.URL,
            })
            
            start := time.Now()
            
            // Download PBF with resume support
            if err := execWithRetry(ctx, Opts{
                Cmd:  "curl",
                Args: []string{"-C", "-", "-o", "data.osm.pbf", cfg.OSM.URL},
                Retries: 3,
            }); err != nil {
                log.Fatal("Failed to download PBF:", err)
            }
            
            // Download checksum
            if err := execWithRetry(ctx, Opts{
                Cmd:  "curl",
                Args: []string{"-sS", "-o", "data.osm.pbf.sha256", cfg.OSM.SHA256},
                Retries: 3,
            }); err != nil {
                log.Fatal("Failed to download checksum:", err)
            }
            
            // Verify checksum
            if err := execWithRetry(ctx, Opts{
                Cmd:  "sha256sum",
                Args: []string{"-c", "data.osm.pbf.sha256"},
                Retries: 1,
            }); err != nil {
                log.Fatal("Checksum verification failed:", err)
            }
            
            duration := time.Since(start)
            
            logJSON("fetch_complete", map[string]any{
                "run_id": runID,
                "duration_seconds": duration.Seconds(),
                "status": "verified",
            })
            
            // Push metrics
            pushMetrics("fetch", duration.Seconds())
        },
    }
}

func loadCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "load",
        Short: "Load OSM data into PostGIS via osm2pgsql",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("load_start", map[string]any{
                "run_id": runID,
                "pbf_path": "data.osm.pbf",
            })
            
            start := time.Now()
            
            // Run osm2pgsql in container
            if err := execWithRetry(ctx, Opts{
                Cmd: "docker",
                Args: []string{
                    "run", "--rm", "--network=host",
                    "-v", fmt.Sprintf("%s:/data.osm.pbf", "data.osm.pbf"),
                    "openmaptiles/osm2pgsql",
                    "--create", "--slim",
                    "-H", "localhost", "-d", "osm", "-U", "postgres",
                    "--cache", "20000", "--number-processes", "4",
                    "/data.osm.pbf",
                },
                Retries: 1,
            }); err != nil {
                log.Fatal("Failed to load OSM data:", err)
            }
            
            duration := time.Since(start)
            
            logJSON("load_complete", map[string]any{
                "run_id": runID,
                "duration_seconds": duration.Seconds(),
                "status": "loaded",
            })
            
            pushMetrics("load", duration.Seconds())
        },
    }
}

func indexCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "index",
        Short: "Create indexes and materialized views",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("index_start", map[string]any{
                "run_id": runID,
            })
            
            start := time.Now()
            
            // Execute SQL indexing script
            sqlScript := `
                -- Add spatial indexes
                CREATE INDEX IF NOT EXISTS idx_roads_geom ON planet_osm_roads USING gist(way);
                CREATE INDEX IF NOT EXISTS idx_line_geom ON planet_osm_line USING gist(way);
                CREATE INDEX IF NOT EXISTS idx_polygon_geom ON planet_osm_polygon USING gist(way);
                
                -- Analyze tables
                ANALYZE planet_osm_roads;
                ANALYZE planet_osm_line;
                ANALYZE planet_osm_polygon;
                
                -- Create materialized views
                CREATE MATERIALIZED VIEW IF NOT EXISTS roads_z8 AS
                SELECT osm_id, highway, name,
                       ST_SimplifyVW(ST_Transform(way, 3857), 30) AS geom
                FROM planet_osm_roads
                WHERE highway IS NOT NULL;
                
                CREATE INDEX IF NOT EXISTS idx_roads_z8_geom ON roads_z8 USING gist(geom);
                
                REFRESH MATERIALIZED VIEW roads_z8;
            `
            
            if err := execWithRetry(ctx, Opts{
                Cmd:  "psql",
                Args: []string{"-d", cfg.DB.DSN, "-c", sqlScript},
                Retries: 3,
            }); err != nil {
                log.Fatal("Failed to create indexes:", err)
            }
            
            duration := time.Since(start)
            
            logJSON("index_complete", map[string]any{
                "run_id": runID,
                "duration_seconds": duration.Seconds(),
                "status": "indexed",
            })
            
            pushMetrics("index", duration.Seconds())
        },
    }
}

func tileCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "tile",
        Short: "Generate tiles with tippecanoe",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("tile_start", map[string]any{
                "run_id": runID,
                "min_zoom": cfg.Tiling.MinZoom,
                "max_zoom": cfg.Tiling.MaxZoom,
            })
            
            start := time.Now()
            
            // Build tippecanoe command with layers
            args := []string{
                "-o", "tiles.mbtiles",
                "-Z", fmt.Sprintf("%d", cfg.Tiling.MinZoom),
                "-z", fmt.Sprintf("%d", cfg.Tiling.MaxZoom),
                "--drop-densest-as-needed",
                "--extend-zooms-if-still-dropping",
                "--no-feature-limit",
                "--no-tile-size-limit",
                "--read-parallel",
            }
            
            // Add layers
            for _, layer := range cfg.Tiling.Layers {
                args = append(args, "-L", fmt.Sprintf("name=%s,sql=%s", layer.Name, layer.SQL))
            }
            
            if err := execWithRetry(ctx, Opts{
                Cmd:  "tippecanoe",
                Args: args,
                Retries: 2,
            }); err != nil {
                log.Fatal("Failed to generate tiles:", err)
            }
            
            duration := time.Since(start)
            
            logJSON("tile_complete", map[string]any{
                "run_id": runID,
                "duration_seconds": duration.Seconds(),
                "output": "tiles.mbtiles",
                "status": "generated",
            })
            
            pushMetrics("tile", duration.Seconds())
        },
    }
}

func publishCmd() *cobra.Command {
    return &cobra.Command{
        Use:   "publish",
        Short: "Publish tiles to S3/MinIO",
        Run: func(cmd *cobra.Command, args []string) {
            ctx := contextWithCancel()
            
            logJSON("publish_start", map[string]any{
                "run_id": runID,
                "bucket": cfg.Publish.S3Bucket,
                "prefix": cfg.Publish.Prefix,
            })
            
            start := time.Now()
            
            // Use rclone to copy to S3/MinIO
            if err := execWithRetry(ctx, Opts{
                Cmd:  "rclone",
                Args: []string{"copy", "tiles.mbtiles", fmt.Sprintf("s3:%s/%s", cfg.Publish.S3Bucket, cfg.Publish.Prefix)},
                Retries: 3,
            }); err != nil {
                log.Fatal("Failed to publish tiles:", err)
            }
            
            duration := time.Since(start)
            
            logJSON("publish_complete", map[string]any{
                "run_id": runID,
                "duration_seconds": duration.Seconds(),
                "status": "published",
            })
            
            pushMetrics("publish", duration.Seconds())
        },
    }
}
```

## Go: Exec Helper, Backoff, Logging, Metrics

### Exec Helper with Retry Logic

```go
// cmd/internal/run/run.go
package run

import (
    "context"
    "encoding/json"
    "log"
    "math/rand"
    "os/exec"
    "time"
)

type Opts struct {
    Cmd     string
    Args    []string
    Env     []string
    Dir     string
    Retries int
}

func ExecWithRetry(ctx context.Context, o Opts) error {
    delay := 1 * time.Second
    maxDelay := 30 * time.Second
    
    for attempt := 0; attempt <= o.Retries; attempt++ {
        cmd := exec.CommandContext(ctx, o.Cmd, o.Args...)
        cmd.Dir = o.Dir
        cmd.Env = append(cmd.Env, o.Env...)
        
        out, err := cmd.CombinedOutput()
        
        logJSON("exec", map[string]any{
            "cmd":     o.Cmd,
            "args":    o.Args,
            "attempt": attempt,
            "out":     string(out),
            "err":     err,
        })
        
        if err == nil {
            return nil
        }
        
        if attempt == o.Retries {
            return err
        }
        
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(delay + time.Duration(rand.Intn(500))*time.Millisecond):
            delay *= 2
            if delay > maxDelay {
                delay = maxDelay
            }
        }
    }
    
    return context.DeadlineExceeded
}

func logJSON(ev string, payload map[string]any) {
    payload["event"] = ev
    payload["timestamp"] = time.Now().UTC().Format(time.RFC3339)
    payload["run_id"] = runID
    
    b, _ := json.Marshal(payload)
    log.Println(string(b))
}

func contextWithCancel() context.Context {
    ctx, cancel := context.WithCancel(context.Background())
    
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)
    
    go func() {
        <-c
        logJSON("shutdown", map[string]any{"signal": "interrupt"})
        cancel()
    }()
    
    return ctx
}
```

### Metrics Push to Pushgateway

```go
func pushMetrics(stage string, duration float64) {
    if cfg.Metrics.Pushgateway == "" {
        return
    }
    
    metrics := fmt.Sprintf(`
# TYPE stage_duration_seconds gauge
stage_duration_seconds{stage="%s",run_id="%s"} %f
`, stage, runID, duration)
    
    execWithRetry(context.Background(), Opts{
        Cmd: "curl",
        Args: []string{
            "-X", "POST",
            "--data-binary", metrics,
            fmt.Sprintf("%s/metrics/job/%s", cfg.Metrics.Pushgateway, stage),
        },
        Retries: 1,
    })
}
```

**Why**: One reliable glue primitive replaces brittle shell scripts; logs are structured; retry semantics are consistent.

## Config & Secrets

### pipeline.yaml

```yaml
osm:
  url: https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf
  sha256: https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf.sha256

db:
  dsn: postgresql://postgres:postgres@localhost:5432/osm?sslmode=disable

tiling:
  min_zoom: 5
  max_zoom: 12
  layers:
    - name: roads
      sql: SELECT osm_id, highway, name, geom FROM roads_z8
    - name: buildings
      sql: SELECT osm_id, building, name, geom FROM buildings_z12

publish:
  s3_bucket: my-maps
  prefix: tiles/

metrics:
  pushgateway: http://localhost:9091
```

### Environment Overrides

```bash
# Override config with environment variables
export OSM_URL="https://download.geofabrik.de/europe/germany-latest.osm.pbf"
export DB_DSN="postgresql://user:pass@db:5432/osm"
export S3_BUCKET="my-production-bucket"
export PUSHGATEWAY_URL="http://pushgateway:9091"
```

### Secrets Management

```bash
# Use AWS SSO for credentials
aws configure sso

# Or MinIO client
mc alias set myminio https://minio.example.com accesskey secretkey

# Never embed credentials in YAML
# Use rclone config for S3/MinIO authentication
rclone config
```

## Ops & Reliability Best Practices

### Idempotency

```go
// Success marker files per stage
func markStageComplete(stage string) error {
    return os.WriteFile(fmt.Sprintf(".%s.ok", stage), []byte(runID), 0644)
}

func isStageComplete(stage string) bool {
    _, err := os.Stat(fmt.Sprintf(".%s.ok", stage))
    return err == nil
}

// Skip stages unless --force flag
func shouldSkipStage(stage string, force bool) bool {
    return !force && isStageComplete(stage)
}
```

### Atomic Outputs

```go
// Write to temp file then move into place
func atomicWrite(filename string, data []byte) error {
    tempFile := filename + ".tmp"
    if err := os.WriteFile(tempFile, data, 0644); err != nil {
        return err
    }
    return os.Rename(tempFile, filename)
}
```

### Health Checks

```go
func waitForPostGIS(ctx context.Context, dsn string) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            db, err := sql.Open("postgres", dsn)
            if err == nil {
                if err := db.Ping(); err == nil {
                    db.Close()
                    return nil
                }
                db.Close()
            }
            time.Sleep(5 * time.Second)
        }
    }
}

func waitForMartin(ctx context.Context, url string) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            resp, err := http.Get(url + "/health")
            if err == nil && resp.StatusCode == 200 {
                resp.Body.Close()
                return nil
            }
            time.Sleep(5 * time.Second)
        }
    }
}
```

### Backpressure Control

```go
// Bound parallelism to CPU cores - 1
func getOptimalParallelism() int {
    cores := runtime.NumCPU()
    if cores > 1 {
        return cores - 1
    }
    return 1
}

// Use in tippecanoe
args := append(args, "--read-parallel", fmt.Sprintf("%d", getOptimalParallelism()))
```

### Observability

```go
type StageMetrics struct {
    Stage     string    `json:"stage"`
    StartTime time.Time `json:"start_time"`
    EndTime   time.Time `json:"end_time"`
    Duration  float64   `json:"duration_seconds"`
    Bytes     int64     `json:"bytes_processed"`
    Tiles     int       `json:"tiles_generated"`
    Status    string    `json:"status"`
}

func recordStageMetrics(stage string, start time.Time, bytes int64, tiles int) {
    duration := time.Since(start).Seconds()
    
    metrics := StageMetrics{
        Stage:     stage,
        StartTime: start,
        EndTime:   time.Now(),
        Duration:  duration,
        Bytes:     bytes,
        Tiles:     tiles,
        Status:    "success",
    }
    
    logJSON("stage_complete", map[string]any{
        "metrics": metrics,
    })
    
    pushMetrics(stage, duration)
}
```

### Cost & Time Optimization

```go
// Cache downloads with checksum gates
func shouldRedownload(pbfPath, sha256Path string) bool {
    if _, err := os.Stat(pbfPath); os.IsNotExist(err) {
        return true
    }
    
    if _, err := os.Stat(sha256Path); os.IsNotExist(err) {
        return true
    }
    
    // Check if checksum matches
    cmd := exec.Command("sha256sum", "-c", sha256Path)
    return cmd.Run() != nil
}

// Reuse MBTiles when input unchanged
func shouldRegenerateTiles(mbtilesPath, pbfPath string) bool {
    if _, err := os.Stat(mbtilesPath); os.IsNotExist(err) {
        return true
    }
    
    // Compare modification times
    mbtilesInfo, _ := os.Stat(mbtilesPath)
    pbfInfo, _ := os.Stat(pbfPath)
    
    return mbtilesInfo.ModTime().Before(pbfInfo.ModTime())
}
```

### Rollbacks

```go
// Keep last N artifacts with versioned prefixes
func publishWithVersion(prefix string) string {
    version := time.Now().Format("200601021504")
    return fmt.Sprintf("%s/v%s/", prefix, version)
}

// Cleanup old versions (keep last 5)
func cleanupOldVersions(bucket, prefix string) error {
    // Implementation depends on your storage backend
    // Use rclone or AWS CLI to list and delete old versions
    return nil
}
```

## Quick Start (TL;DR)

### 1. Bring up services

```bash
docker compose up -d
```

### 2. Build the glue

```bash
go mod init osmglue
go get github.com/spf13/cobra
go get github.com/spf13/viper
go get github.com/lib/pq
go build -o osmglue ./cmd/osmglue
```

### 3. Run stages

```bash
# Start services and wait for health checks
./osmglue boot

# Download OSM data with checksum verification
./osmglue fetch

# Load data into PostGIS via osm2pgsql
./osmglue load

# Create indexes and materialized views
./osmglue index

# Generate tiles with tippecanoe
./osmglue tile

# Start Martin tileserver (optional)
./osmglue serve

# Publish to S3/MinIO (optional)
./osmglue publish

# Check status
./osmglue status
```

### 4. Verify the pipeline

```bash
# Check PostGIS data
psql -d postgresql://postgres:postgres@localhost:5432/osm -c "SELECT COUNT(*) FROM planet_osm_roads;"

# Check Martin tileserver
curl http://localhost:3000/health

# Check generated tiles
ls -la tiles.mbtiles

# Check metrics
curl http://localhost:9091/metrics
```

---

*This tutorial demonstrates how Go can orchestrate complex, multi-stage data pipelines with reliability, observability, and production-grade error handling. The combination of PostGIS, tippecanoe, and object storage creates a powerful geospatial tile serving infrastructure.*
