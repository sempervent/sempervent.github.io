# Go Memory Management Best Practices

**Objective**: Master senior-level Go memory management patterns for production systems. When you need to optimize memory usage, when you want to understand Go's garbage collector, when you need enterprise-grade memory management patterns—these best practices become your weapon of choice.

## Core Principles

- **Understand GC Behavior**: Know how Go's garbage collector works
- **Minimize Allocations**: Reduce unnecessary memory allocations
- **Use Object Pooling**: Reuse objects to reduce GC pressure
- **Monitor Memory Usage**: Track memory consumption and GC activity
- **Optimize Data Structures**: Choose appropriate data structures for memory efficiency

## Go Garbage Collector

### GC Fundamentals

```go
// internal/memory/gc_fundamentals.go
package memory

import (
    "runtime"
    "runtime/debug"
    "time"
)

// GCAnalyzer provides garbage collection analysis
type GCAnalyzer struct {
    logger Logger
}

// NewGCAnalyzer creates a new GC analyzer
func NewGCAnalyzer() *GCAnalyzer {
    return &GCAnalyzer{
        logger: NewLogger(),
    }
}

// AnalyzeGC analyzes garbage collection behavior
func (gca *GCAnalyzer) AnalyzeGC() GCStats {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    return GCStats{
        Alloc:         m.Alloc,
        TotalAlloc:    m.TotalAlloc,
        Sys:           m.Sys,
        Lookups:       m.Lookups,
        Mallocs:       m.Mallocs,
        Frees:         m.Frees,
        HeapAlloc:     m.HeapAlloc,
        HeapSys:       m.HeapSys,
        HeapIdle:      m.HeapIdle,
        HeapInuse:     m.HeapInuse,
        HeapReleased:  m.HeapReleased,
        HeapObjects:   m.HeapObjects,
        StackInuse:    m.StackInuse,
        StackSys:      m.StackSys,
        MSpanInuse:    m.MSpanInuse,
        MSpanSys:      m.MSpanSys,
        MCacheInuse:   m.MCacheInuse,
        MCacheSys:     m.MCacheSys,
        BuckHashSys:   m.BuckHashSys,
        GCSys:         m.GCSys,
        OtherSys:      m.OtherSys,
        NextGC:        m.NextGC,
        LastGC:        m.LastGC,
        PauseTotalNs:  m.PauseTotalNs,
        NumGC:         m.NumGC,
        NumForcedGC:   m.NumForcedGC,
        GCCPUFraction: m.GCCPUFraction,
    }
}

// GCStats represents garbage collection statistics
type GCStats struct {
    Alloc         uint64
    TotalAlloc    uint64
    Sys           uint64
    Lookups       uint64
    Mallocs       uint64
    Frees         uint64
    HeapAlloc     uint64
    HeapSys       uint64
    HeapIdle      uint64
    HeapInuse     uint64
    HeapReleased  uint64
    HeapObjects   uint64
    StackInuse    uint64
    StackSys      uint64
    MSpanInuse    uint64
    MSpanSys      uint64
    MCacheInuse   uint64
    MCacheSys     uint64
    BuckHashSys   uint64
    GCSys         uint64
    OtherSys      uint64
    NextGC        uint64
    LastGC        uint64
    PauseTotalNs  uint64
    NumGC         uint32
    NumForcedGC   uint32
    GCCPUFraction float64
}

// ForceGC forces garbage collection
func (gca *GCAnalyzer) ForceGC() {
    runtime.GC()
}

// SetGCPercent sets the garbage collection target percentage
func (gca *GCAnalyzer) SetGCPercent(percent int) int {
    return debug.SetGCPercent(percent)
}

// SetMemoryLimit sets the memory limit
func (gca *GCAnalyzer) SetMemoryLimit(limit int64) int64 {
    return debug.SetMemoryLimit(limit)
}
```

### GC Tuning

```go
// internal/memory/gc_tuning.go
package memory

import (
    "runtime"
    "runtime/debug"
    "time"
)

// GCTuner provides garbage collection tuning
type GCTuner struct {
    targetPercent int
    memoryLimit   int64
    logger        Logger
}

// NewGCTuner creates a new GC tuner
func NewGCTuner() *GCTuner {
    return &GCTuner{
        targetPercent: 100, // Default GC target
        memoryLimit:   0,    // No limit by default
        logger:        NewLogger(),
    }
}

// TuneGC tunes garbage collection parameters
func (gct *GCTuner) TuneGC(targetPercent int, memoryLimit int64) {
    gct.targetPercent = targetPercent
    gct.memoryLimit = memoryLimit
    
    debug.SetGCPercent(targetPercent)
    if memoryLimit > 0 {
        debug.SetMemoryLimit(memoryLimit)
    }
    
    gct.logger.Info("GC tuning applied", map[string]interface{}{
        "target_percent": targetPercent,
        "memory_limit":   memoryLimit,
    })
}

// MonitorGC monitors garbage collection activity
func (gct *GCTuner) MonitorGC(interval time.Duration) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        gct.logger.Info("GC stats", map[string]interface{}{
            "heap_alloc":     m.HeapAlloc,
            "heap_sys":        m.HeapSys,
            "heap_objects":    m.HeapObjects,
            "num_gc":          m.NumGC,
            "gc_cpu_fraction": m.GCCPUFraction,
        })
    }
}

// OptimizeGC optimizes garbage collection for the workload
func (gct *GCTuner) OptimizeGC(workloadType string) {
    switch workloadType {
    case "high-throughput":
        gct.TuneGC(50, 0) // More aggressive GC
    case "low-latency":
        gct.TuneGC(200, 0) // Less frequent GC
    case "memory-constrained":
        gct.TuneGC(100, 512*1024*1024) // 512MB limit
    default:
        gct.TuneGC(100, 0) // Default settings
    }
}
```

## Memory Allocation Patterns

### Object Pooling

```go
// internal/memory/object_pool.go
package memory

import (
    "sync"
    "time"
)

// ObjectPool represents an object pool
type ObjectPool struct {
    pool     sync.Pool
    newFunc  func() interface{}
    resetFunc func(interface{})
    logger   Logger
}

// NewObjectPool creates a new object pool
func NewObjectPool(newFunc func() interface{}, resetFunc func(interface{})) *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: newFunc,
        },
        newFunc:   newFunc,
        resetFunc: resetFunc,
        logger:    NewLogger(),
    }
}

// Get gets an object from the pool
func (op *ObjectPool) Get() interface{} {
    obj := op.pool.Get()
    if op.resetFunc != nil {
        op.resetFunc(obj)
    }
    return obj
}

// Put puts an object back into the pool
func (op *ObjectPool) Put(obj interface{}) {
    op.pool.Put(obj)
}

// BufferPool represents a buffer pool
type BufferPool struct {
    pool   sync.Pool
    size   int
    logger Logger
}

// NewBufferPool creates a new buffer pool
func NewBufferPool(size int) *BufferPool {
    return &BufferPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]byte, 0, size)
            },
        },
        size:   size,
        logger: NewLogger(),
    }
}

// Get gets a buffer from the pool
func (bp *BufferPool) Get() []byte {
    return bp.pool.Get().([]byte)
}

// Put puts a buffer back into the pool
func (bp *BufferPool) Put(buf []byte) {
    if cap(buf) >= bp.size {
        buf = buf[:0] // Reset length
        bp.pool.Put(buf)
    }
}

// StringPool represents a string pool
type StringPool struct {
    pool   sync.Pool
    logger Logger
}

// NewStringPool creates a new string pool
func NewStringPool() *StringPool {
    return &StringPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &strings.Builder{}
            },
        },
        logger: NewLogger(),
    }
}

// Get gets a string builder from the pool
func (sp *StringPool) Get() *strings.Builder {
    return sp.pool.Get().(*strings.Builder)
}

// Put puts a string builder back into the pool
func (sp *StringPool) Put(sb *strings.Builder) {
    sb.Reset()
    sp.pool.Put(sb)
}
```

### Memory-Efficient Data Structures

```go
// internal/memory/efficient_structures.go
package memory

import (
    "unsafe"
)

// CompactSlice represents a memory-efficient slice
type CompactSlice struct {
    data []byte
    size int
}

// NewCompactSlice creates a new compact slice
func NewCompactSlice(capacity int) *CompactSlice {
    return &CompactSlice{
        data: make([]byte, 0, capacity),
        size: 0,
    }
}

// Append appends data to the slice
func (cs *CompactSlice) Append(data []byte) {
    if cs.size+len(data) > cap(cs.data) {
        newData := make([]byte, cs.size+len(data), (cs.size+len(data))*2)
        copy(newData, cs.data[:cs.size])
        cs.data = newData
    }
    
    copy(cs.data[cs.size:], data)
    cs.size += len(data)
}

// Bytes returns the slice as bytes
func (cs *CompactSlice) Bytes() []byte {
    return cs.data[:cs.size]
}

// Reset resets the slice
func (cs *CompactSlice) Reset() {
    cs.size = 0
}

// MemoryEfficientMap represents a memory-efficient map
type MemoryEfficientMap struct {
    buckets []*bucket
    size    int
    mask    uint32
}

// bucket represents a hash bucket
type bucket struct {
    key   string
    value interface{}
    next  *bucket
}

// NewMemoryEfficientMap creates a new memory-efficient map
func NewMemoryEfficientMap(initialSize int) *MemoryEfficientMap {
    size := 1
    for size < initialSize {
        size <<= 1
    }
    
    return &MemoryEfficientMap{
        buckets: make([]*bucket, size),
        size:     0,
        mask:     uint32(size - 1),
    }
}

// Set sets a key-value pair
func (mem *MemoryEfficientMap) Set(key string, value interface{}) {
    hash := mem.hash(key)
    bucket := &bucket{
        key:   key,
        value: value,
        next:  mem.buckets[hash],
    }
    mem.buckets[hash] = bucket
    mem.size++
}

// Get gets a value by key
func (mem *MemoryEfficientMap) Get(key string) (interface{}, bool) {
    hash := mem.hash(key)
    bucket := mem.buckets[hash]
    
    for bucket != nil {
        if bucket.key == key {
            return bucket.value, true
        }
        bucket = bucket.next
    }
    
    return nil, false
}

// hash calculates the hash for a key
func (mem *MemoryEfficientMap) hash(key string) uint32 {
    hash := uint32(0)
    for _, c := range key {
        hash = hash*31 + uint32(c)
    }
    return hash & mem.mask
}
```

## Memory Monitoring

### Memory Monitor

```go
// internal/memory/memory_monitor.go
package memory

import (
    "runtime"
    "time"
)

// MemoryMonitor monitors memory usage
type MemoryMonitor struct {
    logger     Logger
    interval   time.Duration
    threshold  uint64
    alertFunc  func(uint64)
}

// NewMemoryMonitor creates a new memory monitor
func NewMemoryMonitor(interval time.Duration, threshold uint64) *MemoryMonitor {
    return &MemoryMonitor{
        logger:    NewLogger(),
        interval:  interval,
        threshold: threshold,
    }
}

// SetAlertFunction sets the alert function
func (mm *MemoryMonitor) SetAlertFunction(alertFunc func(uint64)) {
    mm.alertFunc = alertFunc
}

// Start starts memory monitoring
func (mm *MemoryMonitor) Start() {
    ticker := time.NewTicker(mm.interval)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        if m.HeapAlloc > mm.threshold {
            mm.logger.Warn("Memory threshold exceeded", map[string]interface{}{
                "heap_alloc": m.HeapAlloc,
                "threshold":   mm.threshold,
            })
            
            if mm.alertFunc != nil {
                mm.alertFunc(m.HeapAlloc)
            }
        }
        
        mm.logger.Info("Memory stats", map[string]interface{}{
            "heap_alloc":     m.HeapAlloc,
            "heap_sys":       m.Sys,
            "heap_objects":   m.HeapObjects,
            "gc_cpu_fraction": m.GCCPUFraction,
        })
    }
}

// GetMemoryUsage returns current memory usage
func (mm *MemoryMonitor) GetMemoryUsage() MemoryUsage {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    return MemoryUsage{
        HeapAlloc:     m.HeapAlloc,
        HeapSys:       m.HeapSys,
        HeapObjects:   m.HeapObjects,
        StackInuse:    m.StackInuse,
        StackSys:      m.StackSys,
        MSpanInuse:    m.MSpanInuse,
        MSpanSys:      m.MSpanSys,
        MCacheInuse:   m.MCacheInuse,
        MCacheSys:     m.MCacheSys,
        BuckHashSys:   m.BuckHashSys,
        GCSys:         m.GCSys,
        OtherSys:      m.OtherSys,
        NextGC:        m.NextGC,
        LastGC:        m.LastGC,
        NumGC:         m.NumGC,
        GCCPUFraction: m.GCCPUFraction,
    }
}

// MemoryUsage represents memory usage statistics
type MemoryUsage struct {
    HeapAlloc     uint64
    HeapSys       uint64
    HeapObjects   uint64
    StackInuse    uint64
    StackSys      uint64
    MSpanInuse    uint64
    MSpanSys      uint64
    MCacheInuse   uint64
    MCacheSys     uint64
    BuckHashSys   uint64
    GCSys         uint64
    OtherSys      uint64
    NextGC        uint64
    LastGC        uint64
    NumGC         uint32
    GCCPUFraction float64
}
```

### Memory Leak Detection

```go
// internal/memory/leak_detector.go
package memory

import (
    "runtime"
    "time"
)

// MemoryLeakDetector detects memory leaks
type MemoryLeakDetector struct {
    logger      Logger
    interval    time.Duration
    threshold   float64
    baseline    uint64
    samples     []uint64
    maxSamples  int
}

// NewMemoryLeakDetector creates a new memory leak detector
func NewMemoryLeakDetector(interval time.Duration, threshold float64, maxSamples int) *MemoryLeakDetector {
    return &MemoryLeakDetector{
        logger:     NewLogger(),
        interval:   interval,
        threshold:  threshold,
        samples:    make([]uint64, 0, maxSamples),
        maxSamples: maxSamples,
    }
}

// Start starts memory leak detection
func (mld *MemoryLeakDetector) Start() {
    ticker := time.NewTicker(mld.interval)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        if mld.baseline == 0 {
            mld.baseline = m.HeapAlloc
        }
        
        mld.samples = append(mld.samples, m.HeapAlloc)
        if len(mld.samples) > mld.maxSamples {
            mld.samples = mld.samples[1:]
        }
        
        if len(mld.samples) >= 3 {
            if mld.detectLeak() {
                mld.logger.Warn("Potential memory leak detected", map[string]interface{}{
                    "current_alloc": m.HeapAlloc,
                    "baseline":      mld.baseline,
                    "growth_rate":   mld.calculateGrowthRate(),
                })
            }
        }
    }
}

// detectLeak detects if there's a memory leak
func (mld *MemoryLeakDetector) detectLeak() bool {
    if len(mld.samples) < 3 {
        return false
    }
    
    growthRate := mld.calculateGrowthRate()
    return growthRate > mld.threshold
}

// calculateGrowthRate calculates the memory growth rate
func (mld *MemoryLeakDetector) calculateGrowthRate() float64 {
    if len(mld.samples) < 2 {
        return 0
    }
    
    first := mld.samples[0]
    last := mld.samples[len(mld.samples)-1]
    
    if first == 0 {
        return 0
    }
    
    return float64(last-first) / float64(first)
}
```

## Memory Optimization Techniques

### Zero-Copy Operations

```go
// internal/memory/zero_copy.go
package memory

import (
    "unsafe"
)

// ZeroCopyOptimizer provides zero-copy optimization
type ZeroCopyOptimizer struct{}

// NewZeroCopyOptimizer creates a new zero-copy optimizer
func NewZeroCopyOptimizer() *ZeroCopyOptimizer {
    return &ZeroCopyOptimizer{}
}

// StringToBytes converts string to bytes without copying
func (zco *ZeroCopyOptimizer) StringToBytes(s string) []byte {
    return *(*[]byte)(unsafe.Pointer(&s))
}

// BytesToString converts bytes to string without copying
func (zco *ZeroCopyOptimizer) BytesToString(b []byte) string {
    return *(*string)(unsafe.Pointer(&b))
}

// SliceHeader returns the slice header
func (zco *ZeroCopyOptimizer) SliceHeader(slice interface{}) unsafe.Pointer {
    return unsafe.Pointer(&slice)
}

// FastCopy performs fast copying between slices
func (zco *ZeroCopyOptimizer) FastCopy(dst, src []byte) int {
    return copy(dst, src)
}
```

### Memory-Efficient Serialization

```go
// internal/memory/serialization.go
package memory

import (
    "encoding/binary"
    "unsafe"
)

// MemoryEfficientSerializer provides memory-efficient serialization
type MemoryEfficientSerializer struct{}

// NewMemoryEfficientSerializer creates a new memory-efficient serializer
func NewMemoryEfficientSerializer() *MemoryEfficientSerializer {
    return &MemoryEfficientSerializer{}
}

// SerializeInt32 serializes an int32 to bytes
func (mes *MemoryEfficientSerializer) SerializeInt32(value int32) []byte {
    buf := make([]byte, 4)
    binary.LittleEndian.PutUint32(buf, uint32(value))
    return buf
}

// DeserializeInt32 deserializes bytes to int32
func (mes *MemoryEfficientSerializer) DeserializeInt32(data []byte) int32 {
    return int32(binary.LittleEndian.Uint32(data))
}

// SerializeString serializes a string to bytes
func (mes *MemoryEfficientSerializer) SerializeString(value string) []byte {
    return []byte(value)
}

// DeserializeString deserializes bytes to string
func (mes *MemoryEfficientSerializer) DeserializeString(data []byte) string {
    return string(data)
}

// SerializeStruct serializes a struct to bytes
func (mes *MemoryEfficientSerializer) SerializeStruct(value interface{}) []byte {
    // Implement struct serialization
    return nil
}

// DeserializeStruct deserializes bytes to struct
func (mes *MemoryEfficientSerializer) DeserializeStruct(data []byte, value interface{}) error {
    // Implement struct deserialization
    return nil
}
```

## Memory Testing

### Memory Tests

```go
// internal/memory/memory_test.go
package memory

import (
    "testing"
    "time"
)

func TestObjectPool(t *testing.T) {
    pool := NewObjectPool(
        func() interface{} {
            return make([]byte, 1024)
        },
        func(obj interface{}) {
            buf := obj.([]byte)
            for i := range buf {
                buf[i] = 0
            }
        },
    )
    
    // Test pool operations
    obj1 := pool.Get()
    pool.Put(obj1)
    
    obj2 := pool.Get()
    if obj1 != obj2 {
        t.Error("Expected to get the same object from pool")
    }
}

func TestMemoryMonitor(t *testing.T) {
    monitor := NewMemoryMonitor(100*time.Millisecond, 1024*1024) // 1MB threshold
    
    // Test memory monitoring
    go monitor.Start()
    
    // Allocate some memory
    data := make([]byte, 1024*1024)
    _ = data
    
    time.Sleep(200 * time.Millisecond)
}

func TestMemoryLeakDetector(t *testing.T) {
    detector := NewMemoryLeakDetector(100*time.Millisecond, 0.1, 10)
    
    // Test leak detection
    go detector.Start()
    
    // Simulate memory leak
    var data [][]byte
    for i := 0; i < 1000; i++ {
        data = append(data, make([]byte, 1024))
    }
    
    time.Sleep(500 * time.Millisecond)
}

func BenchmarkObjectPool(b *testing.B) {
    pool := NewObjectPool(
        func() interface{} {
            return make([]byte, 1024)
        },
        nil,
    )
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        obj := pool.Get()
        pool.Put(obj)
    }
}

func BenchmarkMemoryAllocation(b *testing.B) {
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = make([]byte, 1024)
    }
}
```

## TL;DR Runbook

### Quick Start

```go
// 1. GC tuning
tuner := NewGCTuner()
tuner.TuneGC(100, 0) // 100% GC target, no memory limit

// 2. Object pooling
pool := NewObjectPool(
    func() interface{} { return make([]byte, 1024) },
    func(obj interface{}) { /* reset logic */ },
)

// 3. Memory monitoring
monitor := NewMemoryMonitor(30*time.Second, 100*1024*1024) // 100MB threshold
go monitor.Start()

// 4. Memory leak detection
detector := NewMemoryLeakDetector(60*time.Second, 0.1, 10)
go detector.Start()
```

### Essential Patterns

```go
// Object pooling
obj := pool.Get()
defer pool.Put(obj)

// Memory monitoring
usage := monitor.GetMemoryUsage()
if usage.HeapAlloc > threshold {
    // Handle high memory usage
}

// GC optimization
runtime.GC()
debug.SetGCPercent(100)
debug.SetMemoryLimit(512 * 1024 * 1024) // 512MB limit
```

---

*This guide provides the complete machinery for optimizing memory usage in Go applications. Each pattern includes implementation examples, monitoring strategies, and real-world usage patterns for enterprise deployment.*
