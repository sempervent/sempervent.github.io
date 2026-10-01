# System resilience, concurrency, and backpressure

Distributed systems fail when work enters faster than you can finish it. Connection pools exhaust, queues grow without bound, and one slow dependency stalls everything upstream. This page is about **bounding concurrency** and **signaling overload** before you run out of memory or threads.

It is not a catalog of every resilience pattern. It focuses on what I reach for first: limits, backpressure, and honest degradation.

## When to care

- **Postgres-heavy services**: pool size is a hard ceiling. Slow queries hold connections; retries multiply load.
- **Async Python / Go / Rust services**: goroutines or tasks are cheap until they are not — memory and file descriptors still cap you.
- **GIS and batch jobs**: large geometries and cold caches create latency spikes that look like “random” timeouts unless you limit fan-out.
- **ML inference**: GPU memory and batch queues need explicit bounds; “just add workers” often makes p95 worse.

If the workload is a once-a-day cron, you probably do not need a circuit breaker. You need a timeout and an alert when the job overruns.

## Concurrency: pick a boundary and enforce it

**Threads / processes (sync code)**  
Use a fixed worker pool size derived from CPU and I/O, not `number_of_cpus * 100`. For DB-bound work, size against **pool connections**, not cores.

**Async (Python asyncio, etc.)**  
Semaphores limit in-flight requests. Without them, every accepted socket spawns unbounded task creation under load.

**Go**  
Worker pools with buffered channels; respect `context` cancellation on shutdown.

**Rust**  
`tokio::sync::Semaphore` or dedicated worker tasks; avoid unbounded `spawn` in handlers.

Illustrative pattern (Python asyncio):

```python
# ILLUSTRATIVE — tune limits from pool size and measured latency
MAX_IN_FLIGHT = 32
sem = asyncio.Semaphore(MAX_IN_FLIGHT)

async def handle(request):
    async with sem:
        return await do_work(request)
```

## Backpressure

Backpressure means **slowing producers** when consumers fall behind.

Mechanisms I actually use:

1. **Bounded queues** — `maxsize` on asyncio queues, channel buffers in Go, Redis stream consumer lag alerts.
2. **HTTP 503 + Retry-After** — when the service is saturated, reject early instead of timing out deep inside the stack.
3. **Admission control at the edge** — API gateway or load balancer max connections; rate limits per tenant.
4. **Database-side guardrails** — `statement_timeout`, connection pool limits, read replicas for read-heavy paths.

Anti-pattern: unbounded in-memory queues “because we will scale workers later.” That is how you OOM during a traffic spike.

## Rate limiting

Use rate limits to protect shared resources (auth endpoints, expensive GIS exports, model inference).

- **Token bucket** — smooth bursts with a sustained average (good for user-facing APIs).
- **Fixed window counters** — simple in Redis; watch boundary effects at window rollovers.
- **Per-tenant keys** — avoid one noisy neighbor taking the whole service.

Do not copy numeric limits from blog posts. Derive them from **measured** capacity and SLOs you actually monitor.

## Load shedding and degradation

When overloaded:

1. Drop or defer **optional** work first (recommendations, previews, non-critical tiles).
2. Serve **cached** or **coarser** results (lower zoom, simplified geometry, stale read).
3. Fail fast on **new** work while draining in-flight requests.

Circuit breakers help when a dependency is **known bad** (repeated errors, health check failing). They are not a substitute for pool sizing.

## Observability (minimum)

Before tuning limits, you need:

- In-flight request count / queue depth
- Pool utilization (DB, Redis, HTTP client)
- p95/p99 latency **per dependency**, not just the edge
- Error rate split by timeout vs 4xx vs 5xx

Prometheus/Grafana/Loki are fine; the point is to see **saturation**, not only CPU.

## GIS-specific notes

- Cap concurrent **tile** or **WFS** exports; large bounding boxes are abuse vectors.
- Prefer **pre-tiled** or **materialized** layers for hot paths; do not let ad hoc `ST_Union` on millions of rows run concurrently without a queue.
- Index discipline (see [PostGIS geometry indexing tutorial](../../tutorials/database-data-engineering/postgis-geometry-indexing.md)) matters more than micro-optimizing thread counts.

## ML inference notes

- Bound batch size and queue depth on GPU workers.
- Separate **online** inference from **batch** scoring where possible.
- Fallback to a smaller model only if you have tested quality impact — not as a default config copied from a diagram.

## What I distrust

- Pages that list twelve languages and fifteen patterns without sizing guidance.
- “Production-ready” sample code that never mentions pool limits or cancellation.
- SLO tables with precise percentages and no measurement method.
- Autoscaling rules copied from hyperscaler docs onto a three-node lab cluster.

## Further reading

- [Failure-oriented system design](../operations/failure-oriented-design.md)
- [Configuration management](configuration-management.md)
- PostgreSQL: [connection pooling tutorial](../../tutorials/database-data-engineering/postgres-pooling.md)
- Google SRE Book — overload and cascading failures (external)
