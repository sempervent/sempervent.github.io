---
tags:
  - doctrine
  - anti-patterns
  - architecture
---

# Architectural Anti-Patterns Index

These are recurring architectural mistakes — patterns adopted with legitimate intent that produce predictable, avoidable failures. Each entry describes the pattern, explains why it fails, and links to the essay that analyzes it in depth.

---

## Premature Microservices

**The pattern**: decomposing a software system into independently deployed services before the domain boundaries are well understood, before the team has the organizational structure to own separate services, or before the operational infrastructure exists to support multi-service deployment.

**Why it fails**: service decomposition is a cost with benefits that only materialize under specific conditions: domain boundaries that are genuinely stable, teams that are large enough to own independent services, and infrastructure capable of handling service mesh, distributed tracing, and independent CI/CD pipelines. Applied prematurely, decomposition produces the overhead costs (network calls, distributed state, versioning complexity) without the isolation and scalability benefits. The resulting system is harder to understand, harder to debug, and harder to change than the monolith it replaced.

**The signal**: a system where every feature change requires coordinating deployments across multiple services, where engineers cannot run the full system locally, or where "microservice" is used to describe services that share a database.

*See: [Why Most Microservices Should Be Monoliths](deep-dives/why-most-microservices-should-be-monoliths.md), [Appropriate Use of Microservices](deep-dives/appropriate-use-of-microservices.md)*

---

## Overusing Kubernetes

**The pattern**: deploying a Kubernetes cluster for workloads that do not require distributed scheduling, dynamic scaling, or multi-service orchestration at the scale that justifies Kubernetes's operational overhead.

**Why it fails**: Kubernetes solves distributed scheduling at large scale. For teams running fewer than 20 services on a handful of nodes, Kubernetes introduces etcd management, cluster upgrade cycles, CNI plugin selection, RBAC configuration, and observability expansion that consume engineering time without providing proportional benefit. The cluster that was adopted to simplify operations becomes an operational commitment that dominates platform engineering capacity.

**The signal**: a team where the majority of Kubernetes-related work is cluster maintenance rather than application delivery; where developers cannot run the production environment locally because it requires a Kubernetes cluster; or where the platform team spends more time on Helm chart management than on feature delivery.

*See: [Why Most Kubernetes Clusters Shouldn't Exist](deep-dives/why-most-kubernetes-clusters-shouldnt-exist.md)*

---

## Real-Time by Default

**The pattern**: choosing streaming or real-time data pipelines because "real-time is better than batch" rather than because the latency requirement of the use case justifies the infrastructure and operational cost of streaming.

**Why it fails**: streaming pipelines require stateful stream processors, backpressure handling, checkpointing, late-arriving data management, and end-to-end observability of a pipeline that runs continuously rather than at a known schedule. These costs are appropriate for use cases where latency genuinely matters: fraud detection, live control loops, safety-critical systems. They are not appropriate for analytical workloads where users are satisfied with hourly batch updates, or for event pipelines where micro-batching (15–30 minute windows) provides adequate latency at a fraction of the operational complexity.

**The signal**: a streaming pipeline that processes data that changes hourly and is consumed by a dashboard refreshed daily; or a team that chose Kafka because "everyone uses Kafka" before measuring whether the latency requirement justifies it.

*See: [The Hidden Cost of Real-Time Systems](deep-dives/the-hidden-cost-of-real-time-systems.md)*

---

## Data Swamp Formation

**The pattern**: a data lake that accumulates data without governance: no ownership model, no schema enforcement, no lineage tracking, no metadata standards, and no partition discipline. The lake grows; its contents become opaque and untrustworthy.

**Why it fails**: object storage is cheap, which makes it easy to store everything and govern nothing. Without ownership, datasets become orphaned when their creators leave. Without schema enforcement, column semantics drift. Without lineage, downstream consumers cannot trace data quality problems to their source. Without partition discipline, query performance degrades as the number of small files grows. The result is a storage estate that is expensive to maintain, slow to query, and impossible to audit.

**The signal**: analysts who must interview multiple people to determine whether a table is current and trustworthy; duplicate pipelines built because existing ones could not be found or trusted; regulatory inquiries that cannot be answered because data lineage was not tracked.

*See: [Why Most Data Lakes Become Data Swamps](deep-dives/why-data-lakes-become-swamps.md), [Metadata as Infrastructure](deep-dives/metadata-as-infrastructure.md), [The Hidden Cost of Metadata Debt](deep-dives/the-hidden-cost-of-metadata-debt.md)*

---

## Serverless Cargo Cult

**The pattern**: adopting serverless architecture because the organizational consensus is "serverless is modern and scalable" without analyzing whether the workload's traffic pattern, latency requirements, stateful complexity, or compliance requirements are compatible with the serverless execution model.

**Why it fails**: serverless is an appropriate model for intermittent, event-driven, latency-tolerant workloads. For sustained high-throughput APIs, stateful workflows, low-latency services, and compliance-sensitive environments, serverless introduces cold start unpredictability, IAM management overhead, observability fragmentation, and vendor lock-in without providing the per-invocation economics that justify these costs. Organizations that adopt serverless for general application development frequently re-architect back to containerized services after encountering the production failure modes that the marketing narrative did not mention.

**The signal**: a serverless application that uses provisioned concurrency (warm instances) to avoid cold starts, effectively paying for reserved capacity — the economic model of always-on services — while retaining the operational complexity of serverless.

*See: [The Myth of Serverless Simplicity](deep-dives/the-myth-of-serverless-simplicity.md)*

---

## Distributed Systems for Small Teams

**The pattern**: adopting distributed system patterns — eventual consistency, CQRS, event sourcing, distributed tracing, saga orchestration — for systems whose scale and team size do not justify the operational and cognitive overhead.

**Why it fails**: distributed systems solve problems that arise at scale: geographic redundancy, massive concurrency, data locality constraints, independent team scaling. At the scale of most organizations (tens of services, hundreds of concurrent users, single-region deployment), these patterns add coordination overhead, debugging complexity, and operational surface without providing the scale benefits they are designed to deliver. The team that implements CQRS for a CRUD application has made the application harder to understand and harder to debug in exchange for no practical benefit.

**The signal**: a system where "it's eventually consistent" is used to explain why the UI shows stale data to users who have not refreshed the page; or a distributed tracing setup that requires 6 hops of instrumentation to diagnose a bug that a stack trace would have identified immediately in a monolith.

*See: [Distributed Systems and the Myth of Infinite Scale](deep-dives/distributed-systems-myth-of-infinite-scale.md), [Why Most Microservices Should Be Monoliths](deep-dives/why-most-microservices-should-be-monoliths.md)*
