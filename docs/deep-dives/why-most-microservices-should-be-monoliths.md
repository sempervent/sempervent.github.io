---
tags:
  - deep-dive
  - architecture
  - systems-design
---

# Why Most Microservices Should Be Monoliths: Coordination Cost and the Myth of Infinite Scale

*This deep dive examines the conditions under which microservice adoption is premature or counterproductive. For the companion analysis of when microservices are architecturally appropriate, see [Appropriate Use of Microservices](appropriate-use-of-microservices.md).*

**Themes:** Architecture · Organizational · Economics

---

## Opening Thesis

Microservices solve scaling problems that most organizations do not have. The architecture was born in large engineering organizations — Netflix, Amazon, Google, Uber — that had reached the limits of monolithic scalability along specific dimensions: team size, deployment velocity, and the operational cost of deploying a 10-million-line codebase as a single unit. These organizations adopted microservices because the coordination costs of a monolith exceeded the coordination costs of distribution.

For organizations that have not reached these limits — which is most organizations building software — the adoption of microservices imports the coordination costs of distribution without the benefits that justify them. The result is an architecture that is harder to develop, harder to debug, harder to operate, and harder to reason about, in service of a scaling capability that the organization does not need.

The argument here is not that microservices are bad. It is that they are expensive, and that the expense is frequently incurred before the benefit is available.

---

## Historical Context

### Service-Oriented Architecture

Service-Oriented Architecture (SOA), as articulated in the early 2000s, proposed decomposing software systems into reusable, communicating services with well-defined interfaces. The enterprise software vendors of the era — IBM, Microsoft, Oracle — implemented SOA through heavyweight XML-based protocols: SOAP, WSDL, WS-* specifications of considerable complexity.

SOA's practical failures were multiple. The SOAP/WSDL stack was verbose, slow, and difficult to implement correctly. The "reusability" premise — that a service designed generically enough could be shared across business domains — produced services that were too abstract to be useful for any specific domain and too inflexible to evolve with any specific domain's requirements. Enterprise service buses (ESBs) accumulated the complexity that should have lived in the applications themselves.

The architectural instinct behind SOA — decompose, isolate, communicate through interfaces — was correct. The implementation was not. The REST revolution that followed replaced SOAP's verbosity with HTTP's simplicity, but it did not change the fundamental premise that services communicating over a network are a useful organizational pattern.

### The REST Era and Independent Deployment

REST APIs, combined with the growth of web-scale software companies in the 2000s, normalized the pattern of services communicating over HTTP. The key observation at this stage was not that decomposition produced scale — it was that independent deployment allowed teams to ship without coordinating their release schedules. A frontend team could deploy its changes without waiting for a backend team; a payments service could be updated without redeploying the entire application.

This benefit was real and remains real. The ability to deploy a service independently, without coupling to other teams' release cycles, is one of the genuine advantages of service decomposition. The question is whether this benefit requires full microservice architecture or whether a modular monolith — a single deployable unit with well-defined internal module boundaries — achieves the same benefit with less overhead.

### The Kubernetes Acceleration

The arrival of Kubernetes as a standard container orchestration platform created a feedback loop that accelerated microservice adoption beyond what organizational maturity could absorb. Kubernetes made it relatively straightforward to deploy many small services, each in its own container, with service discovery, health checks, and rolling updates. The operational machinery for running microservices became more accessible.

What Kubernetes did not provide was any guidance on whether services should be split. It made splitting easier; it did not make splitting correct. Organizations that had been constrained from microservice adoption by operational complexity found that constraint removed, and adopted the architecture without the organizational prerequisites — mature CI/CD, domain ownership, data contract discipline — that make it tractable.

### The Microservices Gold Rush

By the mid-2010s, "microservices" had acquired the character of an industry consensus. Conferences, blog posts, architectural frameworks, and vendor positioning all converged on the message that modern software is built with microservices. Teams that were not adopting microservices were building yesterday's architecture.

This consensus produced adoption without justification. Organizations split services not because the splitting solved a problem they had, but because the architecture was prestigious and because the tooling made it possible. The resulting systems inherited the full operational complexity of distributed systems — network failures, serialization, versioning, distributed tracing, secrets management at scale — without the scale that would justify it.

---

## The Hidden Costs of Microservices

### Network Boundaries

A function call in a monolith completes in nanoseconds. A service call over a network completes in milliseconds — three to six orders of magnitude slower, with variance introduced by network conditions, service load, and connection management. This is not merely a latency concern. It is a reliability concern: network calls can fail in ways that in-process function calls cannot.

```
  Monolith vs distributed system failure modes:
  ─────────────────────────────────────────────────────────────────
  Monolith (function call):
  caller → [in-process call] → callee
  Failure modes: exception (deterministic), OOM (rare)
  Latency: nanoseconds

  Microservices (network call):
  caller → [TCP connect] → [TLS handshake] → [HTTP request]
         → [JSON serialize] → network → [JSON deserialize]
         → callee → [JSON serialize] → network → [JSON deserialize]
         → caller
  Failure modes: timeout, connection refused, partial response,
                 network partition, DNS failure, TLS error,
                 response deserialization error, rate limiting,
                 circuit breaker open
  Latency: milliseconds (with high tail variance)
```

Every service call that crosses a network boundary must implement timeout handling, retry logic with backoff, circuit breaking to prevent cascade failures, and monitoring to detect degradation. This is not optional; without it, a single slow downstream service can exhaust the thread pool or connection pool of every upstream service that calls it. The implementation of robust network call handling is non-trivial and must be repeated at every service boundary.

### Serialization Overhead

In a monolith, data passes between functions as in-memory objects. In a microservices system, data must be serialized to a wire format (JSON, Protobuf, Avro, MessagePack), transmitted, and deserialized. For large or complex data structures, this serialization cost is measurable. For simple data, it is small but present.

More consequentially, serialization forces explicit schema decisions that an in-process API does not require. A function signature in a monolith can be changed and all callers updated in the same commit. A service API change requires versioning: either backward-compatible evolution (adding fields, not removing them) or explicit versioning of the API endpoint. For organizations that have not built the discipline of API versioning, service API evolution becomes a source of breaking changes that must be coordinated across teams.

### Versioning Hell

The service API versioning problem compounds over time. A system with 20 services, each with two or three versions of its API in use simultaneously (some consumers have not updated), generates a combinatorial state space of compatibility requirements. The team responsible for service A must maintain API v1 for the consumer that has not yet migrated, API v2 for the majority of consumers, and API v3 for the new feature that requires a changed contract.

This versioning complexity is manageable with tooling and discipline but requires both. API gateways, schema registries, client libraries, and automated compatibility testing all contribute to managing it. The cost of building and maintaining this infrastructure is real and does not exist in a monolith.

### Deployment Complexity

A monolith has one deployment artifact: the monolith binary or container. A 30-service system has 30 deployment artifacts, each with its own pipeline, its own versioning, its own deployment configuration, and its own health checks. The CI/CD infrastructure that manages 30 independent pipelines is substantially more complex than the infrastructure that manages one.

The deployment complexity has an organizational shadow: each service deployment is an independent event, which means the number of deployments per day is proportional to the number of teams times their deployment frequency. Coordinating deployments that span services — a feature that requires a new API endpoint in service A and a new consumer in service B — requires either backward-compatible API evolution (so B can be deployed after A) or coordinated deployment timing, which reintroduces the coupling that service decomposition was intended to eliminate.

### Distributed Tracing as Necessity

In a monolith, a request failure produces a stack trace. The failure is localized, the call chain is visible, and the debugging tools are standard. In a microservices system, a request failure may produce no trace at all, or a trace that shows a timeout at service boundary N without indicating what happened inside the service that timed out.

Distributed tracing is not optional for a microservices system that operates in production. It is infrastructure that must be built, maintained, and queried. The operational and economic cost of distributed tracing is the direct cost of the architectural choice to split services. Organizations that adopt microservices without building distributed tracing first are creating systems that are not debuggable in production.

### Schema Drift Across Services

In a monolith, a database schema change is a single migration applied to a single database that all code in the monolith reads from. In a microservices system, each service (in the ideal model) has its own database, and the concept that spans services — a "user," an "order," a "product" — may be defined differently in each service's schema.

The "user" in the authentication service has a different schema than the "user" in the preferences service, which has a different schema than the "user" in the analytics service. When these definitions diverge, integration between services requires mapping logic that is fragile, difficult to test, and frequently wrong in edge cases.

---

## Organizational Coupling

Conway's Law observes that organizations produce system designs that mirror their communication structures. A team that owns a monolith produces a monolith whose internal boundaries reflect the team's functional divisions. A set of teams that each own a service produces a service topology that mirrors the organizational structure.

The inverse is also relevant: adopting a microservice architecture in an organization that is not structured to own services does not produce the benefits of service ownership. If the team that is responsible for deploying the authentication service is the same team responsible for the payments service, the shipping service, and the user profile service, those services share all their operational decisions — deployment timing, incident response, operational discipline — and the decomposition has produced overhead without the team autonomy that justifies it.

**Team topology** is the prerequisite that microservice adoption advocates frequently elide. The DevOps Research and Assessment (DORA) research and Team Topologies framework both establish that high-performing software delivery requires small, autonomous, cross-functional teams that own their services end to end. If an organization is not organized this way before adopting microservices, the services will be owned by the wrong teams, and the coordination cost will be paid without the velocity benefit.

---

## When Microservices Are Justified

The conditions under which microservices solve real problems rather than create them are specific:

**Independent scaling requirements**: when different parts of the system have fundamentally different load profiles — a recommendation engine that handles read-heavy bursty traffic, a billing service that handles low-volume transactional traffic, an ingestion service that handles sustained high-throughput writes — collocating them in a monolith creates resource contention that cannot be resolved by vertical scaling. Service decomposition allows each component to be scaled to its own demand curve.

**Strict domain isolation**: when regulatory requirements, security considerations, or organizational policy require that specific functionality be isolated — payment card processing under PCI-DSS, healthcare data under HIPAA, externally available APIs from internal systems — service boundaries enforce isolation that a modular monolith cannot guarantee.

**Polyglot requirements**: when different components genuinely benefit from different technology stacks — a machine learning model that must run in Python, a high-throughput event processor that benefits from Go's concurrency model, a computation-intensive spatial analysis component that requires Rust — service decomposition allows each component to use its optimal technology without inflicting it on the entire system.

**Team autonomy at scale**: when the engineering organization has grown to the point where a single codebase cannot be effectively maintained by one team, and domain ownership has been explicitly defined and staffed, service decomposition enables teams to work autonomously without stepping on each other's changes.

---

## Decision Framework

| Constraint | Monolith | Microservices |
|---|---|---|
| Team size < 20 engineers | Preferred | Premature |
| Independent scaling required | Limited | Preferred |
| Single deployment artifact acceptable | Preferred | Not applicable |
| Domain ownership well-defined | Either | Required |
| CI/CD maturity high | Either | Required |
| Regulatory isolation required | Limited | Preferred |
| Distributed tracing available | Not needed | Required |
| Schema contracts discipline exists | Not needed | Required |
| Polyglot technology stacks needed | Limited | Enables |
| Startup or early-stage product | Strongly preferred | Costly mistake |
| Team organized by function (not domain) | Preferred | Creates overhead |

The decision heuristic that survives the most scrutiny: **if the reason for adopting microservices is not "we have a specific problem that monolithic architecture cannot solve," the reason is not good enough.** Microservices are an organizational and operational investment that must be justified by the problem it solves, not by the architectural aesthetics it expresses.

!!! tip "See also"
    - [Appropriate Use of Microservices](appropriate-use-of-microservices.md) — the companion analysis of when service decomposition is architecturally correct
    - [IaC vs GitOps](iac-vs-gitops.md) — the infrastructure automation layer required before microservice deployment is tractable
    - [Observability vs Monitoring](observability-vs-monitoring.md) — the observability infrastructure that microservices make mandatory
    - [Distributed Systems and the Myth of Infinite Scale](distributed-systems-myth-of-infinite-scale.md) — the coordination and consistency costs that distribution imposes, which microservices also incur
    - [Why Most Kubernetes Clusters Shouldn't Exist](why-most-kubernetes-clusters-shouldnt-exist.md) — the orchestration overhead that microservice deployments typically incur and often do not justify

!!! abstract "Decision Frameworks"
    The structured decision framework from this essay — and its companion on appropriate use — is indexed at [Decision Frameworks → When to Decompose into Microservices](../decision-frameworks.md#when-to-decompose-into-microservices). For the recurring mistake this essay addresses, see [Anti-Patterns → Premature Microservices](../anti-patterns.md#premature-microservices).
