---
tags:
  - deep-dive
  - architecture
  - systems-design
---

# Appropriate Use of Microservices: Domain Boundaries, Autonomy, and Long-Term Maintainability

*This deep dive examines when microservice decomposition is architecturally correct. For the companion analysis of when it is premature or counterproductive, see [Why Most Microservices Should Be Monoliths](why-most-microservices-should-be-monoliths.md).*

**Themes:** Architecture · Governance · Organizational

---

## Opening Thesis

Microservices are architectural tools, not architectural ideology. The question is not "should we adopt microservices?" but "does decomposing this boundary into a service solve a problem we have, and does the cost of that decomposition not exceed the benefit?" Answered honestly, this question disqualifies most proposed service splits and validates a smaller set that are genuinely warranted.

The analysis of appropriate microservice use requires a prior analysis of what boundaries mean. A service boundary is not merely a deployment boundary or a language boundary. It is a boundary of responsibility, of data ownership, and of operational accountability. Services that are defined as technical partitions without corresponding organizational and data ownership partitions do not produce the benefits of service decomposition; they produce its costs.

---

## Domain-Driven Design Perspective

Domain-Driven Design (DDD), articulated by Eric Evans in 2003, provides the most rigorous framework for identifying service boundaries. The central concept is the **bounded context**: a portion of the domain model in which a specific ubiquitous language applies consistently. Within a bounded context, the definition of terms is unambiguous. An "order" in the Order Management bounded context has a precise definition that may differ from the definition of "order" in the Inventory bounded context.

Bounded contexts are not services. They are conceptual boundaries that make the domain model coherent. A bounded context may be implemented as a service, as a module within a monolith, or as a set of tables in a shared database. The architectural form is a consequent decision, not a prerequisite of the bounded context itself.

The value of bounded contexts as a precursor to service design is that they force explicit negotiation of terminology and responsibility before any architectural decision is made. Teams that agree on bounded contexts before building services avoid the pattern of services that share confused definitions — where the payments service's concept of "account" and the user service's concept of "account" gradually diverge because no one negotiated their definitions when the services were created.

**Aggregates** are clusters of domain objects that change together as a unit. The Aggregate root is the entry point through which external interactions with the cluster are made. Aggregates define natural transaction boundaries: changes to an aggregate are atomic, and changes that span aggregates require explicit coordination (events, sagas, or compensating transactions). Aggregates are therefore a natural candidate for service boundaries: each service manages its own aggregates and communicates changes through domain events.

**Ubiquitous language** — the shared vocabulary between domain experts and developers within a bounded context — is both a communication tool and a diagnostic. When the language used by developers and business stakeholders diverges, the bounded context has drifted. When different services use the same term to mean different things, the contexts have been drawn incorrectly.

---

## Architectural Pre-Conditions

Service decomposition is appropriate only when specific pre-conditions are met. These are not aspirational qualities that the organization plans to develop; they are requirements that must exist before the decomposition creates more value than overhead.

**Clear domain isolation**: the service boundary must correspond to a genuine business domain boundary, not a technical partition. A "data access layer" service that wraps a database is not a domain boundary; it is a technical layer that adds network overhead without adding domain clarity. A "payments" service that encapsulates all payment processing logic — charging, refunds, disputes, settlement — corresponds to a genuine business domain with its own regulatory context and team responsibility.

**Explicit data ownership**: each service must own its data. Data that is shared across service boundaries — a database table read and written by multiple services — is not data under service ownership. It is a shared resource that creates coupling between services at the data layer, which is the most difficult coupling to break after the fact. The discipline of "each service owns its own database" is not merely architectural preference; it is the minimum requirement for services to be independently deployable.

```
  Data ownership in service boundaries:
  ───────────────────────────────────────────────────────────────
  Correct: each service owns its data

  [Order Service]          [Inventory Service]
  orders_db                inventory_db
       │                        │
  API: /orders             API: /inventory
  (no direct DB access     (no direct DB access
   to inventory_db)         to orders_db)

  Communication via API or domain events only.

  Incorrect: shared database

  [Order Service]    [Inventory Service]    [Shipping Service]
         └─────────────────┼────────────────────┘
                    [Shared Database]
         (all services read/write each other's tables)
         → coupling through data, not through API
```

**Teams aligned to services**: a service without a team that owns it is a shared responsibility, which in practice means no responsibility. Each service must have a designated owning team responsible for its API contract, its operational health, its schema evolution, and its SLO. When ownership is ambiguous, incidents go unresolved and contracts drift without correction.

**Deployment automation maturity**: independent service deployment requires automated CI/CD pipelines per service, automated testing that validates the service in isolation, and automated health checks that validate deployment success. Manual deployment processes do not scale to multi-service systems; the coordination overhead of manual deployment across 20 services exceeds the benefits of independent deployment.

---

## Data Boundaries

The data boundary question is more consequential than the API boundary question. Services that have independent APIs but share a database are not truly independent: a schema migration in the shared database affects all services simultaneously, defeating the purpose of service isolation.

The correct model is complete data isolation: each service persists its own data in its own storage, with no direct database access from other services. Cross-service data needs are served through APIs or through domain events.

**Event-driven integration** — where services publish domain events to a shared event bus (Kafka, NATS, Kinesis) and other services consume those events — enables loose coupling between services without shared database dependencies. The Order service publishes an `OrderPlaced` event; the Inventory service consumes it and decrements available stock; the Shipping service consumes it and creates a shipment record. No service reads another's database; each processes the events relevant to its domain.

The event-driven model introduces its own complexity: event ordering, at-least-once vs exactly-once delivery semantics, event schema evolution, and the consistency model of the system as a whole (eventual consistency, not the transactional consistency of a shared database). These are real trade-offs, not arguments against event-driven integration — but they must be understood before adopting the pattern.

**API contract discipline** — explicit, versioned, tested API specifications — is the minimum contract between services. An API that is used by other services but not explicitly versioned will break those services when it changes. The practice of contract testing (where consumer services test their assumptions about a provider's API against a contract specification, and the provider validates its API against the same specification) catches breaking changes before deployment.

---

## Failure Isolation

One of the genuine benefits of service decomposition is failure isolation: a failure in one service should not cause failures in other services. In a monolith, a memory leak, an infinite loop, or a corrupted thread pool affects the entire application. In a correctly designed service system, a failed service degrades the capabilities of the system but does not cause other services to fail.

This benefit is not automatic. It requires deliberate design:

**Blast radius reduction** requires that services fail gracefully when their dependencies are unavailable. A service that makes synchronous calls to 10 other services, blocking on each response, has a blast radius equal to any of those 10 services. A service that makes synchronous calls only to services in its critical path, and handles unavailability of non-critical dependencies with graceful degradation, has a smaller blast radius.

**Graceful degradation** means that the service continues to provide partial functionality when dependencies are unavailable. An e-commerce product page that cannot load personalized recommendations falls back to non-personalized recommendations rather than returning an error page. The recommendation service failure is isolated; the product page serves a degraded but functional experience.

**Circuit breakers** prevent a slow or failed downstream service from exhausting the connection pool or thread pool of the upstream service that calls it. When the circuit is open (the downstream service is detected as unhealthy), calls fail fast without waiting for timeouts, and the upstream service's resources are preserved for other work.

```
  Circuit breaker state machine:
  ─────────────────────────────────────────────────────────────────
  [CLOSED] ──(failure threshold exceeded)──► [OPEN]
     ▲                                           │
     │                                      (timeout)
     │                                           ▼
     └──(success)──────────────────────── [HALF-OPEN]
                                          (test call)
  CLOSED: calls pass through normally
  OPEN:   calls fail fast, no downstream calls made
  HALF-OPEN: test calls made; if successful → CLOSED
```

---

## Anti-Patterns

**Premature service splitting** is the most common microservice anti-pattern: splitting a service before the domain boundary is understood, before the data ownership is clear, and before the team can own the resulting services independently. The resulting services share data through databases or direct coupling, defeating the purpose of the split, while inheriting all the operational overhead of distributed systems.

**Nano-services** — services so small that they represent a single function or a single database table — maximize distribution overhead while minimizing the organizational autonomy that justifies distribution. A service that contains three endpoints, is deployed 20 times a day, and requires its own CI/CD pipeline, its own Kubernetes deployment, its own secrets management, and its own observability configuration has an operational cost that vastly exceeds its contribution to the system.

The heuristic that the service should be "small enough to be rewritten in two weeks" (attributed to James Lewis and Martin Fowler in various forms) is intended to bound complexity, not to prescribe size. Services should be as large as necessary to encapsulate a coherent domain and as small as necessary to maintain clear ownership and independent deployability. The two-week rewrite heuristic is a ceiling on complexity, not a target for minimization.

**Database-per-feature confusion** occurs when a service is created for each feature rather than each domain, and each service gets its own database to maintain the data isolation principle. The result is a system with 50 services, each with its own database, where features that span multiple domains — which is most features of any significance — require coordination across many service boundaries and many databases, with eventual consistency everywhere. This is the worst of both worlds: the complexity of distribution without the domain clarity that justifies it.

---

## Decision Framework

The decision to split a service should be evaluated against this checklist. A proposed service split is appropriate only when all items can be answered affirmatively:

**Domain clarity**
- [ ] The proposed service corresponds to a named, well-understood business domain or bounded context
- [ ] The team can articulate the ubiquitous language of the domain and its differences from adjacent domains
- [ ] The domain boundary is stable enough that it is unlikely to be redrawn in the next 12 months

**Data ownership**
- [ ] The proposed service can own all data it needs without accessing another service's database
- [ ] Cross-domain data access can be modeled as events or API calls without shared database access
- [ ] The service's data model is internally consistent and does not require knowledge of other services' schemas

**Team readiness**
- [ ] A specific team or sub-team will own this service end to end: API, operations, on-call, SLO
- [ ] The owning team has the skills to operate the service independently
- [ ] The team has capacity to maintain the additional operational overhead of a new service

**Deployment maturity**
- [ ] Automated CI/CD pipeline for the new service can be created before the service goes to production
- [ ] Automated tests can validate the service in isolation
- [ ] Contract tests can validate the service's API against its consumers

**Operational pre-conditions**
- [ ] Distributed tracing is in place to debug cross-service failures
- [ ] Service health monitoring and alerting will be configured before the service is production traffic-bearing
- [ ] Runbooks for the service's failure modes will be written before the service goes to production

If more than two items in this checklist cannot be affirmed, the service should remain part of the monolith until the pre-conditions are met. The checklist is not a bureaucratic barrier; it is a description of the organizational and technical maturity that makes service decomposition tractable rather than expensive.

!!! tip "See also"
    - [Why Most Microservices Should Be Monoliths](why-most-microservices-should-be-monoliths.md) — the companion analysis of when decomposition is premature
    - [Why Most Data Pipelines Fail](why-most-data-pipelines-fail.md) — the same organizational failure modes (ownership, contracts, incentives) applied to data systems
