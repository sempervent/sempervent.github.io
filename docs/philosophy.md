---
tags:
  - guides
  - philosophy
---

# Systems engineering principles

These are the principles I use when judging architecture, platforms, and organizational trade-offs. The [deep dives](deep-dives/index.md), [best practices](best-practices/index.md), and [tutorials](tutorials/index.md) apply them to specific domains.

---

## Restraint Over Hype

Every decade produces a new class of technologies that promise to eliminate the problems of the previous decade. Microservices promised to fix the monolith. Kubernetes promised to fix microservice deployment. Serverless promised to fix Kubernetes overhead. The promise is always partially true and the cost is always underestimated.

The appropriate response to a new technology is not adoption and not rejection, but analysis: what problem does this solve, what does it cost, and do I have that problem at that cost? The answer is frequently "not yet" or "not for this use case."

Restraint is not conservatism. It is the discipline of not paying complexity costs for capabilities you do not need.

---

## Economics Over Fashion

Technical decisions have economic consequences that outlast the enthusiasm that produced them. A Kubernetes cluster adopted because the industry had standardized on Kubernetes carries its operational costs — cluster upgrades, etcd management, CNI plugin selection, RBAC sprawl — long after the standardization narrative has shifted to the next platform.

Economics here means the full accounting: engineering time, infrastructure cost, opportunity cost, organizational learning overhead, and the switching cost of the decision being wrong. A technology that looks cheap on a benchmark but expensive in operation has been incorrectly evaluated.

Ask what a choice actually costs, over what time horizon, in organizations of what size and maturity. That answer is more durable than any benchmark.

---

## Determinism Over Abstraction

Abstractions are valuable precisely because they hide detail. They become a liability when the hidden detail becomes a failure mode. An engineer who does not understand what happens when a serverless function cold starts, what etcd does when disk latency spikes, or what "eventual consistency" means for their read pattern has adopted an abstraction without understanding its failure envelope.

Determinism — the ability to predict and reason about system behavior under load, failure, and edge conditions — requires understanding the implementation that abstractions hide. Reliable engineering judgment often requires descending below the abstraction layer: into storage physics, network protocol semantics, and scheduling mechanics.

---

## Governance Over Chaos

Data systems, infrastructure, and organizational processes that grow without governance accumulate entropy. Data lakes become swamps. Microservice deployments become dependency graphs no one understands. Kubernetes clusters accumulate Helm releases that no one is responsible for. Metadata becomes stale. Pipelines multiply without ownership.

Governance is not bureaucracy; it is the structural discipline that makes growth sustainable. Schema contracts, ownership models, metadata enforcement, and recorded architectural decisions are mechanisms that keep a system understandable as it grows. The cost of governance is usually lower than the cost of rearchitecting from chaos.

---

## Discipline Over Novelty

The most durable engineering decisions are conservative: they use the simplest technology that solves the problem, they prefer well-understood failure modes over novel ones, and they resist organizational pressure to adopt new tools before existing tools have been exhausted.

That is not hostility to new technology. DuckDB, H3, Apache Iceberg, and OpenTelemetry appear throughout these notes because they represent genuine improvements in specific domains — always with context on what problem they solve, what they cost, and when the older approach remains correct.

Novelty is a feature when it solves a real problem. It is a liability when it is pursued for its own sake.

---

## Related reading

- [Deep Dives](deep-dives/index.md) — long-form essays on specific architectural decisions
- [Reading Tracks](reading-tracks.md) — curated sequences by domain
- [Decision Frameworks](decision-frameworks.md) — structured decision tools extracted from the essays
- [Anti-Patterns](anti-patterns.md) — recurring mistakes these principles are meant to prevent
- [Systems Glossary](systems-glossary.md) — shared vocabulary across the essays
