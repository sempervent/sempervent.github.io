---
tags:
  - deep-dive
  - infrastructure
  - devops
  - architecture
---

# Why Most Kubernetes Clusters Shouldn't Exist: Orchestration, Overhead, and Organizational Maturity

*See also: [Why Most Microservices Should Be Monoliths](why-most-microservices-should-be-monoliths.md) — the service architecture pre-condition that Kubernetes assumes, and [The Human Cost of Automation](the-human-cost-of-automation.md) — the skill atrophy and complexity transfer that Kubernetes deployment introduces.*

**Themes:** Infrastructure · Architecture · Economics

---

## Opening Thesis

Kubernetes solves large-scale distributed scheduling problems that most organizations do not have. It is a platform for running many containerized workloads across many machines with dynamic scheduling, self-healing, and fine-grained resource allocation. It is operationally correct for organizations whose engineering problems include thousands of deployable units, heterogeneous hardware pools, and multi-team platform governance at scale. For the majority of organizations that run a dozen services on a few machines, Kubernetes is an organizational investment in solving problems that do not yet exist, at the cost of introducing operational complexity that does.

The Kubernetes adoption pattern of the late 2010s and early 2020s mirrors the microservices adoption pattern: a technology created to solve real problems at extreme scale was widely adopted before the scale that justifies it was reached, and organizations that adopted it found themselves managing the operational overhead of the platform without the scale benefits that make the overhead worthwhile. The comparison is not coincidental — Kubernetes and microservices are frequently adopted together, each amplifying the other's complexity.

---

## Historical Context

### The VM Era and Slow Provisioning

Before containers, infrastructure provisioning meant virtual machines: isolated compute environments with their own OS kernel, disk image, and lifecycle. VM provisioning was slow (minutes to tens of minutes), resource-inefficient (each VM incurred full OS overhead regardless of workload), and operationally heavyweight (patching, image management, configuration drift). Configuration management tools (Chef, Puppet, Ansible) addressed the configuration drift problem but not the provisioning speed or resource efficiency problems.

The operational model of the VM era was oriented around long-lived, carefully managed instances. Deployment meant updating software on running machines, not replacing machines. Immutable infrastructure — the practice of replacing instances rather than updating them — was technically difficult with VMs because the replacement cycle was too slow to be practical for frequent deployments.

### Docker and the Container Revolution

Docker (2013) changed the operational model for application packaging and deployment. Containers shared the host kernel, eliminating the per-VM OS overhead while providing filesystem and process isolation. Build times dropped from minutes to seconds. Deployment became the act of pulling an image and starting a container, a reproducible operation that took seconds. The immutable infrastructure model became practical.

Docker Compose provided a simple orchestration model for multi-container applications on a single host: define services, their images, their environment variables, and their dependencies in a YAML file, and bring the entire stack up with a single command. For development environments and small production deployments, Docker Compose solved the multi-container coordination problem with minimal operational complexity.

### The Container Orchestration Wars

The success of Docker created a new problem: how to run containers across multiple machines, schedule them based on resource availability, and replace failed containers automatically. Multiple orchestration systems emerged in 2014–2016: Docker Swarm, Apache Mesos + Marathon, CoreOS Fleet, and Google's Kubernetes. This period — the container orchestration wars — produced genuine competition among meaningfully different architectural approaches.

Google open-sourced Kubernetes in 2014, drawing on internal experience with Borg (Google's internal cluster management system since 2003). The Cloud Native Computing Foundation (CNCF) accepted Kubernetes as its first hosted project in 2016. Major cloud providers launched managed Kubernetes services (GKE in 2015, EKS in 2018, AKS in 2017). The orchestration wars ended when cloud provider investment decisively tilted the ecosystem toward Kubernetes.

### Kubernetes Standardization and the CNCF Ecosystem

Kubernetes won the orchestration wars through a combination of technical depth, cloud provider backing, and ecosystem network effects. The CNCF ecosystem grew around it: Helm (package management), Prometheus (monitoring), Istio (service mesh), Argo (GitOps and workflows), Fluentd (logging), Envoy (proxy), and dozens of other projects that addressed the operational gaps Kubernetes left.

This ecosystem was both a strength and a warning sign. Kubernetes required so much additional infrastructure to be operationally usable that an entire ecosystem of complements emerged. A "production-ready" Kubernetes deployment frequently requires Helm, cert-manager, external-secrets, an ingress controller, a CNI plugin, Prometheus + Grafana, a service mesh or at minimum mTLS enforcement, and GitOps tooling. The platform that looked simple from the outside accumulated into a substantial operational commitment.

---

## What Kubernetes Actually Solves

Understanding when Kubernetes is appropriate requires understanding what it actually solves — not what its marketing suggests, but what problems its architecture addresses.

```
  Kubernetes architecture (simplified):
  ─────────────────────────────────────────────────────────────────
  Control Plane (manages desired state)
  ├── API Server     ← single entry point for all cluster ops
  ├── etcd           ← distributed key-value store (cluster state)
  ├── Scheduler      ← assigns pods to nodes based on resources
  └── Controller     ← reconciles desired state → actual state

  Worker Nodes (run workloads)
  ├── Node 1: [Pod A][Pod B][Pod C]   ← kubelet + container runtime
  ├── Node 2: [Pod D][Pod E]
  └── Node N: [Pod F][Pod G][Pod H]

  Users/operators declare desired state via YAML:
  "I want 5 replicas of this container with 500m CPU and 256Mi RAM"
  The scheduler places them; the controller maintains them.
```

**Scheduling**: placing workloads on nodes based on resource requests, affinity/anti-affinity rules, and node selectors. This is valuable when the number of workloads and nodes is large enough that manual placement is impractical. For 5 services across 3 nodes, manual placement is trivial.

**Self-healing**: automatically restarting failed containers, replacing failed nodes, and rescheduling pods when nodes become unhealthy. This is valuable for production workloads that must maintain availability without human intervention. For development or low-criticality workloads, the operational cost of self-healing configuration exceeds the value.

**Service discovery**: allowing services to find each other by DNS name without knowing IP addresses, which change as pods are rescheduled. This is valuable in a dynamic environment where pod IPs change frequently. In a Docker Compose environment, services find each other by service name through the Compose network.

**Horizontal scaling**: automatically adding or removing replicas based on CPU utilization, custom metrics, or scheduled rules. This is valuable for workloads with variable load and when the cost of over-provisioning static capacity exceeds the cost of Kubernetes management. For workloads with predictable, stable load, manual replica counts are simpler.

**Declarative desired state**: expressing the intended state of the system in version-controlled manifests rather than in imperative commands. This is the genuine architectural contribution of Kubernetes that persists regardless of scale — the reconciliation model is a better operational model than imperative deployment scripts. Notably, this benefit is available through lighter alternatives (Docker Compose, Nomad) that do not carry Kubernetes's full operational complexity.

---

## Hidden Costs

| Dimension | Single Node / Docker Compose | Kubernetes |
|---|---|---|
| Learning curve | Days–weeks | Months–years |
| Minimum viable setup | 1 machine, 1 tool | Control plane (3 nodes for HA) + worker nodes |
| Upgrade complexity | Pull new image, restart | Cluster version upgrade with API deprecations |
| Networking | Bridge network (trivial) | CNI plugins, iptables rules, service mesh optional |
| Storage | Volume mounts | PersistentVolumes, StorageClasses, CSI drivers |
| Secrets management | Env files or Compose secrets | Kubernetes secrets (base64, not encrypted), external-secrets operator |
| Observability | Container logs, basic metrics | Prometheus operator, service monitors, node exporters, kube-state-metrics |
| Security surface | Container daemon, host OS | API server, etcd, kubelet, RBAC, admission webhooks, network policies |
| Debugging | `docker logs`, `docker exec` | `kubectl logs`, `kubectl exec`, pod eviction investigation, OOM debugging |
| Operator expertise | One developer | Dedicated platform team recommended |
| Monthly minimum cost | ~$20–50 (single VM) | ~$200–500 (3 control plane + 2+ workers, managed) |

### Cluster Upgrades

Kubernetes releases a new minor version every 4 months. Each minor version is supported for approximately 14 months before reaching end-of-life. Organizations running Kubernetes must therefore upgrade their clusters regularly to maintain security support — a process that involves updating the control plane, updating worker nodes, and validating that workloads continue to function correctly across the version change.

Cluster upgrades in practice are not trivial. Kubernetes deprecates API versions regularly: a manifest written for Kubernetes 1.18 using the `extensions/v1beta1` Ingress API must be rewritten for 1.22, which removed that API version. Helm charts that embed deprecated APIs must be updated. Application code that uses client-go must be updated. The upgrade path requires validating all these changes before upgrading production, which requires a staging environment that mirrors production — itself additional operational infrastructure.

### etcd Fragility

etcd is the distributed key-value store that holds all Kubernetes cluster state. Every manifest, every pod status, every secret, every ConfigMap is stored in etcd. etcd's operational characteristics are demanding: it requires majority quorum for writes, it is sensitive to disk latency (SSDs are effectively required), and it must be backed up regularly because its failure or corruption is a complete cluster failure.

Managing etcd means monitoring its disk usage (etcd has a configurable storage quota that, when exceeded, causes the cluster to become read-only), monitoring its performance (high disk latency produces leader election instability), and maintaining backups with tested restoration procedures. Organizations that adopt managed Kubernetes services (EKS, GKE, AKS) offload etcd management to the cloud provider — but they are also dependent on the cloud provider's etcd management, which is not universally reliable.

### Networking Overlays

Kubernetes's networking model requires that every pod have a unique IP address reachable from any other pod anywhere in the cluster. This flat networking requirement is implemented by Container Network Interface (CNI) plugins: Flannel, Calico, Cilium, Weave Net, and others each implement the pod networking model with different performance characteristics, feature sets, and operational requirements.

CNI plugin selection, configuration, and troubleshooting is a specialized skill. Network policies (firewall rules between pods) require understanding of Kubernetes network policy semantics and CNI plugin behavior. Network performance issues require understanding the CNI plugin's encapsulation model (VXLAN, WireGuard, BGP) and how it interacts with the underlying network infrastructure.

---

## Organizational Maturity Requirements

Kubernetes is not a technology problem. It is an organizational problem: the technology is learnable, but operating it correctly requires organizational structures and cultural practices that most teams have not developed.

**Dedicated platform team**: a Kubernetes cluster that is operated by the same team that develops applications on it is a cluster that will eventually experience a cluster-level failure during a period when the team is focused on application development. Kubernetes is complex enough to warrant dedicated operational ownership — a platform engineering or infrastructure team whose primary responsibility is cluster health, not application delivery.

**CI/CD sophistication**: Kubernetes deployment via `kubectl apply` or Helm manually is operationally inferior to GitOps-driven deployment. But GitOps tooling (Argo CD, Flux) adds additional system complexity. Kubernetes-native deployment requires that the CI/CD pipeline can produce container images, push them to a registry, update Kubernetes manifests with new image tags, and trigger or wait for deployment rollout. This pipeline is substantially more complex than `docker-compose up` or `git push` to a PaaS.

**Monitoring depth**: Kubernetes-based workloads require monitoring at the container, pod, node, and cluster levels simultaneously. A pod that is consistently being OOM-killed needs different investigation than a node that has excessive disk pressure, which needs different investigation than a control plane that has high API server latency. This multi-level monitoring requires understanding what each metric means and how the levels interact.

**Incident response culture**: Kubernetes introduces failure modes that do not exist in simpler deployments — pod eviction due to node pressure, pending pods due to resource exhaustion, failed deployments due to image pull errors, broken readiness probes causing service unavailability. Responding to these incidents requires trained operators with runbooks, not developers who "also handle infrastructure."

---

## When Kubernetes Is Justified

**Multi-service dynamic scaling**: when different services have significantly different load profiles that change dynamically (a recommendation service that spikes during peak traffic hours, a background processing service that handles batch jobs at variable rates), Kubernetes's Horizontal Pod Autoscaler provides value that static Docker Compose replica counts cannot match.

**Regulated multi-tenant environments**: when regulatory requirements mandate strict isolation between tenant workloads, network policy enforcement, and audit trails for all cluster operations, Kubernetes's RBAC, network policies, and audit logging provide a comprehensive control surface that simpler environments cannot replicate.

**Complex deployment topologies**: when deployments require canary releases (10% of traffic to new version), blue-green deployments with instant traffic cutover, A/B testing with user-segment routing, or progressive delivery with automated rollback on SLO violation, Kubernetes provides the infrastructure (deployments, services, ingress) that these patterns require.

**Multi-cluster geographic resilience**: when the application requires active-active deployment across multiple cloud regions, with traffic routing based on user geography and automatic failover on region failure, Kubernetes's multi-cluster tooling (Federation, Admiral, Submariner) provides a structured approach to cross-cluster coordination.

---

## The Illusion of Portability

A frequently cited justification for Kubernetes adoption is cloud portability: Kubernetes workloads can be moved between cloud providers without changing the deployment configuration. This claim is partially true and largely misleading in practice.

The Kubernetes API is consistent across cloud providers. Workloads defined as Kubernetes Deployments and Services can be deployed on GKE, EKS, and AKS without changing the manifest YAML. This is genuine portability at the scheduling and networking layer.

What is not portable is the cloud-native infrastructure that most Kubernetes deployments depend on. Load balancers are provisioned differently on each cloud (LoadBalancer service type triggers cloud-provider-specific API calls). Persistent volumes use cloud-specific storage classes (gp3 on AWS, premium-rwo on GCP). Ingress controllers depend on cloud-specific load balancer annotations. IAM integration for pods (IRSA on AWS, Workload Identity on GCP) is cloud-specific. A "portable" Kubernetes workload that uses any of these — which is essentially every production workload — is not portable without modification.

**Helm sprawl** is the operational manifestation of dependency accumulation. A production Kubernetes cluster typically runs dozens of Helm releases: the application charts, the monitoring stack, the certificate manager, the secret management operator, the ingress controller, the GitOps operator, the cluster autoscaler. Each Helm release introduces upgrade requirements, configuration management complexity, and potential conflicts with other releases. The cluster that was supposed to simplify operations through Kubernetes has become a registry of interdependent Helm releases that must be upgraded in coordination.

---

## Decision Framework

| Factor | Docker Compose / Single Node | Kubernetes |
|---|---|---|
| Team size < 10 engineers | Strongly preferred | Premature |
| Fewer than 20 deployed services | Preferred | Rarely justified |
| No dynamic scaling requirements | Preferred | Adds overhead without benefit |
| Single cloud region | Preferred | Optional |
| Regulated multi-tenant environment | Insufficient | Often required |
| > 50 deployed services | Insufficient | Appropriate |
| Geographic multi-region resilience | Insufficient | Appropriate |
| Dedicated platform team available | Either | Required |
| CI/CD pipeline mature | Either | Required |
| Observability infrastructure in place | Either | Required |

**The question to ask before adopting Kubernetes**: what specific problem will Kubernetes solve that cannot be solved with a simpler alternative? Dynamic scheduling of unpredictable workloads, multi-tenant isolation, complex deployment topologies — these are legitimate answers. "Everyone uses Kubernetes" and "we want to be cloud-native" are not.

**Simpler alternatives that cover most use cases**:
- Docker Compose for development and small production deployments (1–5 services on 1–3 machines)
- Managed PaaS (Railway, Render, Fly.io, AWS ECS) for application workloads without platform engineering overhead
- HashiCorp Nomad for organizations that need container orchestration without Kubernetes's operational complexity
- Managed Kubernetes (EKS, GKE, AKS) with a narrow service surface — if Kubernetes is necessary, the managed control plane is almost always worth the cost

!!! tip "See also"
    - [Why Most Microservices Should Be Monoliths](why-most-microservices-should-be-monoliths.md) — the service decomposition decisions that Kubernetes's value proposition assumes
    - [Appropriate Use of Microservices](appropriate-use-of-microservices.md) — the domain isolation and team alignment prerequisites that make multi-service Kubernetes deployments tractable
    - [The Human Cost of Automation](the-human-cost-of-automation.md) — the skill atrophy and complexity transfer that Kubernetes introduction creates
    - [IaC vs GitOps](iac-vs-gitops.md) — the infrastructure automation model that Kubernetes-native deployments require
    - [The Myth of Serverless Simplicity](the-myth-of-serverless-simplicity.md) — the alternative abstraction that eliminates Kubernetes overhead by outsourcing the control plane to the cloud provider

!!! abstract "Decision Frameworks"
    The structured decision framework from this essay — organized by team size, workload volatility, compliance requirements, and cost tolerance — is indexed at [Decision Frameworks → When to Run Kubernetes](../decision-frameworks.md#when-to-run-kubernetes). For the recurring mistake this essay addresses, see [Anti-Patterns → Overusing Kubernetes](../anti-patterns.md#overusing-kubernetes).
