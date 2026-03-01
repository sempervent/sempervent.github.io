---
tags:
  - deep-dive
  - devops
  - infrastructure
---

# Infrastructure as Code vs GitOps: Control Planes, Drift, and the Politics of Automation

**Themes:** Infrastructure · Governance · Automation

---

## Two Philosophies of Infrastructure Control

Infrastructure as Code (IaC) and GitOps are often described as complementary or as an evolution of one from the other. This is partly true and partly a simplification that obscures a genuine philosophical difference in how each model thinks about infrastructure state, human governance, and the relationship between intent and reality.

IaC is a practice: infrastructure is defined as code, version-controlled, and applied via tooling that translates that code into API calls against cloud providers or infrastructure systems. The model is imperative-adjacent — humans decide when to apply changes, run a plan, and approve execution. The code is the source of truth for intent; actual infrastructure state may diverge.

GitOps is a pattern: a Git repository is the source of truth for desired system state, and an automated agent continuously reconciles the actual system state toward the desired state. No human applies changes manually; the reconciliation loop is always running. Drift is automatically corrected. The model is explicitly declarative and continuously enforced.

These are different control models, not just different tools. The choice between them is a governance decision as much as a technical one.

---

## The Terraform Model

Terraform (HashiCorp, 2014) defines infrastructure through HCL (HashiCorp Configuration Language) resource declarations. A `terraform apply` reads the current state file, queries the actual infrastructure state, computes a diff, and executes the minimum set of API calls to bring actual state into alignment with declared state.

```
  Terraform execution model:
  ──────────────────────────────────────────────────────────────────
  [HCL configuration files]
          │
  terraform plan
          │ compares against state file + actual API state
          ▼
  [Execution plan: create/modify/destroy resources]
          │ human reviews and approves
  terraform apply
          │ executes API calls
          ▼
  [Infrastructure: AWS/GCP/Azure resources]
          │ state written
  [terraform.tfstate]
```

The state file is both Terraform's strength and its primary operational vulnerability. The state file records the mapping between Terraform resource addresses and real infrastructure resource IDs. Without an accurate state file, Terraform cannot know what already exists and cannot plan correctly. State file corruption, concurrent modification by multiple operators, or state file drift from manual infrastructure changes all produce incorrect plans that may destroy resources or fail to create expected resources.

Terraform's drift detection is on-demand, not continuous. A `terraform plan` detects drift at the moment it is run. Between runs, infrastructure can be modified manually, by other automation, or by the cloud provider (capacity events, security mitigations, region maintenance), and Terraform has no visibility into this until the next plan.

The governance model is human-gated: Terraform does nothing without a human running `apply`. This is a feature for many organizations — infrastructure changes require human review and explicit execution — and a limitation for continuous delivery workflows that require infrastructure to self-heal or auto-scale without human intervention.

---

## The Kubernetes Reconciliation Model

Kubernetes, as a platform, is designed around a reconciliation loop at every level of its architecture. A controller watches the desired state (declared in etcd via the API server) and the actual state (running pods, services, deployments) and continuously works to close the gap.

```
  Kubernetes reconciliation model:
  ──────────────────────────────────────────────────────────────────
  [Git repository: YAML manifests]
          │ watched by GitOps operator (Argo CD / Flux)
          ▼
  [Desired state in cluster (etcd)]
          │
  [Controllers: Deployment, Service, etc.]
          │ watch desired state, observe actual state
          ▼
  [Actual state: running pods, services, endpoints]
          │ continuously reconciled
          ▲ any drift → controller corrects
```

GitOps as a pattern (formalized by Weaveworks around 2017–2018) applies this reconciliation model to the deployment lifecycle: a GitOps operator (Argo CD, Flux) watches a Git repository and applies any changes to the cluster automatically. Manual `kubectl apply` commands are replaced by Git commits. Cluster state mirrors Git state continuously.

The GitOps reconciliation loop provides continuous drift correction that Terraform's model does not: if a resource is deleted manually, the GitOps operator recreates it. If a ConfigMap is modified out-of-band, the operator overwrites it. The Git repository is the authoritative source of truth, enforced automatically.

---

## Drift Detection: On-Demand vs Continuous

Drift — the divergence between declared and actual infrastructure state — is the central operational concern for both models. How each model handles drift is its defining characteristic.

```
  Drift detection comparison:
  ──────────────────────────────────────────────────────────────────
  Terraform:
  Time:   t0        t1 (drift)    t2 (plan)    t3 (apply)
  State:  Declared ──► Drifted ────────────────► Corrected
                                  ^
                               Human runs plan/apply

  GitOps (Argo CD):
  Time:   t0        t1 (drift)    t1+30s
  State:  Desired ──► Drifted ──► Auto-corrected
                              ^
                         Reconciliation loop
```

The practical implication of continuous reconciliation is significant: in a GitOps-managed cluster, a developer cannot apply a "quick fix" directly to production with `kubectl` and have it persist. The fix will be overwritten within the reconciliation interval (typically 1–5 minutes). This is the GitOps model's security and governance property: all changes must go through Git, and Git provides the audit trail.

This property is also the source of GitOps friction: for genuinely urgent production situations (a pod crashing due to a bad config, a service endpoint misconfigured), the requirement to commit, push, and wait for a reconciliation cycle adds latency. Teams that adopt GitOps strictly must have practiced runbooks for emergency situations that bypass or accelerate the normal git-commit flow.

---

## Human Governance vs Automation

The governance philosophies of IaC and GitOps reflect different assumptions about trust and risk.

**Terraform's governance model** is human-gated by default. Infrastructure changes require a human operator to run `plan`, review the output, and run `apply`. This creates natural review checkpoints and ensures that no infrastructure change is applied without at least one human seeing it. The trade-off is operational tempo: infrastructure changes are as fast as humans can review and approve them.

CI/CD pipelines can automate `terraform apply` on merge to main, which approaches GitOps behavior, but the state management complexity (locking, state storage) and the absence of continuous reconciliation mean it is not equivalent.

**GitOps' governance model** is automation-gated. Infrastructure changes go through Git — pull request review, CI validation, branch protection — and are then applied automatically without further human action. The governance is front-loaded into the PR review process. The trade-off is that the review process must be robust enough to catch problems before merge, because post-merge there is no "apply approval" step.

---

## When GitOps Fails

GitOps is not universally superior. Its failure modes are specific and worth understanding before adopting it.

**Bootstrapping problem**: to install a GitOps operator, you must have a cluster. To have a cluster, you must provision infrastructure. Terraform (or another IaC tool) is typically used for the initial cluster provisioning; GitOps takes over afterward. The boundary between "what Terraform manages" and "what GitOps manages" requires explicit architectural design.

**Secret management**: secrets cannot be committed to Git in plaintext. GitOps requires an out-of-band secret management mechanism: SealedSecrets, External Secrets Operator, Vault Agent, or AWS Secrets Manager injection. The complexity of this layer is often underestimated.

**Stateful workloads**: databases, stateful sets with persistent volumes, and services with complex upgrade procedures are difficult to manage through a purely declarative reconciliation model. A `helm upgrade` may require manual steps (schema migrations, backup before upgrade) that cannot be expressed as Git state changes.

**Multi-cloud or non-Kubernetes resources**: GitOps operators (Argo CD, Flux) manage Kubernetes resources natively. Managing AWS RDS instances, S3 bucket policies, or Route53 records through GitOps requires either Crossplane (which implements a Kubernetes operator for cloud resources) or a hybrid model where Terraform manages non-Kubernetes resources alongside GitOps for Kubernetes resources.

---

## Hybrid Models in Practice

Most production environments use a hybrid model, whether explicitly designed that way or arrived at through pragmatic evolution.

```
  Practical hybrid architecture:
  ──────────────────────────────────────────────────────────────────────
  Terraform manages:
  ├── VPC, subnets, security groups
  ├── EKS/GKE cluster (Kubernetes control plane)
  ├── RDS databases, Elasticache, S3 buckets
  ├── IAM roles and policies
  └── DNS records, certificates

  GitOps (Argo CD/Flux) manages:
  ├── Kubernetes Deployments, Services, ConfigMaps
  ├── Helm chart releases
  ├── Custom Resource Definitions (operators)
  ├── Namespace RBAC
  └── Application configuration
```

The boundary is typically: Terraform manages infrastructure that exists outside the Kubernetes API surface; GitOps manages everything that is a Kubernetes resource. This division is clean enough to be workable and explicit enough to avoid overlap.

Crossplane blurs this boundary by implementing Kubernetes operators for cloud resources, allowing an S3 bucket or an RDS instance to be declared as a Kubernetes Custom Resource and managed through GitOps. This eliminates the dual-tool model at the cost of additional complexity in the Kubernetes control plane.

---

## Decision Framework

**GitOps is appropriate when**:
- The primary infrastructure surface is Kubernetes or Kubernetes-compatible (via Crossplane)
- The team has strong Git workflow practices and can enforce PR-based governance
- Continuous drift correction is required (compliance, security-sensitive workloads)
- The team can invest in solving the secret management and bootstrap problems

**Terraform is appropriate when**:
- Infrastructure spans multiple clouds or includes significant non-Kubernetes resources
- Human gate-keeping on infrastructure changes is a compliance or governance requirement
- The team lacks Kubernetes expertise but has operational experience with cloud CLIs and APIs
- The state of infrastructure changes slowly enough that on-demand drift detection is acceptable

**Neither model fully solves the governance problem**. Both require human judgment in the design of the system — which resources are managed, what drift correction means for stateful workloads, how secrets are managed, and how emergencies are handled. The automation handles execution; the humans design what should be executed and under what conditions.

!!! tip "See also"
    - [Container Base Image Philosophy](container-base-image-philosophy.md) — the container layer that GitOps and IaC deploy
    - [The Human Cost of Automation](the-human-cost-of-automation.md) — the skill atrophy and complexity transfer that GitOps and IaC introduce as organizational side effects
