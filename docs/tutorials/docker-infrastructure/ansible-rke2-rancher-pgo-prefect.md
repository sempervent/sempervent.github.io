# Building a 2×8 RKE2 Cluster with Rancher, PGO, and Prefect Using Ansible

**Objective**: Master automated Kubernetes cluster provisioning with Ansible. When you need to provision production-grade RKE2 clusters, when you want Rancher for cluster management, when you need Postgres operators and workflow orchestration—this tutorial becomes your weapon of choice.

When you're tired of hand-taming clusters and want a machine to build other machines, this is the path: Ansible rides herd on bare nodes, RKE2 welds them into a cluster, Rancher gives you a GUI throne, PGO breeds Postgres instances like bioengineered livestock, and Prefect runs your flows in the middle of it all.

## Prerequisites

This tutorial assumes:

- You have 10 Linux nodes reachable by SSH:
  - `rke2-head-[1..2]`
  - `rke2-node-[1..8]`
- Ansible control machine with:
  - `ansible ≥ 2.14`
  - `python3`, ssh access
- A DNS or load-balanced endpoint you can point at Rancher later.

## Architecture Overview

### Cluster Layout

```mermaid
flowchart LR
    subgraph ControlPlane["Control Plane (RKE2)"]
        H1["Head-1 (server)"]
        H2["Head-2 (server)"]
    end
    
    subgraph Workers["Worker Nodes (RKE2)"]
        W1["node-1"]
        W2["node-2"]
        W3["node-3"]
        W4["node-4"]
        W5["node-5"]
        W6["node-6"]
        W7["node-7"]
        W8["node-8"]
    end
    
    subgraph K8S["Kubernetes Cluster"]
        R["Rancher"]
        PGO["Zalando PGO Operator"]
        PF["Prefect Agents + Flows"]
    end
    
    ControlPlane --> K8S
    Workers --> K8S
    R --> PF
    PF --> PGO
    PGO -->|"Postgres Clusters"| PF
```

### High-Level Flow

- **Ansible** → installs RKE2 on all 10 nodes
- **Head node 1** bootstraps the cluster; **head node 2** joins as a second server
- **Worker nodes** join as agents
- Once cluster is up, we deploy:
  - **Rancher** via Helm
  - **PGO operator** via Helm/YAML
  - **Prefect** (agent + deployments) into a dedicated namespace

## 1. Ansible Project Layout

Use a clean, boring structure so future-you doesn't have to hunt through rubble.

```
ansible-rke2/
├── ansible.cfg
├── inventories
│   └── prod
│       ├── hosts.ini
│       └── group_vars
│           ├── all.yml
│           ├── rke2_servers.yml
│           └── rke2_agents.yml
├── roles
│   ├── base
│   │   ├── tasks/main.yml
│   │   └── templates/...
│   ├── rke2
│   │   ├── tasks/
│   │   │   ├── install_server.yml
│   │   │   ├── install_agent.yml
│   │   │   └── main.yml
│   │   └── templates/config.yaml.j2
│   ├── rancher
│   │   └── tasks/main.yml
│   ├── pgo
│   │   └── tasks/main.yml
│   └── prefect
│       └── tasks/main.yml
└── site.yml
```

### ansible.cfg

```ini
[defaults]
inventory = inventories/prod/hosts.ini
host_key_checking = False
forks = 20
timeout = 30
deprecation_warnings = False
retry_files_enabled = False
interpreter_python = auto_silent
```

### Inventory: inventories/prod/hosts.ini

```ini
[rke2_servers]
rke2-head-1 ansible_host=10.0.0.11
rke2-head-2 ansible_host=10.0.0.12

[rke2_agents]
rke2-node-1 ansible_host=10.0.0.21
rke2-node-2 ansible_host=10.0.0.22
rke2-node-3 ansible_host=10.0.0.23
rke2-node-4 ansible_host=10.0.0.24
rke2-node-5 ansible_host=10.0.0.25
rke2-node-6 ansible_host=10.0.0.26
rke2-node-7 ansible_host=10.0.0.27
rke2-node-8 ansible_host=10.0.0.28

[all:vars]
ansible_user=your_ssh_user
ansible_become=true
```

### Group vars: inventories/prod/group_vars/all.yml

```yaml
kubernetes_version: "v1.30.0"   # used only for reference/logging
rke2_version: "v1.30.0+rke2r1"

# Where RKE2 stores config and state
rke2_config_dir: /etc/rancher/rke2
rke2_server_token_file: "{{ rke2_config_dir }}/server_token"
rke2_server_url: "https://10.0.0.11:9345"  # points to head-1

# Rancher
rancher_namespace: cattle-system
rancher_version: "2.9.0"
rancher_hostname: "rancher.example.com"

# PGO
pgo_namespace: postgres-operator
pgo_release_name: postgres-operator
pgo_chart_version: "5.6.0"  # adjust as needed

# Prefect
prefect_namespace: prefect
prefect_image: "prefecthq/prefect:3.0.0"   # adjust as needed
prefect_api_url: "https://prefect.your-domain.com/api"  # or your cloud/local URL
prefect_work_queue: "k8s-default"
```

### Group vars: rke2_servers.yml

```yaml
rke2_role: "server"
```

### Group vars: rke2_agents.yml

```yaml
rke2_role: "agent"
```

## 2. Base OS Prep Role

Get the nodes into a predictable, civilized state.

**roles/base/tasks/main.yml**

```yaml
---
- name: Ensure basic packages are installed
  package:
    name:
      - curl
      - wget
      - jq
      - vim
      - net-tools
    state: present

- name: Disable swap
  command: swapoff -a
  when: ansible_swaptotal_mb > 0
  become: true

- name: Comment out swap in fstab
  replace:
    path: /etc/fstab
    regexp: '^(.*\s+swap\s+.*)$'
    replace: '# \1'
  become: true

- name: Set sysctl params for Kubernetes
  copy:
    dest: /etc/sysctl.d/99-kubernetes.conf
    content: |
      net.bridge.bridge-nf-call-ip6tables = 1
      net.bridge.bridge-nf-call-iptables = 1
      net.ipv4.ip_forward = 1
  become: true

- name: Apply sysctl params
  command: sysctl --system
  become: true
```

## 3. RKE2 Installation Role

We'll:

- Install RKE2 server on the two heads
- On the first head:
  - Bootstrap cluster
  - Grab server token
- On the second head + agents:
  - Join cluster using that token

### RKE2 Config Template: roles/rke2/templates/config.yaml.j2

For servers:

```yaml
token: "{{ lookup('file', rke2_server_token_file) if inventory_hostname != groups['rke2_servers'][0] else '' }}"
write-kubeconfig-mode: "0644"
tls-san:
  - "{{ rancher_hostname }}"
  - "10.0.0.11"
  - "10.0.0.12"
node-taint:
  - "CriticalAddonsOnly=true:NoExecute"
```

For agents, we'll use different template logic in tasks.

### roles/rke2/tasks/install_server.yml

```yaml
---
- name: Download RKE2 server install script
  get_url:
    url: https://get.rke2.io
    dest: /tmp/get_rke2.sh
    mode: '0755'

- name: Install RKE2 server
  command: "INSTALL_RKE2_TYPE=server INSTALL_RKE2_VERSION={{ rke2_version }} /tmp/get_rke2.sh"
  args:
    creates: /usr/local/bin/rke2
  become: true

- name: Enable and start RKE2 server
  systemd:
    name: rke2-server
    enabled: true
    state: started
  become: true

- name: Wait for kubeconfig to exist
  stat:
    path: /etc/rancher/rke2/rke2.yaml
  register: kubeconfig
  until: kubeconfig.stat.exists
  retries: 40
  delay: 15
  become: true

- name: Extract server token (on first server only)
  when: inventory_hostname == groups['rke2_servers'][0]
  shell: "cat /var/lib/rancher/rke2/server/node-token"
  register: rke2_token
  become: true

- name: Save server token to file
  when: inventory_hostname == groups['rke2_servers'][0]
  copy:
    dest: "{{ rke2_server_token_file }}"
    content: "{{ rke2_token.stdout }}"
    mode: '0600'
  become: true
```

### roles/rke2/tasks/install_agent.yml

```yaml
---
- name: Copy server token from head-1
  delegate_to: "{{ groups['rke2_servers'][0] }}"
  run_once: true
  slurp:
    src: "{{ rke2_server_token_file }}"
  register: rke2_token_b64
  become: true

- name: Write token file on agents
  copy:
    dest: "{{ rke2_server_token_file }}"
    content: "{{ rke2_token_b64.content | b64decode }}"
    mode: '0600'
  become: true

- name: Download RKE2 agent install script
  get_url:
    url: https://get.rke2.io
    dest: /tmp/get_rke2.sh
    mode: '0755'

- name: Install RKE2 agent
  command: "INSTALL_RKE2_TYPE=agent INSTALL_RKE2_VERSION={{ rke2_version }} /tmp/get_rke2.sh"
  args:
    creates: /usr/local/bin/rke2-agent
  become: true

- name: Configure RKE2 agent
  copy:
    dest: "{{ rke2_config_dir }}/config.yaml"
    content: |
      server: "{{ rke2_server_url }}"
      token: "{{ rke2_token_b64.content | b64decode }}"
  become: true

- name: Enable and start RKE2 agent
  systemd:
    name: rke2-agent
    enabled: true
    state: started
  become: true
```

### roles/rke2/tasks/main.yml

```yaml
---
- name: Include base role
  import_role:
    name: base

- name: Install RKE2 on servers
  when: rke2_role == "server"
  include_tasks: install_server.yml

- name: Install RKE2 on agents
  when: rke2_role == "agent"
  include_tasks: install_agent.yml
```

### Cluster Bring-Up in site.yml

```yaml
---
- hosts: rke2_servers:rke2_agents
  become: true
  roles:
    - rke2
```

Once this play runs successfully, you should be able to:

```bash
# On your Ansible controller, fetch kubeconfig from head-1:
scp your_user@10.0.0.11:/etc/rancher/rke2/rke2.yaml ~/.kube/config-rke2

# Use it:
KUBECONFIG=~/.kube/config-rke2 kubectl get nodes
```

## 4. Installing Rancher via Ansible

We'll install Helm, set up a cert-manager (optional but common), and deploy Rancher.

### roles/rancher/tasks/main.yml

```yaml
---
- name: Install Helm (if not present)
  become: true
  shell: |
    if ! command -v helm >/dev/null 2>&1; then
      curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
    fi
  args:
    executable: /bin/bash

- name: Ensure namespace for Rancher exists
  kubernetes.core.k8s:
    state: present
    api_version: v1
    kind: Namespace
    name: "{{ rancher_namespace }}"
  delegate_to: localhost
  vars:
    ansible_python_interpreter: "{{ ansible_playbook_python }}"
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"

- name: Add Rancher Helm repo
  community.kubernetes.helm_repository:
    name: rancher-stable
    repo_url: https://releases.rancher.com/server-charts/stable
  delegate_to: localhost

- name: Deploy Rancher via Helm
  community.kubernetes.helm:
    name: rancher
    chart_ref: rancher-stable/rancher
    namespace: "{{ rancher_namespace }}"
    create_namespace: false
    values:
      hostname: "{{ rancher_hostname }}"
      replicas: 2
      bootstrapPassword: "ChangeMeNow!"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
```

!!! note "KUBECONFIG Requirement"
    This assumes `KUBECONFIG` is set on your Ansible control machine to point at the RKE2 cluster kubeconfig.

### Update site.yml to add Rancher

```yaml
---
- hosts: rke2_servers:rke2_agents
  become: true
  roles:
    - rke2

- hosts: localhost
  roles:
    - rancher
```

## 5. Installing Zalando PGO Operator

Zalando's Postgres Operator (commonly PGO) can be installed via Helm or raw manifests. Here we'll assume a Helm-based approach.

### roles/pgo/tasks/main.yml

```yaml
---
- name: Ensure PGO namespace exists
  kubernetes.core.k8s:
    state: present
    api_version: v1
    kind: Namespace
    name: "{{ pgo_namespace }}"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"

- name: Add PGO Helm repo
  community.kubernetes.helm_repository:
    name: zalando
    repo_url: https://opensource.zalando.com/postgres-operator/charts/postgres-operator
  delegate_to: localhost

- name: Install PGO Helm chart
  community.kubernetes.helm:
    name: "{{ pgo_release_name }}"
    chart_ref: zalando/postgres-operator
    namespace: "{{ pgo_namespace }}"
    create_namespace: false
    values:
      configGeneral:
        enable_crd_registration: true
      configKubernetes:
        pod_service_account_name: "postgres-operator"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
```

You can also include a simple demo Postgres cluster later, but for now we just want the operator present.

### Add to site.yml

```yaml
---
- hosts: rke2_servers:rke2_agents
  become: true
  roles:
    - rke2

- hosts: localhost
  roles:
    - rancher
    - pgo
```

## 6. Prefect Execution Environment

We'll deploy:

- A Prefect namespace
- A Kubernetes agent pointing to your Prefect API
- Optionally, a simple demo flow deployment manifest

### Example Prefect Agent Deployment (K8s YAML)

**roles/prefect/files/prefect-agent.yaml:**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: prefect
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prefect-agent
  namespace: prefect
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prefect-agent
  template:
    metadata:
      labels:
        app: prefect-agent
    spec:
      containers:
        - name: prefect-agent
          image: prefecthq/prefect:3.0.0
          args:
            - "prefect"
            - "agent"
            - "start"
            - "-q"
            - "k8s-default"
          env:
            - name: PREFECT_API_URL
              value: "https://prefect.your-domain.com/api"
            - name: PREFECT_LOGGING_LEVEL
              value: "INFO"
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
```

### roles/prefect/tasks/main.yml

```yaml
---
- name: Apply Prefect namespace and agent
  kubernetes.core.k8s:
    state: present
    src: "{{ role_path }}/files/prefect-agent.yaml"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
```

### Add the role to site.yml:

```yaml
---
- hosts: rke2_servers:rke2_agents
  become: true
  roles:
    - rke2

- hosts: localhost
  roles:
    - rancher
    - pgo
    - prefect
```

## 7. End-to-End Flow

To see the whole deranged ballet from Ansible to flows:

```mermaid
sequenceDiagram
    participant You as You
    participant Ansible as Ansible Controller
    participant Heads as RKE2 Heads (2)
    participant Nodes as RKE2 Workers (8)
    participant K8S as RKE2 Cluster
    participant Rancher as Rancher
    participant PGO as PGO Operator
    participant Prefect as Prefect Agent
    
    You->>Ansible: run `ansible-playbook site.yml`
    Ansible->>Heads: install base, RKE2 server
    Ansible->>Nodes: install base, RKE2 agent
    Heads-->>K8S: form control plane
    Nodes-->>K8S: join as workers
    Ansible->>K8S: install Rancher via Helm
    Ansible->>K8S: install PGO operator via Helm
    Ansible->>K8S: apply Prefect agent manifest
    Rancher-->>You: UI for cluster & workloads
    PGO-->>K8S: Postgres CRDs & operator
    Prefect-->>K8S: runs flows as jobs
    You-->>Prefect: register flows, create deployments
```

## 8. Creating a Postgres Cluster with PGO

Once PGO is installed, you can create Postgres clusters using Custom Resources. Here's an example cluster definition that Prefect can use.

### roles/pgo/files/postgres-cluster.yaml

```yaml
apiVersion: "acid.zalan.do/v1"
kind: postgresql
metadata:
  name: prefect-postgres
  namespace: default
spec:
  dockerImage: registry.opensource.zalan.do/acid/spilo-15:2.1-p7
  teamId: "prefect"
  volume:
    size: 50Gi
    storageClass: local-path  # adjust to your storage class
  numberOfInstances: 2
  users:
    prefect:
      - superuser
      - createdb
  databases:
    prefect: prefect
  postgresql:
    version: "15"
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "4MB"
      min_wal_size: "1GB"
      max_wal_size: "4GB"
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2
      memory: 2Gi
  enableMasterLoadBalancer: false
  enableReplicaLoadBalancer: false
```

### Update roles/pgo/tasks/main.yml to include cluster creation

Add this task after the PGO operator installation:

```yaml
- name: Create example Postgres cluster for Prefect
  kubernetes.core.k8s:
    state: present
    src: "{{ role_path }}/files/postgres-cluster.yaml"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
  when: create_example_cluster | default(true)
```

### Get Postgres connection details

After the cluster is created, PGO will create a service and secrets. To get connection details:

```bash
# Get the service endpoint
KUBECONFIG=~/.kube/config-rke2 kubectl get svc prefect-postgres -n default

# Get connection credentials
KUBECONFIG=~/.kube/config-rke2 kubectl get secret prefect.prefect-postgres.credentials -n default -o jsonpath='{.data.username}' | base64 -d
KUBECONFIG=~/.kube/config-rke2 kubectl get secret prefect.prefect-postgres.credentials -n default -o jsonpath='{.data.password}' | base64 -d
```

## 9. Prefect Deployment with Postgres Backend

Create a Prefect deployment that uses the Postgres cluster as its result storage and database backend.

### roles/prefect/files/prefect-deployment.yaml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: prefect-postgres-credentials
  namespace: prefect
type: Opaque
stringData:
  username: prefect
  password: ""  # Will be populated from PGO secret
  host: prefect-postgres.default.svc.cluster.local
  port: "5432"
  database: prefect
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prefect-config
  namespace: prefect
data:
  PREFECT_API_URL: "https://prefect.your-domain.com/api"
  PREFECT_RESULTS_PERSIST_BY_DEFAULT: "true"
  PREFECT_RESULTS_SERVER: "postgresql+asyncpg://prefect:CHANGE_ME@prefect-postgres.default.svc.cluster.local:5432/prefect"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: prefect-deployment-example
  namespace: prefect
spec:
  template:
    spec:
      serviceAccountName: prefect-agent
      containers:
        - name: prefect-deployment
          image: prefecthq/prefect:3.0.0
          command:
            - /bin/sh
            - -c
            - |
              # Wait for Postgres to be ready
              until pg_isready -h prefect-postgres.default.svc.cluster.local -p 5432 -U prefect; do
                echo "Waiting for Postgres..."
                sleep 2
              done
              
              # Create deployment
              prefect deployment create \
                --name example-flow \
                --work-queue k8s-default \
                --work-pool-name kubernetes \
                --flow-name example_flow
          envFrom:
            - configMapRef:
                name: prefect-config
            - secretRef:
                name: prefect-postgres-credentials
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
      restartPolicy: Never
```

### Example Prefect Flow

Create a simple example flow that uses Postgres:

**roles/prefect/files/example-flow.py**

```python
from prefect import flow, task
from prefect_sqlalchemy import SqlAlchemyConnector
import asyncio

@task
async def create_table(connector: SqlAlchemyConnector):
    async with await connector.get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_runs (
                id SERIAL PRIMARY KEY,
                flow_name VARCHAR(255),
                run_time TIMESTAMP DEFAULT NOW()
            )
        """)

@task
async def insert_run(connector: SqlAlchemyConnector, flow_name: str):
    async with await connector.get_connection() as conn:
        await conn.execute(
            "INSERT INTO flow_runs (flow_name) VALUES (:name)",
            {"name": flow_name}
        )

@flow(name="example_flow")
async def example_flow():
    connector = SqlAlchemyConnector.load("prefect-postgres")
    await create_table(connector)
    await insert_run(connector, "example_flow")
    print("Flow completed successfully!")

if __name__ == "__main__":
    asyncio.run(example_flow())
```

### Update roles/prefect/tasks/main.yml

Add tasks to create the deployment and sync the flow:

```yaml
---
- name: Apply Prefect namespace and agent
  kubernetes.core.k8s:
    state: present
    src: "{{ role_path }}/files/prefect-agent.yaml"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"

- name: Get Postgres password from PGO secret
  kubernetes.core.k8s_info:
    api_version: v1
    kind: Secret
    name: prefect.prefect-postgres.credentials
    namespace: default
  register: pg_secret
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"

- name: Update Prefect deployment secret with Postgres password
  kubernetes.core.k8s:
    state: present
    definition:
      apiVersion: v1
      kind: Secret
      metadata:
        name: prefect-postgres-credentials
        namespace: prefect
      type: Opaque
      stringData:
        username: prefect
        password: "{{ pg_secret.resources[0].data.password | b64decode }}"
        host: prefect-postgres.default.svc.cluster.local
        port: "5432"
        database: prefect
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
  when: pg_secret.resources | length > 0

- name: Apply Prefect deployment manifest
  kubernetes.core.k8s:
    state: present
    src: "{{ role_path }}/files/prefect-deployment.yaml"
  delegate_to: localhost
  environment:
    KUBECONFIG: "{{ lookup('env', 'KUBECONFIG') }}"
```

## 10. Sanity Checks

Once the smoke clears:

```bash
# Nodes
KUBECONFIG=~/.kube/config-rke2 kubectl get nodes

# Rancher
KUBECONFIG=~/.kube/config-rke2 kubectl get pods -n cattle-system

# PGO
KUBECONFIG=~/.kube/config-rke2 kubectl get pods -n postgres-operator

# Postgres cluster
KUBECONFIG=~/.kube/config-rke2 kubectl get postgresql -n default
KUBECONFIG=~/.kube/config-rke2 kubectl get pods -l application=spilo -n default

# Prefect
KUBECONFIG=~/.kube/config-rke2 kubectl get pods -n prefect
```

If those all return healthy pods and Ready nodes, you've got:

- ✅ A 2×8 RKE2 cluster
- ✅ Rancher watching over it
- ✅ PGO ready to spin Postgres clusters
- ✅ A Postgres cluster running for Prefect
- ✅ Prefect lurking, ready to execute flows with Postgres backend

## 11. Running the Full Stack

Execute the complete automation:

```bash
# Set KUBECONFIG (will be used by localhost tasks)
export KUBECONFIG=~/.kube/config-rke2

# Run the playbook
ansible-playbook site.yml

# Monitor progress
watch -n 5 'KUBECONFIG=~/.kube/config-rke2 kubectl get nodes,pods -A'
```

## See Also

- **[Ansible Playbook Design](../../best-practices/docker-infrastructure/ansible-playbook-design.md)** - Build maintainable, testable automation workflows
- **[Ansible Inventory Management](../../best-practices/docker-infrastructure/ansible-inventory-management.md)** - Master inventory design for scalable automation
- **[RKE2 on Raspberry Pi Farm](../docker-infrastructure/rke2-raspberry-pi.md)** - RKE2 deployment patterns
- **[Prefect FIFO Flow with Redis](../system-administration/prefect-fifo-redis.md)** - Prefect workflow patterns

---

*This tutorial provides the complete machinery for automated Kubernetes cluster provisioning. The patterns scale from development to production, from simple clusters to enterprise-grade deployments with full observability and workflow orchestration.*

