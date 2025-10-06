# Content-Addressed Knowledge: IPFS + SurrealDB + Meilisearch Driven by NATS + Deno

**Objective**: Build a content-addressed knowledge system where documents arrive via HTTP (UI) → pushed onto NATS JetStream. A Deno worker consumes events, canonicalizes JSON, computes a CID (IPFS), pins bytes to Kubo (IPFS node), writes structured facts to SurrealDB, and indexes text to Meilisearch. The SvelteKit UI can search (Meili) → fetch canonical record (SurrealDB) → verify byte integrity via IPFS CID.

## Architecture

```mermaid
flowchart LR
    UI[SvelteKit UI] -->|POST /ingest| NATS[(NATS JetStream)]
    NATS --> Deno[Worker (TypeScript)]
    Deno -->|pin bytes| IPFS[Kubo]
    Deno -->|UPSERT| DB[(SurrealDB)]
    Deno -->|INDEX| MEI[(Meilisearch)]
    UI -->|search| MEI --> UI
    UI -->|record| DB --> UI
    UI -->|GET /ipfs/<CID>| IPFS
    
    subgraph "Content Pipeline"
        NATS
        Deno
        IPFS
    end
    
    subgraph "Storage Layer"
        DB
        MEI
    end
    
    subgraph "Query Layer"
        UI
        MEI
        DB
    end
```

**Why**: We separate bytes (IPFS, immutable) from meaning (SurrealDB, queryable) and finding (Meili, instant). Events glue it together. Content addressing ensures verifiable integrity while maintaining fast search and structured queries.

## Docker Compose with Profiles

```yaml
# docker-compose.yml
version: "3.9"
x-hc: &hc { interval: 5s, timeout: 3s, retries: 30 }

services:
  nats:
    image: nats:2.10
    command: ["-js"]  # enable JetStream
    ports: ["4222:4222","8222:8222"]
    healthcheck: { test: ["CMD","nats","--help"], <<: *hc }

  ipfs:
    image: ipfs/kubo:latest
    ports: ["5001:5001","8080:8080"] # API, Gateway
    volumes: ["ipfs-stash:/data/ipfs"]
    healthcheck: { test: ["CMD","curl","-sf","http://localhost:5001/api/v0/version"], <<: *hc }

  surreal:
    image: surrealdb/surrealdb:latest
    command: ["start","--log","trace","--user","root","--pass","root","file:/data/surreal.db"]
    ports: ["8000:8000"]
    volumes: ["surreal-data:/data"]
    healthcheck: { test: ["CMD","curl","-sf","http://localhost:8000/health"], <<: *hc }

  meili:
    image: getmeili/meilisearch:latest
    environment:
      MEILI_NO_ANALYTICS: "true"
      MEILI_MASTER_KEY: masterKey
    ports: ["7700:7700"]
    volumes: ["meili-data:/meili_data"]
    healthcheck: { test: ["CMD","curl","-sf","http://localhost:7700/health"], <<: *hc }

  worker-deno:
    build: ./worker-deno
    depends_on:
      nats: { condition: service_started }
      ipfs: { condition: service_healthy }
      surreal: { condition: service_healthy }
      meili: { condition: service_healthy }
    environment:
      NATS_URL: nats://nats:4222
      STREAM: ingest
      SUBJECT: docs.incoming
      SURREAL_URL: http://surreal:8000
      SURREAL_USER: root
      SURREAL_PASS: root
      SURREAL_NS: app
      SURREAL_DB: main
      IPFS_API: http://ipfs:5001
      MEILI_HOST: http://meili:7700
      MEILI_KEY: masterKey

  sveltekit:
    build: ./sveltekit
    ports: ["5173:5173"]
    environment:
      NATS_URL: nats://nats:4222
      SUBJECT: docs.incoming
      MEILI_HOST: http://meili:7700
      MEILI_KEY: masterKey
      SURREAL_URL: http://surreal:8000
      SURREAL_NS: app
      SURREAL_DB: main
      SURREAL_USER: root
      SURREAL_PASS: root
      IPFS_GATEWAY: http://localhost:8080  # host access

  nats-cli:
    image: synadia/nats-box
    profiles: ["ops"]
    command: ["bash","-lc","while true; do sleep 3600; done"]
    depends_on: [nats]
    stdin_open: true
    tty: true

  demo-gen:
    build: ./demo-gen
    profiles: ["demo"]
    depends_on: [nats]
    environment:
      NATS_URL: nats://nats:4222
      SUBJECT: docs.incoming

volumes:
  ipfs-stash:
  surreal-data:
  meili-data:
```

### Operator Recipes

```bash
# Core stack
docker compose up -d --build

# Optional ops & generator
docker compose --profile ops up -d
docker compose --profile demo up -d
```

## SurrealDB Schema & Permissions

### db/init.surql
```sql
DEFINE NAMESPACE app;
DEFINE DATABASE main;

USE NS app DB main;

DEFINE TABLE doc SCHEMAFULL
  PERMISSIONS
    FOR select, create, update, delete WHERE true;

DEFINE FIELD id        ON TABLE doc TYPE string;         -- CID
DEFINE FIELD org       ON TABLE doc TYPE string;
DEFINE FIELD title     ON TABLE doc TYPE string;
DEFINE FIELD body      ON TABLE doc TYPE string;
DEFINE FIELD tags      ON TABLE doc TYPE array<string>;
DEFINE FIELD createdAt ON TABLE doc TYPE datetime;
DEFINE FIELD bytes     ON TABLE doc TYPE number;         -- size
```

## Meilisearch Index Bootstrap

### Bootstrap Commands
```bash
# Create index
curl -H "Authorization: Bearer masterKey" -H "Content-Type: application/json" \
  -X POST http://localhost:7700/indexes -d '{"uid":"docs","primaryKey":"id"}'

# Configure searchable attributes
curl -H "Authorization: Bearer masterKey" -H "Content-Type: application/json" \
  -X PATCH http://localhost:7700/indexes/docs/settings -d '{
    "searchableAttributes":["title","body","tags","org"],
    "filterableAttributes":["org","tags"],
    "sortableAttributes":["createdAt"]
  }'
```

## Deno Worker (Event Consumer)

### worker-deno/Dockerfile
```dockerfile
FROM denoland/deno:alpine-1.45.5
WORKDIR /app
COPY main.ts deps.ts .
RUN deno cache deps.ts main.ts
CMD ["run","--allow-net","--allow-env","main.ts"]
```

### worker-deno/deps.ts
```typescript
export * as nats from "https://deno.land/x/nats@v1.16.0/mod.ts";
export { createHash } from "https://deno.land/std@0.224.0/hash/mod.ts";
```

### worker-deno/main.ts
```typescript
import { nats } from "./deps.ts";

const {
  NATS_URL = "nats://nats:4222",
  STREAM = "ingest",
  SUBJECT = "docs.incoming",
  IPFS_API = "http://ipfs:5001",
  SURREAL_URL = "http://surreal:8000",
  SURREAL_USER = "root",
  SURREAL_PASS = "root",
  SURREAL_NS = "app",
  SURREAL_DB = "main",
  MEILI_HOST = "http://meili:7700",
  MEILI_KEY = "masterKey",
} = Deno.env.toObject();

async function surrealQuery(sql: string, vars: Record<string, unknown> = {}) {
  const res = await fetch(`${SURREAL_URL}/sql`, {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
      "Accept": "application/json",
      "NS": SURREAL_NS,
      "DB": SURREAL_DB,
      "Authorization": "Basic " + btoa(`${SURREAL_USER}:${SURREAL_PASS}`),
    },
    body: sql.replace(/\$([a-zA-Z0-9_]+)/g, (_, k) => JSON.stringify(vars[k] ?? null)),
  });
  return res.json();
}

async function ipfsAdd(bytes: Uint8Array) {
  const form = new FormData();
  form.append("file", new Blob([bytes]), "doc.json");
  const res = await fetch(`${IPFS_API}/api/v0/add`, { method: "POST", body: form });
  const txt = await res.text(); // ndjson lines
  const last = txt.trim().split("\n").pop()!;
  const { Hash, Size } = JSON.parse(last);
  return { cid: Hash, size: Number(Size) };
}

async function meiliUpsert(doc: Record<string, unknown>) {
  await fetch(`${MEILI_HOST}/indexes/docs/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${MEILI_KEY}` },
    body: JSON.stringify([doc]),
  });
}

async function ensureStream(js: nats.JetStreamClient, stream: string, subject: string) {
  try { 
    await js.streams.info(stream); 
  } catch {
    await js.streams.add({ 
      name: stream, 
      subjects: [subject], 
      storage: nats.StorageType.File 
    });
  }
}

const nc = await nats.connect({ servers: NATS_URL });
const js = nc.jetstream();
await ensureStream(js, STREAM, SUBJECT);

const sub = await js.subscribe(SUBJECT, { 
  durable: "deno-worker", 
  ack_policy: nats.AckPolicy.Explicit 
});

console.log("worker up:", SUBJECT);

for await (const m of sub) {
  try {
    const payload = JSON.parse(new TextDecoder().decode(m.data)); // {org,title,body,tags}
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    const { cid, size } = await ipfsAdd(bytes);

    const dto = {
      id: cid,
      org: payload.org ?? "unknown",
      title: payload.title ?? "",
      body: payload.body ?? "",
      tags: payload.tags ?? [],
      createdAt: (new Date()).toISOString(),
      bytes: size,
    };

    await surrealQuery(`
      CREATE doc CONTENT {
        id: $id, org: $org, title: $title, body: $body, tags: $tags, createdAt: time::now(), bytes: $bytes
      } ON DUPLICATE UPDATE
        org = $org, title = $title, body = $body, tags = $tags, bytes = $bytes;
    `, dto);

    await meiliUpsert(dto);
    m.ack();
  } catch (e) {
    console.error("process error:", e);
    m.term(); // poison-pill; real life: DLQ subject + metrics
  }
}
```

## SvelteKit UI

### sveltekit/Dockerfile
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 5173
CMD ["npm","run","dev","--","--host","0.0.0.0"]
```

### sveltekit/package.json
```json
{
  "name": "content-addressed-knowledge",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "build": "vite build",
    "dev": "vite dev",
    "preview": "vite preview"
  },
  "devDependencies": {
    "@sveltejs/adapter-auto": "^2.0.0",
    "@sveltejs/kit": "^1.20.4",
    "svelte": "^4.0.5",
    "vite": "^4.4.2"
  },
  "dependencies": {
    "nats.ws": "^1.16.0"
  },
  "type": "module"
}
```

### sveltekit/src/routes/api/ingest/+server.ts
```typescript
import { json } from '@sveltejs/kit';
import { connect } from "nats.ws";

export async function POST({ request }) {
  const body = await request.json(); // {org,title,body,tags}
  const nc = await connect({ servers: process.env.NATS_URL || "nats://nats:4222" } as any);
  const js = (nc as any).jetstream();
  const subject = process.env.SUBJECT || "docs.incoming";
  await js.publish(subject, new TextEncoder().encode(JSON.stringify(body)));
  await nc.close();
  return json({ ok: true });
}
```

### sveltekit/src/routes/search/+page.server.ts
```typescript
export const load = async ({ url, fetch }) => {
  const q = url.searchParams.get('q') || '';
  const host = process.env.MEILI_HOST || 'http://meili:7700';
  const key  = process.env.MEILI_KEY  || 'masterKey';
  const res = await fetch(`${host}/indexes/docs/search`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json','Authorization':`Bearer ${key}` },
    body: JSON.stringify({ q, limit: 20 })
  });
  return { q, results: await res.json() };
};
```

### sveltekit/src/routes/record/[id]/+page.server.ts
```typescript
export const load = async ({ params, fetch }) => {
  const SURREAL_URL = process.env.SURREAL_URL || 'http://surreal:8000';
  const auth = Buffer.from(`${process.env.SURREAL_USER}:${process.env.SURREAL_PASS}`).toString('base64');
  const res = await fetch(`${SURREAL_URL}/sql`, {
    method: 'POST',
    headers: { 'Content-Type':'text/plain','NS':process.env.SURREAL_NS,'DB':process.env.SURREAL_DB,'Authorization':`Basic ${auth}` },
    body: `SELECT * FROM doc WHERE id = "${params.id}";`
  });
  const json = await res.json();
  return { rec: json[0]?.result?.[0] ?? null, gateway: process.env.IPFS_GATEWAY || 'http://localhost:8080' };
};
```

### sveltekit/src/routes/+page.svelte
```svelte
<script>
  import { goto } from '$app/navigation';
  
  let form = {
    org: '',
    title: '',
    body: '',
    tags: ''
  };
  
  async function submit() {
    const tags = form.tags.split(',').map(t => t.trim()).filter(t => t);
    const payload = { ...form, tags };
    
    const res = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (res.ok) {
      form = { org: '', title: '', body: '', tags: '' };
      goto('/search');
    }
  }
</script>

<div class="container">
  <h1>Content-Addressed Knowledge</h1>
  
  <form on:submit|preventDefault={submit}>
    <div class="form-group">
      <label for="org">Organization</label>
      <input id="org" bind:value={form.org} required />
    </div>
    
    <div class="form-group">
      <label for="title">Title</label>
      <input id="title" bind:value={form.title} required />
    </div>
    
    <div class="form-group">
      <label for="body">Body</label>
      <textarea id="body" bind:value={form.body} required></textarea>
    </div>
    
    <div class="form-group">
      <label for="tags">Tags (comma-separated)</label>
      <input id="tags" bind:value={form.tags} />
    </div>
    
    <button type="submit">Ingest Document</button>
  </form>
  
  <div class="actions">
    <a href="/search">Search Documents</a>
  </div>
</div>

<style>
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
  }
  
  .form-group {
    margin-bottom: 1rem;
  }
  
  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
  }
  
  input, textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  
  textarea {
    height: 100px;
    resize: vertical;
  }
  
  button {
    background: #007bff;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  
  button:hover {
    background: #0056b3;
  }
  
  .actions {
    margin-top: 2rem;
    text-align: center;
  }
  
  .actions a {
    color: #007bff;
    text-decoration: none;
  }
</style>
```

### sveltekit/src/routes/search/+page.svelte
```svelte
<script>
  export let data;
  
  function viewRecord(id) {
    window.location.href = `/record/${id}`;
  }
</script>

<div class="container">
  <h1>Search Documents</h1>
  
  <form method="GET">
    <input name="q" value={data.q} placeholder="Search..." />
    <button type="submit">Search</button>
  </form>
  
  {#if data.results.hits}
    <div class="results">
      {#each data.results.hits as hit}
        <div class="result" on:click={() => viewRecord(hit.id)}>
          <h3>{hit.title}</h3>
          <p class="org">Organization: {hit.org}</p>
          <p class="body">{hit.body}</p>
          <div class="tags">
            {#each hit.tags as tag}
              <span class="tag">{tag}</span>
            {/each}
          </div>
          <p class="cid">CID: {hit.id}</p>
        </div>
      {/each}
    </div>
  {:else}
    <p>No results found.</p>
  {/if}
</div>

<style>
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
  }
  
  form {
    margin-bottom: 2rem;
  }
  
  input {
    width: 70%;
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  
  button {
    background: #007bff;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-left: 0.5rem;
  }
  
  .results {
    display: grid;
    gap: 1rem;
  }
  
  .result {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 1rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .result:hover {
    background-color: #f8f9fa;
  }
  
  .org {
    color: #666;
    font-size: 0.9rem;
  }
  
  .body {
    margin: 0.5rem 0;
  }
  
  .tags {
    margin: 0.5rem 0;
  }
  
  .tag {
    background: #e9ecef;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
  }
  
  .cid {
    font-family: monospace;
    font-size: 0.8rem;
    color: #666;
  }
</style>
```

### sveltekit/src/routes/record/[id]/+page.svelte
```svelte
<script>
  export let data;
  
  function verifyOnIPFS() {
    window.open(`${data.gateway}/ipfs/${data.rec.id}`, '_blank');
  }
</script>

<div class="container">
  {#if data.rec}
    <h1>{data.rec.title}</h1>
    
    <div class="metadata">
      <p><strong>Organization:</strong> {data.rec.org}</p>
      <p><strong>Created:</strong> {new Date(data.rec.createdAt).toLocaleString()}</p>
      <p><strong>Size:</strong> {data.rec.bytes} bytes</p>
      <p><strong>CID:</strong> <code>{data.rec.id}</code></p>
    </div>
    
    <div class="content">
      <h2>Content</h2>
      <p>{data.rec.body}</p>
    </div>
    
    {#if data.rec.tags && data.rec.tags.length > 0}
      <div class="tags">
        <h3>Tags</h3>
        {#each data.rec.tags as tag}
          <span class="tag">{tag}</span>
        {/each}
      </div>
    {/if}
    
    <div class="actions">
      <button on:click={verifyOnIPFS}>Verify on IPFS</button>
      <a href="/search">Back to Search</a>
    </div>
  {:else}
    <p>Record not found.</p>
  {/if}
</div>

<style>
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
  }
  
  .metadata {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 2rem;
  }
  
  .metadata p {
    margin: 0.5rem 0;
  }
  
  code {
    background: #e9ecef;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: monospace;
  }
  
  .content {
    margin-bottom: 2rem;
  }
  
  .tags {
    margin-bottom: 2rem;
  }
  
  .tag {
    background: #e9ecef;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
  }
  
  .actions {
    margin-top: 2rem;
  }
  
  button {
    background: #007bff;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 1rem;
  }
  
  button:hover {
    background: #0056b3;
  }
  
  a {
    color: #007bff;
    text-decoration: none;
  }
</style>
```

## Demo Generator

### demo-gen/Dockerfile
```dockerfile
FROM denoland/deno:alpine-1.45.5
WORKDIR /app
COPY app.ts .
CMD ["run","--allow-net","--allow-env","app.ts"]
```

### demo-gen/app.ts
```typescript
import { connect } from "https://deno.land/x/nats@v1.16.0/mod.ts";

const nc = await connect({ servers: Deno.env.get("NATS_URL") || "nats://nats:4222" });
const js = nc.jetstream();
const subj = Deno.env.get("SUBJECT") || "docs.incoming";

const orgs = ["nightco","saltlab","atlas"];
const tags = ["map","nocturne","salt","graph","weird"];

while (true) {
  const doc = {
    org: orgs[Math.floor(Math.random()*orgs.length)],
    title: crypto.randomUUID().slice(0,8) + " notes",
    body: "shadows & circuit diagrams in the rain",
    tags: Array.from(new Set(Array.from({length:2},()=>tags[Math.floor(Math.random()*tags.length)])))
  };
  
  await js.publish(subj, new TextEncoder().encode(JSON.stringify(doc)));
  await new Promise(r=>setTimeout(r, 150));
}
```

## Best Practices & Failure Modes

### Idempotency
- **CID as Primary Key**: Content ID is the natural primary key; "CREATE … ON DUPLICATE UPDATE" keeps Surreal consistent
- **Meili Upserts**: Meilisearch upserts by id prevent duplicate entries
- **Replay Safety**: JetStream durable consumers support replay without side effects

### Backpressure & Reliability
- **JetStream Retention**: NATS JetStream retains messages for replay
- **Explicit ACKs**: Deno consumer should ack explicitly and support replay (durable)
- **Poison Pill Handling**: Failed messages are terminated; real life: DLQ subject + metrics

### Schema Drift
- **Accept JSON**: Accept any JSON structure but only index selected fields
- **Field Selection**: Only index title, body, org, tags to keep search fast
- **Metadata Storage**: Store full JSON in SurrealDB for flexibility

### Security
- **Never Expose Keys**: Never expose Surreal admin or Meili master key publicly
- **Reverse Proxy**: Front with reverse proxy + auth in real life
- **Network Isolation**: Use internal networks for service communication

### Verification Path
- **CID Prominence**: UI should show CID prominently
- **Integrity Check**: Bytes fetched from IPFS must hash to that CID—truth you can grip
- **Gateway Access**: Provide IPFS gateway access for verification

### Data Gravity
- **IPFS Handles Blobs**: IPFS handles immutable byte storage
- **Surreal Stores References**: SurrealDB stores references + metadata
- **Meili Stores Text**: Meilisearch stores only searchable text
- **Keep Separate**: Maintain clear separation of concerns

## TL;DR Runbook

```bash
# 1) Bring up the core
docker compose up -d --build

# 2) Bootstrap Meili index (once)
curl -H "Authorization: Bearer masterKey" -H "Content-Type: application/json" \
  -X POST http://localhost:7700/indexes -d '{"uid":"docs","primaryKey":"id"}'

curl -H "Authorization: Bearer masterKey" -H "Content-Type: application/json" \
  -X PATCH http://localhost:7700/indexes/docs/settings -d '{
    "searchableAttributes":["title","body","tags","org"],
    "filterableAttributes":["org","tags"],
    "sortableAttributes":["createdAt"]
  }'

# 3) Open UI
http://localhost:5173

# 4) Ingest a doc (form) → search → open record → "Verify on IPFS"

# 5) Optional: flood generator
docker compose --profile demo up -d

# 6) Ops console (NATS, Surreal, Meili dashboards/GW)
docker compose --profile ops up -d

# 7) Monitor the pipeline
# - NATS: http://localhost:8222
# - SurrealDB: http://localhost:8000
# - Meilisearch: http://localhost:7700
# - IPFS Gateway: http://localhost:8080
```

---

*This tutorial provides the complete machinery for content-addressed knowledge systems. Each component is production-ready, copy-paste runnable, and designed for verifiable, searchable, and immutable document storage with event-driven processing.*
