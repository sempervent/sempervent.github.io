# Vibe Coding → Heavy Prompts → Agentic Execution (2025 Edition)

**Objective**: Master the art of transforming creative "vibe coding" into structured, executable agentic workflows. Start with mood and intent; end with automated delivery. You are distilling chaos into contracts, and contracts into machines.

## The Pipeline (Mental Model)

```mermaid
flowchart LR
  A[Vibe Brief<br/>(tone, mood, exemplars)] --> B[Heavy Prompt<br/>(spec, constraints, tests)]
  B --> C[Agent Plan<br/>(tasks, tools, graph)]
  C --> D[Execution<br/>(function calls, repos, data)]
  D --> E[Evaluation<br/>(rubrics, tests, traces)]
  E -->|fail| B
  E -->|pass| F[Artifacts<br/>(code, docs, PRs, dashboards)]
```

**Why**: Each phase narrows ambiguity. Vibes inspire; heavy prompts bind; agents obey.

## Vibe Coding — Patterns & Prompts

**Purpose**: Explore tone, UX, visual metaphors, naming, info scent. Fast, messy, generative.

### Best Practices

- **Constrain by reference vibes**: aesthetic adjectives, comparable products, code styles, domain voice
- **Ask for shortlists**: names, metaphors, UX sketches and micro-examples over long essays
- **Always pin negative constraints**: "no animations", "no network calls", "no external deps"

### Vibe Prompt Template (Product/Feature)

```
You are a creative director + staff engineer.
Goal: sketch the feel of a {feature/app}.
Aesthetic: brutalist, nocturnal, functional.
Constraints: no external services; CLI-first; ruthless ergonomics.
Deliver:
1) 5 name options (one-line rationale each).
2) 3 usage narratives (2–3 steps each) in present tense.
3) 2 micro UI sketches (ASCII) showing flow.
4) 5 "don't do this" anti-vibes.
Length caps: names ≤6 words, narratives ≤50 words each.
```

### Vibe Prompt Template (Code Style)

```
Channel a senior {language} engineer.
Produce: 3 tiny code idioms with comments:
- how we name things
- how we handle errors
- how we log
Constraints: functional bias; no globals; no magic.
Return as a Markdown snippet only.
```

## Densify into Heavy Prompts (Spec Contracts)

**Purpose**: Convert moodboards into unambiguous contracts: inputs/outputs, schema, acceptance tests, constraints, risks.

### Best Practices

- **Demand structured output**: JSON with explicit schema
- **Embed acceptance criteria**: edge cases, out-of-scope
- **Include test vectors & self-checks**: the agent will later execute
- **Cap verbosity**: ban hedging; require "Unknown" when uncertain

### Heavy Prompt Template (Spec + Tests)

```
You are the specifier. Transform the following vibe brief into a delivery contract.

Inputs:
- user_story: <text>
- constraints: <list>
- existing_assets: <repo paths/links>

Return JSON only matching this schema:
{
  "tasks": [{"id": "t-#", "desc": "...", "done_when": ["..."]}],
  "apis": [{"name":"...","method":"GET|POST","path":"...","req_schema":{...},"resp_schema":{...}}],
  "data_models": [{"name":"...","schema":{...}}],
  "acceptance_tests": [{"id":"a-#","given":"...","when":"...","then":"..."}],
  "risks": [{"risk":"...","mitigation":"..."}],
  "non_goals": ["..."]
}

Rules:
- No prose, JSON only.
- Prefer additive changes; mark breaking changes.
- Include at least 5 acceptance tests with edge cases.
```

## Agentic Execution — Plans, Tools, Graphs

**Purpose**: Turn the heavy spec into an execution graph with tools and checkpoints.

### State-of-Practice Patterns

- **ReAct**: tool-augmented reasoning (think → act → observe)
- **Plan-Execute split**: first produce a dependency-ordered task list; then execute each with tool calls
- **Self-reflection/critique loops**: "Reflexion"-style with short rubrics
- **Graph orchestration**: state machine/graph for branching flows and retries
- **Function calling/Tools**: code run, file read/write, git ops, web fetch, DB queries, eval runners, vector search/RAG
- **Structured outputs**: via JSON Schema/dataclasses to keep the agent inside the rails

### Agent Planning Prompt (Graph Build)

```
Role: Planner
Input: <heavy_spec_json>
Goal: Emit an execution plan as JSON (no prose):
{
  "nodes":[{"id":"t-1","tool":"git_patch","inputs":{"files":[...]}}],
  "edges":[{"from":"t-1","to":"t-2","on":"success"}],
  "checkpoints":[{"id":"c-1","after":"t-2","run":"acceptance_suite"}]
}
Constraints:
- Only tools: ["git_patch","python_run","shell","db_query","acceptance_suite"]
- Each node has measurable outputs.
- Insert retries: max 2 per node; exponential backoff.
```

### Agent Critic Prompt (Short Rubric)

```
Role: Critic
Assess output of <node_id> against <done_when> and <acceptance_tests>.
Return JSON:
{"pass": true|false, "failures": ["..."], "minimal_patch_hint": "..."}
Rule: keep hints ≤ 200 chars; no rewrites; no essays.
```

## Structured Output & Guardrails

### Best Practices

- **Enforce JSON-only**: with a JSON Schema and reject/retry on parse failure
- **Keep schemas small and composable**: big schemas invite drift
- **Use content filters/PII redaction**: before storage/logging
- **Store traces**: prompt, tool calls, diffs, metrics for audits

### Schema Nudge Prompt

```
When uncertain, emit "Unknown" or an empty list; do not invent.
If a field cannot be derived, omit it (do not use null).
If you exceed limits, truncate and set "truncated": true.
```

## Evaluation: Rubrics, Tests, Golden Files

### State-of-Practice

- **Derive acceptance tests**: from the heavy spec; run them after each critical node
- **Maintain golden artifacts**: snapshots for deterministic compare
- **Use multi-grader strategy**: fast rubric (cheap), deeper rubric (expensive) only on failures
- **Log latency, tokens, cache hits**: failure reasons

### Evaluator Prompt (Content)

```
Role: Evaluator
Given: <artifact>, <acceptance_tests>
Return JSON: {"score": 0..1, "failed_tests": ["a-2","a-5"], "notes": ["..."]}
No style comments; only spec compliance.
```

## Cost, Latency, Caching

- **Plan once, reuse often**: cache the heavy spec & plan by content hash
- **Embed retrieval (RAG)**: to shrink prompts; reference documents by ID
- **Shallow-then-deep**: cheap model to draft, strong model to refine
- **Chunk long tasks**: into nodes; parallelize independent branches
- **Artifact cache**: hash input → skip identical runs

## Safety, Compliance, & Data Hygiene

- **Redact PII/secrets**: pre-prompt; verify no secrets in artifacts
- **Tag outputs with provenance**: model, seed, tools, repo SHA
- **For code changes**: require patch + tests + doc delta before merge
- **Keep human-in-the-loop**: for critical merges and migrations

## Failure Recovery & Observability

- **Use retry with decay**: switch to fallback tool/model on repeated failure
- **Surface explainable errors**: from tools to the agent
- **Emit structured traces**: spans for each node and tool call
- **Keep a kill-switch**: for runaway agents (budget, step, and tool limits)

## Distinct Example Prompts (Ready to Paste)

### A) Vibe → CLI Tool Feel

```
Vibe brief: ruthless CLI for geospatial tiling; grim, minimal, fast.
Deliver:
- 6 command name ideas + 1-line tone each
- 3 sample command invocations (realistic)
- 5 anti-vibes
Keep it tight.
```

### B) Heavy Spec for the Same CLI

```
Transform the vibe into a spec. JSON only:
{
  "commands":[{"name":"tile","flags":[{"--src":"path|s3"},{"--z":"int"}]}],
  "contracts":[{"cmd":"tile","done_when":["outputs PMTiles","≤2GB memory"]}],
  "tests":[{"id":"a1","cmd":"tile --src s3://x --z 10","expect":["200 OK","PMTiles created"]}],
  "risks":[{"risk":"S3 throttling","mitigation":"backoff"}]
}
Rules: no prose, explicit types, ≥5 tests.
```

### C) Agent Plan (Tools)

```
Plan the execution graph for the spec above. Tools:
- git_patch(files:[],diff:string)
- shell(cmd:string,timeout_s:int)
- acceptance_suite(tests:[])
JSON only with nodes/edges/checkpoints. Include retries.
```

### D) Code Patcher (Small, Surgical)

```
Role: Senior Maintainer
Input: failing test <a3>, current code <path/to/file>, diff context.
Output: a unified diff patch only. Keep changes minimal; no refactors.
Ensure tests a1..a5 pass after patch.
```

### E) Self-Critique (Hard Gate)

```
Role: Gatekeeper
Given artifact + acceptance_tests, decide pass/fail.
If fail: list exactly which criteria failed; suggest ≤3-line patch hint.
JSON only: {"pass":true|false,"failures":["..."],"hint":"..."}
```

### F) Doc Delta Generator

```
Given spec + latest code diff, update docs.
Return Markdown only for the changed sections (≤200 lines).
No introductions. Preserve headings.
```

### G) Dynamic Router

```
Role: Router
Given a task, select a tool and minimal prompt for it.
Return: {"tool":"git_patch|shell|db_query|doc_update","prompt":"...","confidence":0..1}
No explanations.
```

### H) Safety Filter

```
You must redact secrets (AWS keys, tokens).
Replace with "***REDACTED***".
Return original structure with redactions. If none found, return unchanged.
```

## Anti-Patterns (Graveyard)

- **Vibes straight to code**: skipping the spec turns into scope creep
- **Unstructured outputs**: prose blobs rot pipelines
- **Monolithic prompts**: one mega-prompt does everything poorly
- **No evals**: if you don't measure, the agent hallucinates success
- **Infinite retries**: limit steps, tokens, and tools

## TL;DR Runbook

- Start with vibe briefs to shape direction
- Densify into heavy, JSON-only specs with tests
- Build an agent plan (graph, tools, retries)
- Execute with structured outputs + guardrails
- Evaluate with rubrics/golden tests; loop until green
- Cache specs/plans/artifacts by content hash
- Log everything; redact secrets; keep a kill-switch

---

*This guide provides the complete machinery for transforming creative vibes into structured, executable agentic workflows. Each pattern includes concrete prompts, schemas, and real-world implementation strategies for enterprise deployment.*
