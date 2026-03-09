---
tags:
  - deep-dive
  - architecture
  - developer-experience
  - systems-design
---

# When to Use a TUI, CLI, or WebApp: Workflow, Environment, and the Interface Decision

**Themes:** Systems Design · Developer Experience · Operations

---

## Opening Thesis

Teams choose the wrong interface modality all the time. A deployment tool that should be a scriptable CLI becomes a heavyweight web dashboard because "we need a UI." A data pipeline console that operators run over SSH gets rebuilt as a web app, importing auth, hosting, and browser compatibility for no gain. A personal task manager that would serve its power users best as a keyboard-driven TUI ships as a web app because that's what the stack knows. The cost of these choices is not merely aesthetic: it shows up in automation gaps, onboarding friction, deployment burden, and the daily cognitive load of the people who use the system.

Interface choice is a systems-design decision. It is shaped by who uses the system, how often they use it, whether their workflow is one-shot or stateful, whether they work alone or in a shared context, and what environment they are in — SSH session, laptop browser, air-gapped datacenter. This essay gives you a decision framework: when a CLI is the right answer, when a TUI is the right middle ground, and when a WebApp is justified despite its infrastructure and operational cost. The goal is not to flatten the question into "it depends." It is to make the dependencies explicit so you can choose with intention and avoid the failure modes that come from defaulting to the wrong surface.

---

## Definitions Without Wasting Your Time

**CLI (command-line interface):** A program invoked from a shell that accepts arguments, flags, and sometimes stdin, and produces stdout/stderr. It is scriptable, composable via pipes and process substitution, and has no persistent visual state between invocations. Examples: `kubectl`, `terraform`, `psql`, `jq`, `ffmpeg`.

**TUI (terminal user interface):** A full-screen or near–full-screen interactive application that runs inside a terminal (or terminal emulator). It uses the alternate screen buffer, captures key events, and often redraws the whole view on input. It is stateful within a single process but does not require a browser or display server. Examples: `vim`, `htop`, `k9s`, `lazygit`, `ncdu`.

**WebApp:** An application that runs in a browser (or browser-like runtime), typically backed by one or more services. It offers rich layout, pointing device interaction, and access from any device with a browser and network. It requires hosting, identity/auth, and browser compatibility. Examples: GitHub, Grafana, internal admin dashboards, customer-facing SaaS.

The boundary between them is not always sharp. A CLI can spawn a TUI (e.g. `git commit` opening an editor). A WebApp can embed a terminal. The important distinction is the primary interaction model and the deployment and operational context that follow from it.

---

## The Real Decision Is About Workflow Shape

The right interface follows the shape of the work.

**One-shot commands** — Run once, get a result, exit. Deploy this revision. Run this query. Export this report. These are CLI territory. Automation and scripting assume exactly this model: no interactive state, versionable commands, repeatable execution.

**Repetitive expert workflows** — The same operator, daily, performing a known sequence: check queue depths, inspect failed jobs, retry, clear alerts. If the sequence is always the same and can be encoded as commands or a small script, a CLI (or CLI plus thin wrapper) wins. If the operator needs to navigate state — lists, filters, drill-down — and the environment is terminal-native (SSH, local terminal), a TUI often fits better than a web app that requires URL, login, and browser.

**Exploratory workflows** — Browsing logs, searching a catalog, filtering a list, following links. Discovery-heavy work benefits from visible structure: menus, breadcrumbs, search bars. A TUI can provide this in the terminal; a WebApp provides it with richer layout and shared links. A CLI is a poor fit when the user does not know in advance what they are looking for.

**Stateful multi-step workflows** — Fill a form, confirm, move to the next step. Wizards, config flows, onboarding. TUIs can do this (wizard-style prompts, multi-screen flows); WebApps excel at it. CLIs can approximate it with interactive prompts, but the experience is brittle and hard to script.

**Collaborative workflows** — Multiple people viewing or editing the same logical object, real-time or near–real-time. Shared dashboards, collaborative editing, comment threads. This is WebApp territory. CLIs and TUIs are single-user by default; collaboration would require custom sync and presence, which web infrastructure already provides.

**Monitoring and operations** — Watch lists, live metrics, alert triage. A TUI is ideal for an operator at a terminal (e.g. `k9s`, `htop`). A WebApp is ideal when the audience is distributed, needs shared views, or requires RBAC and audit trails.

---

## When a CLI Is the Right Answer

A CLI is the right answer when the primary use case is automation, scripting, or one-off commands that must be repeatable and versionable.

**Automation and scripting:** CI/CD pipelines, cron jobs, data pipelines, and glue code invoke commands. They do not click buttons. A tool that exposes its capabilities only through a web UI is automation-hostile: every script becomes a browser automation or API reverse-engineering project. A well-designed CLI is directly scriptable. Flags and arguments are stable; output is parseable (e.g. JSON, tab-separated). The same command that a human runs manually is the command the pipeline runs.

**Composability:** CLIs compose via pipes, subshells, and process substitution. The output of one tool becomes the input of another. This is the Unix philosophy at work: small, focused programs that combine in ways the author did not anticipate. A TUI or WebApp does not compose in this way; it is a closed surface.

**Remote shell workflows:** Operators working over SSH often have only a terminal. Deploying a web app so they can "use a UI" means they must run a browser (local or tunneled), authenticate, and maintain the web stack. A CLI runs in the session they already have. A TUI runs there too and is appropriate when the workflow is interactive; a CLI is appropriate when the workflow is command-driven.

**Low overhead:** No server to run, no browser to load, no auth layer to maintain for the tool itself. The CLI is a binary or script. Updates are a new binary or a pull. This is valuable for internal tools, developer utilities, and operations runbooks.

**Versionable commands:** A runbook that says "run `deploy.sh prod v2.3.1`" is unambiguous. The same runbook in a WebApp world says "log in, go to Deployments, select prod, enter v2.3.1, click Deploy" — and the UI may have changed by the time someone follows it. Commands in scripts and docs stay stable across versions when the CLI maintains backward compatibility.

**Infrastructure and data engineering examples:** Terraform, kubectl, psql, dbt, Prefect CLI, Airflow CLI. These are CLIs because the primary consumer is automation and the expert operator; the interface is the command surface, not a dashboard.

---

## When a CLI Is the Wrong Answer

A CLI is the wrong answer when the workflow requires discovery, dense visual state, or low memorization burden for infrequent users.

**Poor discoverability:** If the user does not already know the command name and approximate flags, a CLI forces them to read docs or run `--help`. There is no "browse and click." For infrequent users or broad audiences, this is a high barrier. For expert users performing known tasks, it is acceptable.

**High memorization burden:** When the tool has dozens of subcommands and hundreds of flags, even experts rely on muscle memory and external cheatsheets. If the workflow is exploratory or the set of actions is large and changing, a visual hierarchy (menus, lists, search) reduces cognitive load. A TUI or WebApp can offer that; a CLI cannot.

**Bad fit for dense visual state:** If the user needs to see many items at once (a long list, a table, a graph) and navigate by selection or filter, a CLI that prints a blob of text is a poor fit. Pagination and filtering in a CLI are possible but clumsy compared to a TUI list or a WebApp table.

**Fragile UX when syntax dominates:** When "doing the right thing" requires remembering exact flag order, quoting rules, or easy-to-get-wrong options, the CLI becomes a source of errors. A form or wizard that constrains input and shows current state is safer for complex or dangerous operations.

---

## When a TUI Is the Right Answer

A TUI is the right answer when the user is terminal-native, the workflow is interactive and stateful, and a full WebApp would be overkill or operationally costly.

**Operator tools:** Runbooks, dashboards, log tailers, queue browsers. The user is an engineer or operator who is already in a terminal (often over SSH). They need lists, filters, and keyboard-driven navigation. A TUI delivers that without a browser, a URL, or a separate auth system. Examples: `k9s` for Kubernetes, `lazygit` for Git, `ncdu` for disk usage.

**Keyboard-centric workflows:** Power users who prefer keys over mouse. TUIs are keyboard-first by design. Navigation, selection, and actions are bound to keys. For someone who lives in the terminal, a TUI often feels faster than switching to a browser and logging in.

**Stateful interaction without browser overhead:** The user needs to see a list, move selection, open detail, maybe run an action — but the environment is the terminal. A TUI provides stateful interaction (selection, focus, multiple panes) without the deployment and hosting cost of a WebApp. No need for TLS, cookies, or session management at the app layer.

**Working over SSH:** In air-gapped, locked-down, or remote-server contexts, the only available interface may be a terminal. A WebApp would require tunneling, a local browser, and often a separate auth story. A TUI runs in the existing SSH session.

**Moderate complexity with strong need for responsiveness:** When the tool has enough structure (lists, detail views, multiple modes) that a pure CLI would be awkward, but the user base is technical and terminal-bound, a TUI is the sweet spot. It is more expressive than a CLI and lighter than a WebApp.

For implementation guidance on building TUIs in Python, Go, and Rust, see the [TUI Applications best-practices (Python)](../best-practices/python/tui-applications.md), [Go](../best-practices/go/tui-applications.md), and [Rust](../best-practices/rust/tui-applications.md), and the corresponding [Python](../tutorials/python-development/building-a-python-tui.md), [Go](../tutorials/go-development/building-a-go-tui.md), and [Rust](../tutorials/rust-development/building-a-rust-tui.md) tutorials.

---

## When a TUI Is the Wrong Answer

A TUI is the wrong answer when collaboration, broad accessibility, or rich media matter more than terminal locality.

**Collaboration-heavy systems:** If multiple people need to see the same view, comment, or work on the same object, a TUI is single-user. Building real-time collaboration into a TUI is possible but not standard; WebApps and shared links are the norm.

**Public-facing products:** Customer-facing tools, demos, and anything used by non-technical users should not assume a terminal. TUIs assume familiarity with terminals, SSH, and keyboard-driven UIs. For a broad audience, a WebApp (or native GUI) is the right default.

**Rich media or broad accessibility:** If the workflow requires charts, images, or accessibility features that terminals do not handle well, a WebApp or GUI is appropriate. TUIs are text and keyboard; they are not the right surface for visual analytics or rich content.

**Terminal constraints as design shackles:** When the thing you need to show (e.g. a complex graph, a rich form) is constantly fighting the terminal's limitations, you are in the wrong modality. Either simplify the TUI to what the terminal handles well or move to a WebApp.

---

## When a WebApp Is the Right Answer

A WebApp is the right answer when accessibility from anywhere, discoverability, collaboration, or role-based access dominates.

**Accessibility from anywhere:** Users who are not on the same network as the server, or who use multiple devices, need a URL and a browser. No SSH, no VPN to the tool's host. WebApps are reachable from anywhere with network and identity.

**Discoverability:** Menus, breadcrumbs, search, and links make structure visible. New users can explore. Documentation can link to specific views. This is where WebApps excel and CLIs are weak.

**Collaboration:** Multiple users, shared state, comments, approvals. Web infrastructure (sessions, auth, real-time backends) supports this. RBAC and audit trails are standard concerns for web applications.

**Easier onboarding for broader audiences:** Non-experts can use a WebApp without learning shell syntax or key bindings. Forms and buttons constrain choices and reduce errors. For internal tools used by mixed audiences (e.g. support, product, ops), a WebApp often reduces training and support burden.

**Visual richness:** Dashboards, charts, drag-and-drop, WYSIWYG. When the value of the tool is in visualization or dense visual interaction, a browser is the right canvas.

---

## When a WebApp Is the Wrong Answer

A WebApp is the wrong answer when the primary users are experts in a terminal context and the added cost of hosting, auth, and browser brings no benefit.

**Over-engineering for internal tools:** A small team's internal deployment helper does not need a web front end, a login page, and a session store. If the only users are developers who run it from their machine or from CI, a CLI (or a TUI for interactive use) is simpler to build, deploy, and maintain.

**Infrastructure burden:** A WebApp implies a server (or serverless), TLS, identity (even if it's "internal SSO"), and browser compatibility. For a tool that could be a single binary or script, this is a large step up in operational surface.

**Poor fit for low-latency terminal-native operations:** An operator debugging a production incident is already in a terminal. Forcing them to open a browser, navigate to the right URL, and log in adds friction and latency. A CLI or TUI that runs in the same session is faster and more reliable in that context.

**Unnecessary abstraction for expert-only utilities:** If the tool is used only by a handful of experts who are comfortable with a CLI or TUI, building a WebApp "for consistency" or "because we have a web stack" wastes effort and adds maintenance without improving the workflow.

---

## Decision Criteria Matrix

The following matrix compares CLI, TUI, and WebApp across criteria that commonly drive the interface decision. Use it as a lens, not a scoring algorithm: the weights of these criteria depend on your context.

| Criterion | CLI | TUI | WebApp |
|-----------|-----|-----|--------|
| **Automation / scripting** | Strong: directly scriptable | Weak: not scriptable in the same way | Weak: requires API or browser automation |
| **Discoverability** | Weak: help and docs | Moderate: menus, keys, structure | Strong: links, menus, search |
| **Onboarding (novices)** | Weak | Moderate (if terminal-familiar) | Strong |
| **Stateful interaction** | Weak: stateless invocations | Strong: in-process state | Strong: server + client state |
| **Collaboration / multi-user** | Weak | Weak (single user) | Strong |
| **Deployment complexity** | Low: binary or script | Low: binary or script | High: server, auth, TLS |
| **Accessibility (any device, any network)** | Weak (needs shell access) | Weak (needs terminal) | Strong (URL + browser) |
| **Remote use (e.g. SSH)** | Strong | Strong | Weak without tunneling |
| **Offline use** | Strong (local binary) | Strong (local binary) | Weak (unless PWA/local) |
| **Observability (logs, metrics)** | Simple: stdout/stderr, process | Moderate: same as CLI | Complex: server-side + client |
| **Speed for experts (terminal-native)** | Strong | Strong | Weaker (context switch) |
| **Suitability for infrequent users** | Weak | Moderate | Strong |
| **Rich visual layout / media** | Weak | Weak (text/ASCII) | Strong |

No single modality wins on every axis. The decision is which criteria matter most for your users and environment.

---

## Environmental and Organizational Constraints

**Air-gapped or locked-down environments:** If the only way to run software is to ship a binary or script and execute it in a restricted network, a WebApp that assumes internet or internal web infrastructure may be infeasible. CLIs and TUIs that run locally fit this constraint.

**Server-admin workflows:** Operators who manage servers via SSH typically want tools that run in that same context. A CLI or TUI that they install once and run from any session is a better fit than a WebApp that requires them to open a browser and authenticate elsewhere.

**Regulated environments:** Compliance may require audit logs, RBAC, and approved software. WebApps centralize these (server-side auth, access logs). CLIs and TUIs can be audited via process and command history, but the story is different. Choose the modality that matches how your organization enforces policy.

**Enterprise auth overhead:** If every internal tool must plug into corporate SSO, the cost of adding a WebApp (auth integration, session handling) is non-trivial. A CLI or TUI may use the same identity (e.g. SSH keys, OS user) without a separate auth layer.

**Internal vs public users:** Internal tools can assume technical users and terminal access. Public or partner-facing tools usually cannot. The same product might expose a CLI for power users and a WebApp for everyone else — a hybrid.

**Maintenance staffing:** A WebApp has a larger surface: front-end, back-end, auth, hosting. A small team may not have the capacity to maintain it. A CLI or TUI can be maintained by one or two people who own the domain logic and the interface together.

---

## Hybrid Patterns That Actually Make Sense

**CLI + WebApp:** The CLI for automation and experts; the WebApp for discoverability, onboarding, and collaboration. Many platforms do this: GitHub (CLI + web), Terraform (CLI + Cloud UI), Datadog (API/CLI + dashboards). The core capabilities are exposed in a scriptable way; the web layer serves different users and use cases.

**CLI + TUI:** A single binary with a default CLI mode and a TUI mode (e.g. `tool run` vs `tool run --tui` or `tool interactive`). The same logic backs both; the operator chooses the interface by context. Good for deployment tools, database clients, and operations consoles.

**TUI + Web backend:** The TUI is the primary operator interface; it talks to the same APIs or services that a future (or existing) WebApp uses. The backend is shared; the front-end is chosen by context (terminal vs browser). This keeps logic in one place and avoids building two separate applications.

**CLI core with multiple frontends:** The application core is a library or service; the CLI, TUI, and WebApp are thin frontends. This is more architectural effort but pays off when you need all three or when you expect to add another frontend later. Not every tool needs this; reserve it for products where multiple interfaces are a requirement, not a guess.

Design around a shared core when you have evidence that more than one modality will be used. Otherwise, build the one interface that matches the primary workflow and add another only when demand is clear.

---

## Anti-Patterns and Failure Modes

**Turning a simple automation tool into a bad web product:** A script or CLI that "just works" in CI and in the shell gets replaced by a WebApp because "we need a UI." The WebApp is under-invested in (no one wants to own the front-end), automation becomes a second-class citizen (no official API, or a bolted-on API), and the team pays hosting and auth cost for no gain. The right move is often to keep the CLI and add a small TUI for interactive use, or to invest seriously in the WebApp and its API so that both humans and automation are first-class.

**Building a TUI where logs and scripts would suffice:** If the real need is "see the last N log lines" or "run these five commands in sequence," a TUI is unnecessary. A CLI that streams logs or a script that runs the sequence is simpler. TUIs are for interactive navigation and stateful choice; don't build one to wrap a single stream or a fixed workflow.

**Using a CLI for workflows that require visibility and navigation:** When users constantly ask "how do I do X?" or "what's the list of Y?", the workflow is discovery-heavy. A CLI forces everything into help text and docs. A TUI or WebApp that shows structure (lists, filters, drill-down) reduces cognitive load and support burden.

**Letting team aesthetics override user workflow:** Choosing a WebApp because the team likes React, or a CLI because "real engineers use the terminal," ignores who actually uses the tool and in what context. The right choice follows from user context, frequency of use, and environment — not from stack preference.

---

## Case Studies and Scenario Comparisons

**Infrastructure deployment tool:** Used by platform engineers and CI. Primary use: scripted deploys and occasional manual runs from a terminal. **Verdict:** CLI first. Automation is the main consumer; humans run the same commands. A WebApp can be added later for visibility (deployment history, status) and for less frequent users; the CLI remains the primary interface for execution.

**Data pipeline operator console:** Used by data engineers to inspect runs, retry failures, and browse dependencies. They work over SSH or from a laptop terminal. **Verdict:** TUI or WebApp. If the team is terminal-native and the console is internal, a TUI (e.g. a custom `k9s`-like view for pipeline runs) is a good fit. If the audience includes non-terminal users or needs shared views and RBAC, a WebApp is better. Avoid a CLI-only interface for this; the workflow is exploratory and stateful.

**Personal note or task manager:** Single user, possibly multiple devices. **Verdict:** Depends on the user. Terminal-native power users are well served by a TUI (or CLI + TUI) with local storage; sync can be file-based or a simple backend. Users who want mobile and browser access need a WebApp (or native mobile) and sync. Don't assume one size fits all; the same app might offer a TUI and a web view over the same data.

**Customer-facing analytics portal:** Used by external customers, multiple roles, need for sharing and access control. **Verdict:** WebApp. Collaboration, discoverability, and accessibility from anywhere are requirements. A CLI or TUI would be wrong for this audience.

**Cluster monitoring utility:** Used by SREs and platform engineers, often in an incident context. They are in a terminal or have a browser open. **Verdict:** Both TUI and WebApp can work. A TUI is ideal for "quick look in the terminal" (e.g. `k9s` for Kubernetes). A WebApp is ideal for shared dashboards, historical views, and alert management. Many teams have both: a TUI for fast terminal use, a WebApp for everything else.

**Secrets or configuration management tool:** Used by developers and automation. Automation needs a CLI or API; humans need to browse, search, and sometimes edit. **Verdict:** CLI (or API) for automation; WebApp or TUI for human browsing. A TUI is viable if the user base is technical and terminal-bound; a WebApp is better if the tool is used by a broad set of people or requires strict RBAC and audit trails in a UI.

---

## Practical Decision Framework

Ask these questions in order; they narrow the interface space.

1. **Who uses it, and how often?** Experts, daily → CLI or TUI. Mixed or infrequent users → WebApp or TUI with good discoverability.
2. **Is automation a primary consumer?** Yes → CLI (and/or API) is required. The WebApp or TUI can coexist but must not be the only interface.
3. **Where do users run it?** Only in a terminal (including SSH) → CLI or TUI. Anywhere, any device → WebApp (or multiple clients with a shared backend).
4. **Is the workflow exploratory or one-shot?** Exploratory (browse, filter, drill down) → TUI or WebApp. One-shot (run command, get result) → CLI.
5. **Do multiple people need to share views or collaborate?** Yes → WebApp. No → CLI or TUI is sufficient.
6. **What are the deployment and maintenance constraints?** Minimal ops → prefer CLI or TUI. Willing to run servers and auth → WebApp is on the table.
7. **Is there a strong reason to avoid a browser?** (e.g. air-gapped, terminal-only, no web stack.) Yes → CLI or TUI. No → WebApp is viable.

**Rubric:** If automation and experts dominate, start with a CLI; add a TUI for interactive use if the workflow is stateful and terminal-native. If discoverability, collaboration, or broad access dominate, build a WebApp; expose an API or CLI for automation if needed. If the primary context is an operator in a terminal with stateful, navigable workflows, a TUI may be the best single interface — and you can add a CLI for scripting and a WebApp later if the need appears.

---

## Conclusion

The best interface is the one that matches the operational truth of the work: who does it, where they are, how often, and whether the workflow is command-driven, exploratory, or collaborative. CLIs excel at automation and composability and are the wrong tool when discoverability and onboarding matter. TUIs excel at terminal-native, stateful, keyboard-driven workflows and are the wrong tool when collaboration or broad accessibility matter. WebApps excel at reach, discoverability, and shared state and are the wrong tool when the only users are experts in a terminal and the infrastructure cost buys nothing.

Choose with intention. Defaulting to a WebApp because "we need a UI" or to a CLI because "we're engineers" ignores the real decision: workflow shape, operator context, and institutional constraints. Get those right, and the right interface follows.

!!! tip "See also"
    - [TUI Applications (Python)](../best-practices/python/tui-applications.md), [Go](../best-practices/go/tui-applications.md), [Rust](../best-practices/rust/tui-applications.md) — production guidance for building TUIs
    - [Building a Python TUI](../tutorials/python-development/building-a-python-tui.md), [Go TUI](../tutorials/go-development/building-a-go-tui.md), [Rust TUI](../tutorials/rust-development/building-a-rust-tui.md) — step-by-step Task Runner TUI tutorials
    - [Cognitive Load Management and Developer Experience](../best-practices/architecture-design/cognitive-load-developer-experience.md) — reducing cognitive load in system design
