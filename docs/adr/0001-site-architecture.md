# ADR-0001: Site Architecture — MkDocs + Material + Repo-as-Docs

**Status**: Accepted

**Date**: 2026-02-28

**Deciders**: Joshua N. Grant (site owner)

**Technical Story**: Build a fast, searchable, maintainable documentation site that supports both professional best practices and playful experimentation — without a CMS, a backend, or a build team.

---

## Context

This site is a personal technical portfolio and documentation hub. It has two distinct audiences with different needs:

1. **Engineers evaluating patterns** — want stable reference material, architectural rationale, and production-ready examples they can adapt. They browse by topic; they don't read linearly.
2. **Curious generalists** — stumble in via search or a link, want something interesting to read or build. They respond to personality and novelty.

The site needed to satisfy both without becoming a bloated CMS or a hand-coded React app. Constraints:

- Single author (no content team, no editorial workflow)
- Content lives in Git (version control, diff-ability, PR review for future collaborators)
- Must deploy to a free static host (GitHub Pages)
- Must be fast to build, cheap to maintain, and not require frontend engineering to update
- Must support full-text search without a search backend
- Must scale to hundreds of pages without navigation collapse

---

## Decision

**MkDocs** with the **Material for MkDocs** theme, deployed to **GitHub Pages** from this repository, with content organized under `docs/` as plain Markdown files.

### Content domains

Three top-level content domains with distinct contracts:

| Domain | Purpose | Structure contract |
|---|---|---|
| **Best Practices** | Stable reference: patterns, governance, architecture | Conceptual; no assumed "do this right now" task |
| **Tutorials** | Task-oriented: step-by-step, copy-paste runnable | Prereqs → Steps → Verify → Troubleshoot |
| **Just for Fun** | Experimental: creative, playful, technically rigorous | No format constraint; must be reproducible |

### Navigation philosophy

- Tabs for top-level domains (`navigation.tabs`)
- Section grouping within tabs (`navigation.sections`)
- Breadcrumbs on every page (`navigation.path`)
- Long subsections (PostgreSQL, Docker) use nested subgroups — not flat 25-item lists
- Every major section has an `index.md` landing page

### Discovery mechanisms

- **Tags**: small, controlled vocabulary (≤ 15 tags total); rendered on a `/tags` index page
- **Search**: lunr-based full-text with symbol-aware separator (`[\s\-\_\.]+`); `search.suggest` and `search.share` enabled
- **What's New**: manually curated `whats-new.md`; updated when new content lands; no automation
- **See Also**: admonition blocks at the bottom of tutorial and best-practice pages; 2–3 links each; contextually adjacent only

### Deployment

GitHub Actions builds the site on push to `main` via `mkdocs gh-deploy`. No server. No CDN to configure. GitHub Pages serves static HTML.

---

## Alternatives Considered

### Hugo

Fast at scale; single binary. Rejected: Go template syntax is a maintenance burden for a Markdown-first author. Material-equivalent theme quality requires significant front-end setup. No meaningful advantage at this site's scale.

### Docusaurus (React)

Strong MDX support; excellent for product docs with interactive components. Rejected: Node.js build pipeline; React authoring expected for advanced layouts; overkill for a site with no interactive components. Bundle size and build complexity add nothing here.

### Sphinx (reStructuredText)

Excellent for API reference documentation; first-class Python ecosystem. Rejected: reST markup is higher friction than Markdown for non-API content; theme ecosystem is weaker; search quality is inferior to Material's lunr integration.

### GitBook / Notion / Confluence

Managed SaaS tools with rich editors and collaboration features. Rejected: content is not Git-native; export lock-in; no control over URL structure; pricing; branding constraints. This site's content is code — it belongs in a code repository.

### Single giant README or GitHub Wiki

Minimal tooling; works for small projects. Rejected: no search, no nav structure, no code block copy buttons, no dark mode, no tagging, no per-section index pages. Collapses at scale.

---

## Consequences

### Positive

- Zero server infrastructure; zero hosting cost
- Content is fully version-controlled; blame, diff, PR review all work
- MkDocs builds in < 15 seconds even at 300+ pages
- Material provides search, dark mode, code copy, admonitions, and tabs out of the box — no custom front-end work
- URL structure is stable (file path = URL); easy to cross-link and share
- `navigation.path` breadcrumbs solve "where am I?" without custom JS

### Negative / Trade-offs

- Nav hierarchy lives in `mkdocs.yml` YAML; verbose at scale and error-prone to maintain manually
- No dynamic content (no comments, no user accounts, no live search suggestions beyond lunr)
- Material plugin upgrades occasionally break config; `requirements.txt` must be pinned and maintained
- "What's New" is manual — if the author doesn't update it, it goes stale
- Tags are manual frontmatter — no auto-tagging

### Follow-ups / Future Work

- [ ] Automate `whats-new.md` generation from `git log` via a pre-commit hook or CI step
- [ ] Add `mkdocs-redirects` entries when pages are moved (plugin already installed)
- [ ] Evaluate `mkdocs-awesome-pages-plugin` to reduce YAML verbosity in nav
- [ ] Add a `projects/` section with structured project cards
- [ ] Revisit tag vocabulary as content grows; target ≤ 20 tags total

---

## Operational Notes (guide future decisions)

**URL preservation**: Never move a page without adding a redirect entry in `mkdocs.yml` via the `redirects` plugin. The plugin is already installed (`mkdocs-redirects`). A broken link from an external site is permanent damage.

**Nav sanity**: Any subsection with more than 12 items should be split into named subgroups. The PostgreSQL section (26 items) is the canonical example — it was split into Core & Design, Performance & Operations, Advanced Features, and Integration & Deployment.

**Cross-linking policy**: Every tutorial gets a `!!! tip "See also"` admonition with 2–3 links to contextually adjacent pages. Best-practice pages link to at least one tutorial that implements the pattern. Links must be relative and verified at build time.

**Tag policy**: Tags are kept to a small, stable vocabulary. Before adding a new tag, check whether an existing tag covers it. Tags should reflect technology domains (`geospatial`, `postgresql`, `docker`), not content types or difficulty levels.

**Content domain assignment**: When a page is ambiguous between Best Practices and Tutorials, assign it to the section that matches the primary reader intent. A page that readers open to *understand* goes in Best Practices. A page that readers open to *do something right now* goes in Tutorials. Cross-link between them.
