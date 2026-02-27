# ADR-0002: Use MkDocs + Material Theme for Documentation Site

**Status**: Accepted

**Date**: 2024-01-15

**Deciders**: Joshua N. Grant

**Tags**: documentation, tooling, site

---

## Context

This site needed a documentation framework that could handle several hundred markdown files, support a rich nav hierarchy, provide fast full-text search, and deploy cleanly to GitHub Pages with zero server infrastructure.

The author writes documentation in Markdown and wanted to keep authoring in Markdown — not learn a template DSL or manage a database. The site should build in CI, not require a local build step for every page edit, and look professional without heavy front-end engineering.

## Decision

Use **MkDocs** (the static site generator) with the **Material for MkDocs** theme.

## Options Considered

### Option 1: MkDocs + Material (chosen)

**Pros:**
- Pure Markdown authoring; no shortcodes or template syntax needed
- Material theme provides tabs, search, code copy, dark mode, admonitions out of the box
- GitHub Pages deployment via `gh-pages` branch is first-class and well-documented
- `navigation.tabs`, `navigation.path`, `search.suggest`, `toc.integrate` cover all UX needs
- Plugin ecosystem: `git-revision-date-localized`, `tags`, `section-index`, `redirects`
- Active development; Material 9.x is stable and widely used

**Cons:**
- Python dependency (minor; managed via `requirements.txt`)
- No JavaScript server-side rendering; purely static
- Large nav trees can produce long YAML in `mkdocs.yml`

### Option 2: Hugo

**Pros:**
- Faster builds at very large scale (thousands of pages)
- Single binary, no Python required

**Cons:**
- Go template syntax is complex for non-Go authors
- Fewer out-of-the-box documentation UX patterns
- Material-equivalent theme quality requires more setup work

### Option 3: Docusaurus (React)

**Pros:**
- First-class MDX support (Markdown + JSX)
- Strong ecosystem for product documentation

**Cons:**
- Node.js build pipeline; heavier dependency surface
- React component authoring expected for advanced layouts
- Overkill for a personal documentation site with no interactive components

## Rationale

MkDocs + Material is the most productive choice for a single-author, Markdown-first documentation site. Material provides the full UX feature set this site needs — tabs, search, dark mode, code copy, admonitions — with zero custom front-end work. Hugo is faster at extreme scale but adds template complexity that doesn't pay off here. Docusaurus is designed for product docs with interactive React components, which this site does not need.

## Consequences

### Positive
- Zero front-end engineering required to maintain the site
- Full-text search works out of the box with Material's lunr integration
- GitHub Actions CI builds and deploys in < 2 minutes

### Negative
- Nav hierarchy is expressed in `mkdocs.yml` YAML, which becomes verbose at large scale
- Material requires specific plugin versions; upgrades occasionally break config

### Neutral / Trade-offs
- Python version pinning matters for reproducible builds; `requirements.txt` must be maintained

## Related Documents

- [Creating MkDocs GitHub Site](../tutorials/quick-start/creating-mkdocs-github-site.md)
- [ADR and Technical Decision Governance](../best-practices/architecture-design/adr-decision-governance.md)
