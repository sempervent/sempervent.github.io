# Portfolio & site reconnaissance (2026-09-30)

Evidence sources: this repository (`main`), live site `https://sempervent.github.io/`,
GitHub API (`sempervent/*` public repos), and HTTP checks against org GitHub Pages paths.

## Current site architecture (before restructure)

| Layer | Behavior |
| --- | --- |
| Stack | MkDocs 1.x + Material 9.x, GitHub Actions → GitHub Pages (`deploy-mkdocs.yml`) |
| Home (`docs/index.md`) | Hero + **Best Practices / Tutorials / Projects / Just for Fun** cards; deep “Start Here” and featured articles; stale **What I'm Working On** and **Latest Updates** |
| Nav (`mkdocs.yml`) | Many top-level tabs: Home, What's New, Tags, Profile, Projects, Technical Documentation, Doctrine, ADRs, Best Practices, Deep Dives, Tutorials, Contact |
| Projects | Single hand-maintained `docs/projects.md` — foregrounds BlackLake, ORNL OpenSAMPL, and legacy web apps |
| Freshness | Manual lists on homepage; `whats-new.md` is explicitly curated (ADR-0001) |
| Project metadata | Duplicated prose in `projects.md` / homepage; **no canonical registry** |
| Plugins installed | `git-revision-date-localized`, `tags`, `minify`; `mkdocs-redirects` / `macros` in `requirements.txt` but **not wired** in `mkdocs.yml` |

## Stale or misleading sections

- Homepage **Featured** projects: Final Fantasy Football, Where I've Been, This Is A Casino, DCRS — not reflected in recent `sempervent` GitHub activity; several lack repo links in `projects.md`.
- **What I'm Working On** — generic doc links, not tied to active repositories (PARQONAUT, `dots`, NUMBRANE).
- **About This Site** — “Everything here is battle-tested” / “production-ready configurations” overstates mixed content (experiments, tutorials, reference).
- **BlackLake** — `projects.md` links “Live Demo” to `s3-rust-data-portal` Pages only; separate `blacklake` Python repo also publishes docs at `/blacklake/`.
- **Footer** — presents ORNL affiliation without personal-site disclaimer.

## Verified live GitHub Pages (HTTP 200)

Org site base: `https://sempervent.github.io/`

| Path | Repo | Notes |
| --- | --- | --- |
| `/` | `sempervent.github.io` | Main portfolio / docs |
| `/PARQONAUT/` | `PARQONAUT` | Homepage URL set on repo |
| `/dots/` | `dots` | Dotfiles docs |
| `/music-rig/` | `music-rig` | Studio setup docs |
| `/blacklake/` | `blacklake` | Python Blacklake docs |
| `/s3-rust-data-portal/` | `s3-rust-data-portal` | Rust portal docs (README also branded Blacklake) |
| `/gi/` | `gi` | |
| `/wildfire-smoke-risk-correlator/` | `wildfire-smoke-risk-correlator` | |
| `/agent-llm-wiki-matrix/` | `agent-llm-wiki-matrix` | |
| `/postgres-query-autopsy-tool/` | `postgres-query-autopsy-tool` | |
| `/smart-farm-wiki/` | `smart-farm-wiki` | |
| `/llm-wiki-template/` | `llm-wiki-template` | |
| `/generative-midi-workbench/` | `generative-midi-workbench` | |

External: [OpenSAMPL docs](https://ornl.github.io/OpenSAMPL/) (`ORNL/OpenSAMPL`).

## Pages enabled but not serving (HTTP 404 on org URL)

GitHub Pages API reports a site; path returned 404 on 2026-09-30:

| Repo | Configured URL |
| --- | --- |
| `mqtt-comparison` | `https://sempervent.github.io/mqtt-comparison/` |
| `embers-of-the-earth` | `https://sempervent.github.io/embers-of-the-earth/` |
| `colony` | `https://sempervent.github.io/colony/` |
| `universe` | `https://sempervent.github.io/universe/` |

Omitted from the public “documentation sites” directory until deploys succeed.

## Orphaned / unlinked live Pages

Live sites above were **not** linked from the main portfolio prior to this work (except BlackLake via wrong repo URL). `PARQONAUT`, `dots`, wikis, and tooling docs were absent from `projects.md`.

## Recently active public repos (portfolio-relevant)

By `updatedAt` on 2026-09-30: `numbrane`, `dots`, `PARQONAUT`, `paraclete`, `music-rig`, `cosmic-architect`, `wildfire-smoke-risk-correlator`, agent/wiki tooling, `postgres-query-autopsy-tool`.

## Recommended hierarchy

1. **Person** — hero + professional scope (geospatial, data, distributed systems, tooling).
2. **Current work** — registry-driven cards: PARQONAUT, dots, NUMBRANE, Paraclete, Cosmic Architect.
3. **Proof** — featured engineering (Blacklake lineages, smoke correlator, autopsy tool).
4. **Technical writing** — Doctrine / Best Practices / Deep Dives / Tutorials (unchanged URLs, grouped under **Writing** tab).
5. **Lab** — creative, games, MIDI, Pi experiments.
6. **Deep docs** — existing taxonomy preserved.

## BlackLake lineage (evidence-based)

| Repo | Implementation | Docs URL | Last push (API) |
| --- | --- | --- | --- |
| `sempervent/blacklake` | Python; git-like S3 dataset store + semantic metadata | `/blacklake/` | 2025-10-09 |
| `sempervent/s3-rust-data-portal` | Rust/Axum ML artifact portal; README title “Blacklake” | `/s3-rust-data-portal/` | 2025-12-16 |

Both are public, both publish MkDocs sites under the org Pages namespace. They appear to be **parallel lineages** sharing a product name rather than a simple rename; this site lists them separately and avoids implying a single merged codebase.

## Deployment vs repository

- CI builds with `mkdocs build --clean` (no `--strict` on `main` deploy); PR preview uses `--strict`.
- No pre-build project generator existed before this change.

## Implementation follow-up (this branch)

- Canonical registry: `data/projects.yaml`
- Generator: `scripts/generate_portfolio.py`
- Validation: `scripts/validate_projects.py`
- Redirect: legacy `projects.md` → `projects/index.md`
