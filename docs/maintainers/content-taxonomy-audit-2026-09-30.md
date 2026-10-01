# Content taxonomy audit (2026-09-30)

Scope: Best Practices vs Tutorials vs Just for Fun under **Writing**, after the portfolio IA pass.

## Category contracts (unchanged)

| Category | Question it answers |
| --- | --- |
| **Best Practices** | What should practitioners understand about design, trade-offs, failure modes, and operations? |
| **Tutorials** | How do I build, configure, or run this specific thing? |
| **Just for Fun** | What happens when curiosity drives the stack — art, games, music, odd integrations? |

**Lab** top-level nav was removed: it duplicated **Projects** (creative repos) and **Just for Fun** (creative tutorials). Redirect: `lab/index.md` → Just for Fun index.

## Moves performed

| Current path | From | To | Reason | Redirect |
| --- | --- | --- | --- | --- |
| `tutorials/just-for-fun/js-glitch-observatory.md` | Tutorials → Python Development | Tutorials → Just for Fun | Browser entropy/art piece; not a Python dev guide | `tutorials/python-development/js-glitch-observatory.md` → new path |

## Reviewed; kept in place

| Path | Category | Notes |
| --- | --- | --- |
| `tutorials/just-for-fun/managing-people-software-dev.md` | Just for Fun (nav) | Leadership essay with tutorial shape; thematically odd but stable URL and useful cross-link target — leave unless a dedicated “Engineering leadership” section appears later |
| `best-practices/creative-fun/*` | Best Practices | Small operational patterns (Celery, LaTeX, time hygiene) — name is cute, content is pattern-oriented; no move |
| `tutorials/best-practices-integration/*` | (mostly unlisted in nav) | Long integration walkthroughs; remain under Tutorials tree for now — audit did not mass-move |
| `deep-dives/*` | Deep Dives (under Writing) | Essay-style; distinct from Best Practices indexes — no change |

## Writing / Projects / Just for Fun hierarchy (decision)

**Option A adopted:**

- **Projects** — repositories and docs sites (registry-driven).
- **Writing** — Best Practices, Tutorials (including **Just for Fun** nested under Tutorials).
- **About** — profile and contact.

No separate **Lab** tab.

## Follow-up (optional, low priority)

- Consider moving `managing-people-software-dev.md` under Best Practices → architecture/leadership if the section grows.
- Periodically spot-check new tutorials landing in `python-development/` for creative misfiles.
