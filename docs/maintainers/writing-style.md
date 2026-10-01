# Writing style (maintainers)

This site has three public contracts:

| Section | Job |
| --- | --- |
| **Best Practices** | Judgment: when to do something, when not to, what breaks, what it costs. |
| **Tutorials** | Build one concrete thing: prerequisites, commands, verification. |
| **Just for Fun** | Creative or odd technical work; tone can be looser. |

## Voice

- Start from the problem. Say what worked, what failed, and the trade-off.
- Prefer concrete nouns, commands, and failure modes over adjectives.
- Do not claim **production-ready**, **enterprise-grade**, or **comprehensive** unless you can point at running systems or cite a benchmark.
- Illustrative code must be labeled **illustrative** or **pseudocode** when it is not maintained and tested.

## Boilerplate to avoid

Do not open with `Objective: Master…` or close with “complete machinery/framework” language.

Avoid template sections that repeat on every page (`What This Guide Covers`, `Key Takeaways`, `Why This Matters`) unless the page truly needs them.

## Verification

Version-sensitive stacks (Kubernetes, Pi OS, PostgreSQL, Python packaging, NiceGUI, cloud APIs) must be checked against current docs before publish.

Remove or archive pages you would not trust at a terminal today.

## Deletion is valid

A generic page that restates official documentation badly is worse than no page.

## Lint

Run `python scripts/lint_writing.py` before large merges. It flags a small set of egregious template phrases in Best Practices and Tutorials (not Just for Fun).
