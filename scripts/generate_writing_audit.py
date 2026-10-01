#!/usr/bin/env python3
"""Generate docs/maintainers/writing-quality-audit-2026-10.md from corpus heuristics."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "maintainers" / "writing-quality-audit-2026-10.md"

SCAN_DIRS = [
    DOCS / "best-practices",
    DOCS / "tutorials",
]

FINGERPRINTS = [
    r"Objective:\s*Master",
    r"weapon of choice",
    r"complete machinery",
    r"complete framework",
    r"This guide provides",
    r"This tutorial provides",
    r"What This Guide Covers",
    r"Key Takeaways",
    r"Why This Matters",
    r"enterprise-grade",
    r"production-ready",
    r"transform your",
    r"gateway to",
    r"comprehensive suite",
    r"Integrated Best Practices Suite",
]

KEEP_HINTS = [
    "postgis",
    "esp32",
    "geoparquet",
    "parquet",
    "mermaid",
    "rke2",
    "nicegui",
    "spark/when-to-use",
    "failure-oriented",
    "geospatial-system-design",
    "just-for-fun",
]


def score_file(text: str, rel: str) -> tuple[int, list[str]]:
    hits: list[str] = []
    score = 0
    for pat in FINGERPRINTS:
        if re.search(pat, text, re.I):
            hits.append(pat)
            score += 2
    if re.match(r"^#\s+.+, .+, .+,", text, re.M):
        score += 3
    lines = len(text.splitlines())
    if lines > 1200:
        score += 4
    elif lines > 800:
        score += 2
    if "best-practices-integration/" in rel:
        score += 5
    for hint in KEEP_HINTS:
        if hint in rel.lower():
            score -= 2
    return score, hits


def disposition(score: int, rel: str, hits: list[str]) -> str:
    if "just-for-fun/" in rel and score < 8:
        return "KEEP"
    if any(h in rel for h in ("postgis-geometry-indexing", "rke2-raspberry-pi", "geospatial-system-design")):
        return "REWRITE" if score > 2 else "LIGHT EDIT"
    if "system-resilience-and-concurrency" in rel:
        return "REWRITE"
    if "best-practices-integration/" in rel:
        return "ARCHIVE/REMOVE"
    if score >= 10:
        return "ARCHIVE/REMOVE"
    if score >= 6:
        return "REWRITE"
    if score >= 3:
        return "LIGHT EDIT"
    if score <= 1 and len(hits) == 0:
        return "KEEP"
    return "LIGHT EDIT"


def reason(disp: str, hits: list[str], lines: int) -> str:
    parts: list[str] = []
    if hits:
        parts.append("fingerprints: " + ", ".join(hits[:3]))
    if lines > 1000:
        parts.append(f"{lines} lines (catalog risk)")
    if disp == "ARCHIVE/REMOVE" and "integration" in " ".join(hits).lower():
        parts.append("synthetic integration tutorial")
    if not parts:
        parts.append("low template signal")
    return "; ".join(parts)


def main() -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    counts: Counter[str] = Counter()

    for base in SCAN_DIRS:
        for path in sorted(base.rglob("*.md")):
            if path.name == "index.md" and path.parent == base:
                category = "index"
            elif "just-for-fun" in path.as_posix():
                category = "just-for-fun"
            elif "best-practices" in path.as_posix():
                category = "best-practices"
            else:
                category = "tutorials"
            rel = path.relative_to(DOCS).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            score, hits = score_file(text, rel)
            disp = disposition(score, rel, hits)
            counts[disp] += 1
            verify = "yes" if disp in ("REWRITE", "LIGHT EDIT", "KEEP") and any(
                k in rel for k in ("rke2", "nicegui", "postgis", "kubernetes", "postgres")
            ) else ("if retained" if disp != "ARCHIVE/REMOVE" else "n/a")
            rows.append(
                (
                    rel,
                    category,
                    disp,
                    reason(disp, hits, len(text.splitlines())),
                    verify,
                    "",
                )
            )

    lines_out = [
        "# Writing quality audit (2026-10)",
        "",
        f"Generated {date.today().isoformat()} by `scripts/generate_writing_audit.py`.",
        "Disposition is heuristic — review before bulk deletes.",
        "",
        "## Summary counts",
        "",
    ]
    for key in (
        "KEEP",
        "LIGHT EDIT",
        "REWRITE",
        "MERGE",
        "MOVE",
        "ARCHIVE/REMOVE",
    ):
        lines_out.append(f"- **{key}**: {counts.get(key, 0)}")
    lines_out.extend(
        [
            "",
            "## Pages",
            "",
            "| Path | Category | Disposition | Reason | Verify commands? | Overlap |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines_out.append("| " + " | ".join(row) + " |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} pages)")


if __name__ == "__main__":
    main()
