#!/usr/bin/env python3
"""Rebuild mkdocs nav: nest doctrine/docs under Writing tab."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MKDOCS = ROOT / "mkdocs.yml"

# Nav sections removed from the public site (skip when merging from main).
_REMOVED_SECTION_HEADERS = frozenset(
    {
        "- Architecture Decisions:",
        "- Diagrams:",
    }
)
_SKIP_LINE_SUBSTRINGS = (
    "Contact & Collaboration",
    "adr/",
    "ADR-",
    "Diagram Style Guide",
    "adr-decision-governance",
)


def _filter_nav_body(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_depth: int | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in _REMOVED_SECTION_HEADERS:
            skip_depth = len(line) - len(line.lstrip())
            continue
        if skip_depth is not None:
            if stripped and (len(line) - len(line.lstrip())) <= skip_depth:
                skip_depth = None
            else:
                continue
        if any(s in line for s in _SKIP_LINE_SUBSTRINGS):
            continue
        filtered.append(line)
    return filtered


def main() -> None:
    original = subprocess.check_output(
        ["git", "show", "main:mkdocs.yml"], text=True, cwd=ROOT
    )
    lines = original.splitlines(keepends=True)

    nav_start = next(i for i, l in enumerate(lines) if l.startswith("nav:"))
    ext_start = next(i for i, l in enumerate(lines) if l.startswith("markdown_extensions:"))

    nav_lines = lines[nav_start:ext_start]
    idx = next(i for i, l in enumerate(nav_lines) if l.strip().startswith("- Doctrine:"))
    body = _filter_nav_body(nav_lines[idx:])

    indented: list[str] = []
    for line in body:
        if not line.strip():
            indented.append(line)
        else:
            indented.append("  " + line)

    header = """nav:
  - Home: index.md
  - Projects:
    - Portfolio: projects/index.md
    - Documentation sites: projects/documentation-sites.md
  - Writing:
    - Technical overview: documentation.md
    - What's New: whats-new.md
"""
    footer = """  - Lab:
    - Overview: lab/index.md
    - Just for Fun: tutorials/just-for-fun/index.md
  - About:
    - Professional Profile: about.md
    - Contact: getting-started.md

"""
    new_content = (
        "".join(lines[:nav_start])
        + header
        + "".join(indented)
        + footer
        + "".join(lines[ext_start:])
    )
    MKDOCS.write_text(new_content, encoding="utf-8")
    print("Rebuilt mkdocs.yml navigation")


if __name__ == "__main__":
    main()
