#!/usr/bin/env python3
"""Rebuild mkdocs nav: nest doctrine/docs under Writing tab."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MKDOCS = ROOT / "mkdocs.yml"


def main() -> None:
    original = subprocess.check_output(
        ["git", "show", "main:mkdocs.yml"], text=True, cwd=ROOT
    )
    lines = original.splitlines(keepends=True)

    nav_start = next(i for i, l in enumerate(lines) if l.startswith("nav:"))
    ext_start = next(i for i, l in enumerate(lines) if l.startswith("markdown_extensions:"))

    nav_lines = lines[nav_start:ext_start]
    idx = next(i for i, l in enumerate(nav_lines) if l.strip().startswith("- Doctrine:"))
    body = nav_lines[idx:]
    body = [l for l in body if "Contact & Collaboration" not in l]

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
    - Contact & Collaboration: getting-started.md

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
