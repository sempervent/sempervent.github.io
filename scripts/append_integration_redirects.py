#!/usr/bin/env python3
"""Append redirect_maps entries for archived best-practices-integration tutorials."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs" / "archive" / "tutorials" / "best-practices-integration"
MKDOCS = ROOT / "mkdocs.yml"
TARGET = "archive/tutorials/best-practices-integration/index.md"


def main() -> None:
    entries = []
    for path in sorted(ARCHIVE.glob("*.md")):
        if path.name == "index.md":
            continue
        old = f"tutorials/best-practices-integration/{path.name}"
        entries.append(f"        {old}: {TARGET}\n")

    text = MKDOCS.read_text(encoding="utf-8")
    marker = "        best-practices/python/python-package-development.md:"
    block = "".join(entries)
    if block.strip() and block.splitlines()[0] not in text:
        text = text.replace(
            "        best-practices/python/python-package-development.md: best-practices/python/python-package.md\n",
            "        best-practices/python/python-package-development.md: best-practices/python/python-package.md\n"
            + block,
        )
        MKDOCS.write_text(text, encoding="utf-8")
        print(f"Added {len(entries)} integration redirects")
    else:
        print("No new redirects added")


if __name__ == "__main__":
    main()
