#!/usr/bin/env python3
"""Remove single-line Objective boilerplate and 'weapon of choice' lines from technical docs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "docs" / "best-practices", ROOT / "docs" / "tutorials"]
SKIP = ("just-for-fun", "archive")


def clean(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**Objective**:"):
            continue
        if re.search(r"weapon of choice", line, re.I):
            continue
        if re.search(r"complete machinery", line, re.I):
            continue
        out.append(line)
    # Collapse triple blank lines
    text = "\n".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text + ("\n" if text.endswith("\n") or not text else "\n")


def main() -> None:
    changed = 0
    for base in DIRS:
        for path in sorted(base.rglob("*.md")):
            if any(s in path.as_posix() for s in SKIP):
                continue
            original = path.read_text(encoding="utf-8")
            updated = clean(original)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                changed += 1
    print(f"Updated {changed} files")


if __name__ == "__main__":
    main()
