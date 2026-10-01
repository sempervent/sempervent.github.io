#!/usr/bin/env python3
"""Fail on egregious generated-prose fingerprints in technical corpus."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "docs" / "best-practices", ROOT / "docs" / "tutorials"]

# Hard failures (serious template voice)
EGREGIOUS = [
    re.compile(r"weapon of choice", re.I),
    re.compile(r"complete machinery", re.I),
    re.compile(r"Objective:\s*Master", re.I),
]

# Allowed in Just for Fun only
JFF = "just-for-fun"


def main() -> int:
    failures: list[str] = []
    for base in DIRS:
        for path in sorted(base.rglob("*.md")):
            rel = path.relative_to(ROOT / "docs").as_posix()
            if JFF in rel:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pat in EGREGIOUS:
                if pat.search(text):
                    failures.append(f"{rel}: {pat.pattern}")
    if failures:
        print("Writing lint failures:", file=sys.stderr)
        for line in failures[:50]:
            print(f"  {line}", file=sys.stderr)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more", file=sys.stderr)
        return 1
    print("OK: no egregious writing fingerprints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
