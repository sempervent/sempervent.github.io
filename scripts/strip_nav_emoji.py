#!/usr/bin/env python3
"""Remove decorative emoji prefixes from mkdocs.yml nav section labels."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MKDOCS = ROOT / "mkdocs.yml"

# Leading emoji / symbol run before a letter or quote in nav labels
EMOJI_PREFIX = re.compile(
    r"^(?P<indent>\s+- )(?P<emoji>[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002700-\U000027BF\U0000FE0F\U0000200D"
    r"\U0001F1E6-\U0001F1FF📌🔗🛠🖼🎨🚀]+)\s+",
    re.UNICODE,
)


def main() -> None:
    text = MKDOCS.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = 0
    for line in lines:
        m = EMOJI_PREFIX.match(line.rstrip("\n"))
        if m:
            newline = m.group("indent") + line[m.end() :]
            if newline != line.rstrip("\n"):
                changed += 1
            out.append(newline if line.endswith("\n") else newline)
        else:
            out.append(line)
    MKDOCS.write_text("".join(out), encoding="utf-8")
    print(f"Stripped emoji from {changed} nav lines in mkdocs.yml")


if __name__ == "__main__":
    main()
