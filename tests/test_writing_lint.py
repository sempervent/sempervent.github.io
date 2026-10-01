"""Writing quality guardrails."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lint_writing_script_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lint_writing.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_redirect_maps_include_merged_python_guides() -> None:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "best-practices/python/typing-in-python.md:" in text
    assert "tutorials/best-practices-integration/event-driven-microservices" in text
