"""Tests for portfolio registry validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validate_projects_script_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_projects.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_generate_portfolio_script_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_portfolio.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_projects_index_omits_empty_sections() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_portfolio.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    index = (ROOT / "docs" / "projects" / "index.md").read_text(encoding="utf-8")
    assert "_None listed._" not in index
    assert "## Active development" not in index
    assert "data/projects.yaml" not in index
