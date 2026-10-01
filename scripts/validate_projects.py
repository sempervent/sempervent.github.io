#!/usr/bin/env python3
"""Validate data/projects.yaml for the portfolio registry."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "projects.yaml"

STATUSES = frozenset({"active", "maintained", "experimental", "historical", "archived"})
CATEGORIES = frozenset(
    {
        "engineering",
        "documentation",
        "creative",
        "games",
        "infrastructure",
        "historical",
    }
)

URL_RE = re.compile(r"^https?://", re.I)


def load_registry() -> dict:
    with REGISTRY.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "projects" not in data:
        raise ValueError("Registry must be a mapping with a 'projects' list")
    if not isinstance(data["projects"], list):
        raise ValueError("'projects' must be a list")
    return data


def validate_url(field: str, url: str, errors: list[str]) -> None:
    if not URL_RE.match(url):
        errors.append(f"{field}: invalid URL {url!r}")
        return
    parsed = urlparse(url)
    if not parsed.netloc:
        errors.append(f"{field}: missing host in {url!r}")


def validate_registry(data: dict) -> list[str]:
    errors: list[str] = []
    slugs: set[str] = set()
    docs_urls: dict[str, str] = {}

    for idx, project in enumerate(data["projects"]):
        prefix = f"projects[{idx}]"
        if not isinstance(project, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue

        slug = project.get("slug")
        if not slug or not isinstance(slug, str):
            errors.append(f"{prefix}: missing or invalid slug")
            continue
        if slug in slugs:
            errors.append(f"duplicate slug: {slug}")
        slugs.add(slug)

        for key in ("name", "summary", "status", "category"):
            if not project.get(key):
                errors.append(f"{prefix} ({slug}): missing required field '{key}'")

        status = project.get("status")
        if status and status not in STATUSES:
            errors.append(f"{prefix} ({slug}): invalid status {status!r}")

        category = project.get("category")
        if category and category not in CATEGORIES:
            errors.append(f"{prefix} ({slug}): invalid category {category!r}")

        if project.get("featured") and not str(project.get("summary", "")).strip():
            errors.append(f"{prefix} ({slug}): featured project missing summary")

        repo = project.get("repo")
        if repo:
            validate_url(f"{prefix} ({slug}).repo", repo, errors)

        docs = project.get("docs")
        if docs:
            validate_url(f"{prefix} ({slug}).docs", docs, errors)
            if docs in docs_urls:
                errors.append(
                    f"docs URL {docs} used by both {docs_urls[docs]} and {slug}"
                )
            docs_urls[docs] = slug

        demo = project.get("demo")
        if demo:
            validate_url(f"{prefix} ({slug}).demo", demo, errors)

    return errors


def main() -> int:
    if not REGISTRY.is_file():
        print(f"Registry not found: {REGISTRY}", file=sys.stderr)
        return 1
    try:
        data = load_registry()
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"Failed to load registry: {exc}", file=sys.stderr)
        return 1

    errors = validate_registry(data)
    if errors:
        print("Project registry validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(data['projects'])} projects validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
