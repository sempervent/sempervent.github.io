"""Map MkDocs doc paths to public URL paths (directory URLs enabled)."""

from __future__ import annotations


def doc_path_to_url(doc_path: str) -> str:
    """Convert a docs-relative markdown path to the site URL path."""
    path = doc_path.strip().replace("\\", "/")
    if not path.endswith(".md"):
        raise ValueError(f"expected .md path, got {doc_path!r}")
    path = path[: -len(".md")]
    if path.endswith("/index"):
        path = path[: -len("/index")]
    if not path:
        return "/"
    return f"/{path}/"
