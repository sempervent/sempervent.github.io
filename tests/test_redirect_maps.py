"""Redirect map sanity checks and HTTP smoke tests against built site."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

REDIRECT_LINE = re.compile(
    r"^\s+(?P<src>[A-Za-z0-9_./-]+\.md):\s*(?P<dst>[A-Za-z0-9_./-]+\.md)\s*$",
    re.MULTILINE,
)


def _load_redirect_maps() -> dict[str, str]:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    maps: dict[str, str] = {}
    for match in REDIRECT_LINE.finditer(text):
        maps[match.group("src")] = match.group("dst")
    assert maps, "redirect_maps not found in mkdocs.yml"
    return maps


def test_redirect_maps_have_distinct_source_and_target_urls() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from redirect_url import doc_path_to_url

    maps = _load_redirect_maps()
    assert maps, "expected at least one redirect"
    collisions: list[str] = []
    for src, dst in maps.items():
        src_url = doc_path_to_url(src)
        dst_url = doc_path_to_url(dst)
        if src_url == dst_url:
            collisions.append(f"{src} -> {dst} both resolve to {src_url}")
    assert not collisions, "self-redirects in mkdocs.yml:\n" + "\n".join(collisions)


def test_projects_index_is_not_meta_redirect() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_portfolio.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    mkdocs = ROOT / ".venv" / "bin" / "mkdocs"
    subprocess.run(
        [str(mkdocs) if mkdocs.exists() else "mkdocs", "build", "--strict"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    html = (SITE / "projects" / "index.html").read_text(encoding="utf-8")
    assert "http-equiv=refresh" not in html.lower()
    assert 'url=/projects/' not in html.lower().replace(" ", "")


def _serve_site() -> tuple[ThreadingHTTPServer, str]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SITE), **kwargs)

        def log_message(self, *_args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, base


def _get_html(base: str, path: str) -> str:
    with urlopen(f"{base}{path}", timeout=10) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _is_mkdocs_redirect_page(html: str) -> bool:
    return "Redirecting..." in html or "http-equiv=refresh" in html.lower()


def test_http_routing_smoke() -> None:
    if not SITE.exists():
        subprocess.run(
            [str(ROOT / ".venv" / "bin" / "mkdocs"), "build", "--strict"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    httpd, base = _serve_site()
    try:
        projects = _get_html(base, "/projects/")
        assert not _is_mkdocs_redirect_page(projects), "/projects/ must not be a redirect stub"
        assert "project-card" in projects or "Portfolio" in projects

        doc_sites = _get_html(base, "/projects/documentation-sites/")
        assert not _is_mkdocs_redirect_page(doc_sites)

        tags = _get_html(base, "/tags/")
        assert _is_mkdocs_redirect_page(tags), "/tags/ should redirect once to Writing"
        assert "../documentation/" in tags or "/documentation/" in tags
        writing = _get_html(base, "/documentation/")
        assert not _is_mkdocs_redirect_page(writing)

        glitch = _get_html(base, "/tutorials/python-development/js-glitch-observatory/")
        assert _is_mkdocs_redirect_page(glitch)
        assert "just-for-fun/js-glitch-observatory" in glitch

        lab = _get_html(base, "/lab/")
        assert _is_mkdocs_redirect_page(lab)
        assert "just-for-fun" in lab
    finally:
        httpd.shutdown()
