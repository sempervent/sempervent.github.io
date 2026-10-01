"""Browser checks for Material light/dark palette consistency."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright


def _build_site() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_projects.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_portfolio.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    mkdocs = ROOT / ".venv" / "bin" / "mkdocs"
    cmd = [str(mkdocs) if mkdocs.exists() else "mkdocs", "build", "--strict"]
    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


def _serve_site() -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SITE), **kwargs)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _contrast_ratio(fg: str, bg: str) -> float:
    def parse(c: str) -> float:
        c = c.strip()
        if c.startswith("rgb"):
            parts = [float(x) for x in c.replace("rgba(", "").replace("rgb(", "").replace(")", "").split(",")[:3]]
            r, g, b = [x / 255 for x in parts]
        else:
            raise ValueError(c)
        def lin(v: float) -> float:
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        rs, gs, bs = lin(r), lin(g), lin(b)
        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs

    l1, l2 = parse(fg), parse(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _card_readable(page, path: str) -> None:
    page.goto(f"{base}{path}", wait_until="networkidle")
    card = page.locator(".project-card").first
    if card.count() == 0:
        return
    scheme = page.locator("html").get_attribute("data-md-color-scheme")
    bg = card.evaluate("el => getComputedStyle(el).backgroundColor")
    fg = card.evaluate("el => getComputedStyle(el).color")
    ratio = _contrast_ratio(fg, bg)
    assert ratio >= 4.0, f"{path} scheme={scheme} card contrast {ratio:.2f} fg={fg} bg={bg}"


@pytest.fixture(scope="module")
def site_server():
    _build_site()
    httpd = _serve_site()
    port = httpd.server_address[1]
    global base
    base = f"http://127.0.0.1:{port}/"
    yield
    httpd.shutdown()


def test_palette_toggle_project_cards(site_server):
    paths = ["/", "/projects/", "/best-practices/geospatial/geospatial-system-design/", "/tutorials/docker-infrastructure/rke2-raspberry-pi/"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme_name, toggle_times in (("default", 0), ("slate", 1)):
            context = browser.new_context()
            page = context.new_page()
            page.goto(base, wait_until="networkidle")
            for _ in range(toggle_times):
                page.locator('label[for="__palette_1"], label[for="__palette_0"]').first.click()
                page.wait_for_timeout(300)
            html_scheme = page.locator("html").get_attribute("data-md-color-scheme")
            assert html_scheme == scheme_name, f"expected {scheme_name}, got {html_scheme}"
            for path in paths:
                _card_readable(page, path)
            context.close()
        browser.close()
