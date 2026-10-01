"""Browser checks for Material light/dark palette and project-card computed colors."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
MKDOCS = ROOT / ".venv" / "bin" / "mkdocs"


def _build_site() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_projects.py")],
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
    cmd = [str(MKDOCS) if MKDOCS.exists() else "mkdocs", "build", "--strict"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)


def _serve_site() -> tuple[ThreadingHTTPServer, str]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SITE), **kwargs)

        def log_message(self, *_args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    return httpd, base


def _relative_luminance(rgb: str) -> float:
    parts = [float(x) for x in rgb.replace("rgba(", "").replace("rgb(", "").replace(")", "").split(",")[:3]]
    r, g, b = [x / 255 for x in parts]

    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    rs, gs, bs = lin(r), lin(g), lin(b)
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs


def _contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _card_metrics(page) -> dict[str, str]:
    page.locator(".project-card-grid .project-card").first.wait_for(timeout=10_000)
    return page.evaluate(
        """() => {
          const card = document.querySelector('.project-card-grid .project-card');
          const body = document.body;
          const cs = getComputedStyle(card);
          const bs = getComputedStyle(body);
          return {
            cardBg: cs.backgroundColor,
            cardFg: cs.color,
            bodyBg: bs.backgroundColor,
            scheme: body.getAttribute('data-md-color-scheme')
              || document.documentElement.getAttribute('data-md-color-scheme'),
          };
        }"""
    )


def _assert_light_card(metrics: dict[str, str], where: str) -> None:
    bg_l = _relative_luminance(metrics["cardBg"])
    fg_l = _relative_luminance(metrics["cardFg"])
    assert bg_l > 0.85, f"{where}: expected light card bg, got {metrics['cardBg']}"
    assert fg_l < 0.4, f"{where}: expected dark card text, got {metrics['cardFg']}"
    assert _contrast_ratio(metrics["cardFg"], metrics["cardBg"]) >= 4.5, metrics


def _assert_dark_card(metrics: dict[str, str], where: str) -> None:
    bg_l = _relative_luminance(metrics["cardBg"])
    fg_l = _relative_luminance(metrics["cardFg"])
    assert bg_l < 0.25, f"{where}: expected dark card bg, got {metrics['cardBg']}"
    assert fg_l > 0.6, f"{where}: expected light card text, got {metrics['cardFg']}"
    assert _contrast_ratio(metrics["cardFg"], metrics["cardBg"]) >= 4.5, metrics


def _click_palette(page, palette_id: str) -> None:
    page.locator(f"label[for='{palette_id}']").click(timeout=5000)
    page.wait_for_timeout(350)


@pytest.fixture(scope="module")
def site_server():
    _build_site()
    httpd, base = _serve_site()
    yield base
    httpd.shutdown()


def test_theme_project_cards_home_and_projects(site_server: str) -> None:
    base = site_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Light (default)
        page.goto(base, wait_until="load")
        _assert_light_card(_card_metrics(page), "home light")

        # Dark
        _click_palette(page, "__palette_1")
        _assert_dark_card(_card_metrics(page), "home dark")

        # Reload preserves dark
        page.reload(wait_until="load")
        _assert_dark_card(_card_metrics(page), "home dark after reload")

        # Projects via direct URL (avoid instant-nav loop on static server)
        page.goto(f"{base}projects/", wait_until="load")
        _assert_dark_card(_card_metrics(page), "projects dark")

        # Back to light
        _click_palette(page, "__palette_0")
        _assert_light_card(_card_metrics(page), "projects light")

        page.goto(base, wait_until="load")
        _assert_light_card(_card_metrics(page), "home light again")

        browser.close()
