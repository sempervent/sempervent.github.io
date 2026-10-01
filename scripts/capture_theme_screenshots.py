#!/usr/bin/env python3
"""Capture light/dark homepage and tutorial screenshots after mkdocs build."""

from __future__ import annotations

import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = ROOT / "maintainers-theme-evidence"


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install playwright in the venv to capture screenshots", file=sys.stderr)
        return 1

    mkdocs = ROOT / ".venv" / "bin" / "mkdocs"
    subprocess.run(
        [str(mkdocs) if mkdocs.exists() else "mkdocs", "build", "--strict"],
        cwd=ROOT,
        check=True,
    )

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(SITE), **kwargs)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for name, path in (
            ("home-light", "/"),
            ("home-dark", "/"),
            ("tutorial-light", "/tutorials/docker-infrastructure/rke2-raspberry-pi/"),
            ("tutorial-dark", "/tutorials/docker-infrastructure/rke2-raspberry-pi/"),
        ):
            page.goto(base + path.lstrip("/"), wait_until="networkidle")
            page.evaluate(
                "() => { const c = document.querySelector('[data-md-component=\"consent\"]'); if (c) c.remove(); }"
            )
            if "dark" in name:
                page.evaluate(
                    "() => { const i = document.querySelector('#__palette_2'); if (i) { i.checked = true; i.dispatchEvent(new Event('change', {bubbles: true})); } }"
                )
                page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
        browser.close()
    httpd.shutdown()
    print(f"Wrote screenshots to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
