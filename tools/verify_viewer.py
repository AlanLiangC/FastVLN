"""Optional Playwright acceptance check against the running real simulator UI."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--screenshot", default="runtime/reports/viewer.png")
    args = parser.parse_args()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.locator("#episode").fill("5")
        page.locator("#reset").click()
        page.wait_for_function("!document.getElementById('once').disabled", timeout=120000)
        page.locator("#once").click()
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('步数')", timeout=120000
        )
        assert page.locator("#error").inner_text() == ""
        assert "480×270" in page.locator("#sensor").inner_text()
        assert page.locator("#frame").evaluate("i=>[i.naturalWidth,i.naturalHeight]") == [480, 270]
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=args.screenshot, full_page=True)
        print(
            json.dumps(
                {
                    "checkpoint": page.locator("#checkpoint").inner_text(),
                    "status": page.locator("#status").inner_text(),
                    "sensor": page.locator("#sensor").inner_text(),
                    "errors": errors,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        browser.close()
    assert not errors


if __name__ == "__main__":
    main()
