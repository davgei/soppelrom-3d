"""Download Polycam capture zips from the web library into data/raw via a controlled browser.

Uses the system Edge (signed executable, works on locked-down machines) with a persistent
profile stored under outputs/browser_profile — log in once, stay logged in.

Modes:
  assist (default): opens the library; YOU click the downloads — every download the browser
      makes is captured, named correctly and saved into data/raw, skipping files you already
      have. Works regardless of Polycam UI changes.
  --auto: experimental full automation — tries to find capture pages and click
      download/raw-data buttons via text heuristics. Expect to tune selectors.

Usage:
  .venv\\Scripts\\python.exe -m src.download_polycam
  .venv\\Scripts\\python.exe -m src.download_polycam --auto --limit 5
"""
from __future__ import annotations

import argparse
import re

from playwright.sync_api import BrowserContext, Download, Page, TimeoutError, sync_playwright

from .prepare_scan import PROJECT_ROOT, RAW_DIR

PROFILE_DIR = PROJECT_ROOT / "outputs" / "browser_profile"
LIBRARY_URL = "https://poly.cam/library"

DOWNLOAD_BUTTON_PATTERN = re.compile(r"download|last ned|export", re.IGNORECASE)
RAW_OPTION_PATTERN = re.compile(r"raw|original|keyframe|source|data", re.IGNORECASE)


def handle_download(download: Download) -> None:
    name = download.suggested_filename
    target = RAW_DIR / name
    if target.exists():
        print(f"  hopper over (finnes): {name}")
        download.cancel()
        return
    download.save_as(target)
    size_mb = target.stat().st_size / 1e6
    print(f"  lagret: {name} ({size_mb:.1f} MB) -> data/raw")


def attach_download_capture(context: BrowserContext) -> None:
    def on_page(page: Page) -> None:
        page.on("download", handle_download)

    for page in context.pages:
        on_page(page)
    context.on("page", on_page)


def _wait_until_browser_closed(context: BrowserContext) -> None:
    try:
        while context.pages:
            context.pages[0].wait_for_timeout(500)
    except Exception:
        pass


def run_assist(context: BrowserContext, url: str) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    print(
        "\nAssistert modus:\n"
        "  1. Logg inn hvis du ikke er det (huskes til neste gang)\n"
        "  2. Klikk deg til skannene og velg nedlasting (raw data / zip)\n"
        "  3. Alle nedlastinger lagres automatisk i data/raw med riktig navn\n"
        "  4. Lukk nettleseren når du er ferdig\n"
    )
    _wait_until_browser_closed(context)


def run_discover(context: BrowserContext, url: str) -> None:
    trace_path = PROJECT_ROOT / "outputs" / "polycam_trace.log"
    trace = open(trace_path, "w", encoding="utf-8")

    def log_response(response) -> None:
        try:
            resource_type = response.request.resource_type
            if resource_type not in ("xhr", "fetch", "document"):
                return
            content_type = response.headers.get("content-type", "")
            trace.write(
                f"{response.status} {response.request.method} {resource_type} "
                f"{content_type} {response.url}\n"
            )
            interesting = any(
                key in response.url.lower()
                for key in ("capture", "library", "asset", "download", "export", "api", "raw")
            )
            if interesting and "json" in content_type:
                try:
                    body = response.text()
                    trace.write("  BODY: " + body[:8000].replace("\n", " ") + "\n")
                except Exception:
                    pass
            trace.flush()
        except Exception:
            pass

    context.on("response", log_response)
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    print(
        "\nDiscover-modus (kartlegger API-et for full automatisering):\n"
        "  1. Logg inn og gå til biblioteket\n"
        "  2. Scroll gjennom biblioteket (så liste-API-et vises)\n"
        "  3. Last ned ETT skann manuelt (raw data / zip)\n"
        "  4. Lukk nettleseren\n"
        f"  -> all trafikk logges til {trace_path}\n"
    )
    _wait_until_browser_closed(context)
    trace.close()
    print(f"trace lagret: {trace_path}")


def run_auto(context: BrowserContext, url: str, limit: int) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    print("Auto-modus (eksperimentell). Logg inn i vinduet hvis nødvendig.")
    input("Trykk Enter her når biblioteket er synlig i nettleseren ... ")

    links = page.locator("a[href*='/captures/']")
    hrefs: list[str] = []
    for i in range(links.count()):
        href = links.nth(i).get_attribute("href")
        if href and href not in hrefs:
            hrefs.append(href)
    print(f"fant {len(hrefs)} captures i biblioteket")

    for href in hrefs[:limit]:
        capture_url = href if href.startswith("http") else f"https://poly.cam{href}"
        print(f"\ncapture: {capture_url}")
        capture_page = context.new_page()
        capture_page.on("download", handle_download)
        try:
            capture_page.goto(capture_url)
            capture_page.wait_for_load_state("networkidle")

            button = capture_page.get_by_role("button", name=DOWNLOAD_BUTTON_PATTERN).first
            button.click(timeout=8000)

            option = capture_page.get_by_text(RAW_OPTION_PATTERN).first
            with capture_page.expect_download(timeout=60000):
                option.click(timeout=8000)
        except TimeoutError:
            print("  fant ikke nedlastingsknapp/valg her — kjør uten --auto og klikk manuelt,")
            print("  eller si ifra hva knappene heter, så justerer vi selektorene.")
        finally:
            capture_page.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Last ned Polycam-skann til data/raw.")
    parser.add_argument("--auto", action="store_true", help="prøv full automatisering (eksperimentell)")
    parser.add_argument("--discover", action="store_true",
                        help="kartlegg API-kallene (logg inn + last ned ETT skann manuelt)")
    parser.add_argument("--url", default=LIBRARY_URL, help="bibliotek-URL")
    parser.add_argument("--limit", type=int, default=10, help="maks captures i auto-modus")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            channel="msedge",
            headless=False,
            accept_downloads=True,
        )
        attach_download_capture(context)
        try:
            if args.discover:
                run_discover(context, args.url)
            elif args.auto:
                run_auto(context, args.url, args.limit)
            else:
                run_assist(context, args.url)
        finally:
            try:
                context.close()
            except Exception:
                pass
    print("\nferdig. Nye skann i data/raw plukkes opp av annoteringsverktøyets bakgrunnsarbeider.")


if __name__ == "__main__":
    main()
