"""Download Polycam capture zips from the web library into data/raw via a controlled browser.

Uses the system Edge (signed executable, works on locked-down machines) with a persistent
profile stored under outputs/browser_profile — log in once, stay logged in.

Modes:
  --dump:  open the first capture and print every button/menu label, so we can lock the
      export selectors to the real UI (run this first if --auto can't find the buttons).
  --auto:  full automation — scroll the whole library, then for each capture open it and
      click export -> raw data -> download, saving each zip into data/raw (dedup by name).
  assist (default): opens the library; YOU click downloads, every one is captured to data/raw.
  --discover: log network traffic while you download one scan manually (API mapping).

Usage:
  .venv\\Scripts\\python.exe -m src.download_polycam --dump
  .venv\\Scripts\\python.exe -m src.download_polycam --auto
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, Download, Page, TimeoutError, sync_playwright

from .prepare_scan import PROJECT_ROOT, RAW_DIR

PROFILE_DIR = PROJECT_ROOT / "outputs" / "browser_profile"
LIBRARY_URL = "https://poly.cam/library"
CDP_PORT = 9222
EDGE_PATHS = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]
STEALTH_ARGS = ["--disable-blink-features=AutomationControlled", "--no-first-run"]

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


def _wait_login(page: Page) -> None:
    print("Venter til du er innlogget og biblioteket vises ...", flush=True)
    for _ in range(600):
        if "/library" in page.url and page.locator("a[href*='/capture/']").count() > 0:
            return
        page.wait_for_timeout(1000)
    print("  (fortsetter uansett)")


def collect_capture_urls(page: Page) -> list[str]:
    """Scroll the library to load every card, then collect all capture URLs."""
    seen: dict[str, None] = {}
    no_growth = 0
    for _ in range(300):
        anchors = page.locator("a[href*='/capture/']")
        for i in range(anchors.count()):
            href = anchors.nth(i).get_attribute("href")
            if href:
                url = href if href.startswith("http") else f"https://poly.cam{href}"
                seen.setdefault(url.split("?")[0], None)
        before = len(seen)
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(700)
        anchors = page.locator("a[href*='/capture/']")
        for i in range(anchors.count()):
            href = anchors.nth(i).get_attribute("href")
            if href:
                url = href if href.startswith("http") else f"https://poly.cam{href}"
                seen.setdefault(url.split("?")[0], None)
        no_growth = no_growth + 1 if len(seen) == before else 0
        if no_growth >= 4:
            break
    return list(seen)


def dump_controls(page: Page, note: str) -> list[str]:
    """List every visible interactive element's text + aria-label, to learn the real UI labels."""
    items: list[str] = []
    for selector in ("button", "[role='menuitem']", "[role='button']", "a"):
        loc = page.locator(selector)
        for i in range(min(loc.count(), 80)):
            element = loc.nth(i)
            try:
                if not element.is_visible():
                    continue
                text = (element.inner_text() or "").strip().replace("\n", " ")
                aria = element.get_attribute("aria-label") or ""
                if text or aria:
                    items.append(f"{selector}: text='{text[:60]}' aria='{aria[:60]}'")
            except Exception:
                continue
    unique = list(dict.fromkeys(items))
    print(f"  [{note}] {len(unique)} synlige kontroller:")
    for line in unique:
        print("    " + line)
    return unique


def export_capture(page: Page, url: str) -> bool:
    """On a capture page: open export, choose raw data, trigger the download. Returns success."""
    page.goto(url)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)

    clicked = False
    for pattern in (DOWNLOAD_BUTTON_PATTERN, re.compile(r"\.\.\.|more|meny", re.IGNORECASE)):
        control = page.get_by_role("button", name=pattern)
        if control.count():
            control.first.click()
            clicked = True
            page.wait_for_timeout(1200)
            break
    if not clicked:
        return False

    for pattern in (RAW_OPTION_PATTERN, DOWNLOAD_BUTTON_PATTERN):
        option = page.get_by_text(pattern)
        if option.count():
            try:
                with page.expect_download(timeout=90000):
                    option.first.click()
                page.wait_for_timeout(500)
                return True
            except TimeoutError:
                continue
    return False


def run_dump(context: BrowserContext, url: str) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    _wait_login(page)
    urls = collect_capture_urls(page)
    print(f"\nfant {len(urls)} captures i biblioteket")
    if not urls:
        return
    print(f"åpner første capture for å kartlegge eksport-menyen:\n  {urls[0]}")
    page.goto(urls[0])
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)
    dump_controls(page, "capture-side")
    for pattern in (DOWNLOAD_BUTTON_PATTERN, re.compile(r"\.\.\.|more|meny", re.IGNORECASE)):
        control = page.get_by_role("button", name=pattern)
        if control.count():
            control.first.click()
            page.wait_for_timeout(1200)
            dump_controls(page, "etter klikk paa eksport/meny")
            break
    print("\nLim inn listen over kontroller over, saa laaser jeg selektorene i auto-modus.")
    input("Trykk Enter for aa lukke ... ")


def run_auto(context: BrowserContext, url: str, limit: int) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    _wait_login(page)
    urls = collect_capture_urls(page)
    existing = {p.stem for p in RAW_DIR.glob("*.zip")}
    print(f"\nfant {len(urls)} captures, {len(existing)} allerede lastet ned")

    done, failed = 0, []
    for index, capture_url in enumerate(urls[:limit], start=1):
        print(f"\n[{index}/{min(len(urls), limit)}] {capture_url}", flush=True)
        try:
            if export_capture(page, capture_url):
                done += 1
            else:
                failed.append(capture_url)
                if len(failed) == 1:
                    dump_controls(page, "FEILET - kontroller paa siden")
        except Exception as error:
            print(f"  feil: {error}")
            failed.append(capture_url)

    print(f"\nferdig: {done} lastet ned, {len(failed)} feilet")
    if failed:
        print("Hvis alt feilet: kjor --dump og lim inn menyvalgene, saa fikser jeg selektorene.")


def _find_edge() -> Path:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    raise SystemExit("fant ikke msedge.exe — er Edge installert?")


def _attach_plain_edge(playwright, url: str) -> BrowserContext:
    """Launch a completely normal Edge (no automation flags at all, only a debug port)
    and connect to it from the outside — indistinguishable from a hand-opened browser."""
    profile = PROJECT_ROOT / "outputs" / "browser_profile_cdp"
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([
        str(_find_edge()),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        url,
    ])
    for _ in range(30):
        try:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{CDP_PORT}")
            return browser.contexts[0]
        except Exception:
            time.sleep(1)
    raise SystemExit(f"fikk ikke kontakt med Edge på port {CDP_PORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Last ned Polycam-skann til data/raw.")
    parser.add_argument("--auto", action="store_true", help="prøv full automatisering (eksperimentell)")
    parser.add_argument("--discover", action="store_true",
                        help="kartlegg API-kallene (logg inn + last ned ETT skann manuelt)")
    parser.add_argument("--dump", action="store_true",
                        help="apne forste capture og skriv ut alle knapper/menyvalg (for selektor-tuning)")
    parser.add_argument("--attach", action="store_true",
                        help="bruk en helt vanlig Edge (omgår bot-deteksjon ved innlogging)")
    parser.add_argument("--url", default=LIBRARY_URL, help="bibliotek-URL")
    parser.add_argument("--limit", type=int, default=10, help="maks captures i auto-modus")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        if args.attach:
            context = _attach_plain_edge(playwright, args.url)
        else:
            context = playwright.chromium.launch_persistent_context(
                str(PROFILE_DIR),
                channel="msedge",
                headless=False,
                accept_downloads=True,
                args=STEALTH_ARGS,
                ignore_default_args=["--enable-automation"],
            )
        attach_download_capture(context)
        try:
            if args.dump:
                run_dump(context, args.url)
            elif args.discover:
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
