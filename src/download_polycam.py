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
import shutil
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
LEDGER = PROJECT_ROOT / "outputs" / "downloaded_captures.txt"

# Global download counter so a download is caught no matter which tab/popup it lands in.
_downloads = {"count": 0}


def handle_download(download: Download) -> None:
    _downloads["count"] += 1
    try:
        name = download.suggested_filename
        print(f"  [nedlasting fanget] {name}  <- {download.url[:100]}", flush=True)
        target = RAW_DIR / name
        if target.exists():
            print(f"  hopper over (finnes): {name}", flush=True)
            return
        download.save_as(target)
        size_mb = target.stat().st_size / 1e6
        print(f"  lagret: {name} ({size_mb:.1f} MB) -> data/raw", flush=True)
    except Exception as error:
        print(f"  (nedlasting kunne ikke lagres: {error})", flush=True)


def _load_ledger() -> set[str]:
    if LEDGER.exists():
        return {line.strip() for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()


def _record_ledger(capture_id: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as handle:
        handle.write(capture_id + "\n")


# The real 'images' export is a normal browser download (opens a popup). We don't try to catch
# it as a network response; we redirect downloads to data/raw and watch for the finished file.
WATCH_DIRS = [RAW_DIR, Path.home() / "Downloads"]


def _snapshot_zips(dirs: list[Path]) -> set[str]:
    found: set[str] = set()
    for directory in dirs:
        if directory.exists():
            for path in directory.glob("*.zip"):
                found.add(str(path))
    return found


def _new_finished_zip(dirs: list[Path], before: set[str], min_bytes: int = 1_000_000) -> Path | None:
    """A .zip that is new, no longer downloading (.crdownload gone), and big enough to be real."""
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.zip"):
            if str(path) in before or Path(str(path) + ".crdownload").exists():
                continue
            try:
                if path.stat().st_size >= min_bytes:
                    return path
            except OSError:
                continue
    return None


def _set_download_path(context: BrowserContext, page: Page, folder: Path) -> None:
    """Point the browser's downloads at `folder` via CDP so the popup download lands there."""
    try:
        session = context.new_cdp_session(page)
        session.send("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(folder)})
        print(f"  nedlastinger settes til {folder}", flush=True)
    except Exception as error:
        print(f"  (kunne ikke omdirigere nedlastinger: {error}; overvaaker data/raw + Downloads)", flush=True)


def _pump(context: BrowserContext, ms: int) -> None:
    """Let Playwright process events for ~ms milliseconds without blocking the download."""
    alive = [p for p in context.pages if not p.is_closed()]
    if alive:
        try:
            alive[0].wait_for_timeout(ms)
            return
        except Exception:
            pass
    time.sleep(ms / 1000)


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


def _settle(page: Page, ms: int = 3000) -> None:
    """Capture pages run a live 3D viewer, so they never reach 'networkidle'. Wait for the DOM
    and then a fixed beat instead, and never let it raise."""
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    try:
        page.wait_for_timeout(ms)
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
            headers = response.headers
            content_type = headers.get("content-type", "")
            content_length = headers.get("content-length", "?")
            disposition = headers.get("content-disposition", "")
            trace.write(
                f"{response.status} {response.request.method} {response.request.resource_type} "
                f"ct={content_type} len={content_length} cd={disposition} {response.url}\n"
            )
            url_lower = response.url.lower()
            keys = ("capture", "download", "export", "asset", "api", "archive", "zip", "raw", "job")
            if "json" in content_type and any(key in url_lower for key in keys):
                try:
                    trace.write("  BODY: " + response.text()[:6000].replace("\n", " ") + "\n")
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
    """Scroll the (virtualized) library to load every card, collecting as we go.

    The list unmounts cards that scroll out of view, so we scroll in SMALL overlapping steps
    (less than one viewport) and collect after every step — otherwise a whole screen of cards
    can mount and unmount between two big scrolls and be missed."""
    print("  venter 10 s paa at biblioteket lastes inn ...", flush=True)
    page.wait_for_timeout(10000)

    viewport = page.viewport_size or {"width": 1280, "height": 800}
    page.mouse.move(viewport["width"] / 2, viewport["height"] / 2)  # so the wheel scrolls the list
    step = int(viewport["height"] * 0.6)  # overlap ~40% between windows

    seen: dict[str, None] = {}

    def collect() -> None:
        anchors = page.locator("a[href*='/capture/']")
        for i in range(anchors.count()):
            href = anchors.nth(i).get_attribute("href")
            if href:
                url = href if href.startswith("http") else f"https://poly.cam{href}"
                seen.setdefault(url.split("?")[0], None)

    no_growth = 0
    for iteration in range(1000):
        collect()
        before = len(seen)
        page.mouse.wheel(0, step)
        page.wait_for_timeout(600)
        collect()
        no_growth = no_growth + 1 if len(seen) == before else 0
        if no_growth >= 12:  # ~7 s of no new cards => reached the bottom
            break
        if iteration % 15 == 0 and iteration:
            print(f"    ... {len(seen)} captures funnet saa langt", flush=True)
    print(f"  ferdig: {len(seen)} captures", flush=True)
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


def export_capture(page: Page, context: BrowserContext, url: str, fmt: str) -> bool:
    """Real Polycam flow: click 'Download' -> the 'images' option (raw keyframe zip) ->
    'Download now' if shown. The browser performs the actual download (its popup works); we've
    redirected downloads to data/raw and just watch for the finished zip to appear."""
    page.goto(url)
    _settle(page)

    download_button = page.get_by_role("button", name="Download", exact=True)
    if not download_button.count():
        return False
    download_button.first.click()
    page.wait_for_timeout(1500)

    option = page.get_by_role("button", name=fmt)  # case-insensitive substring match
    if not option.count():
        print(f"  fant ikke '{fmt}' i nedlastingsmenyen")
        return False

    before = _snapshot_zips(WATCH_DIRS)
    option.first.click()
    page.wait_for_timeout(1000)
    confirm = page.get_by_role("button", name="Download now", exact=True)
    if confirm.count() and confirm.first.is_visible():
        confirm.first.click()

    for i in range(150):
        found = _new_finished_zip(WATCH_DIRS, before)
        if found:
            if found.parent != RAW_DIR:
                destination = RAW_DIR / found.name
                shutil.move(str(found), str(destination))
                found = destination
            print(f"  lagret: {found.name} ({found.stat().st_size / 1e6:.1f} MB) -> data/raw", flush=True)
            return True
        if i and i % 15 == 0:
            pending = any(d.exists() and list(d.glob("*.crdownload")) for d in WATCH_DIRS)
            print(f"    ... venter paa nedlasting{' (paagaar)' if pending else ''} ({i}s)", flush=True)
        _pump(context, 1000)
    print("  ingen ny zip dukket opp (tidsavbrudd)")
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
    _settle(page)
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


def run_auto(context: BrowserContext, url: str, limit: int, fmt: str) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    _wait_login(page)
    _set_download_path(context, page, RAW_DIR)
    urls = collect_capture_urls(page)
    ledger = _load_ledger()
    todo = [u for u in urls if u.rstrip("/").split("/")[-1] not in ledger]
    print(f"\nfant {len(urls)} captures, {len(urls) - len(todo)} allerede lastet ned "
          f"(hopper over), laster ned '{fmt}'")

    done, failed = 0, []
    for index, capture_url in enumerate(todo[:limit], start=1):
        capture_id = capture_url.rstrip("/").split("/")[-1]
        print(f"\n[{index}/{min(len(todo), limit)}] {capture_url}", flush=True)
        try:
            if export_capture(page, context, capture_url, fmt):
                _record_ledger(capture_id)
                done += 1
            else:
                failed.append(capture_url)
                if len(failed) == 1:
                    dump_controls(page, "FEILET - kontroller paa siden")
        except Exception as error:
            print(f"  feil: {error}")
            failed.append(capture_url)

    print(f"\nferdig: {done} lastet ned, {len(failed)} feilet")


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
    parser.add_argument("--limit", type=int, default=100000, help="maks antall skann (standard: alle)")
    parser.add_argument("--format", default="images",
                        help="knappen som lastes ned i auto-modus ('images' = raa keyframe-zip)")
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
                run_auto(context, args.url, args.limit, args.format)
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
