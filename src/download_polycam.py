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
# The download event fires reliably (even in --attach/CDP) and exposes download.url — a
# Cloudflare R2 signed URL. We capture it and fetch it directly with context.request, which is
# far more robust than save_as (CDP quirks) or watching download folders.
_pending_download = {"url": None, "name": None}


def handle_download(download: Download) -> None:
    _pending_download["url"] = download.url
    _pending_download["name"] = download.suggested_filename
    print(f"  [nedlasting fanget] {download.suggested_filename}", flush=True)


def _load_ledger() -> set[str]:
    if LEDGER.exists():
        return {line.strip() for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()


def _record_ledger(capture_id: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as handle:
        handle.write(capture_id + "\n")


# The real 'images' export is a normal browser download (opens a popup). We don't catch it as a
# network response; we watch the actual download folders for the finished file and move it in.
_PARTIAL_SUFFIXES = (".crdownload", ".tmp", ".partial", ".part", ".download")


def _downloads_dir() -> Path:
    """The user's real Downloads folder from the Windows registry (handles OneDrive redirection)."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        value, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
        return Path(value)
    except Exception:
        return Path.home() / "Downloads"


def _watch_dirs() -> list[Path]:
    candidates = [
        RAW_DIR,
        _downloads_dir(),
        Path.home() / "Downloads",
        Path.home() / "OneDrive" / "Downloads",
        Path.home() / "OneDrive - Oslo kommune" / "Downloads",
    ]
    seen: set[str] = set()
    result: list[Path] = []
    for directory in candidates:
        key = str(directory).lower()
        if key not in seen:
            seen.add(key)
            result.append(directory)
    return result


WATCH_DIRS = _watch_dirs()


def _snapshot_files(dirs: list[Path]) -> set[str]:
    found: set[str] = set()
    for directory in dirs:
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    found.add(str(path))
    return found


def _partial_size(dirs: list[Path]) -> int:
    """Largest in-progress (.crdownload) download size across the watched folders, 0 if none."""
    biggest = 0
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.crdownload"):
            try:
                biggest = max(biggest, path.stat().st_size)
            except OSError:
                continue
    return biggest


def _new_finished_file(dirs: list[Path], before: set[str], min_bytes: int = 1_000_000) -> Path | None:
    """Any new, finished (no partial suffix / .crdownload sibling), big-enough file — wherever
    the browser dropped it (Downloads, OneDrive-redirected Downloads, or data/raw)."""
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or str(path) in before:
                continue
            if path.suffix.lower() in _PARTIAL_SUFFIXES:
                continue
            if Path(str(path) + ".crdownload").exists():
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


def collect_capture_urls(page: Page, stop_after: int | None = None) -> list[str]:
    """Scroll the (virtualized) library to load every card, collecting as we go.

    The list unmounts cards that scroll out of view, so we scroll in SMALL overlapping steps
    (less than one viewport) and collect after every step — otherwise a whole screen of cards
    can mount and unmount between two big scrolls and be missed. `stop_after` stops early once
    that many captures are found (fast for small test runs)."""
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
        if stop_after and len(seen) >= stop_after:
            break
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


def _open_capture(page: Page, url: str) -> bool:
    """Navigate to a capture. Use domcontentloaded (the live 3D viewer never finishes 'load',
    which caused ERR_ABORTED/timeouts), and retry once."""
    for attempt in range(2):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            _settle(page)
            return True
        except Exception as error:
            if attempt == 0:
                page.wait_for_timeout(1500)
                continue
            print(f"  kunne ikke aapne siden: {str(error).splitlines()[0]}")
    return False


def export_capture(page: Page, context: BrowserContext, url: str, fmt: str) -> bool:
    """Polycam Export dialog flow: click 'Download' -> select the format tile ('Images', under
    'Other', below the fold) -> click 'Export'. The download is scoped to that click via
    page.expect_download (no global state -> no cross-capture mix-ups), then fetched by its URL
    (with save_as as a fallback)."""
    if not _open_capture(page, url):
        return False

    download_button = page.get_by_role("button", name="Download", exact=True)
    if not download_button.count():
        print("  fant ikke Download-knappen")
        return False
    download_button.first.click()
    page.wait_for_timeout(2000)  # let the Export dialog open

    tile = page.get_by_text(fmt, exact=True)
    if not tile.count():
        print(f"  fant ikke '{fmt}'-flisen i eksportdialogen")
        return False
    try:
        tile.first.scroll_into_view_if_needed(timeout=5000)
    except Exception:
        pass
    tile.first.click()
    page.wait_for_timeout(600)

    export_button = page.get_by_role("button", name="Export", exact=True)
    if not export_button.count():
        print("  fant ikke Export-knappen")
        return False

    try:
        with page.expect_download(timeout=120000) as download_info:
            export_button.last.click()
        download = download_info.value
    except Exception as error:
        print(f"  ingen nedlasting startet: {str(error).splitlines()[0]}")
        return False

    name = download.suggested_filename or f"{url.rstrip('/').split('/')[-1]}.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    target = RAW_DIR / name
    if target.exists():
        print(f"  hopper over (finnes): {name}", flush=True)
        return True

    # Preferred: fetch the signed URL directly. Fallback: Playwright's own save_as.
    try:
        response = context.request.get(download.url, timeout=300000)
        if response.ok:
            body = response.body()
            if len(body) >= 1_000_000:
                target.write_bytes(body)
                print(f"  lagret: {name} ({len(body) / 1e6:.1f} MB) -> data/raw", flush=True)
                return True
            print(f"  URL ga for lite ({len(body)} bytes), proever save_as ...", flush=True)
        else:
            print(f"  URL ga HTTP {response.status}, proever save_as ...", flush=True)
    except Exception as error:
        print(f"  URL-henting feilet ({str(error).splitlines()[0]}), proever save_as ...", flush=True)

    try:
        download.save_as(str(target))
        print(f"  lagret (save_as): {name} ({target.stat().st_size / 1e6:.1f} MB) -> data/raw", flush=True)
        return True
    except Exception as error:
        print(f"  save_as feilet: {str(error).splitlines()[0]}")
        return False


def run_dump(context: BrowserContext, url: str) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    _wait_login(page)
    urls = collect_capture_urls(page)
    print(f"\nfant {len(urls)} captures i biblioteket")
    if not urls:
        return
    shots = PROJECT_ROOT / "outputs"
    shots.mkdir(parents=True, exist_ok=True)
    print(f"åpner første capture:\n  {urls[0]}")
    page.goto(urls[0])
    _settle(page)
    try:
        page.screenshot(path=str(shots / "polycam_capture.png"))
    except Exception as error:
        print(f"  (skjermbilde feilet: {error})")
    dump_controls(page, "capture-side")

    download = page.get_by_role("button", name="Download", exact=True)
    if download.count():
        download.first.click()
        page.wait_for_timeout(2500)
        try:
            page.screenshot(path=str(shots / "polycam_download_panel.png"), full_page=True)
            print(f"  skjermbilde av nedlastingsmenyen -> {shots / 'polycam_download_panel.png'}")
        except Exception as error:
            print(f"  (skjermbilde feilet: {error})")
        dump_controls(page, "etter klikk paa Download")

    print("\nSkjermbilder lagret i outputs/. Send dem, saa ser jeg menyen og laaser 'images'.")
    input("Trykk Enter for aa lukke ... ")


def run_auto(context: BrowserContext, url: str, limit: int, fmt: str) -> None:
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url)
    _wait_login(page)
    # For small test runs, stop scrolling as soon as we have enough captures (the full library
    # scroll takes minutes); for the real bulk run (large --limit) scroll everything.
    stop_after = (limit + 10) if limit <= 50 else None
    urls = collect_capture_urls(page, stop_after=stop_after)
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
    parser.add_argument("--format", default="Images",
                        help="format-flisen i eksportdialogen ('Images' = raa keyframe-zip)")
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
