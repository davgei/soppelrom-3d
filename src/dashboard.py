"""Søppelrom 3D — desktop control panel.

Browse ALL scans in a list, see the rendered result images (room + measurements, free floor,
proposed new-bin placement), and launch the 3D viewer / annotation / entrance-picker — without
typing zip names in a terminal.

    .venv\\Scripts\\python.exe -m src.dashboard
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from . import pipeline
from .annotations import BIN_TYPES
from .set_entrance import ENTRANCE_DIR

VIEWS = [
    ("Rom + mål", "room_topdown.png"),
    ("Ledig gulv", "freespace_over_scene.png"),
    ("Plassering (ny kasse)", "placements.png"),
]

BG = "#1e2228"
PANEL = "#282d35"
ACCENT = "#3d8bfd"
TEXT = "#e6e9ef"
MUTED = "#9aa4b2"


class Dashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Søppelrom 3D — kontrollpanel")
        self.root.geometry("1320x840")
        self.root.minsize(1040, 660)
        self.root.configure(bg=BG)

        self.view = tk.StringVar(value=VIEWS[0][0])
        self.bin_type = tk.StringVar(value="4-hjuls container")
        self.status = tk.StringVar(value="Klar.")
        self._photo: ImageTk.PhotoImage | None = None
        self._pil: Image.Image | None = None
        self._busy = False

        self._build_style()
        self._build_layout()
        self._populate()
        self._signature = self._file_signature()
        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))
        # keyboard shortcuts for the most-used actions (letters shown in the button labels)
        self.root.bind("<g>", lambda _e: self._generate([self._selected()]))  # Generer bilder
        self.root.bind("<G>", lambda _e: self._generate_all())                # Shift+G = Generer alle
        self.root.bind("<o>", lambda _e: self._open_3d())                     # Åpne i 3D
        self.root.bind("<a>", lambda _e: self._annotate())                    # Annotér
        self.root.bind("<f>", lambda _e: self._prepare())                     # Forbered
        # refresh scan statuses the moment the dashboard regains focus (e.g. back from annotating)
        self.root.bind("<FocusIn>", lambda _e: self._refresh_if_changed())

    # ---------- styling ----------

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 16))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Stats.TLabel", background=PANEL, foreground=TEXT, font=("Consolas", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=7)
        style.map("Accent.TButton", background=[("!disabled", ACCENT)], foreground=[("!disabled", "#ffffff")])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=26, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])
        style.configure("TRadiobutton", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, arrowcolor=TEXT)
        # A readonly Combobox ignores the plain configure above; without an explicit state map
        # clam draws the field with its near-white default -> white text on a white box.
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL)],
            background=[("readonly", PANEL)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", PANEL)],
            selectforeground=[("readonly", TEXT)],
            arrowcolor=[("readonly", TEXT)],
        )
        # The drop-down list is a classic Tk Listbox (not ttk), styled via the option database.
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    # ---------- layout ----------

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 6))
        ttk.Label(header, text="Søppelrom 3D", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  velg skann · se resultat · åpne i 3D", style="TLabel").pack(side="left", padx=8)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14, pady=6)

        # left: scan list
        left = ttk.Frame(body, style="Panel.TFrame")
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Skann", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
        self.tree = ttk.Treeview(left, columns=("status", "bins"), show="tree headings", height=26, selectmode="browse")
        self.tree.heading("#0", text="Navn")
        self.tree.heading("status", text="Status")
        self.tree.heading("bins", text="Kasser")
        self.tree.column("#0", width=230)
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("bins", width=60, anchor="center")
        self.tree.tag_configure("annotated", foreground="#4ade80")
        self.tree.tag_configure("prepared", foreground=TEXT)
        self.tree.tag_configure("raw", foreground=MUTED)
        self.tree.pack(fill="y", expand=False, padx=10, pady=(0, 10))
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        # right: viewer
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x")
        for label, _file in VIEWS:
            ttk.Radiobutton(toolbar, text=label, value=label, variable=self.view,
                            command=self._show_image).pack(side="left", padx=(0, 10))
        ttk.Label(toolbar, text="Kassetype:").pack(side="left", padx=(20, 4))
        combo = ttk.Combobox(toolbar, textvariable=self.bin_type, values=list(BIN_TYPES),
                             state="readonly", width=18)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.status.set("Trykk «Generer bilder» for å oppdatere plassering."))

        self.image_holder = ttk.Frame(right, style="Panel.TFrame")
        self.image_holder.pack(fill="both", expand=True, pady=8)
        self.image_label = ttk.Label(self.image_holder, background=PANEL, anchor="center",
                                      text="Velg et skann til venstre.", foreground=MUTED)
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Configure>", lambda _e: self._render_photo())

        self.stats_label = ttk.Label(right, style="Stats.TLabel", justify="left", anchor="w")
        self.stats_label.pack(fill="x", pady=(0, 6))

        # bottom: actions
        actions = ttk.Frame(self.root)
        actions.pack(fill="x", padx=14, pady=(2, 6))

        def separator() -> None:
            ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)

        ttk.Button(actions, text="◀ Forrige (←)", command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(actions, text="Neste (→) ▶", command=lambda: self._step(1)).pack(side="left", padx=6)
        separator()
        ttk.Button(actions, text="Generer bilder (G)", style="Accent.TButton",
                   command=lambda: self._generate([self._selected()])).pack(side="left", padx=4)
        ttk.Button(actions, text="Generer alle (⇧G)", command=self._generate_all).pack(side="left", padx=4)
        separator()
        ttk.Button(actions, text="Åpne i 3D (O)", command=self._open_3d).pack(side="left", padx=4)
        ttk.Button(actions, text="Annotér (A)", command=self._annotate).pack(side="left", padx=4)
        ttk.Button(actions, text="Forbered rå skann (F)", command=self._prepare).pack(side="left", padx=4)

        statusbar = ttk.Frame(self.root, style="Panel.TFrame")
        statusbar.pack(fill="x", side="bottom")
        ttk.Label(statusbar, textvariable=self.status, style="Muted.TLabel").pack(side="left", padx=12, pady=4)

    # ---------- scan list ----------

    def _populate(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for stem in pipeline.list_scans():
            if pipeline.is_annotated(stem):
                status, tag = "✓ annotert", "annotated"
            elif pipeline.is_prepared(stem):
                status, tag = "klar", "prepared"
            else:
                status, tag = "rå", "raw"
            bins = pipeline.existing_bin_count(stem) if pipeline.is_prepared(stem) else ""
            self.tree.insert("", "end", iid=stem, text=stem, values=(status, bins), tags=(tag,))
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _step(self, delta: int) -> None:
        children = list(self.tree.get_children())
        stem = self._selected()
        if not children or stem is None:
            return
        idx = (children.index(stem) + delta) % len(children)
        self.tree.selection_set(children[idx])
        self.tree.focus(children[idx])
        self.tree.see(children[idx])

    # ---------- viewing ----------

    def _view_file(self) -> str:
        return dict(VIEWS)[self.view.get()]

    def _on_select(self) -> None:
        self._load_stats()
        self._show_image()

    def _show_image(self) -> None:
        stem = self._selected()
        if stem is None:
            return
        path = pipeline.preview_dir(stem) / self._view_file()
        if not path.exists():
            self._pil = None
            self.image_label.configure(image="", text="Ingen bilder ennå — trykk «Generer bilder».",
                                       foreground=MUTED)
            return
        self._pil = Image.open(path)
        self._render_photo()

    def _render_photo(self) -> None:
        if self._pil is None:
            return
        w = max(self.image_label.winfo_width(), 200)
        h = max(self.image_label.winfo_height(), 200)
        image = self._pil.copy()
        image.thumbnail((w - 8, h - 8), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo, text="")

    def _load_stats(self) -> None:
        stem = self._selected()
        stats_path = pipeline.preview_dir(stem) / "stats.json" if stem else None
        if not stats_path or not stats_path.exists():
            self.stats_label.configure(text="")
            return
        import json

        s = json.loads(stats_path.read_text(encoding="utf-8"))
        inne = "innendørs" if s.get("indoor") else "utendørs/åpent"
        line = (
            f"  {s.get('address') or stem}    |    {inne}    "
            f"|    rom {s['length_m']}×{s['width_m']} m ({s['area_m2']} m²)    "
            f"|    {s['n_existing']} kasser    |    ledig gulv {s['free_area_m2']} m²    "
            f"|    {s['n_candidates']} nye plasser ({s['bin_type']})    "
            f"|    {s['n_entrances']} inngang ({s['entrance_source']})"
        )
        self.stats_label.configure(text=line)

    # ---------- generation (background) ----------

    def _set_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status.set(message))

    def _generate_all(self) -> None:
        self._generate([s for s in pipeline.list_scans()])

    def _generate(self, stems: list[str | None]) -> None:
        stems = [s for s in stems if s]
        if not stems or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._generate_worker, args=(stems,), daemon=True).start()

    def _generate_worker(self, stems: list[str]) -> None:
        for stem in stems:
            try:
                if not pipeline.is_prepared(stem):
                    self._set_status(f"Forbereder {stem} … (kan ta et par minutter)")
                    subprocess.run(
                        [sys.executable, "-m", "src.prepare_scan", "--scan", str(pipeline.RAW_DIR / f"{stem}.zip")],
                        cwd=str(pipeline.PROJECT_ROOT),
                    )
                self._set_status(f"Analyserer {stem} …")
                pipeline.analyze_and_render(stem, self.bin_type.get())
            except Exception as error:  # noqa: BLE001 - surface any failure in the status bar
                self._set_status(f"Feil på {stem}: {error}")
        self.root.after(0, self._generate_done)

    def _generate_done(self) -> None:
        self._busy = False
        self._populate_keep_selection()
        self._on_select()
        self._set_status("Ferdig.")

    def _populate_keep_selection(self) -> None:
        stem = self._selected()
        self._populate()
        if stem and self.tree.exists(stem):
            self.tree.selection_set(stem)
            self.tree.focus(stem)

    # ---------- launching external Open3D windows ----------

    def _launch(self, module: str, *args: str) -> None:
        subprocess.Popen([sys.executable, "-m", module, *args], cwd=str(pipeline.PROJECT_ROOT))

    def _scan_paths(self) -> tuple[str, str] | None:
        stem = self._selected()
        if stem is None:
            return None
        return (
            str(pipeline.RAW_DIR / f"{stem}.zip"),
            str(pipeline.CACHE_ROOT / stem / "cloud.ply"),
        )

    def _open_3d(self) -> None:
        paths = self._scan_paths()
        if paths:
            self._launch("src.analyze_room", "--scan", paths[0], "--ply", paths[1],
                         "--place", self.bin_type.get(), "--view")
            self._set_status("Åpner 3D-visning …")

    def _annotate(self) -> None:
        stem = self._selected()
        if stem:
            self._launch("src.annotate3d", "--scan", stem)
            self._set_status("Åpner annoteringsverktøyet …")

    def _prepare(self) -> None:
        self._generate([self._selected()])

    # ---------- live refresh: watch annotation/entrance files ----------

    def _file_signature(self) -> dict[str, tuple[float, float]]:
        signature = {}
        for stem in pipeline.list_scans():
            annotation = pipeline.ANNOTATION_DIR / f"{stem}.json"
            entrance = ENTRANCE_DIR / f"{stem}.json"
            signature[stem] = (
                annotation.stat().st_mtime if annotation.exists() else 0.0,
                entrance.stat().st_mtime if entrance.exists() else 0.0,
            )
        return signature

    def _refresh_if_changed(self) -> None:
        try:
            signature = self._file_signature()
            changed = [stem for stem, value in signature.items() if self._signature.get(stem) != value]
            if changed:
                self._signature = signature
                self._populate_keep_selection()
                selected = self._selected()
                if selected in changed and not self._busy:
                    self._set_status(f"Annotering endret — oppdaterer {selected} …")
                    self._generate([selected])
        except Exception:
            pass

    def _poll(self) -> None:
        self._refresh_if_changed()
        self.root.after(1500, self._poll)

    def run(self) -> None:
        self.root.after(1500, self._poll)
        self.root.mainloop()


def main() -> None:
    Dashboard().run()


if __name__ == "__main__":
    main()
