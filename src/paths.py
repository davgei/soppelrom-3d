"""Single source of truth for where data lives.

Large, regenerable or re-downloadable data — the raw scan zips, the reconstruction cache, preview
images and the YOLO training set — lives OUTSIDE the OneDrive-synced project so it is not uploaded
(OneDrive filled up) and cannot be file-locked mid-run. Override the location with the
SOPPELROM_DATA_DIR environment variable; it defaults to %LOCALAPPDATA%\\soppelrom-3d, which is
local-only and never synced.

The small, irreplaceable human work — annotations and entrances — stays inside the repo so it
travels with git / GitHub.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SOPPELROM_DATA_DIR names the PARENT folder: "soppelrom-3d" is appended to it (so D:\ becomes
# D:\soppelrom-3d). Pointing it straight at an existing soppelrom-3d folder is the natural mistake,
# so that is accepted too instead of silently nesting a second soppelrom-3d inside it.
_data_home = os.environ.get("SOPPELROM_DATA_DIR") or os.environ.get("LOCALAPPDATA")
_base = Path(_data_home) if _data_home else Path.home() / ".cache"
DATA_HOME = _base if _base.name.lower() == "soppelrom-3d" else _base / "soppelrom-3d"

# regenerable / re-downloadable — kept out of OneDrive
RAW_DIR = DATA_HOME / "raw"
CACHE_ROOT = DATA_HOME / "cache"
PREVIEW_ROOT = DATA_HOME / "previews"
DATASET_DIR = DATA_HOME / "yolo_dataset"

# Polycam's own .ply exports, deliberately kept OUT of raw/. loader.resolve_ply() picks up any
# "<stem>.ply" sitting next to the zip, and Polycam exports are Z-up while the whole pipeline assumes
# Y-up (ARKit) — so a sibling .ply would silently be read with the height axis 90 degrees off. They
# are also ~3x sparser than our own TSDF reconstruction, so they are an alternative, not an upgrade.
PLY_DIR = DATA_HOME / "ply"

# human work — travels with the repo (git-tracked, tiny)
ANNOTATION_DIR = PROJECT_ROOT / "outputs" / "annotations"
ENTRANCE_DIR = PROJECT_ROOT / "outputs" / "entrances"
