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

def _dir(env: str, default: Path) -> Path:
    """Allow ONE folder to be redirected on its own, e.g. raw on a USB stick with the cache still on
    the fast local disk.

    Why this exists: raw + cache + previews for ~220 scans is 18.7 GB, so it does NOT fit on a 14.5 GB
    stick (raw alone is 11 GB and grows ~51 MB per scan). But `cache` is a BUILD ARTIFACT — 40 MB per
    scan, rebuilt from raw by prepare_scan — so the portable thing to carry between machines is raw
    (the irreplaceable source) while each machine keeps its own cache locally, where it is also much
    faster than FAT32. Set e.g. SOPPELROM_RAW_DIR=E:\\soppelrom-3d\\raw on both PCs and leave the
    rest alone."""
    value = os.environ.get(env)
    return Path(value) if value else default


# regenerable / re-downloadable — kept out of OneDrive
RAW_DIR = _dir("SOPPELROM_RAW_DIR", DATA_HOME / "raw")
CACHE_ROOT = _dir("SOPPELROM_CACHE_DIR", DATA_HOME / "cache")
PREVIEW_ROOT = _dir("SOPPELROM_PREVIEW_DIR", DATA_HOME / "previews")
DATASET_DIR = DATA_HOME / "yolo_dataset"

# Polycam's own .ply exports, deliberately kept OUT of raw/. loader.resolve_ply() picks up any
# "<stem>.ply" sitting next to the zip, and Polycam exports are Z-up while the whole pipeline assumes
# Y-up (ARKit) — so a sibling .ply would silently be read with the height axis 90 degrees off. They
# are also ~3x sparser than our own TSDF reconstruction, so they are an alternative, not an upgrade.
PLY_DIR = DATA_HOME / "ply"

# Polycam's MESH exports (GLTF/OBJ/...). Deliberately NOT in raw/: they arrive as .zip and
# pipeline.list_scans() globs raw/*.zip, so a mesh export dropped there would appear as a phantom
# scan. A mesh is what Polycam actually shows in its own app, and unlike the .ply point export it
# stays sharp when you zoom in — a point cloud renders at a fixed pixel size, so zooming only spreads
# the dots apart.
MESH_DIR = DATA_HOME / "polycam_mesh"

# human work — travels with the repo (git-tracked, tiny)
ANNOTATION_DIR = PROJECT_ROOT / "outputs" / "annotations"
ENTRANCE_DIR = PROJECT_ROOT / "outputs" / "entrances"
