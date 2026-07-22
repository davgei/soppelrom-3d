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

_data_home = os.environ.get("SOPPELROM_DATA_DIR") or os.environ.get("LOCALAPPDATA")
DATA_HOME = (Path(_data_home) if _data_home else Path.home() / ".cache") / "soppelrom-3d"

# regenerable / re-downloadable — kept out of OneDrive
RAW_DIR = DATA_HOME / "raw"
CACHE_ROOT = DATA_HOME / "cache"
PREVIEW_ROOT = DATA_HOME / "previews"
DATASET_DIR = DATA_HOME / "yolo_dataset"

# human work — travels with the repo (git-tracked, tiny)
ANNOTATION_DIR = PROJECT_ROOT / "outputs" / "annotations"
ENTRANCE_DIR = PROJECT_ROOT / "outputs" / "entrances"
