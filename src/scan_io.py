"""Read-only access to a Polycam raw-capture zip (keyframes/images|cameras|depth|confidence|location)."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    timestamp: int
    intrinsics: np.ndarray  # 3x3, expressed at the RGB resolution
    pose_cam_to_world: np.ndarray  # 4x4
    rgb_width: int
    rgb_height: int
    center_depth: float
    blur_score: float


def _parse_camera(data: dict) -> Keyframe:
    intrinsics = np.array(
        [
            [data["fx"], 0.0, data["cx"]],
            [0.0, data["fy"], data["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    pose = np.array(
        [
            [data["t_00"], data["t_01"], data["t_02"], data["t_03"]],
            [data["t_10"], data["t_11"], data["t_12"], data["t_13"]],
            [data["t_20"], data["t_21"], data["t_22"], data["t_23"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return Keyframe(
        timestamp=int(data["timestamp"]),
        intrinsics=intrinsics,
        pose_cam_to_world=pose,
        rgb_width=int(data["width"]),
        rgb_height=int(data["height"]),
        center_depth=float(data["center_depth"]),
        blur_score=float(data.get("blur_score", float("nan"))),
    )


class ScanArchive:
    def __init__(self, zip_path: str | Path) -> None:
        self.zip_path = Path(zip_path)
        self._zip = zipfile.ZipFile(self.zip_path, "r")
        self._names = set(self._zip.namelist())
        self.timestamps = self._paired_timestamps()

    def _timestamps_in(self, folder: str, ext: str) -> set[int]:
        prefix = f"keyframes/{folder}/"
        return {
            int(Path(n).stem)
            for n in self._names
            if n.startswith(prefix) and n.endswith(ext) and not n.endswith("/")
        }

    def _paired_timestamps(self) -> list[int]:
        images = self._timestamps_in("images", ".jpg")
        cameras = self._timestamps_in("cameras", ".json")
        depth = self._timestamps_in("depth", ".png")
        confidence = self._timestamps_in("confidence", ".png")
        return sorted(images & cameras & depth & confidence)

    def keyframe(self, timestamp: int) -> Keyframe:
        data = json.loads(self._zip.read(f"keyframes/cameras/{timestamp}.json"))
        return _parse_camera(data)

    def rgb(self, timestamp: int) -> np.ndarray:
        buffer = np.frombuffer(self._zip.read(f"keyframes/images/{timestamp}.jpg"), np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth_m(self, timestamp: int) -> np.ndarray:
        buffer = np.frombuffer(self._zip.read(f"keyframes/depth/{timestamp}.png"), np.uint8)
        raw_mm = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)  # uint16, millimetres
        return raw_mm.astype(np.float32) / 1000.0

    def confidence(self, timestamp: int) -> np.ndarray:
        buffer = np.frombuffer(self._zip.read(f"keyframes/confidence/{timestamp}.png"), np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    def gps(self, timestamp: int) -> dict | None:
        name = f"keyframes/location/{timestamp}.json"
        if name in self._names:
            return json.loads(self._zip.read(name))
        return None

    def close(self) -> None:
        self._zip.close()
