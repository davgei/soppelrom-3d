"""Write a coloured triangle mesh as a .glb the browser can load.

Why this exists instead of o3d.io.write_triangle_mesh(path.glb): Open3D writes the whole vertex buffer
as a base64 `data:` URI INSIDE the JSON chunk and emits no BIN chunk at all. Measured on
11807_20260727T1110_OF at 120k triangles that is 3.87 MB where the binary is 2.90 MB -- base64 costs a
flat 33% -- and Open3D's own reader then rejects the file it just wrote ("Buffer view with
offset/length (0/720000) is out of range"). A data URI is legal glTF, so three.js would probably
render it, but paying 33% and shipping a file our own tools cannot open is not a good trade when the
container is this simple.

What it writes: glTF 2.0 binary, one node, one mesh, one primitive, positions + normals + vertex
colours + indices in a real BIN chunk. Nothing else -- no textures, no animations, no scene graph.

Coordinates pass through UNCHANGED. glTF is right-handed Y-up and so is the pipeline's gravity-aligned
frame (ARKit), so a bin at plan.json's (x, z) is at three.js's (x, z) with no conversion. Getting this
wrong would put the room on its side, or mirror it, which reads as a bad scan rather than a bug.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# glTF component types
_FLOAT = 5126
_UNSIGNED_BYTE = 5121
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125

# bufferView targets
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_JSON_CHUNK = 0x4E4F534A       # 'JSON'
_BIN_CHUNK = 0x004E4942        # 'BIN\0'

# A vertex index has to fit the index type. 16-bit indices halve the biggest buffer in the file, and
# after decimation most rooms land well under this, so the type is chosen per mesh rather than always
# paying 32 bits.
_UINT16_MAX = 65535


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB 0..1 -> linear 0..1.

    Required, not cosmetic. glTF defines COLOR_0 as LINEAR, and the .ply colours are display sRGB
    straight off the camera frames. Storing them unconverted makes the renderer treat display values
    as linear and then convert them to sRGB again on output, which lifts every midtone: a grey
    concrete floor at 0.50 comes out at 0.74. Converting here means the room in the browser matches
    what place3d shows, which is the whole point of the 3D view.

    The 0.04045 kink is the sRGB piecewise curve, not a plain 2.2 gamma -- the straight-line segment
    near black is what keeps dark corners from crushing to solid black.
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


class _Builder:
    """Accumulates the BIN chunk and the bufferView/accessor tables that describe it."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _view(self, data: bytes, target: int) -> int:
        # Every bufferView starts 4-byte aligned: the spec requires it, and a misaligned view makes
        # the browser copy the data instead of using a typed-array window straight onto the buffer.
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob.extend(data)
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data),
                           "target": target})
        return len(self.views) - 1

    def attribute(self, array: np.ndarray, component: int, kind: str,
                  normalized: bool = False, bounds: bool = False) -> int:
        view = self._view(array.tobytes(), _ARRAY_BUFFER)
        accessor = {"bufferView": view, "componentType": component,
                    "count": int(array.shape[0]), "type": kind}
        if normalized:
            accessor["normalized"] = True
        if bounds:
            # min/max are REQUIRED on POSITION. Without them a viewer cannot compute the bounding
            # volume, and three.js frames the camera on nothing.
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def indices(self, array: np.ndarray) -> int:
        component = _UNSIGNED_SHORT if array.dtype == np.uint16 else _UNSIGNED_INT
        view = self._view(array.tobytes(), _ELEMENT_ARRAY_BUFFER)
        self.accessors.append({"bufferView": view, "componentType": component,
                               "count": int(array.size), "type": "SCALAR"})
        return len(self.accessors) - 1


def _pack(gltf: dict, blob: bytes) -> bytes:
    """glTF JSON + binary -> the .glb container."""
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)          # JSON pads with SPACES
    binary = bytes(blob) + b"\x00" * (-len(blob) % 4)     # BIN pads with ZEROS
    total = 12 + 8 + len(text) + 8 + len(binary)
    return b"".join([
        struct.pack("<4sII", b"glTF", 2, total),
        struct.pack("<II", len(text), _JSON_CHUNK), text,
        struct.pack("<II", len(binary), _BIN_CHUNK), binary,
    ])


def write_mesh(path: Path, vertices: np.ndarray, triangles: np.ndarray,
               colors: np.ndarray | None = None, normals: np.ndarray | None = None,
               name: str = "room") -> Path:
    """Write one coloured mesh to `path` as .glb.

    vertices (n, 3) float, triangles (m, 3) int, colors (n, 3) float in 0..1, normals (n, 3) float.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles)
    builder = _Builder()
    attributes = {"POSITION": builder.attribute(vertices, _FLOAT, "VEC3", bounds=True)}
    if normals is not None and len(normals):
        attributes["NORMAL"] = builder.attribute(
            np.ascontiguousarray(normals, dtype=np.float32), _FLOAT, "VEC3")
    if colors is not None and len(colors):
        # Vertex colours as normalised bytes: photogrammetry colour has nowhere near 8 bits of real
        # precision, and float32 would cost 4x for no visible difference.
        linear = _srgb_to_linear(np.asarray(colors, dtype=np.float64))
        rgb = np.ascontiguousarray(np.clip(linear * 255.0 + 0.5, 0, 255).astype(np.uint8))
        attributes["COLOR_0"] = builder.attribute(rgb, _UNSIGNED_BYTE, "VEC3", normalized=True)

    flat = triangles.reshape(-1)
    dtype = np.uint16 if len(vertices) <= _UINT16_MAX else np.uint32
    index_accessor = builder.indices(np.ascontiguousarray(flat, dtype=dtype))

    gltf = {
        "asset": {"version": "2.0", "generator": "soppelrom-3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name,
                    "primitives": [{"attributes": attributes, "indices": index_accessor,
                                    "material": 0}]}],
        # Unlit-ish but still shaded: roughness 1 / metallic 0 keeps the baked scan colours honest
        # (a metallic surface would tint them by the environment). doubleSided because a Poisson mesh
        # of a room is full of surfaces seen from one side only, and back-face culling would open
        # holes in the walls you look through.
        "materials": [{"name": "scan",
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "doubleSided": True}],
        "buffers": [{"byteLength": len(builder.blob) + (-len(builder.blob) % 4)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pack(gltf, builder.blob))
    return path


def write_open3d_mesh(path: Path, mesh, target_triangles: int | None = None) -> dict:
    """Write an open3d TriangleMesh as .glb, optionally decimating first.

    Returns a small summary for plan.json so the browser knows what it is about to download.
    """
    import open3d as o3d           # local: keeps open3d off the import path for glb's other caller

    triangles = len(mesh.triangles)
    if target_triangles and triangles > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    # Recomputed AFTER decimation: the collapsed vertices have different normals, and stale ones show
    # up as faceting that looks like scanner noise.
    mesh.compute_vertex_normals()
    write_mesh(
        path,
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None,
    )
    return {
        "file": path.name,
        "triangles": int(len(mesh.triangles)),
        "vertices": int(len(mesh.vertices)),
        "triangles_before": int(triangles),
        "bytes": int(path.stat().st_size),
    }
