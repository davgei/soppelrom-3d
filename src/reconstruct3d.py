"""Simplified, SHAPE-FAITHFUL 3D reconstruction of a scan — a readable model instead of the point cloud.

Not a bounding box: the floor follows the TRUE floor shape (footprint.mask, so a triangular room
stays triangular) and walls are built ONLY where real wall structure was detected (scene.wall_mask),
each extruded to its measured height. Nothing is fabricated — no ceiling, no windows, and no wall on
an open side. Bins are simple solid boxes (red = existing, green = proposed) and the push-path is
drawn on the floor.

Two renderers share the same geometry (flat quads):
  * an interactive Open3D window (reuses place3d's proven camera/orbit/navigation), and
  * a headless matplotlib snapshot (previews/<stem>/reconstruction.png) for a quick still.

    .venv\\Scripts\\python.exe -m src.reconstruct3d --scan <stem>            # interactive window
    .venv\\Scripts\\python.exe -m src.reconstruct3d --scan <stem> --snapshot  # just the PNG
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from . import pipeline, place3d
from .annotations import BIN_TYPES
from .paths import PREVIEW_ROOT

FLOOR_COLOR = (0.30, 0.30, 0.32)   # asphalt grey (fallback + snapshot; interactive uses a texture)
WALL_COLOR = (0.62, 0.32, 0.26)    # brick red (fallback + snapshot; interactive uses a texture)
EXIST_COLOR = (0.85, 0.12, 0.12)
PROPOSED_COLOR = (0.18, 0.80, 0.24)
ENTRANCE_COLOR = (1.00, 0.10, 1.00)
BIN_EDGE = (0.0, 0.0, 0.0)
WALL_ALPHA = 0.55
PROPOSED_ALPHA = 0.85

EXISTING_BIN_HEIGHT = 1.2
MIN_WALL_HEIGHT = 0.4
MAX_WALL_HEIGHT = 3.5
BRICK_TILE_M = 0.8         # texture repeat length for brick walls
ASPHALT_TILE_M = 1.3       # texture repeat length for the asphalt floor
WALL_SIMPLIFY_M = 0.30     # approxPolyDP tolerance (metres) — straightens jagged wall outlines
MIN_WALL_EDGE_M = 0.15     # drop wall panels shorter than this (noise slivers)
MIN_WALL_AREA_M2 = 0.15    # drop tiny wall blobs left after cleanup
GROUND_BAND = (0.10, 0.70)  # height band above floor where a real wall's base should have points
GROUND_MIN_POINTS = 3       # a wall cell counts as grounded with at least this many low-band points
GROUND_DILATE_CELLS = 2     # tolerate sparse/occluded wall bases by growing grounded cells this much


@dataclass
class BinBox:
    corners_xz: np.ndarray      # (4, 2)
    y0: float
    y1: float
    kind: str                   # "existing" | "proposed"


@dataclass
class RoomModel:
    floor_y: float
    cell: float
    origin: np.ndarray          # (2,) world X/Z of grid cell (0, 0)
    floor_mask: np.ndarray      # bool [rows, cols] — true floor shape
    wall_mask: np.ndarray       # bool [rows, cols] — where real wall structure is
    wall_height: np.ndarray     # float [rows, cols] — structure height above floor per cell
    bins: list[BinBox]
    entrances: list[tuple[float, float]]


# ---------------------------------------------------------------------------
# geometry (everything is flat quads so both renderers share it)
# ---------------------------------------------------------------------------

def _wall_maps(scene, floor_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (max structure height above floor, #points in a low band near the floor). The low
    band tells us where structure actually reaches the ground — used to drop floating fragments."""
    fs = scene.fs
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape
    points = np.asarray(scene.mesh.vertices) if scene.mesh is not None else np.asarray(scene.aligned.points)
    col = np.floor((points[:, 0] - origin[0]) / cell).astype(int)
    row = np.floor((points[:, 2] - origin[1]) / cell).astype(int)
    height = points[:, 1] - floor_y
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows) & (height > 0.05)
    height_map = np.zeros((rows, cols))
    np.maximum.at(height_map, (row[inside], col[inside]), height[inside])
    ground = inside & (height >= GROUND_BAND[0]) & (height <= GROUND_BAND[1])
    ground_count = np.zeros((rows, cols))
    np.add.at(ground_count, (row[ground], col[ground]), 1.0)
    return height_map, ground_count


def _filter_grounded_walls(wall_mask: np.ndarray, ground_count: np.ndarray) -> np.ndarray:
    """Keep only wall cells whose structure actually reaches near the floor (points in GROUND_BAND).
    Cells that are tall-only with empty space below — overhangs, wires, fragments floating in the
    air — are dropped, so the walls stay straight and grounded. A small dilation of the grounded
    cells tolerates a sparse or occluded wall base without letting far floating bits back in."""
    if not wall_mask.any():
        return wall_mask
    grounded = (ground_count >= GROUND_MIN_POINTS).astype(np.uint8)
    if not grounded.any():
        return wall_mask
    if GROUND_DILATE_CELLS > 0:
        grounded = cv2.dilate(grounded, np.ones((3, 3), np.uint8), iterations=GROUND_DILATE_CELLS)
    return wall_mask & grounded.astype(bool)


def build_room_model(scene) -> RoomModel:
    fs = scene.fs
    floor_y = float(scene.floor_height)
    raw_wall_mask = scene.wall_mask.astype(bool) if scene.wall_mask is not None else np.zeros(fs.free.shape, bool)
    height_map, ground_count = _wall_maps(scene, floor_y)
    wall_mask = _filter_grounded_walls(raw_wall_mask, ground_count)

    bins: list[BinBox] = []
    for bx, bz, bl, bw, byaw in scene.existing:
        rect = cv2.boxPoints(((bx, bz), (bl, bw), byaw)).astype(float)
        bins.append(BinBox(rect, floor_y, floor_y + EXISTING_BIN_HEIGHT, "existing"))
    for cand in scene.result.candidates:
        rect = cv2.boxPoints(cand.rect).astype(float)
        cand_height = BIN_TYPES.get(cand.bin_type, BIN_TYPES["4-hjuls container"])[1]
        bins.append(BinBox(rect, floor_y, floor_y + cand_height, "proposed"))

    return RoomModel(
        floor_y=floor_y,
        cell=float(fs.cell),
        origin=np.asarray(fs.origin, dtype=float),
        floor_mask=fs.floor_observed.astype(bool),
        wall_mask=wall_mask,
        wall_height=height_map,
        bins=bins,
        entrances=[(float(x), float(z)) for x, z in scene.entrances],
    )


def floor_quads(model: RoomModel) -> list[np.ndarray]:
    """Fill the true floor shape with horizontal quads, merging consecutive cells per row."""
    mask, cell, origin, y = model.floor_mask, model.cell, model.origin, model.floor_y
    rows, cols = mask.shape
    quads: list[np.ndarray] = []
    for r in range(rows):
        c = 0
        while c < cols:
            if not mask[r, c]:
                c += 1
                continue
            c0 = c
            while c < cols and mask[r, c]:
                c += 1
            x0, x1 = origin[0] + c0 * cell, origin[0] + c * cell
            z0, z1 = origin[1] + r * cell, origin[1] + (r + 1) * cell
            quads.append(np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]))
    return quads


def _clean_wall_mask(mask: np.ndarray) -> np.ndarray:
    """Close small gaps then drop specks, so the traced outline is smooth rather than jagged."""
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m


def _cell_height(height_map: np.ndarray, col: int, row: int) -> float:
    rows, cols = height_map.shape
    r0, r1 = max(0, row - 1), min(rows, row + 2)
    c0, c1 = max(0, col - 1), min(cols, col + 2)
    local = height_map[r0:r1, c0:c1]
    return float(local.max()) if local.size else 0.0


def _edge_height(height_map: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """One robust (75th-percentile) height sampled ALONG the whole edge, so each wall panel gets a
    single flat top instead of a per-corner sawtooth."""
    n = max(2, int(np.hypot(float(b[0] - a[0]), float(b[1] - a[1]))))
    cols = np.linspace(a[0], b[0], n)
    rows = np.linspace(a[1], b[1], n)
    samples = [_cell_height(height_map, int(round(c)), int(round(r))) for c, r in zip(cols, rows)]
    tall = [v for v in samples if v > MIN_WALL_HEIGHT]
    value = float(np.percentile(tall, 75)) if tall else 2.2
    return float(np.clip(value, MIN_WALL_HEIGHT, MAX_WALL_HEIGHT))


def wall_quads(model: RoomModel) -> list[np.ndarray]:
    """Clean vertical wall panels where wall structure exists, following its real outline.

    The wall_mask is smoothed, traced with contours (so gaps = doorways stay open), simplified hard
    to straight edges, and each edge is extruded from the floor to ONE flat height — so walls read as
    clean planes, not a jagged sawtooth. Tiny slivers/blobs are dropped."""
    mask = _clean_wall_mask(model.wall_mask)
    if not mask.any():
        return []
    cell, origin, floor_y = model.cell, model.origin, model.floor_y
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = WALL_SIMPLIFY_M / cell
    min_area = MIN_WALL_AREA_M2 / (cell * cell)
    min_edge = MIN_WALL_EDGE_M / cell
    quads: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        poly = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        n = len(poly)
        if n < 2:
            continue
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if np.hypot(float(b[0] - a[0]), float(b[1] - a[1])) < min_edge:
                continue
            xa, za = origin[0] + (a[0] + 0.5) * cell, origin[1] + (a[1] + 0.5) * cell
            xb, zb = origin[0] + (b[0] + 0.5) * cell, origin[1] + (b[1] + 0.5) * cell
            h = _edge_height(model.wall_height, a, b)
            quads.append(np.array([
                [xa, floor_y, za],
                [xb, floor_y, zb],
                [xb, floor_y + h, zb],
                [xa, floor_y + h, za],
            ]))
    return quads


def box_quads(corners_xz: np.ndarray, y0: float, y1: float) -> list[np.ndarray]:
    bottom = [[x, y0, z] for x, z in corners_xz]
    top = [[x, y1, z] for x, z in corners_xz]
    faces = [
        [bottom[0], bottom[1], bottom[2], bottom[3]],
        [top[0], top[3], top[2], top[1]],
    ]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([bottom[i], bottom[j], top[j], top[i]])
    return [np.array(face, dtype=float) for face in faces]


def box_lineset(corners_xz: np.ndarray, y0: float, y1: float, color) -> o3d.geometry.LineSet:
    bottom = [[x, y0, z] for x, z in corners_xz]
    top = [[x, y1, z] for x, z in corners_xz]
    points = np.array(bottom + top, dtype=float)
    edges = ([[i, (i + 1) % 4] for i in range(4)]
             + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
             + [[i, 4 + i] for i in range(4)])
    lineset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),
                                   o3d.utility.Vector2iVector(np.array(edges)))
    lineset.paint_uniform_color(list(color))
    return lineset


def _brick_texture(size: int = 256) -> o3d.geometry.Image:
    """Procedural brick pattern (no external files): staggered red bricks with light mortar."""
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (196, 192, 186)  # mortar
    rng = np.random.default_rng(3)
    courses = 6
    course_h = size // courses
    mortar = max(2, size // 90)
    brick_w = size // 3
    for r in range(courses):
        y0, y1 = r * course_h + mortar, (r + 1) * course_h - mortar
        x = -(brick_w // 2 if r % 2 else 0)
        while x < size:
            tone = np.clip(np.array([150, 72, 55]) + rng.integers(-18, 19, 3), 0, 255)
            img[y0:y1, max(0, x + mortar):min(size, x + brick_w - mortar)] = tone
            x += brick_w
    return o3d.geometry.Image(np.ascontiguousarray(img))


def _asphalt_texture(size: int = 256) -> o3d.geometry.Image:
    """Procedural asphalt: dark grey with fine grain and a few lighter aggregate specks."""
    rng = np.random.default_rng(11)
    gray = np.clip(60 + rng.integers(-14, 15, (size, size, 1)), 30, 110).astype(np.uint8)
    img = np.repeat(gray, 3, axis=2)
    img[rng.random((size, size)) > 0.986] = 150
    return o3d.geometry.Image(np.ascontiguousarray(img))


def _quads_to_mesh(quads: list[np.ndarray], two_sided: bool = False,
                   tile_size: float | None = None) -> o3d.geometry.TriangleMesh:
    vertices: list[np.ndarray] = []
    triangles: list[list[int]] = []
    uvs: list[np.ndarray] = []
    for quad in quads:
        base = len(vertices)
        vertices.extend(quad)
        triangles.append([base, base + 1, base + 2])
        triangles.append([base, base + 2, base + 3])
        if tile_size:  # tile the texture at a real-world scale across the quad
            u = float(np.linalg.norm(quad[1] - quad[0])) / tile_size
            v = float(np.linalg.norm(quad[3] - quad[0])) / tile_size
            corner = [np.array([0.0, 0.0]), np.array([u, 0.0]), np.array([u, v]), np.array([0.0, v])]
            uvs += [corner[0], corner[1], corner[2], corner[0], corner[2], corner[3]]
        if two_sided:  # thin wall outlines — a duplicated, reversed back face (separate vertices so
            back = len(vertices)                          # each side gets its own correct normal)
            vertices.extend(quad)
            triangles.append([back, back + 2, back + 1])
            triangles.append([back, back + 3, back + 2])
            if tile_size:
                uvs += [corner[0], corner[2], corner[1], corner[0], corner[3], corner[2]]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    if tile_size and uvs:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uvs, dtype=float))
    return mesh


def _world_bounds(model: RoomModel) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(model.floor_mask)
    xs = model.origin[0] + np.array([cols.min(), cols.max() + 1]) * model.cell
    zs = model.origin[1] + np.array([rows.min(), rows.max() + 1]) * model.cell
    top = model.floor_y + max(float(model.wall_height.max()), 1.5)
    lo = np.array([xs[0], model.floor_y, zs[0]])
    hi = np.array([xs[1], top, zs[1]])
    return lo, hi


# ---------------------------------------------------------------------------
# interactive Open3D viewer (reuses place3d's camera / orbit / navigation)
# ---------------------------------------------------------------------------

class ReconstructionViewer(place3d.PlacementViewer):
    def _render(self, scene) -> None:
        import open3d.visualization.rendering as rendering

        self.scene.scene.clear_geometry()
        model = build_room_model(scene)

        def lit(color, alpha=1.0):
            material = rendering.MaterialRecord()
            if alpha < 1.0:
                material.shader = "defaultLitTransparency"
                material.base_color = [color[0], color[1], color[2], alpha]
            else:
                material.shader = "defaultLit"
                material.base_color = [color[0], color[1], color[2], 1.0]
            return material

        try:
            brick_img, asphalt_img = _brick_texture(), _asphalt_texture()
        except Exception:  # noqa: BLE001 - if texture build fails, fall back to flat colours
            brick_img = asphalt_img = None

        def textured(image):
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [1.0, 1.0, 1.0, 1.0]
            material.albedo_img = image
            return material

        floors = floor_quads(model)
        if floors:
            if asphalt_img is not None:
                self.scene.scene.add_geometry("floor", _quads_to_mesh(floors, tile_size=ASPHALT_TILE_M),
                                              textured(asphalt_img))
            else:
                self.scene.scene.add_geometry("floor", _quads_to_mesh(floors), lit(FLOOR_COLOR))
        walls = wall_quads(model)
        if walls:
            if brick_img is not None:
                self.scene.scene.add_geometry("walls", _quads_to_mesh(walls, two_sided=True, tile_size=BRICK_TILE_M),
                                              textured(brick_img))
            else:
                self.scene.scene.add_geometry("walls", _quads_to_mesh(walls, two_sided=True),
                                              lit(WALL_COLOR, WALL_ALPHA))

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 3.0
        for i, box in enumerate(model.bins):
            mesh = _quads_to_mesh(box_quads(box.corners_xz, box.y0, box.y1))
            color = EXIST_COLOR if box.kind == "existing" else PROPOSED_COLOR
            alpha = 1.0 if box.kind == "existing" else PROPOSED_ALPHA
            self.scene.scene.add_geometry(f"bin_{i}", mesh, lit(color, alpha))
            self.scene.scene.add_geometry(f"bin_edge_{i}", box_lineset(box.corners_xz, box.y0, box.y1, BIN_EDGE), line_mat)

        sphere_mat = rendering.MaterialRecord()
        sphere_mat.shader = "defaultUnlit"
        for i, (ex, ez) in enumerate(model.entrances):
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.15, resolution=12)
            sphere.translate([ex, model.floor_y + 0.15, ez])
            sphere.paint_uniform_color(list(ENTRANCE_COLOR))
            self.scene.scene.add_geometry(f"entrance_{i}", sphere, sphere_mat)

        overlay = self._floor_overlay(scene)
        if overlay is not None:
            omat = rendering.MaterialRecord()
            omat.shader = "defaultUnlit"
            omat.point_size = 7.0
            self.scene.scene.add_geometry("floor_overlay", overlay, omat)

        lo, hi = _world_bounds(model)
        center = (lo + hi) / 2
        span = hi - lo
        bounds = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        self.scene.setup_camera(60.0, bounds, center)
        self._cor = center.copy()
        eye = center + np.array([span[0] * 0.35 + 2.0, span[1] * 1.8 + 3.0, span[2] * 0.9 + 3.0])
        self.scene.look_at(center, eye, [0.0, 1.0, 0.0])

        self.stats_label.text = (
            f"{len(scene.result.candidates)} nye plasser · {len(scene.existing)} eksisterende · "
            f"ledig gulv {scene.fs.free_area_m2:.1f} m² · formtro (gulv + ekte vegger, uten tak)"
        )
        self.window.post_redraw()


# ---------------------------------------------------------------------------
# headless matplotlib snapshot (verification + a quick still)
# ---------------------------------------------------------------------------

def render_snapshot(scene, out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    model = build_room_model(scene)

    def to_plot(quad: np.ndarray) -> list[tuple[float, float, float]]:
        return [(p[0], p[2], p[1]) for p in quad]  # X, Z, Y(up)

    fig = plt.figure(figsize=(11, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    floors = [to_plot(q) for q in floor_quads(model)]
    if floors:
        ax.add_collection3d(Poly3DCollection(floors, facecolor=FLOOR_COLOR, edgecolor="none"))
    walls = [to_plot(q) for q in wall_quads(model)]
    if walls:
        ax.add_collection3d(Poly3DCollection(walls, facecolor=WALL_COLOR, edgecolor="#7d7d85",
                                             linewidths=0.3, alpha=WALL_ALPHA))
    for box in model.bins:
        polys = [to_plot(q) for q in box_quads(box.corners_xz, box.y0, box.y1)]
        color = EXIST_COLOR if box.kind == "existing" else PROPOSED_COLOR
        alpha = 1.0 if box.kind == "existing" else PROPOSED_ALPHA
        ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor="black",
                                             linewidths=0.9, alpha=alpha))

    reachable = scene.result.reachable
    if reachable is not None and reachable.any():
        cell, origin = scene.fs.cell, scene.fs.origin
        ys, xs = np.where(reachable)
        wx = origin[0] + (xs + 0.5) * cell
        wz = origin[1] + (ys + 0.5) * cell
        ax.scatter(wx, wz, np.full(wx.shape, model.floor_y + 0.02), c="#3070ff", s=4, alpha=0.5, depthshade=False)
    if model.entrances:
        ent = np.array(model.entrances)
        ax.scatter(ent[:, 0], ent[:, 1], np.full(len(ent), model.floor_y + 0.15),
                   c="#e0219a", s=40, depthshade=False)

    lo, hi = _world_bounds(model)
    ax.set_box_aspect((max(hi[0] - lo[0], 0.1), max(hi[2] - lo[2], 0.1), max(hi[1] - lo[1], 0.1)))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("høyde (m)")
    ax.view_init(elev=30, azim=-58)
    ax.set_title(scene.address or scene.stem)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Formtro forenklet 3D-rekonstruksjon av et skann")
    parser.add_argument("--scan", default=None, help="skann-stem som skal vises først")
    parser.add_argument("--bin-type", default="4-hjuls container", help="kassetype for forslag")
    parser.add_argument("--snapshot", action="store_true", help="skriv bare et PNG-øyeblikksbilde, ikke vindu")
    args = parser.parse_args()

    if args.snapshot:
        if not args.scan:
            raise SystemExit("--snapshot krever --scan")
        stem = Path(args.scan).stem
        scene = pipeline.compute_scene(stem, args.bin_type)
        path = render_snapshot(scene, PREVIEW_ROOT / stem / "reconstruction.png")
        print(f"Skrev {path}")
        return

    app = ReconstructionViewer(bin_type=args.bin_type)
    if args.scan:
        stem = Path(args.scan).stem
        if stem in app.scans:
            app.index = app.scans.index(stem)
            app._load()
    app.run()


if __name__ == "__main__":
    main()
