"""See-through walls: hide the mesh triangles that face away from the camera.

A room is scanned from the inside, so every wall's normal points INTO the room. Drawing the whole mesh
from outside therefore shows the back of the near wall and the room reads as a closed block -- which is
what "se i 3D" looked like on a long narrow waste room: a white slab with a bin visible at one end.
Dropping the triangles whose normal points away from the eye opens the near side and leaves the far
side standing, a dollhouse view. Looking from inside, nothing is dropped, so the room stays solid.

This lives in its own module because both 3D windows need it and a second copy would drift: the
annotation tool had it, the placement viewer never got it, and that difference was the bug.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

# How far the camera must move before the cull is recomputed, in metres. Rebuilding a 200k-triangle
# mesh on every mouse-move event makes an orbit feel like glue; 2 cm is below what the eye notices on
# a room-sized model.
EYE_EPSILON = 0.02


class BackfaceCuller:
    """Holds one mesh's triangle normals and hands back the subset visible from a given eye."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh) -> None:
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.mesh = mesh
        self._normals = np.asarray(mesh.triangle_normals).copy()
        triangles = np.asarray(mesh.triangles)
        self._centers = np.asarray(mesh.vertices)[triangles].mean(axis=1)
        self._triangles = triangles
        self._last_eye: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the last camera, so the next call always rebuilds."""
        self._last_eye = None

    def culled_for(self, eye, force: bool = False) -> o3d.geometry.TriangleMesh | None:
        """The visible subset as a new mesh, or None when the camera has not moved enough to matter."""
        eye = np.asarray(eye, dtype=float).reshape(3)
        if not force and self._last_eye is not None:
            if float(np.linalg.norm(eye - self._last_eye)) < EYE_EPSILON:
                return None
        self._last_eye = eye

        # A triangle is kept when its normal points towards the eye. The dot product is with the
        # vector from the TRIANGLE to the eye, not the camera's forward axis: on a room-sized mesh the
        # two disagree badly at the edges of the view, and using the forward axis punches holes in
        # walls that are plainly facing you.
        view_dirs = eye - self._centers
        visible = np.einsum("ij,ij->i", self._normals, view_dirs) > 0.0
        culled = o3d.geometry.TriangleMesh(
            self.mesh.vertices,
            o3d.utility.Vector3iVector(self._triangles[visible]),
        )
        culled.vertex_colors = self.mesh.vertex_colors
        culled.vertex_normals = self.mesh.vertex_normals
        return culled
