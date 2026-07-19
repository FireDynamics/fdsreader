"""Regression test for GitHub issue #75: Smoke3D.to_global() raised a ValueError
("all the input array dimensions except for the concatenation axis must match exactly")
whenever two meshes had different resolutions, because the step needed to repeat a
coarser mesh's data was only ever computed for the last axis of the previous loop
(a Python for-loop variable leak) and only when masked=True was passed."""

import numpy as np

from fdsreader.fds_classes import Mesh
from fdsreader.smoke3d import Smoke3D
from fdsreader.utils import Quantity


def _make_mesh(mesh_id, x_coords, y_coords, z_coords):
    coordinates = {"x": np.array(x_coords), "y": np.array(y_coords), "z": np.array(z_coords)}
    extents = {
        "x": (x_coords[0], x_coords[-1]),
        "y": (y_coords[0], y_coords[-1]),
        "z": (z_coords[0], z_coords[-1]),
    }
    return Mesh(coordinates, extents, mesh_id)


def test_to_global_repeats_coarser_mesh_along_every_axis_unmasked():
    # Mesh A is coarse along x (1 cell of size 1.0), mesh B is fine along x (2 cells of size 0.5),
    # placed next to each other. Both have a single cell along y and z.
    mesh_a = _make_mesh("A", [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    mesh_b = _make_mesh("B", [1.0, 1.5, 2.0], [0.0, 1.0], [0.0, 1.0])

    times = np.array([0.0])
    quantity = Quantity("TEST", "test", "-")
    smoke = Smoke3D(root_path="", times=times, quantity=quantity)

    subsmoke_a = smoke._add_subsmoke("a.s3d", mesh_a, upper_bounds=np.array([1.0]))
    subsmoke_a._data = np.ones((1, 2, 2, 2), dtype=np.float32)

    subsmoke_b = smoke._add_subsmoke("b.s3d", mesh_b, upper_bounds=np.array([1.0]))
    subsmoke_b._data = np.full((1, 3, 2, 2), 2.0, dtype=np.float32)

    # Before the fix, this raised ValueError because mesh_a's data was never repeated to
    # match mesh_b's finer x-resolution when masked=False (the default).
    grid = smoke.to_global(masked=False)

    assert grid.shape == (1, 5, 2, 2)
    # Mesh A's single coarse cell (value 1.0) must be repeated twice to match mesh B's
    # finer resolution (value 2.0), instead of only covering a single grid point.
    assert np.array_equal(grid[0, :, 0, 0], [1.0, 1.0, 2.0, 2.0, 2.0])
