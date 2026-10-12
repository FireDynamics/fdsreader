"""Plot3D.to_global() must cope with meshes that have written Plot3D data for fewer time steps than
others, e.g. while FDS is still running (GitHub issue #99)."""

import numpy as np
import pytest

from fdsreader.fds_classes import Mesh
from fdsreader.pl3d import Plot3D
from fdsreader.utils import Quantity


def _make_mesh(mesh_id, x0, x1):
    coordinates = {"x": np.array([x0, x1]), "y": np.array([0.0, 1.0]), "z": np.array([0.0, 1.0])}
    return Mesh(coordinates, {"x": (x0, x1), "y": (0.0, 1.0), "z": (0.0, 1.0)}, mesh_id)


def _plot3d(times_per_mesh):
    """One Plot3D over meshes next to each other along x; each node holds 10 * mesh number + time."""
    pl3d = Plot3D("")
    for n, times in enumerate(times_per_mesh):
        mesh = _make_mesh(f"M{n}", float(n), float(n + 1))
        # Add the time steps in reverse to check that they are sorted per mesh.
        for t in reversed(times):
            pl3d._add_subplot(f"M{n}_{t}.q", t, Quantity("TEST", "test", "-"), 0, mesh)
        pl3d[mesh]._data = np.stack([np.full((2, 2, 2), 10.0 * n + t) for t in times]).astype(np.float32)
    return pl3d


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("times_b", [[0.0, 1.0], [0.0, 2.0], [1.0, 2.0]], ids=["last", "middle", "first"])
def test_to_global_fills_missing_time_steps_with_nan(masked, times_b):
    pl3d = _plot3d([[0.0, 1.0, 2.0], times_b])
    grid = pl3d.to_global(masked=masked, fill=np.nan)

    assert pl3d.times == [0.0, 1.0, 2.0]
    assert grid.shape == (3, 3, 2, 2)
    for t_idx, t in enumerate(pl3d.times):
        np.testing.assert_array_equal(grid[t_idx, 0], t)
        expected_b = 10.0 + t if t in times_b else np.nan
        np.testing.assert_array_equal(grid[t_idx, 1:], expected_b)


def test_to_global_rejects_time_steps_that_cannot_be_told_apart():
    # 100000.0 and 100000.5 are equal within np.isclose, so both meshes' first two steps share one Plot3D time.
    pl3d = _plot3d([[100000.0, 100000.5], [100000.0, 100002.0]])

    with pytest.raises(ValueError, match="cannot be told apart"):
        pl3d.to_global()
