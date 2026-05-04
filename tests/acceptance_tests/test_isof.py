import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def isof_sim():
    return Simulation("./steckler_data")


@pytest.fixture(scope="module")
def isosurface(isof_sim):
    return isof_sim.isosurfaces.filter_by_quantity("TEMP")[0]


def test_isof(isosurface):
    vertices, triangles, _ = isosurface.to_global(len(isosurface.times) - 1)
    assert abs(vertices[-1][0] - 2.80595016) < 1e-6 and abs(vertices[-1][1] - 0.1) < 1e-6 and abs(vertices[-1][2] - 1.83954549) < 1e-6
    assert abs(triangles[-1][-1][0] - 4625) < 1e-6 and abs(triangles[-1][-1][1] - 4627) < 1e-6 and abs(triangles[-1][-1][2] - 4708) < 1e-6


def test_isof_times_non_empty(isosurface):
    assert len(isosurface.times) > 0


def test_isof_times_start_at_zero(isosurface):
    assert isosurface.times[0] == pytest.approx(0.0)


def test_isof_times_monotonic(isosurface):
    times = np.array(isosurface.times)
    assert np.all(np.diff(times) > 0), "ISOF time steps must be monotonically increasing"


def test_isof_to_global_returns_vertices(isosurface):
    vertices, triangles, _ = isosurface.to_global(0)
    assert vertices is not None
    assert triangles is not None
