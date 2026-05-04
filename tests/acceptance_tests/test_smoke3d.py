import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def smoke_sim():
    return Simulation("./steckler_data")


@pytest.fixture(scope="module")
def smoke(smoke_sim):
    return smoke_sim.smoke_3d.get_by_quantity("Temperature")


def test_smoke3d(smoke):
    data, coordinates = smoke.to_global(masked=True, return_coordinates=True)
    assert abs(data[-1, 13, 13, 1] - 77.0) < 1e-6
    assert abs(coordinates["x"][13] - 1.3) < 1e-6


def test_smoke3d_times_non_empty(smoke):
    assert smoke.n_t > 0


def test_smoke3d_times_start_at_zero(smoke):
    assert smoke.times[0] == pytest.approx(0.0)


def test_smoke3d_times_monotonic(smoke):
    assert np.all(np.diff(smoke.times) > 0), "SMOKE3D time steps must be monotonically increasing"


def test_smoke3d_data_shape_matches_times(smoke):
    data = smoke.to_global(masked=True)
    assert data.shape[0] == smoke.n_t


def test_smoke3d_no_nan(smoke):
    data = smoke.to_global(masked=True)
    assert not np.isnan(data.data).any(), "NaN values found in SMOKE3D data"
