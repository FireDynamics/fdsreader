import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def slcf_sim():
    return Simulation("./steckler_data")


def test_slcf(slcf_sim):
    data, coordinates = slcf_sim.slices[0].to_global(masked=True, return_coordinates=True)
    assert abs(data[-1, -1, -1] - 33.311744689941406) < 1e-6
    assert abs(coordinates["x"][0] - 0.0) < 1e-6 and abs(coordinates["x"][-1] - 3.6) < 1e-6


def test_slcf_times_non_empty(slcf_sim):
    assert slcf_sim.slices[0].n_t > 0


def test_slcf_times_start_at_zero(slcf_sim):
    assert slcf_sim.slices[0].times[0] == pytest.approx(0.0)


def test_slcf_times_monotonic(slcf_sim):
    times = slcf_sim.slices[0].times
    assert np.all(np.diff(times) > 0), "Time steps must be strictly monotonically increasing"


def test_slcf_data_shape_matches_times(slcf_sim):
    slc = slcf_sim.slices[0]
    data = slc.to_global(masked=True)
    assert data.shape[0] == slc.n_t


def test_slcf_no_nan_inf(slcf_sim):
    data = slcf_sim.slices[0].to_global(masked=True)
    assert not np.isnan(data.data).any(), "NaN values found in SLCF data"
    assert not np.isinf(data.data).any(), "Inf values found in SLCF data"


def test_slcf_quantity_has_name(slcf_sim):
    assert slcf_sim.slices[0].quantity.name != ""
