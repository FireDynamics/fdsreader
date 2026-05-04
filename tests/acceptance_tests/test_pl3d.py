import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def pl3d_sim():
    return Simulation("./pl3d_data")


@pytest.fixture(scope="module")
def pl3d(pl3d_sim):
    return pl3d_sim.data_3d.get_by_quantity("Temperature")


def test_pl3d(pl3d):
    data, coordinates = pl3d.to_global(masked=True, return_coordinates=True)
    assert abs(data[-1, 41, 27, 0] - 55.85966110229492) < 1e-6
    assert abs(coordinates["x"][41] - 9.25) < 1e-6


def test_pl3d_data_non_empty(pl3d):
    data = pl3d.to_global(masked=True)
    assert data.size > 0


def test_pl3d_no_nan_inf(pl3d):
    data = pl3d.to_global(masked=True)
    assert not np.isnan(data.data).any(), "NaN values found in PL3D data"
    assert not np.isinf(data.data).any(), "Inf values found in PL3D data"
