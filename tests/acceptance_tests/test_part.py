import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def part_sim():
    return Simulation("./part_data")


@pytest.fixture(scope="module")
def particles(part_sim):
    return part_sim.particles["WATER PARTICLES"]


def test_part(particles):
    position = particles.positions[-1][-1]
    color_data = particles.data["PARTICLE DIAMETER"]
    assert (
        abs(position[0] + 0.04036335) < 1e-6
        and abs(position[1] - 0.05389348) < 1e-6
        and abs(position[2] - 13.596354) < 1e-6
    )
    # 321.115 (fits the ~300-2200 range of neighboring particles' diameters) is the value after
    # fixing #87, where multi-quantity particle data was scrambled by a no-op Fortran-order
    # reshape; the previously asserted 2.2600818 was itself a product of that bug.
    assert abs(color_data[-1][-1] - 321.115) < 1e-4


def test_part_positions_non_empty(particles):
    assert len(particles.positions) > 0


def test_part_positions_match_timesteps(particles):
    assert len(particles.positions) == len(particles.data["PARTICLE DIAMETER"])


def test_part_positions_finite(particles):
    for positions_at_t in particles.positions:
        arr = np.array(positions_at_t)
        assert not np.isnan(arr).any(), "NaN in Partikel-Positionen"
        assert not np.isinf(arr).any(), "Inf in Partikel-Positionen"
