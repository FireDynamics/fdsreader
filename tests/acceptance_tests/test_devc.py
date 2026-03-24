import os
from fdsreader import Simulation

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_devc():
    sim = Simulation(os.path.join(TEST_DIR, "../cases/steckler_data"))
    assert abs(sim.devices["TC_Door"][0].data - 23.58) < 1e-6


def test_clear_cache_with_line_devices():
    """Test that clear_cache works with line DEVC devices (issue #104)."""
    sim = Simulation(os.path.join(TEST_DIR, "../cases/devc_data"))
    sim.clear_cache()
