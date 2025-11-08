"""Basic tests to ensure version compatibility."""

from fdsreader import Simulation


def test_sim():
    sim = Simulation("cases/test.smv")
    assert sim.chid == "test"
