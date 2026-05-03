"""Basic integration tests: load a simulation and verify core properties are accessible."""

from pathlib import Path

import pytest

from fdsreader import Simulation

CASES_DIR = Path(__file__).parent / "cases"
TESTS_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def basic_sim():
    return Simulation(str(TESTS_DIR / "test.smv"))


@pytest.fixture(scope="module")
def steckler_sim():
    return Simulation(str(CASES_DIR / "steckler_data"))


def test_basic_chid(basic_sim):
    assert basic_sim.chid == "test"


def test_basic_mesh_count(basic_sim):
    assert len(basic_sim.meshes) >= 1


def test_steckler_loads(steckler_sim):
    assert steckler_sim.chid is not None
    assert len(steckler_sim.meshes) > 0


def test_steckler_slices_accessible(steckler_sim):
    assert len(steckler_sim.slices) > 0


def test_steckler_devices_accessible(steckler_sim):
    assert len(steckler_sim.devices) > 0


def test_steckler_obstructions_accessible(steckler_sim):
    assert len(steckler_sim.obstructions) >= 0
