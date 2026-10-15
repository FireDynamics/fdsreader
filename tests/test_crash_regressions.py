"""Regression tests for small crashes and warnings found while writing the example notebooks."""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings
from fdsreader.evac import EvacCollection
from fdsreader.simulation import _wall_times_to_utc

CASES_DIR = Path(__file__).parent / "cases"


@pytest.mark.parametrize("case", ["bndf_data", "bndf_data_fds6100"])
@pytest.mark.parametrize("obst_id", ["Wand_r", "Golf_brandfläche"])
def test_get_coordinates_of_faces_one_node_wide(case, obst_id):
    # Both obstructions are one cell thick, so some of their faces are one node wide.
    obst = Simulation(str(CASES_DIR / case)).obstructions.get_by_id(obst_id)
    coordinates = obst.get_coordinates()
    assert set(coordinates) == set(obst.orientations)
    for orientation, coords in coordinates.items():
        dim = ("x", "y", "z")[abs(orientation) - 1]
        assert coords[dim] == pytest.approx([obst.bounding_box[dim][0 if orientation < 0 else 1]])
        for other in {"x", "y", "z"} - {dim}:
            assert np.all(np.diff(coords[other]) > 0)
            low, high = obst.bounding_box[other]
            assert np.all((coords[other] >= low - 1e-6) & (coords[other] <= high + 1e-6))


@pytest.mark.parametrize("case", ["geom_data", "geom_data_fds611"])
def test_geometry_quantities(case):
    geom_data = Simulation(str(CASES_DIR / case)).geom_data
    assert set(geom_data.quantities) == {geom.quantity for geom in geom_data}
    assert len(geom_data.quantities) > 0


def test_isosurface_quantities():
    isosurfaces = Simulation(str(CASES_DIR / "steckler_data")).isosurfaces
    assert set(isosurfaces.quantities) == {iso.quantity for iso in isosurfaces}
    assert len(isosurfaces.quantities) > 0


def test_steps_load_without_timezone_warning(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CACHING", False)  # parse the steps file, not a cache
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps = Simulation(str(CASES_DIR / "steckler_data")).steps
    # The wall-clock times are written with an offset of +01:00 and stored as UTC
    assert steps["Wall Time"][0] == np.datetime64("2024-03-14T08:39:53.199")


@pytest.mark.parametrize(
    "stamp, utc",
    [
        ("2024-03-14T09:39:53.199+01:00", "2024-03-14T08:39:53.199"),  # the format FDS writes
        ("2024-03-14T09:39:53.199-05:30", "2024-03-14T15:09:53.199"),
        (" 2024-03-14T09:39:53.199", "2024-03-14T09:39:53.199"),
        ("2024-03-14", "2024-03-14T00:00:00.000"),
        ("2024-03-14T09:39:53.199+0100", "2024-03-14T08:39:53.199"),
        ("2024-03-14T09:39:53.199+01", "2024-03-14T08:39:53.199"),
        ("2024-03-14T09:39:53.199Z", "2024-03-14T09:39:53.199"),
    ],
)
def test_wall_times_to_utc(stamp, utc):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _wall_times_to_utc([stamp])[0] == np.datetime64(utc, "ms")


def test_wall_times_to_utc_per_row_offsets():
    # A run across the switch to daylight saving time changes the offset between rows
    stamps = ["2024-03-31T01:59:59.000+01:00", "2024-03-31T03:00:01.000+02:00"]
    utc = _wall_times_to_utc(stamps)
    assert utc.dtype == np.dtype("datetime64[ms]")
    assert utc[1] - utc[0] == np.timedelta64(2000, "ms")
    assert _wall_times_to_utc([]).shape == (0,)


def test_export_of_slices_without_id(tmp_path):
    pytest.importorskip("multiprocess")
    pytest.importorskip("pathos")
    from fdsreader.export import export_slcf_raw

    slc = Simulation(str(CASES_DIR / "steckler_data")).slices[0]
    assert slc.id == ""
    meta_path = export_slcf_raw(slc, str(tmp_path))
    stem = os.path.splitext(min(subslice.filename for subslice in slc.subslices))[0]
    assert os.path.basename(meta_path) == f"slice2D-{stem.lower()}.yaml"


def test_evac_data_warns(monkeypatch):
    monkeypatch.setattr(EvacCollection, "__len__", lambda self: 1)
    with pytest.warns(FutureWarning, match="FDS\\+Evac") as record:
        Simulation(str(CASES_DIR / "steckler_data"))
    evac_warnings = [w for w in record if "FDS+Evac" in str(w.message)]
    assert evac_warnings[0].filename == __file__  # points at the caller, not into fdsreader


def test_no_evac_warning_without_evac_data():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        Simulation(str(CASES_DIR / "steckler_data"))


def test_evac_classes_without_data_do_not_break_loading(monkeypatch, tmp_path):
    import shutil

    monkeypatch.setattr(settings, "ENABLE_CACHING", False)
    case = tmp_path / "steckler"
    shutil.copytree(CASES_DIR / "steckler_data", case)
    smv = next(case.glob("*.smv"))
    # An Evac class definition without any EVA5 data block, as left by a failed Evac run
    with open(smv, "a") as f:
        f.write("CLASS_OF_HUMANS\n Adult % % \n 1.0 0.0 0.0\n 0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sim = Simulation(str(case))
    assert len(sim.evacs) == 0
