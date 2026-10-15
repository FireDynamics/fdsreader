"""Slices defined with XB that cover only part of their mesh (slcf_partial_fds611, FDS 6.11.1).

The case has two meshes (x 0..1 and 1..2, 0.1 m cells), an obstruction at x 0.6..0.9, y 0.4..0.6,
z 0..0.5, and in the same planes full reference slices and slices limited with XB: 2D at z = 0.4 and
3D over x 0.3..1.6, y 0.2..0.8, z 0.2..0.7, each node-based and cell-centred, plus a 2D slice inside
one mesh only. FDS writes the same values into both, so each partial slice must equal the matching
section of the full one.
"""

from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings

CASES_DIR = Path(__file__).parent / "cases"
PARTIAL_IDS = ["z_part", "z_part_cc", "vol_part", "vol_part_cc", "z_part_m1"]


@pytest.fixture(scope="module")
def sim():
    original = settings.ENABLE_CACHING
    settings.ENABLE_CACHING = False
    try:
        yield Simulation(str(CASES_DIR / "slcf_partial_fds611"))
    finally:
        settings.ENABLE_CACHING = original


def _full_slice(sim, partial):
    return next(
        slc
        for slc in sim.slices
        if not slc.id and slc.type == partial.type and slc.cell_centered == partial.cell_centered
    )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("slice_id", PARTIAL_IDS)
def test_partial_slice_equals_section_of_full_slice(sim, slice_id, masked):
    partial = sim.slices.get_by_id(slice_id)
    full = _full_slice(sim, partial)
    values, coords = partial.to_global(masked=masked, fill=np.nan, return_coordinates=True)
    reference, ref_coords = full.to_global(masked=masked, fill=np.nan, return_coordinates=True)

    section = []
    for dim in ("x", "y", "z") if partial.type == "3D" else ("x", "y"):
        first, last = (int(np.argmin(np.abs(ref_coords[dim] - c))) for c in (coords[dim][0], coords[dim][-1]))
        section.append(slice(first, last + 1))
    n_t = min(len(values), len(reference))  # the 3D reference slices may have one frame more
    np.testing.assert_array_equal(values[:n_t], reference[(slice(0, n_t),) + tuple(section)])
    if masked:
        assert np.isnan(values).any() == np.isnan(reference[(slice(0, n_t),) + tuple(section)]).any()


def test_partial_slice_grid_and_coordinates(sim):
    values, coords = sim.slices.get_by_id("z_part").to_global(return_coordinates=True)
    assert values.shape[1:] == (14, 7)  # nodes x 0.3..1.6, y 0.2..0.8 at 0.1 m
    np.testing.assert_allclose([coords["x"][0], coords["x"][-1]], [0.3, 1.6], atol=1e-6)
    np.testing.assert_allclose([coords["y"][0], coords["y"][-1]], [0.2, 0.8], atol=1e-6)


def test_masked_partial_slice_masks_the_obstruction(sim):
    values, coords = sim.slices.get_by_id("z_part").to_global(masked=True, fill=np.nan, return_coordinates=True)
    x, y = np.meshgrid(coords["x"], coords["y"], indexing="ij")
    inside = (x > 0.6 + 1e-6) & (x < 0.9 - 1e-6) & (y > 0.4 + 1e-6) & (y < 0.6 - 1e-6)
    assert inside.any()
    assert np.isnan(values[:, inside]).all()
    assert not np.isnan(values[:, (x > 1.0)]).any()  # no obstruction in the second mesh


@pytest.mark.parametrize("slice_id", PARTIAL_IDS)
def test_obstruction_mask_matches_subslice_data(sim, slice_id):
    for subslice in sim.slices.get_by_id(slice_id).subslices:
        mask = subslice.mesh.get_obstruction_mask_slice(subslice)
        data = subslice.data
        expected = data.shape if subslice.orientation == 0 else np.expand_dims(data, subslice.orientation).shape
        assert mask.shape == expected
