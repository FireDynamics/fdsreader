"""Slices defined with XB that cover only part of their mesh (slcf_partial_fds611, FDS 6.11.1; input in
tests/cases/fds_inputs/slcf_partial.fds).

The case has three meshes along x: M1 (0..1) and M2 (1..2) with 0.1 m cells and M3 (2..3) with 0.2 m cells,
obstructions at x 0.6..0.9 and 2.4..2.8 (y 0.4..0.6), and in the same planes full reference slices and slices
limited with XB: 2D at z = 0.4 and 3D over x 0.3..2.6, y 0.2..0.8, z 0.2..0.6, each node-based and
cell-centred, plus a 2D slice inside M1 only. FDS writes the same values into both, so each partial slice must
equal the matching section of the full one, and the raw data of each mesh.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings

CASES_DIR = Path(__file__).parent / "cases"
PARTIAL_IDS = ["z_part", "z_part_cc", "vol_part", "vol_part_cc", "z_part_m1"]

# The FDS compatibility workflow runs tests/cases/fds_inputs/slcf_partial.fds through the newest FDS release
# and points FDS_FRESH_OUTPUT_DIR at the output, so the same checks run against that release too.
_CASES = [CASES_DIR / "slcf_partial_fds611"]
_FRESH = Path(os.environ.get("FDS_FRESH_OUTPUT_DIR") or "/nonexistent") / "slcf_partial"
if _FRESH.is_dir():
    _CASES.append(_FRESH)


@pytest.fixture(scope="module", params=_CASES, ids=lambda case: "fresh" if case == _FRESH else "fds611")
def sim(request):
    original = settings.ENABLE_CACHING
    settings.ENABLE_CACHING = False
    try:
        yield Simulation(str(request.param))
    finally:
        settings.ENABLE_CACHING = original


def _full_slice(sim, partial):
    return next(
        slc
        for slc in sim.slices
        if not slc.id and slc.type == partial.type and slc.cell_centered == partial.cell_centered
    )


def _grids(slc, masked):
    """The global grids of a slice with the subslices placed on each; cell-centred 2D slices give one grid per
    plane, here the 0.1 m meshes and the 0.2 m mesh, which FDS writes at different heights."""
    grids = slc.to_global(masked=masked, fill=np.nan)
    if not isinstance(grids, tuple):
        return [(grids, slc.subslices)]
    fine = [sub for sub in slc.subslices if sub.mesh.id != "M3"]
    coarse = [sub for sub in slc.subslices if sub.mesh.id == "M3"]
    return [(grids[0], fine), (grids[1], coarse)]


def _origin_and_step(subslices, dim):
    nodes = [sub.get_coordinates(ignore_cell_centered=True)[dim] for sub in subslices]
    steps = [sub.mesh.coordinates[dim][1] - sub.mesh.coordinates[dim][0] for sub in subslices]
    return min(float(n[0]) for n in nodes), min(steps)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("slice_id", PARTIAL_IDS)
def test_partial_slice_equals_section_of_full_slice(sim, slice_id, masked):
    partial = sim.slices.get_by_id(slice_id)
    full = _full_slice(sim, partial)
    dims = ("x", "y", "z") if partial.type == "3D" else ("x", "y")
    unmasked = _grids(partial, masked=False)
    for (values, partial_subs), (reference, full_subs), (written, _) in zip(
        _grids(partial, masked), _grids(full, masked), unmasked
    ):
        section = []
        for axis, dim in enumerate(dims):
            partial_origin, step = _origin_and_step(partial_subs, dim)
            full_origin, _ = _origin_and_step(full_subs, dim)
            start = int(round((partial_origin - full_origin) / step))
            section.append(slice(start, start + values.shape[axis + 1]))
        reference = reference[(slice(None),) + tuple(section)]
        # FDS may write the slice of a coarser mesh a little further than that of a finer one (vol_part_cc:
        # y up to 1.0 in M3, 0.9 in M1 and M2), so the partial grid has gaps the full slice does not have
        covered = ~np.isnan(written)
        assert covered.mean() > 0.8
        np.testing.assert_array_equal(values[covered], reference[covered])


@pytest.mark.parametrize("slice_id", ["z_part", "vol_part"])
def test_partial_slice_holds_the_raw_data_of_every_mesh(sim, slice_id):
    slc = sim.slices.get_by_id(slice_id)
    grid, coords = slc.to_global(return_coordinates=True)
    dims = ("x", "y", "z") if slc.type == "3D" else ("x", "y")
    for sub in slc.subslices:
        sub_coords = sub.get_coordinates()
        index = [np.searchsorted(coords[dim], sub_coords[dim] - 1e-6) for dim in dims]
        values = grid[(slice(None),) + np.ix_(*index)]
        if sub.mesh.id != "M3":  # the node row at the upper x face is supplied by the next mesh
            values, data = values[:, :-1], sub.data[:, :-1]
        else:
            data = sub.data
        np.testing.assert_array_equal(values, data)


def test_partial_slice_grid_and_coordinates(sim):
    values, coords = sim.slices.get_by_id("z_part").to_global(return_coordinates=True)
    assert values.shape[1:] == (24, 7)  # nodes x 0.3..2.6, y 0.2..0.8 on the 0.1 m grid
    np.testing.assert_allclose([coords["x"][0], coords["x"][-1]], [0.3, 2.6], atol=1e-6)
    np.testing.assert_allclose([coords["y"][0], coords["y"][-1]], [0.2, 0.8], atol=1e-6)


def test_masked_partial_slice_masks_the_obstruction(sim):
    values, coords = sim.slices.get_by_id("z_part").to_global(masked=True, fill=np.nan, return_coordinates=True)
    x, y = np.meshgrid(coords["x"], coords["y"], indexing="ij")
    inside = (x > 0.6 + 1e-6) & (x < 0.9 - 1e-6) & (y > 0.4 + 1e-6) & (y < 0.6 - 1e-6)
    assert inside.any()
    assert np.isnan(values[:, inside]).all()
    assert not np.isnan(values[:, (x > 1.0) & (x < 2.0)]).any()  # no obstruction in M2


@pytest.mark.parametrize("slice_id", PARTIAL_IDS)
def test_obstruction_mask_matches_subslice_data(sim, slice_id):
    for subslice in sim.slices.get_by_id(slice_id).subslices:
        mask = subslice.mesh.get_obstruction_mask_slice(subslice)
        data = subslice.data
        expected = data.shape if subslice.orientation == 0 else np.expand_dims(data, subslice.orientation).shape
        assert mask.shape == expected


def test_obstruction_mask_of_a_region_equals_the_cropped_mask_of_the_mesh(sim):
    for subslice in sim.slices.get_by_id("vol_part").subslices:
        mesh = subslice.mesh
        full = mesh.get_obstruction_mask(subslice.times)
        region = (slice(1, 4), slice(2, 5), slice(0, 3))
        np.testing.assert_array_equal(mesh.get_obstruction_mask(subslice.times, region=region), full[:, 1:4, 2:5, 0:3])
