"""Regression test for GitHub issue #88: cell-centered 3D slices failed with
"could not broadcast input array from shape (nx,ny,nz+1) into shape (nx,ny,nz)"
because SubSlice._load_data only trimmed the ghost point on the first two axes."""

import os
from types import SimpleNamespace

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader.slcf.slice import SubSlice
from fdsreader.utils import Dimension, Extent


def _write_fake_slice_file(path: str, subslice: SubSlice, raw_values: np.ndarray, time: float):
    dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((("f", raw_values.size),)))
    record = np.zeros(1, dtype=dtype_data)
    record["f1"] = time
    record["f4"] = raw_values.flatten(order="F")

    with open(path, "wb") as outfile:
        outfile.write(b"\x00" * subslice._offset)
        record.tofile(outfile)


def test_3d_cell_centered_slice_trims_ghost_points_on_every_axis(tmp_path):
    # Raw (face-centered) mesh has 3x3x4 points -> 2x2x3 cells after removing ghost points.
    dimension = Dimension(3, 3, 4)
    extent = Extent(0, 1, 0, 1, 0, 1)
    parent_slice = SimpleNamespace(cell_centered=True, n_t=1, orientation=0)
    subslice = SubSlice(parent_slice, "slice.sf", dimension, extent, mesh=None)

    raw_values = np.arange(3 * 3 * 4, dtype=np.float32).reshape((3, 3, 4), order="F")
    file_path = os.path.join(tmp_path, "slice.sf")
    _write_fake_slice_file(file_path, subslice, raw_values, time=0.0)

    data_out = np.empty((1,) + dimension.shape(cell_centered=True), dtype=np.float32)
    subslice._load_data(file_path, data_out)

    assert data_out.shape == (1, 2, 2, 3)
    assert np.array_equal(data_out[0], raw_values[1:, 1:, 1:])
