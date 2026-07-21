"""Regression test for the n_vertices == 0 / n_faces == 0 code paths in the two independent
readers of FDS' geometry binary record format: GeomBoundary._load_gcf (fdsreader/geom/geometry.py)
and SubGeomSlice._load_geom_data (fdsreader/slcf/geomslice.py). Both must handle a mesh whose
cutting geometry has no vertices/faces at all - the case where the .gcf/.gsf file simply omits
the VERTS/FACES/... records after the header, per FDS' own "IF (NVERTS>0 .AND. NFACES>0)" guard.

Uses hand-built minimal binary fixtures rather than a real FDS run: this is a narrow binary-format
edge case (an empty record set), not something that needs a full simulation to exercise.
"""

import os

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader.geom.geometry import GeomBoundary
from fdsreader.slcf.geomslice import SubGeomSlice


def _write(outfile, dtype, value):
    record = np.zeros(1, dtype=dtype)
    record["f1"] = value
    record.tofile(outfile)


def _build_empty_geom_file(path):
    """Builds the common header shared by .gcf and .gsf files, with NVERTS = NFACES = 0 so no
    VERTS/FACES/... records follow - matching what FDS itself writes for an empty mesh."""
    with open(path, "wb") as f:
        _write(f, fdtype.INT, 1)  # INTEGER_ONE
        _write(f, fdtype.INT, 2)  # VERSION
        _write(f, fdtype.new((("i", 3),)), [0, 0, 1])  # 0, 0, FIRST_FRAME_STATIC
        _write(f, fdtype.FLOAT, 0.0)  # STIME
        _write(f, fdtype.new((("i", 3),)), [0, 0, 0])  # NVERTS, NFACES, NVOLS


class _StubParentSlice:
    def __init__(self, root_path):
        self._root_path = root_path


def test_load_gcf_handles_empty_mesh(tmp_path):
    gcf_path = os.path.join(tmp_path, "empty.gcf")
    _build_empty_geom_file(gcf_path)

    vertices, faces, n_faces, surf_ind, geom_ind = GeomBoundary._load_gcf(gcf_path)

    assert n_faces == 0
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert surf_ind.shape == (0,)
    assert geom_ind.shape == (0,)


def test_subgeomslice_handles_empty_mesh(tmp_path):
    gsf_path = os.path.join(tmp_path, "empty.gsf")
    _build_empty_geom_file(gsf_path)

    parent = _StubParentSlice(str(tmp_path))
    sub = SubGeomSlice(parent, "dummy.sf", "empty.gsf", extent=None, mesh=None)
    sub._load_geom_data()

    assert sub.n_verts == 0
    assert sub.n_faces == 0
    assert sub.vertices.shape == (0, 3)
    assert sub.faces.shape == (0, 3)


def test_geomslice_aggregation_with_one_empty_and_one_nonempty_submesh(tmp_path):
    """End-to-end version of the GeomSlice.vertices/.faces regression test: this one goes
    through the real SubGeomSlice._load_geom_data (via real files) instead of a hand-built stub,
    so it would have caught the original bug at its actual source."""
    from fdsreader.slcf.geomslice import GeomSlice

    empty_path = os.path.join(tmp_path, "empty.gsf")
    _build_empty_geom_file(empty_path)

    full_path = os.path.join(tmp_path, "full.gsf")
    with open(full_path, "wb") as f:
        _write(f, fdtype.INT, 1)
        _write(f, fdtype.INT, 2)
        _write(f, fdtype.new((("i", 3),)), [0, 0, 1])
        _write(f, fdtype.FLOAT, 0.0)
        _write(f, fdtype.new((("i", 3),)), [3, 1, 0])  # 3 verts, 1 face
        verts = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        _write(f, fdtype.new((("f", 9),)), verts)
        faces = np.array([1, 2, 3], dtype=np.int32)  # FDS' 1-based vertex indices
        _write(f, fdtype.new((("i", 3),)), faces)

    gs = GeomSlice.__new__(GeomSlice)
    gs._subgeomslices = {
        "mesh_empty": SubGeomSlice(_StubParentSlice(str(tmp_path)), "dummy.sf", "empty.gsf", extent=None, mesh=None),
        "mesh_full": SubGeomSlice(_StubParentSlice(str(tmp_path)), "dummy.sf", "full.gsf", extent=None, mesh=None),
    }

    vertices = gs.vertices
    faces = gs.faces

    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)
    assert faces.max() < vertices.shape[0]
