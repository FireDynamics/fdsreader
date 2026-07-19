"""Regression test for GitHub issue #100: an Obstruction lost all but the last
SubObstruction whenever a MULT block created several physical instances of the same
obst_id inside a single mesh, since they were keyed only by mesh id."""

from types import SimpleNamespace

from fdsreader.bndf.obstruction import Obstruction, SubObstruction
from fdsreader.utils import Extent


def test_obstruction_keeps_multiple_subobstructions_per_mesh():
    mesh = SimpleNamespace(id="mesh1")
    obst = Obstruction("OBST-1", -3, 2, (0.0, 0.0, 0.0))

    sub1 = SubObstruction((None,) * 6, (0, 1, 0, 1, 0, 1), Extent(0, 1, 0, 1, 0, 1), mesh)
    sub2 = SubObstruction((None,) * 6, (1, 2, 0, 1, 0, 1), Extent(1, 2, 0, 1, 0, 1), mesh)

    # Mirrors what Simulation._load_obstructions does when parsing the SMV file.
    obst._subobstructions.setdefault(mesh.id, []).append(sub1)
    obst._subobstructions.setdefault(mesh.id, []).append(sub2)

    assert len(obst._all_subobstructions) == 2

    bbox = obst.bounding_box
    assert bbox.x_start == 0
    assert bbox.x_end == 2
