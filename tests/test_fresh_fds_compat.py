"""Structural smoke tests against simulation output freshly produced by the newest FDS
release. Only runs inside the "FDS Compatibility Check" workflow, which points
FDS_FRESH_OUTPUT_DIR at the output of tests/cases/fds_inputs/*.fds run through that release.

Unlike the acceptance tests, these don't assert exact values -- there's no golden baseline
for a version that didn't exist when this test was written. Instead they force every
lazily-parsed collection to materialize, so that a binary-format change in FDS surfaces here
as a parsing crash rather than silently going unnoticed.
"""

import os
from pathlib import Path

import pytest

from fdsreader import Simulation

FRESH_OUTPUT_DIR = os.environ.get("FDS_FRESH_OUTPUT_DIR")


def _case_dirs():
    if not FRESH_OUTPUT_DIR:
        return []
    root = Path(FRESH_OUTPUT_DIR)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


pytestmark = pytest.mark.skipif(
    not _case_dirs(),
    reason="FDS_FRESH_OUTPUT_DIR not set or empty; only runs in the FDS compatibility workflow",
)


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda p: p.name)
def test_fresh_fds_output_loads_without_crashing(case_dir):
    sim = Simulation(str(case_dir))
    assert len(sim.meshes) > 0

    for device in sim.devices:
        for d in device if isinstance(device, list) else [device]:
            _ = d.data

    for s in sim.slices:
        _ = s.to_global(masked=True)

    for gs in sim.geomslices:
        for i in range(len(gs)):
            _ = gs[i].data

    for iso in sim.isosurfaces:
        if len(iso.times) > 0:
            iso.to_global(len(iso.times) - 1)

    for p3d in sim.data_3d:
        _ = p3d.data

    for sm in sim.smoke_3d:
        _ = sm.data

    for particle in sim.particles:
        _ = particle.positions
        _ = particle.data

    for obst in sim.obstructions:
        for quantity in obst.quantities:
            obst.get_global_boundary_data_arrays(quantity)

    for geom in sim.geom_data:
        _ = geom.data
        _ = geom.faces
        _ = geom.vertices
