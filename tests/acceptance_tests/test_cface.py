import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def cface_sim():
    return Simulation("./cface_data_fds611")


@pytest.fixture(scope="module")
def geom(cface_sim):
    return cface_sim.geom_data.filter_by_quantity("Normal Velocity")[0]


def test_cface_geom_ind_matches_number_of_geoms(cface_sim, geom):
    geom_ind = geom.geom_ind
    assert geom_ind.shape[0] == geom.faces.shape[0]
    assert geom_ind.min() >= 0
    assert geom_ind.max() < len(cface_sim.geoms)


def test_cface_geom_ind_distinguishes_geoms(geom):
    # This fixture (thin_object_mass) has two separate &GEOM objects, so faces should be
    # attributed to both of them, not collapsed onto a single one.
    assert len(np.unique(geom.geom_ind)) == 2


def test_cface_surf_ind_matches_surfaces(cface_sim, geom):
    surf_ind = geom.surf_ind
    assert surf_ind.shape[0] == geom.faces.shape[0]
    assert surf_ind.min() >= 0
    assert surf_ind.max() < len(cface_sim.surfaces)


def test_cface_metadata_unavailable_for_legacy_gbf_format():
    legacy_sim = Simulation("./geom_data")
    legacy_geom = legacy_sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]
    with pytest.raises(AttributeError):
        legacy_geom.surf_ind
    with pytest.raises(AttributeError):
        legacy_geom.geom_ind
