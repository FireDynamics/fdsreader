from fdsreader import Simulation


def test_geom():
    sim = Simulation("./geom_data")
    geom = sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]

    assert len(geom.faces) == len(geom.data[-1]) == 19816
    assert len(geom.vertices) == 40624
