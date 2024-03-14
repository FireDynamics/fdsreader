from fdsreader import Simulation


def test_bndf():
    sim = Simulation("./bndf_data")
    obst = sim.obstructions.get_nearest(-0.8, 1, 1)
    face = obst.get_global_boundary_data_arrays("Wall Temperature")[1]

    assert len(face[-1]) == 68
