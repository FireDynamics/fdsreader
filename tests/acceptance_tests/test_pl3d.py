from fdsreader import Simulation


def test_pl3d():
    sim = Simulation("./bndf_data")
    pl_t1 = sim.data_3d.get_by_quantity("Temperature")
    data, coordinates = pl_t1.to_global(masked=True, return_coordinates=True)

    assert abs(data[-1, 41, 27, 0] - 55.85966110229492) < 1e-6
    assert abs(coordinates['x'][41] - 9.25) < 1e-6
