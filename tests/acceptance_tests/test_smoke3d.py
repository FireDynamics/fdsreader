from fdsreader import Simulation


def test_smoke3d():
    sim = Simulation("./steckler_data")
    smoke = sim.smoke_3d.get_by_quantity("Temperature")
    data, coordinates = smoke.to_global(masked=True, return_coordinates=True)

    assert abs(data[-1, 13, 13, 1] - 77.) < 1e-6
    assert abs(coordinates['x'][13] - 1.3) < 1e-6
