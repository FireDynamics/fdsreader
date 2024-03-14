from fdsreader import Simulation


def test_slcf():
    sim = Simulation("./steckler_data")
    data, coordinates = sim.slices[0].to_global(masked=True, return_coordinates=True)
    assert abs(data[-1, -1, -1] - 33.311744689941406) < 1e-6
    assert abs(coordinates['x'][0] - 0.) < 1e-6 and abs(coordinates['x'] - 3.6) < 1e-6
