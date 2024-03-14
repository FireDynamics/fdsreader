from fdsreader import Simulation


def test_devc():
    sim = Simulation("./steckler_data")
    assert abs(sim.devices["TC_Door"][0].data - 23.58) < 1e-6
