from fdsreader import Simulation


def test_part():
    sim = Simulation("./part_data")
    particles = sim.particles["WATER PARTICLES"]
    position = particles.positions[-1]
    color_data = particles.data["PARTICLE DIAMETER"]

    assert abs(position[0] - 0.04036335) < 1e-6 and abs(position[1] - 0.05389348) < 1e-6 and abs(position[2] - 13.596354) < 1e-6
    assert abs(color_data[-1][-1] - 2.2600818) < 1e-6
