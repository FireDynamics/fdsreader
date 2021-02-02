import fdsreader as fds


def main():
    sim = fds.Simulation("../../examples/part/fds_data")

    mesh = sim.meshes[0]

    print(sim.particles[0].positions)


if __name__ == "__main__":
    main()
