import fdsreader as fds


def main():
    sim = fds.Simulation("../../examples/plot3d/fds_data")

    mesh = sim.meshes[0]


if __name__ == "__main__":
    main()
