from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/isof/fds_data")

    mesh = sim.meshes[0]

    isosurfaces = sim.isosurfaces

    print(isosurfaces[0].vertices[mesh])


if __name__ == "__main__":
    main()
