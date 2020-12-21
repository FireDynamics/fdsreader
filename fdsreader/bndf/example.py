from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    mesh = sim.meshes[0]

    obstruction = mesh.obstructions[2]

    boundary = sim.boundaries[0]
    subboundary = boundary[mesh]
    obstruction_data = subboundary[obstruction]

    # print(mesh.obstructions)
    # print(subboundary.obstruction_data)
    # print(obstruction_data)


if __name__ == "__main__":
    main()
