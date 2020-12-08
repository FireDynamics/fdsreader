from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    mesh = sim.meshes[0]

    obstruction = mesh.obstructions[2]
    print(mesh.obstructions)

    boundary = sim.boundaries[0]
    subboundary = boundary[mesh]
    print(subboundary.obstruction_data)
    obstruction_data = subboundary[obstruction]
    print(obstruction_data)


if __name__ == "__main__":
    main()
