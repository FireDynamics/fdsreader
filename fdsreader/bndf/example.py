from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    mesh = sim.meshes[0]

    obstruction = sim.obstructions[1]
    quantities = obstruction.quantities

    # boundary = obstruction.get_boundary_data(quantities[0])
    # obstruction_data = boundary[mesh]

    for obstruction in sim.obstructions.values():
        bndf_data = obstruction.get_boundary_data(quantities[0])
        if bndf_data is not None:
            [face.shape for face in bndf_data.faces.values()]


if __name__ == "__main__":
    main()
