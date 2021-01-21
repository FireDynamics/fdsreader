from fdsreader import Simulation
import matplotlib.pyplot as plt


def main():
    sim = Simulation("../../examples/bndf/fds_small")

    quantities = sim.obstructions[1].quantities

    # fig, ax = plt.subplots(nrows=len(sim.obstructions), ncols=6)
    fig, ax = plt.subplots(nrows=2, ncols=6)
    for i, obst in enumerate(sim.obstructions.values()):
        bndf_data = obst.get_boundary_data(quantities[0])
        if bndf_data is None:
            continue
        for j, face in enumerate((-3, -2, -1, 1, 2, 3)):
            pass
            if face in bndf_data.faces:
                ax[i, j].imshow(bndf_data.faces[face][-2])
    # plt.colorbar(fig)
    plt.show()


if __name__ == "__main__":
    main()
