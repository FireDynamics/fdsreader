from fdsreader import Simulation
import matplotlib.pyplot as plt
import numpy as np


def main():
    sim = Simulation("../../examples/bndf/fds_small")

    quantity = sim.obstructions[1].quantities[0]

    vmin = np.finfo(np.float32).max
    vmax = np.finfo(np.float32).min
    for obst in sim.obstructions.values():
        bndf_data = obst.get_boundary_data(quantity)
        vmin = np.min((np.min(bndf_data.lower_bounds), vmin))
        vmax = np.max((np.max(bndf_data.upper_bounds), vmax))
    vmin = np.min((0, vmin))
    vmax = np.max((25, vmax))

    plt.title(quantity.label + "in" + quantity.unit)
    if len(sim.obstructions.values()) > 1:
        fig, ax = plt.subplots(nrows=len(sim.obstructions), ncols=6)
        for i, obst in enumerate(sim.obstructions.values()):
            print("\n", obst.id, obst.extent)
            bndf_data = obst.get_boundary_data(quantity)
            if bndf_data is None:
                continue
            for j, face in enumerate((-3, -2, -1, 1, 2, 3)):
                pass
                if face in bndf_data.faces:
                    im = ax[i, j].imshow(bndf_data.faces[face][-1], vmin=vmin, vmax=vmax)
    else:
        obst = sim.obstructions[1]
        fig, ax = plt.subplots(ncols=6)
        bndf_data = obst.get_boundary_data(quantity)
        for j, face in enumerate((-3, -2, -1, 1, 2, 3)):
            pass
            if face in bndf_data.faces:
                im = ax[j].imshow(bndf_data.faces[face][-1], vmin=vmin, vmax=vmax)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()


if __name__ == "__main__":
    main()
