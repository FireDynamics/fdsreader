from fdsreader import Simulation
import matplotlib.pyplot as plt
import numpy as np


def main():
    sim = Simulation("./fds_data")

    # Choose a quantity to plot
    quantity = sim.obstructions[1].quantities[0]

    # Get global minimum and maximum values
    vmin = np.finfo(np.float32).max
    vmax = np.finfo(np.float32).min
    for obst in sim.obstructions:
        bndf_data = obst.get_boundary_data(quantity)
        vmin = min(bndf_data.vmin, vmin)
        vmax = max(bndf_data.vmax, vmax)

    # Timestep, t=-1 means the last available timestep
    t = -1

    # Plot all faces of all obstructions
    fig, ax = plt.subplots(nrows=len(sim.obstructions), ncols=6)
    for i, obst in enumerate(sim.obstructions):
        bndf_data = obst.get_boundary_data(quantity)
        for face in bndf_data.faces.keys():
            idx = face + 6 if face < 0 else face-1
            im = ax[i, idx].imshow(bndf_data.faces[face][t].T, origin="lower", vmin=vmin, vmax=vmax)

    # Add global colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(quantity.quantity + " in " + quantity.unit)
    plt.show()


if __name__ == "__main__":
    main()
