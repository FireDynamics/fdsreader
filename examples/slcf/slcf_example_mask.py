import matplotlib.pyplot as plt
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler")

    # Get the first mesh defined in the fds file
    mesh = sim.meshes[0]
    # Get the temperature slice through the door (PBY=0)
    slc = sim.slices.filter_by_quantity("TEMPERATURE").get_nearest(y=0)

    # Get the subslice that cuts through our mesh
    subslice = slc[mesh]

    # Timestep
    t = -1
    # Mask obstructions (here: the wall with the door) with NaN, so they are not plotted
    mask = mesh.get_obstruction_mask_slice(subslice)
    sslc_data = np.where(np.squeeze(mask)[t], subslice.data[t], np.nan)

    # Plot the slice
    coords = subslice.get_coordinates()
    extent = (coords["x"][0], coords["x"][-1], coords["z"][0], coords["z"][-1])
    plt.imshow(sslc_data.T, origin="lower", extent=extent, cmap="coolwarm")
    plt.colorbar(label=f"{slc.quantity.name} [{slc.quantity.unit}]")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.show()


if __name__ == "__main__":
    main()
