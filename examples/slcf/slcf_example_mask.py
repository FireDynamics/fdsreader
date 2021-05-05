import matplotlib.pyplot as plt
import numpy as np

import fdsreader as fds


def main():
    fds.settings.DEBUG = True

    sim = fds.Simulation("./fds_data")

    # Get the first mesh defined in fds file
    mesh = sim.meshes[0]
    # Get the Slice with name (id) "Slice1"
    slc = sim.slices.get_by_id("Slice1")

    # Get subslice that cuts through our mesh
    subslice = slc[mesh]

    # Timestep
    t = -1
    # Fill value for mask
    fill = 0
    # Mask the data
    mask = mesh.get_obstruction_mask_slice(subslice)
    sslc_data = np.where(mask[t], subslice.data[t], fill)

    # Plot the slice
    plt.imshow(sslc_data.T, origin="lower")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
