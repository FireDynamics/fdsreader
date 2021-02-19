import matplotlib.pyplot as plt
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_data")

    # Get the first mesh defined in fds file
    mesh = sim.meshes[0]
    # Get the first slice with quantity temperature
    slc = sim.slices.filter_by_quantity("VELOCITY")[0]

    # Get subslice that cuts through our mesh
    subslice = slc[mesh]

    # Timestep
    t = -1
    # Fill value for mask
    fill = 0
    # Get the value of supporting point in the dimension corresponding to the orientation
    supporting_point_val = subslice.data[0, 0, 0]
    # Mask the data
    mask = mesh.get_obstruction_mask_slice(slc.orientation, supporting_point_val, slc.cell_centered)
    sslc_data = np.where(mask, subslice.data[t], fill)

    # Plot the slice
    plt.imshow(sslc_data.T, origin="lower")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
