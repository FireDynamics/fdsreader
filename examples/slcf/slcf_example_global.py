from matplotlib import cm, pyplot as plt
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_multimesh")

    # Get the first slice
    slc = sim.slices[1]
    data = slc.to_global(masked=True, fill=np.nan)

    # Set colormap
    cmap = cm.get_cmap('coolwarm')
    # Set obsts color
    cmap.set_bad('white')

    # Plot the slice
    plt.imshow(data[-1].T, vmin=0, vmax=slc.vmax, origin="lower", cmap=cmap)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
