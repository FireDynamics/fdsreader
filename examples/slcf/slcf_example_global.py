import matplotlib.pyplot as plt
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_multimesh")

    # Get the first slice
    slc = sim.slices[0]

    data = slc.to_global_nonuniform()

    # Plot the slice
    plt.imshow(data[-1].T, vmin=19, vmax=20, origin="lower")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
