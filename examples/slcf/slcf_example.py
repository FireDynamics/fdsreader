import matplotlib.pyplot as plt

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler")

    # Get the first mesh defined in fds file
    mesh = sim.meshes[0]
    # Get the first slice with quantity temperature
    slc = sim.slices.filter_by_quantity("TEMPERATURE")[0]

    # Output some information about the slice
    print("Quantity:\t\t", slc.quantity, "\nTimes[:5]:\t\t", slc.times[:5])

    # Get subslice that cuts through our mesh
    subslice = slc[mesh]

    # Get the data for one subslice and plot it
    sslc_data = subslice.data
    plt.imshow(sslc_data[-1].T, origin="lower")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
