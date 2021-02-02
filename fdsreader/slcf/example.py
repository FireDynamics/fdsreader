import logging

import matplotlib.pyplot as plt

import fdsreader as fds


def main():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # fds.settings.ENABLE_CACHING = False

    sim = fds.Simulation("C:\\Users\\janv1\\PycharmProjects\\fdsreader\\examples\\slcf\\fds_steckler")

    mesh = sim.meshes[0]
    # Get the second slice
    slc = sim.slices[0]

    # Output some information about our slice
    # print("Quantity:\t\t", slc.quantity, "\nTimes[:5]:\t\t", slc.times[:5])

    # Get subslice that cuts through our mesh
    subslice = slc[mesh]

    sslc_data = subslice.data

    plt.imshow(subslice.data[-1])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
