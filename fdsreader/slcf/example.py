from fdsreader import Simulation

import numpy as np


def main():
    sim = Simulation("C:\\Users\\janv1\\Desktop\\fds\\sample_data\\fds_demo")

    mesh = sim.meshes[0]
    # Get the second slice
    slc = sim.slices[2]

    # Output some information about our slice
    print("Quantity:\t\t", slc.quantity, "\nTimes[:5]:\t\t", slc.times[:5])
    quantity = slc.quantity

    # Get subslice that cuts through our mesh
    subslice = slc.get_subslice(mesh)

    sslc_data = subslice.data

    print(sslc_data)


if __name__ == "__main__":
    main()
