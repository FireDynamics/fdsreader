from fdsreader.utils import scan_directory_smv
from fdsreader import Simulation

import numpy as np


def main():
    # smv_file_paths = scan_directory_smv("../../examples/slcf/fds_data")
    smv_file_paths = scan_directory_smv("C:\\Users\\janv1\\Desktop\\fds\\sample_data\\fds_demo_jan")

    sim = Simulation(smv_file_paths[0])

    mesh = sim.meshes[0]

    # Get the second slice
    slc = sim.slices[2]
    print(slc._subslices[0].mesh)

    # Output some information about our slice
    print("Quantities:\t\t", [q.quantity for q in slc.quantities], "\nTimes[:5]:\t\t",
          slc._times[:5], "\nNumber of subslices:\t", len(slc._subslices))
    quantity = slc.quantities[0].quantity

    # Get subslice that cuts through our mesh
    subslice = slc.get_subslice(mesh)

    sslc_data = subslice.get_data(quantity)
    sslc_data[sslc_data < 21] = np.nan

    # Get actual grid coordinates from mesh
    x, y, z = np.indices(sslc_data[0].shape)
    x = mesh[0][x.flatten() + subslice.extent.x_start]
    y = mesh[1][y.flatten() + subslice.extent.y_start]
    z = mesh[2][z.flatten() + subslice.extent.z_start]


if __name__ == "__main__":
    main()
