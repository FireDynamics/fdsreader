from fdsreader.utils import scan_directory_smv
from fdsreader import Simulation

import numpy as np


def main():
    smv_file_paths = scan_directory_smv("../../examples/slcf/fds_data")

    sim = Simulation(smv_file_paths[0])

    slice = sim.slices[0]
    # print(slice._subslices[0].get_data(slice.quantities[0].quantity, slice.root_path, slice.cell_centered))

    mesh = sim.meshes[0]
    print(mesh.extent)

    slice2 = slice * 2

    # np.array(slice)

    # print(slice2._subslices[0]._data[slice.quantities[0].quantity])


if __name__ == "__main__":
    main()
