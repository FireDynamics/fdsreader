import matplotlib.pyplot as plt
import numpy as np

import fdsreaderutils
import fdsreaderslcf

# locate smokeview file
root_dir = "./fds_smoke_data"
smv_files = fds.utils.scan_directory_smv(root_dir)
print("smv files found: ", smv_files)

# parse smokeview file for slice information
slc_col = fds.slcf.SliceCollection(smv_files[0])
# print all found information
print(slc_col)

# read in meshes
meshes = fds.slcf.MeshCollection(smv_files[0])

# select matching slice
slice_label = 'ext_coef_C0.9H0.1'
slc = slc_col.find_slice_by_label(slice_label)

# read in slice data
slc.read_data()

# map data on mesh
slc.map_data_onto_mesh(meshes)

# get max value
max_coefficient = 0
for it in range(0, slc._times.size):
    cmax = np.max(slc.sd[it])
    max_coefficient = max(cmax, max_coefficient)

# plot slice data
for it in range(0, slc._times.size, 10):
    plt.imshow(slc.sd[it], cmap='Greys', vmax=max_coefficient,
               origin='lower', extent=slc.slice_mesh.dimension)
    plt.title("time = {:.2f}".format(slc._times[it]))
    plt.colorbar(label="{} [{}]".format(slc.quantity, slc.units))
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
