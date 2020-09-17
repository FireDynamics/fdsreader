import matplotlib.pyplot as plt

import fds.slcf
import fds.utils

# locate smokeview file
root_dir = "./fds_data"
smv_files = fds.utils.scan_directory_smv(root_dir)
print("smv files found: ", smv_files)

# parse smokeview file for slice information
slc_col = fds.slcf.SliceCollection(smv_files[0])
# print all found information
print(slc_col)

# read in meshes
meshes = fds.slcf.MeshCollection(smv_files[0])

# select matching slice
slice_label = 'vort_y'
slc = slc_col.find_slice_by_label(slice_label)

# read in slice data
slc.read_data()

# map data on mesh
slc.map_data_onto_mesh(meshes)

# plot slice data
for it in range(0, slc.times.size, 10):
    plt.imshow(slc.sd[it], origin='lower')
    plt.colorbar()
    plt.grid(True)
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
