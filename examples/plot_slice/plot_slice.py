import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt

import fds

# locate smokeview file
root_dir = "./fds_data"
smv_fn = fds.utils.scan_directory_smv(root_dir)
print("smv file found: ", smv_fn)

# parse smokeview file for slice information
slc_col = fds.slices.read_slice_information(os.path.join(root_dir, smv_fn))
# print all found information
slc_col.print()

# read in meshes
meshes = fds.slices.read_meshes(os.path.join(root_dir,smv_fn))

# select matching slice
slice_label = 'vort_y'
sid = -1
for iis in range(len(slc_col.slices)):
    if slc_col[iis].label == slice_label:
        sid = iis
        print("found matching slice")
        break

if sid == -1:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

# read in time information
# sc[sid].readTimes(root_dir)
slc_col.slices[sid].readAllTimes(root_dir)

# read in slice data
slc_col.slices[sid].readData(root_dir)

# map data on mesh
slc_col.slices[sid].mapData(meshes)

# plot slice data
for it in range(0, slc_col.slices[sid].times.size, 10):
    plt.imshow(slc_col.slices[sid].sd[it], origin='lower')
    plt.colorbar()
    plt.grid(True)
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
