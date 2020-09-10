import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt
import numpy as np

import fds

# locate smokeview file
root_dir = "./fds_smoke_data"
smv_filename = fds.utils.scan_directory_smv(root_dir)
print("smv file found: ", smv_filename)

# parse smokeview file for slice information
slice_infos = fds.slices.read_slice_information(os.path.join(root_dir, smv_filename))
# print all found information
print(slice_infos)

# read in meshes
meshes = fds.slices.read_meshes(os.path.join(root_dir, smv_filename))

# select matching slice
slice_label = 'ext_coef_C0.9H0.1'
sid = -1
for iis in range(len(slice_infos.slices)):
    if slice_infos[iis].label == slice_label:
        sid = iis
        print("found matching slice")
        break

if sid == -1:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

slc = slice_infos[sid]

# read in time information
slc.read_all_times(root_dir)

# read in slice data
slc.read_data(root_dir)

# map data on mesh
slc.map_data(meshes)

# get max value
max_coefficient = 0
for it in range(0, slice.times.size):
    cmax = np.max(slice.sd[it])
    max_coefficient = max(cmax, max_coefficient)

# plot slice data
for it in range(0, slice.times.size, 10):
    plt.imshow(slc.sd[it], cmap='Greys', vmax=max_coefficient,
               origin='lower', extent=slc.slice_mesh.extent)
    plt.title("time = {:.2f}".format(slc.times[it]))
    plt.colorbar(label="{} [{}]".format(slc.quantity, slc.units))
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
