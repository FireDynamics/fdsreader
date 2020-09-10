import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt
import numpy as np

import fds

# locate smokeview file
root_dir = "./fds_smoke_data"
smv_fn = fds.utils.scan_directory_smv(root_dir)
print("smv file found: ", smv_fn)

# parse smokeview file for slice information
slc_col = fds.slices.read_slice_information(os.path.join(root_dir, smv_fn))
# print all found information
slc_col.print()

# read in meshes
meshes = fds.slices.read_meshes(os.path.join(root_dir,smv_fn))

# select matching slice
slice_label = 'ext_coef_C0.9H0.1'
sids = []
for iis in range(len(slc_col.slices)):
    if slc_col[iis].label == slice_label:
        sids.append(iis)
        print("found matching slice with id: {}".format(iis))

if sids == []:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

#gather all slice data
slices = []
for iis in sids:
    slc = slc_col[iis]

    # read in time information
    slc.readAllTimes(root_dir)

    # read in slice data
    slc.readData(root_dir)

    # map data on mesh
    slc.mapData(meshes)

    slices.append(slc_col[iis])

mesh, extent, data, mask = fds.slices.combine_slices(slices)

# get max value
max_coefficient = 0
times = slc_col[sids[0]].times.size
for it in range(0, times):
    cmax = np.max(data[it])
    max_coefficient = max(cmax, max_coefficient)

# plot slice data
for it in range(0, times, 10):
    plt.imshow(data[it], cmap='Greys', vmax=max_coefficient,
               origin='lower', extent=extent)
    plt.title("time = {:.2f}".format(slice.times[it]))
    plt.colorbar(label="{} [{}]".format(slice.quantity, slice.units))
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
