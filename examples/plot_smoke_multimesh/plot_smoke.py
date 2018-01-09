import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt
import numpy as np

import fds.slice

# locate smokeview file
root_dir = "./fds_smoke_data"
smv_fn = fds.slice.scanDirectory(root_dir)
print("smv file found: ", smv_fn)

# parse smokeview file for slice information
sc = fds.slice.readSliceInfos(os.path.join(root_dir, smv_fn))
# print all found information
sc.print()

# read in meshes
meshes = fds.slice.readMeshes(os.path.join(root_dir,smv_fn))

# select matching slice
slice_label = 'ext_coef_C0.9H0.1'
sids = []
for iis in range(len(sc.slices)):
    if sc[iis].label == slice_label:
        sids.append(iis)
        print("found matching slice with id: {}".format(iis))

if sids == []:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

#gather all slice data
slices = []
for iis in sids:
    slice = sc[iis]

    # read in time information
    slice.readAllTimes(root_dir)

    # read in slice data
    slice.readData(root_dir)

    # map data on mesh
    slice.mapData(meshes)

    slices.append(sc[iis])

mesh, extent, data, mask = fds.slice.combineSlices(slices)

# get max value
max_coefficient = 0
times = sc[sids[0]].times.size
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
