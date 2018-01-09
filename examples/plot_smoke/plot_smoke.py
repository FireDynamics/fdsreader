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
sid = -1
for iis in range(len(sc.slices)):
    if sc[iis].label == slice_label:
        sid = iis
        print("found matching slice")
        break

if sid == -1:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

slice = sc[sid]

# read in time information
slice.readAllTimes(root_dir)

# read in slice data
slice.readData(root_dir)

# map data on mesh
slice.mapData(meshes)

# get max value
max_coefficient = 0
for it in range(0, slice.times.size):
    cmax = np.max(slice.sd[it])
    max_coefficient = max(cmax, max_coefficient)

# plot slice data
for it in range(0, slice.times.size, 10):
    plt.imshow(slice.sd[it], cmap='Greys', vmax=max_coefficient,
               origin='lower', extent=slice.sm.extent)
    plt.title("time = {:.2f}".format(slice.times[it]))
    plt.colorbar(label="{} [{}]".format(slice.quantity, slice.units))
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
