import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt

import fds.slice as slice

# locate smokeview file
root_dir = "./fds_data"
smv_fn = slice.scanDirectory(root_dir)
print("smv file found: ", smv_fn)

# parse smokeview file for slice information
sc = slice.readSliceInfos(os.path.join(root_dir, smv_fn))
# print all found information
sc.print()

# read in meshes
meshes = slice.readMeshes(os.path.join(root_dir,smv_fn))

# select matching slice
slice_label = 'vort_y'
sid = -1
for iis in range(len(sc.slices)):
    if sc[iis].label == slice_label:
        sid = iis
        print("found matching slice")
        break

if sid == -1:
    print("no slice matching label: {}".format(slice_label))
    sys.exit()

# read in time information
sc[sid].readTimes(root_dir)

# read in slice data
sc.slices[sid].readData(root_dir)

# map data on mesh
sc.slices[sid].mapData(meshes)

# plot slice data
for it in range(0, sc.slices[sid].times.size, 10):
    plt.imshow(sc.slices[sid].sd[it], origin='lower')
    plt.colorbar()
    plt.grid(True)
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
