import sys
sys.path.append("../..")

import os.path
import matplotlib.pyplot as plt
import numpy as np

import fds.slice as slice

root_dir = "./fds_data"
smv_fn = slice.scanDirectory(root_dir)

print("smv file found: ", smv_fn)

sc = slice.readSliceInfos(os.path.join(root_dir, smv_fn))
sc.print()

meshes = slice.readMeshes(os.path.join(root_dir,smv_fn))

for s in sc.slices: s.readTimes(root_dir)

sid = 6
sc.slices[sid].readData(root_dir)
sc.slices[sid].mapData(meshes)

for it in range(0, sc.slices[sid].times.size, 10):
    plt.imshow(sc.slices[sid].sd[it], origin='lower')
    plt.colorbar()
    plt.grid(True)
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()
