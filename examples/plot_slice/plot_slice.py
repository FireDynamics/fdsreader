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
print(sc.slices[sid].times)
sc.slices[sid].readData(root_dir)
sc.slices[sid].mapData(meshes)

for it in range(0, sc.slices[sid].times.size, 20):
    plt.imshow(sc.slices[sid].sd[it], origin='lower')
    # plt.imshow(sc.slices[sid].sd[it], extent=sc.slices[sid].sm.extent, origin='lower')
    plt.colorbar()
    plt.grid(True)
    plt.savefig("single_slice_{:06d}.pdf".format(it))
    plt.clf()



# density_slices = [slices.slices[1], slices.slices[5], slices.slices[9], slices.slices[13]]
density_slices = [sc.slices[6], sc.slices[13], sc.slices[20]]
# density_slices = [sc.slices[3]]

for s in density_slices:
    s.readData(root_dir)
    s.mapData(meshes)

mesh, extent, data, mask = slice.combineSlices(density_slices)
time = density_slices[0].times

for it in range(0, time.size, 20):
    sd = np.ma.array(data[it], mask=mask)
    plt.imshow(sd, extent=extent, origin='lower')
    plt.colorbar()
    plt.savefig("slice_{:06d}.pdf".format(it))
    plt.clf()
