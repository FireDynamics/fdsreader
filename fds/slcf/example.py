import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fds.slcf import MeshCollection, SliceCollection
import fds.utils


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    root_dir = sys.argv[1]
    smv_fn, _ = fds.utils.scan_directory_smv(root_dir)[0]

    meshes = MeshCollection(smv_fn)

    logging.debug(meshes)

    slc_col = SliceCollection(smv_fn)

    slice1 = slc_col[0]
    slice2 = slc_col[6]

    slice1.read_data()
    slice2.read_data()

    slice1.map_data(meshes)
    slice2.map_data(meshes)

    logging.debug(slice1.slice_mesh.extent + " " + slice2.slice_mesh.extent)

    fig, axis = plt.subplots()

    cmin = min(np.amin(slice1.sd[-1]), np.amin(slice2.sd[-1]))
    cmax = max(np.amax(slice1.sd[-1]), np.amax(slice2.sd[-1]))

    im1 = axis.imshow(slice1.sd[-1], extent=slice1.slice_mesh.extent, origin='lower', vmax=cmax,
                      vmin=cmin,
                      animated=True)
    im2 = axis.imshow(slice2.sd[-1], extent=slice2.slice_mesh.extent, origin='lower', vmax=cmax,
                      vmin=cmin,
                      animated=True)

    axis.autoscale()

    plt.xlabel(slice1.slice_mesh.directions[0])
    plt.ylabel(slice1.slice_mesh.directions[1])
    plt.title(slice1)

    plt.colorbar(im1)

    iteration = 0

    def updatefig(frame, *args):
        nonlocal iteration
        iteration += 1
        if iteration >= slice1.times.size:
            iteration = 0
        im1.set_array(slice1.sd[iteration])
        im2.set_array(slice2.sd[iteration])
        return im1, im2

    _ = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)

    plt.show()

    # f = open(os.path.join(root_dir, smv_fn), 'r')
    # list_slice_summary, list_meshes, list_slcf = readGeometry(f)
    # f.close()

    # for slc in list_slice_summary:
    #    direction = 'x='
    #    if slc['dir'] == 1: direction = 'y='
    #    if slc['dir'] == 2: direction = 'z='
    #    print("available slice: ", slc['q'], 'at ', direction, slc['coord'])

    # times = readSliceTimes(os.path.join(root_dir, list_slcf[0]['fn']),
    # list_slcf[0]['n_size'])

    ## print(times)

    # time = 44.9
    # time_step = np.where(times > time)[0][0]
    # print('read in time :', time, 'at step: ', time_step)
    # data = readSliceData(os.path.join(root_dir, slice_fns[0]), time_step)


if __name__ == "__main__":
    main()