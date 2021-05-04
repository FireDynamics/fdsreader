import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from fdsreader import Simulation
from fdsreader.bndf.utils import sort_patches_cartesian


def main():
    sim = Simulation("./fds_data")

    # Get first obstruction
    obst = sim.obstructions.get_nearest_obstruction((-0.8, 1, 1))
    obst = sim.obstructions[3]

    # Get all patches with orientation=1 (equals positive x-dimension)
    orientation = 1
    quantity = "Wall Temperature"
    patches = list()
    for sub_obst in obst.filter_by_orientation(orientation):
        # Get boundary data for a specific quantity
        sub_obst_data = sub_obst.get_data(quantity)
        patches.append(sub_obst_data.data[orientation])

    # Combine patches to a single face for plotting
    patches = sort_patches_cartesian(patches)

    shape_dim1 = sum([patch_row[0].shape[0] for patch_row in patches])
    shape_dim2 = sum([patch.shape[1] for patch in patches[0]])
    n_t = patches[0][0].n_t  # Number of timesteps

    face = np.empty(shape=(n_t, shape_dim1, shape_dim2))
    dim1_pos = 0
    dim2_pos = 0
    for patch_row in patches:
        d1 = patch_row[0].shape[0]
        for patch in patch_row:
            d2 = patch.shape[1]
            face[:, dim1_pos:dim1_pos + d1,
            dim2_pos:dim2_pos + d2] = patch.data
            dim2_pos += d2
        dim1_pos += d1
        dim2_pos = 0

    # Value range
    vmax = np.ceil(obst.vmax(quantity))
    # vmin = np.floor(obst.vmin(quantity))
    vmin = 40 - vmax

    # Value ticks
    ticks = [vmin+i*(vmax-vmin)/10 for i in range(11)]

    t = 0
    fig = plt.figure()
    # Initial contour plot
    cont = plt.contourf(face[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks, cmap="coolwarm")
    plt.colorbar(cont)

    # Contour plot animation
    def animate(i):
        nonlocal t, cont
        for c in cont.collections:
            c.remove()
        cont = plt.contourf(face[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks, cmap="coolwarm")
        t += 1
        return cont
    # Show animation or save it to disk
    anim = animation.FuncAnimation(fig, animate, interval=5, frames=face.shape[0]-2, repeat=False)
    # anim.save("anim.mp4", fps=25)
    plt.show()


if __name__ == "__main__":
    main()
