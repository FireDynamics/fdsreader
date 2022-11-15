import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import fdsreader.settings
from fdsreader import Simulation
from fdsreader.bndf.utils import sort_patches_cartesian


def main():
    sim = Simulation("./fds_data")

    # Get first obstruction
    obst = sim.obstructions.get_nearest(-0.8, 1, 1)

    # Get all patches with orientation=1 (equals positive x-dimension)
    orientation = 1
    quantity = "Wall Temperature"
    face = obst.get_global_boundary_data_arrays(quantity)[orientation]

    # Value range
    vmax = np.ceil(obst.vmax(quantity))
    vmin = np.floor(obst.vmin(quantity))

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
