from fdsreader import Simulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def main():
    sim = Simulation("./fds_data")

    # Filter all obstructions by quantity
    bndf = sim.obstructions.filter_by_quantity("Wall Temperature")[3]
    # Load data of a specific face
    data = bndf.faces[1]

    # Value range
    vmin = np.floor(bndf.vmin)
    vmax = np.ceil(bndf.vmax)

    # Value ticks
    ticks = [vmin+i*(vmax-vmin)/10 for i in range(11)]

    t = 0
    fig = plt.figure()
    # Initial contour plot
    cont = plt.contourf(data[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks)
    plt.colorbar(cont)

    # Contour plot animation
    def animate(i):
        nonlocal t, cont
        t += 1
        for c in cont.collections:
            c.remove()
        cont = plt.contourf(data[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks)
        return cont
    # Show animation or save it to disk
    anim = animation.FuncAnimation(fig, animate, interval=15, frames=data.shape[0]-2, repeat=False)
    # anim.save("anim.mp4", fps=25)
    plt.show()


if __name__ == "__main__":
    main()
