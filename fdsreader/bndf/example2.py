from fdsreader import Simulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    wand_vorne = sim.obstructions[3]
    quantity = "Wall Temperature"
    bndf = wand_vorne[quantity]
    data = bndf.faces[1]

    vmin = np.floor(bndf.vmin)
    vmax = np.ceil(bndf.vmax)

    print(vmin, vmax)

    ticks = [vmin+i*(vmax-vmin)/10 for i in range(11)]

    fig = plt.figure()
    t = 0
    cont = plt.contourf(data[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks)
    plt.colorbar(cont)

    def animate(i):
        nonlocal t, cont
        t += 1
        for c in cont.collections:
            c.remove()
        cont = plt.contourf(data[t].T, origin="lower", vmin=vmin, vmax=vmax, levels=ticks)
        return cont

    anim = animation.FuncAnimation(fig, animate, interval=15, frames=data.shape[0]-2, repeat=False)
    anim.save("anim.mp4", fps=25)
    # plt.show()


if __name__ == "__main__":
    main()
