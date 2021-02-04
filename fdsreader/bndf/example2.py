from fdsreader import Simulation
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    wand_vorne = sim.obstructions[3]
    quantity = "Wall Temperature"
    data = wand_vorne[quantity].faces[1]

    fig = plt.figure()
    t = 0
    cont = plt.contourf(data[t].T, origin="lower")

    def animate(i):
        nonlocal t, cont
        t += 1
        for c in cont.collections:
            c.remove()
        cont = plt.contourf(data[t].T, origin="lower")
        return cont

    anim = animation.FuncAnimation(fig, animate, frames=data.shape[0]-2, repeat=False)
    anim.save("anim.mp4", fps=25)
    # plt.show()


if __name__ == "__main__":
    main()
