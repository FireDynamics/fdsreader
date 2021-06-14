import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler")

    print(sim.devices["TC_Room"].data)


if __name__ == "__main__":
    main()
