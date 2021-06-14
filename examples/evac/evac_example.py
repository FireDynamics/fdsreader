import pyvista as pv
import fdsreader as fds


def main():
    fds.settings.DEBUG = True
    fds.settings.LAZY_LOAD = False

    sim = fds.Simulation("./fds_data")

    # Get all particles with specified id
    evacs = sim.evacs

    print(evacs)

    print(evacs.eff, evacs.xyz, evacs.fed, evacs.meta, evacs[0].data)


if __name__ == "__main__":
    main()
