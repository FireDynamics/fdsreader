import pyvista as pv
import fdsreader as fds


def main():
    sim = fds.Simulation("fds_fed")

    # Get all particles with specified id
    evacs = sim.evacs

    print(evacs, evacs.eff, evacs.xyz, evacs.fed_grid, evacs.fed_corr, evacs.devc, evacs[0],
          evacs.get_unfiltered_positions(), evacs.get_unfiltered_data("HUMAN_SPEED"))


if __name__ == "__main__":
    main()
