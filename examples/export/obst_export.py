import os

import fdsreader as fds
import fdsreader.export


def main():
    base_path = "."
    sim = fds.Simulation(os.path.join(base_path, "fds_data"))

    for obst in sim.obstructions:
        print(obst.id)
        fds.export.export_obst_raw(obst, os.path.join(base_path, "fds_post", "obst"), "F")


if __name__ == "__main__":
    main()
