import os

import fdsreader as fds
import fdsreader.export


def main():
    base_path = "."
    sim = fds.Simulation(os.path.join(base_path, "fds_data"))

    for slc in sim.slices:
        quantity = slc.quantity.name.replace(" ", "_").lower()
        fds.export.export_slcf_raw(slc, os.path.join(base_path, "fds_post", "slices", quantity), "F")


if __name__ == "__main__":
    main()
