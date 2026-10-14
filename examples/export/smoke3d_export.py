import os

import fdsreader as fds
import fdsreader.export


def main():
    base_path = "."
    sim = fds.Simulation(os.path.join(base_path, "fds_data"))

    for smoke in sim.smoke_3d:
        print(smoke.quantity)
        quantity = smoke.quantity.name.replace(" ", "_").lower()
        fds.export.export_smoke_raw(smoke, os.path.join(base_path, "fds_post", "smoke", quantity), "F")


if __name__ == "__main__":
    main()
