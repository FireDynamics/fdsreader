import os
import fdsreader as fds
import fdsreader.export


def main():
    base_path = "C:\\Users\\janv1\\Documents\\Unreal Projects\\VRSmokeVis"
    case = "Apartment"
    sim = fds.Simulation(os.path.join(base_path, "fds_data", case))

    for slc in sim.slices:
        fds.export.export_slcf_raw(slc, os.path.join(base_path, "fds_post", case, "slices", slc.quantity.name.replace(' ', '_').lower()), 'F')


if __name__ == "__main__":
    main()
