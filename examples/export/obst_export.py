import os
import fdsreader as fds
import fdsreader.export


def main():
    base_path = "C:\\Users\\janv1\\Documents\\UnrealProjects\\VRSmokeVis"
    case = "Apartment"
    sim = fds.Simulation(os.path.join(base_path, "fds_data", case))

    for obst in sim.obstructions:
        print(obst.id)
        fds.export.export_obst_raw(obst, os.path.join(base_path, "fds_post", case, "obst"), 'F')


if __name__ == "__main__":
    main()
