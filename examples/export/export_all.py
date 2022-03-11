import os
import fdsreader as fds
import fdsreader.export


def main():
    base_path = "C:\\Users\\janv1\\Documents\\UnrealProjects\\VRSmokeVis"
    case = "Apartment"
    sim = fds.Simulation(os.path.join(base_path, "fds_data", case))

    print(fds.export.export_sim(sim, os.path.join(base_path, "fds_post", case), ordering='F'))


if __name__ == "__main__":
    main()
