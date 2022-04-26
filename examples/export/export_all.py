import os
import fdsreader as fds
import fdsreader.export


def main():
    sim_path = "C:\\Users\\janv1\\Documents\\UnrealProjects\\VRSmokeVis"
    sim = fds.Simulation(os.path.join(sim_path, "fds_data\\Apartment"))

    print(fds.export.export_sim(sim, os.path.join(sim_path, "fds_post\\Apartment"), ordering='F'))


if __name__ == "__main__":
    main()
