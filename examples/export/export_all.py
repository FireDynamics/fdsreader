import os
import fdsreader as fds
import fdsreader.export


def main():
    sim_path = "C:\\Users\\ias7_188\\RiderProjects\\VRSmokeVis"
    sim = fds.Simulation(os.path.join(sim_path, "fds_data"))

    print(fds.export.export_sim(sim, os.path.join(sim_path, "fds_post"), ordering='F'))


if __name__ == "__main__":
    main()
