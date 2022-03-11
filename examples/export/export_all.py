import os
from fdsreader import Simulation
import fdsreader.export as exp


def main():
    base_path = "C:\\Users\\janv1\\Documents\\UnrealProjects\\VRSmokeVis"
    case = "Apartment"
    sim = Simulation(os.path.join(base_path, "fds_data", case))

    print(exp.export_sim(sim, os.path.join(base_path, "fds_post", case), ordering='F'))


if __name__ == "__main__":
    main()
