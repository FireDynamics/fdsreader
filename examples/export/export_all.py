import os
import fdsreader as fds
import fdsreader.export


def main():
    sim_path = "C:\\Users\\janv1\\PycharmProjects\\fdsreader\\examples\\slcf\\fds_multimesh"
    sim = fds.Simulation(sim_path)

    print(fds.export.export_sim(sim, os.path.join(sim_path, "SmokeVisIntermediate"), ordering='F'))


if __name__ == "__main__":
    main()
