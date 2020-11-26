from fdsreader.utils import scan_directory_smv
from simulation import Simulation


def main():
    smv_file_paths = scan_directory_smv("../../examples/plot_3d/fds_data")

    sim = Simulation(smv_file_paths[0])

    plot3d = sim.data_3d[0]

    print(plot3d._subplots[0].get_data())


if __name__ == "__main__":
    main()
