from fdsreader.utils import scan_directory_smv
from fdsreader.simulation import Simulation


def main():

    smv_file_paths = scan_directory_smv("../../examples/bndf/fds_data")

    sim = Simulation(smv_file_paths[0])

    print(sim.boundaries[0].sub_boundaries[0].patches[0].data)


if __name__ == "__main__":
    main()
