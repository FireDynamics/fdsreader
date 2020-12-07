from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/bndf/fds_data")

    print(sim.boundaries[0]._subboundaries[0].patches[0].data)


if __name__ == "__main__":
    main()
