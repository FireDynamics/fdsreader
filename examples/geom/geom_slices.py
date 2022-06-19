from fdsreader import Simulation


def main():
    sim = Simulation("fds_geomslices")

    for slc in sim.geomslices:
        print(slc.vertices, slc.faces, slc.data)


if __name__ == "__main__":
    main()
