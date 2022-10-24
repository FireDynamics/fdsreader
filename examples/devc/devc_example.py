import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler")

    print(sim.devices["TC_Door_Single"].data)

    try:
        df = sim.devices.to_pandas_dataframe()
        print(df)
    except ModuleNotFoundError:
        pass  # pandas not installed


if __name__ == "__main__":
    main()
