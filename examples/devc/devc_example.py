import matplotlib.pyplot as plt

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler")

    # Devices are looked up by the ID used in the input file
    time = sim.devices["Time"].data
    hgl_temp = sim.devices["HGL Temp"]
    print(hgl_temp.quantity, hgl_temp.unit, hgl_temp.data)

    # A device with POINTS returns a list with one device per point. FDS writes these
    # line devices once, at the end of the simulation, so each holds a single value.
    # Here: a line of thermocouples in the door, giving a temperature profile
    tc_door = sim.devices["TC_Door"]
    heights = [devc.xyz[2] for devc in tc_door]
    profile = [devc.data for devc in tc_door]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(time, hgl_temp.data)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(f"{hgl_temp.quantity.name} [{hgl_temp.unit}]")
    ax2.plot(profile, heights)
    ax2.set_xlabel("Temperature at the door, end of simulation [°C]")
    ax2.set_ylabel("Height [m]")
    plt.tight_layout()
    plt.show()

    # With pandas installed, all devices can be exported as one table
    try:
        df = sim.devices.to_pandas_dataframe()
        print(df)
    except ModuleNotFoundError:
        pass  # pandas not installed


if __name__ == "__main__":
    main()
