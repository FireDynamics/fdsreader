"""Regression test for the DEVC csv header parsing: FDS only quotes device names that
contain a comma or space. When none of the names in a given run need quoting (which
happens to be common, e.g. plain identifiers like "TC_Door_Single"), the previous
ad-hoc `split(',"')` parser found no quote-comma boundaries at all and treated the
entire header line as a single device name, raising StopIteration."""

import numpy as np

from fdsreader.devc import Device, DeviceCollection
from fdsreader.simulation import Simulation
from fdsreader.utils import Quantity


def _make_sim(tmp_path, header_line: str, device_ids):
    csv_path = tmp_path / "test_devc.csv"
    units = ",".join(["s"] + ["C"] * len(device_ids))
    csv_path.write_text(f"{units}\n{header_line}\n0.0,1.0,2.0\n1.0,1.5,2.5\n")

    devices = {
        device_id: Device(device_id, Quantity("TEMPERATURE", "TEMP", ""), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for device_id in device_ids
    }
    devices["Time"] = Device("Time", Quantity("TIME", "TIME", "s"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    sim = object.__new__(Simulation)
    sim.devc_path = str(csv_path)
    sim._devices = devices
    sim.devices = DeviceCollection(devices.values())
    return sim


def test_devc_csv_with_no_quoted_names(tmp_path):
    # None of these names need quoting (no commas/spaces), so FDS writes them unquoted.
    sim = _make_sim(tmp_path, "Time,TC_Door_Single,BP_Door_Single", ["TC_Door_Single", "BP_Door_Single"])
    sim._load_DEVC_data()

    assert np.array_equal(sim._devices["Time"]._data, [0.0, 1.0])
    assert np.array_equal(sim._devices["TC_Door_Single"]._data, [1.0, 1.5])
    assert np.array_equal(sim._devices["BP_Door_Single"]._data, [2.0, 2.5])


def test_devc_csv_with_mixed_quoted_and_unquoted_names(tmp_path):
    # "HGL Temp" needs quoting (space), "BP_Door_Single" doesn't.
    sim = _make_sim(tmp_path, 'Time,"HGL Temp",BP_Door_Single', ["HGL Temp", "BP_Door_Single"])
    sim._load_DEVC_data()

    assert np.array_equal(sim._devices["Time"]._data, [0.0, 1.0])
    assert np.array_equal(sim._devices["HGL Temp"]._data, [1.0, 1.5])
    assert np.array_equal(sim._devices["BP_Door_Single"]._data, [2.0, 2.5])
