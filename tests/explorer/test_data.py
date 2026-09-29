"""Reading a simulation that is missing some of its output.

An .smv can name output that was never written -- a run that was cut short, devices that
never fired -- and fdsreader raises a different error for each flavour. None of them
should reach a front end.
"""

import fdsreader
from fdsreader.explorer.data import build_series, device_time, device_values, list_devices


def test_devices_without_data_are_skipped(devc_without_data):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(devc_without_data))

    assert len(list_devices(sim)) >= 1          # the device is declared ...
    assert build_series(sim) == []              # ... but there is nothing to plot


def test_device_values_reports_nothing_rather_than_raising(devc_without_data):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(devc_without_data))
    for device in list_devices(sim):
        assert device_values(device) is None
        assert len(device_time(sim, device)) == 0


def test_a_simulation_with_no_output_at_all(simulation_with_nothing):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(simulation_with_nothing))
    assert build_series(sim) == []
    assert list_devices(sim) == []
    assert len(sim.slices) == 0


def test_a_normal_device_still_works(tiny_case):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(tiny_case))
    series = build_series(sim)
    assert [s["name"] for s in series] == ["TC_1", "TC_2", "HRR"]
    for entry in series:
        assert len(entry["times"]) == len(entry["values"]) > 0
