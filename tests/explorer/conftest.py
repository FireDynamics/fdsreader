"""Fixtures for the explorer tests.

The cases are written from scratch rather than committed as binary fixtures, so the tests
stay small and readable and do not depend on an FDS build.
"""

import csv

import numpy as np
import pytest


@pytest.fixture
def tiny_case(tmp_path):
    """A simulation with devices and an HRR file, written out on the fly."""
    root = tmp_path / "tiny"
    root.mkdir()

    (root / "tiny.smv").write_text(
        "CHID\n tiny\n\n"
        "CSVF\n devc\n tiny_devc.csv\n\n"
        "CSVF\n hrr\n tiny_hrr.csv\n\n"
        "DEVICE\n TC_1 % TEMPERATURE\n 1.0 0.0 1.0 0.0 0.0 1.0\n\n"
        "DEVICE\n TC_2 % TEMPERATURE\n 2.0 0.0 2.0 0.0 0.0 1.0\n\n"
        "DEVICE_ACT\n 2 5.00 1\n"
    )

    times = np.arange(0.0, 10.5, 0.5)
    with open(root / "tiny_devc.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s", "C", "C"])
        writer.writerow(["Time", "TC_1", "TC_2"])
        for t, a, b in zip(times, 20 + 8 * times, 20 + 3 * times):
            writer.writerow([f"{t:.4f}", f"{a:.4f}", f"{b:.4f}"])

    with open(root / "tiny_hrr.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s", "kW"])
        writer.writerow(["Time", "HRR"])
        for t in times:
            writer.writerow([f"{t:.4f}", f"{100 * t:.4f}"])

    return root


class FakeField:
    """A 2D field standing in for a slice, so rendering can be tested without FDS files."""

    kind = "slice"
    drawable = True

    def __init__(self, nx=8, ny=16, n_t=3, quantity="TEMPERATURE", unit="C"):
        self.label = f"{quantity} [{unit}] — x = 0 m"
        self.quantity = quantity
        self.unit = unit
        self.times = np.linspace(0.0, 2.0, n_t)
        self.axes = ("y", "z")
        self.coords = {"x": np.array([0.0]),
                       "y": np.linspace(-2.0, 2.0, nx),
                       "z": np.linspace(0.0, 8.0, ny)}
        self.shape = (nx, ny)
        # a blob that rises and grows, so frames differ both in where the heat is and
        # in how hot it gets -- which is what makes a per-step colour scale visible
        yy, zz = np.meshgrid(self.coords["y"], self.coords["z"], indexing="ij")
        self._frames = [20 + (150 + 180 * k) * np.exp(-(yy ** 2) - (zz - 1 - 2 * k) ** 2)
                        for k in range(n_t)]

    def frame(self, timestep):
        return self._frames[timestep]

    def value_range(self):
        low = min(float(f.min()) for f in self._frames)
        high = max(float(f.max()) for f in self._frames)
        return low, high, False

    @property
    def cmap(self):
        return "inferno"

    def clear_cache(self):
        pass


@pytest.fixture
def fake_field():
    return FakeField()


@pytest.fixture
def fake_series():
    times = np.linspace(0.0, 4.0, 21)
    return [
        {"label": "DEVC  A — TEMPERATURE [C]", "name": "A", "times": times,
         "values": 20 + 10 * times, "quantity": "TEMPERATURE", "unit": "C",
         "source": "device", "device": None},
        {"label": "DEVC  B — TEMPERATURE [C]", "name": "B", "times": times,
         "values": 20 + 4 * times ** 2, "quantity": "TEMPERATURE", "unit": "C",
         "source": "device", "device": None},
    ]
