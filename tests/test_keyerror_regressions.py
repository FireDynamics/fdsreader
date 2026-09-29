"""Regression tests verifying that looking up a missing quantity/id raises a descriptive KeyError
instead of leaking a bare StopIteration from an un-defaulted next(generator) call (a bug fixed in
this session across DeviceCollection.__getitem__, Mesh.get_boundary_data, and
SubObstruction.get_data)."""

import numpy as np
import pytest

from fdsreader.bndf.obstruction import SubObstruction
from fdsreader.devc.device import Device
from fdsreader.devc.device_collection import DeviceCollection
from fdsreader.fds_classes.mesh import Mesh
from fdsreader.utils import Extent, Quantity


def test_device_collection_getitem_raises_keyerror_not_stopiteration():
    coll = DeviceCollection([Device("d1", Quantity("TEMPERATURE", "temp", "C"), (0, 0, 0), (0, 0, 1))])
    assert coll["d1"] is not None
    with pytest.raises(KeyError):
        coll["does-not-exist"]


def test_mesh_get_boundary_data_raises_keyerror_not_stopiteration():
    coords = {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "z": np.array([0.0, 1.0])}
    extents = {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}
    mesh = Mesh(coords, extents, "mesh1")
    with pytest.raises(KeyError):
        mesh.get_boundary_data("TEMPERATURE")


def test_subobstruction_get_data_raises_keyerror_not_stopiteration():
    extent = Extent(0, 1, 0, 1, 0, 1)
    sub = SubObstruction(side_surfaces=(), bound_indices=(0, 1, 0, 1, 0, 1), extent=extent, mesh=None)
    with pytest.raises(KeyError):
        sub.get_data("TEMPERATURE")
