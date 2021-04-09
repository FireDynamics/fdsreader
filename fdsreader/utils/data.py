"""
Collection of internal utilities (convenience functions) for data handling.
"""
import glob
import hashlib
import os
from collections import Iterable
from typing import Tuple
import numpy as np


class Quantity:
    """Object containing information about a quantity with the corresponding label and unit.
    """
    def __init__(self, quantity: str, label: str, unit: str):
        self.label = label
        self.unit = unit
        self.quantity = quantity

    def __eq__(self, other):
        return self.quantity == other.quantity

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        return f"Quantity(label={self.label}, unit={self.unit}, quantity={self.quantity})"

    def __repr__(self):
        return f"Quantity('{self.label}')"


class Device:
    """Represents a single Device.
    """
    def __init__(self, quantity: Quantity, position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float]):
        self.quantity = quantity
        self.position = position
        self.orientation = orientation
        self.data: np.ndarray = None

    @property
    def name(self):
        return self.quantity.quantity

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Device(name={self.name}, position={self.position}, quantity={self.quantity}, mean={np.mean(self.data)})"


def create_hash(path: str):
    """Returns the md5 hash as string for the given file.
    """
    return str(hashlib.md5(open(path, 'rb').read()))


def scan_directory_smv(directory: str):
    """Scanning a directory non-recursively for smv-files.

    :param directory: The directory that will be scanned for smv files.
    :returns: A list containing the path to each smv-file found in the directory.
    """
    return glob.glob(directory + "/**/*.smv", recursive=True)


def get_smv_file(path: str):
    """Get the .smv file in a given directory.

    :param path: Either the path to the directory containing the simulation data or direct path
        to the .smv file for the simulation in case that multiple simulation output was written to
        the same directory.
    """
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        files = scan_directory_smv(path)
        if len(files) > 1:
            raise IOError("There are multiple simulations in this path!")
        elif len(files) == 0:
            raise IOError("There are no simulations in this path!")
        return files[0]
    elif os.path.isfile(path + ".smv"):
        return path + ".smv"
    else:
        raise IOError("Path is invalid!")


class FDSDataCollection:
    """(Abstract) Base class for any collection of FDS data.
    """

    def __init__(self, *elements: Iterable):
        self._elements = tuple(*elements)

    def __getitem__(self, index):
        return self._elements[index]

    def __iter__(self):
        return self._elements.__iter__()

    def __len__(self):
        return len(self._elements)

    def __contains__(self, value):
        return value in self._elements

    def __repr__(self):
        return "[" + ",\n".join(str(e) for e in self._elements) + "]"

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for element in self._elements:
            element.clear_cache()
