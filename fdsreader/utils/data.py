"""
Collection of utilities (convenience functions) for data handling.
"""
import glob
import hashlib
import os


class Quantity:
    def __init__(self, quantity: str, label: str, unit: str):
        self.label = label
        self.unit = unit
        self.quantity = quantity

    def __eq__(self, other):
        return self.quantity == other.quantity


def create_hash(path: str):
    """Returns the md5 hash for the given file.
    """
    return hashlib.md5(open(path, 'rb').read())


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
            raise IOError()
        return files[0]
    else:
        raise IOError("Path is invalid!")
