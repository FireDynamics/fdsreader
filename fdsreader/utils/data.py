"""
Collection of utilities (convenience functions) for data handling.
"""
import glob


class Quantity:
    def __init__(self, quantity: str, label: str, unit: str):
        self.label = label
        self.unit = unit
        self.quantity = quantity


def scan_directory_smv(dir: str):
    """
    Scanning a directory non-recursively for smv-files.

    :param dir: The directory that will be scanned for smv files.
    :returns: A list containing the path to each smv-file found in the directory.
    """
    return glob.glob(dir + "/**/*.smv", recursive=True)
