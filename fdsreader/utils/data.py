"""
Collection of utilities (convenience functions) for data.
"""

import os


class Quantity:
    def __init__(self, quantity: str, label: str, unit: str):
        self.label = label
        self.unit = unit
        self.quantity = quantity


def scan_directory_smv(directory: str):
    """
    Scanning a directory non-recursively for smv-files.
    :param directory: The directory that will be scanned for smv files.
    :return: A list containing the path to each smv-file found in the directory.
    """
    smv_files = list()
    for file in os.listdir(directory):
        if file.endswith(".smv"):
            smv_files.append(os.path.join(directory, file))
    return smv_files
