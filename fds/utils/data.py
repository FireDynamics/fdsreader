"""
Collection of utilities (convenience functions) for data.
"""

import os

def scan_directory_smv(directory: str):
    """
    Scanning a directory non-recursively for smv-files.
    :param directory: The directory that will be scanned for smv files.
    :return: A list containing all smv-files found in the directory.
    """
    smv_files = list()
    for file in os.listdir(directory):
        if file.endswith(".smv"):
            smv_files.append(os.path.join(directory, file))
    return smv_files
