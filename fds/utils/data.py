"""
Collection of utilities (convenience functions) for data.
"""

import os

# As the binary representation of raw data is compiler dependent, this information must be provided
# by the user
FDS_DATA_TYPE_INTEGER = "i4"  # i4 -> 32 bit integer (native endianness, probably little-endian)
FDS_DATA_TYPE_FLOAT = "f4"  # f4 -> 32 bit floating point (native endianness, probably little-endian)
FDS_DATA_TYPE_CHAR = "a"  # a -> 8 bit character
FDS_FORTRAN_BACKWARD = True  # sets weather the blocks are ended with the size of the block


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
