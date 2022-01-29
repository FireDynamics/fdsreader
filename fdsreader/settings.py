LAZY_LOAD = True

ENABLE_CACHING = True

DEBUG = False

IGNORE_ERRORS = False

# As the binary representation of raw data is compiler dependent, this information must be provided
# by the user
FORTRAN_DATA_TYPE_INTEGER = "<i4"  # <i4 -> 4-byte integer (little-endian)
FORTRAN_DATA_TYPE_UINT8 = "<u1"  # <u1 -> 1-byte unsigned integer (little-endian)
FORTRAN_DATA_TYPE_FLOAT = "<f4"  # <f4 -> 4-byte floating point (little-endian)
FORTRAN_DATA_TYPE_CHAR = "S"  # S -> 1-byte char
FORTRAN_BACKWARD = True  # Sets weather the blocks are ended with the size of the block
