LAZY_LOAD = True

ENABLE_CACHING = True

# As the binary representation of raw data is compiler dependent, this information must be provided
# by the user
FORTRAN_DATA_TYPE_INTEGER = "<i4"  # <i4 -> 32 bit integer (little-endian)
FORTRAN_DATA_TYPE_FLOAT = "<f4"  # <f4 -> 32 bit floating point (little-endian)
FORTRAN_DATA_TYPE_CHAR = "S"  # S -> 1-byte char
FORTRAN_BACKWARD = True  # Sets weather the blocks are ended with the size of the block
