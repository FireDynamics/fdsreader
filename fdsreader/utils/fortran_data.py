from typing import Sequence, BinaryIO, Union, Tuple
import numpy as np

from fdsreader.settings import FORTRAN_DATA_TYPE_CHAR, FORTRAN_DATA_TYPE_FLOAT, FORTRAN_DATA_TYPE_INTEGER, \
    FORTRAN_BACKWARD

_BASE_FORMAT = f"{FORTRAN_DATA_TYPE_INTEGER}, {{}}" + (
    f", {FORTRAN_DATA_TYPE_INTEGER}" if FORTRAN_BACKWARD else "")

_DATA_TYPES = {'i': FORTRAN_DATA_TYPE_INTEGER, 'f': FORTRAN_DATA_TYPE_FLOAT,
               'c': FORTRAN_DATA_TYPE_CHAR, '{}': "{}"}


def _get_dtype_output_format(d, n):
    """Returns the correct output format needed to create a numpy dtype depending on input.
    """
    if d == 'c':
        return str(n)
    if type(n) == int or type(n) == np.int32:
        return f"({n},)"
    return str(n)


def new_raw(data_structure: Sequence[Tuple[str, Union[int, str]]]) -> str:
    """Creates the string definition for a fortran-compliant numpy dtype to read in binary fortran data.

    :param data_structure: Tuple consisting of tuples with 2 elements each where the first element
     is a char ('i', 'f', 'c' or '{}') representing the primitive data type to be used and the
     second element an integer representing the number of times this data type was written out in
     Fortran.
    :returns: The definition string for a fortran-compliant numpy dtype with the desired structure.
    """
    return ", ".join([_get_dtype_output_format(d, n) + _DATA_TYPES[d] for d, n in data_structure])


def new(data_structure: Sequence[Tuple[str, Union[int, str]]]) -> np.dtype:
    """Creates a fortran-compliant numpy dtype to read in binary fortran data.

    :param data_structure: Tuple consisting of tuples with 2 elements each where the first element
     is a char ('i', 'f' or 'c') representing the primitive data type to be used and the second
     element an integer representing the number of times this data type was written out in Fortran.
    :returns: The newly created fortran-compliant numpy dtype with the desired structure.
    """
    return np.dtype(_BASE_FORMAT.format(new_raw(data_structure)))


def combine(*dtypes: np.dtype):
    """Combines multiple numpy dtypes into one.

    :param dtypes: An arbitrary amount of numpy dtype objects can be provided.
    :returns: The newly created numpy dtype.
    """
    count = 0
    type_combination = list()
    for types in dtypes:
        for dtype in types.descr:
            type_combination.append(tuple(['f' + str(count)] + list(dtype[1:])))
            count += 1
    return np.dtype(type_combination)


# Commonly used datatypes
CHAR = new((('c', 1),))
INT = new((('i', 1),))
FLOAT = new((('f', 1),))
# Border datatype to get the border of a fortran write
PRE_BORDER = np.dtype(FORTRAN_DATA_TYPE_INTEGER)
HAS_POST_BORDER = FORTRAN_BACKWARD


def read(infile: BinaryIO, dtype: np.dtype, n: int):
    """Convenience function to read in binary data from a file using a numpy dtype.

    :param infile: Already opened binary IO stream.
    :param dtype: Numpy dtype object.
    :param n: The number of times a dtype object should be read in from the stream.
    :returns: Read in data.
    """
    return np.array(
        [[t[i] for i in range(1, len(t), 3)] for t in np.fromfile(infile, dtype=dtype, count=n)],
        dtype=object)
