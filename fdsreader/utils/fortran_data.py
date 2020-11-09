import itertools
from typing import Sequence, BinaryIO, Union, Tuple

import numpy as np

from .settings import FORTRAN_DATA_TYPE_CHAR, FORTRAN_DATA_TYPE_FLOAT, FORTRAN_DATA_TYPE_INTEGER, FORTRAN_BACKWARD

_BASE_FORMAT = f"{FORTRAN_DATA_TYPE_INTEGER}, {{}}" + (f", {FORTRAN_DATA_TYPE_INTEGER}" if FORTRAN_BACKWARD else "")

_DATA_TYPES = {'i': FORTRAN_DATA_TYPE_INTEGER, 'f': FORTRAN_DATA_TYPE_FLOAT, 'c': FORTRAN_DATA_TYPE_CHAR, '{}': "{}"}


def new_raw(data_structure: Sequence[Tuple[str, Union[int, str]]]) -> str:
    """
    Creates the string definition for a fortran-compliant numpy dtype to read in binary fortran data.
    :param data_structure: Tuple consisting of tuples with 2 elements each where the first element is a char ('i', 'f',
     'c' or '{}') representing the primitive data type to be used and the second element an integer representing the
     number of times this data type was written out in Fortran.
    :returns: The definition string for a fortran-compliant numpy dtype with the desired structure.
    """
    return ", ".join([(str(n) if d == 'c' else f"({n},)") + _DATA_TYPES[d] for d, n in data_structure])


def new(data_structure: Sequence[Tuple[str, Union[int, str]]]) -> np.dtype:
    """
    Creates a fortran-compliant numpy dtype to read in binary fortran data.
    :param data_structure: Tuple consisting of tuples with 2 elements each where the first element is a char ('i', 'f'
     or 'c') representing the primitive data type to be used and the second element an integer representing the number
     of times this data type was written out in Fortran.
    :returns: The newly created fortran-compliant numpy dtype with the desired structure.
    """
    return np.dtype(_BASE_FORMAT.format(new_raw(data_structure)))


def combine(*dtypes: np.dtype):
    """
    Combines multiple numpy dtypes into one.
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
INT = new((('i', 1),))
FLOAT = new((('f', 1),))


# CHAR = new((('c', 1),))


def read(infile: BinaryIO, dtype: np.dtype, n: int, offset: int):
    """
    Convenience function to read in binary data from a file using a numpy dtype.
    :param infile: Already opened binary IO stream.
    :param dtype: Numpy dtype object.
    :param n: The number of times an dtype object should be read in from the stream.
    :param offset: Offset where the reader should start reading from the file.
    :returns: Read in data.
    """
    dtypes = dtype.descr
    final_dtypes = combine(
        *[np.dtype((dtypes[i][1], dtypes[i][2] if len(dtypes[i]) > 2 else (1,))) for i in range(1, len(dtypes), 3)])
    return np.array(
        [[t[i] for i in range(1, len(t), 3)] for t in np.fromfile(infile, dtype=dtype, count=n, offset=offset)])
