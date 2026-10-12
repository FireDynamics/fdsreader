from typing import BinaryIO, Sequence, Tuple, Union

import numpy as np

from fdsreader.settings import (
    FORTRAN_BACKWARD,
    FORTRAN_DATA_TYPE_CHAR,
    FORTRAN_DATA_TYPE_FLOAT,
    FORTRAN_DATA_TYPE_INTEGER,
    FORTRAN_DATA_TYPE_UINT8,
)

_BASE_FORMAT = f"{FORTRAN_DATA_TYPE_INTEGER}, {{}}" + (f", {FORTRAN_DATA_TYPE_INTEGER}" if FORTRAN_BACKWARD else "")

_DATA_TYPES = {
    "i": FORTRAN_DATA_TYPE_INTEGER,
    "f": FORTRAN_DATA_TYPE_FLOAT,
    "c": FORTRAN_DATA_TYPE_CHAR,
    "u": FORTRAN_DATA_TYPE_UINT8,
    "{}": "{}",
}


def _get_dtype_output_format(d, n):
    """Returns the correct output format needed to create a numpy dtype depending on input."""
    if d == "c":
        return str(n)
    if isinstance(n, int | np.int32):
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
            type_combination.append(tuple(["f" + str(count)] + list(dtype[1:])))
            count += 1
    return np.dtype(type_combination)


# Commonly used datatypes
CHAR = new((("c", 1),))
INT = new((("i", 1),))
FLOAT = new((("f", 1),))
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
    arr = np.fromfile(infile, dtype=dtype, count=n)
    # Every 3rd field (starting at index 1) is an actual payload field; the fields in between
    # are the fortran record borders (block sizes) that get skipped. Extracting each payload
    # field as a whole column (vectorized) is much faster than accessing it record-by-record,
    # since indexing a structured/void scalar by field goes through numpy's generic (slow)
    # field-lookup machinery on every single access.
    columns = [arr[name] for name in arr.dtype.names[1::3]]
    if len(columns) == 1:
        return np.array([[x] for x in columns[0]], dtype=object)
    return np.array([list(row) for row in zip(*columns)], dtype=object)


def read_columns(infile: BinaryIO, dtype: np.dtype, n: int) -> list:
    """Like :func:`read`, but returns each payload field as its own fully vectorized array of
    shape ``(n, *field_shape)`` instead of boxing every record into a Python list first.

    Useful for callers that post-process every record identically (e.g. reshaping per-timestep
    data), since that post-processing can then be vectorized across all ``n`` records at once
    instead of looping over them in Python.

    :param infile: Already opened binary IO stream.
    :param dtype: Numpy dtype object.
    :param n: The number of times a dtype object should be read in from the stream.
    :returns: One array per payload field, each of shape ``(n, *field_shape)``.
    """
    arr = np.fromfile(infile, dtype=dtype, count=n)
    return [arr[name] for name in arr.dtype.names[1::3]]
