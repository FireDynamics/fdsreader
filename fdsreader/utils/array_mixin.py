import logging
from copy import deepcopy
from typing import Any, Dict, Iterable

import numpy as np


class NumpyArrayMixin(np.lib.mixins.NDArrayOperatorsMixin):
    """Shared numpy-array-protocol implementation (mean, std, __array__, __array_ufunc__,
    __array_function__) for container classes (Slice, GeomSlice, Smoke3D, Plot3D) that wrap
    multiple per-mesh sub-items, each exposing its raw data via a `.data` property backed by a
    `._data` attribute.

    Subclasses must:
      * Override `_array_subitems()` to return an iterable of their per-mesh sub-items.
      * Keep their own module-level `_HANDLED_FUNCTIONS = {}` dict and `implements()` decorator
        (this stays per-module on purpose: `@implements(np.mean)` decorates a method from inside
        the class body, before the class object exists, so a classmethod- or
        __init_subclass__-based registry can't work here - the class body can't reference `cls`
        while it's still executing). Point a class attribute at that dict, e.g.:

            _HANDLED_FUNCTIONS = {}

            def implements(np_function):
                def decorator(func):
                    _HANDLED_FUNCTIONS[np_function] = func
                    return func
                return decorator

            class Slice(NumpyArrayMixin):
                _handled_functions = _HANDLED_FUNCTIONS
                ...
    """

    #: Every subclass must override this with its own module-level `_HANDLED_FUNCTIONS` dict
    #: (see class docstring). Left as an empty dict here only as a safe default.
    _handled_functions: Dict[Any, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._handled_functions is NumpyArrayMixin._handled_functions:
            # Catches this at class-definition time with a clear message instead of letting it
            # surface later as a generic "no implementation found for numpy.mean" TypeError with
            # no indication that a `_handled_functions = _HANDLED_FUNCTIONS` line is missing.
            raise TypeError(
                f"{cls.__name__} must set its own '_handled_functions' class attribute pointing at "
                "its own module-level _HANDLED_FUNCTIONS dict (see NumpyArrayMixin's docstring) - "
                "without it, np.mean(obj)/np.std(obj) will silently return NotImplemented."
            )

    def _array_subitems(self) -> Iterable:
        """Returns an iterable of this object's per-mesh sub-items, each exposing its raw data
        via a `.data` property / `._data` attribute. Must be overridden by subclasses."""
        raise NotImplementedError

    def mean(self):
        """Calculates the mean over all sub-items.

        :returns: The calculated mean value.
        """
        return np.mean([np.mean(item.data) for item in self._array_subitems()])

    def std(self):
        """Calculates the standard deviation over all sub-items.

        :returns: The calculated standard deviation.
        """
        mean = self.mean()
        sum_ = np.sum([np.sum(np.power(item.data - mean, 2)) for item in self._array_subitems()])
        N = np.sum([item.data.size for item in self._array_subitems()])
        return np.sqrt(sum_ / N)

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray."""
        name = type(self).__name__
        raise TypeError(
            f"{name}s can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containing all subitems is"
            " required, use the 'to_global' method and use the returned numpy-array explicitly."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunc with this object as input.

        :returns: A new object of the same type on which the ufunc has been applied.
        """
        if method != "__call__":
            logging.warning(
                "The %s method has been used which is not explicitly implemented. Correctness of"
                " results is not guaranteed. If you require this feature to be implemented please"
                " submit an issue on Github where you explain your use case.",
                method,
            )
        if sum(isinstance(inp, self.__class__) for inp in inputs) > 1:
            name = type(self).__name__.lower()
            raise UserWarning(
                f"The {method} operation is not implemented for multiple {name}s as input yet. If"
                " you require this feature, please request this functionality by submitting an"
                " issue on Github."
            )
        # 'out' (e.g. from in-place operators like +=) references this wrapper object, not a plain
        # array; forwarding it would make numpy re-dispatch to this same method and recurse forever.
        # We always return a new object instead, so the in-place target is discarded here.
        kwargs.pop("out", None)

        new_obj = deepcopy(self)
        for subitem in new_obj._array_subitems():
            args = [subitem.data if isinstance(inp, self.__class__) else inp for inp in inputs]
            subitem._data = ufunc(*args, **kwargs)
        return new_obj

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with this object as input.

        :returns: The output of the array function.
        """
        handled_functions = type(self)._handled_functions
        if func not in handled_functions:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle this type.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return handled_functions[func](*args, **kwargs)
