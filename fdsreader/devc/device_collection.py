
from typing import Iterable, Union, List

from fdsreader.devc import Device
from fdsreader.utils.data import FDSDataCollection


class DeviceCollection(FDSDataCollection):
    """Collection of :class:`Device` objects. Offers additional functionality for working on devices using pandas.
    """

    def __init__(self, *devices: Iterable[Device]):
        super().__init__(*devices)

    def __getitem__(self, key) -> Union[Device, List[Device]]:
        if type(key) == int:
            return self._elements[key]
        else:
            return next(devc for devc in self._elements if (devc.id == key if type(devc) == Device else devc[0].id == key))

    def __contains__(self, value: Union[Device, str]):
        id_matching = any((devc.id == value if type(devc) == Device else devc[0].id == value) for devc in self._elements)
        return value in self._elements or id_matching

    def to_pandas_dataframe(self):
        """Returns a pandas DataFrame with device-IDs as column names and device data as column values.
        """
        import pandas as pd
        data = dict()
        for devc in self:
            if type(devc) == Device:
                data[devc.id] = devc.data
            elif type(devc) == list:
                # It might be the case that there are multiple devices with the same name
                for i, list_devc in enumerate(devc):
                    data[list_devc.id + "_" + str(i)] = list_devc.data
        return pd.DataFrame(data)

