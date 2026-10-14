Installation
============

fdsreader is on `PyPI <https://pypi.org/project/fdsreader/>`_ and needs Python 3.11 or
newer. Its only dependencies are numpy and typing_extensions.

.. code-block:: sh

   pip install fdsreader

Optional extras
---------------

The Jupyter :doc:`explorer <explorer>` needs a plotting stack (matplotlib and ipywidgets),
which is an optional extra:

.. code-block:: sh

   pip install "fdsreader[notebook]"

The terminal explorer (``fdsreader-explorer-cli``) needs nothing beyond numpy and is always
installed. Its interactive view uses ``curses``, which is part of the standard library
everywhere except Windows; there, ``pip install windows-curses`` provides it.

The exporters in :doc:`fdsreader.export <../api/export>` write data to raw binary files and
need a few more packages:

.. code-block:: sh

   pip install "fdsreader[export]"

:meth:`DeviceCollection.to_pandas_dataframe() <fdsreader.devc.device_collection.DeviceCollection.to_pandas_dataframe>`
needs pandas, which is not installed automatically.

Development version
-------------------

To get fixes before they are released, install from the ``master`` branch:

.. code-block:: sh

   pip install "git+https://github.com/FireDynamics/fdsreader.git"

Which FDS versions are supported is listed under :doc:`compatibility`.
