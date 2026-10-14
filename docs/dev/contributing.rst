Contributing
============

Bug reports and pull requests are welcome. Please check the
`issue tracker <https://github.com/FireDynamics/fdsreader/issues>`_ for known problems
first.

`CONTRIBUTING.md <https://github.com/FireDynamics/fdsreader/blob/master/CONTRIBUTING.md>`_
in the repository describes the development setup, how to extract the test data and run
the tests, the code style, and the checklist for pull requests.

To build this documentation locally:

.. code-block:: sh

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html
