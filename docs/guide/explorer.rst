Explorer
========

``fdsreader.explorer`` looks at a simulation without writing plotting code for it: a time
bar driving a 2D slice, and any number of device or HRR quantities plotted against time.
In Jupyter that is an interactive widget; in a terminal you can either explore it
interactively or draw a single view and exit.

In Jupyter
----------

.. code-block:: python

   from fdsreader.explorer import explore

   explorer = explore()

That renders the whole interface — a directory browser, the time bar, the slice on the
left and the curves on the right. Browse to a case and press *Load simulation*.

.. code-block:: python

   explore("./sample_data")            # skip the browser and load this case
   explore(start_dir="/data/cases")    # open the browser somewhere specific
   explore(interactive_canvas=False)   # static images instead of ipympl canvases
   explore(caching=True)               # allow the .pickle cache (writable data only)

``explore()`` returns the explorer, so the simulation and the current selection stay
available in later cells:

.. code-block:: python

   sim = explorer.sim
   explorer.current_time               # where the time bar is, in seconds
   explorer.selection                  # the curves currently ticked
   explorer.field.frame(400)           # the slice on screen, at time step 400

``ipympl`` is optional; with it each plot gains a toolbar to zoom, pan and save. If the
plots come up as ``Failed to load model class 'MPLCanvasModel'``, its browser-side
extension is not loading — pass ``interactive_canvas=False``.

In a terminal
-------------

The command line front end needs nothing but numpy, so it works over SSH on a machine
with no plotting stack and no browser.

``-i`` opens a full-screen interactive view, where the slice and the curves are chosen from
lists rather than named on the command line:

.. code-block:: sh

   fdsreader-explorer-cli ./sample_data -i

.. list-table::
   :header-rows: 1

   * - Key
     - Action
   * - ``←`` ``→`` or ``h`` ``l``
     - step through time
   * - ``page up`` / ``page down``
     - ten steps
   * - ``home`` / ``end``
     - first / last step
   * - ``space``
     - play, pause
   * - ``+`` ``-``
     - play speed
   * - ``f``
     - pick the slice from a list
   * - ``c``
     - pick the curves from a list (``space`` ticks, ``enter`` accepts)
   * - ``[`` ``]``
     - previous / next slice
   * - ``g`` / ``s``
     - colour scale over the whole run / this step
   * - ``?``
     - help
   * - ``q``
     - quit

Without ``-i`` it draws one view and exits, which is what you want in a script:

.. code-block:: sh

   fdsreader-explorer-cli ./sample_data                        # what is in it
   fdsreader-explorer-cli ./sample_data --list                 # slices and curves by name
   fdsreader-explorer-cli ./sample_data --slice 0 --time 4.4   # draw a slice
   fdsreader-explorer-cli ./sample_data --curve TC_1 --curve HRR
   fdsreader-explorer-cli ./sample_data --json                 # for scripting

.. code-block:: text

   TEMPERATURE [C] — x = 0 m  #0   ·   t = 4.401 s (step 630 of 749)

     10.00 |
           |
      8.82 |
           |
           |
           |          ...
           |        .....
      5.88 |       .....
           |      ...:...
           |       .....
           |        ....
           |        ...
      2.94 |       .:..
           |       -.
           |       -:-.
           |       .-*:
           |        +%+.
      0.00 |        +#:
           +------------------
            -2.5           2.5   y [m]
            scale 20 … 1412   ramp ' .:-=+*#%@'   grid 18x18

``--save FILE`` writes the current view as an image or an animation instead
(``.png``/``.pdf``/``.svg``/``.gif``/``.mp4``, needs matplotlib).

Notes
-----

- The interactive terminal view uses ``curses``, which is in the standard library everywhere except
  Windows; there, ``pip install windows-curses`` provides it.
- Slices are read one time step at a time, so stepping through a long run costs a few
  hundred kilobytes rather than loading the whole series into memory.

All classes and functions are listed in the :doc:`API reference <../api/explorer>`.
