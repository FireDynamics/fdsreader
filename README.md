# FDSReader
> Fast and easy-to-use Python reader for FDS data

[![PyPI version](https://badge.fury.io/py/fdsreader.png)](https://badge.fury.io/py/fdsreader)
[![CI](https://github.com/FireDynamics/fdsreader/actions/workflows/ci.yml/badge.svg)](https://github.com/FireDynamics/fdsreader/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FireDynamics/fdsreader/branch/master/graph/badge.svg)](https://codecov.io/gh/FireDynamics/fdsreader)
[![Documentation](https://readthedocs.org/projects/fdsreader/badge/?version=latest)](https://fdsreader.readthedocs.io)

## FDS Version Compatibility

| fdsreader          | FDS 6.7 | FDS 6.8 | FDS 6.9 | FDS 6.10 |
|--------------------|---------|---------|---------|----------|
| ≤ 1.11.x           | ✅      | ✅      | ✅      | ⚠️ (Geometry bug, [#TODO](https://github.com/FireDynamics/fdsreader/issues)) |
| 1.12.x *(planned)* | ✅ | ✅ | ✅ | ✅ |

_Tested against FDS outputs. If you find a compatibility issue please [open an issue](https://github.com/FireDynamics/fdsreader/issues)._

## Installation

The package is available on PyPI and can be installed using pip:
```sh
pip install fdsreader
```

The Jupyter [explorer](#explorer) needs a plotting stack, which is an optional extra:
```sh
pip install "fdsreader[notebook]"
```
Its command line counterpart needs nothing beyond numpy and is always installed.

_FDS Version 6.7.5 and above are fully supported. Versions below 6.7.5 might work, but are not guaranteed to work._

## Usage example

```python
import fdsreader as fds

# Creates an instance of a simulation master-class which manages all data for a given simulation
sim = fds.Simulation("./sample_data")

# Examples of data that can be easily accessed
print(sim.meshes, sim.surfaces, sim.slices, sim.data_3d, sim.smoke_3d, sim.isosurfaces, sim.particles, sim.obstructions)
```

More advanced examples can be found in the respective data type directories inside of the examples directory.

### Configuration
The package provides a few configuration options that can be set using the `settings` module.
```python
fds.settings.KEY = VALUE

# Example
fds.settings.DEBUG = True
```

|      KEY       |  VALUE  | Default | Description |
|----------------|---------|---------|-------------|
|    LAZY_LOAD   | boolean |   True  | Load all data when initially loading the simulation (False) or only when specific data is needed (True). |
| ENABLE_CACHING | boolean |   True  | Cache the loaded simulation to reduce startup times when loading the same simulation again. |
|     DEBUG      | boolean |  False  | Crash on non-critical errors with an exception (True) or output non-critical errors as warnings (False). |
| IGNORE_ERRORS  | boolean |  False  | Ignore any non-critical errors completely. |


### Data structure
![Data structure](https://raw.githubusercontent.com/FireDynamics/fdsreader/master/docs/img/data-structure.svg)

Beware that not all attributes and methods are covered in this diagram. For a complete
documentation of all classes check the API Documentation below.

## Explorer

`fdsreader.explorer` looks at a simulation without writing plotting code for it: a time
bar driving a 2D slice, and any number of device or HRR quantities plotted against time.
In Jupyter that is an interactive widget; in a terminal it draws one view and exits.

### In Jupyter

```python
from fdsreader.explorer import explore

explorer = explore()
```

That renders the whole interface — a directory browser, the time bar, the slice on the
left and the curves on the right. Browse to a case and press *Load simulation*.

```python
explore("./sample_data")            # skip the browser and load this case
explore(start_dir="/data/cases")    # open the browser somewhere specific
explore(interactive_canvas=False)   # static images instead of ipympl canvases
explore(caching=True)               # allow the .pickle cache (writable data only)
```

`explore()` returns the explorer, so the simulation and the current selection stay
available in later cells:

```python
sim = explorer.sim
explorer.current_time               # where the time bar is, in seconds
explorer.selection                  # the curves currently ticked
explorer.field.frame(400)           # the slice on screen, at time step 400
```

`ipympl` is optional; with it each plot gains a toolbar to zoom, pan and save. If the
plots come up as `Failed to load model class 'MPLCanvasModel'`, its browser-side
extension is not loading — pass `interactive_canvas=False`.

### In a terminal

The command line front end draws one view and exits. It needs nothing but numpy, so it
works over SSH on a machine with no plotting stack and no browser:

```sh
fdsreader-explorer-cli ./sample_data                        # what is in it
fdsreader-explorer-cli ./sample_data --list                 # slices and curves by name
fdsreader-explorer-cli ./sample_data --slice 0 --time 4.4   # draw a slice
fdsreader-explorer-cli ./sample_data --curve TC_1 --curve HRR
fdsreader-explorer-cli ./sample_data --json                 # for scripting
```

```
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
```

`--save FILE` writes the current view as an image or an animation instead
(`.png`/`.pdf`/`.svg`/`.gif`/`.mp4`, needs matplotlib).

Slices are read one time step at a time, so stepping through a long run costs a few
hundred kilobytes rather than loading the whole series into memory.

## API Documentation
[https://fdsreader.readthedocs.io](https://fdsreader.readthedocs.io)

## Releasing a new version

Versioning is handled automatically via Git tags using `setuptools-scm`.

```bash
# New release
git tag -a v1.12.0 -m "Version 1.12.0"
git push origin v1.12.0
```

This triggers the release workflow which builds the package and publishes it to PyPI.

## Meta

*  Jan Vogelsang – j.vogelsang@fz-juelich.de
*  Marc Fehr - m.fehr@fz-juelich.de
*  Kristian Börger - k.boerger@fz-juelich.de
*  Prof. Dr. Lukas Arnold - l.arnold@fz-juelich.de

Distributed under the LGPLv3 (GNU Lesser General Public License v3) license. See ``LICENSE`` for more information.

[https://github.com/FireDynamics/fdsreader](https://github.com/FireDynamics/fdsreader)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, how to run tests, and the PR checklist.
