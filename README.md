# FDSReader
> Fast and easy-to-use Python reader for FDS data

[![Build Status](https://travis-ci.com/FireDynamics/fdsreader.svg?branch=master)](https://travis-ci.com/FireDynamics/fdsreader)  
[![PyPI version](https://badge.fury.io/py/fdsreader.png)](https://badge.fury.io/py/fdsreader)  


## Installation

The package is available on PyPI and can be installed using pip:  
```sh
pip install fdsreader
```

## Usage example
```python
import fdsreader as fds

# Creates an instance of a simulation master-class which manages all data for a given simulation
sim = fds.Simulation("./sample_data")

# Examples of data that can be easily accessed
print(sim.meshes, sim.surfaces, sim.slices, sim.data_3d, sim.isosurfaces, sim.particles, sim.obstructions, sim.obstructions[0].get_boundary_data("temperature"))
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
|     DEBUG      | boolean |  False  | Crash on non-critical errors with an exception (True) or hide non-critical errors (False). |


### Data structure
![Data structure](https://raw.githubusercontent.com/FireDynamics/fdsreader/master/docs/img/data-structure.svg)

Beware that not all attributes and methods are covered in this diagram. For a complete  
documentation of all classes check the API Documentation below.  

## API Documentation
[https://firedynamics.github.io/fdsreader/](https://firedynamics.github.io/fdsreader/)

## Meta

*  Jan Vogelsang – j.vogelsang@fz-juelich.de
*  Prof. Dr. Lukas Arnold - l.arnold@fz-juelich.de

Distributed under the LGPLv3 (GNU Lesser General Public License v3) license. See ``LICENSE`` for more information.

[https://github.com/FireDynamics/fdsreader](https://github.com/FireDynamics/fdsreader)

## Contributing

1. Fork it (<https://github.com/FireDynamics/fdsreader/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
