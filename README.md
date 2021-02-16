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

### Data structure
![Data structure](https://raw.githubusercontent.com/FireDynamics/fdsreader/master/docs/img/data-structure.svg)


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

## Release History
* (unreleased) 1.0.0
    * (First official version will be released after sufficient public testing in beta stage)

### Beta *(Q1 2021)*
* (unreleased) 0.9.0
    * (Entering beta status after extensive testing with selected participants in alpha stage)
    
### Alpha *(02/2021)*
* (unreleased) 0.x.0
    * (Entering alpha status after extensive private testing in pre-alpha stage)
    
### Pre-Alpha *(current stage)*
* 0.6.2
    * Improved documentation
* 0.6.1
    * Added multimesh part support
* 0.6.0
    * Added part example
    * Added pl3d example
    * Added slcf example
    * Added two bndf examples
    * Added isof example
* 0.5.3
    * Usability improvements for bndf
* 0.5.2
    * Bugfixes for bndf
* 0.5.1
    * Several bugfixes and improvements
* 0.5.0
    * Preparing for alpha release
    * Usability improvements for simulation
    * Usability improvements for part
    * Added devc support (devices)
    * Added part support (particles)
    * Bugfixes for bndf
* 0.4.10
    * Bugfixes for slcf
* 0.4.9
    * Bugfixes for bndf
    * Improved 2D-Slice functionality
* 0.4.8
    * Complete rework of internal reading process (higher performance)
    * Complete rework of bndf
    * Bugfixes (obstructions, extents, simulation)
* 0.4.7
    * Added cache clearing functionality
    * Bugfixes
* 0.4.6
    * Added automatic caching for simulations (significant loading time reduction) 
    * Reworked internal slcf data structure
    * Fixed isof reader (now correctly reads in data for all time steps)
    * Connected bndf data to obstructions
    * Simplified instantiation of Simulation objects  
* 0.4.5
    * Added multimesh isof support
    * Improved slcf stability
* 0.4.4
    * Bugfixes (bndf and pl3d)
* 0.4.3
    * Bugfixes (slcf and isof)
* 0.4.2
    * Completed API documentation
* 0.4.1
    * Bugfixes (python import issues) 
* 0.4.0
    * Added bndf support (boundaries)
* 0.3.0
    * Added multimesh pl3d support
* 0.3.0
    * Added basic pl3d support
* 0.2.0
    * Added isof support (isosurfaces)
* 0.1.2
    * Added numpy support for slices
* 0.1.1
    * Added multimesh slcf support
    * Added API documentation
    * Package available on PyPI
* 0.1.0
    * Added basic slcf support (2D + 3D slices)
