# FDSReader
> Fast and easy-to-use Python reader for FDS data

[![Build Status](https://travis-ci.com/FireDynamics/fdsreader.svg?branch=master)](https://travis-ci.com/FireDynamics/fdsreader)

## Installation

The package is available on PyPI and can be installed using pip:  
```sh
python -m pip install --upgrade fdsreader
```

## Usage example
```python
import fdsreader as fds

# Creates an instance of a simulation master-class which manages all data for a given simulation
sim = fds.Simulation("./sample_data")

# Examples of data that can be easily accessed
print(sim.meshes, sim.meshes[0].obstructions, sim.surfaces, sim.slices, sim.boundaries, sim.data_3d, sim.isosurfaces)
```

### Data structure
![Data structure](https://raw.githubusercontent.com/FireDynamics/fdsreader/master/docs/img/data-structure.svg)


## API Documentation
[https://firedynamics.github.io/fdsreader/](https://firedynamics.github.io/fdsreader/)

## Release History
* (unreleased) 1.0.0
    * (First official version will be released after sufficient public testing in beta stage)

### Beta *(Q1 2021)*
* (unreleased) 0.9.0
    * (Entering beta status after extensive testing with selected participants in alpha stage)
    
### Alpha *(01/2021)*
* (unreleased) 0.x.0
    * (Entering alpha status after extensive private testing in pre-alpha stage)
    
### Pre-Alpha *(current stage)*
* 0.4.7
    * Added cache clearing functionality
    * Bugfixes for bndf
* 0.4.6
    * Added automatic caching for simulations (which significantly reduces simulation loading time) 
    * Reworked internal slcf data structure
    * Fixed isof reader (now correctly reads in data for all time steps)
    * Connected bndf data to obstructions
    * Simplified instantiation of Simulation objects  
* 0.4.5
    * Added Multimesh isof support
    * Improved slcf and bndf stability
* 0.4.4
    * Bugfixes for bndf and plot3d
* 0.4.3
    * Bugfixes for slcf and isof readers
* 0.4.2
    * Completed API documentation
* 0.4.1
    * Fixed import issues 
* 0.4.0
    * Added BNDF support (boundaries)
* 0.3.0
    * Added Multimesh Plot3D support
* 0.3.0
    * Added basic Plot3D support
* 0.2.0
    * Added isof support (isosurfaces)
* 0.1.2
    * Added numpy support for slices
* 0.1.1
    * Added multimesh slcf support (multimesh slices)
    * Added API documentation
    * Package available on PyPI
* 0.1.0
    * Added basic slcf support (slices)

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
