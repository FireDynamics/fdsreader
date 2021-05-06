## Release History
* 1.0.7
    * Caching bugfixes
* 1.0.6
    * Caching bugfixes
* 1.0.5
    * Bugfixes for obst slice masks
    * Added tag convencience function for particles
* 1.0.4
    * Bugfixes for obst masks
* 1.0.3
    * Bufixes for part
* 1.0.2
    * Bugfixes for Python 3.6/3.7
* 1.0.1
    * Vmin/Vmax and numpy min/max support for slcf
* 1.0.0
    * Reworked bndf data structure
    * Fixed multiple bugs
    * Added quantities property to all collections

### Beta
* 0.9.22
    * Bugfixes for bndf
* 0.9.21
    * Slice and Pl3D now support np.mean
* 0.9.20
    * Slice coordinates now correctly represent cell-centered slices
* 0.9.19
    * Added more utility functions for slices
* 0.9.18
    * Improved slice filtering
* 0.9.17
    * Made collection output easier to read
* 0.9.16
    * Added verbosity to Simulation class
* 0.9.15
    * Improved collection filtering
* 0.9.14
    * Resolved possible requirement version conflicts
* 0.9.13
    * Refactored requirements
* 0.9.12
    * Bugfixes for slcf
* 0.9.11
    * Added option to retrieve coordinates from slices
* 0.9.10
    * Improved slice filtering functionality
* 0.9.9
    * Added bndf support for older versions of fds
* 0.9.8
    * Fixed an issue that caused numpy to crash on Unix-based machines
* 0.9.7
    * Added slcf support for older versions of fds
* 0.9.6
    * Bugfixes for complex geometry
* 0.9.5
    * Fixed issue with invalid pickle files (cache)
* 0.9.4
    * Bugfixes for complex geometry
* 0.9.3
    * Added lazy loading for complex geometry data
* 0.9.2
    * Improved documentation for complex geometry
* 0.9.1
    * Added documentation for complex geometry
* 0.9.0
    * Added rudimentary support for complex geometry
    
### Alpha
* 0.8.3
    * Added collections to documentation
* 0.8.2
    * Fixed slcf examples
* 0.8.1
    * Bugfixes for vents
* 0.8.0
    * Major bugfixes
    
### Pre-Alpha
* 0.7.4
    * Added obst hole support (bndf)
* 0.7.3
    * Added exception handling for optional modules
* 0.7.2
    * Improved slice combination functionality
* 0.7.1
    * Made PyVista (Vtk) requirement optional
* 0.7.0
    * Added automatic version compatibility testing
    * Usability improvements for slcf
* 0.6.5
    * Bugfixes
* 0.6.4
    * Added more convenience functions (mainly filters)
    * Added color-data isof support
* 0.6.3
    * Added some convenience functions (filters, obstruction masks, ...)
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