## Release History
* 1.9.8
    * Fixed to_global for 3D slices
* 1.9.7
    * Fixed obstructions for simulations with restart in between
* 1.9.6
    * Supporting obstructions with holes
* 1.9.5
    * Added option to read the actual obstruction id set in the fds-file for FDS 6.7.10+
* 1.9.4
    * Fixed bug when reading in SHOW/HIDE OBST lines
* 1.9.3
    * Reading FDS version from .smv file is now backwards compatible
* 1.9.2
    * Removed prints
* 1.9.1
    * Improved slcf get_coordinates and added option to return coordinates for to_global method
* 1.9.0
    * Several bugfixes
* 1.8.7
    * Fixed devices and improved global slcf example
* 1.8.6
    * Fixed __repr__ of Simulation
* 1.8.5
    * Fixed particle reader
* 1.8.4
    * Obst vmax and vmin now correctly return floats instead of numpy scalars
* 1.8.3
    * Fixed obst exporter and added safe guards to obstructions for when there is no boundary data 
* 1.8.2
    * Added option to mask global slice and fixed csv and slice mask bugs
* 1.8.1
    * Lots of improvements and convenience methods for obstructions
* 1.8.0
    * Reworked Devices: Now using a DeviceCollection with pandas support, devices do now lazyload as well
* 1.7.5
    * Fixed bug in get_subslice method
* 1.7.4
    * Several bugfixes
* 1.7.3
    * Fixed bug in sublices get_coordinates method
* 1.7.2
    * Fixed naming in \_\_repr\_\_ string
* 1.7.1
    * Fixed cpu, hrr and steps reader
* 1.7.0
    * Changed the behavior of the to_global slcf method to return two slices for cases where slices cut right through mesh boundaries
* 1.6.6
    * Fixed exception when devices with duplicated IDs are given
* 1.6.5
    * Fixed slcf to_global methods for specific cell-centered slice scenarios
* 1.6.4
    * Adding hash to sim exporter
* 1.6.3
    * Fixed slcf reader for irregular meshes
* 1.6.2
    * Added cell-centered information to slcf exporter
* 1.6.1
    * Fixed slcf exporter
* 1.6.0
    * Fixed evac reader, now works with latest evac changes
* 1.5.1
    * Fixed sim exporter
* 1.5.0
    * Added python 3.10+ support
* 1.4.4
    * Added sim exporter
* 1.4.3
    * Fixed some slcf bugs
* 1.4.2
    * Added cli-friendliness to exporters
* 1.4.1
    * Fixed slcf reader slice_id bug and improved export example
* 1.4.0
    * Added option to ignore all non-critical errors and warnings to provide optional cli-friendliness
* 1.3.12
    * Fixed a bug that did not allow spaces in Mesh IDs
* 1.3.11
    * Fixed obst exporter
* 1.3.10
    * Fixed some dependency conflicts
* 1.3.9
    * Fixed rounding issues with slices
* 1.3.8
    * Made exporter imports optional
* 1.3.7
    * Fixed bug for 3d to 2d slcf function
* 1.3.6
    * Fixed a typo in code
* 1.3.5
    * Added obst/bndf exporter
    * Fixed some slcf bugs
* 1.3.4
    * Fixing an issue with pip when the working directory contains chinese characters 
* 1.3.3
    * Fixed the reading of the column names of HRR/CPU data
* 1.3.2
    * Fixed a bndf bug when no data existed for one or more Subobstructions
* 1.3.1
    * Fixed imports 
* 1.3.0
    * Fixed documentation 
* 1.2.12
    * Added a convenience function for quantity filtering to pl3d
* 1.2.11
    * Finished the slcf exporter
* 1.2.10
    * Fixed some edge cases for evac when no data existed 
* 1.2.9
    * Bugfixes for smoke3d exporter 
* 1.2.8
    * Modified smoke3d exporter to export yaml instead of mhd
* 1.2.7
    * Fixed issues with imports 
* 1.2.6 
    * Modified smoke3d exporter to export 4d data instead of 3d
* 1.2.5
    * Made usage of quantities easier 
* 1.2.4
    * Added to_global convenience function for non-uniform meshes
* 1.2.3
    * Support for profile data (prof)
* 1.2.2
    * Support for mesh boundary data
* 1.2.1
    * Added gslcf support (geomslices)
* 1.2.0
    * Added evac support
    * Optimized manual cache clearing functionality
* 1.1.1
    * Slcf hotfixes
* 1.1.0
    * Added smoke3d support
    * Bugfixes for devices
    * Devices now also contain units of output quantities
* 1.0.11
    * Bugfixes for cell-centered slices
    * Added get_by_id filter for obstructions and meshes
* 1.0.10
    * Improvements for bndf
* 1.0.9
    * Improvements for complex geometry
* 1.0.8
    * Bugfixes for obst slice masks
* 1.0.7
    * Caching bugfixes
* 1.0.6
    * Caching bugfixes
* 1.0.5
    * Bugfixes for obst slice masks
    * Added tag convenience function for particles
* 1.0.4
    * Bugfixes for obst masks
* 1.0.3
    * Bugfixes for part
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
    * Slice and pl3d now support numpy.mean
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
* 0.3.1
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