---
title: 'FDSReader'
tags:
  - Python
  - TODO
authors:
  - name: Jan Niklas Vogelsang
    orcid: 0000-0003-2336-6630
    affiliation: 1
  - name: Prof. Dr. Lukas Arnold
    orcid: TODO
    affiliation: 2
affiliations:
  - name: PGI-15, Forschungszentrum Jülich, Germany
    index: 1
  - name: IAS-7, Forschungszentrum Jülich, Germany
    index: 2
date: 15 March 2024
bibliography: paper.bib
---

# Summary

In recent years, Python has become the most popular programming language in the scientific community. It offers
extensive functionality for preparing, post-processing and visualising data of all types and origins. By
a large number of freely available packages, the programming environment can be flexibly adapted to your own
needs. However, until now it was not possible to automatically import and process fire simulation data as
produced by the fire dynamics simulator (FDS) [@FDS] for example. Using SmokeView [@SMV], it was
already possible to visualize generated data, but it was not possible to process it directly or perform detailed
analysis on the data itself. The introduction of the fdsreader made this possible in an uncomplicated and efficient way.

# Statement of need

Up until now, most working groups in both industry and research have had no solution for efficiently and
automatically reading the binary data output by FDS to process it further. One would either have to use Fds2Ascii, a
Fortran program to manually convert the binary output of FDS into a text file and read in the text file with their
preferred language or write their own scripts to read the binary data output of FDS and further process the data. Many
groups did end up with their own set of, typically Python-based, scripts to read the various types of data an FDS
simulation can produce, e.g. slices, boundary data, etc.
As the internal data structures of FDS are non-trivial and the source code is not trivial to understand for an outsider,
the scripts usually require lots of time to write, ended up having strict limitations to the simulation parameters,
e.g. only  single mesh based simulations, and could not process all types of data. In addition, new versions of FDS
often meant that some scripts no longer worked as expected, requiring even more time investment to maintain these scripts.
The fdsreader Python package provides functionality to automatically import the full set of FDS outputs for both old and
new versions of FDS.
When first loading a simulation, all metadata is collected and cached to reduce subsequent load times. The actual
data produced during the simulation such as the slice data for each time step, is not loaded until the data is accessed by the
user via Python, which then triggers loading the data into memory in the background and converting it to equivalent
Python datastructures. This method minimizes initial loading time without sacrificing the ability to easily filter and
select the desired data. The fdsreader collects all data of the same type into one collection and offers its own set
functions for each type to easily select the data which should be further processed. These functions can even be called
in an automated fashion, to run some predefined postprocessing routines on simulation data without having to manually
interact with the data, as one would have to do when using SmokeView or Fds2Ascii.
The fdsreader is able to read all data types available in FDS including most of the additional metadata. To be concrete,
the package contains modules for slices (slcf), boundary data (bndf), volumes (plot3d, smoke3d) particles (part),
isosurfaces (isof), evacuations (evac), complex geometry boundary data (geom), devices (devc) and meshes. Slices,
boundary data, isosurfaces and volumes are each respectively seperated into multiple parts, one for each mesh. While FDS
outputs the data separately for each mesh, the fdsreader provides methods to combine all these parts automatically and
operate on data across meshes.

# Acknowledgements
TODO

# References
