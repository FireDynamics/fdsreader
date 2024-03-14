---
title: 'FDSReader'
tags:
  - Python
  - TODO
authors:
  - name: Jan Niklas Vogelsang
    orcid: 0000-0003-2336-6630
    equal-contrib: true
    affiliation: 1
  - name: Prof. Dr. Lukas Arnold
    orcid: TODO
    equal-contrib: false
    affiliation: 2
affiliations:
 - name: PGI-15, Forschungszentrum Juelich, Germany
   index: 1
 - name: IAS-7, Forschungszentrum Juelich, Germany
   index: 2
date: 15 March 2024
bibliography: paper.bib
---

# Summary
Over the past few years, Python has become the most popular programming language in the scientific community. It offers extensive functionality for the preparation, post-processing and visualization of data of any type and origin. By conveniently importing a variety of freely available packages, the programming environment can be flexibly adapted to one’s own needs. Until now, however, it was not possible to automatically import and process fire simulation data as procuded by the fire dynamics simulator FDS (Mcdermott10?) for example. Using SmokeView (Forney00?), it was already possible to visualize generated data, but it was not possible to process it directly or analyze it exactly. With the introduction of the fdsreader (FDSReader:2022?), this is now possible in an uncomplicated and efficient way.

# Statement of need
Up until now, most working groups did not have any solution to efficiently and automatically read in the binary data output by FDS to process it further. One would have to either use Fds2Ascii, a Fortran program to manually convert the binary output into a text file, and read in the text file with their preferred language or write their own scripts to read the binary data output of FDS and further process the data. Many groups did end up with their own set of, typically python-based, scripts to read the various types of data an FDS simulation can produce, e.g. slices, boundary data, etc. As the FDS internal data structures are non-trivial and the source code uneasy to understand as an outsider, the scripts usually took large amounts of time to write, ended up having strict limitations to the simulation parameters, e.g. only single mesh based simulations, and could not process all types of data. Furthermore, new versions of FDS often led to some scripts not working as expected anymore, so maintaining these scripts cost even more additional time.
The fdsreader python package provides functionality to automatically import everything FDS outputs in a simulation run. When first loading a simulation, all metadata is collected and cached to reduce subsequent loading times. The actual data produced during the simulation, e.g. slice data for each timestep, is not loaded until the data is accessed by the user via python, which then triggers loading the data into memory in the background and converting it to equivalent python datastructures. This methods minimizes initial loading time without sacrificing the ability to easily filter and select the desired data. The fdsreader collects all data of the same type into one collection and offers its own set functions for each type to easily select the data which should be further processed. These functions can even be called in an automated fashion, to run some predefined postprocessing routines on simulation data without having to manually interact with the data, as one would have to do when using SmokeView or Fds2Ascii.
The fdsreader is able to read all data types available in FDS including most of the additional metadata. To be concrete, the package contains modules for slices (slcf), boundary data (bndf), volumes (plot3d, smoke3d) particles (part), isosurfaces (isof), evacuations (evac), complex geometry boundary data (geom), devices (devc) and meshes. Slices, boundary data, isosurfaces and volumes are each respectively seperated into multiple parts, one for each mesh. While FDS outputs the data seperately for each mesh, the fdsreader provides methods to combine all these parts automatically.

# Acknowledgements
We acknowledge contributions from TODO

# References
