"""Regression test for GitHub issue #87: particle quantity data got scrambled for any
particle class with more than one quantity. FDS writes the quantity block in Fortran
(column-major) order, but the previous code read it directly into an already
(n_particles, n_quantities)-shaped dtype field (which numpy always fills in C order)
and then called `.reshape(same_shape, order="F")` — a no-op, since reshaping to an
identical shape never rearranges data regardless of the order argument."""

import os

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader.part import Particle, ParticleCollection
from fdsreader.utils import Quantity


def _write(outfile, dtype, value):
    record = np.zeros(1, dtype=dtype)
    record["f1"] = value
    record.tofile(outfile)


def _build_prt5_file(path, n_particles, quantity_values):
    """quantity_values: list of 1D arrays, one per quantity, each of length n_particles."""
    n_quantities = len(quantity_values)
    with open(path, "wb") as f:
        # Endian marker, FDS version, and class count are each their own Fortran record.
        _write(f, fdtype.INT, 1)
        _write(f, fdtype.INT, 610)
        _write(f, fdtype.INT, 1)
        _write(f, fdtype.new((("i", 2),)), [n_quantities, 0])
        for i in range(n_quantities):
            _write(f, fdtype.new((("c", 30),)), f"QUANTITY_{i}".encode().ljust(30))
            _write(f, fdtype.new((("c", 30),)), b"unit".ljust(30))

        _write(f, fdtype.FLOAT, 0.0)  # time
        _write(f, fdtype.INT, n_particles)
        positions = np.zeros(3 * n_particles, dtype=np.float32)
        _write(f, fdtype.new((("f", 3 * n_particles),)), positions)
        tags = np.arange(1, n_particles + 1, dtype=np.int32)
        _write(f, fdtype.new((("i", n_particles),)), tags)

        # Fortran/column-major order: all particles for quantity 1, then quantity 2, ...
        flat = np.concatenate(quantity_values).astype(np.float32)
        _write(f, fdtype.new((("f", n_particles * n_quantities),)), flat)


def test_particle_quantities_not_scrambled(tmp_path):
    n_particles = 4
    quantity_values = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([100.0, 200.0, 300.0, 400.0]),
        np.array([-1.0, -2.0, -3.0, -4.0]),
    ]
    file_path = os.path.join(tmp_path, "test_1.prt5")
    _build_prt5_file(file_path, n_particles, quantity_values)

    quantities = [Quantity(f"QUANTITY_{i}", f"QUANTITY_{i}", "unit") for i in range(3)]
    particle = Particle("TEST_PART", quantities, (0.0, 0.0, 0.0))
    particle.n_particles["mesh1"] = [n_particles]

    collection = ParticleCollection([0.0], [particle])
    collection._file_paths["mesh1"] = file_path
    collection._load_data()

    for i, expected in enumerate(quantity_values):
        assert np.array_equal(particle.data[f"QUANTITY_{i}"][0], expected), f"quantity {i} was scrambled"
