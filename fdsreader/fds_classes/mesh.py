import math
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Literal

from fdsreader.bndf import Boundary, Patch
from fdsreader.bndf.obstruction import _find_boundary
from fdsreader.utils import Dimension, Extent, Quantity


class Mesh:
    """3-dimensional Mesh of fixed, defined size.

    :ivar coordinates: Coordinate values for each of the 3 dimension.
    :ivar dimensions: :class:`Dimension` describing the size of the 3 dimensions regarding indices.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar n: Number of elements for each of the 3 dimensions.
    :ivar n_size: Total number of blocks in this mesh.
    :var id: Mesh id/short_name assigned to this mesh.
    """

    def __init__(
        self,
        coordinates: Dict[Literal["x", "y", "z"], np.ndarray],
        extents: Dict[Literal["x", "y", "z"], Tuple[float, float]],
        mesh_id: str,
    ):
        """
        :param coordinates: Coordinate values of the three axes.
        :param extents: Extent of the mesh in each dimension.
        :param mesh_id: ID of this mesh.
        """
        self.id = mesh_id
        self.coordinates = coordinates
        self.dimension = Dimension(
            coordinates["x"].size if coordinates["x"].size > 0 else 1,
            coordinates["y"].size if coordinates["y"].size > 0 else 1,
            coordinates["z"].size if coordinates["z"].size > 0 else 1,
        )

        self.n_size = self.dimension.size()
        self.extent = Extent(
            extents["x"][0], extents["x"][1], extents["y"][0], extents["y"][1], extents["z"][0], extents["z"][1]
        )

        self.obstructions = list()

        self._boundary_data: Dict[int, Boundary] = dict()

    def get_obstruction_mask(
        self,
        times: Sequence[float],
        cell_centered=False,
        region: Tuple[slice, slice, slice] | None = None,
        mode: Literal["obstructions", "placeholders"] = "obstructions",
    ) -> np.ndarray:
        """Marks all cells (or nodes) which are blocked by an obstruction.

        With ``mode="obstructions"`` (the default) the mask shows where the obstructions are: the cells inside an
        obstruction, or the nodes inside it and on its faces. An obstruction without thickness (a thin wall) blocks
        the nodes on it but no cells, as no cell lies inside it. For cell-centred data this is what Smokeview blanks;
        for nodes Smokeview blanks the areas between them next to solid cells, so a gas cell between two
        obstructions one cell apart has all its nodes masked here but is shown in Smokeview.

        With ``mode="placeholders"`` it marks only the values FDS fills with a placeholder (the ambient value)
        instead of a value of the gas: the cells inside an obstruction, and the nodes where all cells of this mesh
        around them are, as FDS averages a node from the gas cells around it (CORNER_VALUE in dump.f90). So the
        nodes on the faces of an obstruction keep their gas values, except where obstructions touch each other or
        an obstruction ends at the edge of the mesh. For cell-centred data both modes are the same.

        Both masks are derived from &OBST input only; solids defined with &GEOM are not included, and FDS writes
        face quantities such as velocities differently.

        :param times: All timesteps of the simulation.
        :param region: Index ranges along x, y and z, as slices without a step, to which the mask is limited.
            The whole mesh by default.
        :param mode: ``"obstructions"`` or ``"placeholders"``, see above.
        :returns: A 4-dimensional array with time as first and x,y,z as last dimensions, False where blocked. The
            array depends on time as obstructions may be hidden as specific points in time.
        """
        if mode not in ("obstructions", "placeholders"):
            raise ValueError(f"mode must be 'obstructions' or 'placeholders', not {mode!r}")
        shape = self.dimension.shape(cell_centered=cell_centered)
        if region is None:
            region = (slice(None),) * 3
        if any(r.step not in (None, 1) for r in region):
            raise ValueError(f"The region must not have a step, got {region}")
        # Resolve open ends and negative indices like numpy indexing does
        region = tuple(slice(*r.indices(n)[:2]) for r, n in zip(region, shape))
        region = tuple(slice(r.start, max(r.start, r.stop)) for r in region)
        times = np.asarray(times, dtype=float)

        if mode == "obstructions":
            mask = np.ones((len(times),) + tuple(r.stop - r.start for r in region), dtype=bool)
            for visible, blocked in self._obstruction_parts(times, region, nodes=not cell_centered):
                mask[(visible,) + blocked] = False
            return mask

        # Placeholders: the cells needed are those of the region, or for nodes the cells on both sides of each node
        n_cells = self.dimension.shape(cell_centered=True)
        offset = 0 if cell_centered else 1
        window = tuple(slice(r.start - offset, r.stop) for r in region)
        inside = tuple(slice(max(w.start, 0), min(w.stop, n)) for w, n in zip(window, n_cells))
        solid = np.zeros((len(times),) + tuple(max(c.stop - c.start, 0) for c in inside), dtype=bool)

        for visible, cells in self._obstruction_parts(times, inside, nodes=False):
            solid[(visible,) + cells] = True

        # Cells outside the mesh are left out of a node's neighbourhood, i.e. count as solid
        padding = [(0, 0)] + [(c.start - w.start, w.stop - c.stop) for c, w in zip(inside, window)]
        solid = np.pad(solid, padding, constant_values=True)

        if cell_centered:
            return ~solid
        # A node is blocked if all (up to 8) cells around it are solid
        nodes = np.ones((len(times),) + tuple(r.stop - r.start for r in region), dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    nodes &= solid[:, dx : dx + nodes.shape[1], dy : dy + nodes.shape[2], dz : dz + nodes.shape[3]]
        return ~nodes

    def _obstruction_parts(self, times: np.ndarray, window: Tuple[slice, slice, slice], nodes: bool):
        """The parts of all obstructions in this mesh that lie in the window (index ranges along x, y, z): for each,
        the indices of the times it is visible at and its cells (low..high-1 between the nodes of its faces) or
        nodes (low..high), relative to the window."""
        for obst in self.obstructions:
            for subobst in obst.get_subobstructions(self):
                ranges = list()
                for dim, w in zip(("x", "y", "z"), window):
                    low, high = subobst.bound_indices[dim]
                    high = high + 1 if nodes else high
                    ranges.append(slice(max(low, w.start) - w.start, max(min(high, w.stop) - w.start, 0)))
                if any(r.stop <= r.start for r in ranges):
                    continue  # none in the window, or no cells (an obstruction without thickness)
                yield self._time_indices(times, subobst.get_visible_times(times)), tuple(ranges)

    @staticmethod
    def _time_indices(times: np.ndarray, selected: Sequence[float]) -> np.ndarray:
        """Indices of the given times (a subset of times) in the sorted array times."""
        selected = np.asarray(selected, dtype=float)
        if len(times) < 2 or selected.size == 0:
            return np.zeros(selected.size, dtype=int)
        idx = np.clip(np.searchsorted(times, selected), 1, len(times) - 1)
        return idx - (np.abs(times[idx - 1] - selected) <= np.abs(times[idx] - selected))

    def get_obstruction_mask_slice(self, subslice, mode: Literal["obstructions", "placeholders"] = "obstructions"):
        """Marks the cells, or nodes, of a single subslice which are blocked by an obstruction, by the rules of
        :meth:`get_obstruction_mask` for the given mode.

        :returns: A 4-dimensional array with time as first and x,y,z as last dimensions, matching the shape of the
            subslice's data. The array depends on time as obstructions may be hidden at specific points in time.
            For a 2D slice the axis of its orientation has length 1.
        """
        ranges = subslice.index_ranges()
        region = tuple(ranges[dim] for dim in ("x", "y", "z"))
        return self.get_obstruction_mask(subslice.times, cell_centered=subslice.cell_centered, region=region, mode=mode)

    def coordinate_to_index(
        self,
        coordinate: Tuple[float, ...],
        dimension: Tuple[Literal[1, 2, 3, "x", "y", "z"], ...] = ("x", "y", "z"),
        cell_centered=False,
    ) -> Tuple[int, ...]:
        """Finds the nearest point in the mesh's grid and returns its indices.

        :param coordinate: Tuple of 3 floats. If the dimension parameter is supplied, up to 2
            dimensions can be left out from the tuple.
        :param dimension: The dimensions in which to return the indices (1=x, 2=y, 3=z).
        :param cell_centered: Instead of finding the nearest point on the mesh, find the center of the nearest cell.
        """
        # Convert possible integer input to chars
        dimension = tuple(("x", "y", "z")[dim - 1] if isinstance(dim, int) else dim for dim in dimension)

        ret = list()
        for i, dim in enumerate(dimension):
            co = coordinate[i]
            coords = self.coordinates[dim]
            if cell_centered:
                coords = coords[:-1] + np.diff(coords) / 2
            idx = np.searchsorted(coords, co, side="left")
            if idx > 0 and (idx == len(coords) or math.fabs(co - coords[idx - 1]) < math.fabs(co - coords[idx])):
                ret.append(idx - 1)
            else:
                ret.append(idx)
        return tuple(ret)

    def get_nearest_coordinate(
        self,
        coordinate: Tuple[float, ...],
        dimension: Tuple[Literal[1, 2, 3, "x", "y", "z"], ...] = ("x", "y", "z"),
        cell_centered=False,
    ) -> Tuple[float, ...]:
        """Finds the nearest point in the mesh's grid.

        :param coordinate: Tuple of 3 floats. If the dimension parameter is supplied, up to 2
            dimensions can be left out from the tuple.
        :param dimension: The dimensions in which to return the indices (1=x, 2=y, 3=z).
        :param cell_centered: Instead of finding the nearest point on the mesh, find the center of the nearest cell.
        """
        indices = self.coordinate_to_index(coordinate, dimension, cell_centered)
        ret = list()
        for i, dim in enumerate(dimension):
            coords = self.coordinates[dim]
            if cell_centered:
                coords = coords[:-1] + np.diff(coords) / 2
            ret.append(coords[indices[i]])
        return tuple(ret)

    def _add_patches(
        self,
        bid: int,
        cell_centered: bool,
        quantity: str,
        short_name: str,
        unit: str,
        patches: List[Patch],
        times: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(
                Quantity(quantity, short_name, unit), cell_centered, times, patches, lower_bounds, upper_bounds
            )
            # Add reference to parent boundary class in patches
            for patch in patches:
                patch._boundary_parent = self._boundary_data[bid]

    def get_boundary_data(self, quantity: Union[str, Quantity]) -> Boundary:
        """The boundary data of a quantity on the faces of this mesh that are not faces of obstructions (those belong
        to :class:`Obstruction`), e.g. the floor of the simulation domain. With ``MB`` vents in a multi-mesh case (see
        the FDS User Guide), FDS may also write faces between two meshes.

        A face may consist of several patches with the same orientation (see :attr:`Boundary.patches`); to combine
        them over all meshes, use :meth:`MeshCollection.get_global_boundary_data_arrays`.

        :param quantity: The quantity's name or short name.
        """
        return _find_boundary(self._boundary_data, quantity)

    def __getitem__(self, dimension: Literal[0, 1, 2, "x", "y", "z"]) -> np.ndarray:
        """Get all values in given dimension.

        :param dimension: The dimension in which to return all grid values (0=x, 1=y, 2=z).
        """
        # Convert possible integer input to chars
        if isinstance(dimension, int):
            dimension = ("x", "y", "z")[dimension]
        return self.coordinates[dimension]

    def __eq__(self, other):
        if not isinstance(other, Mesh):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self, *args, **kwargs):
        return f'Mesh(id="{self.id}", extent={str(self.extent)}, dimension={str(self.dimension)})'
