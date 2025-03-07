from typing import Optional

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl  # type: ignore
from dolfinx.mesh import create_mesh

__all__ = ["OpenFOAMReader", "find_closest_value"]


class OpenFOAMReader:
    """
    Reads an OpenFOAM results file and converts the velocity data into a
    dolfinx.fem.Function

        Args:
            filename: the filename
            OF_mesh_cell_type_value: cell type id (12 corresponds to HEXAHEDRON)

        Attributes:
            filename: the filename
            OF_mesh_cell_type_value: cell type id (12 corresponds to HEXAHEDRON)
            reader: pyvista OpenFOAM reader for .foam files
            OF_mesh: the mesh from the openfoam file
            OF_cells: an array of the cells with associated vertices
            connectivity: The OpenFOAM mesh cell connectivity with vertices reordered
                in a sorted order for mapping with the dolfinx mesh.
            mesh_element: the basix element used in the dolfinx mesh
            dolfinx_mesh: the dolfinx mesh
            function_space: the function space of the dolfinx function returned in
                create_dolfinx_function()
    """

    filename: str
    OF_mesh_cell_type_value: int

    reader: pyvista.POpenFOAMReader
    OF_mesh: pyvista.pyvista_ndarray | pyvista.DataSet
    OF_cells: np.ndarray
    connectivity: np.ndarray
    mesh_element: basix.ufl._BlockedElement
    dolfinx_mesh: dolfinx.mesh.Mesh
    function_space: dolfinx.fem.FunctionSpace

    def __init__(self, filename, OF_mesh_cell_type_value: int = 12):
        self.filename = filename
        self.OF_mesh_cell_type_value = OF_mesh_cell_type_value

        self.reader = pyvista.POpenFOAMReader(self.filename)

    def _read_with_pyvista(self, t: float):
        """reads the filename dolfinx.fem.Function from the OpenFOAM file.

        Args:
            t: timestamp of the data to read
            name: Name of the field in the openfoam file, defaults to "U" for velocity

        Returns:
            the dolfinx function
        """
        self.reader.set_active_time_value(t)  # Set the time value to read data from
        OF_multiblock = self.reader.read()  # Read the data from the OpenFOAM file
        self.OF_mesh = OF_multiblock["internalMesh"]  # Extract the internal mesh

        # Dictionary mapping cell type to connectivity
        assert hasattr(self.OF_mesh, "cells_dict")  # Ensure the mesh has cell data
        OF_cells_dict = self.OF_mesh.cells_dict  # Get the cell dictionary

        self.OF_cells = OF_cells_dict.get(self.OF_mesh_cell_type_value)

        # Raise error if OF_mesh is mixed topology
        if len(OF_cells_dict.keys()) > 1:
            raise NotImplementedError("Cannot support mixed-topology meshes")

        # Raise error if no cells of the specified type are found in the OF_mesh
        if self.OF_cells is None:
            raise ValueError(
                f"No {self.OF_mesh_cell_type_value} cells found in the mesh. Found "
                f"{OF_cells_dict.keys()}"
            )

    def _create_dolfinx_mesh(self):
        """Creates a dolfinx.mesh.Mesh based on the elements within the OpenFOAM mesh"""

        # create the connectivity between the OpenFOAM and dolfinx meshes
        args_conn = np.argsort(self.OF_cells, axis=1)  # Sort the cell connectivity
        rows = np.arange(self.OF_cells.shape[0])[:, None]  # Create row indices
        self.connectivity = self.OF_cells[rows, args_conn]  # Reorder connectivity

        # Define mesh element
        if self.OF_mesh_cell_type_value == 12:
            shape = "hexahedron"
        elif self.OF_mesh_cell_type_value == 10:
            shape = "tetrahedron"
        else:
            raise ValueError(
                f"Cell type: {self.OF_mesh_cell_type_value}, not supported, please use"
                " either 12 (hexahedron) or 10 (tetrahedron) cells in OF mesh"
            )
        degree = 1  # Set polynomial degree
        cell = ufl.Cell(shape)
        self.mesh_vector_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )
        self.mesh_scalar_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=()
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(self.mesh_vector_element)
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.connectivity, self.OF_mesh.points, mesh_ufl
        )

    def create_dolfinx_function(
        self, t: Optional[float] = None, name: Optional[str] = "U"
    ) -> dolfinx.fem.Function:
        """Creates a dolfinx.fem.Function from the OpenFOAM file.

        Args:
            t: timestamp of the data to read
            name: Name of the field in the openfoam file, defaults to "U" for velocity

        Returns:
            the dolfinx function
        """
        self._read_with_pyvista(t=t)

        # Create dolfinx mesh if it doesn't exist
        if not hasattr(self, "dolfinx_mesh"):
            self._create_dolfinx_mesh()

        if name == "U":
            element = self.mesh_vector_element
        else:
            element = self.mesh_scalar_element

        self.function_space = dolfinx.fem.functionspace(self.dolfinx_mesh, element)
        u = dolfinx.fem.Function(self.function_space)

        num_vertices = (
            self.dolfinx_mesh.topology.index_map(0).size_local
            + self.dolfinx_mesh.topology.index_map(0).num_ghosts
        )
        vertex_map = np.empty(num_vertices, dtype=np.int32)

        # Get cell-to-vertex connectivity
        c_to_v = self.dolfinx_mesh.topology.connectivity(
            self.dolfinx_mesh.topology.dim, 0
        )

        # Map the OF_mesh vertices to dolfinx_mesh vertices
        num_cells = (
            self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).size_local
            + self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).num_ghosts
        )
        vertices = np.array([c_to_v.links(cell) for cell in range(num_cells)])
        flat_vertices = np.concatenate(vertices)
        cell_indices = np.repeat(np.arange(num_cells), [len(v) for v in vertices])
        vertex_positions = np.concatenate([np.arange(len(v)) for v in vertices])

        vertex_map[flat_vertices] = self.connectivity[
            self.dolfinx_mesh.topology.original_cell_index
        ][cell_indices, vertex_positions]

        # Assign values in OF_mesh to dolfinx_mesh
        assert hasattr(self.OF_mesh, "point_data")
        u.x.array[:] = self.OF_mesh.point_data[name][vertex_map].flatten()

        return u


def find_closest_value(values: list[float], target: float) -> float:
    """
    Finds the closest value in a NumPy array of floats to a given target float.

    Parameters:
        values (np.ndarray): Array of float values.
        target (float): The target float value.

    Returns:
        float: The closest value from the array.
    """
    values_ = np.asarray(values)  # Ensure input is a NumPy array
    return values_[np.abs(values_ - target).argmin()]
