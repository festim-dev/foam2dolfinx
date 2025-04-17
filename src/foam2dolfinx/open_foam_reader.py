from typing import Optional, Dict

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
            cell_type: cell type id (12 corresponds to HEXAHEDRON)

        Attributes:
            filename: the filename
            cell_type: cell type id (12 corresponds to HEXAHEDRON)
            reader: pyvista OpenFOAM reader for .foam files
            multidomain: boolean indicating if the mesh is multi-domain
            OF_mesh: the mesh from the openfoam file
            OF_cells: an array of the cells with associated vertices
            connectivity: The OpenFOAM mesh cell connectivity with vertices reordered
                in a sorted order for mapping with the dolfinx mesh.
            dolfinx_mesh: the dolfinx mesh
            function_space: the function space of the dolfinx function returned in
                create_dolfinx_function()

        Notes:
            The cell type refers to the VTK cell type, a full list of cells and their
            respective integers can be found at: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    """

    filename: str
    cell_type: int

    reader: pyvista.POpenFOAMReader
    multidomain: bool
    OF_mesh: Dict[str, pyvista.pyvista_ndarray | pyvista.DataSet]
    OF_cells: Dict[str, np.ndarray]
    connectivity: Dict[str, np.ndarray]
    dolfinx_mesh: Dict[str, dolfinx.mesh.Mesh]
    function_space: Dict[str, dolfinx.fem.FunctionSpace]

    def __init__(self, filename, cell_type: int = 12):
        self.filename = filename
        self.cell_type = cell_type

        self.reader = pyvista.POpenFOAMReader(self.filename)
        self.multidomain = False
        self.OF_mesh = {}
        self.OF_cells_dict = {}
        self.OF_cells = {}
        self.dolfinx_mesh = {}

    @property
    def cell_type(self):
        return self._cell_type

    @cell_type.setter
    def cell_type(self, value):
        if not isinstance(value, int):
            raise TypeError("cell_type value should be an int")
        self._cell_type = value

    def _read_with_pyvista(self, t: float, subdomain: Optional[str] = "default"):
        """
        Reads the openfoam data in the filename provided, passes details of the
        openfoam mesh to OF_mesh and details of the cells to OF_cells.

        Args:
            t: timestamp of the data to read
            subdomain: Name of the subdmain in the openfoam file, from which a field is
                extracted

        """
        self.reader.set_active_time_value(t)  # Set the time value to read data from
        OF_multiblock = self.reader.read()  # Read the data from the OpenFOAM file

        # Check if the reader has a multiblock dataset block named "internalMesh"
        if "internalMesh" not in OF_multiblock.keys():
            self.multidomain = True
            if subdomain is None:
                raise ValueError("Subdomain must be specified for multi-domain meshes")
            if subdomain not in OF_multiblock.keys():
                raise ValueError(
                    f"Subdomain {subdomain} not found in the OpenFOAM file. "
                    f"Available subdomains: {OF_multiblock.keys()}"
                )

        # Extract the internal mesh
        if self.multidomain:
            for cell_array_name in OF_multiblock.keys():
                self.OF_mesh[cell_array_name] = OF_multiblock[cell_array_name][
                    "internalMesh"
                ]
        else:
            self.OF_mesh[subdomain] = OF_multiblock["internalMesh"]

        assert hasattr(
            self.OF_mesh[subdomain], "cells_dict"
        )  # Ensure the mesh has cell data
        self.OF_cells_dict[subdomain] = self.OF_mesh[
            subdomain
        ].cells_dict  # Get the cell dictionary

        self.OF_cells[subdomain] = self.OF_cells_dict[subdomain].get(self.cell_type)

        # Raise error if OF_mesh is mixed topology
        if len(self.OF_cells_dict[subdomain].keys()) > 1:
            raise NotImplementedError("Cannot support mixed-topology meshes")

        # Raise error if no cells of the specified type are found in the OF_mesh
        cell_types_in_mesh = [int(k) for k in self.OF_cells_dict[subdomain].keys()]
        if self.OF_cells[subdomain] is None:
            raise ValueError(
                f"No cell type {self.cell_type} found in the mesh. Found "
                f"{cell_types_in_mesh}"
            )

    def _create_dolfinx_mesh(self, subdomain: Optional[str] = "default"):
        """Creates a dolfinx.mesh.Mesh based on the elements within the OpenFOAM mesh"""

        # Define mesh element and define args conn based on the OF cell type
        if self.cell_type == 12:
            shape = "hexahedron"
            args_conn = np.tile(
                np.array([0, 1, 3, 2, 4, 5, 7, 6]), (len(self.OF_cells[subdomain]), 1)
            )

        elif self.cell_type == 10:
            shape = "tetrahedron"
            args_conn = np.argsort(
                self.OF_cells[subdomain], axis=1
            )  # Sort the cell connectivity

        else:
            raise ValueError(
                f"Cell type: {self.cell_type}, not supported, please use"
                " either 12 (hexahedron) or 10 (tetrahedron) cells in OF mesh"
            )

        # create the connectivity between the OpenFOAM and dolfinx meshes
        # Create row indices
        rows = np.arange(self.OF_cells[subdomain].shape[0])[:, None]
        # Reorder connectivity
        self.connectivity = self.OF_cells[subdomain][rows, args_conn]

        # Define mesh element
        if self.cell_type == 12:
            shape = "hexahedron"
        elif self.cell_type == 10:
            shape = "tetrahedron"
        else:
            raise ValueError(
                f"Cell type: {self.cell_type}, not supported, please use"
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
        self.dolfinx_mesh[subdomain] = create_mesh(
            MPI.COMM_WORLD, self.connectivity, self.OF_mesh[subdomain].points, mesh_ufl
        )

    def create_dolfinx_function(
        self, t: float, name: str = "U", subdomain: Optional[str] = "default"
    ) -> dolfinx.fem.Function:
        """Creates a dolfinx.fem.Function from the OpenFOAM file.

        Args:
            t: timestamp of the data to read
            name: Name of the field in the openfoam file, defaults to "U" for velocity
            subdomain: Name of the subdmain in the openfoam file, from which a field is
                extracted

        Returns:
            the dolfinx function
        """

        # read the openfoam data in the filename provided
        self._read_with_pyvista(t=t, subdomain=subdomain)

        # create the dolfinx mesh
        if subdomain not in self.dolfinx_mesh:
            self._create_dolfinx_mesh(subdomain=subdomain)

        mesh = self.dolfinx_mesh[subdomain]

        if name == "U":
            element = self.mesh_vector_element
        else:
            element = self.mesh_scalar_element

        function_space = dolfinx.fem.functionspace(mesh, element)
        u = dolfinx.fem.Function(function_space)

        num_vertices = (
            mesh.topology.index_map(0).size_local
            + mesh.topology.index_map(0).num_ghosts
        )
        vertex_map = np.empty(num_vertices, dtype=np.int32)

        # Get cell-to-vertex connectivity
        c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
        # Map the OF_mesh vertices to dolfinx_mesh vertices
        num_cells = (
            mesh.topology.index_map(mesh.topology.dim).size_local
            + mesh.topology.index_map(mesh.topology.dim).num_ghosts
        )
        vertices = np.array([c_to_v.links(cell) for cell in range(num_cells)])
        flat_vertices = np.concatenate(vertices)
        cell_indices = np.repeat(np.arange(num_cells), [len(v) for v in vertices])
        vertex_positions = np.concatenate([np.arange(len(v)) for v in vertices])

        vertex_map[flat_vertices] = self.connectivity[
            mesh.topology.original_cell_index
        ][cell_indices, vertex_positions]

        # Assign values in OF_mesh to dolfinx_mesh
        assert hasattr(self.OF_mesh[subdomain], "point_data")
        u.x.array[:] = self.OF_mesh[subdomain].point_data[name][vertex_map].flatten()

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
