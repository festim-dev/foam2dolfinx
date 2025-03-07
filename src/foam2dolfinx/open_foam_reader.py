from mpi4py import MPI

import basix
import numpy as np
import pyvista
import ufl
from dolfinx import fem
from dolfinx.mesh import create_mesh


class OpenFOAMReader:
    """
    Reads an OpenFOAM results file and converts the velocity data into a
    dolfinx.fem.Function

        Args:
            filename: the filename
            OF_mesh_type_value: cell type id (12 corresponds to HEXAHEDRON)

        Attributes:
            filename: the filename
            OF_mesh_type_value: cell type id (12 corresponds to HEXAHEDRON)
    """

    def __init__(self, filename, OF_mesh_type_value: int = 12):
        self.filename = filename
        self.OF_mesh_type_value = OF_mesh_type_value

        self.OF_mesh = None
        self.dolfinx_mesh = None

        self.reader = pyvista.POpenFOAMReader(self.filename)

    def _read_with_pyvista(self, t: float):
        self.reader.set_active_time_value(t)
        OF_multiblock = self.reader.read()
        self.OF_mesh = OF_multiblock["internalMesh"]

        # Dictionary mapping cell type to connectivity
        OF_cells_dict = self.OF_mesh.cells_dict

        self.OF_cells = OF_cells_dict.get(self.OF_mesh_type_value)
        if self.OF_cells is None:
            raise ValueError(
                f"No {self.OF_mesh_type_value} cells found in the mesh. Found "
                f"{OF_cells_dict.keys()}"
            )

    def _create_dolfinx_mesh(self):
        # Connectivity of the mesh (topology) - The second dimension indicates the type
        # of cell used

        args_conn = np.argsort(self.OF_cells, axis=1)
        rows = np.arange(self.OF_cells.shape[0])[:, None]
        self.connectivity = self.OF_cells[rows, args_conn]

        # Define mesh element
        if self.OF_mesh_type_value == 12:
            shape = "hexahedron"
        degree = 1
        cell = ufl.Cell(shape)
        self.mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        mesh_ufl = ufl.Mesh(self.mesh_element)

        # Create Dolfinx Mesh
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.connectivity, self.OF_mesh.points, mesh_ufl
        )
        self.dolfinx_mesh.topology.index_map(self.dolfinx_mesh.topology.dim).size_global

    def create_dolfinx_function(self, t=None) -> fem.Function:
        self._read_with_pyvista(t=t)
        self._create_dolfinx_mesh()
        self.function_space = fem.functionspace(self.dolfinx_mesh, self.mesh_element)
        u = fem.Function(self.function_space)

        num_vertices = (
            self.dolfinx_mesh.topology.index_map(0).size_local
            + self.dolfinx_mesh.topology.index_map(0).num_ghosts
        )
        vertex_map = np.empty(num_vertices, dtype=np.int32)
        c_to_v = self.dolfinx_mesh.topology.connectivity(
            self.dolfinx_mesh.topology.dim, 0
        )

        num_cells = (
            self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).size_local
            + self.dolfinx_mesh.topology.index_map(
                self.dolfinx_mesh.topology.dim
            ).num_ghosts
        )
        for cell in range(num_cells):
            vertices = c_to_v.links(cell)
            for i, vertex in enumerate(vertices):
                vertex_map[vertex] = self.connectivity[
                    self.dolfinx_mesh.topology.original_cell_index
                ][cell][i]

        u.x.array[:] = self.OF_mesh.point_data["U"][vertex_map].flatten()

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
    values = np.asarray(values)  # Ensure input is a NumPy array
    return values[np.abs(values - target).argmin()]
