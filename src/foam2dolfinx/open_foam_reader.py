from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx.mesh import create_mesh, meshtags

from .helpers import tag_boundary_patch

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
            times: list of time values in the OpenFOAM file
            multidomain: boolean indicating if the mesh is multi-domain
            OF_meshes_dict: dictionary of meshes from the OpenFOAM file
            OF_cells_dict: dictionary of arrays of the cells with associated vertices
            connectivities_dict: dictionary of the OpenFOAM mesh cell connectivity with
                vertices reordered in a sorted order for mapping with the dolfinx mesh.
            dolfinx_meshes_dict: dictionary of dolfinx meshes
            vertex_maps_dict: dictionary of vertex maps per subdomain, mapping dolfinx
                vertex indices to the original OpenFOAM point indices. Built once per
                subdomain on the first call to create_dolfinx_function_with_point_data
                and cached for subsequent calls.

        Notes:
            The cell type refers to the VTK cell type, a full list of cells and their
            respective integers can be found at: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html

            If only one mesh is present in the OpenFOAM file, all data will be under the
            key: "default"
    """

    filename: str
    cell_type: int

    reader: pyvista.POpenFOAMReader
    times: list[float]
    multidomain: bool
    OF_meshes_dict: dict[str, pyvista.pyvista_ndarray | pyvista.DataSet]
    OF_cells_dict: dict[str, np.ndarray]
    connectivities_dict: dict[str, np.ndarray]
    dolfinx_meshes_dict: dict[str, dolfinx.mesh.Mesh]
    vertex_maps_dict: dict[str, np.ndarray]

    def __init__(self, filename, cell_type: int = 12):
        self.filename = filename
        self.cell_type = cell_type

        self.reader = pyvista.POpenFOAMReader(self.filename)
        self.OF_multiblock = None
        self.times = self.reader.time_values
        self.multidomain = False
        self.OF_meshes_dict = {}
        self.OF_cells_dict = {}
        self.connectivities_dict = {}
        self.dolfinx_meshes_dict = {}
        self.vertex_maps_dict = {}

    @property
    def cell_type(self):
        return self._cell_type

    @cell_type.setter
    def cell_type(self, value):
        if not isinstance(value, int):
            raise TypeError("cell_type value should be an int")
        self._cell_type = value

    def _read_with_pyvista(self, t: float, subdomain: str | None = "default"):
        """
        Reads the OpenFOAM data in the filename provided, passes details of the
        OpenFOAM mesh to OF_mesh and details of the cells to OF_cells.

        Args:
            t: timestamp of the data to read
            subdomain: Name of the subdmain in the OpenFOAM file, from which a field is
                extracted

        """
        self.reader.set_active_time_value(t)  # Set the time value to read data from
        self.OF_multiblock = self.reader.read()  # Read the data from the OpenFOAM file

        # Check if the reader has a multiblock dataset block named "internalMesh"
        if "internalMesh" not in self.OF_multiblock.keys():
            self.multidomain = True
            if subdomain not in self.OF_multiblock.keys():
                raise ValueError(
                    f"Subdomain {subdomain} not found in the OpenFOAM file. "
                    f"Available subdomains: {self.OF_multiblock.keys()}"
                )

        # Extract the internal mesh
        if self.multidomain:
            for cell_array_name in self.OF_multiblock.keys():
                self.OF_meshes_dict[cell_array_name] = self.OF_multiblock[
                    cell_array_name
                ]["internalMesh"]
        else:
            self.OF_meshes_dict[subdomain] = self.OF_multiblock["internalMesh"]

        # obtain dictionary of cell types in OF_mesh
        OF_cell_type_dict = self.OF_meshes_dict[subdomain].cells_dict

        cell_types_in_mesh = [int(k) for k in OF_cell_type_dict.keys()]

        # Raise error if OF_mesh is mixed topology
        if len(cell_types_in_mesh) > 1:
            raise NotImplementedError("Cannot support mixed-topology meshes")

        self.OF_cells_dict[subdomain] = OF_cell_type_dict.get(self.cell_type)

        # Raise error if no cells of the specified type are found in the OF_mesh
        if self.OF_cells_dict[subdomain] is None:
            raise ValueError(
                f"No cell type {self.cell_type} found in the mesh. Found "
                f"{cell_types_in_mesh}"
            )

    def _create_dolfinx_mesh(self, subdomain: str | None = "default"):
        """Creates a dolfinx.mesh.Mesh based on the elements within the OpenFOAM mesh"""

        # Define mesh element and define args conn based on the OF cell type
        if self.cell_type == 12:
            shape = "hexahedron"
            args_conn = np.tile(
                np.array([0, 1, 3, 2, 4, 5, 7, 6]),
                (len(self.OF_cells_dict[subdomain]), 1),
            )

        elif self.cell_type == 10:
            shape = "tetrahedron"
            args_conn = np.argsort(
                self.OF_cells_dict[subdomain], axis=1
            )  # Sort the cell connectivity

        else:
            raise ValueError(
                f"Cell type: {self.cell_type}, not supported, please use"
                " either 12 (hexahedron) or 10 (tetrahedron) cells in OF mesh"
            )

        # create the connectivity between the OpenFOAM and dolfinx meshes
        # Create row indices
        rows = np.arange(self.OF_cells_dict[subdomain].shape[0])[:, None]
        # Reorder connectivity
        self.connectivities_dict[subdomain] = self.OF_cells_dict[subdomain][
            rows, args_conn
        ]

        degree = 1  # Set polynomial degree
        cell = ufl.Cell(shape)
        # ufl.Cell.cellname became a property after dolfinx v0.10
        cell_name = cell.cellname() if callable(cell.cellname) else cell.cellname
        self.mesh_vector_element = basix.ufl.element(
            "Lagrange", cell_name, degree, shape=(3,)
        )
        self.mesh_scalar_element = basix.ufl.element(
            "Lagrange", cell_name, degree, shape=()
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(
            basix.ufl.element("Lagrange", cell_name, degree, shape=(3,))
        )
        self.dolfinx_meshes_dict[subdomain] = create_mesh(
            comm=MPI.COMM_WORLD,
            cells=self.connectivities_dict[subdomain],
            x=self.OF_meshes_dict[subdomain].points,
            e=mesh_ufl,
        )

    def _get_mesh(
        self, t: float, name: str, subdomain: str | None
    ) -> dolfinx.mesh.Mesh:
        """Reads pyvista data for the given time and subdomain, validates that the
        requested field exists, builds the dolfinx mesh on first call, and returns it.

        Args:
            t: timestamp of the data to read
            name: name of the field to validate against the available cell data keys
            subdomain: subdomain key in the OpenFOAM multiblock dataset

        Returns:
            the dolfinx mesh for this subdomain
        """
        self._read_with_pyvista(t=t, subdomain=subdomain)

        if name not in self.OF_meshes_dict[subdomain].cell_data.keys():
            raise ValueError(
                f"Function name: {name} not found in the subdomain: {subdomain}, "
                "in the OpenFOAM file. "
                f"Available functions in subdomain: {subdomain} : "
                f"{self.OF_meshes_dict[subdomain].cell_data.keys()}"
            )

        if subdomain not in self.dolfinx_meshes_dict:
            self._create_dolfinx_mesh(subdomain=subdomain)

        return self.dolfinx_meshes_dict[subdomain]

    def create_dolfinx_function_with_cell_data(
        self, t: float, name: str = "U", subdomain: str | None = "default"
    ) -> dolfinx.fem.Function:
        """Creates a dolfinx.fem.Function from the OpenFOAM file using cell data.

        Args:
            t: timestamp of the data to read
            name: Name of the field in the OpenFOAM file, defaults to "U" for velocity
            subdomain: Name of the subdmain in the OpenFOAM file, from which a field is
                extracted

        Returns:
            the dolfinx function
        """
        mesh = self._get_mesh(t, name, subdomain)

        data = self.OF_meshes_dict[subdomain].cell_data[name]
        # () for scalars, (3,) for vectors
        data_shape = data.shape[1:] if data.ndim > 1 else ()
        element = basix.ufl.element(
            "DG", mesh.topology.cell_name(), 0, shape=data_shape
        )

        function_space = dolfinx.fem.functionspace(mesh, element)
        u = dolfinx.fem.Function(function_space)
        u.x.array[:] = data[mesh.topology.original_cell_index].flatten()

        return u

    def create_dolfinx_function_with_point_data(
        self, t: float, name: str = "U", subdomain: str | None = "default"
    ) -> dolfinx.fem.Function:
        """Creates a dolfinx.fem.Function from the OpenFOAM file using point data.

        Args:
            t: timestamp of the data to read
            name: Name of the field in the OpenFOAM file, defaults to "U" for velocity
            subdomain: Name of the subdmain in the OpenFOAM file, from which a field is
                extracted

        Returns:
            the dolfinx function
        """
        mesh = self._get_mesh(t, name, subdomain)

        # select scalar or vector element based on data dimensionality
        data = self.OF_meshes_dict[subdomain].point_data[name]
        element = (
            self.mesh_vector_element if data.ndim > 1 else self.mesh_scalar_element
        )

        if subdomain not in self.vertex_maps_dict:
            num_vertices = (
                mesh.topology.index_map(0).size_local
                + mesh.topology.index_map(0).num_ghosts
            )
            vertex_map = np.empty(num_vertices, dtype=np.int32)
            c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
            num_cells = (
                mesh.topology.index_map(mesh.topology.dim).size_local
                + mesh.topology.index_map(mesh.topology.dim).num_ghosts
            )
            vertices = np.array([c_to_v.links(cell) for cell in range(num_cells)])
            flat_vertices = np.concatenate(vertices)
            cell_indices = np.repeat(np.arange(num_cells), [len(v) for v in vertices])
            vertex_positions = np.concatenate([np.arange(len(v)) for v in vertices])
            vertex_map[flat_vertices] = self.connectivities_dict[subdomain][
                mesh.topology.original_cell_index
            ][cell_indices, vertex_positions]
            self.vertex_maps_dict[subdomain] = vertex_map

        function_space = dolfinx.fem.functionspace(mesh, element)
        u = dolfinx.fem.Function(function_space)
        u.x.array[:] = (
            self.OF_meshes_dict[subdomain]
            .point_data[name][self.vertex_maps_dict[subdomain]]
            .flatten()
        )

        return u

    def create_facet_meshtags(
        self, t: float = 0, name: str = "U", subdomain: str | None = "default"
    ) -> dolfinx.mesh.MeshTags:
        """Creates a dolfinx.mesh.MeshTags for the facets of the mesh based on the
        boundary patches in the OpenFOAM file.

        Args:
            t: timestamp of the data to read, default to 0.
            name: Name of the field in the OpenFOAM file, defaults to "U" for velocity
            subdomain: Name of the subdmain in the OpenFOAM file, from which a field is
                extracted

        Returns:
            the dolfinx MeshTags for the facets of the mesh
        """
        mesh = self._get_mesh(t, name, subdomain)

        boundary = self.OF_multiblock["boundary"]

        all_facets = np.array([], dtype=np.int32)
        all_tags = np.array([], dtype=np.int32)

        for i, name in enumerate(boundary.keys()):
            facets, tags = tag_boundary_patch(mesh, boundary[name], i + 1)
            all_facets = np.concatenate([all_facets, facets])
            all_tags = np.concatenate([all_tags, tags])

            print(f"Tagging {len(facets)} facets for patch {name} with ID {i + 1}")

        facet_meshtags = meshtags(
            mesh,
            mesh.topology.dim - 1,
            all_facets,
            all_tags,
        )

        return facet_meshtags


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
