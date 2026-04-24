from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx.mesh import create_mesh, exterior_facet_indices, meshtags
from scipy.spatial import cKDTree

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
            subdomain_cell_offsets: dictionary mapping each subdomain name to a
                (start, end) tuple of cell index ranges in the merged pyvista dataset.
                Populated by _create_global_dolfinx_mesh and used to assign subdomain
                IDs when building cell and interface facet meshtags.

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
    subdomain_cell_offsets: dict[str, tuple[int, int]]

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
        self.subdomain_cell_offsets = {}

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

    def _read_with_pyvista_all(self, t: float):
        """Reads the OpenFOAM multiblock data at time t for all subdomains.

        Populates OF_multiblock and OF_meshes_dict for every subdomain without
        requiring a specific subdomain to be named. For single-domain files,
        also populates OF_cells_dict["default"]. For multidomain files, cell
        data for each subdomain is read lazily by _create_global_dolfinx_mesh.

        Args:
            t: timestamp of the data to read
        """
        self.reader.set_active_time_value(t)
        self.OF_multiblock = self.reader.read()

        if "internalMesh" not in self.OF_multiblock.keys():
            self.multidomain = True
            for name in self.OF_multiblock.keys():
                self.OF_meshes_dict[name] = self.OF_multiblock[name]["internalMesh"]
        else:
            self.multidomain = False
            self.OF_meshes_dict["default"] = self.OF_multiblock["internalMesh"]
            OF_cell_type_dict = self.OF_meshes_dict["default"].cells_dict
            cell_types = [int(k) for k in OF_cell_type_dict.keys()]
            if len(cell_types) > 1:
                raise NotImplementedError("Cannot support mixed-topology meshes")
            self.OF_cells_dict["default"] = OF_cell_type_dict.get(self.cell_type)
            if self.OF_cells_dict["default"] is None:
                raise ValueError(
                    f"No cell type {self.cell_type} found in the mesh. "
                    f"Found {cell_types}"
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

    def _create_global_dolfinx_mesh(self):
        """Merges all subdomain pyvista meshes into a single global dolfinx mesh.

        Reads cell data for any subdomains not yet in OF_cells_dict, records
        cumulative cell offsets in subdomain_cell_offsets (used later for cell
        and interface facet meshtags), merges all pyvista meshes with clean()
        to deduplicate shared interface points, and creates a single dolfinx
        mesh stored under the key "_global" in dolfinx_meshes_dict.
        """
        subdomain_names = list(self.OF_meshes_dict.keys())

        # Populate OF_cells_dict for any subdomains not yet read
        for name in subdomain_names:
            if name not in self.OF_cells_dict:
                OF_cell_type_dict = self.OF_meshes_dict[name].cells_dict
                cell_types = [int(k) for k in OF_cell_type_dict.keys()]
                if len(cell_types) > 1:
                    raise NotImplementedError("Cannot support mixed-topology meshes")
                cells = OF_cell_type_dict.get(self.cell_type)
                if cells is None:
                    raise ValueError(
                        f"No cell type {self.cell_type} found in subdomain {name}. "
                        f"Found {cell_types}"
                    )
                self.OF_cells_dict[name] = cells

        # Record cumulative cell offsets before merging (cell order is preserved)
        cumulative = 0
        for name in subdomain_names:
            count = len(self.OF_cells_dict[name])
            self.subdomain_cell_offsets[name] = (cumulative, cumulative + count)
            cumulative += count

        # Merge all pyvista meshes; clean() deduplicates coincident interface points
        merged = pyvista.merge(
            [self.OF_meshes_dict[name] for name in subdomain_names]
        ).clean()

        OF_cells = merged.cells_dict.get(self.cell_type)

        if self.cell_type == 12:
            shape = "hexahedron"
            args_conn = np.tile(np.array([0, 1, 3, 2, 4, 5, 7, 6]), (len(OF_cells), 1))
        elif self.cell_type == 10:
            shape = "tetrahedron"
            args_conn = np.argsort(OF_cells, axis=1)
        else:
            raise ValueError(
                f"Cell type: {self.cell_type}, not supported, please use"
                " either 12 (hexahedron) or 10 (tetrahedron) cells in OF mesh"
            )

        rows = np.arange(OF_cells.shape[0])[:, None]
        connectivity = OF_cells[rows, args_conn]
        self.connectivities_dict["_global"] = connectivity

        degree = 1
        cell = ufl.Cell(shape)
        # ufl.Cell.cellname became a property after dolfinx v0.10
        cell_name = cell.cellname() if callable(cell.cellname) else cell.cellname
        self.mesh_vector_element = basix.ufl.element(
            "Lagrange", cell_name, degree, shape=(3,)
        )
        self.mesh_scalar_element = basix.ufl.element(
            "Lagrange", cell_name, degree, shape=()
        )

        mesh_ufl = ufl.Mesh(
            basix.ufl.element("Lagrange", cell_name, degree, shape=(3,))
        )
        self.dolfinx_meshes_dict["_global"] = create_mesh(
            comm=MPI.COMM_WORLD,
            cells=connectivity,
            x=merged.points,
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

    def create_facet_meshtags(self, t: float | None = None) -> dolfinx.mesh.MeshTags:
        """Creates a dolfinx.mesh.MeshTags for all tagged facets of the mesh.

        For single-domain meshes, tags external boundary patches (IDs starting at 1).
        For multidomain meshes, also tags interface facets between subdomains with
        sequential IDs continuing from the last boundary patch ID.

        Args:
            t: timestamp of the data to read, defaults to the first available time.

        Returns:
            the dolfinx MeshTags for the facets of the mesh
        """
        t = t if t is not None else next(tv for tv in self.times if tv != 0)
        self._read_with_pyvista_all(t=t)

        if self.multidomain:
            if "_global" not in self.dolfinx_meshes_dict:
                self._create_global_dolfinx_mesh()
            mesh = self.dolfinx_meshes_dict["_global"]
        else:
            if "default" not in self.dolfinx_meshes_dict:
                self._create_dolfinx_mesh(subdomain="default")
            mesh = self.dolfinx_meshes_dict["default"]

        # Collect boundary patches — in multidomain files each subdomain block
        # holds its own "boundary" child; single-domain has one top-level "boundary"
        if self.multidomain:
            patches = []
            for sd_name in self.OF_meshes_dict.keys():
                sd_block = self.OF_multiblock[sd_name]
                if "boundary" in sd_block.keys():
                    boundary = sd_block["boundary"]
                    for patch_name in boundary.keys():
                        patches.append((patch_name, boundary[patch_name]))
        else:
            boundary = self.OF_multiblock["boundary"]
            patches = [(name, boundary[name]) for name in boundary.keys()]

        # build shared data once across all patches
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, 0)
        mesh.topology.create_connectivity(0, fdim)
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        facet_indices = exterior_facet_indices(mesh.topology)
        c_to_v = mesh.topology.connectivity(fdim, 0)
        facet_vertices = np.vstack([c_to_v.links(f) for f in facet_indices])
        tree = cKDTree(mesh.geometry.x)

        all_facets = []
        all_tags = []
        patch_summary = {}

        for i, (patch_name, patch_dataset) in enumerate(patches):
            facets, tags = tag_boundary_patch(
                mesh,
                patch_dataset,
                i + 1,
                tree=tree,
                facet_indices=facet_indices,
                facet_vertices=facet_vertices,
            )
            all_facets.append(facets)
            all_tags.append(tags)
            patch_summary[patch_name] = {"id": i + 1, "n_facets": len(facets)}

        print("Boundary patch summary:")
        for patch_name, info in patch_summary.items():
            print(f"  {patch_name}: id={info['id']}, n_facets={info['n_facets']}")

        next_tag = len(patches) + 1

        if self.multidomain:
            # Build cell subdomain tag array in dolfinx ordering
            total_pv_cells = sum(e for _, e in self.subdomain_cell_offsets.values())
            pv_cell_tags = np.zeros(total_pv_cells, dtype=np.int32)
            for j, (sd_name, (start, end)) in enumerate(
                self.subdomain_cell_offsets.items()
            ):
                pv_cell_tags[start:end] = j + 1
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            dolfinx_cell_tags = pv_cell_tags[
                mesh.topology.original_cell_index[:num_cells]
            ]

            # Find internal facets where adjacent cells belong to different subdomains
            f_to_c = mesh.topology.connectivity(fdim, mesh.topology.dim)
            num_facets = mesh.topology.index_map(fdim).size_local
            n_adj = np.array([len(f_to_c.links(f)) for f in range(num_facets)])
            internal_facet_ids = np.where(n_adj == 2)[0].astype(np.int32)
            cell_pairs = np.array([f_to_c.links(f) for f in internal_facet_ids])
            tags_a = dolfinx_cell_tags[cell_pairs[:, 0]]
            tags_b = dolfinx_cell_tags[cell_pairs[:, 1]]
            is_interface = tags_a != tags_b
            interface_facet_ids = internal_facet_ids[is_interface]
            tags_a_iface = tags_a[is_interface]
            tags_b_iface = tags_b[is_interface]

            # Assign one sequential tag per unique subdomain pair
            pair_to_tag: dict[tuple[int, int], int] = {}
            interface_tags = np.empty(len(interface_facet_ids), dtype=np.int32)
            for j in range(len(interface_facet_ids)):
                pair_key = (
                    min(tags_a_iface[j], tags_b_iface[j]),
                    max(tags_a_iface[j], tags_b_iface[j]),
                )
                if pair_key not in pair_to_tag:
                    pair_to_tag[pair_key] = next_tag
                    next_tag += 1
                interface_tags[j] = pair_to_tag[pair_key]

            all_facets.append(interface_facet_ids)
            all_tags.append(interface_tags)

            print("Interface summary:")
            sd_names = list(self.subdomain_cell_offsets.keys())
            for (id_a, id_b), tag in pair_to_tag.items():
                n_iface = int(np.sum(interface_tags == tag))
                print(
                    f"  {sd_names[id_a - 1]}-{sd_names[id_b - 1]}: "
                    f"id={tag}, n_facets={n_iface}"
                )

        all_facets_arr = np.concatenate(all_facets)
        all_tags_arr = np.concatenate(all_tags)
        sort_idx = np.argsort(all_facets_arr)
        return meshtags(
            mesh,
            fdim,
            all_facets_arr[sort_idx].astype(np.int32),
            all_tags_arr[sort_idx].astype(np.int32),
        )

    def create_cell_meshtags(self, t: float | None = None) -> dolfinx.mesh.MeshTags:
        """Creates a dolfinx.mesh.MeshTags for the cells of the mesh.

        For single-domain meshes, all cells are tagged with ID 1.
        For multidomain meshes, cells are tagged with their subdomain ID (1-indexed
        in the order subdomains appear in the OpenFOAM file).

        Args:
            t: timestamp of the data to read, defaults to the first available time.

        Returns:
            the dolfinx MeshTags for the cells of the mesh
        """
        t = t if t is not None else next(tv for tv in self.times if tv != 0)
        self._read_with_pyvista_all(t=t)

        if self.multidomain:
            if "_global" not in self.dolfinx_meshes_dict:
                self._create_global_dolfinx_mesh()
            mesh = self.dolfinx_meshes_dict["_global"]

            total_pv_cells = sum(e for _, e in self.subdomain_cell_offsets.values())
            pv_cell_tags = np.zeros(total_pv_cells, dtype=np.int32)
            for j, (_, (start, end)) in enumerate(self.subdomain_cell_offsets.items()):
                pv_cell_tags[start:end] = j + 1
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_tag_array = pv_cell_tags[mesh.topology.original_cell_index[:num_cells]]
        else:
            if "default" not in self.dolfinx_meshes_dict:
                self._create_dolfinx_mesh(subdomain="default")
            mesh = self.dolfinx_meshes_dict["default"]
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_tag_array = np.ones(num_cells, dtype=np.int32)

        cell_indices = np.arange(num_cells, dtype=np.int32)
        return meshtags(mesh, mesh.topology.dim, cell_indices, cell_tag_array)


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
