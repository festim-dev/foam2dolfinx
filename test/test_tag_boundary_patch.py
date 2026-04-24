import dolfinx.mesh
import numpy as np
import pyvista
import pytest
from mpi4py import MPI
from scipy.spatial import cKDTree
from dolfinx.mesh import exterior_facet_indices

from foam2dolfinx.helpers import tag_boundary_patch


@pytest.fixture
def unit_cube_topology():
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [1, 1, 1]],
        [2, 2, 2],
        dolfinx.mesh.CellType.hexahedron,
    )
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(0, fdim)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facet_indices = exterior_facet_indices(mesh.topology)
    c_to_v = mesh.topology.connectivity(fdim, 0)
    facet_vertices = np.vstack([c_to_v.links(f) for f in facet_indices])
    tree = cKDTree(mesh.geometry.x)
    return mesh, tree, facet_indices, facet_vertices


def test_tag_boundary_patch_returns_empty_when_no_points_match(unit_cube_topology):
    mesh, tree, facet_indices, facet_vertices = unit_cube_topology
    patch = pyvista.PolyData(np.array([[100.0, 100.0, 100.0]]))
    matched, tags = tag_boundary_patch(
        mesh,
        patch,
        1,
        tree=tree,
        facet_indices=facet_indices,
        facet_vertices=facet_vertices,
    )
    assert len(matched) == 0
    assert len(tags) == 0


def test_tag_boundary_patch_tags_equal_patch_id(unit_cube_topology):
    mesh, tree, facet_indices, facet_vertices = unit_cube_topology
    x0_points = mesh.geometry.x[mesh.geometry.x[:, 0] < 1e-10]
    patch = pyvista.PolyData(x0_points)
    _, tags = tag_boundary_patch(
        mesh,
        patch,
        42,
        tree=tree,
        facet_indices=facet_indices,
        facet_vertices=facet_vertices,
    )
    assert np.all(tags == 42)


def test_tag_boundary_patch_matched_facets_are_subset_of_exterior(unit_cube_topology):
    mesh, tree, facet_indices, facet_vertices = unit_cube_topology
    x0_points = mesh.geometry.x[mesh.geometry.x[:, 0] < 1e-10]
    patch = pyvista.PolyData(x0_points)
    matched, _ = tag_boundary_patch(
        mesh,
        patch,
        1,
        tree=tree,
        facet_indices=facet_indices,
        facet_vertices=facet_vertices,
    )
    assert np.all(np.isin(matched, facet_indices))


@pytest.mark.parametrize(
    "patch_points",
    [
        np.array([[100.0, 100.0, 100.0]]),  # no match
        None,  # match (x=0 face, filled below)
    ],
)
def test_tag_boundary_patch_output_arrays_are_int32(unit_cube_topology, patch_points):
    mesh, tree, facet_indices, facet_vertices = unit_cube_topology
    if patch_points is None:
        patch_points = mesh.geometry.x[mesh.geometry.x[:, 0] < 1e-10]
    patch = pyvista.PolyData(patch_points)
    matched, tags = tag_boundary_patch(
        mesh,
        patch,
        1,
        tree=tree,
        facet_indices=facet_indices,
        facet_vertices=facet_vertices,
    )
    assert matched.dtype == np.int32
    assert tags.dtype == np.int32
