import numpy as np
from dolfinx.mesh import exterior_facet_indices
from scipy.spatial import cKDTree


def tag_boundary_patch(
    dolfinx_mesh,
    patch_dataset,
    patch_id,
    tol=1e-6,
    *,
    tree=None,
    facet_indices=None,
    facet_vertices=None,
):
    """Tags the facets of a dolfinx mesh that belong to a given OpenFOAM boundary patch.

    Args:
        dolfinx_mesh: the dolfinx mesh
        patch_dataset: the pyvista dataset for the boundary patch
        patch_id: integer tag to assign to matched facets
        tol: spatial tolerance for matching patch points to mesh vertices
        tree: optional pre-built cKDTree on mesh geometry points — pass this when
            calling for multiple patches to avoid rebuilding the tree each time
        facet_indices: optional pre-computed exterior facet indices — pass together
            with facet_vertices when calling for multiple patches
        facet_vertices: optional pre-computed 2D array (n_facets, n_verts_per_facet)
            of vertex indices for each exterior facet

    Returns:
        tuple of (matched_facet_indices, tags) as int32 arrays
    """
    fdim = dolfinx_mesh.topology.dim - 1

    if facet_indices is None or facet_vertices is None:
        dolfinx_mesh.topology.create_connectivity(fdim, 0)
        dolfinx_mesh.topology.create_connectivity(0, fdim)
        dolfinx_mesh.topology.create_connectivity(fdim, dolfinx_mesh.topology.dim)
        facet_indices = exterior_facet_indices(dolfinx_mesh.topology)
        c_to_v = dolfinx_mesh.topology.connectivity(fdim, 0)
        facet_vertices = np.vstack([c_to_v.links(f) for f in facet_indices])

    if tree is None:
        tree = cKDTree(dolfinx_mesh.geometry.x)

    matched = tree.query_ball_point(patch_dataset.points, tol)
    matched_idx = np.unique(
        np.fromiter((i for sub in matched for i in sub), dtype=np.intp)
    )

    # boolean mask over all vertices, then check all vertices of each facet at once
    vertex_matched = np.zeros(len(dolfinx_mesh.geometry.x), dtype=bool)
    if len(matched_idx):
        vertex_matched[matched_idx] = True

    facet_mask = vertex_matched[facet_vertices].all(axis=1)
    matched_facets = facet_indices[facet_mask]

    return matched_facets, np.full(len(matched_facets), patch_id, dtype=np.int32)
