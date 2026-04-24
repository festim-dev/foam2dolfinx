import numpy as np
from scipy.spatial import cKDTree


def tag_boundary_patch(
    dolfinx_mesh,
    patch_dataset,
    patch_id,
    tol=1e-6,
    *,
    tree: cKDTree,
    facet_indices: np.ndarray,
    facet_vertices: np.ndarray,
):
    """Tags the facets of a dolfinx mesh that belong to a given OpenFOAM boundary patch.

    Args:
        dolfinx_mesh: the dolfinx mesh
        patch_dataset: the pyvista dataset for the boundary patch
        patch_id: integer tag to assign to matched facets
        tol: spatial tolerance for matching patch points to mesh vertices
        tree: pre-built cKDTree on mesh geometry points
        facet_indices: pre-computed exterior facet indices
        facet_vertices: pre-computed 2D array (n_facets, n_verts_per_facet)
            of vertex indices for each exterior facet

    Returns:
        tuple of (matched_facet_indices, tags) as int32 arrays
    """
    matched = tree.query_ball_point(patch_dataset.points, tol)
    matched_idx = np.unique(
        np.fromiter((i for sub in matched for i in sub), dtype=np.intp)
    )

    vertex_matched = np.zeros(len(dolfinx_mesh.geometry.x), dtype=bool)
    if len(matched_idx):
        vertex_matched[matched_idx] = True

    facet_mask = vertex_matched[facet_vertices].all(axis=1)
    matched_facets = facet_indices[facet_mask]

    return matched_facets, np.full(len(matched_facets), patch_id, dtype=np.int32)
