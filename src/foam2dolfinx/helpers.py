import numpy as np
from dolfinx.mesh import exterior_facet_indices
from scipy.spatial import cKDTree


def tag_boundary_patch(dolfinx_mesh, patch_dataset, patch_id, tol=1e-6):
    fdim = dolfinx_mesh.topology.dim - 1
    dolfinx_mesh.topology.create_connectivity(fdim, 0)
    dolfinx_mesh.topology.create_connectivity(0, fdim)
    dolfinx_mesh.topology.create_connectivity(fdim, dolfinx_mesh.topology.dim)

    facet_indices = exterior_facet_indices(dolfinx_mesh.topology)
    x = dolfinx_mesh.geometry.x
    patch_points = patch_dataset.points
    tree = cKDTree(x)
    matched_vertex_indices = tree.query_ball_point(patch_points, tol)
    matched_vertex_indices = list(set(i for sub in matched_vertex_indices for i in sub))

    matched_facets = []
    for facet in facet_indices:
        vertices = dolfinx_mesh.topology.connectivity(fdim, 0).links(facet)
        if all(v in matched_vertex_indices for v in vertices):
            matched_facets.append(facet)

    # print(f"Tagging {len(matched_facets)} facets for patch ID {patch_id}")
    return np.array(matched_facets, dtype=np.int32), np.full(
        len(matched_facets), patch_id, dtype=np.int32
    )
