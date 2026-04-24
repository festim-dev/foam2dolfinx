import numpy as np
import pytest
from pyvista import examples

from foam2dolfinx import OpenFOAMReader


# --- create_cell_meshtags: multidomain ---


def test_create_cell_meshtags_dim_equals_mesh_dim(two_regions_reader):
    ct = two_regions_reader.create_cell_meshtags()
    mesh = two_regions_reader.dolfinx_meshes_dict["_global"]
    assert ct.dim == mesh.topology.dim


def test_create_cell_meshtags_multidomain_fluid_count(two_regions_reader):
    ct = two_regions_reader.create_cell_meshtags()
    assert np.sum(ct.values == 1) == 2100


def test_create_cell_meshtags_multidomain_solid_count(two_regions_reader):
    ct = two_regions_reader.create_cell_meshtags()
    assert np.sum(ct.values == 2) == 945


def test_create_cell_meshtags_multidomain_no_untagged_cells(two_regions_reader):
    ct = two_regions_reader.create_cell_meshtags()
    assert np.all(ct.values > 0)


# --- create_cell_meshtags: single domain ---


def test_create_cell_meshtags_single_domain_dim_equals_mesh_dim():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    ct = reader.create_cell_meshtags(t=0)
    mesh = reader.dolfinx_meshes_dict["default"]
    assert ct.dim == mesh.topology.dim


def test_create_cell_meshtags_single_domain_all_values_are_one():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    ct = reader.create_cell_meshtags(t=0)
    assert np.all(ct.values == 1)


# --- create_facet_meshtags: multidomain ---


def test_create_facet_meshtags_dim_equals_fdim(two_regions_reader):
    ft = two_regions_reader.create_facet_meshtags()
    mesh = two_regions_reader.dolfinx_meshes_dict["_global"]
    assert ft.dim == mesh.topology.dim - 1


@pytest.mark.parametrize("tag, expected_count", [(1, 15), (2, 15), (3, 70)])
def test_create_facet_meshtags_boundary_patch_counts(
    two_regions_reader, tag, expected_count
):
    ft = two_regions_reader.create_facet_meshtags()
    assert np.sum(ft.values == tag) == expected_count


def test_create_facet_meshtags_interface_count(two_regions_reader):
    ft = two_regions_reader.create_facet_meshtags()
    # interface facets receive the highest tag (assigned after all boundary patches)
    assert np.sum(ft.values == ft.values.max()) == 60


def test_create_facet_meshtags_all_tag_values_positive(two_regions_reader):
    ft = two_regions_reader.create_facet_meshtags()
    assert np.all(ft.values > 0)


# --- create_facet_meshtags: single domain ---


def test_create_facet_meshtags_single_domain_dim_equals_fdim():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    ft = reader.create_facet_meshtags()
    mesh = reader.dolfinx_meshes_dict["default"]
    assert ft.dim == mesh.topology.dim - 1
