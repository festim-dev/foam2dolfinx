import dolfinx
import numpy as np
import pytest
from pyvista import examples

from foam2dolfinx import OpenFOAMReader


@pytest.fixture
def hex_reader():
    return OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)


@pytest.fixture
def tet_reader():
    return OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=10)


@pytest.mark.parametrize(
    "cell_type, n_verts, expected_shape",
    [(12, 8, "hexahedron"), (10, 4, "tetrahedron")],
)
def test_get_connectivity_returns_correct_shape_name(
    cell_type, n_verts, expected_shape
):
    reader = OpenFOAMReader(
        filename=examples.download_cavity(load=False), cell_type=cell_type
    )
    shape_name, _ = reader._get_connectivity(np.zeros((3, n_verts), dtype=int))
    assert shape_name == expected_shape


@pytest.mark.parametrize("cell_type, n_verts", [(12, 8), (10, 4)])
def test_get_connectivity_output_shape_matches_input(cell_type, n_verts):
    reader = OpenFOAMReader(
        filename=examples.download_cavity(load=False), cell_type=cell_type
    )
    cells = np.arange(5 * n_verts).reshape(5, n_verts)
    _, connectivity = reader._get_connectivity(cells)
    assert connectivity.shape == cells.shape


def test_get_connectivity_raises_for_unsupported_cell_type():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    reader._cell_type = 5
    with pytest.raises(ValueError, match="not supported"):
        reader._get_connectivity(np.zeros((3, 4), dtype=int))


def test_get_connectivity_hexahedron_applies_vtk_to_dolfinx_permutation(hex_reader):
    cells = np.arange(8).reshape(1, 8)
    _, connectivity = hex_reader._get_connectivity(cells)
    np.testing.assert_array_equal(connectivity[0], [0, 1, 3, 2, 4, 5, 7, 6])


def test_get_connectivity_tetrahedron_sorts_vertices(tet_reader):
    cells = np.array([[40, 10, 30, 20]])
    _, connectivity = tet_reader._get_connectivity(cells)
    np.testing.assert_array_equal(connectivity[0], [10, 20, 30, 40])


def test_build_dolfinx_mesh_returns_dolfinx_mesh(hex_reader):
    hex_reader._read_with_pyvista(t=0)
    shape, connectivity = hex_reader._get_connectivity(
        hex_reader.OF_cells_dict["default"]
    )
    result = hex_reader._build_dolfinx_mesh(
        hex_reader.OF_meshes_dict["default"].points, connectivity, shape
    )
    assert isinstance(result, dolfinx.mesh.Mesh)


@pytest.mark.parametrize("attr", ["mesh_vector_element", "mesh_scalar_element"])
def test_build_dolfinx_mesh_sets_element_attributes(hex_reader, attr):
    hex_reader._read_with_pyvista(t=0)
    shape, connectivity = hex_reader._get_connectivity(
        hex_reader.OF_cells_dict["default"]
    )
    hex_reader._build_dolfinx_mesh(
        hex_reader.OF_meshes_dict["default"].points, connectivity, shape
    )
    assert getattr(hex_reader, attr) is not None
