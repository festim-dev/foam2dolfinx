import pytest
from pyvista import examples

from foam2dolfinx import OpenFOAMReader


def test_error_rasied_when_using_mixed_topology_mesh():
    my_reader = OpenFOAMReader(filename=examples.download_openfoam_tubes(load=False))

    with pytest.raises(
        NotImplementedError, match="Cannot support mixed-topology meshes"
    ):
        my_reader._read_with_pyvista(t=0)


def test_error_rasied_when_cells_wanted_are_not_in_file_provided():
    my_reader = OpenFOAMReader(
        filename=examples.download_cavity(load=False), OF_mesh_cell_type_value=1
    )

    with pytest.raises(
        ValueError,
        match="No 1 cells found in the mesh. Found dict_keys([np.uint8(12)])",
    ):
        my_reader._read_with_pyvista(t=0)
