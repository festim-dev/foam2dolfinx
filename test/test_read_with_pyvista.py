import pytest
from pyvista import examples

from foam2dolfinx import OpenFOAMReader


def test_error_rasied_when_using_mixed_topology_mesh():
    my_reader = OpenFOAMReader(filename=examples.download_openfoam_tubes(load=False))

    with pytest.raises(
        NotImplementedError, match="Cannot support mixed-topology meshes"
    ):
        my_reader._read_with_pyvista(t=0)
