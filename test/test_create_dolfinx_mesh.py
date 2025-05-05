import numpy as np
import pytest
from pyvista import examples
import re

from foam2dolfinx import OpenFOAMReader


def test_error_rasied_when_using_mixed_topology_mesh():
    test_value = 1
    my_reader = OpenFOAMReader(
        filename=examples.download_cavity(load=False),
        cell_type=test_value,
    )

    # Create a random number generator
    rng = np.random.default_rng()

    # Create a 400x8 array filled with random values
    my_reader.OF_cells_dict["default"] = rng.random((400, 8))

    error_message = "Cell type: 1, not supported, please use either 12 (hexahedron) or 10 (tetrahedron) cells in OF mesh"
    pattern = re.escape(error_message)

    with pytest.raises(
        ValueError,
        match=pattern,
    ):
        my_reader._create_dolfinx_mesh()
