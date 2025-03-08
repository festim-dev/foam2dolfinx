import numpy as np
import pytest
from pyvista import examples

from foam2dolfinx import OpenFOAMReader


@pytest.mark.parametrize("value", ["tetra", 1.0, np.array([1.0])])
def test_error_raised_when_cell_type_not_int(value):
    "test that an error is raised when an integer is not given as an arg to cell type"

    with pytest.raises(TypeError):
        OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=value)
