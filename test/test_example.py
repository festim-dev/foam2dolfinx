from foam2dolfinx import OpenFOAMReader
import dolfinx
from pyvista import examples


def test_reading_and_writing_cavity_example():
    my_of_reader = OpenFOAMReader(filename=examples.download_cavity(load=False))
    vel = my_of_reader.create_dolfinx_function(t=2.5)

    assert isinstance(vel, dolfinx.fem.Function)


def test_reading_example_2():
    my_of_reader = OpenFOAMReader(filename=examples.download_openfoam_tubes(load=False))
    vel = my_of_reader.create_dolfinx_function(t=1000.0)

    assert isinstance(vel, dolfinx.fem.Function)
