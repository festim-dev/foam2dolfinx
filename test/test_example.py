from foam2dolfinx import OpenFOAMReader
import dolfinx
from pyvista import examples


def test_reading_and_writing_cavity_example():
    my_of_reader = OpenFOAMReader(filename=examples.download_cavity(load=False))
    vel = my_of_reader.create_dolfinx_function(t=2.5)

    assert isinstance(vel, dolfinx.fem.Function)


def test_baby_example():
    time = 2812.0
    my_of_reader = OpenFOAMReader(
        filename="test/data/baby_example/pv.foam", OF_mesh_type_value=10
    )

    vel = my_of_reader.create_dolfinx_function(t=time, name="U")
    T = my_of_reader.create_dolfinx_function(t=time, name="T")

    assert isinstance(vel, dolfinx.fem.Function)
    assert isinstance(T, dolfinx.fem.Function)
