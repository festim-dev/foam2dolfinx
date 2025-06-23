from foam2dolfinx import OpenFOAMReader
import dolfinx
from pyvista import examples
import zipfile
from pathlib import Path


def test_reading_and_writing_cavity_example():
    my_of_reader = OpenFOAMReader(filename=examples.download_cavity(load=False))
    vel = my_of_reader.create_dolfinx_function(t=2.5)

    assert isinstance(vel, dolfinx.fem.Function)


def test_baby_example(tmpdir):
    time = 2812.0

    zip_path = Path("test/data/baby_example.zip")
    extract_path = Path(tmpdir) / "baby_example"

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Construct the path to the .foam file
    foam_file = extract_path / "baby_example/pv.foam"

    # read the .foam file
    my_of_reader = OpenFOAMReader(filename=str(foam_file), cell_type=10)

    vel = my_of_reader.create_dolfinx_function(t=time, name="U")
    T = my_of_reader.create_dolfinx_function(t=time, name="T")

    assert isinstance(vel, dolfinx.fem.Function)
    assert isinstance(T, dolfinx.fem.Function)


def test_hot_room(tmpdir):
    time = 2000.0

    zip_path = Path("test/data/hotRoom.zip")
    extract_path = Path(tmpdir) / "hotRoom"

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Construct the path to the .foam file
    foam_file = extract_path / "hotRoom/hotRoom.foam"

    # read the .foam file
    my_of_reader = OpenFOAMReader(filename=str(foam_file), cell_type=12)

    vel = my_of_reader.create_dolfinx_function(t=time, name="U")
    T = my_of_reader.create_dolfinx_function(t=time, name="T")
    nut = my_of_reader.create_dolfinx_function(t=time, name="nut")

    assert isinstance(vel, dolfinx.fem.Function)
    assert isinstance(T, dolfinx.fem.Function)
    assert isinstance(nut, dolfinx.fem.Function)
