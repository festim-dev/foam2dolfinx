import zipfile
from pathlib import Path

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
        filename=examples.download_cavity(load=False), cell_type=1
    )

    with pytest.raises(
        ValueError,
        match=r"No cell type 1 found in the mesh\. Found \[.*12]",
    ):
        my_reader._read_with_pyvista(t=0)


def test_error_rasied_when_subdomain_is_not_given_in_multidomain_case(tmpdir):
    zip_path = Path("test/data/test_2Regions.zip")
    extract_path = Path(tmpdir) / "test_2Regions"

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Construct the path to the .foam file
    foam_file = extract_path / "test_2Regions/pv.foam"

    # read the .foam file
    my_of_reader = OpenFOAMReader(filename=str(foam_file), cell_type=12)

    with pytest.raises(
        ValueError,
        match="Subdomain None not found in the OpenFOAM file\. Available subdomains: \['defaultRegion', 'fluid', 'solid']",
    ):
        my_of_reader._read_with_pyvista(t=20.0, subdomain=None)


@pytest.mark.parametrize("subdomain", ["fluid", "solid"])
def test_read_with_pyvista_finds_all_mesh_data(tmpdir, subdomain):
    zip_path = Path("test/data/test_2Regions.zip")
    extract_path = Path(tmpdir) / "test_2Regions"

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Construct the path to the .foam file
    foam_file = extract_path / "test_2Regions/pv.foam"

    # read the .foam file
    my_of_reader = OpenFOAMReader(filename=str(foam_file), cell_type=12)

    my_of_reader._read_with_pyvista(t=20.0, subdomain=subdomain)

    assert subdomain in my_of_reader.OF_meshes_dict
