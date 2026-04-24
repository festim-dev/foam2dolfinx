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


def test_error_rasied_when_wrong_subdomain_given_in_multidomain_case(tmpdir):
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
        match=(
            r"Subdomain coucou not found in the OpenFOAM file\. "
            r"Available subdomains: \['fluid', 'solid']"
        ),
    ):
        my_of_reader._read_with_pyvista(t=20.0, subdomain="coucou")


def test_read_with_pyvista_no_subdomain_single_domain_populates_cells():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    reader._read_with_pyvista(t=0)
    assert "default" in reader.OF_cells_dict
    assert reader.OF_cells_dict["default"] is not None


def test_read_with_pyvista_no_subdomain_single_domain_sets_multidomain_false():
    reader = OpenFOAMReader(filename=examples.download_cavity(load=False), cell_type=12)
    reader._read_with_pyvista(t=0)
    assert reader.multidomain is False


def test_read_with_pyvista_no_subdomain_multidomain_populates_all_meshes(
    two_regions_reader,
):
    two_regions_reader._read_with_pyvista(t=20.0)
    assert "fluid" in two_regions_reader.OF_meshes_dict
    assert "solid" in two_regions_reader.OF_meshes_dict


def test_read_with_pyvista_no_subdomain_multidomain_excludes_defaultRegion(
    two_regions_reader,
):
    two_regions_reader._read_with_pyvista(t=20.0)
    assert "defaultRegion" not in two_regions_reader.OF_meshes_dict


def test_read_with_pyvista_no_subdomain_multidomain_does_not_populate_cells(
    two_regions_reader,
):
    two_regions_reader._read_with_pyvista(t=20.0)
    assert len(two_regions_reader.OF_cells_dict) == 0


def test_read_with_pyvista_no_subdomain_multidomain_sets_multidomain_true(
    two_regions_reader,
):
    two_regions_reader._read_with_pyvista(t=20.0)
    assert two_regions_reader.multidomain is True


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
