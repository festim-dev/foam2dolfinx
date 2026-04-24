import re
import zipfile
from pathlib import Path

import pytest

from foam2dolfinx import OpenFOAMReader


def test_not_finding_function_cell_data(tmpdir):
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
        match=re.escape(
            "Function name: coucou not found in the subdomain: solid, in the OpenFOAM file. "
            "Available functions in subdomain: solid : ['T']"
        ),
    ):
        my_of_reader.create_dolfinx_function_with_cell_data(
            t=20.0, subdomain="solid", name="coucou"
        )


def test_not_finding_function_point_data(tmpdir):
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
        match=re.escape(
            "Function name: coucou not found in the subdomain: solid, in the OpenFOAM file. "
            "Available functions in subdomain: solid : ['T']"
        ),
    ):
        my_of_reader.create_dolfinx_function_with_point_data(
            t=20.0, subdomain="solid", name="coucou"
        )
