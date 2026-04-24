import zipfile
from pathlib import Path

import pytest

from foam2dolfinx import OpenFOAMReader


@pytest.fixture
def two_regions_reader(tmp_path):
    zip_path = Path("test/data/test_2Regions.zip")
    extract_path = tmp_path / "test_2Regions"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)
    foam_file = extract_path / "test_2Regions/pv.foam"
    return OpenFOAMReader(filename=str(foam_file), cell_type=12)
