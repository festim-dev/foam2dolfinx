# foam2dolfinx

[![CI](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml/badge.svg)](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

foam2dolfinx is a tool for converting OpenFOAM output files to functions that can be used within [dolfinx](https://github.com/FEniCS/dolfinx).

> [!NOTE]  
> This small package was inspired by Stefano Riva's [ROSE-pyforce](https://github.com/ERMETE-Lab/ROSE-pyforce) repository.

## Installation

```python
conda create -n my-env
conda activate my-env
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista
```

## Example usage

> [!NOTE]  
> Currently only domains with a unique cell type across the domain are supported. Furthermore, only vtk type cells 10 - tetrahedron and 12 - hexhedron are supported.

Consider a case where you want to read the velocity and temperature fields from a domain with tetrahedron cells at a time of 100s:

```python
from foam2dolfinx import OpenFOAMReader

my_reader = OpenFOAMReader(filename="my_file.foam", cell_type=10)

my_dolfinx_func = my_reader.create_dolfinx_function(t=100, name="T")
```