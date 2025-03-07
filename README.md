# foam2dolfinx

[![CI](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml/badge.svg)](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

foam2dolfinx is a tool for converting OpenFOAM output files to functions that can be used within [dolfinx](https://github.com/FEniCS/dolfinx).

> [!NOTE]  
> This small package was inspired by Stefano Riva's [ROSE-pyforce](https://github.com/ERMETE-Lab/ROSE-pyforce) repository.

## Installation

```python
conda env create -n my-env
conda activate my-env
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista
```