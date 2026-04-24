# foam2dolfinx

[![NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/)
[![Conda CI](https://github.com/festim-dev/foam2dolfinx/actions/workflows/ci_conda.yml/badge.svg?branch=main)](https://github.com/festim-dev/foam2dolfinx/actions/workflows/ci_conda.yml)
[![Docker CI](https://github.com/festim-dev/foam2dolfinx/actions/workflows/ci_docker.yml/badge.svg?branch=main)](https://github.com/festim-dev/foam2dolfinx/actions/workflows/ci_docker.yml)
[![codecov](https://codecov.io/gh/festim-dev/foam2dolfinx/branch/main/graph/badge.svg?token=AK3A9CV2D3)](https://codecov.io/gh/festim-dev/foam2dolfinx)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/944519483.svg)](https://doi.org/10.5281/zenodo.17297276)

foam2dolfinx converts OpenFOAM output files into dolfinx meshes, meshtags, and functions for use in [DOLFINx](https://github.com/FEniCS/dolfinx)-based finite element workflows.

> [!NOTE]
> This package was inspired by Stefano Riva's [ROSE-pyforce](https://github.com/ERMETE-Lab/ROSE-pyforce) repository.

## Installation

```bash
conda create -n foam2dolfinx-env
conda activate foam2dolfinx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install foam2dolfinx
```

> [!NOTE]
> Only single cell-type meshes are supported. Supported VTK cell types: `10` (tetrahedron) and `12` (hexahedron).

## Usage

### Reading field data

Cell-centred data (e.g. from a finite volume solution) and point data are both supported:

```python
from foam2dolfinx import OpenFOAMReader

reader = OpenFOAMReader(filename="my_case.foam", cell_type=12)

# Cell data — maps to a DG-0 function space
T_cells = reader.create_dolfinx_function_with_cell_data(t=1.0, name="T")

# Point data — maps to a CG-1 function space
U_points = reader.create_dolfinx_function_with_point_data(t=1.0, name="U")
```

### Multi-domain cases

For multi-region OpenFOAM cases, pass the subdomain name to read fields from a specific region:

```python
reader = OpenFOAMReader(filename="my_case.foam", cell_type=12)

T_fluid = reader.create_dolfinx_function_with_cell_data(t=1.0, name="T", subdomain="fluid")
T_solid = reader.create_dolfinx_function_with_cell_data(t=1.0, name="T", subdomain="solid")
```

### Meshtags

Boundary patches and cell zones can be extracted as `dolfinx.mesh.MeshTags` for use in variational forms and boundary condition assignment.

For multi-domain cases, a single merged global mesh is built automatically. Interface facets between subdomains are detected and tagged with sequential IDs continuing after the boundary patch IDs.

```python
from mpi4py import MPI
from dolfinx.io import XDMFFile
from foam2dolfinx import OpenFOAMReader

reader = OpenFOAMReader(filename="my_case.foam", cell_type=12)

facet_tags = reader.create_facet_meshtags()
cell_tags  = reader.create_cell_meshtags()

mesh = reader.dolfinx_meshes_dict["_global"]  # single-domain: use "default"

with XDMFFile(MPI.COMM_WORLD, "facet_meshtags.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_meshtags(facet_tags, mesh.geometry)

with XDMFFile(MPI.COMM_WORLD, "cell_meshtags.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_meshtags(cell_tags, mesh.geometry)
```

Both methods print a summary to stdout showing each patch/zone name, its assigned integer ID, and how many facets or cells carry that tag.

### Finding available time values

```python
reader = OpenFOAMReader(filename="my_case.foam", cell_type=12)
print(reader.times)  # e.g. [0.0, 0.5, 1.0, 2.0]
```

To find the closest available time to a target value:

```python
from foam2dolfinx import find_closest_value

t = find_closest_value(reader.times, 1.3)  # returns 1.0 or 2.0
```
