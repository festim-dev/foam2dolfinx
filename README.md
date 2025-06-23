# foam2dolfinx

[![CI](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml/badge.svg)](https://github.com/festim-dev/FESTIM/actions/workflows/ci.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

foam2dolfinx is a tool for converting OpenFOAM output files to functions that can be used within [dolfinx](https://github.com/FEniCS/dolfinx).

> [!NOTE]  
> This small package was inspired by Stefano Riva's [ROSE-pyforce](https://github.com/ERMETE-Lab/ROSE-pyforce) repository.

## Installation

```bash
conda create -n foam2dolfinx-env
conda activate foam2dolfinx-env
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista
```
Once in the created in environment:
```bash
pip install git+https://github.com/festim-dev/foam2dolfinx
```

# Example usage

## Standard case 

```python
from foam2dolfinx import OpenFOAMReader
from pyvista import examples

# use foam data from the examples in pyvista
foam_example = examples.download_cavity(load=False)

# instantiate reader:
my_reader = OpenFOAMReader(filename=foam_example, cell_type=10)

# read velocity field at t=2.5s
vel = my_of_reader.create_dolfinx_function(t=2.5, name="U")
```

> [!NOTE]  
> Currently only domains with a unique cell type across the domain are supported. Furthermore, only vtk type cells 10 - tetrahedron and 12 - hexhedron are supported.

## Multiple fields

Consider a case where in the same file there is both a temperature and velocity field to read at

```python
from foam2dolfinx import OpenFOAMReader

# instantiate reader:
my_reader = OpenFOAMReader(filename="my_local_file.foam")

# read velocity and temperature fields at t=1s
vel = my_of_reader.create_dolfinx_function(t=1.0, name="U")
T = my_of_reader.create_dolfinx_function(t=1.0, name="T")
```

## Multiple subdomains
```python
from foam2dolfinx import OpenFOAMReader

# instantiate reader:
my_reader =OpenFOAMReader(filename="my_local_file.foam")

# read velocity and temperature fields at t=1s
vel1 = my_of_reader.create_dolfinx_function(t=3.0, name="U", subdomain="sub1")
vel2 = my_of_reader.create_dolfinx_function(t=3.0, name="U", subdomain="sub2")
```

## Tips and tricks

If you are unaware of the time values with data within the OpenFOAM data, you can check with the `time_values` function within the 'reader' attribute of the 'OpenFOAMReader' class:

```python
from foam2dolfinx import OpenFOAMReader

# instantiate reader:
my_reader = OpenFOAMReader(filename="my_local_file.foam")

# find the time values
print(my_reader.reader.time_values)
```
This should return a list of floats with the time values in the file:
```
[1.0, 2.0, 3.0]
```
