# TEAM-30 model
[![Time-domain verification](https://github.com/Wells-Group/TEAM30/actions/workflows/time-domain.yml/badge.svg)](https://github.com/Wells-Group/TEAM30/actions/workflows/time-domain.yml)

This repository contains a DOLFINx implementation of the [TEAM 30 model](http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf).

- `generate_team30_meshes.py`: A script that generates the two TEAM 30 models (single and three phase) meshes and saves them to xdmf format. To learn about input parameters, run `python3 generate_team30_meshes.py --help`.
- `team30_A_phi.py`: Script for solving the TEAM 30 model for either a single phase or three phase engine. To learn about input parameters, run `python3 team30_A_phi.py --help`.
- `parameteric_study.py`: Script for doing a parametric sweep for either model and comparing with reference data. To learn about input parameters, run `python3 parameteric_study.py --help`
- `utils.py`: File containing utillity functions used in the `team30_A_phi.py`, including post processing and quantities derived from Az
- `test_team30.py` Testing script verifying the single and three phase implementation for first and second order elements by comparing to reference data. Executed with `python3 -m pytest -xvs 

## Dependencies
The code relies on [DOLFINx](https://github.com/FEniCS/dolfinx/) which can for instanced by ran by using docker:
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared/ --shm-size=512m --name=dolfinx_main dolfinx/dolfinx
```
which can then be restarted at a later stage using
```bash
docker container start -i dolfinx_main
```

### Progress bar
We use `tqdm` for progress bar plots. This package can be installed with 
```bash
pip3 install tqdm
```
### Mesh generation
To generate the meshes, `gmsh>=4.8.0` is required, alongside with `mpi4py`, `h5py` and `meshio`. 
To install the `meshio` and `h5py` in the `DOLFINx` docker container call:
```bash
export HDF5_MPI="ON"
export CC=mpicc
export HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/"
pip3 install --no-cache-dir --no-binary=h5py h5py meshio
```

### Post-processing
We use `pandas` and `matplotlib` for post processing and comparison with reference data.
```bash
pip3 install pandas matplotlib
```