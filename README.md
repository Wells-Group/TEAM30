# TEAM-30 model

This repository contains a DOLFINx implementation of the [TEAM 30 model](http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf).

- `generate_team30_meshes.py`: A script that generates the two TEAM 30 models (single and three phase) meshes and saves them to xdmf format. To learn about input paramemters, run `python3 generate_team30_meshes.py --help`


## Dependencies
To generate the meshes, `gmsh>=4.8.0` is required, alongside with `mpi4py`, `h5py` and `meshio`. 
To install the `meshio` and `h5py` in the `DOLFINx` docker container call:
```bash
export HDF5_MPI="ON"
export CC=mpicc
export HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/"
pip3 install --no-cache-dir --no-binary=h5py h5py meshio
```