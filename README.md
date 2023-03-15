# TEAM-30 model
[![Time-domain verification](https://github.com/Wells-Group/TEAM30/actions/workflows/time-domain.yml/badge.svg)](https://github.com/Wells-Group/TEAM30/actions/workflows/time-domain.yml)

This repository contains a DOLFINx implementation of the [TEAM 30 model](http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf).
## 2D modelling
- `generate_team30_meshes.py`: A script that generates the two TEAM 30 models (single and three phase) meshes and saves them to xdmf format. To learn about input parameters, run `python3 generate_team30_meshes.py --help`.
- `team30_A_phi.py`: Script for solving the TEAM 30 model for either a single phase or three phase engine. To learn about input parameters, run `python3 team30_A_phi.py --help`.
- `parameteric_study.py`: Script for doing a parametric sweep for either model and comparing with reference data. To learn about input parameters, run `python3 parameteric_study.py --help`
- `utils.py`: File containing utillity functions used in the `team30_A_phi.py`, including post processing and quantities derived from Az
- `test_team30.py` Testing script verifying the single and three phase implementation for first and second order elements by comparing to reference data. Executed with `python3 -m pytest -xvs 

## 3D modelling
- `generate_team30_meshes_3D.py`: A script that generates the two 3D TEAM 30 models (single and three phase) meshes and saves them to xdmf format. To learn about input parameters, run `python3 generate_team30_meshes_3D.py --help`.

## Installation
The list of requirements can be found in [requirements.txt](requirements.txt).

For an out of the box docker image, go to the [Github package](https://github.com/users/jorgensd/packages/container/package/dolfinx_team30).
 
The docker image can then be started with the following command:
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared/ --shm-size=512m --name=team30 ghcr.io/jorgensd/dolfinx_team30:v0.6.0
```
