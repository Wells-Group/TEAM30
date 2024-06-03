FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

COPY pyproject.toml pyproject.toml

RUN python3 -m pip install .