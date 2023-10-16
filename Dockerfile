FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt --user