FROM ghcr.io/fenics/dolfinx/dolfinx:v0.6.0-r1

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt --user