FROM dolfinx/dolfinx:v0.5.0

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --user