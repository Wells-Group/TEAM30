FROM dolfinx/dolfinx:nightly

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt --user