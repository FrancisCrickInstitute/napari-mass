FROM ghcr.io/napari/napari:0.4.19

COPY . napari-mass
WORKDIR napari-mass

RUN pip install --upgrade pip
RUN pip install .

ENTRYPOINT /bin/bash
#ENTRYPOINT napari -w napari-mass
