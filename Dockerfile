# https://napari.org/stable/howtos/docker.html
# Note: requires X-launch server running on Windows/Mac

FROM ghcr.io/napari/napari:latest
#FROM ghcr.io/napari/napari-xpra:latest

COPY . napari-mass
WORKDIR napari-mass

RUN pip install --upgrade pip
RUN pip install .

#ENTRYPOINT /bin/bash
ENTRYPOINT napari -w napari-mass
