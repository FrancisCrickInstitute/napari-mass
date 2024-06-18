FROM ghcr.io/napari/napari:latest

#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt

COPY . napari-mass

WORKDIR /napari-mass

RUN pip install .

#ENTRYPOINT /bin/bash
