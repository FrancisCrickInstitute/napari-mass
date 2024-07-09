# How to run the napari plugin in a container

## Requirements
- This repository (cloned locally)
- Docker
- X server (on Windows/Mac)

### Building the container
`docker build -t napari-mass .`

### Running the container
`docker run napari-mass:latest`
### Running the container using X server
`docker run -e DISPLAY=host.docker.internal:0 napari-mass:latest`

### References
- Github: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
- Docker: https://docs.docker.com/reference/cli/docker/image/build/
- X server: https://napari.org/stable/howtos/docker.html#base-napari-image
- napari & Docker: https://napari.org/stable/howtos/docker.html
