# Installation notes

## Environment / prerequisites installation

### Creating napari conda environment
    conda create -n napari-env python=3.9
    conda activate napari-env
    conda install -c conda-forge pyqt napari

### Installation of latest development version
    git clone git@github.com:FrancisCrickInstitute/napari-mass.git
    cd napari-mass
    pip install -e .

Don’t include PySide2 or PyQt5 in your plugin’s dependencies!
(https://napari.org/stable/plugins/best_practices.html)

For probreg / open3d on Mac (to override the local clang compiler):

    conda install cxx-compiler
