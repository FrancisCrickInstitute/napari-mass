# Installation notes

## Environment / prerequisites installation

### Creating napari conda environment
    conda create -n napari-env python=3.10
    conda activate napari-env
    conda install -c conda-forge pyqt napari

### Installation of latest published version
In napari from the main menu select Plugins -> Install/Uninstall Plugins, then search for 'napari-mass' and install it.
It should then appear at the bottom of the Plugins menu.
Also see https://napari.org/stable/plugins/find_and_install_plugin.html

### Installation of latest development version
    git clone git@github.com:FrancisCrickInstitute/napari-mass.git
    cd napari-mass
    pip install -e .

For probreg / open3d on Mac (to override the local clang compiler):

    conda install cxx-compiler

### References
* https://napari.org/stable/plugins/find_and_install_plugin.html
* https://napari.org/stable/plugins/best_practices.html
