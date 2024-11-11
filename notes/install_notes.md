# Installation notes

## Environment / prerequisites installation

### Creating napari conda environment
    conda create -n napari-env python=3.10
    conda activate napari-env
    conda install -c conda-forge pyqt napari

### Creating napari+ conda environment from file

The napari environment including all the packages required for this plugin can be created by using
the pre-defined environment.yml file:

    conda env create -f environment.yml

Napari can be tested by running:

    (conda activate napari-env)
    napari

Troubleshooting: On Windows machines ensure the environment is created in a place with write access, such as in the current user folder.
This can be forced using the prefix argument e.g.: 

    conda env create --prefix=C:/Users/<user>/.conda/envs/napari-env -f environment.yml

### Plugin installation of latest development version
    git clone git@github.com:FrancisCrickInstitute/napari-mass.git
    cd napari-mass
    pip install -e .

Troubleshooting: On Mac for probreg / open3d, to override the local clang compiler:

    conda install cxx-compiler

### References
* https://napari.org/stable/tutorials/fundamentals/installation.html
* https://napari.org/stable/plugins/find_and_install_plugin.html (https://napari.org/0.4.17/plugins/find_and_install_plugin.html)
* https://napari.org/stable/plugins/best_practices.html (https://napari.org/0.4.15/plugins/best_practices.html)
