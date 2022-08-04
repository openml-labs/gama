:orphan:

Installation
------------

For regular usage, you can install GAMA with pip::

    pip install gama

GAMA features optional dependencies for visualization and development.
You can install them with::

    pip install gama[OPTIONAL]

where `OPTIONAL` is one or more (comma separated):

 - `vis`: allows you to use the prototype dash app to visualize optimization traces.
 - `dev`: sets up all required dependencies for development of GAMA.
 - `doc`: sets up all required dependencies for building documentation of GAMA.

To see exactly what dependencies will be installed, see `setup.py <https://github.com/openml-labs/gama/blob/master/setup.py>`_.
If you plan on developing GAMA, cloning the repository and installing locally with test and doc dependencies is advised::

    git clone https://github.com/PGijsbers/gama.git
    cd gama
    pip install -e ".[doc,test]"

This installation will refer to your local GAMA files.
Changes to the code directly affect the installed GAMA package without requiring a reinstall.
