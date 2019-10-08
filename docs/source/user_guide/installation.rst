:orphan:

Installation
------------

For regular usage, you can install GAMA with pip::

    pip install gama

GAMA features optional dependencies for visualization and development.
You can install them with::

    pip install gama[OPTIONAL]

where `OPTIONAL` is one of:

 - `vis`: allows you to use the prototype dash app to visualize optimization traces.
 - `test`: to run GAMA's unit tests locally.
 - `doc`: to build docs locally.
 - `all`: all of the above.

To see exactly what dependencies will be installed, see `setup.py <https://github.com/PGijsbers/gama/blob/master/setup.py>`_.
If you plan on developing GAMA, cloning the repository and installing locally is advised::

    git clone https://github.com/PGijsbers/gama.git
    cd gama
    pip install -e .[all]

This installation will refer to your local GAMA files.
Changes to the code directly affect the installed GAMA package without reinstalling requiring a reinstall.
