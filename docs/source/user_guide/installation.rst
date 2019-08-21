:orphan:

Installation
------------

GAMA makes use of optional dependencies for its test environment (``test``) and its dash app (``vis``).
To install GAMA first clone the repository::

    git clone https://github.com/PGijsbers/gama.git
    cd gama

Then install GAMA with optional dependencies as desired.
Installing only the required dependencies allows you to use all of GAMA's AutoML functionality::

    pip install -e .

Installing the visualization dependencies additionally allows you to use the prototype dash app to visualize optimization traces::

    pip install -e .[vis]

If you plan on developing GAMA, install the test environment::

    pip install -e .[test]

To see what dependencies will be installed, see `setup.py <https://github.com/PGijsbers/gama/blob/master/setup.py>`_.

