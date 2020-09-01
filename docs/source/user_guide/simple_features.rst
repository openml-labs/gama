:orphan:

Simple Features
---------------
This section features a couple of simple to use features that might be interesting for a wide audience.
For more advanced features, see the :ref:`advanced_guide_index`.

Command Line Interface
**********************

GAMA may also be called from a terminal, but the tool currently supports only part of all Python functionality.
In particular it can only load data from `.csv` or `.arff` files and AutoML pipeline configuration is not available.
The tool will produce a single pickled scikit-learn model (by default named 'gama_model.pkl'),
code export is also available.
Please see `gama -h` for all options.

Code Export
***********
It is possible to have GAMA export the final model definition as a Python file, see :meth:`gama.Gama.export_script`.
