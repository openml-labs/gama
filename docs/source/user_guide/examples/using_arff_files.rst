:orphan:

Using ARFF files
****************

GAMA supports data in ARFF files directly, utilizing extra information given, such as which features are categorical.
In the example below, make sure to replace the file paths to the ARFF files to be used.
The example script can be run by using e.g.
`breast_cancer_train.arff <https://github.com/PGijsbers/gama/tree/master/gama/tests/data/breast_cancer_train.arff>`_ and
`breast_cancer_test.arff <https://github.com/PGijsbers/gama/tree/master/gama/tests/data/breast_cancer_test.arff>`_.
The target should always be specified as the last column.


.. file below is copied in by conf.py
.. literalinclude:: /user_guide/examples/arff_example.py

The GamaRegressor also has ARFF support.