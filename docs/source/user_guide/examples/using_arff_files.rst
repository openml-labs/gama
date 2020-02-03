:orphan:

Using ARFF files
****************

GAMA supports data in `ARFF <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_ files directly, utilizing extra information given, such as which features are categorical.
In the example below, make sure to replace the file paths to the ARFF files to be used.
The example script can be run by using e.g.
`breast_cancer_train.arff <https://github.com/PGijsbers/gama/tree/master/gama/tests/data/breast_cancer_train.arff>`_ and
`breast_cancer_test.arff <https://github.com/PGijsbers/gama/tree/master/gama/tests/data/breast_cancer_test.arff>`_.
The target should always be specified as the last column.


.. file below is copied in by conf.py
.. literalinclude:: /user_guide/examples/arff_example.py

The GamaRegressor also has ARFF support.

The advantage of using an ARFF file over something like a numpy-array, is that attribute types are specified.
When supplying only numpy-arrays (e.g. through ``fit(X, y)``), GAMA can not know if a particular feature is ordinal or numeric.
This means that GAMA might use a wrong feature transformation for the data (e.g. one-hot encoding on a numeric feature or scaling on a categorical feature).
Note that this is not unique to GAMA, but any framework which accepts numeric input without meta-data.

Unfortunately the ``date`` and ``string`` formats the ARFF file allows is not (fully) supported in GAMA yet,
for the latest news, see `issue#2 <https://github.com/PGijsbers/gama/issues/2>`_.