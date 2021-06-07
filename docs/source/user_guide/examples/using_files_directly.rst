:orphan:

Using Files Directly
********************

You can load data directly from csv and `ARFF <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_ files.
For ARFF files, GAMA can utilize extra information given, such as which features are categorical.
For csv files GAMA will infer column types, but this might lead to mistakes.
In the example below, make sure to replace the file paths to the files to be used.
The example script can be run by using e.g.
`breast_cancer_train.arff <https://github.com/openml-labs/gama/tree/master/gama/tests/data/breast_cancer_train.arff>`_ and
`breast_cancer_test.arff <https://github.com/openml-labs/gama/tree/master/gama/tests/data/breast_cancer_test.arff>`_.
The target should always be specified as the last column, unless the `target_column` is specified.
Make sure you adjust the file path if not executed from the examples directory.


.. file below is copied in by conf.py
.. literalinclude:: /user_guide/examples/arff_example.py

The GamaRegressor also has csv and ARFF support.

The advantage of using an ARFF file over something like a numpy-array or a csv file is that attribute types are specified.
When supplying only numpy-arrays (e.g. through ``fit(X, y)``), GAMA can not know if a particular feature is nominal or numeric.
This means that GAMA might use a wrong feature transformation for the data (e.g. one-hot encoding on a numeric feature or scaling on a categorical feature).
Note that this is not unique to GAMA, but any framework which accepts numeric input without meta-data.

.. note::
    Unfortunately the ``date`` and ``string`` formats the ARFF file allows is not (fully) supported in GAMA yet,
    for the latest news, see `issue#2 <https://github.com/openml-labs/gama/issues/2>`_.
