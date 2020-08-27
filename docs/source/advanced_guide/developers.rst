:orphan:

Developers Notes
----------------

.. note:: This is not set in stone. As more AutoML pipeline steps are added by more people,
    we expect to identify parts of the interface to be improved. We can't do this without your feedback!
    Feel free to get in touch, preferably in the form of a public discussion on a Github issue, and let us
    know what difficulties you encounter, or what works well!


General Overview
****************

To be written.


Adding a Search Method
**********************

A Search Method defines how to search for good machine learning pipelines.
For instance, it can iteratively design better pipelines by using evolutionary optimization.
To make search methods easily configurable, they must share a similar interface.
To achieve this, each search method must inherit from the `BaseSearch <https://github.com/PGijsbers/gama/blob/develop/gama/search_methods/base_search.py>`_ class.

Your custom search method should at least provide an implementation for the `search` method.

.. autoclass:: AsynchronousSuccessiveHalving

.. file below is copied in by conf.py
.. literalinclude:: ../../../gama/search_methods/base_search.py

Hello this is some text.
This is code::

    from something import thatotherthing

    for i in range(10):
        wheee!

Hello this is not code.

Adding a Post Processing Method
*******************************

Word.