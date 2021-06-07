..  _contributing_index:

Contributing
============

We greatly appreciate contributions and are happy to see you reading this section!
This section explains the general workflow and has some documentation for development.
If you need more information, please check out :ref:`contributing_howto`.
To ensure a pleasant environment for everyone we ask that all our contributors adhere to the `code of conduct <https://github.com/openml-labs/gama/blob/master/code_of_conduct.md>`_.

The best way to get started is by looking at the `Github Issues <https://github.com/openml-labs/gama/issues>`_.
See if there is an open issue you would like to work on.
If so, please comment on the issue first to make sure no double work is done.
If you are interested in adding or improving something unlisted, please open a new issue first.
Describe the planned contribution to see if it fits within our vision of the package,
and for discussion on its implementation.

Contributions should be developed on a separate branch created from the latest development branch.
When you are finished, make sure all the unit tests work and the documentation is updated before submitting a pull request.
In the pull request, mention the issue that discusses the contribution and please mention any additional information
that might be useful for reviewing the changes.

.. note::
    We recently `blackened <https://black.readthedocs.io/en/stable/>`_ our code.
    Before that we maintained a 120 character line limit, which far exceeds the limit
    of 88 by black. Refactoring the code to read well at 88 characters is an ongoing process.


.. include:: contributing.inc
    :start-line: 1
