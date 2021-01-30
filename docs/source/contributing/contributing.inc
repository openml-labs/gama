:orphan:

.. _contributing_howto:

Github workflow for your contribution
*************************************
The workflow GAMA maintains is described in more detail in this
`great blog post <https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/>`_
by `@scott_lowe <https://twitter.com/scott_lowe>`_.
Here follows a short version with little explanation, just some commands.

First fork the GAMA Github repository:

    - visit `GAMA's Github page <https://github.com/PGijsbers/gama>`_
    - click the 'fork' button in the top-right corner.

Clone *your fork* to your local machine.
Find the URL behind the green button 'clone or download', it should look like this::

    git clone https://github.com/GITHUB_USER_NAME/gama.git

Add the original GAMA repository as an additional remote::

    git remote add upstream https://github.com/PGijsbers/gama.git

Create a branch to work on your contribution (from the directory that has your cloned repository)::

    git checkout -b <your_branch_name>

Now you can start working on your contribution!
Push your changes back to your ``origin`` remote (the name of the automatically added remote)::

    git push origin <your_branch_name>

If you are ready to have your changes added to the GAMA repository, open a pull request.
This can be done using the Github website, where a button should pop up on your fork's repository.

When opening a pull request, make sure you have:

    - updated the documentation
    - added or updated tests where reasonable
    - if there is a loss of test coverage, you should provide a reason
    - refer to the issue and mention any additional information that might make reviewing the PR easier.

Setting up the development dependencies
***************************************
For development, you will need some additional packages for code style checks and running tests.
Navigate to your GAMA directory and run::

    pip install -e .[dev]

This installs test dependencies such as ``pytest`` and plugins.
It also installs `pre-commit <https://pre-commit.com/>`_,
a tool for automatically running several style checks before any commit.
Pre-commit will run `black <https://black.readthedocs.io/en/stable/>`_ code formatter,
`flake8 <https://flake8.pycqa.org/en/latest/>`_ style checker and
`mypy <https://mypy.readthedocs.io/en/stable/>`_ type checker before any commit.
While pre-commit is installed through the above command,
it itself needs to install the dependencies specified for our pre-commit workflow::

    pre-commit install

Run ``pre-commit run --all-files`` to see the pre-commit in action, it should output::

    black....................................................................Passed
    mypy.....................................................................Passed
    flake8...................................................................Passed

You should now be set up for development!
Pre-commit will run these tests whenever you attempt to make a commit.
If any of the checks fail, the commit will be aborted and you will be able to make the required changes before attempting to commit again.
Remember to add the updated file to your commit!
We use pre-commit to ensure a consistent code style and make PR reviews easier.
Note that pre-commit does not run any code tests.

Running code tests
******************
If you followed the instructions above, the required test dependencies should be installed on your system.
The tests are separated into unit and system tests, based on scope.

Unit tests are for quickly testable isolated functions.
Examples include creation of machine learning pipelines, selection procedures or data formatting.

System tests perform full AutoML runs on tiny datasets with limited resources.
They are meant to ensure that the individual snippets tested in unit tests are integrated well together.
It also functions as a sanity check that the system as a whole produces reasonable results.
While complete state coverage of the code will always be impossible for this type of system,
the system tests aim to cover as much as possible within a small time frame.

To run a test group run::

    pytest -sv -n 4 tests/SUITE

where ``SUITE`` is either ``unit`` or ``system``.
The ``-sv -n 4`` flags are optional,
but give succinct feedback on what is being run (``-sv``) and run tests in parallel (``-n 4``).
For more information, run ``pytest -h``.

To get a coverage report that reports which lines of code are (not) covered by tests,
also provide the arguments ``--cov=gama --cov-report term-missing``.
This can help identify if parts of your PR are not covered by unit tests.
When you generate a PR, a coverage report will also be generated and posted.
Should the coverage decline, the PR will not be accepted unless a good reason is given for the decreased coverage.

Some unit tests require files.
The paths to these files are specified in a way that assume the repository root directory as the working directory.
For example, the tests in `tests/unit/test_data.py` refer to the file `tests/unit/data/openml_d_23380.csv`.
When running your unit tests from an IDE, this may cause the files not to be found if it is not the default working directory for tests.
In PyCharm, go to `Run > Edit Configurations... > Templates > Python tests > Pytest `
and fill in `working directory` in the template to be the directory where `tests` resides (e.g. `~/gama`).

Generating Documentation
************************
The documentation is generated with `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
After updating the documentation, please build the new docs to verify everything works and looks as intended::

    sphinx-build -b html docs\source docs\build

The html pages are now in the ``gama\docs\build`` directory.
There is no need to upload the generated documentation.
A commit to master or a release branch will trigger the build and upload of the latest documentation.
