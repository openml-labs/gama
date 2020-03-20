:orphan:

Dashboard
---------

.. note::
    The GAMA Dashboard is not done.
    However, it is functional and released to get some early feedback.

GAMA Dashboard is a graphical user interface to start and monitor the AutoML search.
It is available when GAMA has been installed with its visualization optional dependencies (`pip install gama[vis]`).
To start GAMA Dashboard call `gamadash` from the command line.

Starting GAMA Dashboard will open a new tab in your webbrowser which will show the GAMA Dashboard Home page:

.. image:: images/DashboardHome.png

On the left you can configure GAMA, on the right you can select the dataset you want to perform AutoML on.
To provide a dataset, specify the path to the ARFF-file which contains your data.
Once the dataset has been set, the `Go!`-button on the bottom left will be enabled.
When you are satisfied with your chosen configuration, press the `Go!`-button to start GAMA.
This will take you to the 'Running' tab.

The running tab will look similar to this:

.. image:: images/DashboardRunning.png

You see four main components on this page:

 1. A visualization of search results. In this scatter plot, each scored pipeline is represented by a marker.
    The larger markers represent the most recent evaluations. Their location is determined by the pipeline's
    length (on the y-axis) and score (on the x-axis). You can hover over the markers to get precise scores,
    and click on the pipeline to select it. A selected pipeline is represented with a large red marker.

 2. Output of the search command. This field provides a textual progress report on GAMA's AutoML process.

 3. Full specification of the selected pipeline. This view of the selected pipeline specifies hyperparametersettings
    for each step in the pipeline.

 4. A table of each evaluated pipeline. Similar to the plot (1), here you find all pipelines evaluated during search.
    It is possible to sort and filter based on performance.

Selecting a pipeline in the table or through the plot will update the other components.
