:orphan:

Visualization
-------------

If you have installed GAMA with the optional visualization dependencies (``pip install -e .[vis]``),
you can start the dashboard like so::

    from gama.visualization import dash_app
    dash_app.run_server()

