import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq

import pandas as pd


dashboard = dash.Dash('GamaDashboard')
dashboard.config.suppress_callback_exceptions = True


def create_empty_page():
    return html.Div(
        id="no-pages",
        children=[
            html.P("Sorry, no pages found for this app.")
        ]
    )


if __name__ == '__main__':
    dashboard.layout = create_empty_page()
    dashboard.run_server(debug=True)
