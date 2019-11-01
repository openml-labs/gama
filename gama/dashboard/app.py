from bisect import bisect
from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq

import pandas as pd

from gama.dashboard.pages.base_page import BasePage

dashboard = dash.Dash('GamaDashboard')
dashboard.config.suppress_callback_exceptions = True


# === Construct UI elements ===

def build_app():
    from gama.dashboard.pages import pages
    base = create_generic_layout()
    base['tabs'].children = create_tabs(pages)
    return base


def create_generic_layout():
    """ Creates the generic layout of tabs and their content pages. """
    return html.Div(
        id="page",
        children=[
            html.Div(id="tabs"),
            html.Div(id="content")
        ]
    )


def create_tabs(pages: List[BasePage]):
    if pages == []:
        raise ValueError("Must have at least one tab.")
    # Sort pages by alignment
    sorted_pages = sorted(pages, key=lambda p: p.alignment)
    grouped_pages = ([page for page in sorted_pages if page.alignment >= 0] +
                     [page for page in sorted_pages if page.alignment < 0])
    tabs = [create_tab(page.name) for page in grouped_pages]
    return [dcc.Tabs(id="page-tabs", value=tabs[0].value, children=tabs)]


def create_tab(name: str):
    return dcc.Tab(
        id=f"{name}-tab",
        label=name,
        value=name,
        style={'width': '10%'}
    )


# === Callbacks ===

@dashboard.callback(
    [Output("content", "children")],
    [Input("page-tabs", "value")]
)
def display_page_content(page_name):
    from gama.dashboard.pages import pages
    page = [page for page in pages if page.name == page_name][0]
    return [page.build_page()]


if __name__ == '__main__':
    dashboard.layout = build_app()
    dashboard.run_server(debug=True)
