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
    for page in pages:
        page.build_page(dashboard)
    return base


def create_generic_layout():
    """ Creates the generic layout of tabs and their content pages. """
    tab_banner_style = {
        'border-top-left-radius': '3px',
        'background-color': '#f9f9f9',
        'padding': '0px 24px',
        'border-bottom': '1px solid #d6d6d6'
    }

    return html.Div(
        id="page",
        children=[
            html.Div(id="tabs", style=tab_banner_style),
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
    tab_style = {
        'color': 'black',
        'width': '10%',
        'border-top-left-radius': '3px',  # round tab corners
        'border-top-right-radius': '3px',
        'border-bottom': '0px',  # bottom box-shadow still present
        'padding': '6px'
    }
    selected_tab_style = {
        ** tab_style,
        'border-top': '3px solid #c81818',  # Highlight color (TU/e colored)
        'box-shadow': '1px 1px 0px white'  # removes bottom edge
    }
    return dcc.Tab(
        id=f"{name}-tab",
        label=name,
        value=name,
        style=tab_style,
        selected_style=selected_tab_style
    )


# === Callbacks ===

@dashboard.callback(
    [Output("content", "children")],
    [Input("page-tabs", "value")]
)
def display_page_content(page_name):
    from gama.dashboard.pages import pages
    page = [page for page in pages if page.name == page_name][0]
    return [page.content]


if __name__ == '__main__':
    dashboard.layout = build_app()
    dashboard.run_server(debug=True)