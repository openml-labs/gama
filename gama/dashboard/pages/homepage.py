import multiprocessing
from typing import Optional

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State

from gama.dashboard.pages.base_page import BasePage


class HomePage(BasePage):
    callbacks = []

    def __init__(self):
        super().__init__(name='Home', alignment=0)

    def build_page(self, app: Optional = None):
        self._build_content()
        if app is not None:
            self._register_callbacks(app)

    def _build_content(self) -> html.Div:
        """ Build all the components of the page. """
        configuration = build_configuration_menu()
        configuration.style['width'] = '35%'
        configuration.style['float'] = 'left'
        data_navigator = build_data_navigator()
        data_navigator.style['width'] = '65%'
        data_navigator.style['float'] = 'right'
        self._content = html.Div(
            id="home-content",
            children=[
                configuration,
                data_navigator
            ]
        )
        return self._content

    def _register_callbacks(self, app):
        for (io, fn) in HomePage.callbacks:
            app.callback(*io)(fn)
        HomePage.callbacks = []


# === Configuration Menu ===

def cpu_slider():
    n_cpus = multiprocessing.cpu_count()
    id_ = 'cpu_slider'
    cpu_input = dbc.FormGroup(
        [
            dbc.Label("N Jobs", html_for=id_, width=5),
            dbc.Col(
                dcc.Slider(id=id_, min=1, max=n_cpus, updatemode='drag',
                           value=1, marks={1: '1', n_cpus: str(n_cpus)})
            )
        ],
        row=True
    )
    HomePage.callbacks.append(((
        Output(id_, "marks"),
        [Input(id_, "value")],
        [State(id_, "min"), State(id_, "max")]),
        update_marks
    ))
    return cpu_input


def time_nud(label_text: str, hour_id: str, hour_default: int, minute_id: str, minute_default: int):
    time_input = dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=hour_id, width=6),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Input(id=hour_id, type='number', min=0, max=99, step=1, value=hour_default),
                        dbc.InputGroupAddon("H", addon_type="append"),
                    ]
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Input(id=minute_id, type='number', min=0, max=59, step=1, value=minute_default),
                        dbc.InputGroupAddon("M", addon_type="append"),
                    ]
                )
            )
        ],
        row=True
    )
    return time_input


def markdown_header(text: str, level: int = 4, with_horizontal_rule: bool = True):
    hr = '\n---'
    markdown = f"{'#' * level} {text}{hr if with_horizontal_rule else ''}"
    return dcc.Markdown(markdown)


def build_configuration_menu() -> html.Div:
    cpu_input = cpu_slider()
    max_total_time_input = time_nud('Max Runtime',
                                    hour_id='max_total_h',
                                    hour_default=1,
                                    minute_id='max_total_m',
                                    minute_default=0)
    max_eval_time_input = time_nud('Max time per pipeline',
                                   hour_id='max_eval_h',
                                   hour_default=0,
                                   minute_id='max_eval_m',
                                   minute_default=5)
    return html.Div(
        children=[markdown_header('Configure GAMA', level=2),
                  markdown_header('Resources'),
                  dbc.Form([cpu_input, max_total_time_input, max_eval_time_input]),
                  markdown_header('Resources 2'),
                  dbc.Form([cpu_input, max_total_time_input, max_eval_time_input])],
        style={'box-shadow': '1px 1px 1px black', 'padding': '2%'}
    )


def update_marks(selected_value, min_, max_):
    return {min_: str(min_), selected_value: str(selected_value), max_: str(max_)}


def build_data_navigator() -> html.Div:
    return html.Div([html.P("Data Navigator")], style={'box-shadow': '1px 1px 1px black'})
