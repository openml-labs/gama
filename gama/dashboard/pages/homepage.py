import multiprocessing
from typing import Optional

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
parameter_input_row = {'margin': '4%', 'line-height': 2}


def create_slider_input(id_: str, min_: int, max_: int, label: Optional[str] = None):
    HomePage.callbacks.append(((
            Output(id_, "marks"),
            [Input(id_, "value")],
            [State(id_, "min"), State(id_, "max")]),
            update_marks
    ))
    slider = daq.Slider(id=id_, min=min_, max=max_, updatemode='drag',
                        value=min_, marks={min_: min_, max_: max_})
    slider_div = html.Div(id=f"{id_}-slider-div", children=[slider])
    label_text = label if label is not None else id_
    label_div = html.Div(id=f"{id_}-label-div", children=label_text, style={'width': '46%', 'float': 'left'})
    return html.Div(id=f"{id_}-row", children=[label_div, slider_div], className="control-element")


def build_configuration_menu() -> html.Div:
    n_cpus = multiprocessing.cpu_count()
    cpu_slider = create_slider_input('n_jobs', 1, n_cpus, label='N Jobs')

    nud_width = 60
    max_time_label = html.Div(id='max_time_label', children='Max Runtime', style={'width': '46%', 'float': 'left', 'text-align': 'middle'})
    max_hours_input = daq.NumericInput(id='max_hours_input', max=99, size=nud_width)
    hours_input_div = html.Div(id='hours_input_div', children=[max_hours_input], style={'float': 'right', 'width': f'{nud_width}px'})
    hours_label = html.Label(id='hours_label', children='h', style={'float': 'right', 'width': '2%', 'vertical-align': 'center'})
    max_minutes_input = daq.NumericInput(id='max_minutes_input', max=59, size=nud_width)
    minutes_input_div = html.Div(id='minutes_input_div', children=[max_minutes_input], style={'float': 'right', 'width': f'{nud_width}px'})
    minutes_label = html.Label(id='minutes_label', children='m', style={'float': 'right', 'width': '5%', 'vertical-align': 'center'})

    time_row = html.Div(id='time_row',
                        children=[max_time_label, html.Div(), hours_input_div, hours_label, minutes_input_div, minutes_label],
                        className="control-element")

    return html.Div(
        children=[html.P("Configuration Menu"), cpu_slider, time_row],
        style={'box-shadow': '1px 1px 1px black'}
    )


def update_marks(selected_value, min_, max_):
    return {min_: min_, selected_value: selected_value, max_: max_}


def build_data_navigator() -> html.Div:
    return html.Div([html.P("Data Navigator")], style={'box-shadow': '1px 1px 1px black'})