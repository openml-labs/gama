import multiprocessing
import os
import shlex
import subprocess
from typing import Optional, List, Dict

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
            dbc.Label("N Jobs", html_for=id_, width=6),
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
    return dbc.FormGroup(
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


def toggle_button(label_text: str, id_: str, start_on: bool = True):
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=id_, width=6),
            dbc.Col(dbc.Checklist(id=id_, options=[{'label': '', 'value': 'on'}], switch=True,
                                  value='on' if start_on else 'off')),
        ],
        row=True
    )


def dropdown(label_text: str, id_: str, options: Dict[str, str], value: Optional[str] = None):
    """ options formatted as {LABEL_KEY: LABEL_TEXT, ...} """
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=id_, width=6),
            dbc.Col(
                dcc.Dropdown(
                    id=id_,
                    options=[
                        {'label': text, 'value': key}
                        for key, text in options.items()
                    ],
                    clearable=False,
                    value=value
                ),
            ),
        ],
        row=True
    )


def button_header(text: str, id_: str, level: int = 4):
    header = f"{'#' * level} {text}"
    return dbc.FormGroup([dbc.Button([dcc.Markdown(header)], id=id_, block=True, color='primary')])


def markdown_header(text: str, level: int = 4, with_horizontal_rule: bool = True):
    header = f"{'#' * level} {text}"
    hr = f"\n{'-'*(level + 1 + len(text))}"  # matching length '-' not required but nicer.
    return dcc.Markdown(f"{header}{hr if with_horizontal_rule else ''}")


def toggle_collapse(click, is_open: bool):
    if click:
        return not is_open
    return is_open


def collapsable_section(header: str, controls: List[dbc.FormGroup], start_open: bool = True):
    header_id = f"{header}-header"
    form_id = f"{header}-form"

    form_header = button_header(header, id_=header_id)
    collapsable_form = dbc.Collapse(
        id=form_id,
        children=[dbc.Form(controls)],
        is_open=start_open
    )

    HomePage.callbacks.append(((
        Output(form_id, "is_open"),
        [Input(header_id, "n_clicks")],
        [State(form_id, "is_open")]),
        toggle_collapse
    ))
    return form_header, collapsable_form


def start_gama(
        n_clicks, metric, regularize, n_jobs,
        max_total_time_h, max_total_time_m, max_eval_time_h, max_eval_time_m,
        input_file):
    # For some reason, 0 input registers as None.
    max_total_time_h = 0 if max_total_time_h is None else max_total_time_h
    max_total_time_m = 0 if max_total_time_m is None else max_total_time_m
    max_eval_time_h = 0 if max_eval_time_h is None else max_eval_time_h
    max_eval_time_m = 0 if max_eval_time_m is None else max_eval_time_m
    max_total_time = (max_total_time_h * 60 + max_total_time_m)
    max_eval_time = (max_eval_time_h * 60 + max_eval_time_m)
    command = f'gama "{input_file}" -n {n_jobs} -t {max_total_time} --time_pipeline {max_eval_time}'
    if regularize != 'on':
        command += ' --long'
    if metric != 'default':
        command += f' -m {metric}'
    print('calling ', command)
    command = shlex.split(command)
    process = subprocess.Popen(command, shell=True)
    return 'danger', dcc.Markdown("#### Stop!")


def build_configuration_menu() -> html.Div:
    # Optimization
    from gama.utilities.metrics import all_metrics
    metrics = {m: m.replace('_', ' ') for m in all_metrics}
    metrics.update({'default': 'default'})
    scoring_input = dropdown('Metric', 'metric_dropdown', options=metrics, value='default')
    regularize_input = toggle_button('Prefer short pipelines', 'regularize_length_switch')
    optimization = collapsable_section("Optimization", [scoring_input, regularize_input])

    # Resources
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
    resources = collapsable_section("Resources", [cpu_input, max_total_time_input, max_eval_time_input])

    # Advanced
    advanced = collapsable_section("Advanced", [], start_open=False)

    # Go!
    go_button = dbc.Button([dcc.Markdown("#### Go!")], id='go-button', block=True, color='success', disabled=True)

    HomePage.callbacks.append(((
        [Output('go-button', "color"),
         Output('go-button', "children")],
        [Input('go-button', "n_clicks")],
        [State('metric_dropdown', "value"),
         State('regularize_length_switch', "value"),
         State('cpu_slider', "value"),
         State('max_total_h', "value"),
         State('max_total_m', "value"),
         State('max_eval_h', "value"),
         State('max_eval_m', "value"),
         State('file-path-input', 'value')]),
        start_gama
    ))
    #def start_gama(metric, regularize, n_jobs, max_total_time, max_eval_time, input_file):

    return html.Div(
        children=[markdown_header('Configure GAMA', level=2),
                  *optimization,
                  *resources,
                  *advanced,
                  go_button],
        style={'box-shadow': '1px 1px 1px black', 'padding': '2%'}
    )


def update_marks(selected_value, min_, max_):
    return {min_: str(min_), selected_value: str(selected_value), max_: str(max_)}


def build_data_navigator() -> html.Div:
    # Unfortunately, at this time we need a full path, which is not available with dcc.Upload
    #
    # default_message = ['Drag and Drop or ', html.A('Select Files')]
    # upload_file = dcc.Upload(
    #     id='upload-data',
    #     children=html.Div(
    #         id='upload-data-text',
    #         children=default_message
    #     ),
    #     style={'borderWidth': '1px', 'borderRadius': '5px', 'borderStyle': 'dashed', 'textAlign': 'center'}
    # )

    # def update_output(list_of_contents, filename, list_of_dates):
    #     if list_of_contents is not None:
    #         children = html.Div([f"Showing '{filename}'", html.A(' (Click to change)')])
    #         return children, filename
    #     return default_message, filename
    #
    # ---------------------------------------------

    upload_file = dbc.Input(
        id='file-path-input',
        placeholder='Path to data file, e.g. ~/data/mydata.arff',
        type='text'
    )

    table_container = html.Div(
        id='table-container',
        children=['No data loaded.']
    )

    def update_data_table(filename):
        if filename is not None and os.path.isfile(filename):
            return 'found a file!', False
        return filename, True

    HomePage.callbacks.append(((
        [Output('table-container', 'children'),
         Output('go-button', 'disabled')],
        [Input('file-path-input', 'value')]),
        update_data_table
    ))

    return html.Div(
        children=[markdown_header("Data Navigator", level=2),
                  upload_file,
                  table_container],
        style={'box-shadow': '1px 1px 1px black', 'padding': '2%'}
    )
