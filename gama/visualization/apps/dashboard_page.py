import base64
from typing import List, Dict, Optional

import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport
from gama.visualization.apps.plotting import individual_plot, aggregate_plot, plot_preset_graph

reports = {}
aggregate_dataframe: Optional[pd.DataFrame] = None

###########################################################
#                      HEADER BOX                         #
###########################################################
presets = [
    {'label': '#Pipeline by learner', 'value': 'number_pipeline_by_learner'},
    {'label': '#Pipeline by size', 'value': 'number_pipeline_by_size'},
    {'label': 'Best score over time', 'value': 'best_over_time'},
    {'label': 'Best score over iterations', 'value': 'best_over_n'},
    {'label': 'Size vs Metric', 'value': 'size_vs_metric'},
    {'label': 'Evaluation Times', 'value': 'evaluation_times_dist'},
    {'label': 'Evaluations by Rung', 'value': 'n_by_rung'},
    {'label': 'Time by Rung', 'value': 'time_by_rung'},
    {'label': 'Custom', 'value': 'custom'},
]

preset_container = html.Div(
    id='preset-container',
    children=[
        html.Div('Visualization Presets'),
        dcc.Dropdown(
            id='preset-dropdown',
            options=presets,
            value='best_over_n',
            style=dict(width='90%')
        )
    ],
    style=dict(width='50%', display='inline-block', float='left')
)

sep_agg_radio = dcc.RadioItems(
    id='sep-agg-radio',
    options=[
        {'label': 'separate', 'value': 'separate-line'},
        {'label': 'aggregate', 'value': 'aggregate'}
    ],
    value='separate-line',
    style={'width': '90%', 'display': 'inline-block'}
)

sep_agg_container = html.Div(
    id='sep_agg_container',
    children=[
        html.Div('Style'),
        sep_agg_radio
    ],
    style=dict(display='inline-block', width='50%', float='left')
)

dashboard_header = [preset_container, sep_agg_container]

###########################################################
#                      LEFT COLUMN                        #
###########################################################
dashboard_graph = dcc.Graph(id='dashboard-graph')

third_width = {'width': '30%', 'display': 'inline-block'}
plot_control_container = html.Div(
    id='plot-controls',
    children=[
        html.Div([
            html.Label('x-axis'),
            dcc.Dropdown(id='x-axis-metric')
            ],
            style=third_width
        ),
        html.Div([
            html.Label('y-axis'),
            dcc.Dropdown(id='y-axis-metric')
        ],
            style=third_width
        ),
        html.Div([
            html.Label('plot type'),
            dcc.Dropdown(
                id='plot-type',
                options=[
                    {'label': 'scatter', 'value': 'markers'},
                    {'label': 'line', 'value': 'lines'}
                ],
                value='lines'
            )],
            style=third_width
        )
    ],
    style=dict(width='80%', display='none'),
    hidden=True
)

graph_settings_container = html.Div(
    id='graph-settings-container',
    children=[plot_control_container]
)

visualization_container = html.Div(
    id='visualization-container',
    children=[dashboard_graph,
              graph_settings_container],
    style={'float': 'left', 'width': '85%'}
)

###########################################################
#                      RIGHT COLUMN                       #
###########################################################
file_select = dcc.Checklist(id='select-log-checklist')

report_select_container = html.Div(
    id='report-select-container',
    children=[file_select],
    style={'width': '14%', 'float': 'right', 'padding-right': '1%'}
)

###########################################################
#                      Main Page                          #
###########################################################
dashboard_page = html.Div(
    id='dashboard-main',
    children=[
        visualization_container,
        report_select_container
    ]
)


###########################################################
#                      Callbacks                          #
###########################################################
@app.callback([Output('x-axis-metric', 'options'),
               Output('y-axis-metric', 'options')],
              [Input('select-log-checklist', 'value')])
def update_valid_axis_options(logs: List[str]):
    if logs is None or logs == []:
        return [], []
    shared_attributes = set(attribute
                            for logname, report in reports.items()
                            for attribute in report.evaluations.columns
                            if logname in logs)
    dropdown_options = [{'label': att, 'value': att} for att in shared_attributes]
    return dropdown_options, dropdown_options


@app.callback(Output('dashboard-graph', 'figure'),
              [Input('select-log-checklist', 'value'),
               Input('sep-agg-radio', 'value'),
               Input('x-axis-metric', 'value'),
               Input('y-axis-metric', 'value'),
               Input('plot-type', 'value'),
               Input('preset-dropdown', 'value')])
def update_graph(logs: List[str], aggregate: str = 'separate-line', xaxis: str = None, yaxis: str = None, mode: str = None, preset_value: str = None):
    print(logs, aggregate, xaxis, yaxis, mode, preset_value)
    if preset_value == 'custom':
        if logs is None or logs == [] or xaxis is None or yaxis is None:
            title = 'Load and select a log on the right'
            plots = []
        else:
            title = f'{aggregate} plot of {len(logs)} logs'
            if aggregate == 'separate-line':
                plots = [individual_plot(reports[log], xaxis, yaxis, mode) for log in logs]
            if aggregate == 'aggregate':
                plots = aggregate_plot([reports[log] for log in logs], xaxis, yaxis)
        return {
            'data': plots,
            'layout': {
                'title': title,
                'xaxis': {'title': f'{xaxis}'},
                'yaxis': {'title': f'{yaxis}'},
                'hovermode': 'closest' if mode == 'markers' else 'x'
            }
        }
    elif logs is not None:
        return plot_preset_graph([reports[log] for log in logs], aggregate_dataframe, preset_value, aggregate)
    else:
        return {}


@app.callback(Output('select-log-checklist', 'options'),
              [Input('upload-box', 'contents')],
              [State('upload-box', 'filename')])
def load_logs(list_of_contents, list_of_names):
    global aggregate_dataframe
    if list_of_contents is not None:
        for content, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            log_lines = decoded.splitlines()
            report = GamaReport(log_lines=log_lines, name=filename)
            reports[filename] = report

            eval_copy = report.evaluations.copy()
            eval_copy['search_method'] = report.search_method
            if aggregate_dataframe is None:
                aggregate_dataframe = eval_copy
            else:
                aggregate_dataframe = pd.concat([aggregate_dataframe, eval_copy])
            print(report.search_method)
        return [{'label': logname, 'value': logname} for logname in reports]
    return []


@app.callback(Output('plot-controls', 'style'),
              [Input('preset-dropdown', 'value')],
              [State('plot-controls', 'style')])
def toggle_plot_controls(preset, plot_controls_style):
    if preset == 'custom':
        plot_controls_style['display'] = 'inline-block'
    else:
        plot_controls_style['display'] = 'none'
    return plot_controls_style
