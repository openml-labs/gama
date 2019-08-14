import base64
from typing import List

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport
from gama.visualization.apps.plotting import individual_plot, aggregate_plot, plot_preset_graph

reports = {}


###########################################################
#                      LEFT COLUMN                        #
###########################################################
dashboard_graph = dcc.Graph(id='dashboard-graph')

presets = [
    {'label': '#Pipeline by learner', 'value': 'number_pipeline_by_learner'},
    {'label': '#Pipeline by size', 'value': 'number_pipeline_by_size'},
    {'label': 'Best score over time', 'value': 'best_over_time'},
    {'label': 'Best score over iterations', 'value': 'best_over_n'},
    {'label': 'Size vs Metric', 'value': 'size_vs_metric'},
    {'label': 'Custom', 'value': 'custom'},
]
preset_control_container = html.Div(
    id='preset-control-container',
    children=[
        html.Div('Visualization Presets'),
        dcc.RadioItems(
            id='preset-radio',
            options=presets,
            value='best_over_n'
        )
    ],
    style=dict(width='20%', display='inline-block')
)

third_width = {'width': '30%', 'display': 'inline-block'}
plot_control_container = html.Div(
    id='plot-controls',
    children=[
        dcc.RadioItems(
            id='sep-agg-radio',
            options=[
                {'label': 'separate', 'value': 'separate-line'},
                {'label': 'aggregate', 'value': 'aggregate'}
            ],
            value='separate-line',
            style={'width': '10%', 'display': 'inline-block'}
        ),
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
    children=[preset_control_container, plot_control_container]
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
upload_box = dcc.Upload(
    id='upload-box',
    children=html.Div([html.A('Select or drop log(s).')]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    },
    multiple=True
)

file_select = dcc.Checklist(id='select-log-checklist')

report_select_container = html.Div(
    id='report-select-container',
    children=[upload_box, file_select],
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
               Input('preset-radio', 'value')])
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
        return plot_preset_graph([reports[log] for log in logs], preset_value)
    else:
        return {}


@app.callback(Output('select-log-checklist', 'options'),
              [Input('upload-box', 'contents')],
              [State('upload-box', 'filename')])
def load_logs(list_of_contents, list_of_names):
    if list_of_contents is not None:
        for content, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            log_lines = decoded.splitlines()
            reports[filename] = GamaReport(log_lines=log_lines, name=filename)
        return [{'label': logname, 'value': logname} for logname in reports]
    return []


@app.callback(Output('plot-controls', 'style'),
              [Input('preset-radio', 'value')],
              [State('plot-controls', 'style')])
def toggle_plot_controls(preset, plot_controls_style):
    if preset == 'custom':
        plot_controls_style['display'] = 'inline-block'
    else:
        plot_controls_style['display'] = 'none'
    return plot_controls_style
