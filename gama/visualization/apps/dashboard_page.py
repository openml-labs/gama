import base64
from typing import List

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport
from gama.visualization.apps.plotting import individual_plot, aggregate_plot

reports = {}


###########################################################
#                      LEFT COLUMN                        #
###########################################################
dashboard_graph = dcc.Graph(id='dashboard-graph')

third_width = {'width': '30%', 'display': 'inline-block'}
graph_settings_container = html.Div(
    id='graph-settings-container',
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
    ]
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
               Output('x-axis-metric', 'value'),
               Output('y-axis-metric', 'options'),
               Output('y-axis-metric', 'value')],
              [Input('select-log-checklist', 'value')],
              [State('x-axis-metric', 'value'),
               State('y-axis-metric', 'value')])
def update_valid_axis_options(logs: List[str], x_value: str, y_value: str):
    if logs is None or logs == []:
        return [], None, [], None
    shared_attributes = set(attribute
                            for logname, report in reports.items()
                            for attribute in report.evaluations.columns
                            if logname in logs)
    dropdown_options = [{'label': att, 'value': att} for att in shared_attributes]
    x_value = x_value if x_value is not None else 'n'
    y_value = y_value if y_value is not None else f'{reports[logs[0]].metrics[0]}_cummax'
    return dropdown_options, x_value, dropdown_options, y_value


@app.callback(Output('dashboard-graph', 'figure'),
              [Input('select-log-checklist', 'value'),
               Input('sep-agg-radio', 'value'),
               Input('x-axis-metric', 'value'),
               Input('y-axis-metric', 'value'),
               Input('plot-type', 'value')])
def update_graph(logs: List[str], aggregate: str = 'separate-line', xaxis: str = None, yaxis: str = None, mode: str=None):
    print(logs, aggregate, xaxis, yaxis, mode)
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
