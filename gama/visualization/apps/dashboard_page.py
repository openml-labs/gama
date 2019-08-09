import base64
from typing import List, Optional

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport

reports = {}


dashboard_graph = dcc.Graph(id='dashboard-graph')

graph_settings_container = html.Div(
    id='graph-settings-container',
    children=[
        dcc.RadioItems(
            id='sep-agg-radio',
            options=[
                {'label': 'separate', 'value': 'separate-line'},
                {'label': 'aggregate', 'value': 'aggregate'}
            ],
            value='separate-line'
        )
    ]
)

visualization_container = html.Div(
    id='visualization-container',
    children=[dashboard_graph,
              graph_settings_container],
    style={'float': 'left', 'width': '85%'}
)


def aggregate_plot(logs, xaxis, yaxis):
    reports_to_combine = [reports[log] for log in logs]
    concat_df = pd.concat([report.evaluations for report in reports_to_combine])
    concat_df = concat_df[concat_df[yaxis] != -float('inf')]
    agg_df = concat_df.groupby(by=xaxis).agg({yaxis: ['mean', 'std']}).reset_index()
    agg_df.columns = [xaxis, yaxis, 'std']
    upper_bound = go.Scatter(
        name=f'UB',
        x=agg_df[xaxis],
        y=agg_df[yaxis] + agg_df['std'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    mean_performance = go.Scatter(
        name=f'Mean',
        x=agg_df[xaxis],
        y=agg_df[yaxis],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    lower_bound = go.Scatter(
        name=f'LB',
        x=agg_df[xaxis],
        y=agg_df[yaxis] - agg_df['std'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0)
    )
    aggregate_data = [lower_bound, mean_performance, upper_bound]
    return aggregate_data


def individual_plots(logs, xaxis, yaxis, mode):
    plots = []
    for log in logs:
        plots.append(go.Scatter(
            name=f'{log}',
            x=reports[log].evaluations[xaxis],
            y=reports[log].evaluations[yaxis],
            #text=reports[log].evaluations.pipeline,
            mode=mode
        ))
    return plots


@app.callback(Output('dashboard-graph', 'figure'),
              [Input('select-log-checklist', 'value'),
               Input('sep-agg-radio', 'value')])
def update_graph(logs: List[str], mode: str = 'separate-line', xaxis: str = 'n', yaxis: Optional[str] = None):
    if logs is None or logs == []:
        title = 'Load and select a log on the right'
        plots = []
    else:
        title = f'{mode} plot of {len(logs)} logs'
        if yaxis is None:
            yaxis = f'{reports[logs[0]].metrics[0]}_cummax'
        if mode == 'separate-line':
            plots = individual_plots(logs, xaxis, yaxis, mode='lines')
        if mode == 'aggregate':
            plots = aggregate_plot(logs, xaxis, yaxis)
    return {
        'data': plots,
        'layout': {
            'title': title,
            'xaxis': {'title': f'{xaxis}'},
            'yaxis': {'title': f'{yaxis}'}
        }
    }


###########################################################
#                      RIGHT COLUMN                       #
###########################################################


@app.callback(Output('select-log-checklist', 'options'),
              [Input('upload-box', 'contents')],
              [State('upload-box', 'filename'),
               State('upload-box', 'last_modified')])
def load_logs(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        for content, filename in zip(list_of_contents, list_of_names):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            log_lines = decoded.splitlines()
            reports[filename] = GamaReport(log_lines=log_lines)
        return [{'label': logname, 'value': logname} for logname in reports]
    return []


upload_box = dcc.Upload(
    id='upload-box',
    children=html.Div([html.A('Select or drop file(s).')]),
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
