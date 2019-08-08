from typing import List

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport

report = None


@app.callback(Output('optimization-graph', 'figure'),
              [Input('x-axis-metric', 'value'),
               Input('y-axis-metric', 'value'),
               Input('plot-type', 'value')])
def show_label_value(xaxis, yaxis, mode):
    return make_figure(report, xaxis, yaxis, mode)


def make_figure(report, xaxis, yaxis, mode='lines'):
    return {
         'data': [
             go.Scatter(
                 name=f'GAMA',
                 x=report.evaluations[xaxis],
                 y=report.evaluations[yaxis],
                 text=report.evaluations.pipeline,
                 mode=mode
             )
         ],
         'layout': {
             'title': f'log name',
             'xaxis': {'title': f'n'},
             'yaxis': {'title': f'{yaxis}'}
         }
     }


def single_report_page(log_lines: List[str], log_name: str):
    """ Generates a html page with dash visualizing"""
    global report
    report = GamaReport(log_lines=log_lines)

    max_phasename_length = max(len(phase[0]) for phase in report.phases)
    max_algorithm_length = max(len(phase[1]) for phase in report.phases)
    max_phasetime_length = max(len(f'{phase[2]:.3f}') for phase in report.phases)
    phases_summary = [html.Pre(
        f'{phase[0]: <{max_phasename_length}} {phase[1]: <{max_algorithm_length}} {phase[2]:{max_phasetime_length}.3f}s'
    ) for phase in report.phases
    ]

    third_width = {'width': '33%', 'display': 'inline-block'}
    layout = html.Div(children=[
        html.Div(children=phases_summary, style={'textAlign': 'center'}),
        dcc.Graph(
            id='optimization-graph'
        ),
        html.Div([
            html.Label('x-axis'),
            dcc.Dropdown(
                id='x-axis-metric',
                options=[{'label': metric, 'value': metric}
                         for metric in report.evaluations.columns],
                value=f'n'
            )],
            style=third_width
        ),
        html.Div([
            html.Label('y-axis'),
            dcc.Dropdown(
                id='y-axis-metric',
                options=[{'label': metric, 'value': metric}
                         for metric in report.evaluations.columns],
                value=f'{report.metrics[0]}_cummax'
            )],
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
    ])
    return layout
