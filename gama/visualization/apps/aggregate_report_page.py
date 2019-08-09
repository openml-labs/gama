from typing import List
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from gama.visualization.app import app
from gama.logging.GamaReport import GamaReport

reports: List[GamaReport] = []


@app.callback(Output('combined-optimization-graph', 'figure'),
              [Input('x-axis-metric', 'value'),
               Input('y-axis-metric', 'value'),
               Input('plot-type', 'value')])
def show_label_value(xaxis, yaxis, mode):
    return make_figure(xaxis, yaxis, mode)


def make_figure(xaxis, yaxis, mode='lines'):
    if mode == 'agg':
        concat_df = pd.concat([report.evaluations for report in reports])
        concat_df = concat_df[concat_df[yaxis] != -float('inf')]
        agg_df = concat_df.groupby(by=xaxis).agg({yaxis: ['mean', 'std']}).reset_index()
        agg_df.columns = [xaxis, yaxis, 'std']
        upper_bound = go.Scatter(
            name=f'UB',
            x=agg_df[xaxis],
            y=agg_df[yaxis] + agg_df['std'],
            mode='lines',
            marker=dict(color="#444"),  # Not immediately clear what anything below this is.
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
            marker=dict(color="#444"),  # Not immediately clear what anything below this is.
            line=dict(width=0)
        )
        data = [lower_bound, mean_performance, upper_bound]
    elif mode == 'sep':
        data = [
            go.Scatter(
                name=f'{i}',
                x=report.evaluations[xaxis],
                y=report.evaluations[yaxis],
                text=report.evaluations.pipeline
            )
            for i, report in enumerate(reports)
        ]

    return {
         'data': data,
         'layout': {
             'title': f'log name',
             'xaxis': {'title': f'{xaxis}'},
             'yaxis': {'title': f'{yaxis}'}
         }
     }



def aggregate_report_page(list_log_lines: List[List[str]]):
    """ Generates a html page with dash visualizing"""
    global reports
    for log_lines in list_log_lines:
        reports.append(GamaReport(log_lines=log_lines))

    third_width = {'width': '33%', 'display': 'inline-block'}
    layout = html.Div(children=[
        dcc.Graph(
            id='combined-optimization-graph'
        ),
        html.Div([
            html.Label('x-axis'),
            dcc.Dropdown(
                id='x-axis-metric',
                options=[{'label': metric, 'value': metric}
                         for metric in reports[0].evaluations.columns],
                value=f'n'
            )],
            style=third_width
        ),
        html.Div([
            html.Label('y-axis'),
            dcc.Dropdown(
                id='y-axis-metric',
                options=[{'label': metric, 'value': metric}
                         for metric in reports[0].evaluations.columns],
                value=f'{reports[0].metrics[0]}_cummax'
            )],
            style=third_width
        ),
        html.Div([
            html.Label('plot type'),
            dcc.Dropdown(
                id='plot-type',
                options=[
                    {'label': 'aggregate', 'value': 'agg'},
                    {'label': 'separate', 'value': 'sep'}
                ],
                value='agg'
            )],
            style=third_width
        )
    ])
    return layout
