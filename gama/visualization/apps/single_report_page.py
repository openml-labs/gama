from typing import List

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from gama.logging.GamaReport import GamaReport


def single_report_page(log_lines: List[str], log_name: str):
    """ Generates a html page with dash visualizing"""
    report = GamaReport(log_lines=log_lines)

    max_phasename_length = max(len(phase[0]) for phase in report.phases)
    max_algorithm_length = max(len(phase[1]) for phase in report.phases)
    max_phasetime_length = max(len(f'{phase[2]:.3f}') for phase in report.phases)
    phases_summary = [html.Pre(
        f'{phase[0]: <{max_phasename_length}} {phase[1]: <{max_algorithm_length}} {phase[2]:{max_phasetime_length}.3f}s'
    ) for phase in report.phases
    ]

    graph = go.Scatter(
        name=f'GAMA',
        x=report.evaluations.n,
        y=report.evaluations[f'{report.metrics[0]}_cummax'],
        mode='lines'
    )

    return html.Div(children=[
        html.H1(children="GAMA Dashboard"),
        html.Div(children=phases_summary),
        dcc.Graph(
            id='optimization-graph',
            figure={
                'data': [graph],
                'layout': {
                    'title': f'{log_name}',
                    'xaxis': {'title': f'n'},
                    'yaxis': {'title': f'{report.metrics[0]}'}
                }
            },
        )
    ])
