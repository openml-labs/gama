import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
from dash.dependencies import Input, Output, State

from gama.dashboard.components.cli_window import CLIWindow
from gama.dashboard.pages.base_page import BasePage


class RunningPage(BasePage):

    def __init__(self):
        super().__init__(name='Running', alignment=-1)
        self.cli = None
        self.id = 'running-page'

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        plot_area = self.plot_area()
        pl_viz = self.pipeline_viz()
        pl_list = self.pipeline_list(app)
        self._content = html.Div(
            id=self.id,
            children=[
                dbc.Row([
                    dbc.Col(plot_area, width=8),
                    dbc.Col(self.cli.html),
                ]),
                dbc.Row([
                    dbc.Col(pl_viz, width=8),
                    dbc.Col(pl_list)
                ])
            ]
        )
        return self._content

    def gama_started(self, process, log_file):
        self.cli.monitor(process)

    def plot_area(self):
        import numpy as np
        evals = 2*np.random.random((50, 2))
        evaluations = evals
        sizes = list(range(20, 1, -1))[:len(evaluations)] + [10] * max(len(evaluations) - 20, 0)
        evaluation_color = '#301cc9'  # (247°, 86%, 79%)
        pareto = [(2.1, 2.1), (2.05, 2.2), (2.2, 2.05)]
        pareto_color = '#c81818'

        all_scatter = go.Scatter(
            x=[f[0] for f in evaluations],
            y=[f[1] for f in evaluations],
            mode='markers',
            marker={'color': evaluation_color, 'size': sizes},
            name='all evaluations'
        )
        pareto_scatter = go.Scatter(
            x=[f[0] for f in pareto],
            y=[f[1] for f in pareto],
            mode='markers',
            marker={'color': pareto_color, 'size': 30},
            name='pareto front'
        )

        scatter = dcc.Graph(
            figure={'data': [all_scatter, pareto_scatter]}
        )

        return html.Div(scatter, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_list(self, app):
        pipelines = [
            'GaussianNB',
            'SelectFWE>RandomForestClassifier',
            'MinMaxScaler>DecisionTree',
            'GaussianNB',
            'SelectFWE>RandomForestClassifier',
            'MinMaxScaler>DecisionTree',
            'GaussianNB',
            'SelectFWE>RandomForestClassifier',
            'MinMaxScaler>DecisionTree',
            'GaussianNB',
            'SelectFWE>RandomForestClassifier',
            'MinMaxScaler>DecisionTree'
        ]
        ta = dash_table.DataTable(
            id='pipeline-table',
            columns=[{'name': 'Pipeline', 'id': 'pl'}],
            data=[{'pl': pl} for pl in pipelines],
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll'
            },
            persistence_type='session', persistence=True
        )

        def process_selection(active_cell):
            # https://community.plot.ly/t/datatable-accessing-value-of-active-cell/20378
            print(active_cell)
            return [str(active_cell)]
        app.callback(
            [Output('pl-viz', 'children')],
            [Input('pipeline-table', 'active_cell')]
        )(process_selection)
        return html.Div(ta, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div(id='pl-viz', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})
