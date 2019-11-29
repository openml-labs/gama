import os

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from dash.dependencies import Input, Output, State

from gama.dashboard.components.cli_window import CLIWindow
from gama.dashboard.pages.base_page import BasePage
from gama.logging.GamaReport import GamaReport


class RunningPage(BasePage):

    def __init__(self):
        super().__init__(name='Running', alignment=-1)
        self.cli = None
        self.id = 'running-page'
        self.report = None
        self.log_file = None

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        plot_area = self.plot_area()
        pl_viz = self.pipeline_viz()
        pl_list = self.pipeline_list(app)
        ticker = dcc.Interval(id='ticker', interval=1000)
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
                ]),
                ticker
            ]
        )

        def update(_):
            if ((self.report is None and (self.log_file is None or not os.path.exists(self.log_file)))
                    or self.report is not None and not self.report.update()):
                # The report does not exist, or exists but nothing is updated.
                raise PreventUpdate
            elif self.report is None:
                self.report = GamaReport(self.log_file)

            scatters = self.create_scatter_plots(self.report)
            datas = self.fill_pipeline_table(self.report)
            return {'data': scatters}, datas

        app.callback(
            [Output('evaluation-graph', 'figure'),
             Output('pipeline-table', 'data')],
            [Input('ticker', 'n_intervals')]
        )(update)

        return self._content

    def gama_started(self, process, log_file):
        self.cli.monitor(process)
        self.log_file = log_file

    def fill_pipeline_table(self, report):
        return [{'pl': report.individuals[id_].short_name, 'id': id_} for id_ in report.evaluations.id]

    def create_scatter_plots(self, report):
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

        return [all_scatter, pareto_scatter]

    def plot_area(self):
        scatter = dcc.Graph(
            id='evaluation-graph',
            figure={'data': []}
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
            # Want to refer to the individual's ID, look up the individual and display the
            # full hyperparameter configuration of the individual.
            print(active_cell)
            return [str(active_cell)]
        app.callback(
            [Output('pl-viz', 'children')],
            [Input('pipeline-table', 'active_cell')]
        )(process_selection)
        return html.Div(ta, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div(id='pl-viz', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})
