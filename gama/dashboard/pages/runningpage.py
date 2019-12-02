import os

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd

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
        self.selected_pipeline = None

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        plot_area = self.plot_area(app)
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
                    dbc.Col(pl_viz, width=4),
                    dbc.Col(pl_list)
                ]),
                ticker
            ]
        )

        def update(_, active_cell):
            selected_pipeline = None if active_cell is None else active_cell['row_id']
            new_pipeline_selected = self.selected_pipeline != selected_pipeline
            self.selected_pipeline = selected_pipeline
            if ((self.report is None and (self.log_file is None or not os.path.exists(self.log_file)))
                    or (self.report is not None and not self.report.update() and not new_pipeline_selected)):
                # The report does not exist, or exists but nothing is updated.
                raise PreventUpdate
            elif self.report is None:
                self.report = GamaReport(self.log_file)

            scatters = self.create_scatter_plots(self.report, selected_pipeline)
            datas = self.fill_pipeline_table(self.report)
            figure = {
                'data': scatters,
                'layout': dict(
                    hovermode='closest',
                    clickmode='event+select'
                )
            }

            return figure, datas
        app.callback(
            [Output('evaluation-graph', 'figure'),
             Output('pipeline-table', 'data')],
            [Input('ticker', 'n_intervals'),
             Input('pipeline-table', 'active_cell')]
        )(update)

        return self._content

    def gama_started(self, process, log_file):
        self.cli.monitor(process)
        self.log_file = log_file

    def fill_pipeline_table(self, report):
        return [{'pl': report.individuals[id_].short_name(' > '), 'id': id_} for id_ in report.evaluations.id]

    def create_scatter_plots(self, report, selected_pipeline=None):
        if len(report.evaluations) == 0:
            raise PreventUpdate

        with pd.option_context('mode.use_inf_as_na', True):
            evaluations = report.evaluations.dropna()
        scores = evaluations.loc[:, report.metrics].values
        ids = [row.id for i, row in evaluations.iterrows()]
        sizes = [6] * max(len(scores) - 20, 0) + list(range(6, 26))[:len(scores)]
        evaluation_color = '#301cc9'  # (247°, 86%, 79%)
        pareto_color = '#c81818'
        colors = [pareto_color if id_ == selected_pipeline else evaluation_color
                  for id_ in ids]
        sizes = [30 if id_ == selected_pipeline else size for id_, size in zip(ids, sizes)]

        pareto = [(2.1, 2.1), (2.05, 2.2), (2.2, 2.05)]

        all_scatter = go.Scatter(
            x=[-f[0] for f in scores],
            y=[-f[1] for f in scores],
            mode='markers',
            marker={'color': colors, 'size': sizes},
            name='all evaluations',
            text=[self.report.individuals[row.id].short_name() for i, row in evaluations.iterrows()],
            customdata=ids,
        )

        pareto_scatter = go.Scatter(
            x=[f[0] for f in pareto],
            y=[f[1] for f in pareto],
            mode='markers',
            marker={'color': pareto_color, 'size': 30},
            name='pareto front'
        )
        return [all_scatter] #, pareto_scatter]

    def plot_area(self, app):
        scatter = dcc.Graph(
            id='evaluation-graph',
            figure={
                'data': [],
                'layout': dict(
                    hovermode='closest',
                    transition={'duration': 500},
                )
            }
        )

        def display_click_data(clickData):
            return ['7f1bb0ed-b80c-4d78-a23a-0c456721fa3f']
            #return clickData["points"][0]['customdata']

        app.callback(
            Output('pipeline-table', 'selected_row_ids'),
            [Input('evaluation-graph', 'clickData')]
        )(display_click_data)


        return html.Div(scatter, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_list(self, app):
        ta = dash_table.DataTable(
            id='pipeline-table',
            columns=[{'name': 'Pipeline', 'id': 'pl'}],
            data=[],
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
            return [self.report.individuals[active_cell['row_id']].pipeline_str()]
        app.callback(
            [Output('pl-viz', 'children')],
            [Input('pipeline-table', 'active_cell')]
        )(process_selection)
        return html.Div(ta, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div(id='pl-viz', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})
