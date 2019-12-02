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
        self.selected_pipeline_changed = False

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        plot_area = self.plot_area(app)
        pl_viz = self.pipeline_viz()
        pl_list = self.pipeline_list()
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
                ticker,
                # A div as sink for Output
                html.Div(
                    id='selected-pipeline',
                    style=dict(display='none')
                )
            ]
        )

        app.callback(
            [Output('evaluation-graph', 'figure'),
             Output('pipeline-table', 'data'),
             Output('pl-viz', 'children')],
            [Input('ticker', 'n_intervals'),
             Input('selected-pipeline', 'children')]
        )(self.update_page)

        app.callback(
            [Output('selected-pipeline', 'children')],
            [Input('evaluation-graph', 'clickData'),
             Input('pipeline-table', 'active_cell')],
            [State('selected-pipeline', 'children')]
        )(self.update_selection)

        return self._content

    def update_page(self, _, selected_pipeline):
        if ((self.report is None and (self.log_file is None or not os.path.exists(self.log_file)))
                or (self.report is not None and not self.report.update() and not self.selected_pipeline_changed)):
            # The report does not exist, or exists but nothing is updated.
            raise PreventUpdate
        elif self.report is None:
            self.report = GamaReport(self.log_file)

        self.selected_pipeline_changed = False
        scatters = self.scatter_plot(self.report, selected_pipeline)
        figure = {
            'data': scatters,
            'layout': dict(
                hovermode='closest',
                clickmode='event+select'
            )
        }

        pl_table_data = [{'pl': self.report.individuals[id_].short_name(' > '), 'id': id_}
                         for id_ in self.report.evaluations.id]

        pl_viz_data = None if selected_pipeline is None else self.report.individuals[selected_pipeline].pipeline_str()

        return figure, pl_table_data, pl_viz_data

    def scatter_plot(self, report, selected_pipeline: str = None):
        with pd.option_context('mode.use_inf_as_na', True):
            evaluations = report.evaluations.dropna()

        metric_one, metric_two = report.metrics[:2]

        # Marker size indicates recency of the evaluations, recent evaluations are bigger.
        biggest_size = 25
        smallest_size = 5
        selected_size = 30
        d_size_min_max = biggest_size - smallest_size

        sizes = list(range(smallest_size, biggest_size))[-len(evaluations):]
        if len(evaluations) > d_size_min_max:
            sizes = [smallest_size] * (len(evaluations) - d_size_min_max) + sizes
        if selected_pipeline is not None:
            sizes = [size if id_ != selected_pipeline else selected_size
                     for size, id_ in zip(sizes, evaluations.id)]

        default_color = '#301cc9'
        selected_color = '#c81818'

        colors = [default_color if id_ != selected_pipeline else selected_color
                  for id_ in evaluations.id]

        all_scatter = go.Scatter(
            x=-evaluations[metric_one],
            y=-evaluations[metric_two],
            mode='markers',
            marker={'color': colors, 'size': sizes},
            name='all evaluations',
            text=[self.report.individuals[id_].short_name() for id_ in evaluations.id],
            customdata=evaluations.id,
        )
        return [all_scatter]

    def gama_started(self, process, log_file):
        self.cli.monitor(process)
        self.log_file = log_file

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
        return html.Div(scatter, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def update_selection(self, click_data, active_cell, previous_selected):
        click_selected = None if click_data is None else click_data["points"][0]['customdata']
        cell_selected = None if active_cell is None else active_cell['row_id']
        if click_selected == cell_selected == previous_selected:
            raise PreventUpdate

        self.selected_pipeline_changed = True
        if click_selected is not None and previous_selected != click_selected:
            return [click_selected]
        elif cell_selected is not None and previous_selected != cell_selected:
            return [cell_selected]

    def pipeline_list(self):
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

        return html.Div(ta, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div(id='pl-viz', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})
