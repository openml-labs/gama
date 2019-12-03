import os
import time

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

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        plot_area = self.plot_area()
        pl_viz = self.pipeline_viz()
        pl_list = self.pipeline_list()
        ticker = dcc.Interval(id='ticker', interval=5000)
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

        app.callback(
            [Output('evaluation-graph', 'figure'),
             Output('pipeline-table', 'data'),
             Output('pl-viz', 'children'),
             Output('pipeline-table', 'selected_rows'),
             Output('pipeline-table', 'selected_row_ids'),
             Output('evaluation-graph', 'clickData')],
            [Input('ticker', 'n_intervals'),
             Input('running-page-store', 'data')]
        )(self.update_page)

        app.callback(
            [Output('running-page-store', 'data')],
            [Input('evaluation-graph', 'clickData'),
             Input('pipeline-table', 'selected_row_ids')],
            [State('running-page-store', 'data')]
        )(self.update_selection)

        return self._content

    def update_selection(self, click_data, selected_row_ids, page_store):
        cell_selected = None if selected_row_ids is None else selected_row_ids[0]
        click_selected = None if click_data is None else click_data["points"][0]['customdata']
        # Selected row ids and click data are always set back to None. The value that is not None is the new value.
        if click_data is not None or cell_selected is not None:
            self.need_update = True
            page_store['selected_pipeline'] = click_selected if click_selected is not None else cell_selected
            return [page_store]
        # First call or sync call.
        raise PreventUpdate

    def update_page(self, _, page_store):
        start_update = time.time()
        selected_pipeline = page_store.get('selected_pipeline', None)
        if ((self.report is None and (self.log_file is None or not os.path.exists(self.log_file)))
                or (self.report is not None and not self.report.update() and not self.need_update)):
            # The report does not exist, or exists but nothing is updated.
            raise PreventUpdate
        elif self.report is None:
            self.report = GamaReport(self.log_file)

        with pd.option_context('mode.use_inf_as_na', True):
            evaluations = self.report.evaluations.dropna()

        self.need_update = False
        scatters = self.scatter_plot(evaluations, self.report.metrics, selected_pipeline)
        figure = {
            'data': scatters,
            'layout': dict(
                hovermode='closest',
                clickmode='event+select'
            )
        }

        pl_table_data = [{'pl': self.report.individuals[id_].short_name(' > '), 'id': id_}
                         for id_ in evaluations.id]
        row_id = [i for i, id_ in enumerate(evaluations.id) if id_ == selected_pipeline]

        def format_pipeline(ind):
            pipeline_elements = []
            for primitive_node in reversed(ind.primitives):
                pipeline_elements.append(html.B(str(primitive_node._primitive)))
                pipeline_elements.append(html.Br())
                for terminal in primitive_node._terminals:
                    pipeline_elements.append(f'    {terminal}')
                    pipeline_elements.append(html.Br())
            return pipeline_elements
        pl_viz_data = None if selected_pipeline is None else format_pipeline(self.report.individuals[selected_pipeline])

        print('Update complete in ', time.time() - start_update)
        return figure, pl_table_data, pl_viz_data, row_id, None, None

    def scatter_plot(self, evaluations, metrics, selected_pipeline: str = None):
        metric_one, metric_two = metrics

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

    def plot_area(self):
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

    def pipeline_list(self):
        ta = dash_table.DataTable(
            id='pipeline-table',
            columns=[{'name': 'Pipeline', 'id': 'pl'}],
            data=[],
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll'
            },
            row_selectable='single',
            persistence_type='session',
            persistence=True
        )

        return html.Div(ta, style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div(id='pl-viz', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%', 'whiteSpace': 'pre-wrap'})
