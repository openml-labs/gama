import dash_html_components as html
import dash_bootstrap_components as dbc

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
        pl_list = self.pipeline_list()
        pl_viz = self.pipeline_viz()
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
        return html.Div('plot', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_list(self):
        return html.Div('pl-list', style={'height': '100%', 'box-shadow': '1px 1px 1px black', 'padding': '2%'})

    def pipeline_viz(self):
        return html.Div('pl-viz', style={'box-shadow': '1px 1px 1px black', 'padding': '2%'})
