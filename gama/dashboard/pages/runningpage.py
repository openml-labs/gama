import dash_html_components as html

from gama.dashboard.components.cli_window import CLIWindow
from gama.dashboard.pages.base_page import BasePage


class RunningPage(BasePage):

    def __init__(self):
        super().__init__(name='Running', alignment=-1)
        self.cli = None
        self.id = 'running-page'

    def build_page(self, app, controller):
        self.cli = CLIWindow('cli', app)
        self._content = html.Div(id=self.id, children=[self.cli.html])
        return self._content

    def gama_started(self, process, log_file):
        self.cli.monitor(process)
