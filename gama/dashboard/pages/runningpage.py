import dash_html_components as html

from gama.dashboard.pages.base_page import BasePage


class RunningPage(BasePage):

    def __init__(self):
        super().__init__(name='Running', alignment=-1)

    def build_page(self):
        return html.Div([html.P("Running Page Placeholder")])
