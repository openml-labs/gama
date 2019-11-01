import dash_html_components as html

from gama.dashboard.pages.base_page import BasePage


class HomePage(BasePage):

    def __init__(self):
        super().__init__(name='Home', alignment=0)

    def build_page(self):
        return html.Div([html.P("Home Page Placeholder")])
