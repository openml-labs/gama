import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State

from gama.dashboard.pages.base_page import BasePage


class HomePage(BasePage):

    def __init__(self):
        super().__init__(name='Home', alignment=0)

    def build_page(self) -> html.Div:
        configuration = build_configuration_menu()
        configuration.style['width'] = '50%'
        configuration.style['float'] = 'left'
        data_navigator = build_data_navigator()
        data_navigator.style['width'] = '50%'
        data_navigator.style['float'] = 'right'
        return html.Div(
            id="home-content",
            children=[
                configuration,
                data_navigator
            ]
        )

    def register_callbacks(self, app):
        app.callback(
            Output("n_jobs", "marks"),
            [Input("n_jobs", "value")],
            [State("n_jobs", "min"), State("n_jobs", "max")]
        )(update_marks)


def create_slider_input(label: str, min_: int, max_: int):
    return daq.Slider(id=label, min=min_, max=max_, updatemode='drag')


def build_configuration_menu() -> html.Div:
    cpu_slider = create_slider_input('n_jobs', 1, 16)
    return html.Div(
        children=[html.P("Configuration Menu"), cpu_slider],
        style={'box-shadow': '1px 1px 1px black'}
    )


def update_marks(selected_value, min_, max_):
    return {min_: min_, selected_value: selected_value, max_: max_}


def build_data_navigator() -> html.Div:
    return html.Div([html.P("Data Navigator")], style={'box-shadow': '1px 1px 1px black'})


# class AutomarkSlider(daq.Slider):
#     app = None
#
#     def __init__(self, **kwargs):
#         self._min = kwargs.get('min', 0)
#         self._max = kwargs.get('max', None)
#         self._requested_marks = kwargs.get('marks', {})
#         kwargs['marks'] = self._update_marks(None)
#         super().__init__(**kwargs)
#
#         print('registering callback for ', self.id)
#         AutomarkSlider.app.callback(
#             Output(self.id, 'marks'),
#             [Input(self.id, 'value')]
#         )(self._update_marks)
#
#     def _update_marks(self, selected_value: int):
#         self._marks = dict(self._requested_marks)
#         self._marks[self._min] = self._min
#         if self._max is not None:
#             self._marks[self._max] = self._max
#         if selected_value is not None:
#             self._marks[selected_value] = selected_value
#         self.marks = self._marks
#         print(self.marks)
#         return self.marks


