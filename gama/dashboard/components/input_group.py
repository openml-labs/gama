from typing import Dict

from dash import Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc


def automark_slider(app: Dash, id_: str, label: str, slider_kwargs: Dict):
    defaults = dict(min=1, max=10, value=1, updatemode="drag")
    defaults.update(slider_kwargs)
    marks = {defaults["min"]: defaults["min"], defaults["max"]: defaults["max"]}

    cpu_input = dbc.FormGroup(
        [
            dbc.Label(label, html_for=id_, width=6),
            dbc.Col(dcc.Slider(id=id_, marks=marks, **defaults)),
        ],
        row=True,
    )
    app.callback.append(
        Output(id_, "marks"),
        [Input(id_, "value")],
        [State(id_, "min"), State(id_, "max")],
    )(_update_marks)
    return cpu_input


def _update_marks(selected_value, min_, max_):
    return {min_: str(min_), selected_value: str(selected_value), max_: str(max_)}


class ToggleButton:
    def __init__(self, button_id: str, app: Dash, label: str, start_on: bool = True):
        self._button_id = button_id
        self.html = self._build_content(label, start_on)

    def _build_content(self, label: str, start_on: bool):
        return dbc.FormGroup(
            [
                dbc.Label(label, html_for=self._button_id, width=6),
                dbc.Col(
                    dbc.Checklist(
                        id=self._button_id,
                        options=[{"label": "", "value": "on"}],
                        switch=True,
                        value="on" if start_on else "off",
                    )
                ),
            ],
            row=True,
        )
