from typing import List

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html


def button_header(text: str, id_: str, level: int = 4):
    header = f"{'#' * level} {text}"
    return dbc.FormGroup(
        [dbc.Button([dcc.Markdown(header)], id=id_, block=True, color="primary")]
    )


def markdown_header(text: str, level: int = 4, with_horizontal_rule: bool = True):
    header = f"{'#' * level} {text}"
    # matching length '-' not required but nicer.
    hr = f"\n{'-'*(level + 1 + len(text))}"
    return dcc.Markdown(f"{header}{hr if with_horizontal_rule else ''}")


class CollapsableSection:
    """ A Form with a ButtonHeader which when presses collapses/expands the Form. """

    def __init__(
        self, header: str, controls: List[dbc.FormGroup], start_open: bool = True
    ):
        self._header = header
        self._start_open = start_open
        self._header_id = f"{header}-header"
        self._form_id = f"{header}-form"
        self._controls = controls

        self.html = self._build_content()

    def _build_content(self) -> html.Div:
        form_header = button_header(self._header, id_=self._header_id)
        self.form = dbc.Form(self._controls)
        collapsable_form = dbc.Collapse(
            id=self._form_id, children=[self.form], is_open=self._start_open
        )
        return html.Div([form_header, collapsable_form])

    def register_callbacks(self, app):
        app.callback(
            Output(self._form_id, "is_open"),
            [Input(self._header_id, "n_clicks")],
            [State(self._form_id, "is_open")],
        )(_toggle_collapse)


def _toggle_collapse(click, is_open: bool):
    if click:
        return not is_open
    return is_open
