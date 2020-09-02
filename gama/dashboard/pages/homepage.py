import multiprocessing
import os
from typing import Optional, List, Dict, Tuple, Callable

import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from gama.dashboard.pages.base_page import BasePage
from gama.data_loading import file_to_pandas, load_feature_metadata_from_file


class HomePage(BasePage):
    callbacks: List[Tuple[Tuple, Callable]] = []

    def __init__(self):
        super().__init__(name="Home", alignment=0)
        self.id = "home-page"

    def build_page(self, app, controller):
        self._build_content(app, controller)
        if app is not None:
            self._register_callbacks(app)

    def _build_content(self, app, controller) -> html.Div:
        """ Build all the components of the page. """
        configuration = build_configuration_menu(app, controller)
        configuration.style["width"] = "35%"
        configuration.style["float"] = "left"
        data_navigator = build_data_navigator()
        data_navigator.style["width"] = "65%"
        data_navigator.style["float"] = "right"
        self._content = html.Div(id=self.id, children=[configuration, data_navigator])
        return self._content

    def _register_callbacks(self, app):
        for (io, fn) in HomePage.callbacks:
            app.callback(*io)(fn)
        HomePage.callbacks = []

    def load(self):
        pass

    def unload(self):
        pass


# === Configuration Menu ===


def cpu_slider():
    n_cpus = multiprocessing.cpu_count()
    id_ = "cpu_slider"
    cpu_input = dbc.FormGroup(
        [
            dbc.Label("N Jobs", html_for=id_, width=6),
            dbc.Col(
                dcc.Slider(
                    id=id_,
                    min=1,
                    max=n_cpus,
                    updatemode="drag",
                    value=1,
                    marks={1: "1", n_cpus: str(n_cpus)},
                    persistence_type="session",
                    persistence=True,
                )
            ),
        ],
        row=True,
    )
    HomePage.callbacks.append(
        (
            (
                Output(id_, "marks"),
                [Input(id_, "value")],
                [State(id_, "min"), State(id_, "max")],
            ),
            update_marks,
        )
    )
    return cpu_input


def time_nud(
    label_text: str,
    hour_id: str,
    hour_default: int,
    minute_id: str,
    minute_default: int,
):
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=hour_id, width=6),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id=hour_id,
                            type="number",
                            min=0,
                            max=99,
                            step=1,
                            value=hour_default,
                        ),
                        dbc.InputGroupAddon("H", addon_type="append"),
                    ]
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id=minute_id,
                            type="number",
                            min=0,
                            max=59,
                            step=1,
                            value=minute_default,
                        ),
                        dbc.InputGroupAddon("M", addon_type="append"),
                    ]
                )
            ),
        ],
        row=True,
    )


def toggle_button(label_text: str, id_: str, start_on: bool = True):
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=id_, width=6),
            dbc.Col(
                dbc.Checklist(
                    id=id_,
                    options=[{"label": "", "value": "on"}],
                    switch=True,
                    value="on" if start_on else "off",
                )
            ),
        ],
        row=True,
    )


def text_input(label_text: str, default_text: str, id_: str):
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=id_, width=6),
            dbc.Col(
                dbc.Input(
                    id=id_, type="text", placeholder=default_text, value=default_text
                )
            ),
        ],
        row=True,
    )


def dropdown(
    label_text: str, id_: str, options: Dict[str, str], value: Optional[str] = None
):
    """ options formatted as {LABEL_KEY: LABEL_TEXT, ...} """
    return dbc.FormGroup(
        [
            dbc.Label(label_text, html_for=id_, width=6),
            dbc.Col(
                dcc.Dropdown(
                    id=id_,
                    options=[
                        {"label": text, "value": key} for key, text in options.items()
                    ],
                    clearable=False,
                    value=value,
                    persistence_type="session",
                    persistence=True,
                ),
            ),
        ],
        row=True,
    )


def button_header(text: str, id_: str, level: int = 4):
    header = f"{'#' * level} {text}"
    return dbc.FormGroup(
        [dbc.Button([dcc.Markdown(header)], id=id_, block=True, color="primary")]
    )


def markdown_header(text: str, level: int = 4, with_horizontal_rule: bool = True):
    header = f"{'#' * level} {text}"
    hr = f"\n{'-'*(level + 1 + len(text))}"  # matching length '-' not required but nice
    return dcc.Markdown(f"{header}{hr if with_horizontal_rule else ''}")


def toggle_collapse(click, is_open: bool):
    if click:
        return not is_open
    return is_open


def collapsable_section(
    header: str, controls: List[dbc.FormGroup], start_open: bool = True
):
    header_id = f"{header}-header"
    form_id = f"{header}-form"

    form_header = button_header(header, id_=header_id)
    collapsable_form = dbc.Collapse(
        id=form_id, children=[dbc.Form(controls)], is_open=start_open
    )

    HomePage.callbacks.append(
        (
            (
                Output(form_id, "is_open"),
                [Input(header_id, "n_clicks")],
                [State(form_id, "is_open")],
            ),
            toggle_collapse,
        )
    )
    return form_header, collapsable_form


def build_configuration_menu(app, controller) -> html.Div:
    # Optimization
    from gama.utilities.metrics import all_metrics

    metrics = {m: m.replace("_", " ") for m in all_metrics}
    metrics.update({"default": "default"})
    scoring_input = dropdown(
        "Metric", "metric_dropdown", options=metrics, value="default"
    )
    regularize_input = toggle_button(
        "Prefer short pipelines", "regularize_length_switch"
    )
    optimization = collapsable_section(
        "Optimization", [scoring_input, regularize_input]
    )

    # Resources
    cpu_input = cpu_slider()
    max_total_time_input = time_nud(
        "Max Runtime",
        hour_id="max_total_h",
        hour_default=1,
        minute_id="max_total_m",
        minute_default=0,
    )
    max_eval_time_input = time_nud(
        "Max time per pipeline",
        hour_id="max_eval_h",
        hour_default=0,
        minute_id="max_eval_m",
        minute_default=5,
    )
    resources = collapsable_section(
        "Resources", [cpu_input, max_total_time_input, max_eval_time_input]
    )

    # Advanced
    log_path = text_input("Log Directory", "~/GamaLog", "logpath")
    advanced = collapsable_section("Advanced", [log_path], start_open=False)

    # Go!
    go_button = dbc.Button(
        [dcc.Markdown("#### Go!")],
        id="go-button",
        block=True,
        color="success",
        disabled=True,
    )

    def start_gama(n_click, running_tab_style, *args):
        controller.start_gama(*args)
        running_tab_style["display"] = "inline"
        return "danger", dcc.Markdown("#### Stop!"), "Running", running_tab_style

    app.callback(
        [
            Output("go-button", "color"),
            Output("go-button", "children"),
            Output("page-tabs", "value"),
            Output("Running-tab", "style"),
        ],
        [Input("go-button", "n_clicks")],
        [
            State("Running-tab", "style"),
            State("metric_dropdown", "value"),
            State("regularize_length_switch", "value"),
            State("cpu_slider", "value"),
            State("max_total_h", "value"),
            State("max_total_m", "value"),
            State("max_eval_h", "value"),
            State("max_eval_m", "value"),
            State("file-path-input", "value"),
            State("logpath", "value"),
            State("target_dropdown", "value"),
        ],
    )(start_gama)

    return html.Div(
        children=[
            markdown_header("Configure GAMA", level=2),
            *optimization,
            *resources,
            *advanced,
            go_button,
        ],
        style={"box-shadow": "1px 1px 1px black", "padding": "2%"},
    )


def update_marks(selected_value, min_, max_):
    return {min_: str(min_), selected_value: str(selected_value), max_: str(max_)}


def build_data_navigator() -> html.Div:
    upload_file = dbc.Input(
        id="file-path-input",
        placeholder="Path to data file, e.g. ~/data/mydata.arff",
        type="text",
    )

    modes = ["None", "Small", "All"]
    settings = dbc.FormGroup(
        [
            dbc.Label("Target", html_for="target_dropdown", width=2),
            dbc.Col(
                dcc.Dropdown(
                    id="target_dropdown",
                    options=[{"label": "-", "value": "a"}],
                    clearable=False,
                    value="a",
                    # persistence_type="session",
                    # persistence=True,
                ),
                width=4,
            ),
            dbc.Label("Preview Mode", html_for="preview_dropdown", width=2),
            dbc.Col(
                dcc.Dropdown(
                    id="preview_dropdown",
                    options=[{"label": m, "value": m.lower()} for m in modes],
                    clearable=False,
                    value="none",
                    # persistence_type="session",
                    # persistence=True,
                ),
                width=4,
            ),
        ],
        row=True,
    )

    table_container = html.Div(id="table-container", children=["No data loaded."])

    data_settings = html.Div(
        id="data-settings-container",
        children=[settings, table_container],
        style={"margin": "10px"},
    )

    def update_data_table(filename, mode):
        if filename is not None and os.path.isfile(filename):
            if mode in ["all", "small"]:
                df = file_to_pandas(filename)
                if mode == "small":
                    df = df.head(50)

                data_table = dash_table.DataTable(
                    id="table",
                    columns=[{"name": c, "id": c} for c in df.columns],
                    data=df.to_dict("records"),
                    editable=False,
                    style_table={"maxHeight": "500px", "overflowY": "scroll"},
                )
                attributes = list(df.columns)
            else:
                data_table = "Preview not enabled."
                attributes = list(load_feature_metadata_from_file(filename))

            target_options = [{"label": c, "value": c} for c in attributes]
            default_target = attributes[-1]

            return [data_table], target_options, default_target, False
        return ["No data loaded"], [{"label": "-", "value": "a"}], "a", True

    HomePage.callbacks.append(
        (
            (
                [
                    Output("table-container", "children"),
                    Output("target_dropdown", "options"),
                    Output("target_dropdown", "value"),
                    Output("go-button", "disabled"),
                ],
                [
                    Input("file-path-input", "value"),
                    Input("preview_dropdown", "value"),
                ],
            ),
            update_data_table,
        )
    )

    return html.Div(
        children=[
            markdown_header("Data Navigator", level=2),
            upload_file,
            data_settings,
        ],
        style={"box-shadow": "1px 1px 1px black", "padding": "2%"},
    )
