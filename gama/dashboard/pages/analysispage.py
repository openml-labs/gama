import base64
import itertools
import os
import shutil
import uuid
from typing import Dict, List, Optional

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd

from gama.dashboard.pages.base_page import BasePage
from gama.logging.GamaReport import GamaReport
from .runningpage import format_pipeline
from ..plotting import plot_preset_graph


class AnalysisPage(BasePage):
    def __init__(self):
        super().__init__(name="Analysis", alignment=-1)
        self.id = "analysis-page"
        self.reports: Dict[str, GamaReport] = {}

    def build_page(self, app, controller):
        upload_box = html.Div(
            id="upload-container",
            children=[
                dcc.Upload(
                    id="upload-box",
                    children=html.Div([html.A("Select or drop log(s).")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "display": "inline-block",
                    },
                    multiple=True,
                )
            ],
            style=dict(
                width=f'{len("Select or drop log(s).")}em',
                display="inline-block",
                float="right",
            ),
        )

        # Top

        presets = [
            {"label": "#Pipeline by learner", "value": "number_pipeline_by_learner"},
            {"label": "#Pipeline by size", "value": "number_pipeline_by_size"},
            {"label": "Best score over time", "value": "best_over_time"},
            {"label": "Best score over iterations", "value": "best_over_n"},
            {"label": "Size vs Metric", "value": "size_vs_metric"},
            {"label": "Evaluation Times", "value": "evaluation_times_dist"},
            {"label": "Evaluations by Rung", "value": "n_by_rung"},
            {"label": "Time by Rung", "value": "time_by_rung"},
            {"label": "Table", "value": "table"},
            # {"label": "Custom", "value": "custom"},
        ]

        preset_container = html.Div(
            id="preset-container",
            children=[
                html.Div("Visualization Presets"),
                dcc.Dropdown(
                    id="preset-dropdown",
                    options=presets,
                    value="best_over_n",
                    style=dict(width="90%"),
                ),
            ],
            style=dict(width="100%", display="inline-block", float="left"),
        )

        # sep_agg_radio = dcc.RadioItems(
        #     id="sep-agg-radio",
        #     options=[
        #         {"label": "separate", "value": "separate-line"},
        #         {"label": "aggregate", "value": "aggregate"},
        #     ],
        #     value="separate-line",
        #     style={"width": "90%", "display": "inline-block"},
        # )

        # sep_agg_container = html.Div(
        #     id="sep_agg_container",
        #     children=[html.Div("Style"), sep_agg_radio],
        #     style=dict(display="inline-block", width="50%", float="left"),
        # )

        # left

        dashboard_graph = dcc.Graph(id="dashboard-graph")
        self.dbg = dashboard_graph
        dashboard_table = html.Div(id="db-table", children=["'Tis a table"])
        self.dbt = dashboard_table

        # third_width = {"width": "30%", "display": "inline-block"}
        # plot_control_container = html.Div(
        #     id="plot-controls",
        #     children=[
        #         html.Div(
        #             [html.Label("x-axis"), dcc.Dropdown(id="x-axis-metric")],
        #             style=third_width
        #         ),
        #         html.Div(
        #             [html.Label("y-axis"), dcc.Dropdown(id="y-axis-metric")],
        #             style=third_width
        #         ),
        #         html.Div(
        #             [
        #                 html.Label("plot type"),
        #                 dcc.Dropdown(
        #                     id="plot-type",
        #                     options=[
        #                         {"label": "scatter", "value": "markers"},
        #                         {"label": "line", "value": "lines"},
        #                     ],
        #                     value="lines",
        #                 ),
        #             ],
        #             style=third_width,
        #         ),
        #     ],
        #     style=dict(width="80%", display="none"),
        #     hidden=True,
        # )
        #
        # graph_settings_container = html.Div(
        #     id="graph-settings-container", children=[plot_control_container]
        # )

        # graph_update_timer = dcc.Interval(id="update-timer", interval=2 * 1000)  # ms
        #
        # graph_update_trigger = dcc.Store(id="update-trigger")

        visualization_container = html.Div(
            id="visualization-container",
            children=[
                dashboard_graph,
                # graph_settings_container,
                # graph_update_timer,
                # graph_update_trigger,
            ],
            style={"float": "left", "width": "85%", "height": "1000px"},
        )

        # right

        file_select = dcc.Checklist(id="select-log-checklist")

        report_select_container = html.Div(
            id="report-select-container",
            children=[preset_container, upload_box, file_select],
            style={"width": "14%", "float": "right", "padding-right": "1%"},
        )

        self._content = html.Div(
            id=self.id, children=[visualization_container, report_select_container],
        )

        app.callback(
            Output("select-log-checklist", "options"),
            [Input("upload-box", "contents")],
            [State("upload-box", "filename")],
        )(self.load_logs)

        app.callback(
            [
                Output("visualization-container", "children"),
                Output("dashboard-graph", "figure"),
            ],
            [
                Input("select-log-checklist", "value"),
                Input("preset-dropdown", "value"),
            ],
        )(self.update_graph)

        return self._content

    def load_logs(self, list_of_contents, list_of_names):
        # global aggregate_dataframe
        if list_of_contents is not None:
            tmp_dir = f"tmp_{str(uuid.uuid4())}"
            os.makedirs(tmp_dir)
            for content, filename in zip(list_of_contents, list_of_names):
                content_type, content_string = content.split(",")
                decoded = base64.b64decode(content_string).decode("utf-8")
                with open(os.path.join(tmp_dir, filename), "w") as fh:
                    fh.write(decoded)

            report = GamaReport(tmp_dir)
            report_name = report.search_method
            for i in itertools.count():
                if f"{report_name}_{i}" not in self.reports:
                    break
            self.reports[f"{report_name}_{i}"] = report
            shutil.rmtree(tmp_dir)
            return [{"label": logname, "value": logname} for logname in self.reports]
        return []

    def update_graph(self, logs: List[str], preset_value: Optional[str] = None):
        print(logs, preset_value)  # , aggregate, xaxis, yaxis, mode,
        # if preset_value == "custom":
        #     if logs is None or logs == [] or xaxis is None or yaxis is None:
        #         title = "Load and select a log on the right"
        #         plots = []
        #     else:
        #         title = f"{aggregate} plot of {len(logs)} logs"
        #         if aggregate == "separate-line":
        #             plots = [
        #                 individual_plot(reports[log], xaxis, yaxis, mode)
        #                 for log in logs
        #             ]
        #         if aggregate == "aggregate":
        #             plots = aggregate_plot(
        #               [reports[log] for log in logs], xaxis, yaxis
        #             )
        #     return {
        #         "data": plots,
        #         "layout": {
        #             "title": title,
        #             "xaxis": {"title": f"{xaxis}"},
        #             "yaxis": {"title": f"{yaxis}"},
        #             "hovermode": "closest" if mode == "markers" else "x",
        #         },
        #     }
        if logs is not None:
            # filtered_aggregate = aggregate_dataframe[
            #     aggregate_dataframe.filename.isin(logs)
            # ]
            if preset_value == "table":
                return [self.make_table({log: self.reports[log] for log in logs})], {}
            else:
                print("plotting")
                return (
                    [self.dbg],
                    plot_preset_graph(
                        [self.reports[log] for log in logs], preset_value
                    ),
                )
        else:
            return [self.dbg], {}

    def make_table(self, reports):
        combined_df = pd.DataFrame()

        for name, report in reports.items():
            report.evaluations["log"] = name
            df = report.evaluations
            df[[report.metrics[0], "length"]] = df["score"].apply(
                lambda x: pd.Series(x[1:-1].split(","))
            )
            df["length"] = -df["length"].astype(float)
            combined_df = combined_df.append(df)

        def full_to_short(pl: str) -> str:
            steps = pl.split(",")[0].split("(")[:-1]
            return " > ".join(list(reversed(steps)))

        combined_df["pipeline"] = combined_df["pipeline"].apply(full_to_short)

        show_cols = ["log", "n", "length", report.metrics[0], "pipeline"]
        data_table = dash_table.DataTable(
            id="table",
            columns=[{"name": c, "id": c} for c in show_cols],
            data=combined_df.to_dict("records"),
            editable=False,
            #  fixed_rows={'headers': True}, adds a scroll y
            style_cell={"textAlign": "left"},
            tooltip_data=[
                {
                    report.metrics[0]: {
                        "value": row["error"] if row["error"] != "None" else "",
                        "type": "markdown",
                    },
                    "pipeline": {
                        "value": "\n".join(
                            format_pipeline(
                                report.individuals[row["id"]], how="markdown"
                            )
                        ),
                        "type": "markdown",
                    },
                }
                for row in combined_df.to_dict("rows")
            ],
            style_cell_conditional=[{"if": {"column_id": "length"}, "width": "80px"}],
            tooltip_duration=None,
            filter_action="native",
            sort_action="native",
        )

        return data_table
