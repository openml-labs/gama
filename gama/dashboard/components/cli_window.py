import shlex
import subprocess
import threading
import queue
from typing import List

from dash import Dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import visdcc


def enqueue_output(out, queue_: queue.Queue):
    for line in iter(out.readline, b""):
        queue_.put(line)
    out.close()


class CLIWindow:
    """ A Component for Dash App which simulates a console window """

    def __init__(
        self,
        id_: str,
        app: Dash,
        update_interval_s: float = 1.0,
        auto_scroll: bool = True,
    ):
        self._update_interval_s = update_interval_s
        self.auto_scroll = auto_scroll

        self.console_id = f"{id_}-text"
        self.timer_id = f"{id_}-interval"
        self.js_id = f"{id_}-js"
        self.id_ = id_

        self.autoscroll_script = (
            f"document.getElementById('{self.console_id}').scrollTop"
            f" = document.getElementById('{self.console_id}').scrollHeight"
        )
        self.process = None
        self._thread = None
        self._queue = None
        self._lines: List[str] = []

        self.html = self._build_component()
        self._register_callbacks(app)

    def _build_component(self) -> html.Div:
        timer = dcc.Interval(
            id=self.timer_id, interval=self._update_interval_s * 1000, n_intervals=0
        )
        scroller = visdcc.Run_js(id=self.js_id, run="")
        self.console = dcc.Textarea(
            id=self.console_id,
            contentEditable="false",
            style={
                "height": "100%",
                "width": "100%",
                "borderWidth": "1px",
                "borderRadius": "5px",
                "borderStyle": "dashed",
            },
            persistence_type="session",
            persistence=True,
        )
        return html.Div(
            id=self.id_,
            children=[timer, self.console, scroller],
            style={"height": "100%"},
        )

    def _register_callbacks(self, app):
        app.callback(
            [Output(self.console_id, "value"), Output(self.js_id, "run")],
            [Input(self.timer_id, "n_intervals")],
            [State(self.console_id, "value")],
        )(self.update_console)

    def monitor(self, process):
        self.process = process
        # Because there are only blocking reads to the pipe,
        # we need to read them on a separate thread.
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=enqueue_output, args=(self.process.stdout, self._queue), daemon=True
        )
        self._thread.start()

    def call(self, command: str):
        cmd = shlex.split(command)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        self.monitor(process)

    def update_console(self, _, current_text):
        if self.process is None:
            return [None, None]

        # We want to update the text field if there is new output from the process,
        # or if we detect the text value has been reset (due to e.g. switching tabs).
        try:
            line = self._queue.get_nowait()
            self._lines.append(line.decode("utf-8"))
        except queue.Empty:
            # No new message, update only required if value field had been reset.
            if current_text is not None:
                raise PreventUpdate

        script = self.autoscroll_script if self.auto_scroll else ""
        return ["".join(self._lines), script]
