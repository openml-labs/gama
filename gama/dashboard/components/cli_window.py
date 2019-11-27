import shlex
import subprocess
import threading
import queue

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import visdcc


def enqueue_output(out, queue_: queue.Queue):
    for line in iter(out.readline, b''):
        queue_.put(line)
    out.close()


class CLIWindow:
    """ A Component for Dash App which simulates a console window.


    """
    def __init__(self, id_: str, app: 'Dash', update_interval_s: float = 1.0, auto_scroll: bool = True):
        self._update_interval_s = update_interval_s
        self.auto_scroll = auto_scroll

        self.console_id = f'{id_}-text'
        self.timer_id = f'{id_}-interval'
        self.js_id = f'{id_}-js'
        self.id_ = id_

        self.autoscroll_script = f"document.getElementById('{self.console_id}').scrollTop = document.getElementById('{self.console_id}').scrollHeight"
        self.process = None
        self._thread = None
        self._queue = None
        self._lines = []

        self.html = self._build_component()
        self._register_callbacks(app)

    def _build_component(self) -> html.Div:
        timer = dcc.Interval(
            id=self.timer_id,
            interval=self._update_interval_s * 1000,
            n_intervals=0
        )
        scroller = visdcc.Run_js(id=self.js_id, run='')
        console = dcc.Textarea(
            id=self.console_id,
            contentEditable='false',
            style={'height': '200px', 'width': '100%', 'borderWidth': '1px', 'borderRadius': '5px', 'borderStyle': 'dashed'}
        )
        return html.Div(
            id=self.id_,
            children=[timer, console, scroller]
        )

    def _register_callbacks(self, app):
        app.callback(
            [Output(self.console_id, 'value'),
             Output(self.console_id, 'disabled'),
             Output(self.js_id, 'run')],
            [Input(self.timer_id, 'n_intervals')]
        )(self.update_console)

    def call(self, command: str):
        command = shlex.split(command)
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        # Because there are only blocking reads to the pipe,
        # we need to read them on a separate thread.
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=enqueue_output, args=(self.process.stdout, self._queue), daemon=True)
        self._thread.start()

    def update_console(self, _):
        if self.process is None:
            return [None, True, None]
        try:
            line = self._queue.get_nowait()
            self._lines.append(line.decode('utf-8'))
        except queue.Empty:
            # No new message, no update required.
            raise PreventUpdate
        return [''.join(self._lines), True, self.autoscroll_script if self.auto_scroll else '']
