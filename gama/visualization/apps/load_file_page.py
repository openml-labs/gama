import base64
import io

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

from gama.visualization.app import app
from gama.visualization.apps.single_report_page import single_report_page


load_file_page = html.Div([
    html.H1(children="GAMA Dashboard"),
    html.Div(children="Please select a log file to visualize:"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # For now we only visualize single traces:
        multiple=False
    ),
    html.Div(id='load-error-message')
])


@app.callback(Output('page-content', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        log_lines = decoded.splitlines()
        return single_report_page(log_lines)
    return load_file_page
