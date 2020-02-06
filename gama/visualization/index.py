import dash_core_components as dcc
import dash_html_components as html

from gama.visualization.app import dash_app
from gama.visualization.apps.dashboard_page import dashboard_page, dashboard_header

app_title = 'GAMA Dashboard'


upload_box = html.Div(
    id='upload-container',
    children=[
        dcc.Upload(
            id='upload-box',
            children=html.Div([html.A('Select or drop log(s).')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'display': 'inline-block'
            },
            multiple=True
        )
    ],
    style=dict(width=f'{len("Select or drop log(s).")}em', display='inline-block', float='right')
)

top_bar = html.Div(
    id='top-bar',
    children=[
        html.H1(children=f"{app_title}", style={'float': 'left', 'width': f'30%', 'display': 'inline-block'}),
        html.Div(id='header-box', children=[], style=dict(float='left', width='40%')),
        upload_box,
 ])

dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    top_bar,
    html.Div(id='page-content')
])

dash_app.layout['page-content'].children = dashboard_page
dash_app.layout['header-box'].children = dashboard_header


if __name__ == '__main__':
    dash_app.run_server(debug=True, port=5001)
