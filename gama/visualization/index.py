import dash_core_components as dcc
import dash_html_components as html

from gama.visualization.app import app
from gama.visualization.apps.load_file_page import load_file_page


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1(children="GAMA Dashboard", style={'textAlign': 'center'}),
    html.Div(id='page-content')
])


if __name__ == '__main__':
    app.layout['page-content'].children = load_file_page
    app.run_server(debug=True)
