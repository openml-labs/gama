from typing import List

try:
    import dash
    import dash_core_components as dcc
    import dash_bootstrap_components as dbc
    import dash_html_components as html
    from dash.dependencies import Input, Output
except ImportError:
    print(
        "Unable to import Dash modules. "
        "Make sure GAMA is installed with the optional visualization dependencies."
        "E.g.: pip install gama[vis]"
    )
    raise

from gama.dashboard.controller import Controller
from gama.dashboard.pages.base_page import BasePage


external_scripts = [
    "https://code.jquery.com/jquery-3.3.1.min.js",
    "https://cdn.datatables.net/v/dt/dt-1.10.18/datatables.min.js",
]
dashboard = dash.Dash(
    "GamaDashboard",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=external_scripts,
)
dashboard.config.suppress_callback_exceptions = True

controller = Controller()


# === Construct UI elements ===


def build_app():
    from gama.dashboard.pages import pages

    base = create_generic_layout()
    base["tabs"].children = create_tabs(pages)
    page_stores = []
    for page in pages:
        page.build_page(dashboard, controller)
        if hasattr(page, "gama_started"):
            controller.gama_started(page.gama_started)
        page_stores.append(
            dcc.Store(id=f"{page.id}-store", storage_type="session", data={})
        )
    base["page-stores"].children = page_stores
    return base


def create_generic_layout():
    """ Creates the generic layout of tabs and their content pages. """
    tab_banner_style = {
        "border-top-left-radius": "3px",
        "background-color": "#f9f9f9",
        "padding": "0px 24px",
        "border-bottom": "1px solid #d6d6d6",
    }

    return html.Div(
        id="page",
        children=[
            html.Div(id="tabs", style=tab_banner_style),
            html.Div(id="content"),
            html.Div(id="page-stores"),
        ],
        style={"font-family": "'Open Sans Semi Bold', sans-serif"},
    )


def create_tabs(pages: List[BasePage]):
    if pages == []:
        raise ValueError("Must have at least one tab.")

    # Sort pages by alignment
    sorted_pages = sorted(pages, key=lambda p: p.alignment)
    left_aligned = [page for page in sorted_pages if page.alignment >= 0]
    right_aligned = [page for page in sorted_pages if page.alignment < 0]
    grouped_pages = left_aligned + right_aligned
    tabs = [create_tab(page.name, page.starts_hidden) for page in grouped_pages]
    return [dcc.Tabs(id="page-tabs", value=tabs[0].value, children=tabs)]


def create_tab(name: str, invisible: bool):
    tab_style = {
        "color": "black",
        "width": "10%",
        "border-top-left-radius": "3px",  # round tab corners
        "border-top-right-radius": "3px",
        "border-bottom": "0px",  # bottom box-shadow still present
        "padding": "6px",
        "display": "none" if invisible else "inline",
    }
    selected_tab_style = {
        **tab_style,
        "border-top": "3px solid #c81818",  # Highlight color (TU/e colored)
        "box-shadow": "1px 1px 0px white",  # removes bottom edge
        "display": "inline",
    }
    return dcc.Tab(
        id=f"{name}-tab",
        label=name,
        value=name,
        style=tab_style,
        selected_style=selected_tab_style,
    )


# === Callbacks ===


@dashboard.callback([Output("content", "children")], [Input("page-tabs", "value")])
def display_page_content(page_name):
    from gama.dashboard.pages import pages

    page = [page for page in pages if page.name == page_name][0]
    page.need_update = True
    return [page.content]


def main(open_window: bool = True, **server_kwargs):
    dashboard.layout = build_app()
    # if open_window:
    #     import webbrowser
    #     from threading import Timer
    #
    #     port = server_kwargs.get("port", 8050)
    #     localhost = f"http://127.0.0.1:{port}"
    #
    #     Timer(1, webbrowser.open_new, kwargs=dict(url=localhost)).start()
    dashboard.run_server(**server_kwargs)


if __name__ == "__main__":
    main(debug=True)
