# app.py
import dash
import dash_bootstrap_components as dbc  # Required for layout and callbacks
from layout import create_layout
from callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Audio Analysis Dashboard"
server = app.server

# Set layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, port=8054)