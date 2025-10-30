"""
Main Doppler application file
"""
import os
import sys
import warnings

# Add project root to path to import shared modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from models import load_models
from layout import create_main_layout
from callbacks import register_callbacks

# Dash imports
import dash
import dash_bootstrap_components as dbc

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Doppler Effect Simulator"
app.config.suppress_callback_exceptions = True
server = app.server

# Load models
VELOCITY_MODEL_LOADED = load_models()

# Set up layout
app.layout = create_main_layout()

# Register callbacks
register_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8055)