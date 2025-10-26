# --- SEARCHABLE COMMENT: Imports ---
from dash import Dash
# Local module imports
import layout         # UI layout structure
import callbacks      # Application interactivity
import model_utils    # For preloading the model

# ============================================
# --- SEARCHABLE COMMENT: Dash App Initialization ---
# ============================================
# Create the main Dash app instance
# suppress_callback_exceptions is needed because callbacks are defined in a separate file
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server # Expose server instance for deployments (e.g., Gunicorn)
app.title = "Drone Sound Analysis" # Set browser tab title

# ============================================
# --- SEARCHABLE COMMENT: Assign Layout ---
# ============================================
# Create the layout by calling the function from layout.py
app.layout = layout.create_layout()

# ============================================
# --- SEARCHABLE COMMENT: Register Callbacks ---
# ============================================
# Register all callbacks defined in callbacks.py with the app instance
callbacks.register_callbacks(app)

# ============================================
# --- SEARCHABLE COMMENT: Main Execution Block ---
# ============================================
if __name__ == "__main__":
    # --- SEARCHABLE COMMENT: Preload Model ---
    # Attempt to load the AI model when the application starts
    print("Attempting to preload model...")
    model_utils.ensure_model()

    # --- SEARCHABLE COMMENT: Run Dash Server ---
    # Start the Dash development server
    print("Starting Dash server...")
    app.run(debug=True, port=8051) # Run on port 8051 with debugging enabled

