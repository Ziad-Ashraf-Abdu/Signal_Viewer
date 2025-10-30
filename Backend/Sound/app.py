import sys
import os

# Correct path setup - shared is in Backend/shared/
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', 'shared')

print(f"Looking for shared module at: {shared_path}")
print(f"Path exists: {os.path.exists(shared_path)}")

if shared_path not in sys.path:
    sys.path.insert(0, shared_path)
    print("✅ Shared path added to Python path")

from dash import Dash
import layout
import callbacks

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Drone Sound Analysis"

app.layout = layout.create_layout()
callbacks.register_callbacks(app)

if __name__ == "__main__":
    print("Starting Drone Sound Analysis server...")
    app.run(debug=True, port=8051)