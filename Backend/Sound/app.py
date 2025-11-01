import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# We want to add the 'Backend' directory, which is one level up
backend_path = os.path.normpath(os.path.join(current_dir, '..'))
print(f"Looking for modules in: {backend_path}")
if os.path.exists(backend_path):
    print(f"Path exists: True")
    sys.path.append(backend_path)  # <-- THIS IS THE FIX
    print("✅ Backend path added to Python path")
else:
    print(f"Path exists: False. Could not find: {backend_path}")

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