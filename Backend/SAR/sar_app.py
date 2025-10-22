import base64
import io
from datetime import datetime

# --- Core Data Handling Libraries ---
import numpy as np
import pandas as pd
from PIL import Image # Python Imaging Library for image manipulation

# --- Dash Framework Libraries ---
import dash
from dash import dcc, html, Input, Output, State, ctx # Core components, callback context
import plotly.graph_objs as go # Plotly for creating graphs
from dash.exceptions import PreventUpdate # To prevent callback updates when not needed


# ---------- Helper Functions ---------------------------------

# --- Image Handling: Decode Base64 ---
def pil_from_base64(b64_string):
    """Convert a base64 string (from dcc.Upload) to a PIL Image."""
    # Decode the base64 string, read bytes as an image, convert to RGBA
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert('RGBA')

# --- Image Handling: Encode Base64 ---
def image_to_base64_bytes(img, fmt='PNG'):
    """Convert a PIL Image to a base64 ASCII string (content only, no header)."""
    # Save image to a byte buffer
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    # Encode bytes to base64 and decode to ASCII string
    return base64.b64encode(buf.getvalue()).decode('ascii')

# --- Analysis: Compute Stats & Histogram ---
def compute_stats_and_histogram(pil_img, bins=50):
    """Calculate basic statistics and intensity histogram from a PIL Image."""
    # Convert image to grayscale for intensity calculations
    gray = pil_img.convert('L')
    # Flatten pixel intensities into a 1D numpy array
    intensities = np.array(gray).flatten()
    # Calculate basic descriptive statistics
    stats = {
        'mean': round(float(np.mean(intensities)), 2), 'median': round(float(np.median(intensities)), 2),
        'stdDev': round(float(np.std(intensities)), 2), 'min': round(float(np.min(intensities)), 2),
        'max': round(float(np.max(intensities)), 2), 'pixels': int(intensities.size),
        'width': pil_img.width, 'height': pil_img.height,
    }
    # Calculate histogram counts and bin edges
    hist_counts, bin_edges = np.histogram(intensities, bins=bins, range=(0, 255))
    # Create a Pandas DataFrame for the histogram data (bin centers and counts)
    histogram = pd.DataFrame({'intensity': 0.5 * (bin_edges[:-1] + bin_edges[1:]), 'count': hist_counts})
    return stats, histogram

# --- Processing: Apply Intensity Threshold ---
def apply_threshold_to_image(pil_img, threshold_percent):
    """Apply a threshold filter: Pixels below threshold are set to black."""
    # Calculate the actual intensity threshold value (0-255) based on percentage
    thr_value = (threshold_percent / 100.0) * 255.0
    # Convert image to grayscale to create the mask
    gray = pil_img.convert('L')
    # Create a boolean mask where pixels are below the threshold
    mask = np.array(gray) < thr_value
    # Convert original image to RGBA numpy array to modify pixels
    rgba_arr = np.array(pil_img.convert('RGBA'))
    # Apply the mask: set RGB values of masked pixels to 0 (black), keep original alpha
    rgba_arr[mask, :3] = 0
    # Convert the modified numpy array back to a PIL Image
    return Image.fromarray(rgba_arr)


# ---------- Dash App Layout -------------------------
# --- Initialize Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True) # Allow callbacks on components generated later
app.title = 'SAR Analysis Dashboard' # Set browser tab title

# --- UI STYLE CONSTANTS --- (Define styles for consistent appearance)
PRIMARY_COLOR = "#4A90E2"
BACKGROUND_COLOR = "#F7F9FC"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
SUBTLE_TEXT_COLOR = "#666666"
BORDER_COLOR = "#EAEAEA"
FONT_FAMILY = "Inter, system-ui, sans-serif"

# --- Reusable Card Style ---
CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
}

# --- Define App Layout ---
app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        # --- Main Content Container ---
        html.Div(style={"maxWidth": "1100px", "margin": "0 auto"}, children=[
            # --- Header ---
            html.Div([
                html.H1("🛰️ SAR Data Analysis Platform",
                        style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                html.P("Synthetic Aperture Radar Signal Processing & Feature Extraction",
                       style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '16px'}),
            ], style={'marginBottom': '20px'}),

            # --- Upload Area ---
            dcc.Upload(
                id='upload-image', children=html.Div(['📂 Drag & Drop or ', html.A('Select a SAR Image File')]),
                style={ # Styling for the upload component
                    'width': '100%', 'height': '100px', 'lineHeight': '100px', 'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px', 'textAlign': 'center', 'marginBottom': '20px', 'cursor': 'pointer',
                    'borderColor': BORDER_COLOR, 'backgroundColor': CARD_BACKGROUND_COLOR
                },
                accept='image/*', multiple=False # Accept image files, only one at a time
            ),

            # --- Tabs for Content Sections ---
            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Analysis', value='analysis'),
                dcc.Tab(label='Processing', value='processing')
            ], style={'marginBottom': '20px'}),

            # --- Tab Content Area ---
            # Content for the selected tab will be rendered here by a callback
            html.Div(id='tab-content'),

            # --- Hidden Stores for Data Persistence ---
            # Store base64 string of the originally uploaded/processed image
            dcc.Store(id='original-image-b64'),
            # Store base64 string of the image after applying processing steps (like threshold)
            dcc.Store(id='processed-image-b64'),
            # Store calculated image statistics (as a dictionary)
            dcc.Store(id='stats-store'),
            # Store histogram data (as a list of dictionaries for DataFrame reconstruction)
            dcc.Store(id='hist-store'),

            # --- Hidden Download Components ---
            # Used to trigger downloads via callbacks
            dcc.Download(id='download-csv'),
            dcc.Download(id='download-image')
        ])
    ])

# ---------- Callbacks -----------------------------

# --- Callback: Handle File Upload ---
@app.callback(
    [Output('original-image-b64', 'data'),      # Store for the initially processed image
     Output('processed-image-b64', 'data'),    # Store for image used in processing tab (initially same)
     Output('stats-store', 'data'),            # Store calculated statistics
     Output('hist-store', 'data')],            # Store histogram data
    [Input('upload-image', 'contents')],       # Triggered by file upload content change
    prevent_initial_call=True                  # Don't run on app start before upload
)
def handle_upload(contents):
    """Processes uploaded image, calculates stats/hist, and stores results."""
    # Prevent update if no content is provided (e.g., initial load)
    if not contents: raise PreventUpdate
    try:
        # Split the base64 string header from the content
        _, b64 = contents.split(',', 1)
        # Decode base64 and create PIL image
        pil_img = pil_from_base64(b64)
        # Compute statistics and histogram
        stats, hist_df = compute_stats_and_histogram(pil_img)
        # Encode the initial image back to base64 (content only)
        b64_str = image_to_base64_bytes(pil_img)
        # Return data for the stores (initial processed image is same as original view)
        return b64_str, b64_str, stats, hist_df.to_dict('records')
    except Exception as e:
        # Print error and return None to clear stores if upload/processing fails
        print(f"Upload error: {e}")
        return None, None, None, None

# --- Callback: Render Tab Content ---
@app.callback(
    Output('tab-content', 'children'),           # Target the Div to render tab content into
    [Input('tabs', 'value'),                    # Triggered when active tab changes
     Input('original-image-b64', 'data'),     # Triggered when original image updates
     Input('stats-store', 'data'),             # Triggered when stats update
     Input('hist-store', 'data'),              # Triggered when histogram updates
     Input('processed-image-b64', 'data')]      # Triggered when processing modifies image
)
def render_tab(tab, orig_b64, stats, hist_records, proc_b64):
    """Renders the appropriate layout based on the selected tab and available data."""
    # Show message if no data (stats) is loaded yet
    if not stats:
        return html.Div(style=CARD_STYLE, children=[
            html.H3('Upload SAR data to begin analysis', style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR}),
        ])

    # Recreate histogram DataFrame from stored list of dictionaries
    hist_df = pd.DataFrame(hist_records)

    # --- Overview Tab Layout ---
    if tab == 'overview':
        # Select specific statistics to display nicely
        stat_items = {k: v for k, v in stats.items() if k in ['mean', 'median', 'stdDev', 'min', 'max', 'pixels']}
        # Use CSS Grid for layout: Image on left (2/3 width), Stats on right (1/3 width)
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '24px'}, children=[
            # Left Column: Image Display Card
            html.Div(style=CARD_STYLE, children=[
                html.H3('SAR Intensity Image', style={'marginTop': 0}),
                # Display the currently processed image (proc_b64)
                html.Img(src=f'data:image/png;base64,{proc_b64}', style={'maxWidth': '100%', 'borderRadius': '8px'})
            ]),
            # Right Column: Properties & Stats Cards (using Flexbox for vertical stacking)
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[
                # Image Properties Card
                html.Div(style=CARD_STYLE, children=[
                    html.H3('Image Properties', style={'marginTop': 0}),
                    html.Div(f"Resolution: {stats['width']} x {stats['height']} px", style={'marginBottom': '5px'}),
                    html.Div(f"Total Pixels: {stats['pixels']:,}") # Format pixels with commas
                ]),
                # Signal Statistics Card
                html.Div(style=CARD_STYLE, children=[
                    html.H3('Signal Statistics', style={'marginTop': 0}),
                    # Loop through stat_items to create label/value pairs
                    html.Div([
                        html.Div([html.Span(k, style={'color': SUBTLE_TEXT_COLOR}), # Stat label
                                  html.Span(v, style={'fontWeight': 'bold'})],    # Stat value
                                 style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '5px 0'})
                        for k, v in stat_items.items()
                    ])
                ]),
            ])
        ])

    # --- Analysis Tab Layout ---
    if tab == 'analysis':
        # Create histogram figure using Plotly Graph Objects
        fig = go.Figure(data=[go.Bar(x=hist_df['intensity'], y=hist_df['count'], marker_color=PRIMARY_COLOR)])
        # Style the histogram figure
        fig.update_layout(title_text='Intensity Distribution', plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family=FONT_FAMILY, color=TEXT_COLOR), margin={'t': 40, 'b': 40})
        # Display the histogram in a card
        return html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig)])

    # --- Processing Tab Layout ---
    if tab == 'processing':
        # Use CSS Grid: Threshold controls on left, Export buttons on right
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, children=[
            # Left Column: Threshold Filter Card
            html.Div(style=CARD_STYLE, children=[
                html.H3('Threshold Filter', style={'marginTop': 0}),
                html.P("Adjust the slider to highlight pixels below a certain intensity.",
                       style={'color': SUBTLE_TEXT_COLOR}),
                # Slider for threshold percentage
                dcc.Slider(id='threshold-slider', min=0, max=100, step=1, value=50, marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}), # Show tooltip always
                # Button to apply the filter
                html.Button('Apply Filter', id='apply-threshold', n_clicks=0, style={ # Basic button styling
                    'marginTop': '20px', 'width': '100%', 'backgroundColor': PRIMARY_COLOR, 'color': 'white',
                    'border': 'none', 'borderRadius': '8px', 'padding': '12px', 'fontSize': '15px', 'cursor': 'pointer'
                }),
            ]),
            # Right Column: Export Data Card
            html.Div(style=CARD_STYLE, children=[
                html.H3('Export Data', style={'marginTop': 0}),
                # Button to export statistics
                html.Button('Export Statistics (CSV)', id='export-csv', n_clicks=0,
                            style={'width': '100%', 'padding': '12px', 'marginBottom': '10px'}), # Basic styling
                # Button to export the processed image
                html.Button('Export Processed Image (PNG)', id='export-image', n_clicks=0,
                            style={'width': '100%', 'padding': '12px'}), # Basic styling
            ])
        ])

# --- Callback: Apply Threshold Filter ---
@app.callback(
    # Update the 'processed-image-b64' store. Allow duplicate needed as handle_upload also sets it initially.
    Output('processed-image-b64', 'data', allow_duplicate=True),
    Input('apply-threshold', 'n_clicks'),        # Triggered by the 'Apply Filter' button
    [State('threshold-slider', 'value'),        # Get the current threshold slider value
     State('original-image-b64', 'data')],       # Get the *original* uploaded image data
    prevent_initial_call=True                   # Don't run on initial load
)
def handle_threshold(n_clicks, threshold_value, orig_b64):
    """Applies the threshold filter to the original image and updates the processed image store."""
    # Prevent update if no original image data exists
    if not orig_b64: raise PreventUpdate
    # Decode the original image from base64
    pil_img = pil_from_base64(orig_b64)
    # Apply the thresholding function
    processed_img = apply_threshold_to_image(pil_img, threshold_value)
    # Encode the thresholded image back to base64 and return it
    return image_to_base64_bytes(processed_img)

# --- Callback: Export Statistics CSV ---
@app.callback(
    Output('download-csv', 'data'),              # Target the dcc.Download component for CSV
    Input('export-csv', 'n_clicks'),             # Triggered by the 'Export Statistics (CSV)' button
    State('stats-store', 'data'),                # Get the stored statistics data
    prevent_initial_call=True                   # Don't run on initial load
)
def export_csv(n_clicks, stats):
    """Generates and triggers the download of the statistics CSV file."""
    # Prevent update if no stats data exists
    if not stats: raise PreventUpdate
    # Convert the stats dictionary into a Pandas DataFrame
    df = pd.DataFrame([stats])
    # Use dcc.send_data_frame to create the CSV download data
    # Generate a filename with a timestamp
    return dcc.send_data_frame(df.to_csv, f"sar_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

# --- Callback: Export Processed Image PNG ---
@app.callback(
    Output('download-image', 'data'),            # Target the dcc.Download component for the image
    Input('export-image', 'n_clicks'),           # Triggered by the 'Export Processed Image (PNG)' button
    State('processed-image-b64', 'data'),        # Get the stored base64 string of the processed image
    prevent_initial_call=True                   # Don't run on initial load
)
def export_image(n_clicks, proc_b64):
    """Generates and triggers the download of the processed image PNG file."""
    # Prevent update if no processed image data exists
    if not proc_b64: raise PreventUpdate
    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(proc_b64)
    # Use dcc.send_bytes to create the PNG download data
    # Generate a filename with a timestamp
    return dcc.send_bytes(image_bytes, f"processed_sar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the Dash app server
    app.run(debug=True, port=8053) # Enable debug mode, run on specified port

