import base64
import io
import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

# Add the parent directory to Python path to find shared module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# --- Shared Utilities ---
from shared.config import *
from shared.file_utils import pil_from_base64, image_to_base64_bytes
from shared.analysis_utils import AnalysisUtils
from shared.plotting_utils import PlottingUtils
from shared.ui_components import create_upload_component, create_card, create_button

# Initialize shared utilities
analysis_utils = AnalysisUtils()
plotting_utils = PlottingUtils()

# ---------- Helper Functions ---------------------------------

def apply_threshold_to_image(pil_img, threshold_percent):
    """Apply a threshold filter: Pixels below threshold are set to black."""
    return analysis_utils.apply_image_threshold(pil_img, threshold_percent)

# ---------- Dash App Layout -------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'SAR Analysis Dashboard'

app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        html.Div(style={"maxWidth": "1100px", "margin": "0 auto"}, children=[
            # --- Header ---
            html.Div([
                html.H1("🛰️ SAR Data Analysis Platform",
                        style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                html.P("Synthetic Aperture Radar Signal Processing & Feature Extraction",
                       style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '16px'}),
            ], style={'marginBottom': '20px'}),

            # --- Upload Area using shared component ---
            create_upload_component(
                'upload-image', 
                accept_types='image/*',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select a SAR Image File')])
            ),

            # --- Tabs for Content Sections ---
            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Analysis', value='analysis'),
                dcc.Tab(label='Processing', value='processing')
            ], style={'marginBottom': '20px'}),

            html.Div(id='tab-content'),

            # --- Hidden Stores for Data Persistence ---
            dcc.Store(id='original-image-b64'),
            dcc.Store(id='processed-image-b64'),
            dcc.Store(id='stats-store'),
            dcc.Store(id='hist-store'),

            dcc.Download(id='download-csv'),
            dcc.Download(id='download-image')
        ])
    ])

# ---------- Callbacks -----------------------------

@app.callback(
    [Output('original-image-b64', 'data'),
     Output('processed-image-b64', 'data'),
     Output('stats-store', 'data'),
     Output('hist-store', 'data')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def handle_upload(contents):
    """Processes uploaded image, calculates stats/hist, and stores results."""
    if not contents: 
        raise PreventUpdate
    try:
        _, b64 = contents.split(',', 1)
        pil_img = pil_from_base64(b64)
        stats, hist_df = analysis_utils.compute_image_stats_and_histogram(pil_img)
        b64_str = image_to_base64_bytes(pil_img)
        return b64_str, b64_str, stats, hist_df.to_dict('records')
    except Exception as e:
        print(f"Upload error: {e}")
        return None, None, None, None

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('original-image-b64', 'data'),
     Input('stats-store', 'data'),
     Input('hist-store', 'data'),
     Input('processed-image-b64', 'data')]
)
def render_tab(tab, orig_b64, stats, hist_records, proc_b64):
    if not stats:
        return create_card('', html.H3('Upload SAR data to begin analysis', 
                         style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR}))

    hist_df = pd.DataFrame(hist_records)

    if tab == 'overview':
        stat_items = {k: v for k, v in stats.items() if k in ['mean', 'median', 'stdDev', 'min', 'max', 'pixels']}
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '24px'}, children=[
            create_card('SAR Intensity Image', 
                html.Img(src=f'data:image/png;base64,{proc_b64}', 
                        style={'maxWidth': '100%', 'borderRadius': '8px'})),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '24px'}, children=[
                create_card('Image Properties', html.Div([
                    html.Div(f"Resolution: {stats['width']} x {stats['height']} px", 
                            style={'marginBottom': '5px'}),
                    html.Div(f"Total Pixels: {stats['pixels']:,}")
                ])),
                create_card('Signal Statistics', html.Div([
                    html.Div([html.Span(k, style={'color': SUBTLE_TEXT_COLOR}),
                              html.Span(v, style={'fontWeight': 'bold'})],
                             style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '5px 0'})
                    for k, v in stat_items.items()
                ])),
            ])
        ])

    elif tab == 'analysis':
        fig = plotting_utils.create_histogram_plot(hist_df, 'Intensity Distribution')
        return create_card('Analysis', dcc.Graph(figure=fig))

    elif tab == 'processing':
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, children=[
            create_card('Threshold Filter', [
                html.H3('Threshold Filter', style={'marginTop': 0}),
                html.P("Adjust the slider to highlight pixels below a certain intensity.",
                       style={'color': SUBTLE_TEXT_COLOR}),
                dcc.Slider(id='threshold-slider', min=0, max=100, step=1, value=50, marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
                create_button('Apply Filter', 'apply-threshold', variant='primary', 
                            style={'marginTop': '20px'})
            ]),
            create_card('Export Data', [
                html.H3('Export Data', style={'marginTop': 0}),
                create_button('Export Statistics (CSV)', 'export-csv', variant='secondary',
                            style={'marginBottom': '10px'}),
                create_button('Export Processed Image (PNG)', 'export-image', variant='secondary')
            ])
        ])

@app.callback(
    Output('processed-image-b64', 'data', allow_duplicate=True),
    Input('apply-threshold', 'n_clicks'),
    [State('threshold-slider', 'value'),
     State('original-image-b64', 'data')],
    prevent_initial_call=True
)
def handle_threshold(n_clicks, threshold_value, orig_b64):
    if not orig_b64: 
        raise PreventUpdate
    pil_img = pil_from_base64(orig_b64)
    processed_img = apply_threshold_to_image(pil_img, threshold_value)
    return image_to_base64_bytes(processed_img)

@app.callback(
    Output('download-csv', 'data'),
    Input('export-csv', 'n_clicks'),
    State('stats-store', 'data'),
    prevent_initial_call=True
)
def export_csv(n_clicks, stats):
    if not stats: 
        raise PreventUpdate
    df = pd.DataFrame([stats])
    return dcc.send_data_frame(df.to_csv, f"sar_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

@app.callback(
    Output('download-image', 'data'),
    Input('export-image', 'n_clicks'),
    State('processed-image-b64', 'data'),
    prevent_initial_call=True
)
def export_image(n_clicks, proc_b64):
    if not proc_b64: 
        raise PreventUpdate
    image_bytes = base64.b64decode(proc_b64)
    return dcc.send_bytes(image_bytes, f"processed_sar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

if __name__ == '__main__':
    app.run(debug=True, port=8053)