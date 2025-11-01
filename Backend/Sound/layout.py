import sys
import os

# Correct path setup - shared is in Backend/shared/
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', 'shared')

if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from dash import dcc, html
from shared.config import *
from shared.ui_components import (
    create_upload_component, 
    create_card, 
    create_button,
    create_navigation_header,
    create_main_container
)

def create_layout():
    """Creates the main layout structure for the Sound app using shared components."""
    
    return create_main_container([
        create_navigation_header(
            "Drone Sound Analysis & Sampling Explorer",
            "Upload audio, classify using AI, and explore the Nyquist theorem interactively",
            icon="🚁"
        ),

        create_card("📁 Upload Audio", [
            create_upload_component(
                "upload-audio", 
                accept_types='audio/*',
                children=html.Div(['📤 Drag & Drop or ', html.A('Select an Audio File')])
            ),
            create_button(
                "🚀 Analyze Original Audio", 
                "classify-btn", 
                variant="primary", 
                disabled=True,
                style_overrides={'marginTop': '15px'}
            ),
            html.Div(
                id="upload-error-output",
                style={'color': ERROR_COLOR, 'marginTop': '15px', 'fontSize': '14px', 'textAlign': 'center'}
            )
        ]),

        html.Div(
            id="results-card",
            style={'display': 'none'},
            children=[
                create_card("Analysis Results", [
    html.Div(id="file-name", style={"marginBottom": "15px", "fontWeight": "600", "color": SUBTLE_TEXT_COLOR}),

    # --- THIS IS THE ADDED WRAPPER ---
    dcc.Loading(id="loading-original-prediction", type="circle", children=[
        html.Div(id="classification-result", style={"fontSize": "22px", "fontWeight": "700", "textAlign": "center", "marginBottom": "20px", "padding": "20px", "borderRadius": "10px", 'minHeight': '60px'})
    ]),
    # --- END OF ADDED WRAPPER ---

    html.Audio(id="audio-player", controls=True, style={"width": "100%", "marginTop": "10px", "marginBottom": "25px"}),
    create_button("🔬 Explore Sampling & Aliasing", "show-sampling-btn", variant="secondary", style_overrides={'display': 'none', 'marginTop': '15px'}),
                    
                    html.Div(id="sampling-controls", style={'display': 'none', 'marginTop': '25px'}, children=[
                        html.Hr(style={'borderTop': f'2px solid {BORDER_COLOR}', 'margin': '25px 0'}),
                        create_card("Interactive Sampling Explorer", [
                            html.P("Adjust the sampling frequency (Fs) slider below. When Fs < Nyquist Rate, high frequencies alias into lower frequencies, creating distortion.", style={'fontSize': '14px', 'color': SUBTLE_TEXT_COLOR, 'marginBottom': '20px'}),
                            html.Div(id='nyquist-info', style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap':'15px', 'margin': '20px 0', 'textAlign': 'center', 'backgroundColor': BACKGROUND_COLOR, 'padding': '15px', 'borderRadius': '8px', 'fontSize': '14px', 'border': f'1px solid {BORDER_COLOR}'}),
                            dcc.Slider(id='sampling-freq-slider', min=500, max=48000, step=100, value=8000, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginTop': '25px'}, children=[
                                create_button("▶️ Play Sampled", "play-sampled-btn", variant="warning"),
                                create_button("🧠 Predict Sampled", "predict-sampled-btn", variant="success"),
                            ]),
                            html.Div(id="playback-warning", style={'color': WARNING_COLOR, 'textAlign': 'center', 'fontSize': '14px', 'fontWeight':'500', 'marginTop': '15px', 'padding':'10px', 'borderRadius':'6px', 'backgroundColor':'#FFF8E1', 'minHeight': '20px'}),
                            html.Audio(id="sampled-audio-player", controls=True, style={"width": "100%", "marginTop": "15px", 'display': 'none'}),
                            dcc.Loading(id="loading-sampled-prediction", type="circle", children=[
                                html.Div(id="sampled-classification-result", style={"fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px", "padding": "20px", "borderRadius": "10px", 'minHeight': '60px'}),
                            ])
                        ])
                    ]),
                    dcc.Graph(id="waveform-plot", config={"displayModeBar": False, 'staticPlot': False}, style={'marginTop':'25px'}),
                ])
            ]
        ),
        dcc.Store(id='audio-data-store'),
    ])