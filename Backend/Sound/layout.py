# --- SEARCHABLE COMMENT: Imports ---
from dash import dcc, html
import config # Import style constants and card styles

# ============================================
# --- SEARCHABLE COMMENT: Dash Layout Definition ---
# Defines the structure and components of the web application's UI.
# ============================================

def create_layout():
    """Creates the main layout structure for the Dash app."""
    layout = html.Div(
        # --- SEARCHABLE COMMENT: Main App Container Style ---
        style={
            "fontFamily": config.FONT_FAMILY,
            "backgroundColor": config.BACKGROUND_COLOR,
            "padding": "40px 20px", # Padding top/bottom and left/right
            "minHeight": "100vh" # Ensure background covers full height
        },
        children=[
            # --- SEARCHABLE COMMENT: Centered Content Container ---
            html.Div(
                style={ "maxWidth": "900px", "margin": "0 auto" }, # Center content horizontally
                children=[
                    # --- SEARCHABLE COMMENT: Header Section ---
                    html.Div(
                        style={"marginBottom": "32px"},
                        children=[
                            html.H1(
                                "🚁 Drone Sound Analysis & Sampling Explorer",
                                style={ 'textAlign': 'center', 'color': config.TEXT_COLOR, 'fontWeight': '800',
                                        'letterSpacing': '-0.5px', 'fontSize': '2.2rem', 'marginBottom': '8px' }
                            ),
                            html.P(
                                "Upload audio, classify using AI, and explore the Nyquist theorem interactively.",
                                style={ 'textAlign': 'center', 'color': config.SUBTLE_TEXT_COLOR, 'fontSize': '1.1rem', 'margin': '0' }
                            ),
                        ]
                    ), # End Header

                    # --- SEARCHABLE COMMENT: Upload Card ---
                    html.Div(
                        style=config.CARD_STYLE,
                        children=[
                            # --- SEARCHABLE COMMENT: Upload Component ---
                            dcc.Upload(
                                id="upload-audio",
                                children=html.Div(["📤 Drag & Drop or ", html.A("Select an Audio File", style={'color': config.PRIMARY_COLOR, 'fontWeight': '500'})]),
                                style=config.UPLOAD_STYLE,
                                multiple=False, # Allow only single file uploads
                            ),
                            # --- SEARCHABLE COMMENT: Analyze Button ---
                            html.Button(
                                "🚀 Analyze Original Audio",
                                id="classify-btn",
                                n_clicks=0,
                                disabled=True, # Initially disabled, enabled by callback
                                style=config.BUTTON_STYLE_PRIMARY # Apply primary button style
                            ),
                            # --- SEARCHABLE COMMENT: Upload Error Output ---
                            html.Div(
                                id="upload-error-output",
                                style={ 'color': config.ERROR_COLOR, 'marginTop': '15px', 'fontSize': '14px',
                                        'textAlign': 'center', 'fontWeight': '500' }
                            )
                        ]
                    ), # End Upload Card

                    # --- SEARCHABLE COMMENT: Results Card ---
                    # This card contains all analysis outputs and controls, initially hidden.
                    html.Div(
                        id="results-card",
                        style={**config.CARD_STYLE, 'display': 'none'}, # Initially hidden
                        children=[
                            # --- SEARCHABLE COMMENT: Loading Indicator (Analysis) ---
                            dcc.Loading(
                                id="loading-analysis",
                                type="circle", # Use circle style loader
                                children=[
                                    html.Div(
                                        id="results-content",
                                        children=[
                                            # --- SEARCHABLE COMMENT: Results Header ---
                                            html.H3(
                                                "Analysis Results",
                                                style={ "marginTop": 0, "borderBottom": f"2px solid {config.BORDER_COLOR}", # Thicker border
                                                        "paddingBottom": "15px", "marginBottom": "20px", "color": config.TEXT_COLOR }
                                            ),
                                            # --- SEARCHABLE COMMENT: File Name Display ---
                                            html.Div(
                                                id="file-name",
                                                style={ "marginBottom": "15px", "fontWeight": "600",
                                                        "color": config.SUBTLE_TEXT_COLOR, "fontSize": "15px" }
                                            ),
                                            # --- SEARCHABLE COMMENT: Classification Result (Original) ---
                                            # Content and style updated dynamically by callback
                                            html.Div(
                                                id="classification-result",
                                                style={ "fontSize": "22px", "fontWeight": "700", "textAlign": "center",
                                                        "marginBottom": "20px", "padding": "20px", "borderRadius": "10px",
                                                        'transition': 'all 0.3s ease-in-out' }
                                            ),
                                            # --- SEARCHABLE COMMENT: Audio Player (Original) ---
                                            html.Audio(
                                                id="audio-player", controls=True,
                                                style={ "width": "100%", "marginTop": "10px", "marginBottom": "25px" }
                                            ),

                                            # --- SEARCHABLE COMMENT: Explore Sampling Button ---
                                            # Toggles the visibility of the sampling controls section
                                            html.Button(
                                                "🔬 Explore Sampling & Aliasing", id="show-sampling-btn", n_clicks=0,
                                                style={**config.BUTTON_STYLE_SECONDARY, 'display': 'none'} # Hidden until analysis runs
                                            ),

                                            # --- SEARCHABLE COMMENT: Sampling Controls Section ---
                                            # Contains slider, plots, and buttons related to resampling. Initially hidden.
                                            html.Div(
                                                id="sampling-controls",
                                                style={'display': 'none', 'marginTop': '25px'}, # Hidden, appears on button click
                                                children=[
                                                    html.Hr(style={'borderTop': f'2px solid {config.BORDER_COLOR}', 'margin': '25px 0'}),
                                                    html.H4(
                                                        "Interactive Sampling Explorer",
                                                        style={ "marginTop": "0", "marginBottom": "10px", 'color': config.TEXT_COLOR }
                                                    ),
                                                    html.P(
                                                        "Adjust the sampling frequency (Fs) slider below. When Fs < Nyquist Rate, high frequencies alias into lower frequencies, creating distortion.",
                                                        style={ 'fontSize': '14px', 'color': config.SUBTLE_TEXT_COLOR,
                                                                'marginBottom': '20px', 'lineHeight': '1.5' }
                                                    ),
                                                    # --- SEARCHABLE COMMENT: Nyquist Info Display ---
                                                    # Shows calculated rates
                                                    html.Div(
                                                        id='nyquist-info',
                                                        style={ 'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap':'15px',
                                                                'margin': '20px 0', 'textAlign': 'center',
                                                                'backgroundColor': config.BACKGROUND_COLOR, 'padding': '15px',
                                                                'borderRadius': '8px', 'fontSize': '14px', 'border': f'1px solid {config.BORDER_COLOR}'}
                                                    ),
                                                    # --- SEARCHABLE COMMENT: Sampling Frequency Slider ---
                                                    dcc.Slider(
                                                        id='sampling-freq-slider', min=500, max=48000, step=100, value=8000,
                                                        tooltip={"placement": "bottom", "always_visible": True},
                                                        className='custom-slider' # Class for potential future CSS styling
                                                    ),

                                                    # --- SEARCHABLE COMMENT: Sampled Audio Buttons ---
                                                    # Grid layout for Play and Predict buttons
                                                    html.Div(
                                                        style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginTop': '25px'},
                                                        children=[
                                                            # --- SEARCHABLE COMMENT: Play Sampled Button ---
                                                            html.Button("▶️ Play Sampled", id="play-sampled-btn", n_clicks=0, style=config.BUTTON_STYLE_WARNING),
                                                            # --- SEARCHABLE COMMENT: Predict Sampled Button ---
                                                            html.Button("🧠 Predict Sampled", id="predict-sampled-btn", n_clicks=0, style=config.BUTTON_STYLE_SUCCESS),
                                                        ]
                                                    ),

                                                    # --- SEARCHABLE COMMENT: Playback Warning ---
                                                    # Displays warnings related to low sample rates or aliasing
                                                    html.Div(
                                                        id="playback-warning",
                                                        style={ 'color': config.WARNING_COLOR, 'textAlign': 'center', 'fontSize': '14px',
                                                                'fontWeight':'500', 'marginTop': '15px', 'padding':'10px',
                                                                'borderRadius':'6px', 'backgroundColor':'#FFF8E1', 'minHeight': '20px'} # Ensure space even when empty
                                                    ),
                                                    # --- SEARCHABLE COMMENT: Audio Player (Sampled) ---
                                                    html.Audio(
                                                        id="sampled-audio-player", controls=True,
                                                        style={"width": "100%", "marginTop": "15px", 'display': 'none'} # Initially hidden
                                                    ),

                                                    # --- SEARCHABLE COMMENT: Classification Result (Sampled) ---
                                                    # Includes loading indicator
                                                    dcc.Loading(
                                                        id="loading-sampled-prediction", type="circle",
                                                        children=[
                                                            html.Div(
                                                                id="sampled-classification-result",
                                                                # Style updated dynamically by callback
                                                                style={ "fontSize": "18px", "fontWeight": "600", "textAlign": "center",
                                                                        "marginTop": "25px", "padding": "20px", "borderRadius": "10px",
                                                                        'transition': 'all 0.3s ease-in-out', 'minHeight': '60px'}
                                                            ),
                                                        ]
                                                    )
                                                ]
                                            ), # End Sampling Controls Section

                                            # --- SEARCHABLE COMMENT: Waveform Plot ---
                                            # Displays either the initial preview or the interactive resampled view
                                            dcc.Graph(
                                                id="waveform-plot",
                                                config={"displayModeBar": False, 'staticPlot': False}, # Allow dynamic updates
                                                style={'marginTop':'25px'} # Add margin
                                            ),
                                        ]
                                    ) # End results-content
                                ] # End children of Loading
                            ) # End Loading
                        ] # End children of results-card
                    ), # End Results Card

                    # --- SEARCHABLE COMMENT: Data Store ---
                    # Hidden component to store intermediate data (like audio arrays) in the browser session
                    dcc.Store(id='audio-data-store'),
                ] # End children of main content container
            ) # End main content container
        ] # End children of app layout
    ) # End app layout
    return layout
