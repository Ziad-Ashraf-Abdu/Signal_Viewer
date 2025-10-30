"""
Layout components for Doppler application
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
from shared.ui_components import *

def create_main_layout():
    """Create the main layout for Doppler application"""
    return html.Div([
        # Header
        create_navigation_header(
            "Doppler Effect Simulator",
            "Upload a WAV file to analyze aliasing and simulate Doppler effect with real audio",
            icon="🚗"
        ),

        # Hidden stores
        dcc.Store(id='simulation-running', data=False),
        dcc.Store(id='time-elapsed', data=0),
        dcc.Store(id='sound-freq', data=0),
        dcc.Store(id='uploaded-audio-data', data=None),
        dcc.Store(id='aliasing-audio', data=None),
        dcc.Store(id='predicted-velocity-store', data=None),
        html.Div(id='sound-init', style={'display': 'none'}),
        html.Div(id='sound-div', style={'display': 'none'}),
        dcc.Interval(id='interval', interval=100, n_intervals=0, disabled=True),

        # Upload Section
        create_card(
            "📁 Upload Car Audio (WAV)",
            [
                create_upload_component(
                    'upload-audio',
                    accept_types='.wav',
                    children=html.Div(['Drag and Drop or ', html.A('Select a WAV File')])
                ),
                html.Div(id='upload-status', style={'fontSize': '14px', 'color': '#6b7280', 'marginBottom': '10px'}),
            ],
            card_id="upload-card"
        ),

        # Aliasing Section
        create_card(
            "📉 Aliasing Demonstration",
            [
                html.Div([
                    html.Label("Target Sampling Frequency (Hz):", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='fs-slider',
                        min=1000,
                        max=44100,
                        step=100,
                        value=22050,
                        disabled=True
                    ),
                    html.Div(id='slider-value-display', 
                            style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': '18px'})
                ], style={'padding': '20px', 'backgroundColor': '#f0fdf4', 'borderRadius': '10px', 'marginBottom': '20px'}),

                html.Div([
                    html.H4("🔊 Original Audio"),
                    html.Audio(id='audio-orig', controls=True, style={'width': '100%'})
                ]),
                html.Div([
                    html.H4("🔊 Aliased Audio (Downsampled – No Anti-Aliasing)"),
                    html.Audio(id='audio-alias', controls=True, style={'width': '100%'})
                ])
            ],
            card_id="aliasing-card"
        ),

        # Analysis & Prediction Section
        create_card(
            "🎵 Analysis & Prediction",
            [
                html.H4(id='detected-freq-display', children="Detected Frequency: — Hz", style={
                    'color': '#ef4444', 'fontSize': '20px', 'fontWeight': '700', 'marginBottom': '15px'
                }),
                html.H4(id='predicted-velocity-display', children="Predicted Velocity: — m/s", style={
                    'color': '#10b981', 'fontSize': '20px', 'fontWeight': '700', 'marginBottom': '15px'
                }),
                html.Div([
                    create_button('📊 Use This Frequency', 'use-audio-freq-btn', disabled=True, className="me-3"),
                    create_button('🚗 Use This Velocity', 'use-audio-velocity-btn', disabled=True, variant="success")
                ]),
                dcc.Graph(id='audio-spectrum-graph', style={'height': '350px'})
            ],
            card_id="analysis-card"
        ),

        # Doppler Simulation Section
        create_card(
            "🔊 Doppler Effect Simulation",
            [
                # Source and Observer Panels
                dbc.Row([
                    dbc.Col(create_source_panel(), width=6),
                    dbc.Col(create_observer_panel(), width=6)
                ], className="mb-4"),

                # Frequency Input
                html.Div([
                    html.Div([
                        create_frequency_input()
                    ], style={
                        'display': 'inline-block',
                        'padding': '20px 40px',
                        'backgroundColor': 'white',
                        'borderRadius': '15px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                        'border': '2px solid #e0e0e0'
                    })
                ], style={'textAlign': 'center', 'marginBottom': '25px'}),

                # Control Buttons
                html.Div([
                    create_button('▶️ Start', 'start-btn', variant="success", className="me-3"),
                    create_button('⏸️ Pause', 'pause-btn', variant="warning", className="me-3"),
                    create_button('⏹️ Reset', 'reset-btn', variant="error", className="me-3"),
                    create_button('🔈 Mute', 'mute-btn', variant="secondary"),
                    html.Span(id='mute-label', children='', style={
                        'marginLeft': '16px', 'fontWeight': '600', 'fontSize': '16px', 'color': '#374151'
                    })
                ], style={'textAlign': 'center', 'marginBottom': '25px'}),

                # Frequency Display
                html.Div(id='frequency-display', style={
                    'fontSize': '20px', 'textAlign': 'center', 'padding': '20px',
                    'fontWeight': '600', 'color': '#1f2937', 'backgroundColor': 'white',
                    'borderRadius': '12px', 'marginBottom': '25px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                    'border': '2px solid #e5e7eb'
                }),

                # Simulation Graph
                dcc.Graph(id='simulation-graph', style={'height': '60vh'})
            ],
            card_id="simulation-card"
        )
    ], style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        'backgroundColor': '#f9fafb',
        'minHeight': '100vh',
        'paddingBottom': '40px'
    })

def create_source_panel():
    """Create sound source control panel"""
    return html.Div([
        html.H3("🔊 Sound Source", style={
            'color': '#667eea', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': '700',
            'borderBottom': '3px solid #667eea', 'paddingBottom': '10px'
        }),
        dcc.RadioItems(
            id='source-type',
            options=[{'label': ' Moving', 'value': 'moving'}, {'label': ' Static', 'value': 'static'}],
            value='moving', inline=True, style={'marginBottom': '20px'},
            labelStyle={'marginRight': '20px', 'fontSize': '15px', 'fontWeight': '500'}
        ),
        labeled_input("Start X (m):", 'source-x0', -200),
        labeled_input("Start Y (m):", 'source-y0', 0),
        html.Div(id='source-vel-inputs', children=[
            labeled_input("Speed (m/s):", 'source-speed', 30),
            labeled_input("Direction (°):", 'source-dir', 0)
        ])
    ], style={
        'padding': '25px', 'backgroundColor': '#f8f9ff', 'borderRadius': '15px',
        'border': '2px solid #667eea', 'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.15)'
    })

def create_observer_panel():
    """Create observer control panel"""
    return html.Div([
        html.H3("👂 Observer", style={
            'color': '#f093fb', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': '700',
            'borderBottom': '3px solid #f093fb', 'paddingBottom': '10px'
        }),
        dcc.RadioItems(
            id='observer-type',
            options=[{'label': ' Moving', 'value': 'moving'}, {'label': ' Static', 'value': 'static'}],
            value='moving', inline=True, style={'marginBottom': '20px'},
            labelStyle={'marginRight': '20px', 'fontSize': '15px', 'fontWeight': '500'}
        ),
        labeled_input("Start X (m):", 'observer-x0', 0),
        labeled_input("Start Y (m):", 'observer-y0', 0),
        html.Div(id='observer-vel-inputs', children=[
            labeled_input("Speed (m/s):", 'observer-speed', 10),
            labeled_input("Direction (°):", 'observer-dir', 180)
        ])
    ], style={
        'padding': '25px', 'backgroundColor': '#fff8fd', 'borderRadius': '15px',
        'border': '2px solid #f093fb', 'boxShadow': '0 4px 12px rgba(240, 147, 251, 0.15)'
    })

def create_frequency_input():
    """Create frequency input component"""
    return labeled_input("Emitted Frequency (Hz):", 'freq-input', 500, width=100)

def labeled_input(label, id, value, width=80):
    """Create labeled input field"""
    from dash import dcc, html
    return html.Div([
        html.Label(label, style={
            'display': 'inline-block', 'width': '140px', 'fontWeight': '600',
            'color': '#2c3e50', 'fontSize': '14px'
        }),
        dcc.Input(id=id, type='number', value=value, style={
            'width': f'{width}px', 'padding': '8px 12px', 'border': '2px solid #e0e0e0',
            'borderRadius': '8px', 'fontSize': '14px', 'outline': 'none'
        })
    ], style={'marginBottom': '15px'})