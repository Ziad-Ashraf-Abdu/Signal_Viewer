"""
Layout components for Human application
"""
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc, html
from shared.ui_components import *

def create_audio_players(original_data, downsampled_data, reconstructed_data):
    """Generate audio players for different audio versions"""
    audio_data_dict = {}
    
    if original_data:
        audio_data_dict['original'] = {
            'data': np.array(original_data['waveform']), 
            'sr': original_data['sample_rate']
        }
    
    if downsampled_data:
        audio_data_dict['downsampled'] = {
            'data': np.array(downsampled_data['waveform']), 
            'sr': downsampled_data['sample_rate']
        }
    
    if reconstructed_data:
        audio_data_dict['reconstructed'] = {
            'data': np.array(reconstructed_data['waveform']), 
            'sr': reconstructed_data['sample_rate']
        }
    
    return create_audio_comparison_players(audio_data_dict)

def create_main_layout():
    """Create the main layout for Human application"""
    return create_main_container([
        # Header Section
        dbc.Row(
            dbc.Col(
                create_navigation_header(
                    "Interactive Audio Sampling & Analysis", 
                    "Upload an audio file to downsample, reconstruct, and test the effects on an AI model.",
                    icon="🎧"
                )
            )
        ),

        # Hidden stores
        dcc.Store(id='original-audio-data'),
        dcc.Store(id='downsampled-audio-data'),
        dcc.Store(id='reconstructed-audio-data'),

        # Section 1: Audio Upload
        create_card(
            "1. Load Audio",
            [
                create_upload_component(
                    'upload-audio',
                    accept_types='audio/*',
                    children=html.Div(['Drag and Drop or ', html.A('Select Audio File')])
                ),
                html.Div(id='upload-status', className="mt-3")
            ],
            card_id="upload-card"  # Added card_id
        ),

        # Section 2: Audio Processing Controls
        create_card(
            "2. Process Audio",
            [
                html.Div([
                    html.H4("Downsample Frequency", className="text-danger mb-2"),
                    dcc.Slider(  # Using direct dcc.Slider instead of create_slider
                        id='target-sr-slider',
                        min=500,
                        max=16000,
                        step=500,
                        value=8000,
                        marks={
                            500: '0.5k',
                            **{i: f'{i // 1000}k' for i in range(2000, 17000, 2000)}
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], className="mb-4"),
                html.Div([
                    dbc.Button(  # Using dbc.Button directly
                        'Apply AI Reconstruction',
                        id='model-reconstruct-btn',
                        color="success",
                        className="me-3"
                    ),
                    html.Span(id='model-status', style={'fontWeight': 'bold'})
                ], className="text-center mt-3"),
            ],
            card_id="processing-card"  # Added card_id
        ),

        # Section 3: AI Model Testing
        create_card(
            "3. Test Audio on Model",
            [
                html.P("Select an audio signal and test its effect on the model's prediction.", 
                       className="text-center mb-3"),
                dcc.Dropdown(
                    id='test-audio-selector',
                    options=[
                        {'label': 'Original Audio', 'value': 'original'},
                        {'label': 'Downsampled (Aliased) Audio', 'value': 'downsampled'},
                        {'label': 'AI Reconstructed Audio', 'value': 'reconstructed'}
                    ],
                    value='original',
                    clearable=False,
                    className="mb-3"
                ),
                dbc.Button(  # Using dbc.Button directly
                    'Run Gender Detection Test',
                    id='run-test-btn',
                    color="primary",
                    className="w-100 mb-3"
                ),
                dcc.Loading(
                    id="loading-test-result",
                    type="circle",
                    children=[html.Div(id='test-result-output', className="mt-3")]
                )
            ],
            card_id="testing-card"  # Added card_id
        ),

        # Section 4: Listening and Analysis
        create_card(
            "4. Listen & Analyze",
            [
                html.Div(id='gender-prediction-output', className="text-center mt-2 mb-4 h4", 
                        style={'color': '#0d6efd'}),
                
                # Controls for graph display
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Signals to Display:", className="fw-bold d-block mb-2"),
                        dbc.Checklist(
                            id='signal-selector-checklist',
                            options=[
                                {'label': 'Original', 'value': 'original'},
                                {'label': 'Downsampled', 'value': 'downsampled'},
                                {'label': 'Reconstructed', 'value': 'reconstructed'},
                            ],
                            value=['original', 'downsampled', 'reconstructed'],
                            inline=True,
                            className="mb-2 mb-md-0"
                        ),
                    ], width=12, md=6),
                    dbc.Col([
                        html.Label("Graph View Mode:", className="fw-bold d-block mb-2"),
                        dbc.RadioItems(
                            id='graph-view-selector',
                            options=[
                                {'label': 'Overlap', 'value': 'overlap'},
                                {'label': 'Separate', 'value': 'separate'},
                            ],
                            value='overlap',
                            inline=True
                        ),
                    ], width=12, md=6),
                ], className="mb-4 text-center"),

                # Main output area
                dcc.Loading(
                    id="loading-main-output",
                    type="default",
                    children=html.Div([
                        html.Div(id='audio-players', className="mb-4"),
                        html.Div(id='spectrum-plot-container', className="mt-4")
                    ])
                )
            ],
            card_id="analysis-card"  # Added card_id
        )
    ])