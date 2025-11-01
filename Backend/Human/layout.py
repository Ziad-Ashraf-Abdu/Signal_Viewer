# layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html

def create_layout():
    return dbc.Container([
        # Header
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H1("🎧 Interactive Audio Sampling & Analysis", className="text-center mb-3"),
                    html.P("Upload an audio file to downsample, reconstruct, and test the effects on an AI model.",
                           className="text-center text-muted"),
                ], className="bg-light p-4 rounded mb-4 shadow-sm")
            )
        ),

        # Hidden stores
        dcc.Store(id='original-audio-data'),
        dcc.Store(id='downsampled-audio-data'),
        dcc.Store(id='reconstructed-audio-data'),

        # Upload Card
        dbc.Card(
            dbc.CardBody([
                html.H3("1. Load Audio", className="card-title mb-3"),
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div(['Drag and Drop or ', html.A('Select Audio File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center'
                    },
                    className="mb-3"
                ),
                html.Div(id='upload-status', className="mt-3")
            ]), className="mb-4 shadow-sm"
        ),

        # Processing Controls
        dbc.Card(
            dbc.CardBody([
                html.H3("2. Process Audio", className="card-title mb-3"),
                html.Div([
                    html.H4("Downsample Frequency", className="text-danger mb-2"),
                    dcc.Slider(
                        id='target-sr-slider',
                        min=500,
                        max=16000,
                        step=500,
                        value=8000,
                        marks={500: '0.5k', **{i: f'{i//1000}k' for i in range(2000, 17000, 2000)}},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-3"
                    ),
                ], className="mb-4"),
                html.Div([
                    dbc.Button('Apply AI Reconstruction', id='model-reconstruct-btn', n_clicks=0, color="success", className="me-3"),
                    html.Span(id='model-status', style={'fontWeight': 'bold'})
                ], className="text-center mt-3"),
            ]), className="mb-4 shadow-sm"
        ),

        # Model Test Card
        dbc.Card(
            dbc.CardBody([
                html.H3("3. Test Audio on Model", className="card-title mb-3"),
                html.P("Select an audio signal and test its effect on the model's prediction.", className="text-center mb-3"),
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
                dbc.Button('Run Gender Detection Test', id='run-test-btn', n_clicks=0, color="primary", className="w-100 mb-3"),
                dcc.Loading(
                    id="loading-test-result",
                    type="circle",
                    children=[html.Div(id='test-result-output', className="mt-3")]
                )
            ]), className="mb-4 shadow-sm"
        ),

        # Listen & Analyze Card
        dbc.Card(
            dbc.CardBody([
                html.H3("4. Listen & Analyze", className="card-title mb-3"),
                html.Div(id='gender-prediction-output', className="text-center mt-2 mb-4 h4", style={'color': '#0d6efd'}),
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
                dcc.Loading(
                    id="loading-main-output",
                    type="default",
                    children=html.Div([
                        html.Div(id='audio-players', className="mb-4"),
                        html.Div(id='spectrum-plot-container', className="mt-4")
                    ])
                )
            ]), className="shadow-sm mb-4"
        ),
    ], fluid=False, className="py-4 bg-light")