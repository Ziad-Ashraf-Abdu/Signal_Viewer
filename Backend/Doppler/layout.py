# layout.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from components import header, labeled_input

def create_layout():
    return html.Div([
        header(),

        # Upload Section
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("📁 Upload Car Audio (WAV)", style={'color': '#667eea', 'marginBottom': '15px', 'fontSize': '24px', 'fontWeight': '700'}),
                            dcc.Upload(
                                id='upload-audio',
                                children=html.Div(['Drag and Drop or ', html.A('Select a WAV File')]),
                                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '2px',
                                       'borderStyle': 'dashed', 'borderRadius': '10px', 'textAlign': 'center',
                                       'marginBottom': '15px', 'backgroundColor': '#f0f4ff', 'borderColor': '#667eea'},
                                multiple=False, accept='.wav'
                            ),
                            html.Div(id='upload-status', style={'fontSize': '14px', 'color': '#6b7280', 'marginBottom': '10px'}),
                        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '12px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'marginBottom': '20px'})
                    ])
                ])
            ], fluid=True)
        ], style={'padding': '0 20px'}),

        # Aliasing Section
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("📉 Aliasing Demonstration", style={'color': '#10b981', 'marginBottom': '15px'}),
                        html.Div([
                            html.Label("Target Sampling Frequency (Hz):", style={'fontWeight': 'bold'}),
                            dcc.Slider(id='fs-slider', min=1000, max=44100, step=100, value=22050, disabled=True),
                            html.Div(id='slider-value-display', style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': '18px'})
                        ], style={'padding': '20px', 'backgroundColor': '#f0fdf4', 'borderRadius': '10px', 'marginBottom': '20px'}),
                        html.Div([html.H4("🔊 Original Audio"), html.Audio(id='audio-orig', controls=True, style={'width': '100%'})]),
                        html.Div([html.H4("🔊 Aliased Audio (Downsampled – No Anti-Aliasing)"), html.Audio(id='audio-alias', controls=True, style={'width': '100%'})])
                    ])
                ])
            ], fluid=True)
        ], style={'padding': '0 20px', 'marginBottom': '40px'}),

        # Analysis & Prediction
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("🎵 Analysis & Prediction", style={'color': '#667eea', 'marginBottom': '15px', 'fontSize': '24px', 'fontWeight': '700'}),
                            html.H4(id='detected-freq-display', children="Detected Frequency: — Hz", style={'color': '#ef4444', 'fontSize': '20px', 'fontWeight': '700', 'marginBottom': '15px'}),
                            html.H4(id='predicted-velocity-display', children="Predicted Velocity: — m/s", style={'color': '#10b981', 'fontSize': '20px', 'fontWeight': '700', 'marginBottom': '15px'}),
                            html.Div([
                                html.Button('📊 Use This Frequency', id='use-audio-freq-btn', n_clicks=0, disabled=True, style={'padding': '10px 20px', 'backgroundColor': '#667eea', 'color': 'white', 'border': 'none', 'borderRadius': '8px', 'fontSize': '14px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.3)'}),
                                html.Button('🚗 Use This Velocity', id='use-audio-velocity-btn', n_clicks=0, disabled=True, style={'padding': '10px 20px', 'backgroundColor': '#10b981', 'color': 'white', 'border': 'none', 'borderRadius': '8px', 'fontSize': '14px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.3)', 'marginLeft': '10px'})
                            ])
                        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '12px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'marginBottom': '20px'})
                    ])
                ]),
                dbc.Row([dbc.Col([dcc.Graph(id='audio-spectrum-graph', style={'height': '350px'})])])
            ], fluid=True)
        ], style={'padding': '0 20px', 'marginBottom': '30px'}),

        # Doppler Controls
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("🔊 Sound Source", style={'color': '#667eea', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': '700', 'borderBottom': '3px solid #667eea', 'paddingBottom': '10px'}),
                        dcc.RadioItems(id='source-type', options=[{'label': ' Moving', 'value': 'moving'}, {'label': ' Static', 'value': 'static'}], value='moving', inline=True, style={'marginBottom': '20px'}, labelStyle={'marginRight': '20px', 'fontSize': '15px', 'fontWeight': '500'}),
                        html.Div([
                            labeled_input("Start X (m):", 'source-x0', -200),
                            labeled_input("Start Y (m):", 'source-y0', 0),
                            html.Div(id='source-vel-inputs', children=[
                                labeled_input("Speed (m/s):", 'source-speed', 30),
                                labeled_input("Direction (°):", 'source-dir', 0)
                            ])
                        ])
                    ], style={'padding': '25px', 'backgroundColor': '#f8f9ff', 'borderRadius': '15px', 'border': '2px solid #667eea', 'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.15)'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.Div([
                        html.H3("👂 Observer", style={'color': '#f093fb', 'marginBottom': '20px', 'fontSize': '22px', 'fontWeight': '700', 'borderBottom': '3px solid #f093fb', 'paddingBottom': '10px'}),
                        dcc.RadioItems(id='observer-type', options=[{'label': ' Moving', 'value': 'moving'}, {'label': ' Static', 'value': 'static'}], value='moving', inline=True, style={'marginBottom': '20px'}, labelStyle={'marginRight': '20px', 'fontSize': '15px', 'fontWeight': '500'}),
                        html.Div([
                            labeled_input("Start X (m):", 'observer-x0', 0),
                            labeled_input("Start Y (m):", 'observer-y0', 0),
                            html.Div(id='observer-vel-inputs', children=[
                                labeled_input("Speed (m/s):", 'observer-speed', 10),
                                labeled_input("Direction (°):", 'observer-dir', 180)
                            ])
                        ])
                    ], style={'padding': '25px', 'backgroundColor': '#fff8fd', 'borderRadius': '15px', 'border': '2px solid #f093fb', 'boxShadow': '0 4px 12px rgba(240, 147, 251, 0.15)'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
            ], style={'marginBottom': '30px', 'padding': '0 20px'}),

            html.Div([html.Div([labeled_input("Emitted Frequency (Hz):", 'freq-input', 500, width=100)], style={'display': 'inline-block', 'padding': '20px 40px', 'backgroundColor': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '2px solid #e0e0e0'})], style={'textAlign': 'center', 'marginBottom': '25px'}),

            html.Div([
                html.Button('▶️ Start', id='start-btn', n_clicks=0, style={'marginRight': '12px', 'padding': '12px 28px', 'backgroundColor': '#10b981', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'fontSize': '16px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.3)'}),
                html.Button('⏸️ Pause', id='pause-btn', n_clicks=0, style={'marginRight': '12px', 'padding': '12px 28px', 'backgroundColor': '#f59e0b', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'fontSize': '16px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(245, 158, 11, 0.3)'}),
                html.Button('⏹️ Reset', id='reset-btn', n_clicks=0, style={'marginRight': '12px', 'padding': '12px 28px', 'backgroundColor': '#ef4444', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'fontSize': '16px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(239, 68, 68, 0.3)'}),
                html.Button('🔈 Mute', id='mute-btn', n_clicks=0, style={'padding': '12px 24px', 'backgroundColor': '#6b7280', 'color': 'white', 'border': 'none', 'borderRadius': '10px', 'fontSize': '16px', 'fontWeight': '600', 'cursor': 'pointer', 'boxShadow': '0 4px 12px rgba(107, 114, 128, 0.3)'}),
                html.Span(id='mute-label', children='', style={'marginLeft': '16px', 'fontWeight': '600', 'fontSize': '16px', 'color': '#374151'})
            ], style={'textAlign': 'center', 'marginBottom': '25px'}),

            html.Div(id='frequency-display', style={'fontSize': '20px', 'textAlign': 'center', 'padding': '20px', 'fontWeight': '600', 'color': '#1f2937', 'backgroundColor': 'white', 'borderRadius': '12px', 'margin': '0 20px 25px 20px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'border': '2px solid #e5e7eb'}),
            html.Div([dcc.Graph(id='simulation-graph', style={'height': '60vh'})], style={'padding': '0 20px', 'marginBottom': '30px'}),
        ]),

        # Hidden Stores
        dcc.Store(id='simulation-running', data=False),
        dcc.Store(id='time-elapsed', data=0),
        dcc.Store(id='sound-freq', data=0),
        dcc.Store(id='uploaded-audio-data', data=None),
        dcc.Store(id='aliasing-audio', data=None),
        dcc.Store(id='predicted-velocity-store', data=None),
        html.Div(id='sound-init', style={'display': 'none'}),
        html.Div(id='sound-div', style={'display': 'none'}),
        dcc.Interval(id='interval', interval=50, n_intervals=0, disabled=True)
    ], style={'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', 'backgroundColor': '#f9fafb', 'minHeight': '100vh', 'paddingBottom': '40px'})