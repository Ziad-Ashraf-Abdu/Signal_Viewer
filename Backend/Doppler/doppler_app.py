import os
import h5py
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import librosa

# ---------- CONFIG ----------
SPEED_OF_SOUND = 343  # m/s
AUDIO_FILE = "CitroenC4Picasso_51.wav"
H5_FILE = "speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"
# ----------------------------

# Derived names
vehiclename_full = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
vehiclename_prefix = vehiclename_full.split('_')[0] if '_' in vehiclename_full else vehiclename_full

# HDF5 stats placeholders
h5_loaded = False
speed_mode = None
speed_mean = None
speed_count = 0
speed_est_array = None
h5_error_msg = None
h5_used_key = None

# Try to read HDF5 dataset using prefix-based matching
if os.path.exists(H5_FILE):
    try:
        with h5py.File(H5_FILE, 'r') as hf:
            available_keys = list(hf.keys())
            candidates = [k for k in available_keys if k.startswith(vehiclename_prefix)]
            pref1 = f"{vehiclename_prefix}_speeds_est_all"
            pref2 = f"{vehiclename_prefix}_speeds_gt"

            chosen_key = None
            if pref1 in hf:
                chosen_key = pref1
            elif pref2 in hf:
                chosen_key = pref2
            elif candidates:
                chosen_key = candidates[0]

            if chosen_key:
                h5_used_key = chosen_key
                speed_est_array = np.array(hf[chosen_key])
                speed_count = int(speed_est_array.size)
                print(
                    f"Using HDF5 key '{chosen_key}' for prefix '{vehiclename_prefix}' (derived from '{vehiclename_full}').")
                print(f"Found {speed_count} speed estimates in '{H5_FILE}' under key '{chosen_key}'.")

                for s in speed_est_array:
                    print(s)

                valid = speed_est_array[~np.isnan(speed_est_array)]
                if valid.size > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    mode_val = vals[np.argmax(counts)]
                    mean_val = float(np.mean(valid))
                    speed_mode = float(mode_val)
                    speed_mean = mean_val
                    h5_loaded = True
                    print(f"Mode (most frequent): {speed_mode}")
                    print(f"Mean speed: {speed_mean:.6f}")
                else:
                    print("No valid (non-NaN) speed estimates to compute statistics.")
            else:
                h5_error_msg = (f"No HDF5 keys starting with '{vehiclename_prefix}' were found in '{H5_FILE}'. "
                                f"Available keys: {available_keys}")
                print(h5_error_msg)
    except Exception as e:
        h5_error_msg = f"Error reading H5 file '{H5_FILE}': {e}"
        print(h5_error_msg)
else:
    h5_error_msg = f"H5 file '{H5_FILE}' not found."
    print(h5_error_msg)

# ---------- AUDIO ANALYSIS ----------
dominant_freq = 500
audio_fig = None
audio_loaded = False

if os.path.exists(AUDIO_FILE):
    try:
        print(f"Loading audio: {AUDIO_FILE}")
        y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
        print(f"Loaded: {len(y)} samples at {sr} Hz")

        N = len(y)
        yf = fft(y)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])

        min_freq = 20
        min_idx = np.argmax(xf >= min_freq)
        peak_idx, _ = find_peaks(magnitude[min_idx:], height=np.max(magnitude) * 0.1, distance=100)

        if len(peak_idx) > 0:
            dominant_freq = float(xf[min_idx + peak_idx[0]])

        audio_fig = go.Figure()
        audio_fig.add_trace(go.Scatter(
            x=xf, y=magnitude,
            mode='lines',
            name='Spectrum',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        if dominant_freq > 0:
            audio_fig.add_vline(
                x=dominant_freq,
                line=dict(color='#f093fb', dash='dash', width=3),
                annotation_text=f"Dominant: {dominant_freq:.1f} Hz",
                annotation_position="top right",
                annotation=dict(font=dict(size=14, color='#f093fb', family='Arial'))
            )
        audio_fig.update_layout(
            title=dict(text="Car Sound Frequency Spectrum", font=dict(size=18, color='#2d3748', family='Arial')),
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis_range=[0, 2000],
            plot_bgcolor='rgba(247, 250, 252, 0.5)',
            paper_bgcolor='white',
            font=dict(family='Arial', color='#4a5568'),
            margin=dict(l=50, r=30, t=60, b=50)
        )
        audio_loaded = True
        print(f"Audio analysis complete. Dominant frequency: {dominant_freq:.1f} Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        dominant_freq = 500
else:
    print(f"Audio file '{AUDIO_FILE}' not found. Using default frequency.")

# ---------- DASH APP ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True


def labeled_input(label, id, value, width=80):
    return html.Div([
        html.Label(label, style={
            'display': 'inline-block',
            'width': '140px',
            'fontWeight': '600',
            'color': '#2d3748',
            'fontSize': '14px'
        }),
        dcc.Input(
            id=id,
            type='number',
            value=value,
            style={
                'width': f'{width}px',
                'padding': '10px 14px',
                'borderRadius': '8px',
                'border': '2px solid #e2e8f0',
                'fontSize': '14px',
                'transition': 'all 0.3s ease',
                'outline': 'none'
            },
            className='custom-input'
        )
    ], style={'marginBottom': '14px'})


def hdf5_summary_lines():
    cnt = speed_count if (speed_count is not None and speed_count != 0) else "N/A"
    mode_str = f"{speed_mode}" if speed_mode is not None else "N/A"
    mean_str = f"{speed_mean:.3f}" if speed_mean is not None else "N/A"
    return html.Div([
        html.P(f"AUDIO_FILE full: {AUDIO_FILE}, Speed estimates found: {cnt}",
               style={'margin': '0', 'fontSize': '14px', 'color': '#4a5568'}),
        html.Br(),
        html.P(f"Mode (most frequent): {mode_str} m/s", style={'margin': '0', 'fontSize': '14px', 'color': '#4a5568'}),
        html.Br(),
        html.P(f"Mean speed: {mean_str} m/s", style={'margin': '0', 'fontSize': '14px', 'color': '#4a5568'})
    ], style={
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'border': '1px solid #e2e8f0'
    })


# Add custom CSS to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-input:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            }
            .custom-button {
                transition: all 0.3s ease;
            }
            .custom-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔊 Advanced Doppler Effect Simulator", style={
            'textAlign': 'center',
            'color': 'white',
            'padding': '32px 24px 16px 24px',
            'margin': '0',
            'fontWeight': '700',
            'fontSize': '36px',
            'letterSpacing': '-0.5px'
        }),
        html.P("Interactive Physics Simulation with Real-Time Audio & Frequency Analysis", style={
            'textAlign': 'center',
            'color': 'rgba(255, 255, 255, 0.9)',
            'fontSize': '16px',
            'margin': '0',
            'paddingBottom': '32px'
        })
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'marginBottom': '30px',
        'boxShadow': '0 10px 25px rgba(102, 126, 234, 0.3)'
    }),

    # HDF5 Summary
    dbc.Container([
        dbc.Row([
            dbc.Col([
                hdf5_summary_lines()
            ])
        ])
    ], fluid=True, style={'marginBottom': '30px'}),

    # Audio Analysis Section
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("🎵 Audio Frequency Analysis", style={
                            'color': '#2d3748',
                            'fontSize': '22px',
                            'fontWeight': '700',
                            'marginBottom': '12px'
                        }),
                        html.P(f"Analyzed car sound file: {AUDIO_FILE}", style={
                            'color': '#718096',
                            'fontSize': '14px',
                            'marginBottom': '8px'
                        }),
                        html.H4(f"Detected Frequency: {dominant_freq:.1f} Hz", style={
                            'color': '#667eea',
                            'fontSize': '28px',
                            'fontWeight': '700',
                            'marginBottom': '16px'
                        }),
                        html.Button('📊 Use This Frequency',
                                    id='use-audio-freq-btn',
                                    n_clicks=0,
                                    className='custom-button',
                                    style={
                                        'backgroundColor': '#667eea',
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '12px 24px',
                                        'borderRadius': '8px',
                                        'fontSize': '15px',
                                        'fontWeight': '600',
                                        'cursor': 'pointer'
                                    }
                                    )
                    ], style={
                        'padding': '24px',
                        'backgroundColor': 'white',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        'border': '1px solid #e2e8f0',
                        'marginBottom': '20px'
                    })
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(figure=audio_fig, style={'height': '400px'})
                    ], style={
                        'backgroundColor': 'white',
                        'borderRadius': '12px',
                        'padding': '16px',
                        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        'border': '1px solid #e2e8f0'
                    })
                ])
            ])
        ], fluid=True)
    ], style={'marginBottom': '30px'}) if audio_loaded else html.Div(),

    # Controls
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("🔊 Sound Source", style={
                        'color': '#2d3748',
                        'fontSize': '20px',
                        'fontWeight': '700',
                        'marginBottom': '16px'
                    }),
                    dcc.RadioItems(
                        id='source-type',
                        options=[
                            {'label': ' Moving', 'value': 'moving'},
                            {'label': ' Static', 'value': 'static'}
                        ],
                        value='moving',
                        inline=True,
                        style={'marginBottom': '16px', 'fontSize': '14px'},
                        labelStyle={'marginRight': '20px', 'color': '#4a5568'}
                    ),
                    labeled_input("Start X (m):", 'source-x0', -200),
                    labeled_input("Start Y (m):", 'source-y0', 0),
                    html.Div(id='source-vel-inputs', children=[
                        labeled_input("Speed (m/s):", 'source-speed', 30),
                        labeled_input("Direction (°):", 'source-dir', 0)
                    ])
                ], style={
                    'padding': '24px',
                    'backgroundColor': 'white',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    'border': '1px solid #e2e8f0',
                    'height': '100%'
                })
            ], md=6),

            dbc.Col([
                html.Div([
                    html.H3("👂 Observer", style={
                        'color': '#2d3748',
                        'fontSize': '20px',
                        'fontWeight': '700',
                        'marginBottom': '16px'
                    }),
                    dcc.RadioItems(
                        id='observer-type',
                        options=[
                            {'label': ' Moving', 'value': 'moving'},
                            {'label': ' Static', 'value': 'static'}
                        ],
                        value='moving',
                        inline=True,
                        style={'marginBottom': '16px', 'fontSize': '14px'},
                        labelStyle={'marginRight': '20px', 'color': '#4a5568'}
                    ),
                    labeled_input("Start X (m):", 'observer-x0', 0),
                    labeled_input("Start Y (m):", 'observer-y0', 0),
                    html.Div(id='observer-vel-inputs', children=[
                        labeled_input("Speed (m/s):", 'observer-speed', 10),
                        labeled_input("Direction (°):", 'observer-dir', 180)
                    ])
                ], style={
                    'padding': '24px',
                    'backgroundColor': 'white',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    'border': '1px solid #e2e8f0',
                    'height': '100%'
                })
            ], md=6)
        ], style={'marginBottom': '30px'})
    ], fluid=True),

    # Frequency Input
    html.Div([
        html.Div([
            labeled_input("Emitted Frequency (Hz):", 'freq-input', int(dominant_freq), width=120)
        ], style={
            'display': 'inline-block',
            'padding': '20px 40px',
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            'border': '1px solid #e2e8f0'
        })
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Control Buttons
    html.Div([
        html.Button('▶️ Start',
                    id='start-btn',
                    n_clicks=0,
                    className='custom-button',
                    style={
                        'marginRight': '12px',
                        'backgroundColor': '#48bb78',
                        'color': 'white',
                        'border': 'none',
                        'padding': '14px 28px',
                        'borderRadius': '10px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 6px rgba(72, 187, 120, 0.3)'
                    }
                    ),
        html.Button('⏸️ Pause',
                    id='pause-btn',
                    n_clicks=0,
                    className='custom-button',
                    style={
                        'marginRight': '12px',
                        'backgroundColor': '#ed8936',
                        'color': 'white',
                        'border': 'none',
                        'padding': '14px 28px',
                        'borderRadius': '10px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 6px rgba(237, 137, 54, 0.3)'
                    }
                    ),
        html.Button('⏹️ Reset',
                    id='reset-btn',
                    n_clicks=0,
                    className='custom-button',
                    style={
                        'marginRight': '12px',
                        'backgroundColor': '#f56565',
                        'color': 'white',
                        'border': 'none',
                        'padding': '14px 28px',
                        'borderRadius': '10px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 6px rgba(245, 101, 101, 0.3)'
                    }
                    ),
        html.Button('🔈 Mute',
                    id='mute-btn',
                    n_clicks=0,
                    className='custom-button',
                    style={
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'border': 'none',
                        'padding': '14px 28px',
                        'borderRadius': '10px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'boxShadow': '0 4px 6px rgba(102, 126, 234, 0.3)'
                    }
                    ),
        html.Span(id='mute-label', style={
            'marginLeft': '16px',
            'fontWeight': '700',
            'fontSize': '16px',
            'color': '#667eea'
        })
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Frequency Display
    html.Div(id='frequency-display', style={
        'textAlign': 'center',
        'padding': '20px',
        'fontWeight': '700',
        'fontSize': '18px',
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'margin': '0 20px 30px 20px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'border': '1px solid #e2e8f0',
        'color': '#2d3748'
    }),

    # Simulation Graph
    html.Div([
        dcc.Graph(id='simulation-graph', style={'height': '60vh'})
    ], style={
        'margin': '0 20px',
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'padding': '16px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'border': '1px solid #e2e8f0'
    }),

    # Hidden stores
    dcc.Store(id='simulation-running', data=False),
    dcc.Store(id='time-elapsed', data=0),
    dcc.Store(id='sound-freq', data=0),
    html.Div(id='sound-init', style={'display': 'none'}),
    html.Div(id='sound-div', style={'display': 'none'}),
    dcc.Interval(id='interval', interval=100, n_intervals=0, disabled=True),
    dcc.Store(id='h5-stats', data={
        'loaded': h5_loaded,
        'mode': speed_mode,
        'mean': speed_mean,
        'count': speed_count,
        'used_key': h5_used_key,
        'error': h5_error_msg
    })
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#f7fafc',
    'minHeight': '100vh',
    'paddingBottom': '60px'
})


# ---------- CALLBACKS ----------

@app.callback(
    Output('freq-input', 'value'),
    Input('use-audio-freq-btn', 'n_clicks'),
    prevent_initial_call=True
)
def use_audio_freq(n_clicks):
    return int(dominant_freq)


@app.callback(
    Output('source-vel-inputs', 'style'),
    Input('source-type', 'value')
)
def toggle_source_vel(source_type):
    return {} if source_type == 'moving' else {'display': 'none'}


@app.callback(
    Output('observer-vel-inputs', 'style'),
    Input('observer-type', 'value')
)
def toggle_observer_vel(observer_type):
    return {} if observer_type == 'moving' else {'display': 'none'}


@app.callback(
    [Output('interval', 'disabled'),
     Output('simulation-running', 'data'),
     Output('time-elapsed', 'data')],
    [Input('start-btn', 'n_clicks'),
     Input('pause-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('interval', 'n_intervals')],
    [State('simulation-running', 'data'),
     State('time-elapsed', 'data')]
)
def control_simulation(start_clicks, pause_clicks, reset_clicks, n_intervals, is_running, time_elapsed):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, 0

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-btn':
        return True, False, 0

    if trigger_id == 'start-btn':
        return False, True, time_elapsed

    if trigger_id == 'pause-btn':
        return True, False, time_elapsed

    if trigger_id == 'interval' and is_running:
        return False, True, round(time_elapsed + 0.1, 4)

    return True, False, time_elapsed


@app.callback(
    [Output('simulation-graph', 'figure'),
     Output('frequency-display', 'children'),
     Output('sound-freq', 'data')],
    [Input('simulation-running', 'data'),
     Input('time-elapsed', 'data'),
     Input('freq-input', 'value'),
     Input('source-type', 'value'),
     Input('observer-type', 'value'),
     Input('source-x0', 'value'),
     Input('source-y0', 'value'),
     Input('observer-x0', 'value'),
     Input('observer-y0', 'value'),
     Input('source-speed', 'value'),
     Input('source-dir', 'value'),
     Input('observer-speed', 'value'),
     Input('observer-dir', 'value')]
)
def update_display(is_running, t, f_emit,
                   src_type, obs_type,
                   src_x0, src_y0, obs_x0, obs_y0,
                   src_speed, src_dir, obs_speed, obs_dir):
    src_x0 = src_x0 if src_x0 is not None else -200
    src_y0 = src_y0 if src_y0 is not None else 0
    obs_x0 = obs_x0 if obs_x0 is not None else 0
    obs_y0 = obs_y0 if obs_y0 is not None else 0
    f_emit = f_emit if f_emit is not None else int(dominant_freq)
    src_speed = src_speed if src_speed is not None else (30 if src_type == 'moving' else 0)
    src_dir = src_dir if src_dir is not None else 0
    obs_speed = obs_speed if obs_speed is not None else (10 if obs_type == 'moving' else 0)
    obs_dir = obs_dir if obs_dir is not None else 0

    if src_type == 'moving':
        theta_s = np.radians(src_dir)
        src_x = src_x0 + src_speed * np.cos(theta_s) * t
        src_y = src_y0 + src_speed * np.sin(theta_s) * t
    else:
        src_x, src_y = src_x0, src_y0

    if obs_type == 'moving':
        theta_o = np.radians(obs_dir)
        obs_x = obs_x0 + obs_speed * np.cos(theta_o) * t
        obs_y = obs_y0 + obs_speed * np.sin(theta_o) * t
    else:
        obs_x, obs_y = obs_x0, obs_y0

    dx = obs_x - src_x
    dy = obs_y - src_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    v_src_rad = 0.0
    v_obs_rad = 0.0

    if distance > 1e-6:
        ur_x = dx / distance
        ur_y = dy / distance

        if src_type == 'moving':
            v_sx = src_speed * np.cos(np.radians(src_dir))
            v_sy = src_speed * np.sin(np.radians(src_dir))
            v_src_rad = v_sx * ur_x + v_sy * ur_y

        if obs_type == 'moving':
            v_ox = obs_speed * np.cos(np.radians(obs_dir))
            v_oy = obs_speed * np.sin(np.radians(obs_dir))
            v_obs_rad = -(v_ox * ur_x + v_oy * ur_y)

    denominator = SPEED_OF_SOUND - v_src_rad
    if abs(denominator) < 1e-6:
        f_perceived = f_emit
    else:
        f_perceived = f_emit * (SPEED_OF_SOUND + v_obs_rad) / denominator

    f_perceived = max(20, min(20000, f_perceived))
    f_perceived = round(f_perceived, 1)

    if not is_running:
        sound_freq = 0
        status = "Running" if is_running else ("Paused" if t > 0 else "Stopped")
        freq_text = f"{status}  |  Emitted: {f_emit} Hz  →  Perceived: {f_perceived} Hz  |  Time: {t:.1f} s"
    else:
        sound_freq = f_perceived
        freq_text = f"🔊 Emitted: {f_emit} Hz  →  Perceived: {f_perceived} Hz  |  Time: {t:.1f} s"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[obs_x], y=[obs_y],
        mode='markers+text',
        marker=dict(size=20, color='#f093fb', line=dict(color='white', width=3)),
        text=['👂 Observer'],
        textposition='top center',
        textfont=dict(size=14, color='#2d3748', family='Arial')
    ))

    fig.add_trace(go.Scatter(
        x=[src_x], y=[src_y],
        mode='markers',
        marker=dict(size=25, color='rgba(0,0,0,0)'),
        hovertext=f'Car (Source)\nPosition: ({src_x:.1f}, {src_y:.1f})',
        hoverinfo='text',
        showlegend=False
    ))

    car_svg = '''
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <g transform="translate(50,50) rotate({angle}) translate(-50,-50)">
            <rect x="20" y="40" width="60" height="25" rx="5" fill="#667eea" stroke="white" stroke-width="2"/>
            <path d="M 30 40 L 35 25 L 65 25 L 70 40 Z" fill="#667eea" stroke="white" stroke-width="2"/>
            <rect x="36" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
            <rect x="52" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
            <circle cx="32" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
            <circle cx="68" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
        </g>
    </svg>
    '''.format(angle=src_dir if src_type == 'moving' else 0)

    fig.add_layout_image(
        dict(
            source=f"data:image/svg+xml;charset=utf-8,{car_svg.replace('#', '%23').replace('<', '%3C').replace('>', '%3E').replace(' ', '%20').replace('\"', '%22')}",
            x=src_x,
            y=src_y,
            xref="x",
            yref="y",
            sizex=40,
            sizey=40,
            xanchor="center",
            yanchor="middle",
            layer="above"
        )
    )

    try:
        wave_count = 6
        for n in range(wave_count):
            t_emit = max(0, t - n * (1.0 / max(1, f_emit)))
            radius = SPEED_OF_SOUND * (t - t_emit)
            if radius > 0:
                opacity = max(0.05, 1 - (n / wave_count) * 0.7)
                fig.add_shape(type="circle",
                              xref="x", yref="y",
                              x0=src_x - radius, y0=src_y - radius,
                              x1=src_x + radius, y1=src_y + radius,
                              line=dict(color=f"rgba(102,126,234,{opacity})", dash="dot", width=2))
    except Exception:
        pass

    fig.update_layout(
        xaxis=dict(
            title="X Position (meters)",
            range=[-300, 300],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=2
        ),
        yaxis=dict(
            title="Y Position (meters)",
            range=[-150, 150],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=2
        ),
        showlegend=False,
        title=dict(
            text="Real-Time Doppler Effect Visualization",
            font=dict(size=20, color='#2d3748', family='Arial')
        ),
        plot_bgcolor='rgba(247, 250, 252, 0.5)',
        paper_bgcolor='white',
        font=dict(family='Arial', color='#4a5568'),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig, freq_text, sound_freq


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks < 1) return '';
        if (typeof window._doppAudioInit === 'undefined') {
            window._doppAudioInit = true;
        }
        try {
            if (!window.audioCtx) {
                window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                window.gainNode = window.audioCtx.createGain();
                window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                window.gainNode.connect(window.audioCtx.destination);
                window.oscillator = null;
                window.muted = false;
            }
            if (window.audioCtx.state === 'suspended') {
                window.audioCtx.resume();
            }
        } catch (e) {
            console.log('Audio init error:', e);
        }
        return '';
    }
    """,
    Output('sound-init', 'children'),
    Input('start-btn', 'n_clicks')
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (typeof window._doppAudioInit === 'undefined') {
            try {
                window._doppAudioInit = true;
                window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                window.gainNode = window.audioCtx.createGain();
                window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                window.gainNode.connect(window.audioCtx.destination);
                window.oscillator = null;
                window.muted = false;
                if (window.audioCtx.state === 'suspended') window.audioCtx.resume();
            } catch(e) {}
        }
        if (!n_clicks) return '';
        window.muted = (n_clicks % 2) === 1;
        try {
            if (window.gainNode && window.audioCtx) {
                window.gainNode.gain.setValueAtTime(window.muted ? 0 : 1, window.audioCtx.currentTime);
            }
        } catch(e) {}
        return window.muted ? 'Muted' : 'Unmuted';
    }
    """,
    Output('mute-label', 'children'),
    Input('mute-btn', 'n_clicks')
)

app.clientside_callback(
    """
    function(freq) {
        try {
            if (typeof window._doppAudioInit === 'undefined') {
                window._doppAudioInit = true;
                window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                window.gainNode = window.audioCtx.createGain();
                window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                window.gainNode.connect(window.audioCtx.destination);
                window.oscillator = null;
                window.muted = false;
            }
            if (window.audioCtx && window.audioCtx.state === 'suspended') {
                window.audioCtx.resume();
            }
            if (freq && freq > 0 && !window.muted) {
                if (!window.oscillator) {
                    window.oscillator = window.audioCtx.createOscillator();
                    window.oscillator.type = 'sine';
                    window.oscillator.connect(window.gainNode);
                    window.oscillator.start();
                }
                try {
                    window.oscillator.frequency.linearRampToValueAtTime(freq, window.audioCtx.currentTime + 0.05);
                } catch(e) {
                    window.oscillator.frequency.setValueAtTime(freq, window.audioCtx.currentTime);
                }
            } else {
                if (window.oscillator) {
                    try {
                        window.oscillator.stop();
                    } catch(e) {}
                    try {
                        window.oscillator.disconnect();
                    } catch(e) {}
                    window.oscillator = null;
                }
            }
        } catch (e) {
            console.log('Sound clientside error:', e);
        }
        return '';
    }
    """,
    Output('sound-div', 'children'),
    Input('sound-freq', 'data')
)

if __name__ == '__main__':
    app.run(debug=True)