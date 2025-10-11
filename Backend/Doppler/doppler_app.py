import os
import h5py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import librosa

# ---------- CONFIG ----------
SPEED_OF_SOUND = 343  # m/s
AUDIO_FILE = "CitroenC4Picasso_51.wav"
H5_FILE = "speed_estimations_NN_1000-200-50-10-1_reg1e-3_lossMSE.h5"

# ---------- UI STYLE CONSTANTS ----------
PRIMARY_COLOR = "#4A90E2"
ACCENT_COLOR = "#50E3C2"
BACKGROUND_COLOR = "#F7F9FC"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
SUBTLE_TEXT_COLOR = "#666666"
BORDER_COLOR = "#EAEAEA"
FONT_FAMILY = "Inter, system-ui, sans-serif"

CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR,
    "borderRadius": "12px",
    "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)",
    "border": f"1px solid {BORDER_COLOR}",
}

BUTTON_STYLE = {
    "backgroundColor": PRIMARY_COLOR,
    "color": "white",
    "border": "none",
    "borderRadius": "8px",
    "padding": "12px 24px",
    "fontSize": "15px",
    "fontWeight": "600",
    "cursor": "pointer",
    "transition": "all 0.2s ease",
}

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

# Try to read HDF5 dataset
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
                valid = speed_est_array[~np.isnan(speed_est_array)]
                if valid.size > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    speed_mode = float(vals[np.argmax(counts)])
                    speed_mean = float(np.mean(valid))
                    h5_loaded = True
            else:
                h5_error_msg = f"No HDF5 keys for '{vehiclename_prefix}' found. Available: {available_keys}"
    except Exception as e:
        h5_error_msg = f"Error reading H5 file: {e}"
else:
    h5_error_msg = f"H5 file '{H5_FILE}' not found."

# ---------- AUDIO ANALYSIS ----------
dominant_freq = 500
audio_fig = None
audio_loaded = False

if os.path.exists(AUDIO_FILE):
    try:
        y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
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
        audio_fig.add_trace(
            go.Scatter(x=xf, y=magnitude, mode='lines', name='Spectrum', line=dict(color=PRIMARY_COLOR, width=2)))
        audio_fig.add_vline(x=dominant_freq, line=dict(color=ACCENT_COLOR, dash='dash', width=2),
                            annotation_text=f"{dominant_freq:.1f} Hz")
        audio_fig.update_layout(
            title=dict(text="Audio Frequency Spectrum", font=dict(size=18, color=TEXT_COLOR)),
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
            xaxis_range=[0, 2000],
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family=FONT_FAMILY, color=SUBTLE_TEXT_COLOR),
            margin=dict(l=50, r=30, t=60, b=50)
        )
        audio_loaded = True
    except Exception as e:
        print(f"Error loading audio: {e}")

# ---------- DASH APP ----------
app = dash.Dash(__name__)


def labeled_input(label, id, value, width=80):
    return html.Div([
        html.Label(label, style={'fontWeight': '500', 'color': SUBTLE_TEXT_COLOR, "marginRight": "10px"}),
        dcc.Input(id=id, type='number', value=value, style={
            'width': f'{width}px', 'padding': '8px 12px', 'borderRadius': '6px',
            'border': f'1px solid {BORDER_COLOR}', 'fontSize': '14px', 'textAlign': 'center'
        })
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '12px'})


def hdf5_summary_card():
    if not h5_loaded:
        return html.Div(style=CARD_STYLE, children=[
            html.H3("HDF5 Summary", style={"marginTop": 0}),
            html.P(h5_error_msg or "HDF5 data could not be loaded.")
        ])

    return html.Div(style=CARD_STYLE, children=[
        html.H3("HDF5 Summary", style={"marginTop": 0, "marginBottom": "20px"}),
        html.Div([
            html.Div([html.Span("Estimates Found", style={"color": SUBTLE_TEXT_COLOR}),
                      html.Span(speed_count, style={"fontWeight": "bold", "fontSize": "18px"})],
                     style={'textAlign': 'center'}),
            html.Div([html.Span("Mode Speed (m/s)", style={"color": SUBTLE_TEXT_COLOR}),
                      html.Span(f"{speed_mode:.2f}", style={"fontWeight": "bold", "fontSize": "18px"})],
                     style={'textAlign': 'center'}),
            html.Div([html.Span("Mean Speed (m/s)", style={"color": SUBTLE_TEXT_COLOR}),
                      html.Span(f"{speed_mean:.2f}", style={"fontWeight": "bold", "fontSize": "18px"})],
                     style={'textAlign': 'center'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '10px'})
    ])


app.layout = html.Div(style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px"},
                      children=[
                          html.Div(style={"maxWidth": "1200px", "margin": "0 auto"}, children=[
                              # Header
                              html.Div([
                                  html.H1("🔊 Doppler Effect Simulator",
                                          style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                                  html.P(
                                      "An interactive physics simulation with real-time audio and frequency analysis",
                                      style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '16px'}),
                              ], style={'marginBottom': '40px'}),

                              # Top Row: HDF5 Summary and Audio Analysis
                              html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 2fr', 'gap': '24px',
                                              'marginBottom': '24px'}, children=[
                                  hdf5_summary_card(),
                                  html.Div(style=CARD_STYLE, children=[
                                      html.H3(f"Audio Analysis: {os.path.basename(AUDIO_FILE)}",
                                              style={"marginTop": 0}),
                                      html.Div([
                                          html.P(f"Detected Dominant Frequency:",
                                                 style={'color': SUBTLE_TEXT_COLOR, 'margin': '0'}),
                                          html.H4(f"{dominant_freq:.1f} Hz",
                                                  style={'color': PRIMARY_COLOR, 'fontSize': '28px',
                                                         'fontWeight': 'bold', 'margin': '5px 0 15px 0'}),
                                          html.Button('Use This Frequency', id='use-audio-freq-btn', n_clicks=0,
                                                      style=BUTTON_STYLE)
                                      ], style={'marginBottom': '20px'}),
                                      dcc.Graph(figure=audio_fig,
                                                style={'height': '250px'}) if audio_loaded else html.P(
                                          "Audio file not found or failed to load.")
                                  ])
                              ]),

                              # Middle Row: Controls
                              html.Div(style={**CARD_STYLE, 'marginBottom': '24px'}, children=[
                                  html.Div(
                                      style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '24px',
                                             'alignItems': 'start'}, children=[
                                          # Source Controls
                                          html.Div([
                                              html.H3("🔊 Sound Source", style={'marginTop': 0}),
                                              dcc.RadioItems(id='source-type',
                                                             options=[{'label': ' Moving', 'value': 'moving'},
                                                                      {'label': ' Static', 'value': 'static'}],
                                                             value='moving', inline=True,
                                                             labelStyle={'marginRight': '15px'}),
                                              html.Hr(
                                                  style={'border': f'1px solid {BORDER_COLOR}', 'margin': '15px 0'}),
                                              labeled_input("Start X (m):", 'source-x0', -200),
                                              labeled_input("Start Y (m):", 'source-y0', 0),
                                              html.Div(id='source-vel-inputs', children=[
                                                  labeled_input("Speed (m/s):", 'source-speed', 30),
                                                  labeled_input("Direction (°):", 'source-dir', 0)
                                              ])
                                          ]),
                                          # Observer Controls
                                          html.Div([
                                              html.H3("👂 Observer", style={'marginTop': 0}),
                                              dcc.RadioItems(id='observer-type',
                                                             options=[{'label': ' Moving', 'value': 'moving'},
                                                                      {'label': ' Static', 'value': 'static'}],
                                                             value='moving', inline=True,
                                                             labelStyle={'marginRight': '15px'}),
                                              html.Hr(
                                                  style={'border': f'1px solid {BORDER_COLOR}', 'margin': '15px 0'}),
                                              labeled_input("Start X (m):", 'observer-x0', 0),
                                              labeled_input("Start Y (m):", 'observer-y0', 0),
                                              html.Div(id='observer-vel-inputs', children=[
                                                  labeled_input("Speed (m/s):", 'observer-speed', 10),
                                                  labeled_input("Direction (°):", 'observer-dir', 180)
                                              ])
                                          ]),
                                          # Frequency & Sim Controls
                                          html.Div([
                                              html.H3("⚙️ Simulation", style={'marginTop': 0}),
                                              labeled_input("Emitted Freq (Hz):", 'freq-input', int(dominant_freq),
                                                            width=100),
                                              html.Hr(
                                                  style={'border': f'1px solid {BORDER_COLOR}', 'margin': '15px 0'}),
                                              html.Div([
                                                  html.Button('▶️ Start', id='start-btn', n_clicks=0,
                                                              style={**BUTTON_STYLE, 'backgroundColor': '#4CAF50'}),
                                                  html.Button('⏸️ Pause', id='pause-btn', n_clicks=0,
                                                              style={**BUTTON_STYLE, 'backgroundColor': '#FFA500'}),
                                                  html.Button('⏹️ Reset', id='reset-btn', n_clicks=0,
                                                              style={**BUTTON_STYLE, 'backgroundColor': '#F44336'}),
                                                  html.Button('🔈 Mute', id='mute-btn', n_clicks=0, style=BUTTON_STYLE),
                                              ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                                                        'gap': '10px'}),
                                              html.Span(id='mute-label',
                                                        style={'textAlign': 'center', 'display': 'block',
                                                               'marginTop': '10px', 'color': PRIMARY_COLOR,
                                                               'fontWeight': 'bold'})
                                          ])
                                      ])
                              ]),

                              # Bottom Row: Simulation
                              html.Div(style=CARD_STYLE, children=[
                                  html.Div(id='frequency-display',
                                           style={'textAlign': 'center', 'padding': '15px', 'fontWeight': '500',
                                                  'fontSize': '18px', 'color': TEXT_COLOR,
                                                  'backgroundColor': BACKGROUND_COLOR, 'borderRadius': '8px',
                                                  'marginBottom': '20px'}),
                                  dcc.Graph(id='simulation-graph', style={'height': '500px'})
                              ]),

                              # Hidden stores and intervals
                              dcc.Store(id='simulation-running', data=False),
                              dcc.Store(id='time-elapsed', data=0),
                              dcc.Store(id='sound-freq', data=0),
                              html.Div(id='sound-init', style={'display': 'none'}),
                              html.Div(id='sound-div', style={'display': 'none'}),
                              dcc.Interval(id='interval', interval=100, n_intervals=0, disabled=True),
                          ])
                      ])


# ---------- CALLBACKS ----------
@app.callback(Output('freq-input', 'value'), Input('use-audio-freq-btn', 'n_clicks'), prevent_initial_call=True)
def use_audio_freq(n_clicks):
    return int(dominant_freq)


@app.callback(Output('source-vel-inputs', 'style'), Input('source-type', 'value'))
def toggle_source_vel(source_type):
    return {} if source_type == 'moving' else {'display': 'none'}


@app.callback(Output('observer-vel-inputs', 'style'), Input('observer-type', 'value'))
def toggle_observer_vel(observer_type):
    return {} if observer_type == 'moving' else {'display': 'none'}


@app.callback(
    [Output('interval', 'disabled'), Output('simulation-running', 'data'), Output('time-elapsed', 'data')],
    [Input('start-btn', 'n_clicks'), Input('pause-btn', 'n_clicks'), Input('reset-btn', 'n_clicks'),
     Input('interval', 'n_intervals')],
    [State('simulation-running', 'data'), State('time-elapsed', 'data')]
)
def control_simulation(start_clicks, pause_clicks, reset_clicks, n_intervals, is_running, time_elapsed):
    ctx = dash.callback_context
    if not ctx.triggered: return True, False, 0
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'reset-btn': return True, False, 0
    if trigger_id == 'start-btn': return False, True, time_elapsed
    if trigger_id == 'pause-btn': return True, False, time_elapsed
    if trigger_id == 'interval' and is_running: return False, True, round(time_elapsed + 0.1, 4)
    return True, False, time_elapsed


@app.callback(
    [Output('simulation-graph', 'figure'), Output('frequency-display', 'children'), Output('sound-freq', 'data')],
    [Input('time-elapsed', 'data'), Input('freq-input', 'value'), Input('source-type', 'value'),
     Input('observer-type', 'value'),
     Input('source-x0', 'value'), Input('source-y0', 'value'), Input('observer-x0', 'value'),
     Input('observer-y0', 'value'),
     Input('source-speed', 'value'), Input('source-dir', 'value'), Input('observer-speed', 'value'),
     Input('observer-dir', 'value')]
)
def update_display(t, f_emit, src_type, obs_type, src_x0, src_y0, obs_x0, obs_y0, src_speed, src_dir, obs_speed,
                   obs_dir):
    is_running = dash.callback_context.triggered_id != 'time-elapsed' or t > 0

    src_x0, src_y0 = (src_x0 or -200), (src_y0 or 0)
    obs_x0, obs_y0 = (obs_x0 or 0), (obs_y0 or 0)
    f_emit = f_emit or int(dominant_freq)
    src_speed = src_speed if (src_type == 'moving' and src_speed is not None) else 0
    obs_speed = obs_speed if (obs_type == 'moving' and obs_speed is not None) else 0
    src_dir, obs_dir = src_dir or 0, obs_dir or 0

    src_x = src_x0 + src_speed * np.cos(np.radians(src_dir)) * t
    src_y = src_y0 + src_speed * np.sin(np.radians(src_dir)) * t
    obs_x = obs_x0 + obs_speed * np.cos(np.radians(obs_dir)) * t
    obs_y = obs_y0 + obs_speed * np.sin(np.radians(obs_dir)) * t

    dx, dy = obs_x - src_x, obs_y - src_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    v_src_rad, v_obs_rad = 0.0, 0.0
    if distance > 1e-6:
        ur_x, ur_y = dx / distance, dy / distance
        v_sx = src_speed * np.cos(np.radians(src_dir))
        v_sy = src_speed * np.sin(np.radians(src_dir))
        v_src_rad = v_sx * ur_x + v_sy * ur_y
        v_ox = obs_speed * np.cos(np.radians(obs_dir))
        v_oy = obs_speed * np.sin(np.radians(obs_dir))
        v_obs_rad = -(v_ox * ur_x + v_oy * ur_y)

    denominator = SPEED_OF_SOUND - v_src_rad
    f_perceived = f_emit * (SPEED_OF_SOUND + v_obs_rad) / denominator if abs(denominator) > 1e-6 else f_emit
    f_perceived = round(max(20, min(20000, f_perceived)), 1)

    status = "Running" if is_running and t > 0 else ("Paused" if t > 0 else "Stopped")
    freq_text = f"Status: {status} | Emitted: {f_emit} Hz → Perceived: {f_perceived} Hz | Time: {t:.1f} s"
    sound_freq = f_perceived if is_running and t > 0 else 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[obs_x], y=[obs_y], mode='markers+text', marker=dict(size=20, color=ACCENT_COLOR), text=['👂'],
                   textposition='middle center', textfont=dict(size=24)))
    fig.add_trace(
        go.Scatter(x=[src_x], y=[src_y], mode='markers+text', marker=dict(size=25, color=PRIMARY_COLOR), text=['🚗'],
                   textposition='middle center', textfont=dict(size=24)))

    for n in range(6):
        t_emit = max(0, t - n * (0.8 / max(1, f_emit / 100)))
        radius = SPEED_OF_SOUND * (t - t_emit)
        if radius > 0:
            opacity = max(0.05, 1 - (n / 6) * 0.7)
            fig.add_shape(type="circle", xref="x", yref="y", x0=src_x - radius, y0=src_y - radius, x1=src_x + radius,
                          y1=src_y + radius, line=dict(color=f"rgba(74,144,226,{opacity})", dash="dot", width=2))

    fig.update_layout(
        xaxis=dict(title="X Position (m)", range=[-300, 300]), yaxis=dict(title="Y Position (m)", range=[-150, 150]),
        showlegend=False,
        title=dict(text="Real-Time Doppler Effect Visualization", font=dict(size=20, color=TEXT_COLOR)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=SUBTLE_TEXT_COLOR), margin=dict(l=60, r=40, t=80, b=60)
    )
    return fig, freq_text, sound_freq


app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks || n_clicks < 1) return '';
        try {
            if (!window.audioCtx) {
                window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                window.gainNode = window.audioCtx.createGain();
                window.gainNode.gain.setValueAtTime(1, window.audioCtx.currentTime);
                window.gainNode.connect(window.audioCtx.destination);
                window.oscillator = null; window.muted = false;
            }
            if (window.audioCtx.state === 'suspended') window.audioCtx.resume();
        } catch (e) { console.log('Audio init error:', e); }
        return '';
    }
    """,
    Output('sound-init', 'children'), Input('start-btn', 'n_clicks')
)
app.clientside_callback(
    """
    function(n_clicks) {
        if (typeof window.audioCtx === 'undefined') return '';
        window.muted = (n_clicks % 2) === 1;
        try {
            window.gainNode.gain.setValueAtTime(window.muted ? 0 : 1, window.audioCtx.currentTime);
        } catch(e) {}
        return window.muted ? 'Muted' : 'Unmuted';
    }
    """,
    Output('mute-label', 'children'), Input('mute-btn', 'n_clicks')
)
app.clientside_callback(
    """
    function(freq) {
        try {
            if (!window.audioCtx) return '';
            if (window.audioCtx.state === 'suspended') window.audioCtx.resume();
            if (freq && freq > 0 && !window.muted) {
                if (!window.oscillator) {
                    window.oscillator = window.audioCtx.createOscillator();
                    window.oscillator.type = 'sine';
                    window.oscillator.connect(window.gainNode);
                    window.oscillator.start();
                }
                window.oscillator.frequency.linearRampToValueAtTime(freq, window.audioCtx.currentTime + 0.05);
            } else {
                if (window.oscillator) {
                    window.oscillator.stop();
                    window.oscillator.disconnect();
                    window.oscillator = null;
                }
            }
        } catch (e) { console.log('Sound clientside error:', e); }
        return '';
    }
    """,
    Output('sound-div', 'children'), Input('sound-freq', 'data')
)

if __name__ == '__main__':
    app.run(debug=True)