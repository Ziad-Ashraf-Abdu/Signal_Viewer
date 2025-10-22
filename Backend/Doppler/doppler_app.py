import os
import base64
import io
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.io import wavfile
import urllib.parse
import tensorflow as tf
import librosa
from pydub import AudioSegment

# --- Model Integration Start ---

# Load the pre-trained Keras model for velocity prediction
MODEL_PATH = 'velocity_regressor_dense.h5'
if os.path.exists(MODEL_PATH):
    try:
        velocity_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Velocity prediction model loaded successfully.")
    except Exception as e:
        velocity_model = None
        print(f"❌ Error loading velocity model: {e}")
else:
    velocity_model = None
    print(f"⚠️ Velocity prediction model not found at '{MODEL_PATH}'. Prediction will be disabled.")

# --- Model Integration End ---


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
app.title = "Doppler"

SPEED_OF_SOUND = 343  # m/s


# --- Model Integration Start ---

def predict_velocity(audio_data, sr):
    if velocity_model is None:
        return None
    try:
        target_sr = 22050
        # Convert to float32 if not already
        audio_data = np.asarray(audio_data, dtype=np.float32)

        # Resample if needed
        if sr != target_sr:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)

        # Ensure exactly 2 seconds at 22050 Hz → 44100 samples
        target_length = target_sr * 2  # 44100 samples
        if len(audio_data) > target_length:
            # Center crop
            start = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start:start + target_length]
        else:
            # Pad with zeros at the end (librosa's default behavior)
            audio_data = librosa.util.fix_length(data=audio_data, size=target_length)

        # Extract MFCCs (same params as training!)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sr, n_mfcc=30, n_fft=2048, hop_length=512)
        mfccs_mean = np.mean(mfccs, axis=1)  # shape: (30,)

        # Predict
        features = mfccs_mean.reshape(1, -1)
        prediction = velocity_model.predict(features, verbose=0)
        return float(prediction[0][0])
    except Exception as e:
        print(f"Error during velocity prediction: {e}")
        return None

# --- Model Integration End ---


# Helper to create labeled input fields
def labeled_input(label, id, value, width=80):
    return html.Div([
        html.Label(label, style={
            'display': 'inline-block',
            'width': '140px',
            'fontWeight': '600',
            'color': '#2c3e50',
            'fontSize': '14px'
        }),
        dcc.Input(id=id, type='number', value=value, style={
            'width': f'{width}px',
            'padding': '8px 12px',
            'border': '2px solid #e0e0e0',
            'borderRadius': '8px',
            'fontSize': '14px',
            'transition': 'all 0.3s ease',
            'outline': 'none'
        })
    ], style={'marginBottom': '15px'})


# 🔑 Makes any audio playable in browser
def make_playable_wav(audio, sr):
    if len(audio) == 0:
        return ""
    playback_sr = 22050
    if sr >= 3000:
        final_audio = audio
        final_sr = sr
    else:
        duration_sec = len(audio) / sr
        new_len = int(duration_sec * playback_sr)
        if new_len == 0:
            return ""
        indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
        final_audio = audio[indices]
        final_sr = playback_sr
    final_audio = np.clip(final_audio, -1.0, 1.0)
    wav_data = (final_audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, final_sr, wav_data)
    wav_bytes = buf.getvalue()
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"


# === LAYOUT ===
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚗 Doppler ", style={
            'textAlign': 'center',
            'color': 'white',
            'margin': '0',
            'padding': '30px',
            'fontSize': '36px',
            'fontWeight': '700',
            'letterSpacing': '1px',
            'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
        }),
        html.P("Upload a WAV file to analyze aliasing and simulate Doppler effect with real audio", style={
            'textAlign': 'center',
            'color': 'rgba(255,255,255,0.9)',
            'margin': '0',
            'paddingBottom': '20px',
            'fontSize': '16px'
        })
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    }),

    # === SHARED AUDIO UPLOAD ===
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("📁 Upload Car Audio (WAV)", style={
                            'color': '#667eea',
                            'marginBottom': '15px',
                            'fontSize': '24px',
                            'fontWeight': '700'
                        }),
                        dcc.Upload(
                            id='upload-audio',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select a WAV File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'marginBottom': '15px',
                                'backgroundColor': '#f0f4ff',
                                'borderColor': '#667eea'
                            },
                            multiple=False,
                            accept='.wav'
                        ),
                        html.Div(id='upload-status',
                                 style={'fontSize': '14px', 'color': '#6b7280', 'marginBottom': '10px'}),
                    ], style={
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                        'marginBottom': '20px'
                    })
                ])
            ])
        ], fluid=True)
    ], style={'padding': '0 20px'}),

    # === ALIASING SECTION ===
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("📉 Aliasing Demonstration", style={'color': '#10b981', 'marginBottom': '15px'}),
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
                    ], style={'padding': '20px', 'backgroundColor': '#f0fdf4', 'borderRadius': '10px',
                              'marginBottom': '20px'}),

                    html.Div([
                        html.H4("🔊 Original Audio"),
                        html.Audio(id='audio-orig', controls=True,
                                   style={'width': '100%', 'margin': '10px auto', 'display': 'block'})
                    ]),

                    html.Div([
                        html.H4("🔊 Aliased Audio (Downsampled – No Anti-Aliasing)"),
                        html.Audio(id='audio-alias', controls=True,
                                   style={'width': '100%', 'margin': '10px auto', 'display': 'block'})
                    ])
                ])
            ])
        ], fluid=True)
    ], style={'padding': '0 20px', 'marginBottom': '40px'}),

    # === DOPPLER SECTION ===
    html.Div([
        dbc.Container([
            # Audio Analysis for Doppler
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("🎵 Analysis & Prediction", style={
                            'color': '#667eea',
                            'marginBottom': '15px',
                            'fontSize': '24px',
                            'fontWeight': '700'
                        }),
                        html.H4(id='detected-freq-display', children="Detected Frequency: — Hz", style={
                            'color': '#ef4444',
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'marginBottom': '15px'
                        }),
                        html.H4(id='predicted-velocity-display', children="Predicted Velocity: — m/s", style={
                            'color': '#10b981',
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'marginBottom': '15px'
                        }),
                        html.Div([
                            html.Button('📊 Use This Frequency', id='use-audio-freq-btn', n_clicks=0, disabled=True,
                                        style={
                                            'padding': '10px 20px',
                                            'backgroundColor': '#667eea',
                                            'color': 'white',
                                            'border': 'none',
                                            'borderRadius': '8px',
                                            'fontSize': '14px',
                                            'fontWeight': '600',
                                            'cursor': 'pointer',
                                            'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.3)',
                                            'transition': 'all 0.3s ease'
                                        }),
                            html.Button('🚗 Use This Velocity', id='use-audio-velocity-btn', n_clicks=0, disabled=True,
                                        style={
                                            'padding': '10px 20px',
                                            'backgroundColor': '#10b981',
                                            'color': 'white',
                                            'border': 'none',
                                            'borderRadius': '8px',
                                            'fontSize': '14px',
                                            'fontWeight': '600',
                                            'cursor': 'pointer',
                                            'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.3)',
                                            'transition': 'all 0.3s ease',
                                            'marginLeft': '10px'
                                        })
                        ])
                    ], style={
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                        'marginBottom': '20px'
                    })
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='audio-spectrum-graph', style={'height': '350px'})
                ])
            ])
        ], fluid=True)
    ], style={'padding': '0 20px', 'marginBottom': '30px'}),

    # Doppler Controls (Source + Observer)
    html.Div([
        html.Div([
            # Source Panel
            html.Div([
                html.Div([
                    html.H3("🔊 Sound Source", style={
                        'color': '#667eea',
                        'marginBottom': '20px',
                        'fontSize': '22px',
                        'fontWeight': '700',
                        'borderBottom': '3px solid #667eea',
                        'paddingBottom': '10px'
                    }),
                    dcc.RadioItems(
                        id='source-type',
                        options=[{'label': ' Moving', 'value': 'moving'},
                                 {'label': ' Static', 'value': 'static'}],
                        value='moving',
                        inline=True,
                        style={'marginBottom': '20px'},
                        labelStyle={
                            'marginRight': '20px',
                            'fontSize': '15px',
                            'fontWeight': '500'
                        }
                    ),
                    html.Div([
                        labeled_input("Start X (m):", 'source-x0', -200),
                        labeled_input("Start Y (m):", 'source-y0', 0),
                        html.Div(id='source-vel-inputs', children=[
                            labeled_input("Speed (m/s):", 'source-speed', 30),
                            labeled_input("Direction (°):", 'source-dir', 0)
                        ])
                    ])
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#f8f9ff',
                    'borderRadius': '15px',
                    'border': '2px solid #667eea',
                    'boxShadow': '0 4px 12px rgba(102, 126, 234, 0.15)'
                })
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }),

            # Observer Panel
            html.Div([
                html.Div([
                    html.H3("👂 Observer", style={
                        'color': '#f093fb',
                        'marginBottom': '20px',
                        'fontSize': '22px',
                        'fontWeight': '700',
                        'borderBottom': '3px solid #f093fb',
                        'paddingBottom': '10px'
                    }),
                    dcc.RadioItems(
                        id='observer-type',
                        options=[{'label': ' Moving', 'value': 'moving'},
                                 {'label': ' Static', 'value': 'static'}],
                        value='moving',
                        inline=True,
                        style={'marginBottom': '20px'},
                        labelStyle={
                            'marginRight': '20px',
                            'fontSize': '15px',
                            'fontWeight': '500'
                        }
                    ),
                    html.Div([
                        labeled_input("Start X (m):", 'observer-x0', 0),
                        labeled_input("Start Y (m):", 'observer-y0', 0),
                        html.Div(id='observer-vel-inputs', children=[
                            labeled_input("Speed (m/s):", 'observer-speed', 10),
                            labeled_input("Direction (°):", 'observer-dir', 180)
                        ])
                    ])
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#fff8fd',
                    'borderRadius': '15px',
                    'border': '2px solid #f093fb',
                    'boxShadow': '0 4px 12px rgba(240, 147, 251, 0.15)'
                })
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'marginLeft': '4%'
            })
        ], style={'marginBottom': '30px', 'padding': '0 20px'}),

        # Global Settings
        html.Div([
            html.Div([
                labeled_input("Emitted Frequency (Hz):", 'freq-input', 500, width=100)
            ], style={
                'display': 'inline-block',
                'padding': '20px 40px',
                'backgroundColor': 'white',
                'borderRadius': '15px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                'border': '2px solid #e0e0e0'
            })
        ], style={'textAlign': 'center', 'marginBottom': '25px'}),

        # Buttons
        html.Div([
            html.Button('▶️ Start', id='start-btn', n_clicks=0, style={
                'marginRight': '12px',
                'padding': '12px 28px',
                'backgroundColor': '#10b981',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontSize': '16px',
                'fontWeight': '600',
                'cursor': 'pointer',
                'boxShadow': '0 4px 12px rgba(16, 185, 129, 0.3)',
                'transition': 'all 0.3s ease',
                'letterSpacing': '0.5px'
            }),
            html.Button('⏸️ Pause', id='pause-btn', n_clicks=0, style={
                'marginRight': '12px',
                'padding': '12px 28px',
                'backgroundColor': '#f59e0b',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontSize': '16px',
                'fontWeight': '600',
                'cursor': 'pointer',
                'boxShadow': '0 4px 12px rgba(245, 158, 11, 0.3)',
                'transition': 'all 0.3s ease',
                'letterSpacing': '0.5px'
            }),
            html.Button('⏹️ Reset', id='reset-btn', n_clicks=0, style={
                'marginRight': '12px',
                'padding': '12px 28px',
                'backgroundColor': '#ef4444',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontSize': '16px',
                'fontWeight': '600',
                'cursor': 'pointer',
                'boxShadow': '0 4px 12px rgba(239, 68, 68, 0.3)',
                'transition': 'all 0.3s ease',
                'letterSpacing': '0.5px'
            }),
            html.Button('🔈 Mute', id='mute-btn', n_clicks=0, style={
                'padding': '12px 24px',
                'backgroundColor': '#6b7280',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontSize': '16px',
                'fontWeight': '600',
                'cursor': 'pointer',
                'boxShadow': '0 4px 12px rgba(107, 114, 128, 0.3)',
                'transition': 'all 0.3s ease',
                'letterSpacing': '0.5px'
            }),
            html.Span(id='mute-label', children='', style={
                'marginLeft': '16px',
                'fontWeight': '600',
                'fontSize': '16px',
                'color': '#374151'
            })
        ], style={'textAlign': 'center', 'marginBottom': '25px'}),

        # Display
        html.Div(id='frequency-display', style={
            'fontSize': '20px',
            'textAlign': 'center',
            'padding': '20px',
            'fontWeight': '600',
            'color': '#1f2937',
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'margin': '0 20px 25px 20px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
            'border': '2px solid #e5e7eb'
        }),

        html.Div([
            dcc.Graph(id='simulation-graph', style={'height': '60vh'})
        ], style={
            'padding': '0 20px',
            'marginBottom': '30px'
        }),
    ]),

    # === HIDDEN STORES ===
    dcc.Store(id='simulation-running', data=False),
    dcc.Store(id='time-elapsed', data=0),
    dcc.Store(id='sound-freq', data=0),
    dcc.Store(id='uploaded-audio-data', data=None),
    dcc.Store(id='aliasing-audio', data=None),
    dcc.Store(id='predicted-velocity-store', data=None),

    html.Div(id='sound-init', style={'display': 'none'}),
    html.Div(id='sound-div', style={'display': 'none'}),

    dcc.Interval(id='interval', interval=100, n_intervals=0, disabled=True)
], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'backgroundColor': '#f9fafb',
    'minHeight': '100vh',
    'paddingBottom': '40px'
})


# === MAIN UPLOAD HANDLER (feeds both sections) ===
# (Keep all other code the same, only replace this function)

@callback(
    [
        Output('uploaded-audio-data', 'data'),
        Output('aliasing-audio', 'data'),
        Output('upload-status', 'children'),
        Output('fs-slider', 'disabled'),
        Output('fs-slider', 'max'),
        Output('fs-slider', 'value'),
        Output('fs-slider', 'marks'),
        Output('audio-orig', 'src'),
        Output('detected-freq-display', 'children'),
        Output('use-audio-freq-btn', 'disabled'),
        Output('audio-spectrum-graph', 'figure'),
        Output('predicted-velocity-display', 'children'),
        Output('predicted-velocity-store', 'data'),
        Output('use-audio-velocity-btn', 'disabled'),
    ],
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def handle_upload_and_analyze(contents, filename):
    if contents is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Audio Uploaded", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
        return [None] * 7 + ["", "Detected Frequency: — Hz", True, empty_fig, "Predicted Velocity: — m/s", None, True]

    try:
        _, b64 = contents.split(',')
        wav_bytes = base64.b64decode(b64)
        buffer = io.BytesIO(wav_bytes)

        # Load audio with librosa or fallback to pydub
        try:
            data_float, sr = librosa.load(buffer, sr=None, mono=True)
        except Exception:
            buffer.seek(0)
            try:
                audio_segment = AudioSegment.from_file(buffer, format="wav")
                sr = audio_segment.frame_rate
                audio_segment = audio_segment.set_channels(1)
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.sample_width == 2:
                    data_float = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4:
                    data_float = samples.astype(np.float32) / 2147483648.0
                else:
                    data_float = samples.astype(np.float32) / 128.0
            except Exception as pydub_error:
                user_error_message = f"❌ Error: The file '{filename}' is not a valid or supported WAV file. Please try exporting it again from an audio editor like Audacity."
                print(f"Pydub fallback failed: {pydub_error}")
                empty_fig = go.Figure().update_layout(title="File Error")
                return (None, None, user_error_message, True, 44100, 22050, {}, "",
                        "Detected Frequency: — Hz", True, empty_fig,
                        "Predicted Velocity: — m/s", None, True)

        aliasing_audio = data_float.tolist()

        # === Frequency Analysis – Highest Significant Frequency ===
        N = len(data_float)
        yf = fft(data_float)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])

        # Define significance threshold (e.g., 10% of max magnitude)
        threshold = 0.1 * np.max(magnitude)
        significant_indices = np.where(magnitude >= threshold)[0]

        if len(significant_indices) > 0:
            # Get the HIGHEST frequency with magnitude above threshold
            highest_significant_idx = significant_indices[-1]
            dominant_freq = float(xf[highest_significant_idx])
        else:
            # Fallback if no significant peaks
            dominant_freq = 500.0

        # --- Velocity Prediction ---
        predicted_velocity = predict_velocity(data_float, sr)
        if predicted_velocity is not None:
            velocity_text = f"Predicted Velocity: {predicted_velocity:.2f} m/s"
            velocity_button_disabled = False
        else:
            velocity_text = "Predicted Velocity: — m/s"
            velocity_button_disabled = True

        # Spectrum plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xf, y=magnitude, mode='lines', name='Spectrum', line=dict(color='#667eea')))
        fig.add_vline(
            x=dominant_freq,
            line=dict(color='#ef4444', dash='dash', width=3),
            annotation_text=f"Highest Sig.: {dominant_freq:.1f} Hz",
            annotation_position="top right",
            annotation_font=dict(size=14, color='#ef4444', family='Arial Black')
        )
        fig.update_layout(
            title="Uploaded Audio Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            xaxis_range=[0, min(2000, sr // 2)]
        )

        # Slider marks
        marks = {i: f"{i // 1000}k" for i in range(1000, sr + 1, max(2000, sr // 8))}
        marks[sr] = f"{sr // 1000}k"

        # Original audio playable
        orig_src = make_playable_wav(data_float, sr)

        # Store for Doppler section
        doppler_data = {'y': data_float.tolist(), 'sr': int(sr), 'dominant_freq': dominant_freq}

        return (
            doppler_data, aliasing_audio, f"✅ Loaded: {filename} | Fs = {sr} Hz",
            False, sr, min(sr, 22050), marks, orig_src,
            f"Detected Frequency: {dominant_freq:.1f} Hz", False, fig,
            velocity_text, predicted_velocity, velocity_button_disabled
        )

    except Exception as e:
        empty_fig = go.Figure().update_layout(title="Error")
        return (
            None, None, f"❌ An unexpected error occurred: {str(e)}", True, 44100, 22050, {}, "",
            "Detected Frequency: — Hz", True, empty_fig,
            "Predicted Velocity: — m/s", None, True
        )

# === ALIASING CALLBACKS ===
@callback(
    Output('audio-alias', 'src'),
    Input('fs-slider', 'value'),
    State('aliasing-audio', 'data'),
    State('uploaded-audio-data', 'data')
)
def update_aliasing(target_fs, audio_list, doppler_data):
    if audio_list is None or doppler_data is None:
        return ""
    data = np.array(audio_list, dtype=np.float32)
    orig_sr = doppler_data['sr']

    if target_fs >= orig_sr:
        aliased = data
        aliased_sr = orig_sr
    else:
        ratio = orig_sr / target_fs
        new_len = int(len(data) / ratio)
        indices = (np.arange(new_len) * ratio).astype(int)
        indices = indices[indices < len(data)]
        aliased = data[indices]
        aliased_sr = target_fs

    return make_playable_wav(aliased, aliased_sr)


@callback(
    Output('slider-value-display', 'children'),
    Input('fs-slider', 'value')
)
def display_slider_value(value):
    return f"Aliasing Sampling Frequency: {value} Hz"


# === DOPPLER CALLBACKS ===
@callback(
    [Output('source-speed', 'value'),
     Output('source-type', 'value')],
    Input('use-audio-velocity-btn', 'n_clicks'),
    State('predicted-velocity-store', 'data'),
    prevent_initial_call=True
)
def use_audio_velocity(n_clicks, velocity):
    if velocity is not None:
        return round(abs(velocity), 2), 'moving'
    return no_update, no_update


@callback(
    Output('freq-input', 'value'),
    Input('use-audio-freq-btn', 'n_clicks'),
    State('uploaded-audio-data', 'data'),
    prevent_initial_call=True
)
def use_audio_freq(n_clicks, audio_data):
    if audio_data and 'dominant_freq' in audio_data:
        return int(audio_data['dominant_freq'])
    return 500


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
    f_emit = f_emit if f_emit is not None else 500
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
        marker=dict(size=16, color='#f093fb', line=dict(color='white', width=2)),
        text=['👂 Observer'],
        textposition='top center',
        textfont=dict(size=14, color='#1f2937', family='Arial Black')
    ))

    car_svg_clean = '''
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(50,50) rotate({angle}) translate(-50,-50)">
        <rect x="20" y="40" width="60" height="25" rx="5" fill="#667eea" stroke="white" stroke-width="2"/>
        <path d="M 30 40 L 35 25 L 65 25 L 70 40 Z" fill="#667eea" stroke="white" stroke-width="2"/>
        <rect x="36" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
        <rect x="52" y="28" width="12" height="10" rx="2" fill="#E3F2FD"/>
        <circle cx="32" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
        <circle cx="68" cy="65" r="6" fill="#2c3e50" stroke="white" stroke-width="1.5"/>
        <circle cx="78" cy="45" r="2.5" fill="#FFF59D"/>
        <circle cx="78" cy="55" r="2.5" fill="#FFF59D"/>
    </g>
</svg>
    '''.format(angle=src_dir if src_type == 'moving' else 0)

    svg_encoded = urllib.parse.quote(car_svg_clean.strip(), safe='')
    svg_data_url = f"data:image/svg+xml;charset=utf-8,{svg_encoded}"

    fig.add_layout_image(
        dict(
            source=svg_data_url,
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
                opacity = 1 - (n / wave_count) * 0.7
                fig.add_shape(type="circle",
                              xref="x", yref="y",
                              x0=src_x - radius, y0=src_y - radius,
                              x1=src_x + radius, y1=src_y + radius,
                              line=dict(color="rgba(102, 126, 234, " + str(opacity) + ")",
                                        dash="dot", width=2))
    except Exception:
        pass

    fig.update_layout(
        xaxis=dict(
            title="X Position (meters)",
            range=[-300, 300],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            title="Y Position (meters)",
            range=[-150, 150],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        showlegend=False,
        title={
            'text': "Real-Time Doppler Effect Visualization",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f2937', 'family': 'Arial Black'}
        },
        plot_bgcolor='rgba(249, 250, 251, 0.5)',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(family='Arial, sans-serif', size=12, color='#374151')
    )

    return fig, freq_text, sound_freq


# === CLIENT-SIDE AUDIO CALLBACKS (Doppler) ===
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