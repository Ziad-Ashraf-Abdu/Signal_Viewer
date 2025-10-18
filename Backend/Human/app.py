import os
import io
import base64
import warnings
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf
import torch
from scipy.io import wavfile  # ✅ Added for robust audio playback

warnings.filterwarnings('ignore')

# --- Model Loading ---

try:
    from keras.models import load_model
    import tensorflow as tf

    ANTI_ALIASING_MODEL = load_model('./Anti-Aliasing.keras')
    MODEL_LOADED = True
    print("✅ AI model for audio reconstruction loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load reconstruction model. App will run in fallback mode. Error: {e}")
    MODEL_LOADED = False
    ANTI_ALIASING_MODEL = None

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
    gender_feature_extractor = AutoFeatureExtractor.from_pretrained(GENDER_MODEL_NAME)
    gender_model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL_NAME)
    GENDER_MODEL_LOADED = True
    print("✅ Gender detection model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load gender detection model. Feature will be disabled. Error: {e}")
    GENDER_MODEL_LOADED = False
    gender_feature_extractor = None
    gender_model = None


# --- Helper Functions ---

def analyze_audio(y, sr):
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    magnitude = np.abs(fft)
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)
    return {
        'waveform': y.tolist(), 'sample_rate': sr, 'duration': len(y) / sr,
        'frequencies': freqs.tolist(), 'magnitude': magnitude.tolist(),
        'nyquist_freq': sr / 2
    }


def apply_model_reconstruction(y_input, sr_input):
    if not MODEL_LOADED:
        print("Reconstruction model not available. Performing standard upsampling.")
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=16000)
    try:
        model_sr, model_len = 16000, 48000
        y_upsampled = librosa.resample(y_input, orig_sr=sr_input, target_sr=model_sr)
        reconstructed_chunks = []
        for i in range(0, len(y_upsampled), model_len):
            chunk = y_upsampled[i:i + model_len]
            len_chunk = len(chunk)
            if len_chunk < model_len:
                chunk = np.pad(chunk, (0, model_len - len_chunk), 'constant')
            model_input = chunk[np.newaxis, ..., np.newaxis]
            pred_chunk = np.squeeze(ANTI_ALIASING_MODEL.predict(model_input, verbose=0))
            if len_chunk < model_len:
                pred_chunk = pred_chunk[:len_chunk]
            reconstructed_chunks.append(pred_chunk)
        return np.concatenate(reconstructed_chunks) if reconstructed_chunks else np.array([])
    except Exception as e:
        print(f"❌ Error during model reconstruction: {e}")
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=16000)


# ✅ MODIFIED: Replaced with the robust function from your provided script
def create_audio_data(audio, sr):
    """Makes any audio playable in browser, handling very low sample rates."""
    if len(audio) == 0:
        return ""

    playback_sr = 22050  # A browser-safe sample rate

    # If the sample rate is too low for browsers, upsample without interpolation
    if sr < 3000:
        duration_sec = len(audio) / sr
        new_len = int(duration_sec * playback_sr)
        if new_len == 0: return ""
        # Create new indices by stretching the old ones
        indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
        final_audio = audio[indices]
        final_sr = playback_sr
    else:
        final_audio = audio
        final_sr = sr

    # Clip, convert to 16-bit PCM, and encode
    final_audio = np.clip(final_audio, -1.0, 1.0)
    wav_data = (final_audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, final_sr, wav_data)
    wav_bytes = buf.getvalue()
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"


def predict_gender(y, sr):
    if not GENDER_MODEL_LOADED:
        return "Gender detection model not available."
    try:
        if sr != 16000:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_resampled = y

        inputs = gender_feature_extractor(y_resampled, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = gender_model(inputs.input_values).logits

        pred_id = torch.argmax(logits, dim=-1).item()
        predicted_label = gender_model.config.id2label[pred_id].capitalize()
        return f"Predicted Gender: {predicted_label}"
    except Exception as e:
        print(f"❌ Error during gender prediction: {e}")
        return "Could not predict gender from audio."


# --- UI Component Creation Functions ---

def create_audio_players(original_data, downsampled_data, reconstructed_data):
    players = [html.Div([
        html.H5("1. Original Audio", style={'color': '#007bff'}),
        html.Audio(controls=True, style={'width': '100%'},
                   src=create_audio_data(np.array(original_data['waveform']), original_data['sample_rate']))
    ], style={'flex': '1', 'padding': '10px'})]
    if downsampled_data:
        players.append(html.Div([
            html.H5("2. Downsampled (Aliased) Audio", style={'color': '#dc3545'}),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(np.array(downsampled_data['waveform']), downsampled_data['sample_rate']))
        ], style={'flex': '1', 'padding': '10px'}))
    if reconstructed_data:
        reconstruction_title = "3. AI Reconstructed Audio" if MODEL_LOADED else "3. Upsampled Audio (Fallback)"
        reconstruction_color = '#28a745' if MODEL_LOADED else '#fd7e14'
        players.append(html.Div([
            html.H5(reconstruction_title, style={'color': reconstruction_color}),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(np.array(reconstructed_data['waveform']),
                                             reconstructed_data['sample_rate']))
        ], style={'flex': '1', 'padding': '10px'}))
    return html.Div(players, style={'display': 'flex', 'gap': '20px', 'justifyContent': 'space-around'})


# --- App Layout ---
app = dash.Dash(__name__)
app.title = "Audio Analysis Dashboard"
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("🎧 Interactive Audio Sampling & Analysis", style={'textAlign': 'center'}),
        html.P("Upload an audio file to downsample, reconstruct, and test the effects on an AI model.",
               style={'textAlign': 'center', 'color': '#6c757d'}),
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    dcc.Store(id='original-audio-data'),
    dcc.Store(id='downsampled-audio-data'),
    dcc.Store(id='reconstructed-audio-data'),

    html.Div([
        html.H3("1. Load Audio"),
        dcc.Upload(id='upload-audio', children=html.Div(['Drag and Drop or ', html.A('Select Audio File')]),
                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'}),
        html.Div(id='upload-status')
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'marginBottom': '20px'}),

    html.Div([
        html.H3("2. Process Audio"),
        html.Div([
            html.H4("Sampling Frequency", style={'color': '#dc3545'}),
            # ✅ MODIFIED: Lowered min to 500 and updated marks
            dcc.Slider(id='target-sr-slider', min=500, max=16000, step=500, value=8000,
                       marks={
                           500: '0.5k',
                           **{i: f'{i // 1000}k' for i in range(2000, 17000, 2000)}
                       },
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Button('Apply AI Reconstruction', id='model-reconstruct-btn', n_clicks=0,
                        style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'padding': '12px 24px',
                               'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px'}),
            html.Span(id='model-status', style={'marginLeft': '20px', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center'}),
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'marginBottom': '20px'}),

    html.Div([
        html.H3("3. Test Audio on Model"),
        html.P("Select an audio signal and test its effect on the model's prediction.", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='test-audio-selector',
            options=[
                {'label': 'Original Audio', 'value': 'original'},
                {'label': 'Downsampled (Aliased) Audio', 'value': 'downsampled'},
                {'label': 'AI Reconstructed Audio', 'value': 'reconstructed'}
            ],
            value='original',
            clearable=False,
            style={'marginBottom': '10px'}
        ),
        html.Button('Run Gender Detection Test', id='run-test-btn', n_clicks=0,
                    style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '12px 24px',
                           'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '16px', 'width': '100%'}),
        dcc.Loading(
            id="loading-test-result", type="circle",
            children=html.H4(id='test-result-output',
                             style={'textAlign': 'center', 'marginTop': '15px', 'color': '#17a2b8'})
        )
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'marginBottom': '20px'}),

    html.Div([
        html.H3("4. Listen & Analyze"),
        html.H4(id='gender-prediction-output', style={'textAlign': 'center', 'color': '#6a0dad', 'marginTop': '10px'}),

        html.Div([
            html.Div([
                html.Label("Select Signals to Display:"),
                dcc.Checklist(
                    id='signal-selector-checklist',
                    options=[
                        {'label': 'Original', 'value': 'original'},
                        {'label': 'Downsampled', 'value': 'downsampled'},
                        {'label': 'Reconstructed', 'value': 'reconstructed'},
                    ],
                    value=['original', 'downsampled', 'reconstructed'],
                    labelStyle={'display': 'inline-block', 'margin-right': '15px'}
                ),
            ], style={'margin-bottom': '10px'}),
            html.Div([
                html.Label("Graph View Mode:"),
                dcc.RadioItems(
                    id='graph-view-selector',
                    options=[
                        {'label': 'Overlap', 'value': 'overlap'},
                        {'label': 'Separate', 'value': 'separate'},
                    ],
                    value='overlap',
                    labelStyle={'display': 'inline-block', 'margin-right': '15px'}
                ),
            ]),
        ], style={'textAlign': 'center', 'margin-bottom': '20px'}),

        dcc.Loading(
            id="loading-main-output", type="default",
            children=html.Div([
                html.Div(id='audio-players'),
                html.Div(id='spectrum-plot-container', style={'marginTop': '20px'})
            ])
        )
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'}),
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


# --- Callbacks ---

@app.callback(
    Output('original-audio-data', 'data'),
    Output('upload-status', 'children'),
    Output('model-status', 'children'),
    Output('gender-prediction-output', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def upload_audio_file(contents, filename):
    recon_status = f"Recon AI: {'✅' if MODEL_LOADED else '❌'}"
    gender_status = f"Gender AI: {'✅' if GENDER_MODEL_LOADED else '❌'}"
    model_status_text = f"{recon_status} | {gender_status}"
    if contents is None:
        return None, "Please upload a WAV, MP3, or FLAC file.", model_status_text, ""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Use soundfile to read, as it's more robust with formats
        y, sr = sf.read(io.BytesIO(decoded), dtype='float32')
        if y.ndim > 1: y = y.mean(axis=1)
        if len(y) > sr * 20: y = y[:sr * 20]

        gender_text = "Initial " + predict_gender(y, sr)
        status_msg = f"✅ Loaded: {filename} ({len(y) / sr:.2f}s)"
        return analyze_audio(y, sr), status_msg, model_status_text, gender_text
    except Exception as e:
        return None, f"❌ Error: {e}", model_status_text, ""


@app.callback(
    Output('downsampled-audio-data', 'data'),
    Output('reconstructed-audio-data', 'data'),
    Input('target-sr-slider', 'value'),
    Input('model-reconstruct-btn', 'n_clicks'),
    State('original-audio-data', 'data')
)
def process_audio(target_sr, model_clicks, original_data):
    if original_data is None: return None, None
    ctx = callback_context
    triggered_id = ctx.triggered_id if ctx.triggered else 'No-ID'
    y_original = np.array(original_data['waveform'])
    sr_original = original_data['sample_rate']
    y_downsampled = librosa.resample(y_original, orig_sr=sr_original, target_sr=target_sr)
    current_downsampled_data = analyze_audio(y_downsampled, target_sr)
    if triggered_id == 'model-reconstruct-btn' and model_clicks > 0:
        y_input = np.array(current_downsampled_data['waveform'])
        sr_input = current_downsampled_data['sample_rate']
        y_reconstructed = apply_model_reconstruction(y_input, sr_input)
        reconstructed_data = analyze_audio(y_reconstructed, 16000)
        return current_downsampled_data, reconstructed_data
    return current_downsampled_data, dash.no_update if triggered_id == 'target-sr-slider' else None


@app.callback(
    Output('audio-players', 'children'),
    Output('spectrum-plot-container', 'children'),
    Input('original-audio-data', 'data'),
    Input('downsampled-audio-data', 'data'),
    Input('reconstructed-audio-data', 'data'),
    Input('graph-view-selector', 'value'),
    Input('signal-selector-checklist', 'value')
)
def update_ui(original_data, downsampled_data, reconstructed_data, view_mode, selected_signals):
    if original_data is None:
        return html.Div("Upload an audio file to begin."), None

    players = create_audio_players(original_data, downsampled_data, reconstructed_data)

    all_signals = {
        'original': {'data': original_data, 'name': 'Original', 'color': 'blue'},
        'downsampled': {'data': downsampled_data, 'name': 'Downsampled', 'color': 'red'},
        'reconstructed': {
            'data': reconstructed_data,
            'name': "AI Reconstructed" if MODEL_LOADED else "Upsampled (Fallback)",
            'color': '#28a745' if MODEL_LOADED else '#fd7e14'
        }
    }

    signals_to_plot = {key: val for key, val in all_signals.items() if
                       key in selected_signals and val['data'] is not None}

    if not signals_to_plot:
        return players, html.Div("Select a signal to display its graph.")

    if view_mode == 'overlap':
        fig = go.Figure()
        for key, sig_info in signals_to_plot.items():
            data = sig_info['data']
            fig.add_trace(go.Scatter(x=data['frequencies'], y=data['magnitude'],
                                     name=f"{sig_info['name']} ({data['sample_rate']} Hz)",
                                     line=dict(color=sig_info['color'], width=2)))
        if 'downsampled' in signals_to_plot:
            nyquist_freq = signals_to_plot['downsampled']['data']['nyquist_freq']
            fig.add_vline(x=nyquist_freq, line_dash="dash", line_color="red",
                          annotation_text=f"Nyquist: {nyquist_freq:.0f} Hz")
        fig.update_layout(title='Frequency Spectrum Comparison', hovermode='x unified', height=500)
        return players, dcc.Graph(figure=fig)

    else:  # 'separate' view
        graphs = []
        for key, sig_info in signals_to_plot.items():
            data = sig_info['data']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['frequencies'], y=data['magnitude'],
                                     name=sig_info['name'], line=dict(color=sig_info['color'])))
            title = f"{sig_info['name']} Spectrum ({data['sample_rate']} Hz)"
            if key == 'downsampled':
                fig.add_vline(x=data['nyquist_freq'], line_dash="dash", line_color="red",
                              annotation_text=f"Nyquist: {data['nyquist_freq']:.0f} Hz")
            fig.update_layout(title=title, height=300, margin=dict(t=40, b=40))
            graphs.append(dcc.Graph(figure=fig, style={'marginBottom': '10px'}))
        return players, graphs


@app.callback(
    Output('test-result-output', 'children'),
    Input('run-test-btn', 'n_clicks'),
    State('test-audio-selector', 'value'),
    State('original-audio-data', 'data'),
    State('downsampled-audio-data', 'data'),
    State('reconstructed-audio-data', 'data'),
    prevent_initial_call=True
)
def run_model_test(n_clicks, selected_audio_type, original_data, downsampled_data, reconstructed_data):
    if not GENDER_MODEL_LOADED: return "Gender model is not available for testing."
    data_map = {'original': original_data, 'downsampled': downsampled_data, 'reconstructed': reconstructed_data}
    selected_data = data_map.get(selected_audio_type)
    if selected_data is None: return f"Please generate '{selected_audio_type}' audio first."
    y = np.array(selected_data['waveform'])
    sr = selected_data['sample_rate']
    prediction = predict_gender(y, sr)
    return f"Test Result: {prediction}"


if __name__ == '__main__':
    app.run(debug=True, port=8054)