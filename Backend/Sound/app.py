import os
import io
import base64
import threading

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objs as go

# ========================
# Hugging Face model setup
# ========================
MODEL_ID = "preszzz/drone-audio-detection-05-12"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

_processor, _model, _device = None, None, "cpu"
_model_lock = threading.Lock()


def ensure_model():
    """Loads the Hugging Face model if it hasn't been loaded yet."""
    global _processor, _model, _device
    if _model is not None: return _processor, _model, _device, None
    with _model_lock:
        if _model is not None: return _processor, _model, _device, None
        try:
            print(f"Loading model: {MODEL_ID}")
            auth_token = {"token": HF_TOKEN} if HF_TOKEN else {}
            proc = AutoProcessor.from_pretrained(MODEL_ID, **auth_token)
            mod = AutoModelForAudioClassification.from_pretrained(MODEL_ID, **auth_token)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mod.to(dev)
            mod.eval()
            _processor, _model, _device = proc, mod, dev
            print(f"Model loaded on {dev}")
            return _processor, _model, _device, None
        except Exception as e:
            return None, None, None, f"Failed to load model: {e}"


# ========================
# Prediction helper
# ========================
def predict_with_local_model(processor, model, device, audio, sr, chunk_s=2):
    """Runs inference on the audio data in chunks."""
    results = []
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    chunk_len = chunk_s * sr
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i + chunk_len]
        if len(chunk) < sr: continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probs))
        results.append({
            "chunk": i // chunk_len,
            "label": model.config.id2label[label_id],
            "score": float(probs[label_id]),
        })
    return results


# ========================
# Dash app
# ========================
app = Dash(__name__)

# --- UI STYLE CONSTANTS ---
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#6c757d"
BACKGROUND_COLOR = "#F7F9FC"
CARD_BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
SUBTLE_TEXT_COLOR = "#666666"
BORDER_COLOR = "#EAEAEA"
FONT_FAMILY = "Inter, system-ui, sans-serif"

CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
}


# --- Plotting Helpers ---
def create_initial_figure(audio_data):
    """Generates a simple waveform plot for the initial view."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=audio_data, mode="lines", line=dict(color=PRIMARY_COLOR, width=1)))
    fig.update_layout(
        title="Waveform Preview", margin=dict(l=40, r=20, t=40, b=20), height=250,
        xaxis_title="Samples", yaxis_title="Amplitude",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=SUBTLE_TEXT_COLOR),
    )
    return fig


def create_resampled_figure(original_audio, original_sr, new_sr, max_freq):
    """Generates a Plotly figure showing original and resampled audio."""
    # To prevent visual clutter, only show a short slice of the original audio
    display_duration_s = 0.05
    display_samples = int(min(len(original_audio), original_sr * display_duration_s))
    original_audio_segment = original_audio[:display_samples]
    time_original = np.linspace(0, len(original_audio_segment) / original_sr, num=len(original_audio_segment))

    # Simulate sampling by taking every Nth sample
    step = max(1, int(round(original_sr / new_sr)))
    sampled_audio = original_audio_segment[::step]
    time_sampled = time_original[::step]

    fig = go.Figure()

    # Plot the original, high-resolution signal in the background
    fig.add_trace(go.Scatter(
        x=time_original, y=original_audio_segment, mode="lines",
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1.5), name="Original Signal"
    ))

    # Plot the sampled points and the reconstructed line
    fig.add_trace(go.Scatter(
        x=time_sampled, y=sampled_audio, mode="lines+markers",
        line=dict(color=PRIMARY_COLOR),
        marker=dict(color=PRIMARY_COLOR, size=6), name=f"Sampled at {new_sr} Hz"
    ))

    nyquist_rate = 2 * max_freq
    title_text = "Waveform Sampling (Zoomed View)"
    if new_sr < nyquist_rate:
        title_text += f" - ⚠️ Aliasing likely (Fs &lt; {nyquist_rate:.0f} Hz Nyquist Rate)"

    fig.update_layout(
        title=title_text, margin=dict(l=40, r=20, t=40, b=20), height=300,
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=SUBTLE_TEXT_COLOR),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig


app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        html.Div(style={"maxWidth": "800px", "margin": "0 auto", "display": "flex", "flexDirection": "column",
                        "gap": "24px"},
                 children=[
                     # Header
                     html.Div([
                         html.H1("🎤 Drone Sound Analysis",
                                 style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': '800',
                                        'letterSpacing': '-1px'}),
                         html.P("Upload an audio file to classify it and explore the effects of sampling.",
                                style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '18px',
                                       'maxWidth': '600px', 'margin': '0 auto'}),
                     ]),

                     # Upload Card
                     html.Div(style=CARD_STYLE, children=[
                         dcc.Upload(
                             id="upload-audio",
                             children=html.Div(["📤 Drag & Drop or ", html.A("Select an Audio File")]),
                             style={
                                 "width": "100%", "height": "120px", "lineHeight": "120px", "borderWidth": "2px",
                                 "borderStyle": "dashed", "borderRadius": "10px", "borderColor": "#d0d0d0",
                                 "textAlign": "center", "cursor": "pointer", "backgroundColor": "#fafafa",
                                 'transition': 'all 0.3s ease-in-out'
                             },
                             multiple=False,
                         ),
                         html.Button("🚀 Analyze Audio", id="classify-btn", n_clicks=0, disabled=True, style={
                             "marginTop": "20px", "width": "100%", "backgroundColor": PRIMARY_COLOR, "color": "white",
                             "border": "none", "borderRadius": "8px", "padding": "16px", "fontSize": "16px",
                             "fontWeight": "600",
                             "cursor": "pointer", 'transition': 'background-color 0.3s ease-in-out',
                             'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                         }),
                     ]),

                     # Results Card
                     html.Div(id="results-card", style={**CARD_STYLE, 'display': 'none'}, children=[
                         dcc.Loading(
                             id="loading-analysis", type="dot", children=[
                                 html.Div(id="results-content", children=[
                                     html.H3("Analysis Results",
                                             style={"marginTop": 0, "borderBottom": f"1px solid {BORDER_COLOR}",
                                                    "paddingBottom": "15px"}),
                                     html.Div(id="file-name", style={"marginBottom": "10px", "fontWeight": "bold",
                                                                     "color": SUBTLE_TEXT_COLOR}),
                                     html.Div(id="classification-result",
                                              style={"fontSize": "20px", "fontWeight": "600", "color": TEXT_COLOR,
                                                     "textAlign": "center", "marginBottom": "15px", "padding": "15px",
                                                     "borderRadius": "8px", "backgroundColor": BACKGROUND_COLOR,
                                                     "border": f"1px solid {BORDER_COLOR}",
                                                     'transition': 'all 0.3s ease-in-out'}),
                                     html.Audio(id="audio-player", controls=True,
                                                style={"width": "100%", "marginTop": "10px"}),
                                     html.Button("🔬 Explore Sampling & Aliasing", id="show-sampling-btn",
                                                 style={'display': 'none'}),
                                     html.Div(id="sampling-controls", style={'display': 'none'}, children=[
                                         html.Hr(style={'border': f'1px solid {BORDER_COLOR}', 'margin': '20px 0'}),
                                         html.H4("Waveform Sampling Explorer",
                                                 style={"marginTop": "20px", "marginBottom": "10px"}),
                                         html.P(
                                             "Adjust the sampling frequency to see its effect. If the frequency is less than twice the signal's maximum frequency (Nyquist Rate), aliasing occurs, creating a distorted representation.",
                                             style={'fontSize': '14px', 'color': SUBTLE_TEXT_COLOR}),
                                         html.Div(id='nyquist-info',
                                                  style={'display': 'flex', 'justifyContent': 'space-around',
                                                         'margin': '15px 0', 'textAlign': 'center',
                                                         'backgroundColor': BACKGROUND_COLOR, 'padding': '10px',
                                                         'borderRadius': '8px'}),
                                         dcc.Slider(id='sampling-freq-slider', min=500, max=48000, step=100, value=8000,
                                                    tooltip={"placement": "bottom", "always_visible": True}),
                                         html.Button("🎵 Play Sampled Audio", id="play-sampled-btn",
                                                     style={'marginTop': '20px', 'width': '100%',
                                                            'backgroundColor': PRIMARY_COLOR, 'color': 'white',
                                                            'border': 'none', 'borderRadius': '8px', 'padding': '12px',
                                                            'fontSize': '15px', 'cursor': 'pointer'}),
                                         html.Div(id="playback-warning",
                                                  style={'color': '#D32F2F', 'textAlign': 'center', 'fontSize': '14px',
                                                         'marginTop': '10px'}),
                                         html.Audio(id="sampled-audio-player", controls=True,
                                                    style={"width": "100%", "marginTop": "10px", 'display': 'none'}),
                                     ]),
                                     dcc.Graph(id="waveform-plot", config={"displayModeBar": False}),
                                 ])
                             ]
                         )
                     ]),
                     dcc.Store(id='audio-data-store'),  # Store for audio data
                 ]
                 )
    ]
    )


@app.callback(
    Output("classify-btn", "disabled"),
    Input("upload-audio", "contents")
)
def update_button_state(contents):
    """Enable the analyze button only when a file is uploaded."""
    return not contents


@app.callback(
    Output("results-card", "style"),
    Output("file-name", "children"),
    Output("waveform-plot", "figure"),
    Output("audio-player", "src"),
    Output("audio-data-store", "data"),
    Output("classification-result", "children"),
    Output("show-sampling-btn", "style"),
    Output("sampling-controls", "style"),
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def display_uploaded_audio(contents, filename):
    """Callback to display the audio file immediately after upload."""
    if not contents:
        return [no_update] * 8

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        audio_data, sr = sf.read(io.BytesIO(decoded))
        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        err_msg = f"⚠️ Error: Could not read audio file '{filename}'. Reason: {e}"
        fig = go.Figure()
        return {**CARD_STYLE, 'display': 'block'}, filename, fig, None, None, err_msg, {'display': 'none'}, {
            'display': 'none'}

    fig = create_initial_figure(audio_data)
    store_data = {'original_audio': audio_data.tolist(), 'original_sr': sr}
    initial_text = "File loaded. Click 'Analyze Audio' to begin."

    return {**CARD_STYLE, 'display': 'block'}, f"File: {filename}", fig, contents, store_data, initial_text, {
        'display': 'none'}, {'display': 'none'}


@app.callback(
    Output("classification-result", "children", allow_duplicate=True),
    Output("classification-result", "style"),
    Output("show-sampling-btn", "style", allow_duplicate=True),
    Output("sampling-freq-slider", "max"),
    Output("sampling-freq-slider", "value"),
    Output("sampling-freq-slider", "marks"),
    Output("nyquist-info", "children"),
    Output("waveform-plot", "figure", allow_duplicate=True),
    Output("audio-data-store", "data", allow_duplicate=True),
    Input("classify-btn", "n_clicks"),
    State("audio-data-store", "data"),
    prevent_initial_call=True,
)
def analyze_audio(n_clicks, stored_data):
    """Main callback to run analysis after the 'Analyze' button is clicked."""
    if not n_clicks or not stored_data:
        return [no_update] * 9

    audio_data = np.array(stored_data['original_audio'])
    sr = stored_data['original_sr']

    # --- Signal Analysis for Nyquist Rate ---
    n = len(audio_data)
    max_freq = sr / 4  # Default fallback
    if n > 100:
        yf = np.fft.rfft(audio_data)
        xf = np.fft.rfftfreq(n, 1 / sr)
        energy = np.abs(yf) ** 2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]
        try:
            freq_99_percentile_idx = np.where(cumulative_energy >= total_energy * 0.99)[0][0]
            max_freq = xf[freq_99_percentile_idx]
        except (IndexError, TypeError):
            pass

    max_freq = max(500, max_freq)
    nyquist_rate = 2 * max_freq
    initial_sampling_rate = int(min(sr, max(1000, nyquist_rate)))

    # --- AI Prediction ---
    processor, model, device, load_err = ensure_model()
    classification = f"⚠️ Model Error: {load_err}" if load_err else ""
    if not classification:
        preds = predict_with_local_model(processor, model, device, audio_data, sr)
        top_pred = max(preds, key=lambda x: x['score'])
        classification = "⚠️ No prediction (file too short)." if not preds else f"Prediction: {top_pred['label']} ({top_pred['score'] * 100:.2f}%)"

    # --- Create Dynamic Styles and Outputs ---
    result_style = {"fontSize": "20px", "fontWeight": "600", "textAlign": "center", "marginBottom": "15px",
                    "padding": "15px", "borderRadius": "8px", 'transition': 'all 0.3s ease-in-out'}
    if 'drone' in classification.lower():
        result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': '#FFCCC7', 'color': '#A82D26'})
    else:
        result_style.update({'backgroundColor': '#F0F9FF', 'borderColor': '#BDE2FF', 'color': '#1E40AF'})

    fig = create_resampled_figure(audio_data, sr, initial_sampling_rate, max_freq)
    stored_data['max_freq'] = max_freq  # Add max_freq to the store
    slider_marks = {int(nyquist_rate): {'label': 'Nyquist', 'style': {'color': PRIMARY_COLOR, 'fontWeight': 'bold'}}}

    nyquist_text = [
        html.Div(f"Original SR: {sr} Hz", style={'flex': 1}),
        html.Div(f"Est. Max Freq: {max_freq:.0f} Hz", style={'flex': 1}),
        html.Div(f"Nyquist Rate: {nyquist_rate:.0f} Hz", style={'flex': 1, 'fontWeight': 'bold'})
    ]

    show_sampling_btn_style = {'marginTop': '20px', 'width': '100%', 'backgroundColor': SECONDARY_COLOR,
                               'color': 'white', 'border': 'none', 'borderRadius': '8px', 'padding': '14px',
                               'fontSize': '15px', 'cursor': 'pointer', 'fontWeight': '600',
                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}

    return classification, result_style, show_sampling_btn_style, sr, initial_sampling_rate, slider_marks, nyquist_text, fig, stored_data


@app.callback(
    Output("sampling-controls", "style", allow_duplicate=True),
    Input("show-sampling-btn", "n_clicks"),
    State("sampling-controls", "style"),
    prevent_initial_call=True
)
def toggle_sampling_controls(n_clicks, current_style):
    """Toggles the visibility of the sampling controls div."""
    if n_clicks:
        if current_style and current_style.get('display') == 'block':
            return {'display': 'none'}
        return {'display': 'block'}
    return no_update


@app.callback(
    Output("waveform-plot", "figure", allow_duplicate=True),
    Output("playback-warning", "children"),
    Input("sampling-freq-slider", "value"),
    State("audio-data-store", "data"),
    prevent_initial_call=True
)
def update_waveform_on_sample(new_sr, stored_data):
    """Updates the waveform plot and warning message when the sampling slider is adjusted."""
    if not stored_data or 'max_freq' not in stored_data:
        return no_update, ""

    audio_data = np.array(stored_data['original_audio'])
    fig = create_resampled_figure(audio_data, stored_data['original_sr'], new_sr, stored_data['max_freq'])

    warning_text = ""
    if new_sr < 3000:
        warning_text = "⚠️ Playback may fail or be silent at frequencies below 3000 Hz due to browser limitations."

    return fig, warning_text


@app.callback(
    Output("sampled-audio-player", "src"),
    Output("sampled-audio-player", "style"),
    Input("play-sampled-btn", "n_clicks"),
    State("audio-data-store", "data"),
    State("sampling-freq-slider", "value"),
    prevent_initial_call=True
)
def play_resampled_audio(n_clicks, stored_data, new_sr):
    """Resamples the full audio and provides a data URI to the sampled audio player."""
    if not n_clicks or not stored_data: return no_update, no_update

    original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
    original_sr = stored_data['original_sr']

    # Bypass resampling if the new sample rate is the same as the original
    if int(new_sr) == int(original_sr):
        processed_audio = original_audio
    else:
        processed_audio = librosa.resample(y=original_audio, orig_sr=original_sr, target_sr=new_sr, res_type='soxr_hq')

    buffer = io.BytesIO()
    sf.write(buffer, processed_audio, int(new_sr), format='WAV', subtype='PCM_16')
    buffer.seek(0)

    encoded_sound = base64.b64encode(buffer.read()).decode()
    data_uri = f"data:audio/wav;base64,{encoded_sound}"

    return data_uri, {"width": "100%", "marginTop": "10px", 'display': 'block'}


if __name__ == "__main__":
    app.run(debug=True, port=8051)

