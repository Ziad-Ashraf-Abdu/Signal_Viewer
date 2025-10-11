import os
import io
import base64
import threading

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

# ========================
# Hugging Face model setup
# ========================
MODEL_ID = "preszzz/drone-audio-detection-05-12"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

_processor, _model, _device = None, None, "cpu"
_model_lock = threading.Lock()


def ensure_model():
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
    results = []
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
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

app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        html.Div(style={"maxWidth": "800px", "margin": "0 auto", "display": "flex", "flexDirection": "column",
                        "gap": "24px"}, children=[
            # Header
            html.Div([
                html.H1("🎤 Drone Sound Detection",
                        style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                html.P("Upload an audio file to classify if it contains a drone.",
                       style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '16px'}),
            ]),

            # Upload Card
            html.Div(style=CARD_STYLE, children=[
                dcc.Upload(
                    id="upload-audio",
                    children=html.Div(["📂 Drag & Drop or ", html.A("Select an Audio File")]),
                    style={
                        "width": "100%", "height": "100px", "lineHeight": "100px",
                        "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "10px",
                        "borderColor": BORDER_COLOR,
                        "textAlign": "center", "cursor": "pointer", "backgroundColor": BACKGROUND_COLOR,
                    },
                    multiple=False,
                ),
                html.Button("🚀 Classify Audio", id="classify-btn", n_clicks=0, style={
                    "marginTop": "20px", "width": "100%", "backgroundColor": PRIMARY_COLOR, "color": "white",
                    "border": "none", "borderRadius": "8px", "padding": "14px", "fontSize": "16px", "fontWeight": "600",
                    "cursor": "pointer",
                }),
            ]),

            # Results Card
            html.Div(id="results-card", style={**CARD_STYLE, 'display': 'none'}, children=[
                html.H3("Analysis Results", style={"marginTop": 0}),
                html.Div(id="file-name",
                         style={"marginBottom": "10px", "fontWeight": "bold", "color": SUBTLE_TEXT_COLOR}),
                dcc.Graph(id="waveform-plot", style={"height": "200px"}, config={"displayModeBar": False}),
                html.Audio(id="audio-player", controls=True, style={"width": "100%", "marginTop": "10px"}),
                html.Hr(style={'border': f'1px solid {BORDER_COLOR}', 'margin': '20px 0'}),
                html.Div(id="classification-result",
                         style={"fontSize": "22px", "fontWeight": "bold", "color": PRIMARY_COLOR,
                                "textAlign": "center"}),
            ]),
        ])
    ])


@app.callback(
    [Output("results-card", "style"), Output("file-name", "children"), Output("waveform-plot", "figure"),
     Output("audio-player", "src"), Output("classification-result", "children")],
    [Input("classify-btn", "n_clicks")],
    [State("upload-audio", "contents"), State("upload-audio", "filename")],
    prevent_initial_call=True,
)
def process_file(n_clicks, contents, filename):
    if not contents:
        return {'display': 'none'}, "", go.Figure(), None, ""

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        audio_data, sr = sf.read(io.BytesIO(decoded))
        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        return {**CARD_STYLE, 'display': 'block'}, f"Error reading {filename}", go.Figure(), None, f"⚠️ Error: {e}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=audio_data, mode="lines", line=dict(color=PRIMARY_COLOR)))
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=20), height=200,
        xaxis_title="Samples", yaxis_title="Amplitude",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=SUBTLE_TEXT_COLOR),
    )

    processor, model, device, load_err = ensure_model()
    if load_err:
        return {**CARD_STYLE, 'display': 'block'}, f"File: {filename}", fig, contents, f"⚠️ Model Error: {load_err}"

    preds = predict_with_local_model(processor, model, device, audio_data, sr)
    if not preds:
        return {**CARD_STYLE,
                'display': 'block'}, f"File: {filename}", fig, contents, "⚠️ No prediction (file might be too short)."

    top = max(preds, key=lambda x: x["score"])
    classification = f"Prediction: {top['label']} ({top['score'] * 100:.2f}%)"

    return {**CARD_STYLE, 'display': 'block'}, f"File: {filename}", fig, contents, classification


if __name__ == "__main__":
    app.run(debug=True, port=8051)