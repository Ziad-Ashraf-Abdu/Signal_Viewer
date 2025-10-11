import os
import io
import base64
import threading

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go


# ========================
# Hugging Face model setup
# ========================
MODEL_ID = "preszzz/drone-audio-detection-05-12"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # optional, only needed if model is private

# Globals for lazy loading
_processor = None
_model = None
_device = "cpu"
_model_lock = threading.Lock()


def ensure_model():
    """
    Lazily load the processor & model once and cache them.
    Returns (processor, model, device, error_message).
    """
    global _processor, _model, _device

    if _model is not None and _processor is not None:
        return _processor, _model, _device, None

    with _model_lock:
        if _model is not None and _processor is not None:
            return _processor, _model, _device, None

        try:
            print("Loading model:", MODEL_ID)
            if HF_TOKEN:
                proc = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
                mod = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
            else:
                proc = AutoProcessor.from_pretrained(MODEL_ID)
                mod = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mod.to(dev)
            mod.eval()

            _processor, _model, _device = proc, mod, dev
            print("Model loaded on", dev)
            return _processor, _model, _device, None
        except Exception as e:
            err = f"Failed to load model: {e}"
            print(err)
            return None, None, None, err


# ========================
# Prediction helper
# ========================
def predict_with_local_model(processor, model, device, audio, sr, chunk_s=2):
    """Split audio into chunks, classify each, return results."""
    results = []
    target_sr = 16000

    if sr != target_sr:
        print(f"Resampling from {sr} -> {target_sr}")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    chunk_len = chunk_s * sr
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i + chunk_len]
        if len(chunk) < sr:  # skip very small chunks
            continue

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        label_id = int(torch.argmax(probs))
        label = model.config.id2label[label_id]
        score = float(probs[label_id])

        results.append({
            "chunk": i // chunk_len,
            "label": label,
            "score": score,
            "all": {model.config.id2label[j]: float(p) for j, p in enumerate(probs)}
        })

    return results


# ========================
# Dash app
# ========================
app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "30px", "maxWidth": "800px", "margin": "auto"},
    children=[
        html.H2("🎤 Drone Sound Detection", style={"textAlign": "center"}),

        dcc.Upload(
            id="upload-audio",
            children=html.Div(["📂 Drag & Drop or ", html.A("Select an Audio File")]),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "marginBottom": "20px",
                "cursor": "pointer",
                "backgroundColor": "#f9f9f9",
            },
            multiple=False,
        ),

        html.Div(id="file-name", style={"marginBottom": "10px", "fontWeight": "bold"}),

        dcc.Graph(
            id="waveform-plot",
            style={"height": "250px"},
            config={"displayModeBar": False},
        ),

        html.Audio(id="audio-player", controls=True, style={"width": "100%", "marginTop": "10px"}),

        html.Button(
            "🚀 Classify",
            id="classify-btn",
            n_clicks=0,
            style={
                "marginTop": "20px",
                "padding": "10px 20px",
                "borderRadius": "8px",
                "border": "none",
                "backgroundColor": "#4CAF50",
                "color": "white",
                "fontSize": "16px",
                "cursor": "pointer",
            },
        ),

        html.Div(id="classification-result", style={"marginTop": "20px", "fontSize": "18px", "fontWeight": "bold", "color": "#333"}),
    ],
)


@app.callback(
    [
        Output("file-name", "children"),
        Output("waveform-plot", "figure"),
        Output("audio-player", "src"),
        Output("classification-result", "children"),
    ],
    [Input("classify-btn", "n_clicks")],
    [State("upload-audio", "contents"), State("upload-audio", "filename")],
    prevent_initial_call=True,
)
def process_file(n_clicks, contents, filename):
    if not contents:
        return "No file uploaded yet", go.Figure(), None, ""

    # Decode base64 audio
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        audio_data, sr = sf.read(io.BytesIO(decoded))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        return f"Error reading file: {e}", go.Figure(), None, ""

    # Waveform plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=audio_data, mode="lines", line=dict(color="blue")))
    fig.update_layout(
        margin=dict(l=40, r=40, t=20, b=20),
        height=250,
        xaxis_title="Samples",
        yaxis_title="Amplitude",
    )

    audio_src = contents  # reuse base64 for HTML audio

    # Load model if needed
    processor, model, device, load_err = ensure_model()
    if load_err:
        return f"Uploaded file: {filename}", fig, audio_src, f"⚠️ Model load failed: {load_err}"

    # Predict
    preds = predict_with_local_model(processor, model, device, audio_data, sr)
    if not preds:
        return f"Uploaded file: {filename}", fig, audio_src, "⚠️ No predictions (file too short?)"

    top = max(preds, key=lambda x: x["score"])
    classification = f"Prediction: {top['label']} ({top['score']*100:.2f}%)"

    return f"Uploaded file: {filename}", fig, audio_src, classification


if __name__ == "__main__":
    app.run(debug=True, port=8051)
