import os
import io
import base64
import threading

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

# Import PreventUpdate and callback_context for richer callback control
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

# ========================
# Hugging Face model setup
# ========================
MODEL_ID = "preszzz/drone-audio-detection-05-12"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") # Ensure this environment variable is set if needed

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
            # Updated auth token handling for newer transformers versions
            auth_token = HF_TOKEN if HF_TOKEN else None
            # Use trust_remote_code=True if required by the model
            proc = AutoProcessor.from_pretrained(MODEL_ID, token=auth_token, trust_remote_code=True)
            mod = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=auth_token, trust_remote_code=True)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mod.to(dev)
            mod.eval()
            _processor, _model, _device = proc, mod, dev
            print(f"Model loaded on {dev}")
            return _processor, _model, _device, None
        except Exception as e:
            print(f"Error loading model: {e}") # Print error for debugging
            return None, None, None, f"Failed to load model: {e}"


# ========================
# Audio Processing Helpers
# ========================

def resample_audio_decimation(audio_data, original_sr, target_sr):
    """
    Resamples audio using simple decimation (point sampling) for downsampling
    and sinc interpolation for upsampling. This creates clear aliasing artifacts
    when downsampling below Nyquist rate.
    """
    # Ensure audio_data is float32
    if not np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        if max_abs > 1.5:
             if max_abs > 0: audio_data /= max_abs

    if int(original_sr) == int(target_sr):
        return audio_data

    ratio = original_sr / target_sr
    
    if target_sr < original_sr:
        # DOWNSAMPLING: Use decimation (take every nth point)
        # This intentionally causes aliasing when sampling below Nyquist
        decimation_factor = int(np.round(ratio))
        if decimation_factor < 1:
            decimation_factor = 1
        
        print(f"Decimating audio by factor {decimation_factor} (taking every {decimation_factor}th sample)")
        print(f"⚠️ NO anti-aliasing filter applied - aliasing artifacts expected!")
        
        # Simple decimation: take every nth sample
        resampled = audio_data[::decimation_factor]
        
        return resampled
    else:
        # UPSAMPLING: Use sinc interpolation
        print(f"Upsampling audio from {original_sr} Hz to {target_sr} Hz using sinc interpolation")
        
        # Calculate new length
        new_length = int(len(audio_data) * target_sr / original_sr)
        
        # Create new time indices
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        
        # Sinc interpolation
        resampled = np.zeros(new_length, dtype=np.float32)
        
        # Use windowed sinc interpolation for efficiency
        window_size = 10  # Half-width of sinc window
        
        for i, new_idx in enumerate(new_indices):
            # Find nearby samples
            center = int(np.round(new_idx))
            start = max(0, center - window_size)
            end = min(len(audio_data), center + window_size + 1)
            
            # Calculate sinc weights
            for j in range(start, end):
                diff = new_idx - j
                if abs(diff) < 1e-6:
                    weight = 1.0
                else:
                    # Sinc function with Hamming window
                    sinc_val = np.sin(np.pi * diff) / (np.pi * diff)
                    # Hamming window
                    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * (j - start) / (end - start))
                    weight = sinc_val * hamming
                
                resampled[i] += audio_data[j] * weight
        
        return resampled

# ========================
# Prediction helper
# ========================
def predict_with_local_model(processor, model, device, audio, sr, chunk_s=2):
    """Runs inference on the audio data in chunks."""
    # Ensure audio is float32, as expected by librosa and the model
    if not np.issubdtype(audio.dtype, np.floating):
         audio = audio.astype(np.float32)
         # Normalize if it looks like integer audio might be scaled wrong
         max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0
         if max_abs > 1.5: # Use a threshold slightly > 1
              print(f"Warning: Input audio max abs value is {max_abs}. Assuming int16 scale and normalizing.")
              if max_abs > 0 : audio /= 32767.0 # Normalize based on typical int16 range

    # Ensure audio is mono
    if audio.ndim > 1:
        print("Audio has multiple channels, converting to mono.")
        audio = np.mean(audio, axis=1)

    results = []
    target_sr_model = 16000 # Model's expected sample rate

    # Resample audio to the model's target rate IF NECESSARY
    if sr != target_sr_model:
        print(f"Resampling audio from {sr} Hz to model's required {target_sr_model} Hz for prediction.")
        try:
            # Use float32 for resampling
            audio_resampled_for_model = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr_model, res_type='soxr_hq')
            current_sr_for_model = target_sr_model
        except Exception as e:
             print(f"Error during resampling for model prediction: {e}")
             return [{"error": f"Resampling failed: {e}"}]
    else:
        audio_resampled_for_model = audio
        current_sr_for_model = sr


    chunk_len = int(chunk_s * current_sr_for_model) # Ensure integer
    if len(audio_resampled_for_model) < current_sr_for_model: # Handle very short audio (less than 1s)
         print(f"Audio too short for prediction after potential resampling ({len(audio_resampled_for_model)} samples < {current_sr_for_model}).")
         return [{"label": "N/A", "score": 0.0, "message": "Audio too short"}]

    num_chunks_processed = 0
    for i in range(0, len(audio_resampled_for_model), chunk_len):
        chunk = audio_resampled_for_model[i:i + chunk_len]
        # Skip last chunk if significantly shorter than required (e.g., less than 0.5 second)
        if len(chunk) < current_sr_for_model * 0.5:
            print(f"Skipping final chunk of length {len(chunk)} < {current_sr_for_model * 0.5:.0f} samples.")
            continue

        num_chunks_processed += 1
        try:
            # Processor usually handles padding if needed with padding="longest" or similar
            inputs = processor(chunk, sampling_rate=current_sr_for_model, return_tensors="pt", padding="longest")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs))
            results.append({
                "chunk": i // chunk_len,
                "label": model.config.id2label[label_id],
                "score": float(probs[label_id]),
            })
        except Exception as e:
            print(f"Error during model inference on chunk {i // chunk_len}: {e}")
            results.append({"error": f"Inference failed on chunk: {e}"})
            # Optionally break or continue based on desired error handling
            # break

    if num_chunks_processed == 0 and len(audio_resampled_for_model) >= current_sr_for_model * 0.5:
         print("Warning: No chunks were processed despite audio being long enough. Check chunking/padding.")
         return [{"label": "N/A", "score": 0.0, "message": "Chunk processing error"}]
    elif num_chunks_processed == 0:
        # This case implies audio was shorter than 0.5s * model_sr
         print(f"Audio too short for prediction ({len(audio_resampled_for_model)} samples).")
         return [{"label": "N/A", "score": 0.0, "message": "Audio too short"}]


    return results


# ========================
# Dash app
# ========================
app = Dash(__name__, suppress_callback_exceptions=True) # suppress_callback_exceptions needed for dynamic layout

# --- UI STYLE CONSTANTS ---
PRIMARY_COLOR = "#4A90E2" # Blue
SECONDARY_COLOR = "#6c757d" # Greyish color for buttons/elements
BACKGROUND_COLOR = "#F7F9FC" # Light grey background
CARD_BACKGROUND_COLOR = "#FFFFFF" # White cards
TEXT_COLOR = "#333333" # Dark text
SUBTLE_TEXT_COLOR = "#666666" # Lighter grey text
BORDER_COLOR = "#EAEAEA" # Light border
FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' # Use Inter font
ERROR_COLOR = '#D32F2F' # Red for errors/warnings
SUCCESS_COLOR = '#388E3C' # Green for success messages
WARNING_COLOR = "#F59E0B" # Orange for warnings

# Base card style
CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
}

# Base button style
BUTTON_STYLE = {
    "backgroundColor": PRIMARY_COLOR, "color": "white", "border": "none",
    "borderRadius": "8px", "padding": "14px 24px", "fontSize": "16px", # Slightly larger padding/font
    "fontWeight": "600", "cursor": "pointer", 'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'textAlign': 'center', 'display':'inline-block', # Ensure centering
    'lineHeight': '1.2' # Adjust line height
}
# Hover effect base (can be merged)
BUTTON_HOVER_STYLE = {
    'filter': 'brightness(110%)',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.15)'
}
# Disabled style base (can be merged)
BUTTON_DISABLED_STYLE = {
    'backgroundColor': '#cccccc',
    'cursor': 'not-allowed',
    'boxShadow': 'none'
}

# Specific button styles by merging
BUTTON_STYLE_PRIMARY = {**BUTTON_STYLE}
BUTTON_STYLE_SECONDARY = {**BUTTON_STYLE, "backgroundColor": SECONDARY_COLOR}
BUTTON_STYLE_SUCCESS = {**BUTTON_STYLE, "backgroundColor": SUCCESS_COLOR}
BUTTON_STYLE_WARNING = {**BUTTON_STYLE, "backgroundColor": WARNING_COLOR}
BUTTON_STYLE_ERROR = {**BUTTON_STYLE, "backgroundColor": ERROR_COLOR}

# Style for Upload component
UPLOAD_STYLE = {
    "width": "100%", "height": "120px", "lineHeight": "120px", "borderWidth": "2px",
    "borderStyle": "dashed", "borderRadius": "10px", "borderColor": "#d0d0d0",
    "textAlign": "center", "cursor": "pointer", "backgroundColor": "#fafafa",
    'transition': 'all 0.3s ease-in-out', "marginBottom": "20px"
}
UPLOAD_HOVER_STYLE = { # Define separately if needed, or handle with CSS :hover
    "borderColor": PRIMARY_COLOR,
    "backgroundColor": "#f0f8ff"
}


# --- Plotting Helpers ---
def create_initial_figure(audio_data):
    """Generates a simple waveform plot for the initial view with enhanced styling."""
    fig = go.Figure()
    max_points = 50000
    if len(audio_data) <= max_points:
        step = 1
        x_display = np.arange(len(audio_data))
        y_display = audio_data
    else:
        step = max(1, len(audio_data) // max_points)
        y_display = audio_data[::step]
        x_display = np.arange(len(y_display)) * step

    fig.add_trace(go.Scattergl(x=x_display, y=y_display, mode="lines", line=dict(color=PRIMARY_COLOR, width=1.5))) # Slightly thicker line
    fig.update_layout(
        title="Waveform Preview",
        margin=dict(l=50, r=30, t=50, b=50), # Adjusted margins
        height=280, # Slightly taller
        xaxis_title="Sample Index", yaxis_title="Amplitude",
        plot_bgcolor=CARD_BACKGROUND_COLOR, # Match card background
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR), # Use main text color
        xaxis=dict(gridcolor=BORDER_COLOR), # Lighter grid lines
        yaxis=dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR)
    )
    return fig


def create_resampled_figure(original_audio, original_sr, new_sr, max_freq):
    """
    Generates a Plotly figure showing original vs decimated/resampled audio.
    Shows both the waveform and the actual downsampled result.
    """
    fig = go.Figure()
    nyquist_rate = 2 * max_freq
    title_text = "Original vs Decimated Waveform"
    is_aliasing = new_sr < nyquist_rate
    if is_aliasing:
        title_text = f"⚠️ Aliasing Present (Fs={new_sr} Hz < Nyquist={nyquist_rate:.0f} Hz)"
        title_font_color = ERROR_COLOR
    else:
        title_font_color = TEXT_COLOR

    # Show longer duration to see aliasing effects better
    display_duration_s = 0.1  # Increased from 0.05
    display_samples_orig = int(min(len(original_audio), original_sr * display_duration_s))

    if display_samples_orig < 2:
         fig.update_layout(title="Audio segment too short to display sampling")
         return fig

    original_audio_segment = original_audio[:display_samples_orig]
    time_original = np.linspace(0, display_samples_orig / original_sr, num=display_samples_orig)

    # Plot original signal
    fig.add_trace(go.Scattergl(
        x=time_original, y=original_audio_segment, mode="lines",
        line=dict(color='rgba(150, 150, 150, 0.6)', width=1.5), 
        name="Original Signal"
    ))

    # Perform actual decimation on this segment
    decimation_factor = int(np.round(original_sr / new_sr))
    if decimation_factor < 1:
        decimation_factor = 1
    
    decimated_segment = original_audio_segment[::decimation_factor]
    time_decimated = np.arange(len(decimated_segment)) * decimation_factor / original_sr

    # Plot decimated samples with connecting lines
    line_color = ERROR_COLOR if is_aliasing else PRIMARY_COLOR
    fig.add_trace(go.Scattergl(
        x=time_decimated, y=decimated_segment, mode="lines+markers",
        line=dict(color=line_color, width=2),
        marker=dict(color=line_color, size=8, symbol='circle'),
        name=f"Decimated (every {decimation_factor}th sample)"
    ))

    fig.update_layout(
        title=dict(text=title_text, font=dict(color=title_font_color, size=16)),
        margin=dict(l=50, r=30, t=60, b=50), height=320,
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        xaxis=dict(gridcolor=BORDER_COLOR),
        yaxis=dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.7)')
    )
    return fig

# --- Dash Layout ---
app.layout = html.Div(
    style={"fontFamily": FONT_FAMILY, "backgroundColor": BACKGROUND_COLOR, "padding": "40px", "minHeight": "100vh"},
    children=[
        html.Div(style={"maxWidth": "800px", "margin": "0 auto", "display": "flex", "flexDirection": "column",
                        "gap": "24px"},
                 children=[
                     # Header
                     html.Div([
                         # Consider adding an icon before the H1
                         html.H1("🚁 Drone Sound Analysis & Sampling Explorer",
                                 style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': '800',
                                        'letterSpacing': '-0.5px', 'fontSize': '2.2rem'}), # Larger, bolder header
                         html.P("Upload audio, classify using AI, and explore the Nyquist theorem interactively.",
                                style={'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '1.1rem', # Slightly larger subtext
                                       'maxWidth': '600px', 'margin': '5px auto 20px auto'}), # Adjusted margin
                     ]),

                     # Upload Card
                     html.Div(style=CARD_STYLE, children=[
                         dcc.Upload(
                             id="upload-audio",
                             children=html.Div(["📤 Drag & Drop or ", html.A("Select an Audio File", style={'color': PRIMARY_COLOR, 'fontWeight': '500'})]),
                             style=UPLOAD_STYLE,
                             multiple=False,
                         ),
                         html.Button("🚀 Analyze Original Audio", id="classify-btn", n_clicks=0, disabled=True, style=BUTTON_STYLE_PRIMARY),
                         html.Div(id="upload-error-output", style={'color': ERROR_COLOR, 'marginTop': '15px', 'fontSize': '14px', 'textAlign': 'center', 'fontWeight': '500'})
                     ]),

                     # Results Card (initially hidden)
                     html.Div(id="results-card", style={**CARD_STYLE, 'display': 'none'}, children=[
                         dcc.Loading(
                             id="loading-analysis", type="circle", # Changed loader type
                             children=[
                                 html.Div(id="results-content", children=[
                                     html.H3("Analysis Results",
                                             style={"marginTop": 0, "borderBottom": f"1px solid {BORDER_COLOR}",
                                                    "paddingBottom": "15px", "marginBottom": "20px", "color": TEXT_COLOR}), # Increased margin
                                     html.Div(id="file-name", style={"marginBottom": "15px", "fontWeight": "600", # Bolder filename
                                                                      "color": SUBTLE_TEXT_COLOR, "fontSize": "15px"}),
                                     # Classification result for ORIGINAL audio
                                     html.Div(id="classification-result",
                                              style={"fontSize": "22px", "fontWeight": "700", # Bolder result
                                                     "textAlign": "center", "marginBottom": "20px", "padding": "20px", # More padding
                                                     "borderRadius": "10px", # More rounded
                                                     'transition': 'all 0.3s ease-in-out'}), # Style set dynamically by callback
                                     html.Audio(id="audio-player", controls=True,
                                                style={"width": "100%", "marginTop": "10px", "marginBottom": "25px"}), # Increased margin

                                     # Button to show/hide sampling controls
                                     html.Button("🔬 Explore Sampling & Aliasing", id="show-sampling-btn", n_clicks=0,
                                                 style={**BUTTON_STYLE_SECONDARY, 'width': '100%', 'display':'none'}), # Initially hidden

                                     # Sampling Controls Section (initially hidden within results card)
                                     html.Div(id="sampling-controls", style={'display': 'none', 'marginTop': '25px'}, children=[
                                         html.Hr(style={'borderTop': f'2px solid {BORDER_COLOR}', 'margin': '25px 0'}), # Thicker separator
                                         html.H4("Interactive Sampling Explorer",
                                                 style={"marginTop": "20px", "marginBottom": "10px", 'color': TEXT_COLOR}),
                                         html.P(
                                             "Adjust the sampling frequency (Fs) slider below. The plot shows actual decimation - taking every Nth sample. When Fs < Nyquist Rate, high frequencies will alias into lower frequencies, creating audible distortion.",
                                             style={'fontSize': '14px', 'color': SUBTLE_TEXT_COLOR, 'marginBottom': '20px', 'lineHeight': '1.5'}),
                                         # Improved Nyquist info layout
                                         html.Div(id='nyquist-info',
                                                  style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap':'10px',
                                                         'margin': '20px 0', 'textAlign': 'center',
                                                         'backgroundColor': BACKGROUND_COLOR, 'padding': '15px', # More padding
                                                         'borderRadius': '8px', 'fontSize': '14px', 'border': f'1px solid {BORDER_COLOR}'}),
                                         dcc.Slider(id='sampling-freq-slider', min=500, max=48000, step=100, value=8000,
                                                    tooltip={"placement": "bottom", "always_visible": True}, className='custom-slider'), # Add class for potential CSS styling

                                         # Buttons for sampled audio
                                         html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginTop': '25px'}, children=[
                                              html.Button("▶️ Play Sampled", id="play-sampled-btn", n_clicks=0, style=BUTTON_STYLE_WARNING),
                                              html.Button("🧠 Predict Sampled", id="predict-sampled-btn", n_clicks=0, style=BUTTON_STYLE_SUCCESS),
                                         ]),

                                         html.Div(id="playback-warning", # Styled warning area
                                                  style={'color': WARNING_COLOR, 'textAlign': 'center', 'fontSize': '14px',
                                                         'fontWeight':'500', 'marginTop': '15px', 'padding':'8px', 'borderRadius':'6px', 'backgroundColor':'#FFF8E1'}),
                                         html.Audio(id="sampled-audio-player", controls=True,
                                                    style={"width": "100%", "marginTop": "15px", 'display': 'none'}),

                                         # Output for sampled audio prediction
                                         dcc.Loading(
                                                id="loading-sampled-prediction", type="circle", children=[
                                                    html.Div(id="sampled-classification-result",
                                                            style={"fontSize": "18px", "fontWeight": "600",
                                                                    "textAlign": "center", "marginTop": "25px", "padding": "20px",
                                                                    "borderRadius": "10px",
                                                                    'transition': 'all 0.3s ease-in-out', 'minHeight': '60px'}), # Style set dynamically
                                                ]
                                         )
                                     ]),
                                     # Waveform Plot (Updates based on slider when sampling controls are visible)
                                     dcc.Graph(id="waveform-plot", config={"displayModeBar": False, 'staticPlot': False}, style={'marginTop':'20px'}), # Ensure updates
                                 ])
                             ]
                         )
                     ]),
                     dcc.Store(id='audio-data-store'),  # Store for original audio data and metadata
                 ]
                 )
    ]
)

# --- Callbacks (Remain the same logically, but ensure styles match new definitions) ---

@app.callback(
    Output("classify-btn", "disabled"),
    Output("classify-btn", "style"), # Control style for disabled state
    Input("upload-audio", "contents"),
    State("classify-btn", "disabled"),
)
def update_button_state(contents, current_disabled_state):
    """Enable/disable the analyze button and update its style."""
    is_disabled = not contents
    # Prevent unnecessary style updates if disabled state hasn't changed
    if is_disabled == current_disabled_state:
        raise PreventUpdate

    new_style = {**BUTTON_STYLE_PRIMARY} # Start with base style
    if is_disabled:
        new_style.update(BUTTON_DISABLED_STYLE)

    return is_disabled, new_style


@app.callback(
    Output("results-card", "style", allow_duplicate=True),
    Output("file-name", "children"),
    Output("waveform-plot", "figure", allow_duplicate=True), # Allow duplicate for sampling plot
    Output("audio-player", "src"),
    Output("audio-data-store", "data"),
    Output("classification-result", "children", allow_duplicate=True), # Allow duplicate for error message
    Output("show-sampling-btn", "style", allow_duplicate=True), # Allow duplicate
    Output("sampling-controls", "style", allow_duplicate=True), # Allow duplicate
    Output("upload-error-output", "children"), # Output for upload errors
    Output("sampled-classification-result", "children", allow_duplicate=True), # Reset sampled result on new upload
    Output("sampled-classification-result", "style", allow_duplicate=True), # Reset sampled style
    Output("sampled-audio-player", "src", allow_duplicate=True), # Reset sampled player src
    Output("sampled-audio-player", "style", allow_duplicate=True), # Hide sampled player
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def display_uploaded_audio(contents, filename):
    """Callback to display the audio file immediately after upload, store data, but don't analyze yet."""
    if not contents:
        raise PreventUpdate

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    err_msg = ""

    try:
        audio_data, sr = sf.read(io.BytesIO(decoded))
        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1) # Convert to mono
        # Ensure float32 for consistency and normalize
        if not np.issubdtype(audio_data.dtype, np.floating):
            max_val = np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else 32767.0
            if max_val != 0: audio_data = audio_data.astype(np.float32) / max_val
            else: audio_data = audio_data.astype(np.float32)
        elif np.max(np.abs(audio_data)) > 1.5:
             print("Warning: Float audio data exceeds [-1, 1] range. Normalizing.")
             max_abs = np.max(np.abs(audio_data))
             if max_abs > 0: audio_data /= max_abs

    except Exception as e:
        err_msg = f"⚠️ Error reading audio file '{filename}': Please ensure it's a valid audio format. ({e})"
        print(err_msg)
        return ({'display': 'none'}, "", go.Figure(), None, None, "",
                {'display': 'none'}, {'display': 'none'}, err_msg,
                "", no_update, "", {'display':'none'})

    fig = create_initial_figure(audio_data)
    store_data = {'original_audio': audio_data.tolist(), 'original_sr': sr, 'filename': filename}
    initial_text = html.Span("File loaded. Click 'Analyze Original Audio' to classify.", style={'color': SUBTLE_TEXT_COLOR, 'fontStyle': 'italic'})

    sampled_result_style = {"fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px",
                           "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                           "border": f"1px solid {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px'}

    # Show results card, display waveform, enable player, prepare store, hide sampling
    return ({**CARD_STYLE, 'display': 'block'}, f"File: {filename}", fig, contents, store_data, initial_text,
            {'display': 'none'}, {'display': 'none'}, "", # Hide sampling button and controls initially
            " ", sampled_result_style, # Reset sampled result text and style
            "", {'display':'none'}) # Reset sampled player


@app.callback(
    Output("classification-result", "children", allow_duplicate=True),
    Output("classification-result", "style", allow_duplicate=True),
    Output("show-sampling-btn", "style", allow_duplicate=True),
    Output("sampling-freq-slider", "max"),
    Output("sampling-freq-slider", "value"), # Set initial slider value
    Output("sampling-freq-slider", "marks"),
    Output("nyquist-info", "children"),
    Output("audio-data-store", "data", allow_duplicate=True), # Update store with max_freq
    Input("classify-btn", "n_clicks"),
    State("audio-data-store", "data"),
    prevent_initial_call=True,
)
def analyze_original_audio(n_clicks, stored_data):
    """Callback to run analysis ONLY on the original audio when 'Analyze' button is clicked."""
    if not n_clicks or not stored_data or 'original_audio' not in stored_data:
        raise PreventUpdate

    try:
        audio_data = np.array(stored_data['original_audio'], dtype=np.float32)
        sr = stored_data['original_sr']
        filename = stored_data.get('filename', 'Unknown file')

        # --- AI Prediction on ORIGINAL audio ---
        print(f"Analyzing original audio: {filename} ({len(audio_data)} samples @ {sr} Hz)")
        processor, model, device, load_err = ensure_model()
        classification = f"⚠️ Model Load Error: {load_err}" if load_err else ""
        top_pred_label = "N/A"

        if not classification:
            preds = predict_with_local_model(processor, model, device, audio_data.copy(), sr)
            if preds and isinstance(preds, list) and len(preds) > 0:
                if any("error" in p for p in preds):
                    classification = f"⚠️ Prediction Error: {next((p['error'] for p in preds if 'error' in p), 'Unknown')}"
                    top_pred_label = "Error"
                elif "label" in preds[0] and "score" in preds[0]:
                    valid_preds = [p for p in preds if "error" not in p and "label" in p]
                    if valid_preds:
                        top_pred = max(valid_preds, key=lambda x: x.get('score', 0))
                        classification = html.Div([
                            html.Span("Original Prediction: "),
                            html.Strong(f"{top_pred['label']}", style={'color': PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else ERROR_COLOR}),
                            html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':SUBTLE_TEXT_COLOR})
                        ])
                        top_pred_label = top_pred['label']
                    else:
                        classification = "⚠️ Prediction valid but failed (e.g., all chunks failed)."
                        top_pred_label = "Error"
                elif "message" in preds[0]:
                     classification = html.Span(f"⚠️ {preds[0]['message']}", style={'color': WARNING_COLOR})
                     top_pred_label = "N/A"
                else:
                    classification = "⚠️ Prediction format unexpected."
                    top_pred_label = "Error"
            else:
                 classification = "⚠️ No prediction returned (file might be too short or model error)."
                 top_pred_label = "N/A"

        # --- Signal Analysis for Nyquist Rate ---
        n = len(audio_data)
        max_freq = sr / 4
        if n > 100:
            try:
                yf = np.fft.rfft(audio_data)
                xf = np.fft.rfftfreq(n, 1 / sr)
                if len(yf) > 1:
                    energy = np.abs(yf[1:]) ** 2
                    cumulative_energy = np.cumsum(energy)
                    total_energy = cumulative_energy[-1] if cumulative_energy.size > 0 else 0
                    if total_energy > 1e-9:
                        freq_99_percentile_idx = np.where(cumulative_energy >= total_energy * 0.99)[0][0]
                        actual_idx = freq_99_percentile_idx + 1
                        if actual_idx < len(xf): max_freq = xf[actual_idx]
                        elif len(xf) > 1: max_freq = xf[-1]
                    elif len(xf) > 1: max_freq = xf[-1]
                elif len(xf) > 0: max_freq = xf[-1] if len(xf) > 0 else sr / 4
            except (IndexError, TypeError, ValueError) as fft_err:
                print(f"Warning: FFT analysis for max_freq failed: {fft_err}")
                if n > 0 and 'xf' in locals() and len(xf) > 0: max_freq = xf[-1]

        max_freq = max(500, max_freq)
        nyquist_rate = 2 * max_freq
        slider_default = max(1000, np.ceil(nyquist_rate / 100.0) * 100)
        initial_sampling_rate = int(min(sr + 500, slider_default + 500, sr))

        # --- Styles and Outputs ---
        result_style = {"fontSize": "22px", "fontWeight": "700", "textAlign": "center", "marginBottom": "20px",
                        "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                        "border": f"2px solid"} # Base style, border color set below
        if 'drone' in top_pred_label.lower():
            result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': ERROR_COLOR, 'color': ERROR_COLOR})
        elif top_pred_label not in ["N/A", "Error"]:
             result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': PRIMARY_COLOR, 'color': PRIMARY_COLOR})
        else:
             result_style.update({'backgroundColor': BACKGROUND_COLOR, 'borderColor': BORDER_COLOR, 'color': TEXT_COLOR})

        stored_data['max_freq'] = max_freq
        slider_max = max(sr + 1000, 48000)
        # Adjusted mark styling for clarity
        slider_marks = {
             500: {'label': '0.5 kHz', 'style': {'fontSize': '11px'}},
             int(nyquist_rate): {'label': f'Nyquist ({nyquist_rate / 1000:.1f} kHz)', 'style': {'color': PRIMARY_COLOR, 'fontWeight': 'bold', 'fontSize': '11px', 'whiteSpace':'nowrap'}},
             int(slider_max): {'label': f'{int(slider_max/1000)} kHz', 'style': {'fontSize': '11px'}}
        }
        # Avoid overlapping marks
        if abs(500 - nyquist_rate) < slider_max * 0.05: del slider_marks[500]
        if abs(slider_max - nyquist_rate) < slider_max * 0.05: del slider_marks[int(slider_max)]


        nyquist_text = [ # Improved formatting
            html.Div([html.Strong("Original SR:"), html.Span(f" {sr} Hz")]),
            html.Div([html.Strong("Est. Max Freq:"), html.Span(f" {max_freq:.0f} Hz")]),
            html.Div([html.Strong("Nyquist Rate:"), html.Span(f" {nyquist_rate:.0f} Hz")], style={'fontWeight': 'bold', 'color': PRIMARY_COLOR})
        ]

        show_sampling_btn_style = {**BUTTON_STYLE_SECONDARY, 'width': '100%', 'marginTop': '15px', 'display': 'block'}

        return classification, result_style, show_sampling_btn_style, slider_max, initial_sampling_rate, slider_marks, nyquist_text, stored_data

    except Exception as e:
         print(f"Error during analysis: {e}")
         import traceback
         traceback.print_exc()
         error_text = html.Div([html.Strong("⚠️ Analysis Error: "), f"{e}"])
         error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                        "padding": "15px", "borderRadius": "8px", 'backgroundColor': ERROR_COLOR,
                        "border": f"1px solid {ERROR_COLOR}"}
         return error_text, error_style, {'display':'none'}, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("sampling-controls", "style", allow_duplicate=True),
    Input("show-sampling-btn", "n_clicks"),
    State("sampling-controls", "style"),
    prevent_initial_call=True
)
def toggle_sampling_controls(n_clicks, current_style):
    """Toggles the visibility of the sampling controls div."""
    if n_clicks % 2 == 1:
         return {'display': 'block', 'marginTop': '25px'} # Show with margin
    else:
         return {'display': 'none', 'marginTop': '25px'} # Hide


@app.callback(
    Output("waveform-plot", "figure", allow_duplicate=True),
    Output("playback-warning", "children"),
    Input("sampling-freq-slider", "value"),
    State("audio-data-store", "data"),
    State("sampling-controls", "style"),
    prevent_initial_call=True
)
def update_waveform_on_sample(new_sr, stored_data, sampling_style):
    """Updates the waveform plot ONLY when the sampling controls are visible."""
    if not stored_data or 'max_freq' not in stored_data or sampling_style.get('display') == 'none':
        raise PreventUpdate

    try:
        audio_data = np.array(stored_data['original_audio'], dtype=np.float32)
        fig = create_resampled_figure(audio_data, stored_data['original_sr'], new_sr, stored_data['max_freq'])

        warning_text = ""
        is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2
        if new_sr < 3000:
            # Use stronger warning style
            warning_text = html.Span("⚠️ Playback may fail or be distorted below ~3 kHz.", style={'color':WARNING_COLOR, 'fontWeight':'500'})
        elif is_aliasing:
             # Use span for specific styling
             warning_text = html.Span(["📉 Aliasing likely: ", html.B(f"Fs ({new_sr} Hz)"), f" < Nyquist ({stored_data.get('max_freq', 0) * 2:.0f} Hz)."], style={'color':WARNING_COLOR, 'fontWeight':'500'})

        return fig, warning_text
    except Exception as e:
         print(f"Error updating waveform plot: {e}")
         return no_update, html.Span(f"Error plotting: {e}", style={'color':ERROR_COLOR})


@app.callback(
    Output("sampled-audio-player", "src", allow_duplicate=True),
    Output("sampled-audio-player", "style", allow_duplicate=True),
    Input("play-sampled-btn", "n_clicks"),
    State("audio-data-store", "data"),
    State("sampling-freq-slider", "value"),
    prevent_initial_call=True
)
def play_resampled_audio(n_clicks, stored_data, new_sr):
    """Resamples the full audio and provides a data URI to the sampled audio player."""
    if not n_clicks or not stored_data: return no_update, {'display': 'none'}

    try:
        original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
        original_sr = stored_data['original_sr']

        print(f"Resampling audio from {original_sr} Hz to {new_sr} Hz for playback.")
        processed_audio = resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

        buffer = io.BytesIO()
        if len(processed_audio) == 0:
             print("Warning: Resampled audio is empty.")
             return "", {'display': 'none'}

        max_abs_val = np.max(np.abs(processed_audio)) if len(processed_audio) > 0 else 0
        processed_audio_int = np.zeros_like(processed_audio, dtype=np.int16)

        if max_abs_val > 0:
             # Scale float audio (assumed in [-X, X]) to int16 range [-32767, 32767]
             processed_audio_int = np.int16(processed_audio / max_abs_val * 32767)

        sf.write(buffer, processed_audio_int, int(new_sr), format='WAV', subtype='PCM_16')
        buffer.seek(0)

        encoded_sound = base64.b64encode(buffer.read()).decode()
        data_uri = f"data:audio/wav;base64,{encoded_sound}"

        return data_uri, {"width": "100%", "marginTop": "15px", 'display': 'block'} # Show player, more margin
    except Exception as e:
        print(f"Error resampling audio for playback: {e}")
        import traceback
        traceback.print_exc()
        return "", {'display': 'none'}


# --- Callback for Predicting Sampled Audio ---
@app.callback(
    Output("sampled-classification-result", "children", allow_duplicate=True),
    Output("sampled-classification-result", "style", allow_duplicate=True),
    Input("predict-sampled-btn", "n_clicks"),
    State("audio-data-store", "data"),
    State("sampling-freq-slider", "value"),
    prevent_initial_call=True
)
def predict_sampled_audio(n_clicks, stored_data, new_sr):
    """Resamples audio and runs prediction on the resampled version."""
    if not n_clicks or not stored_data or 'original_audio' not in stored_data:
        # Default text when no prediction has run
        default_style = {"fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
                        "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                        "border": f"1px dashed {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
                        "color": SUBTLE_TEXT_COLOR}
        return "Prediction result for sampled audio will appear here.", default_style

    # Loading message while processing
    loading_text = "🧠 Analyzing sampled audio..."
    loading_style = {"fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
                    "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                    "border": f"1px solid {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
                    "color": SUBTLE_TEXT_COLOR}


    try:
        original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
        original_sr = stored_data['original_sr']
        filename = stored_data.get('filename', 'Audio File') # Use default if filename missing

        print(f"Resampling {filename} from {original_sr} Hz to {new_sr} Hz for prediction.")
        # --- Resample audio using decimation ---
        resampled_audio_for_pred = resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

        # --- Predict on RESAMPLED audio ---
        print(f"Predicting on resampled audio ({len(resampled_audio_for_pred)} samples @ {new_sr} Hz)...")
        processor, model, device, load_err = ensure_model()
        classification_html = html.Span(f"⚠️ Model Load Error: {load_err}", style={'color': ERROR_COLOR}) if load_err else ""
        top_pred_label = "N/A"

        if not load_err:
            preds = predict_with_local_model(processor, model, device, resampled_audio_for_pred, new_sr)
            # --- Process prediction results ---
            if preds and isinstance(preds, list) and len(preds) > 0:
                if any("error" in p for p in preds):
                    error_msg = next((p['error'] for p in preds if 'error' in p), 'Unknown prediction error')
                    classification_html = html.Span(f"⚠️ Prediction Error: {error_msg}", style={'color': ERROR_COLOR})
                    top_pred_label = "Error"
                elif "label" in preds[0] and "score" in preds[0]:
                    valid_preds = [p for p in preds if "error" not in p and "label" in p]
                    if valid_preds:
                        top_pred = max(valid_preds, key=lambda x: x.get('score', 0))
                        # Format with label bolded and colored
                        pred_color = PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else ERROR_COLOR
                        classification_html = html.Div([
                            html.Span(f"Sampled Prediction ({new_sr} Hz): "),
                            html.Strong(f"{top_pred['label']}", style={'color': pred_color}),
                            html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':SUBTLE_TEXT_COLOR})
                        ])
                        top_pred_label = top_pred['label']
                    else:
                        classification_html = html.Span("⚠️ Prediction valid but failed (e.g., all chunks failed).", style={'color': WARNING_COLOR})
                        top_pred_label = "Error"
                elif "message" in preds[0]:
                     classification_html = html.Span(f"⚠️ {preds[0]['message']} (at {new_sr} Hz)", style={'color': WARNING_COLOR})
                     top_pred_label = "N/A"
                else:
                    classification_html = html.Span("⚠️ Prediction format unexpected.", style={'color': ERROR_COLOR})
                    top_pred_label = "Error"
            else:
                 classification_html = html.Span(f"⚠️ No prediction (audio might be too short after resampling to {new_sr} Hz).", style={'color': WARNING_COLOR})
                 top_pred_label = "N/A"


        # --- Style the output div ---
        result_style = {"fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px",
                        "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                        "border": f"2px solid", 'minHeight': '60px'} # Base style

        is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2

        # Update background/border/text color based on prediction
        if 'drone' in top_pred_label.lower():
            result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': ERROR_COLOR, 'color': ERROR_COLOR})
        elif top_pred_label not in ["N/A", "Error"]:
             result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': PRIMARY_COLOR, 'color': PRIMARY_COLOR})
        else: # Error or N/A
             result_style.update({'backgroundColor': BACKGROUND_COLOR, 'borderColor': BORDER_COLOR, 'color': TEXT_COLOR})

        # Add aliasing note if relevant
        aliasing_note_html = ""
        if is_aliasing:
             aliasing_note_html = html.P("Note: Prediction used audio sampled below Nyquist rate (aliasing likely occurred).",
                                  style={'fontSize': '12px', 'color': SUBTLE_TEXT_COLOR, 'marginTop': '10px', 'fontStyle': 'italic'})

        return html.Div([classification_html, aliasing_note_html]), result_style

    except Exception as e:
        print(f"Error during sampled prediction: {e}")
        import traceback
        traceback.print_exc()
        error_text = html.Div([html.Strong("⚠️ Sampled Prediction Error: "), f"{e}"])
        # Consistent error styling
        error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                       "padding": "15px", "borderRadius": "8px", 'backgroundColor': ERROR_COLOR,
                       "border": f"1px solid {ERROR_COLOR}", 'minHeight': '60px', "marginTop": "25px"}
        return error_text, error_style


if __name__ == "__main__":
    # Ensure model is loaded (or attempted) on startup
    ensure_model()
    # Consider adding host='0.0.0.0' if running in Docker or need external access
    app.run(debug=True, port=8051)