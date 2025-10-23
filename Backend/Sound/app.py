# --- SEARCHABLE COMMENT: Imports ---
import os
import io
import base64
import threading

import numpy as np
import soundfile as sf # For reading audio files
import librosa # For audio resampling
import torch # PyTorch for AI model
from transformers import AutoProcessor, AutoModelForAudioClassification # Hugging Face model components

# Added missing imports for filtering
from scipy.signal import butter, filtfilt

# Import Dash components for building the web application
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate # To prevent callback updates under certain conditions
import plotly.graph_objs as go # For creating plots


# ============================================
# --- SEARCHABLE COMMENT: Model Setup ---
# Configuration for the Hugging Face audio classification model.
# ============================================
MODEL_ID = "preszzz/drone-audio-detection-05-12" # Model identifier on Hugging Face Hub
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") # Optional: Hugging Face API token for private models

# --- SEARCHABLE COMMENT: Global Model Variables ---
# Global variables to hold the loaded model and processor, avoiding reloading on every request.
_processor, _model, _device = None, None, "cpu"
_model_lock = threading.Lock() # Lock to prevent race conditions during model loading


# --- SEARCHABLE COMMENT: Ensure Model Function ---
def ensure_model():
    """
    Loads the Hugging Face model and processor into global variables if they haven't been loaded yet.
    Uses a lock to ensure thread safety during loading.

    Returns:
        tuple: (processor, model, device, error_message)
    """
    global _processor, _model, _device
    # Check if already loaded (fast path)
    if _model is not None: return _processor, _model, _device, None

    # Acquire lock for thread-safe loading
    with _model_lock:
        # Double-check if loaded by another thread while waiting for the lock
        if _model is not None: return _processor, _model, _device, None
        try:
            print(f"Loading model: {MODEL_ID}")
            # --- SEARCHABLE COMMENT: Model Authentication ---
            # Use token if provided, otherwise attempts public download
            auth_token = HF_TOKEN if HF_TOKEN else None
            # --- SEARCHABLE COMMENT: Model Loading ---
            # trust_remote_code=True might be needed for some custom model architectures
            proc = AutoProcessor.from_pretrained(MODEL_ID, token=auth_token, trust_remote_code=True)
            mod = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=auth_token, trust_remote_code=True)

            # --- SEARCHABLE COMMENT: Device Selection (CPU/GPU) ---
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mod.to(dev) # Move model to the selected device
            mod.eval() # Set model to evaluation mode (disables dropout, etc.)

            # Store loaded components in global variables
            _processor, _model, _device = proc, mod, dev
            print(f"Model loaded successfully on {dev}")
            return _processor, _model, _device, None
        except Exception as e:
            # --- SEARCHABLE COMMENT: Model Load Error Handling ---
            print(f"Error loading model: {e}") # Log error for debugging
            # Return error message if loading fails
            return None, None, None, f"Failed to load model: {e}"


# ============================================
# --- SEARCHABLE COMMENT: Audio Processing Helpers ---
# Functions for resampling, filtering, and preparing audio data.
# ============================================

# --- SEARCHABLE COMMENT: Resample Audio Function (Decimation/Interpolation) ---
def resample_audio_decimation(audio_data, original_sr, target_sr):
    """
    Resamples audio using simple decimation (point sampling) for downsampling
    and basic sinc interpolation for upsampling.
    ***This version explicitly DOES NOT use an anti-aliasing filter during decimation***
    to clearly demonstrate aliasing artifacts.

    Args:
        audio_data (np.ndarray): Input audio signal.
        original_sr (int): Original sampling rate.
        target_sr (int): Desired new sampling rate.

    Returns:
        np.ndarray: Resampled audio signal.
    """
    # Ensure audio is float32 and normalized roughly to [-1, 1] for processing consistency
    if not np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        # Normalize assuming integer type if max_abs is large
        if max_abs > 1.5:
            if max_abs > 0: audio_data /= 32767.0 # Assume int16 range if large float

    # If rates are the same, no resampling needed
    if int(original_sr) == int(target_sr):
        return audio_data

    ratio = original_sr / target_sr

    # --- SEARCHABLE COMMENT: Downsampling (Decimation) ---
    if target_sr < original_sr: # Downsampling
        decimation_factor = int(np.round(ratio))
        # Ensure factor is at least 1
        if decimation_factor < 1:
            decimation_factor = 1

        print(f"Decimating audio by factor {decimation_factor} (taking every {decimation_factor}th sample)")
        print(f"⚠️ WARNING: No anti-aliasing filter applied before decimation - Aliasing artifacts are expected!")

        # Perform decimation by simple slicing
        resampled = audio_data[::decimation_factor]
        return resampled
    # --- SEARCHABLE COMMENT: Upsampling (Sinc Interpolation) ---
    else: # Upsampling
        print(f"Upsampling audio from {original_sr} Hz to {target_sr} Hz using basic sinc interpolation")
        new_length = int(len(audio_data) * target_sr / original_sr)
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)

        # Basic windowed sinc interpolation
        resampled = np.zeros(new_length, dtype=np.float32)
        window_size = 10 # Half-width of the sinc window

        for i, new_idx in enumerate(new_indices):
            center = int(np.round(new_idx))
            start = max(0, center - window_size)
            end = min(len(audio_data), center + window_size + 1)

            # Apply sinc function weighted by Hamming window
            for j in range(start, end):
                diff = new_idx - j
                if abs(diff) < 1e-6: # Avoid division by zero at the center
                    weight = 1.0
                else:
                    sinc_val = np.sin(np.pi * diff) / (np.pi * diff)
                    # Hamming window calculation
                    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * (j - start) / (end - start -1e-9)) # Avoid div by zero if end-start=0
                    weight = sinc_val * hamming

                resampled[i] += audio_data[j] * weight

        return resampled


# ============================================
# --- SEARCHABLE COMMENT: Prediction Helper ---
# Function to run the AI model on audio data.
# ============================================
def predict_with_local_model(processor, model, device, audio, sr, chunk_s=2):
    """
    Runs inference using the loaded Hugging Face model on the provided audio data.
    Handles necessary preprocessing like mono conversion and resampling to the model's required rate (16kHz).
    Processes audio in chunks to handle potentially long files.

    Args:
        processor: The Hugging Face processor.
        model: The Hugging Face model.
        device: The device (CPU or CUDA) the model is on.
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate of the input audio.
        chunk_s (int): Duration of audio chunks in seconds for processing.

    Returns:
        list: A list of dictionaries, each containing prediction results ('label', 'score') for a chunk,
              or an error message.
    """
    # --- SEARCHABLE COMMENT: Prediction Audio Preprocessing ---
    # Ensure audio is float32 and roughly normalized
    if not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0
        if max_abs > 1.5:
            print(f"Warning: Input audio max abs value is {max_abs}. Assuming int16 scale and normalizing.")
            if max_abs > 0 : audio /= 32767.0 # Normalize based on typical int16 range

    # Ensure audio is mono
    if audio.ndim > 1:
        print("Audio has multiple channels, converting to mono.")
        audio = np.mean(audio, axis=1)

    results = []
    target_sr_model = 16000 # The specific sample rate the model was trained on

    # --- SEARCHABLE COMMENT: Prediction Resampling (to Model Rate) ---
    # Resample the input audio *only if needed* to match the model's expected sample rate
    if sr != target_sr_model:
        print(f"Resampling audio from {sr} Hz to model's required {target_sr_model} Hz for prediction.")
        try:
            # Using librosa's high-quality resampler
            audio_resampled_for_model = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr_model, res_type='soxr_hq')
            current_sr_for_model = target_sr_model
        except Exception as e:
            print(f"Error during resampling for model prediction: {e}")
            return [{"error": f"Resampling failed: {e}"}] # Return error if resampling fails
    else:
        # No resampling needed if sample rates already match
        audio_resampled_for_model = audio
        current_sr_for_model = sr

    # --- SEARCHABLE COMMENT: Prediction Chunking ---
    chunk_len = int(chunk_s * current_sr_for_model) # Calculate chunk length in samples
    # Check if the entire audio is too short (less than 1 second)
    if len(audio_resampled_for_model) < current_sr_for_model:
        print(f"Audio too short for prediction after potential resampling ({len(audio_resampled_for_model)} samples < {current_sr_for_model}).")
        return [{"label": "N/A", "score": 0.0, "message": "Audio too short"}]

    num_chunks_processed = 0
    # Iterate through the audio in chunks
    for i in range(0, len(audio_resampled_for_model), chunk_len):
        chunk = audio_resampled_for_model[i:i + chunk_len]

        # --- SEARCHABLE COMMENT: Prediction Chunk Handling ---
        # Skip the last chunk if it's too short (e.g., less than 0.5 seconds)
        # This avoids potential issues with the model on very short inputs
        min_chunk_len = current_sr_for_model * 0.5
        if len(chunk) < min_chunk_len:
            print(f"Skipping final chunk of length {len(chunk)} < {min_chunk_len:.0f} samples.")
            continue

        num_chunks_processed += 1
        try:
            # --- SEARCHABLE COMMENT: Model Inference ---
            # Process the chunk using the Hugging Face processor
            # padding="longest" handles chunks shorter than model's expected input size
            inputs = processor(chunk, sampling_rate=current_sr_for_model, return_tensors="pt", padding="longest")
            # Run the model inference (disabling gradient calculation for efficiency)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits # Send input to the correct device
            # --- SEARCHABLE COMMENT: Model Postprocessing ---
            # Apply softmax to get probabilities and find the most likely class
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs))
            # Store the result for this chunk
            results.append({
                "chunk": i // chunk_len,
                "label": model.config.id2label[label_id], # Map label ID to human-readable label
                "score": float(probs[label_id]), # Confidence score
            })
        except Exception as e:
            # --- SEARCHABLE COMMENT: Prediction Error Handling ---
            print(f"Error during model inference on chunk {i // chunk_len}: {e}")
            results.append({"error": f"Inference failed on chunk: {e}"})
            # Optionally stop processing chunks on error: break

    # Handle cases where no chunks were processed (e.g., if initial audio > 1s but < 0.5s after chunking logic)
    if num_chunks_processed == 0 and len(audio_resampled_for_model) >= min_chunk_len :
        print("Warning: No chunks were processed despite audio being long enough. Check chunking/padding.")
        return [{"label": "N/A", "score": 0.0, "message": "Chunk processing error"}]
    elif num_chunks_processed == 0:
        # Already handled by the initial length check if audio < 1s
        pass

    return results # Return the list of predictions (or errors) for each chunk


# ============================================
# --- SEARCHABLE COMMENT: Dash App Initialization ---
# Setting up the Dash application instance.
# ============================================
app = Dash(__name__, suppress_callback_exceptions=True) # suppress_callback_exceptions allows callbacks for dynamically generated components

# ============================================
# --- SEARCHABLE COMMENT: UI Style Constants ---
# Centralized definitions for colors, fonts, and layout styles used throughout the UI.
# ============================================
PRIMARY_COLOR = "#4A90E2" # Blue
SECONDARY_COLOR = "#6c757d" # Greyish color for buttons/elements
BACKGROUND_COLOR = "#F7F9FC" # Light grey background
CARD_BACKGROUND_COLOR = "#FFFFFF" # White cards
TEXT_COLOR = "#333333" # Dark text
SUBTLE_TEXT_COLOR = "#666666" # Lighter grey text
BORDER_COLOR = "#EAEAEA" # Light border
FONT_FAMILY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' # Use Inter font stack
ERROR_COLOR = '#D32F2F' # Red for errors/warnings
SUCCESS_COLOR = '#388E3C' # Green for success messages
WARNING_COLOR = "#F59E0B" # Orange for warnings

# --- SEARCHABLE COMMENT: Card Style ---
# Style dictionary for the main content cards.
CARD_STYLE = {
    "backgroundColor": CARD_BACKGROUND_COLOR, "borderRadius": "12px", "padding": "24px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)", "border": f"1px solid {BORDER_COLOR}",
    "marginBottom": "24px" # Added default bottom margin
}

# --- SEARCHABLE COMMENT: Button Styles ---
# Base button style and variations for different states/purposes.
BUTTON_STYLE = {
    "backgroundColor": PRIMARY_COLOR, "color": "white", "border": "none",
    "borderRadius": "8px", "padding": "14px 24px", "fontSize": "16px",
    "fontWeight": "600", "cursor": "pointer", 'transition': 'all 0.2s ease',
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'textAlign': 'center', 'display':'inline-block',
    'lineHeight': '1.2', 'width': '100%' # Default buttons to full width
}
BUTTON_DISABLED_STYLE = { # Style for disabled buttons
    'backgroundColor': '#cccccc', 'cursor': 'not-allowed', 'boxShadow': 'none'
}
BUTTON_STYLE_PRIMARY = {**BUTTON_STYLE}
BUTTON_STYLE_SECONDARY = {**BUTTON_STYLE, "backgroundColor": SECONDARY_COLOR}
BUTTON_STYLE_SUCCESS = {**BUTTON_STYLE, "backgroundColor": SUCCESS_COLOR}
BUTTON_STYLE_WARNING = {**BUTTON_STYLE, "backgroundColor": WARNING_COLOR}
BUTTON_STYLE_ERROR = {**BUTTON_STYLE, "backgroundColor": ERROR_COLOR}

# --- SEARCHABLE COMMENT: Upload Style ---
UPLOAD_STYLE = {
    "width": "100%", "height": "120px", "lineHeight": "120px", "borderWidth": "2px",
    "borderStyle": "dashed", "borderRadius": "10px", "borderColor": "#d0d0d0",
    "textAlign": "center", "cursor": "pointer", "backgroundColor": "#fafafa",
    'transition': 'all 0.3s ease-in-out', "marginBottom": "20px"
}


# ============================================
# --- SEARCHABLE COMMENT: Plotting Helpers ---
# Functions to create the Plotly figures used in the dashboard.
# ============================================

# --- SEARCHABLE COMMENT: Initial Waveform Plot Function ---
def create_initial_figure(audio_data):
    """
    Generates a Plotly figure showing a preview of the uploaded audio waveform.
    Uses Scattergl for better performance with potentially large datasets.
    Downsamples the displayed data if the audio file is very long.

    Args:
        audio_data (np.ndarray): The audio signal data.

    Returns:
        plotly.graph_objs.Figure: The generated figure object.
    """
    fig = go.Figure()
    max_points = 50000 # Limit points on the plot for performance
    if len(audio_data) <= max_points:
        step = 1
        x_display = np.arange(len(audio_data)) # Use sample index for x-axis
        y_display = audio_data
    else:
        # Downsample data for plotting if too long
        step = max(1, len(audio_data) // max_points)
        y_display = audio_data[::step]
        x_display = np.arange(len(y_display)) * step # Adjust x-axis for downsampling

    # --- SEARCHABLE COMMENT: Initial Waveform Trace ---
    fig.add_trace(go.Scattergl(x=x_display, y=y_display, mode="lines", line=dict(color=PRIMARY_COLOR, width=1.5))) # Slightly thicker line

    # --- SEARCHABLE COMMENT: Initial Waveform Layout ---
    fig.update_layout(
        title="Waveform Preview",
        margin=dict(l=50, r=30, t=50, b=50), # Adjusted margins for better spacing
        height=280, # Slightly taller plot
        xaxis_title="Sample Index", yaxis_title="Amplitude",
        plot_bgcolor=CARD_BACKGROUND_COLOR, # Match card background for seamless look
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR), # Use consistent font and color
        xaxis=dict(gridcolor=BORDER_COLOR), # Lighter grid lines
        yaxis=dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR) # Lighter grid and zero lines
    )
    return fig

# --- SEARCHABLE COMMENT: Resampled Waveform Plot Function ---
# --- SEARCHABLE COMMENT: Aliasing Visualization ---
def create_resampled_figure(original_audio, original_sr, new_sr, max_freq):
    """
    Generates a Plotly figure comparing a segment of the original audio
    with the points that would be sampled at the `new_sr`.
    Visually demonstrates the effect of sampling and potential aliasing.

    Args:
        original_audio (np.ndarray): The original full audio signal.
        original_sr (int): The original sampling rate.
        new_sr (int): The target sampling rate from the slider.
        max_freq (float): Estimated maximum frequency in the original signal.

    Returns:
        plotly.graph_objs.Figure: The generated figure object.
    """
    fig = go.Figure()
    nyquist_rate = 2 * max_freq
    title_text = "Waveform Sampling (Zoomed View)"
    is_aliasing = new_sr < nyquist_rate # --- SEARCHABLE COMMENT: Aliasing Check ---

    # --- SEARCHABLE COMMENT: Aliasing Warning Title ---
    if is_aliasing:
        # Modify title to warn about aliasing
        title_text = f"⚠️ Aliasing Likely (Fs={new_sr} Hz < Nyquist={nyquist_rate:.0f} Hz)"
        title_font_color = ERROR_COLOR # Use error color for title
    else:
        title_font_color = TEXT_COLOR # Use default text color

    # Display a short segment (e.g., 50ms) for clarity
    display_duration_s = 0.05
    display_samples_orig = int(min(len(original_audio), original_sr * display_duration_s))

    # Handle cases where the segment is too short to plot
    if display_samples_orig < 2:
        fig.update_layout(title="Audio segment too short to display sampling visualization.")
        return fig

    original_audio_segment = original_audio[:display_samples_orig]
    # Time axis for the original segment
    time_original = np.linspace(0, display_samples_orig / original_sr, num=display_samples_orig)

    # --- SEARCHABLE COMMENT: Original Signal Trace (Resampled Plot) ---
    # Plot the original signal segment faintly in the background
    fig.add_trace(go.Scattergl(
        x=time_original, y=original_audio_segment, mode="lines",
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2), name="Original Signal" # Thicker faint line
    ))

    # --- SEARCHABLE COMMENT: Sample Point Calculation ---
    # Calculate which points *would be* sampled at the new rate within this segment
    num_samples_new = int(display_duration_s * new_sr)
    if num_samples_new < 2: # Need at least 2 points to show sampling
        fig.update_layout(title=title_text + " - (Target SR too low for visualization)", title_font_color=title_font_color)
        return fig

    # Determine the indices in the *original* segment array corresponding to the new sample times
    max_index = display_samples_orig - 1
    sample_indices_orig = np.linspace(0, max_index, num=num_samples_new, dtype=int)
    # Ensure indices are within bounds (important due to potential floating point inaccuracies)
    sample_indices_orig = np.clip(sample_indices_orig, 0, max_index)

    # Get the amplitude values and times for the sampled points
    sampled_audio = original_audio_segment[sample_indices_orig]
    time_sampled = sample_indices_orig / original_sr # Calculate exact time of each sample

    # --- SEARCHABLE COMMENT: Sampled Signal Trace (Resampled Plot) ---
    # Plot the sampled points and connect them with lines
    line_color = ERROR_COLOR if is_aliasing else PRIMARY_COLOR # --- SEARCHABLE COMMENT: Aliasing Color Change ---
    fig.add_trace(go.Scattergl(
        x=time_sampled, y=sampled_audio, mode="lines+markers",
        line=dict(color=line_color, width=1.5), # Sampled line style
        marker=dict(color=line_color, size=7, symbol='circle-open'), # Marker style for sample points
        name=f"Sampled at {new_sr} Hz"
    ))

    # --- SEARCHABLE COMMENT: Resampled Plot Layout ---
    fig.update_layout(
        title=dict(text=title_text, font=dict(color=title_font_color, size=16)), # Styled title
        margin=dict(l=50, r=30, t=60, b=50), height=320, # Adjusted margins and height
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR, # Match card background
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR), # Consistent font
        xaxis=dict(gridcolor=BORDER_COLOR), # Lighter grid
        yaxis=dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.7)') # Semi-transparent legend
    )
    return fig

# ============================================
# --- SEARCHABLE COMMENT: Dash Layout Definition ---
# Defines the structure and components of the web application's UI.
# ============================================
app.layout = html.Div(
    # --- SEARCHABLE COMMENT: Main App Container Style ---
    style={
        "fontFamily": FONT_FAMILY,
        "backgroundColor": BACKGROUND_COLOR,
        "padding": "40px 20px", # Padding top/bottom and left/right
        "minHeight": "100vh" # Ensure background covers full height
    },
    children=[
        # --- SEARCHABLE COMMENT: Centered Content Container ---
        html.Div(
            style={ "maxWidth": "900px", "margin": "0 auto" }, # Center content horizontally
            children=[
                # --- SEARCHABLE COMMENT: Header Section ---
                html.Div(
                    style={"marginBottom": "32px"},
                    children=[
                        html.H1(
                            "🚁 Drone Sound Analysis & Sampling Explorer",
                            style={ 'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': '800',
                                    'letterSpacing': '-0.5px', 'fontSize': '2.2rem', 'marginBottom': '8px' }
                        ),
                        html.P(
                            "Upload audio, classify using AI, and explore the Nyquist theorem interactively.",
                            style={ 'textAlign': 'center', 'color': SUBTLE_TEXT_COLOR, 'fontSize': '1.1rem', 'margin': '0' }
                        ),
                    ]
                ), # End Header

                # --- SEARCHABLE COMMENT: Upload Card ---
                html.Div(
                    style=CARD_STYLE,
                    children=[
                        # --- SEARCHABLE COMMENT: Upload Component ---
                        dcc.Upload(
                            id="upload-audio",
                            children=html.Div(["📤 Drag & Drop or ", html.A("Select an Audio File", style={'color': PRIMARY_COLOR, 'fontWeight': '500'})]),
                            style=UPLOAD_STYLE,
                            multiple=False, # Allow only single file uploads
                        ),
                        # --- SEARCHABLE COMMENT: Analyze Button ---
                        html.Button(
                            "🚀 Analyze Original Audio",
                            id="classify-btn",
                            n_clicks=0,
                            disabled=True, # Initially disabled, enabled by callback
                            style=BUTTON_STYLE_PRIMARY # Apply primary button style
                        ),
                        # --- SEARCHABLE COMMENT: Upload Error Output ---
                        html.Div(
                            id="upload-error-output",
                            style={ 'color': ERROR_COLOR, 'marginTop': '15px', 'fontSize': '14px',
                                    'textAlign': 'center', 'fontWeight': '500' }
                        )
                    ]
                ), # End Upload Card

                # --- SEARCHABLE COMMENT: Results Card ---
                # This card contains all analysis outputs and controls, initially hidden.
                html.Div(
                    id="results-card",
                    style={**CARD_STYLE, 'display': 'none'}, # Initially hidden
                    children=[
                        # --- SEARCHABLE COMMENT: Loading Indicator (Analysis) ---
                        dcc.Loading(
                            id="loading-analysis",
                            type="circle", # Use circle style loader
                            children=[
                                html.Div(
                                    id="results-content",
                                    children=[
                                        # --- SEARCHABLE COMMENT: Results Header ---
                                        html.H3(
                                            "Analysis Results",
                                            style={ "marginTop": 0, "borderBottom": f"2px solid {BORDER_COLOR}", # Thicker border
                                                    "paddingBottom": "15px", "marginBottom": "20px", "color": TEXT_COLOR }
                                        ),
                                        # --- SEARCHABLE COMMENT: File Name Display ---
                                        html.Div(
                                            id="file-name",
                                            style={ "marginBottom": "15px", "fontWeight": "600",
                                                    "color": SUBTLE_TEXT_COLOR, "fontSize": "15px" }
                                        ),
                                        # --- SEARCHABLE COMMENT: Classification Result (Original) ---
                                        # Content and style updated dynamically by callback
                                        html.Div(
                                            id="classification-result",
                                            style={ "fontSize": "22px", "fontWeight": "700", "textAlign": "center",
                                                    "marginBottom": "20px", "padding": "20px", "borderRadius": "10px",
                                                    'transition': 'all 0.3s ease-in-out' }
                                        ),
                                        # --- SEARCHABLE COMMENT: Audio Player (Original) ---
                                        html.Audio(
                                            id="audio-player", controls=True,
                                            style={ "width": "100%", "marginTop": "10px", "marginBottom": "25px" }
                                        ),

                                        # --- SEARCHABLE COMMENT: Explore Sampling Button ---
                                        # Toggles the visibility of the sampling controls section
                                        html.Button(
                                            "🔬 Explore Sampling & Aliasing", id="show-sampling-btn", n_clicks=0,
                                            style={**BUTTON_STYLE_SECONDARY, 'display': 'none'} # Hidden until analysis runs
                                        ),

                                        # --- SEARCHABLE COMMENT: Sampling Controls Section ---
                                        # Contains slider, plots, and buttons related to resampling. Initially hidden.
                                        html.Div(
                                            id="sampling-controls",
                                            style={'display': 'none', 'marginTop': '25px'}, # Hidden, appears on button click
                                            children=[
                                                html.Hr(style={'borderTop': f'2px solid {BORDER_COLOR}', 'margin': '25px 0'}),
                                                html.H4(
                                                    "Interactive Sampling Explorer",
                                                    style={ "marginTop": "0", "marginBottom": "10px", 'color': TEXT_COLOR }
                                                ),
                                                html.P(
                                                    "Adjust the sampling frequency (Fs) slider below. When Fs < Nyquist Rate, high frequencies alias into lower frequencies, creating distortion.",
                                                    style={ 'fontSize': '14px', 'color': SUBTLE_TEXT_COLOR,
                                                            'marginBottom': '20px', 'lineHeight': '1.5' }
                                                ),
                                                # --- SEARCHABLE COMMENT: Nyquist Info Display ---
                                                # Shows calculated rates
                                                html.Div(
                                                    id='nyquist-info',
                                                    style={ 'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap':'15px',
                                                            'margin': '20px 0', 'textAlign': 'center',
                                                            'backgroundColor': BACKGROUND_COLOR, 'padding': '15px',
                                                            'borderRadius': '8px', 'fontSize': '14px', 'border': f'1px solid {BORDER_COLOR}'}
                                                ),
                                                # --- SEARCHABLE COMMENT: Sampling Frequency Slider ---
                                                dcc.Slider(
                                                    id='sampling-freq-slider', min=500, max=48000, step=100, value=8000,
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className='custom-slider' # Class for potential future CSS styling
                                                ),

                                                # --- SEARCHABLE COMMENT: Sampled Audio Buttons ---
                                                # Grid layout for Play and Predict buttons
                                                html.Div(
                                                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginTop': '25px'},
                                                    children=[
                                                        # --- SEARCHABLE COMMENT: Play Sampled Button ---
                                                        html.Button("▶️ Play Sampled", id="play-sampled-btn", n_clicks=0, style=BUTTON_STYLE_WARNING),
                                                        # --- SEARCHABLE COMMENT: Predict Sampled Button ---
                                                        html.Button("🧠 Predict Sampled", id="predict-sampled-btn", n_clicks=0, style=BUTTON_STYLE_SUCCESS),
                                                    ]
                                                ),

                                                # --- SEARCHABLE COMMENT: Playback Warning ---
                                                # Displays warnings related to low sample rates or aliasing
                                                html.Div(
                                                    id="playback-warning",
                                                    style={ 'color': WARNING_COLOR, 'textAlign': 'center', 'fontSize': '14px',
                                                            'fontWeight':'500', 'marginTop': '15px', 'padding':'10px',
                                                            'borderRadius':'6px', 'backgroundColor':'#FFF8E1', 'minHeight': '20px'} # Ensure space even when empty
                                                ),
                                                # --- SEARCHABLE COMMENT: Audio Player (Sampled) ---
                                                html.Audio(
                                                    id="sampled-audio-player", controls=True,
                                                    style={"width": "100%", "marginTop": "15px", 'display': 'none'} # Initially hidden
                                                ),

                                                # --- SEARCHABLE COMMENT: Classification Result (Sampled) ---
                                                # Includes loading indicator
                                                dcc.Loading(
                                                    id="loading-sampled-prediction", type="circle",
                                                    children=[
                                                        html.Div(
                                                            id="sampled-classification-result",
                                                            # Style updated dynamically by callback
                                                            style={ "fontSize": "18px", "fontWeight": "600", "textAlign": "center",
                                                                    "marginTop": "25px", "padding": "20px", "borderRadius": "10px",
                                                                    'transition': 'all 0.3s ease-in-out', 'minHeight': '60px'}
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ), # End Sampling Controls Section

                                        # --- SEARCHABLE COMMENT: Waveform Plot ---
                                        # Displays either the initial preview or the interactive resampled view
                                        dcc.Graph(
                                            id="waveform-plot",
                                            config={"displayModeBar": False, 'staticPlot': False}, # Allow dynamic updates
                                            style={'marginTop':'25px'} # Add margin
                                        ),
                                    ]
                                ) # End results-content
                            ] # End children of Loading
                        ) # End Loading
                    ] # End children of results-card
                ), # End Results Card

                # --- SEARCHABLE COMMENT: Data Store ---
                # Hidden component to store intermediate data (like audio arrays) in the browser session
                dcc.Store(id='audio-data-store'),
            ] # End children of main content container
        ) # End main content container
    ] # End children of app layout
) # End app layout


# ============================================
# --- SEARCHABLE COMMENT: Callbacks ---
# Functions that define the application's interactivity.
# They trigger based on user input (e.g., button clicks, slider changes)
# and update the output components (e.g., plots, text).
# ============================================

# --- SEARCHABLE COMMENT: Enable Analyze Button Callback ---
@app.callback(
    Output("classify-btn", "disabled"), # Output: Whether the button is disabled
    Output("classify-btn", "style"),    # Output: Button's style (to change appearance when disabled)
    Input("upload-audio", "contents"),  # Input: Triggered when file content is uploaded
    State("classify-btn", "disabled"), # State: Current disabled state (to prevent unnecessary updates)
)
def update_button_state(contents, current_disabled_state):
    """Enable/disable the 'Analyze Original Audio' button based on file upload."""
    is_disabled = not contents # Button is disabled if there are no contents
    # Only update if the disabled state actually changes
    if is_disabled == current_disabled_state:
        raise PreventUpdate

    new_style = {**BUTTON_STYLE_PRIMARY} # Start with the default primary style
    if is_disabled:
        new_style.update(BUTTON_DISABLED_STYLE) # Apply disabled style if needed

    return is_disabled, new_style

# --- SEARCHABLE COMMENT: File Upload Callback ---
@app.callback(
    # --- Outputs for Upload Callback ---
    Output("results-card", "style", allow_duplicate=True),             # Show/hide the results card
    Output("file-name", "children"),                                  # Update file name display
    Output("waveform-plot", "figure", allow_duplicate=True),          # Update waveform plot (initial preview)
    Output("audio-player", "src"),                                     # Set source for original audio player
    Output("audio-data-store", "data"),                               # Store audio data and sample rate
    Output("classification-result", "children", allow_duplicate=True),# Initial text in classification area
    Output("show-sampling-btn", "style", allow_duplicate=True),       # Hide sampling button initially
    Output("sampling-controls", "style", allow_duplicate=True),       # Hide sampling controls initially
    Output("upload-error-output", "children"),                        # Display errors during file reading
    Output("sampled-classification-result", "children", allow_duplicate=True), # Reset sampled result text
    Output("sampled-classification-result", "style", allow_duplicate=True),    # Reset sampled result style
    Output("sampled-audio-player", "src", allow_duplicate=True),      # Reset sampled player source
    Output("sampled-audio-player", "style", allow_duplicate=True),     # Hide sampled player
    # --- Input for Upload Callback ---
    Input("upload-audio", "contents"),                                # Triggered by file upload
    # --- State for Upload Callback ---
    State("upload-audio", "filename"),                                # Get the name of the uploaded file
    prevent_initial_call=True, # Don't run when the app first loads
)
def display_uploaded_audio(contents, filename):
    """
    Handles file uploads. Reads the audio data, displays an initial waveform preview,
    enables the original audio player, stores the data, and resets dependent components.
    Does NOT perform AI analysis yet.
    """
    if not contents:
        # Should not happen if button state callback works, but good practice
        raise PreventUpdate

    # --- SEARCHABLE COMMENT: Base64 Decoding ---
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    err_msg = "" # Initialize error message variable

    try:
        # --- SEARCHABLE COMMENT: Audio File Reading ---
        # Use soundfile to read audio data from the decoded bytes
        audio_data, sr = sf.read(io.BytesIO(decoded))
        # --- SEARCHABLE COMMENT: Mono Conversion ---
        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1) # Convert stereo to mono

        # --- SEARCHABLE COMMENT: Audio Normalization ---
        # Ensure audio is float32 and normalized roughly to [-1, 1] range
        if not np.issubdtype(audio_data.dtype, np.floating):
            max_val = np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else 32767.0
            if max_val != 0: audio_data = audio_data.astype(np.float32) / max_val
            else: audio_data = audio_data.astype(np.float32) # Avoid division by zero for silence
        elif np.max(np.abs(audio_data)) > 1.5: # Handle float data already loaded but possibly out of range
             print("Warning: Float audio data exceeds [-1, 1] range. Normalizing.")
             max_abs = np.max(np.abs(audio_data))
             if max_abs > 0: audio_data /= max_abs

    except Exception as e:
        # --- SEARCHABLE COMMENT: File Read Error Handling ---
        err_msg = f"⚠️ Error reading audio file '{filename}': Please ensure it's a valid audio format. ({e})"
        print(err_msg) # Log detailed error
        # Update UI to show error and hide results
        return ({'display': 'none'}, "", go.Figure(), None, None, "",
                {'display': 'none'}, {'display': 'none'}, err_msg, # Display error message
                "", no_update, "", {'display':'none'}) # Reset other outputs

    # --- SEARCHABLE COMMENT: Initial Plot Creation ---
    fig = create_initial_figure(audio_data) # Generate the waveform preview plot
    # --- SEARCHABLE COMMENT: Storing Audio Data ---
    # Store audio data (as list), sample rate, and filename in dcc.Store
    store_data = {'original_audio': audio_data.tolist(), 'original_sr': sr, 'filename': filename}
    # Initial message for the classification result area
    initial_text = html.Span("File loaded. Click 'Analyze Original Audio' to classify.", style={'color': SUBTLE_TEXT_COLOR, 'fontStyle': 'italic', 'fontSize': '16px'})

    # Default style for the sampled result area (to reset it visually)
    sampled_result_style = {
        "fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
        "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
        "border": f"1px dashed {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
        "color": SUBTLE_TEXT_COLOR
    }

    # --- SEARCHABLE COMMENT: Update UI after Upload ---
    # Return values to update all the specified Output components
    return (
        {**CARD_STYLE, 'display': 'block'}, # Show results card
        f"File: {filename}",               # Display filename
        fig,                               # Display initial waveform plot
        contents,                          # Set src for original audio player
        store_data,                        # Save data to dcc.Store
        initial_text,                      # Show initial message in classification area
        {'display': 'none'},               # Keep sampling button hidden
        {'display': 'none'},               # Keep sampling controls hidden
        "",                                # Clear any previous upload error
        "Prediction result for sampled audio will appear here.", # Reset sampled result text
        sampled_result_style,              # Reset sampled result style
        "",                                # Clear sampled audio player source
        {'display':'none'}                 # Hide sampled audio player
    )


# --- SEARCHABLE COMMENT: Analyze Original Audio Callback ---
@app.callback(
    # --- Outputs for Analyze Callback ---
    Output("classification-result", "children", allow_duplicate=True), # Display prediction result
    Output("classification-result", "style", allow_duplicate=True),    # Style prediction result (e.g., color)
    Output("show-sampling-btn", "style", allow_duplicate=True),       # Show the "Explore Sampling" button
    Output("sampling-freq-slider", "max"),                            # Set slider's max value based on original SR
    Output("sampling-freq-slider", "value"),                           # Set initial slider value
    Output("sampling-freq-slider", "marks"),                           # Set marks (like Nyquist rate) on slider
    Output("nyquist-info", "children"),                               # Display calculated Nyquist info
    Output("audio-data-store", "data", allow_duplicate=True),         # Update stored data with calculated max_freq
    # --- Input for Analyze Callback ---
    Input("classify-btn", "n_clicks"),                                # Triggered by clicking the analyze button
    # --- State for Analyze Callback ---
    State("audio-data-store", "data"),                                # Get stored audio data
    prevent_initial_call=True,
)
def analyze_original_audio(n_clicks, stored_data):
    """
    Runs the AI model classification on the *original* uploaded audio data.
    Also calculates the signal's estimated maximum frequency and Nyquist rate.
    Updates the UI with the classification result and makes the sampling controls available.
    """
    # Prevent running if button wasn't clicked or data isn't ready
    if not n_clicks or not stored_data or 'original_audio' not in stored_data:
        raise PreventUpdate

    try:
        # --- SEARCHABLE COMMENT: Retrieving Stored Data ---
        audio_data = np.array(stored_data['original_audio'], dtype=np.float32) # Convert list back to numpy array
        sr = stored_data['original_sr']
        filename = stored_data.get('filename', 'Unknown file')

        # --- SEARCHABLE COMMENT: AI Prediction (Original Audio) ---
        print(f"Analyzing original audio: {filename} ({len(audio_data)} samples @ {sr} Hz)")
        processor, model, device, load_err = ensure_model() # Load model if needed
        classification = "" # Initialize classification result variable
        top_pred_label = "N/A" # Default label if prediction fails

        if load_err:
            classification = html.Span(f"⚠️ Model Load Error: {load_err}", style={'color': ERROR_COLOR})
            top_pred_label = "Error"
        else:
            # --- SEARCHABLE COMMENT: Calling Prediction Function ---
            preds = predict_with_local_model(processor, model, device, audio_data.copy(), sr) # Pass a copy

            # --- SEARCHABLE COMMENT: Processing Prediction Results ---
            # Handle various outcomes from the prediction function
            if preds and isinstance(preds, list) and len(preds) > 0:
                if any("error" in p for p in preds): # Check if any chunk resulted in an error
                    error_msg = next((p['error'] for p in preds if 'error' in p), 'Unknown prediction error')
                    classification = html.Span(f"⚠️ Prediction Error: {error_msg}", style={'color': ERROR_COLOR})
                    top_pred_label = "Error"
                elif "label" in preds[0] and "score" in preds[0]: # Check if we got valid predictions
                    valid_preds = [p for p in preds if "error" not in p and "label" in p]
                    if valid_preds:
                        # Find the prediction with the highest confidence score across all chunks
                        top_pred = max(valid_preds, key=lambda x: x.get('score', 0))
                        # --- SEARCHABLE COMMENT: Formatting Classification Output ---
                        pred_color = PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else ERROR_COLOR
                        classification = html.Div([
                            html.Span("Original Prediction: "),
                            html.Strong(f"{top_pred['label']}", style={'color': pred_color}),
                            html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':SUBTLE_TEXT_COLOR})
                        ])
                        top_pred_label = top_pred['label']
                    else:
                        classification = html.Span("⚠️ Prediction error occurred in all chunks.", style={'color': WARNING_COLOR})
                        top_pred_label = "Error"
                elif "message" in preds[0]: # Handle specific messages like "Audio too short"
                     classification = html.Span(f"⚠️ {preds[0]['message']}", style={'color': WARNING_COLOR})
                     top_pred_label = "N/A"
                else: # Unexpected format
                    classification = html.Span("⚠️ Prediction format unexpected.", style={'color': ERROR_COLOR})
                    top_pred_label = "Error"
            else: # No predictions returned
                 classification = html.Span("⚠️ No prediction returned (audio might be too short or model error).", style={'color': WARNING_COLOR})
                 top_pred_label = "N/A"

        # --- SEARCHABLE COMMENT: Nyquist Rate Calculation ---
        # Estimate the maximum frequency (f_max) in the signal using FFT energy percentile.
        n = len(audio_data)
        max_freq = sr / 4 # Conservative default (half Nyquist of original SR)
        if n > 100: # Only run FFT if signal is reasonably long
            try:
                # --- SEARCHABLE COMMENT: FFT Calculation ---
                yf = np.fft.rfft(audio_data) # Real FFT
                xf = np.fft.rfftfreq(n, 1 / sr) # Frequencies corresponding to FFT output

                if len(yf) > 1: # Ensure we have frequency components besides DC
                    energy = np.abs(yf[1:]) ** 2 # Calculate energy (magnitude squared), excluding DC
                    cumulative_energy = np.cumsum(energy)
                    total_energy = cumulative_energy[-1] if cumulative_energy.size > 0 else 0

                    if total_energy > 1e-9: # Avoid calculations for pure silence
                        # Find the frequency below which 99% of the energy lies
                        freq_99_percentile_idx = np.where(cumulative_energy >= total_energy * 0.99)[0][0]
                        # Index needs +1 offset (because we excluded DC), map back to xf frequencies
                        actual_idx = freq_99_percentile_idx + 1
                        if actual_idx < len(xf):
                             max_freq = xf[actual_idx]
                        elif len(xf) > 1:
                            max_freq = xf[-1] # Fallback to max calculated frequency if index out of bounds
                    elif len(xf) > 1:
                         # Very low energy, use max frequency as fallback
                         max_freq = xf[-1]
                elif len(xf) > 0 : # Only DC component? Use max freq as fallback
                     max_freq = xf[-1] if len(xf) > 0 else sr / 4

            except (IndexError, TypeError, ValueError) as fft_err:
                # --- SEARCHABLE COMMENT: FFT Error Handling ---
                print(f"Warning: FFT analysis for max_freq failed: {fft_err}")
                # Fallback if FFT fails
                if n > 0 and 'xf' in locals() and xf is not None and len(xf) > 0: max_freq = xf[-1]


        max_freq = max(500, max_freq) # Ensure a minimum reasonable f_max (e.g., 500 Hz)
        nyquist_rate = 2 * max_freq # Calculate the Nyquist rate

        # --- SEARCHABLE COMMENT: Slider Configuration ---
        # Set initial slider value based on calculated Nyquist or original SR
        slider_default = max(1000, np.ceil(nyquist_rate / 100.0) * 100) # Round up to nearest 100Hz
        initial_sampling_rate = int(min(sr + 500, slider_default + 500, sr)) # Start <= original SR, slightly above Nyquist if possible

        # --- SEARCHABLE COMMENT: Dynamic UI Styling (Classification Result) ---
        result_style = {
            "fontSize": "22px", "fontWeight": "700", "textAlign": "center", "marginBottom": "20px",
            "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
            "border": "2px solid" # Border color set below
        }
        # Color-code the result box based on the prediction
        if 'drone' in top_pred_label.lower():
            result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': ERROR_COLOR, 'color': ERROR_COLOR}) # Red style
        elif top_pred_label not in ["N/A", "Error"]:
            result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': PRIMARY_COLOR, 'color': PRIMARY_COLOR}) # Blue style
        else: # Error or N/A
            result_style.update({'backgroundColor': BACKGROUND_COLOR, 'borderColor': BORDER_COLOR, 'color': TEXT_COLOR}) # Default style

        # --- SEARCHABLE COMMENT: Update Stored Data ---
        stored_data['max_freq'] = max_freq # Save calculated max_freq in the dcc.Store

        # --- SEARCHABLE COMMENT: Slider Marks Configuration ---
        slider_max = max(sr + 1000, 48000) # Ensure slider range is adequate
        # Define marks for slider (min, Nyquist, max)
        slider_marks = {
             500: {'label': '0.5 kHz', 'style': {'fontSize': '11px'}},
             int(nyquist_rate): {'label': f'Nyquist ({nyquist_rate / 1000:.1f} kHz)', 'style': {'color': PRIMARY_COLOR, 'fontWeight': 'bold', 'fontSize': '11px', 'whiteSpace':'nowrap'}},
             int(slider_max): {'label': f'{int(slider_max/1000)} kHz', 'style': {'fontSize': '11px'}}
        }
        # Remove min/max marks if they overlap significantly with the Nyquist mark
        if abs(500 - nyquist_rate) < slider_max * 0.05: del slider_marks[500]
        if abs(slider_max - nyquist_rate) < slider_max * 0.05: del slider_marks[int(slider_max)]

        # --- SEARCHABLE COMMENT: Nyquist Info Text ---
        # Format the text to display SR, f_max, and Nyquist rate
        nyquist_text = [
            html.Div([html.Strong("Original SR:"), html.Span(f" {sr} Hz")]),
            html.Div([html.Strong("Est. Max Freq:"), html.Span(f" {max_freq:.0f} Hz")]),
            html.Div([html.Strong("Nyquist Rate:"), html.Span(f" {nyquist_rate:.0f} Hz")], style={'fontWeight': 'bold', 'color': PRIMARY_COLOR})
        ]

        # --- SEARCHABLE COMMENT: Show Sampling Button ---
        # Make the "Explore Sampling" button visible
        show_sampling_btn_style = {**BUTTON_STYLE_SECONDARY, 'width': '100%', 'marginTop': '15px', 'display': 'block'}

        # --- SEARCHABLE COMMENT: Return Values for Analyze Callback ---
        return (classification, result_style, show_sampling_btn_style, slider_max, initial_sampling_rate,
                slider_marks, nyquist_text, stored_data)

    except Exception as e:
        # --- SEARCHABLE COMMENT: Analysis Error Handling ---
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc() # Print full stack trace for debugging
        error_text = html.Div([html.Strong("⚠️ Analysis Error: "), f"{e}"])
        error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                       "padding": "15px", "borderRadius": "8px", 'backgroundColor': ERROR_COLOR,
                       "border": f"1px solid {ERROR_COLOR}"}
        # Return error message, hide sampling button, update minimal outputs
        return error_text, error_style, {'display':'none'}, no_update, no_update, no_update, no_update, no_update


# --- SEARCHABLE COMMENT: Toggle Sampling Controls Callback ---
@app.callback(
    Output("sampling-controls", "style", allow_duplicate=True), # Output: Style (display property)
    Input("show-sampling-btn", "n_clicks"),                     # Input: Triggered by button click
    State("sampling-controls", "style"),                        # State: Current style to check visibility
    prevent_initial_call=True,
)
def toggle_sampling_controls(n_clicks, current_style):
    """Toggles the visibility of the sampling controls div when the button is clicked."""
    if n_clicks % 2 == 1: # If odd number of clicks (1, 3, 5...)
        return {'display': 'block', 'marginTop': '25px'} # Show the controls
    else: # If even number of clicks (0, 2, 4...)
        return {'display': 'none', 'marginTop': '25px'} # Hide the controls


# --- SEARCHABLE COMMENT: Update Waveform on Slider Callback ---
# --- SEARCHABLE COMMENT: Aliasing Plot Update ---
@app.callback(
    Output("waveform-plot", "figure", allow_duplicate=True), # Output: The updated plot
    Output("playback-warning", "children"),                  # Output: Warning text (aliasing, low SR)
    Input("sampling-freq-slider", "value"),                 # Input: Triggered by slider value change
    State("audio-data-store", "data"),                      # State: Get original audio data
    State("sampling-controls", "style"),                    # State: Check if sampling controls are visible
    prevent_initial_call=True,
)
def update_waveform_on_sample(new_sr, stored_data, sampling_style):
    """
    Updates the waveform plot dynamically as the sampling frequency slider is moved.
    Only updates if the sampling controls section is currently visible.
    Calls `create_resampled_figure` to generate the plot comparing original and sampled points.
    """
    # --- SEARCHABLE COMMENT: Prevent Plot Update if Hidden ---
    # Only update the plot if the sampling section is visible to avoid unnecessary computation
    if not stored_data or 'max_freq' not in stored_data or sampling_style.get('display') == 'none':
        raise PreventUpdate # Stop the callback if controls aren't visible

    try:
        audio_data = np.array(stored_data['original_audio'], dtype=np.float32)
        # --- SEARCHABLE COMMENT: Calling Resampled Plot Function ---
        # Generate the figure showing original vs sampled points
        fig = create_resampled_figure(audio_data, stored_data['original_sr'], new_sr, stored_data['max_freq'])

        # --- SEARCHABLE COMMENT: Aliasing/Playback Warning ---
        warning_text = ""
        is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2
        if new_sr < 3000: # Warn about potential playback issues at very low rates
            warning_text = html.Span("⚠️ Playback may fail or be distorted below ~3 kHz.", style={'color':WARNING_COLOR, 'fontWeight':'500'})
        elif is_aliasing: # Warn about likely aliasing
             warning_text = html.Span(["📉 Aliasing likely: ", html.B(f"Fs ({new_sr} Hz)"), f" < Nyquist ({stored_data.get('max_freq', 0) * 2:.0f} Hz)."], style={'color':WARNING_COLOR, 'fontWeight':'500'})

        return fig, warning_text # Return the new figure and warning text
    except Exception as e:
        print(f"Error updating waveform plot: {e}")
        # Return no_update for figure (keep the old one) and display error message
        return no_update, html.Span(f"Error plotting: {e}", style={'color':ERROR_COLOR})


# --- SEARCHABLE COMMENT: Play Sampled Audio Callback ---
@app.callback(
    Output("sampled-audio-player", "src", allow_duplicate=True),   # Output: Data URI for the audio player
    Output("sampled-audio-player", "style", allow_duplicate=True), # Output: Style to show/hide the player
    Input("play-sampled-btn", "n_clicks"),                         # Input: Triggered by button click
    State("audio-data-store", "data"),                           # State: Get original audio data
    State("sampling-freq-slider", "value"),                      # State: Get target sample rate
    prevent_initial_call=True,
)
def play_resampled_audio(n_clicks, stored_data, new_sr):
    """
    Resamples the *entire* original audio to the currently selected `new_sr`
    using the decimation/interpolation method. Encodes it as a WAV data URI
    and sends it to the `sampled-audio-player` component.
    """
    if not n_clicks or not stored_data:
        return no_update, {'display': 'none'} # Hide player if not triggered or no data

    try:
        original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
        original_sr = stored_data['original_sr']

        print(f"Resampling audio from {original_sr} Hz to {new_sr} Hz for playback (using decimation/sinc).")
        # --- SEARCHABLE COMMENT: Calling Playback Resampling Function ---
        # Use the decimation/interpolation resampler for playback
        processed_audio = resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

        # --- SEARCHABLE COMMENT: WAV Encoding ---
        buffer = io.BytesIO() # Create in-memory buffer
        if len(processed_audio) == 0:
            print("Warning: Resampled audio for playback is empty.")
            return "", {'display': 'none'} # Return empty src, hide player

        # Scale float audio (assumed in [-X, X]) to int16 range [-32767, 32767] for WAV format
        max_abs_val = np.max(np.abs(processed_audio)) if len(processed_audio) > 0 else 0
        processed_audio_int = np.zeros_like(processed_audio, dtype=np.int16) # Initialize with zeros
        if max_abs_val > 0: # Avoid division by zero for silence
            scale_factor = 32767.0 / max_abs_val
            processed_audio_int = np.int16(np.clip(processed_audio * scale_factor, -32767, 32767))


        # Write WAV data to buffer
        sf.write(buffer, processed_audio_int, int(new_sr), format='WAV', subtype='PCM_16')
        buffer.seek(0) # Rewind buffer to the beginning

        # --- SEARCHABLE COMMENT: Data URI Creation ---
        # Encode WAV data as Base64 and create data URI
        encoded_sound = base64.b64encode(buffer.read()).decode()
        data_uri = f"data:audio/wav;base64,{encoded_sound}"

        # Return data URI and style to show the player
        return data_uri, {"width": "100%", "marginTop": "15px", 'display': 'block'}
    except Exception as e:
        # --- SEARCHABLE COMMENT: Playback Error Handling ---
        print(f"Error resampling audio for playback: {e}")
        import traceback
        traceback.print_exc() # Print full stack trace for debugging
        # Return empty source and hide player on error
        return "", {'display': 'none'}


# --- SEARCHABLE COMMENT: Predict Sampled Audio Callback ---
@app.callback(
    Output("sampled-classification-result", "children", allow_duplicate=True), # Output: Prediction text/HTML
    Output("sampled-classification-result", "style", allow_duplicate=True),    # Output: Style of the result box
    Input("predict-sampled-btn", "n_clicks"),                                 # Input: Triggered by button click
    State("audio-data-store", "data"),                                       # State: Get original audio data
    State("sampling-freq-slider", "value"),                                  # State: Get target sample rate
    prevent_initial_call=True,
)
def predict_sampled_audio(n_clicks, stored_data, new_sr):
    """
    Resamples the original audio to the currently selected `new_sr` (using decimation/interpolation).
    Then, runs the AI classification model on this *resampled* audio.
    Displays the prediction result and highlights potential issues due to aliasing.
    """
    # Default message and style before prediction runs
    default_style = {
        "fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
        "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
        "border": f"1px dashed {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
        "color": SUBTLE_TEXT_COLOR
    }
    if not n_clicks or not stored_data or 'original_audio' not in stored_data:
        return "Prediction result for sampled audio will appear here.", default_style

    # --- SEARCHABLE COMMENT: Loading State for Sampled Prediction ---
    # Display a loading message immediately
    loading_text = "🧠 Analyzing sampled audio..."
    loading_style = {"fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
                    "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                    "border": f"1px solid {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
                    "color": SUBTLE_TEXT_COLOR}

    try:
        original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
        original_sr = stored_data['original_sr']
        filename = stored_data.get('filename', 'Audio File')

        print(f"Resampling {filename} from {original_sr} Hz to {new_sr} Hz for prediction (using decimation/sinc).")
        # --- SEARCHABLE COMMENT: Resampling for Sampled Prediction ---
        # Resample using the decimation/interpolation method (no anti-aliasing filter)
        resampled_audio_for_pred = resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

        # --- SEARCHABLE COMMENT: AI Prediction (Sampled Audio) ---
        print(f"Predicting on resampled audio ({len(resampled_audio_for_pred)} samples @ {new_sr} Hz)...")
        processor, model, device, load_err = ensure_model()
        classification_html = "" # Initialize variable for result HTML
        top_pred_label = "N/A"

        if load_err:
            classification_html = html.Span(f"⚠️ Model Load Error: {load_err}", style={'color': ERROR_COLOR})
            top_pred_label = "Error"
        else:
            # --- SEARCHABLE COMMENT: Calling Prediction on Sampled Data ---
            # IMPORTANT: Pass the RESAMPLED audio and its NEW sample rate to the model
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
                        pred_color = PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else ERROR_COLOR
                        classification_html = html.Div([
                            html.Span(f"Sampled Prediction ({new_sr} Hz): "),
                            html.Strong(f"{top_pred['label']}", style={'color': pred_color}),
                            html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':SUBTLE_TEXT_COLOR})
                        ])
                        top_pred_label = top_pred['label']
                    else:
                        classification_html = html.Span("⚠️ Prediction valid but failed.", style={'color': WARNING_COLOR})
                        top_pred_label = "Error"
                 elif "message" in preds[0]: # Handle "Audio too short" etc.
                     classification_html = html.Span(f"⚠️ {preds[0]['message']} (at {new_sr} Hz)", style={'color': WARNING_COLOR})
                     top_pred_label = "N/A"
                 else: # Unexpected format
                    classification_html = html.Span("⚠️ Prediction format unexpected.", style={'color': ERROR_COLOR})
                    top_pred_label = "Error"
            else: # No predictions returned
                 classification_html = html.Span(f"⚠️ No prediction (audio might be too short after resampling to {new_sr} Hz).", style={'color': WARNING_COLOR})
                 top_pred_label = "N/A"

        # --- SEARCHABLE COMMENT: Dynamic UI Styling (Sampled Classification Result) ---
        result_style = {
            "fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px",
            "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
            "border": "2px solid", 'minHeight': '60px' # Base style
        }

        # Check for aliasing based on stored max_freq
        is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2

        # Update background/border/text color based on prediction label
        if 'drone' in top_pred_label.lower():
            result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': ERROR_COLOR, 'color': ERROR_COLOR})
        elif top_pred_label not in ["N/A", "Error"]:
             result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': PRIMARY_COLOR, 'color': PRIMARY_COLOR})
        else: # Error or N/A
             result_style.update({'backgroundColor': BACKGROUND_COLOR, 'borderColor': BORDER_COLOR, 'color': TEXT_COLOR})

        # --- SEARCHABLE COMMENT: Aliasing Note ---
        # Add a note if aliasing was likely during the resampling for this prediction
        aliasing_note_html = ""
        if is_aliasing:
             aliasing_note_html = html.P(
                 "Note: Prediction used audio sampled below Nyquist rate (aliasing likely occurred).",
                 style={ 'fontSize': '12px', 'color': SUBTLE_TEXT_COLOR, 'marginTop': '10px',
                         'marginBottom': '0', 'fontStyle': 'italic' }
             )

        # Return the formatted classification result and the calculated style
        return html.Div([classification_html, aliasing_note_html]), result_style

    except Exception as e:
        # --- SEARCHABLE COMMENT: Sampled Prediction Error Handling ---
        print(f"Error during sampled prediction: {e}")
        import traceback
        traceback.print_exc() # Log full error trace
        error_text = html.Div([html.Strong("⚠️ Sampled Prediction Error: "), f"{e}"])
        # Consistent error styling for the result box
        error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                       "padding": "15px", "borderRadius": "8px", 'backgroundColor': ERROR_COLOR,
                       "border": f"1px solid {ERROR_COLOR}", 'minHeight': '60px', "marginTop": "25px"}
        return error_text, error_style


# ============================================
# --- SEARCHABLE COMMENT: Main Execution Block ---
# Runs when the script is executed directly.
# ============================================
if __name__ == "__main__":
    # --- SEARCHABLE COMMENT: Preload Model ---
    # Attempt to load the model on startup to potentially speed up the first analysis
    ensure_model()
    # --- SEARCHABLE COMMENT: Run Dash Server ---
    # Start the Dash development server
    # debug=True enables hot-reloading and error messages in the browser
    # port=8051 specifies the port number (default is 8050)
    app.run(debug=True, port=8051)

