import os
import io
import base64
import warnings
import dash
import dash_bootstrap_components as dbc # Import Bootstrap components
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf # Used for robust audio file reading
import torch
from scipy.io import wavfile # Used for writing WAV data for playback

# Suppress warnings (use cautiously)
warnings.filterwarnings('ignore')

# --- Model Loading ---

# Try loading the Keras model for audio anti-aliasing/reconstruction
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

# Try loading the Hugging Face Transformers model for gender detection
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

# Function to analyze audio data (waveform, FFT, etc.)
def analyze_audio(y, sr):
    """Calculates basic audio features including FFT magnitude spectrum."""
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    magnitude = np.abs(fft)
    # Normalize magnitude
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)
    return {
        'waveform': y.tolist(), 'sample_rate': sr, 'duration': len(y) / sr,
        'frequencies': freqs.tolist(), 'magnitude': magnitude.tolist(),
        'nyquist_freq': sr / 2
    }


# Function to apply the AI model for audio reconstruction (upsampling + anti-aliasing)
def apply_model_reconstruction(y_input, sr_input):
    """Applies the AI model for reconstruction or falls back to standard resampling."""
    if not MODEL_LOADED:
        print("Reconstruction model not available. Performing standard upsampling.")
        # Fallback: standard Librosa resampling
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=16000)
    try:
        model_sr, model_len = 16000, 48000 # Model's expected sample rate and input length
        # Initial upsampling to the model's target rate
        y_upsampled = librosa.resample(y_input, orig_sr=sr_input, target_sr=model_sr)
        reconstructed_chunks = []
        # Process audio in chunks compatible with the model's input size
        for i in range(0, len(y_upsampled), model_len):
            chunk = y_upsampled[i:i + model_len]
            len_chunk = len(chunk)
            # Pad the chunk if it's shorter than the model's expected length
            if len_chunk < model_len:
                chunk = np.pad(chunk, (0, model_len - len_chunk), 'constant')
            # Prepare chunk for the model (add batch and channel dimensions)
            model_input = chunk[np.newaxis, ..., np.newaxis]
            # Get prediction from the anti-aliasing model
            pred_chunk = np.squeeze(ANTI_ALIASING_MODEL.predict(model_input, verbose=0))
            # Trim padding if it was added
            if len_chunk < model_len:
                pred_chunk = pred_chunk[:len_chunk]
            reconstructed_chunks.append(pred_chunk)
        # Combine the processed chunks
        return np.concatenate(reconstructed_chunks) if reconstructed_chunks else np.array([])
    except Exception as e:
        print(f"❌ Error during model reconstruction: {e}")
        # Fallback on error during prediction
        return librosa.resample(y_input, orig_sr=sr_input, target_sr=16000)


# Function to create base64 encoded WAV data URL for HTML audio playback
def create_audio_data(audio, sr):
    """Makes any audio playable in browser, handling very low sample rates."""
    audio_array = np.array(audio) # Ensure it's a numpy array
    if len(audio_array) == 0:
        return ""

    playback_sr = 22050  # A browser-safe sample rate

    # If the sample rate is too low for browsers, upsample without interpolation
    # This prevents pitch changes for extremely low sample rates.
    if sr < 3000:
        duration_sec = len(audio_array) / sr
        new_len = int(duration_sec * playback_sr)
        if new_len == 0: return ""
        # Create new indices by stretching the old ones (nearest neighbor)
        indices = np.round(np.linspace(0, len(audio_array) - 1, new_len)).astype(int)
        final_audio = audio_array[indices]
        final_sr = playback_sr
    else:
        final_audio = audio_array
        final_sr = sr

    # Clip audio to [-1.0, 1.0], convert to 16-bit PCM
    final_audio = np.clip(final_audio, -1.0, 1.0)
    wav_data = (final_audio * 32767).astype(np.int16)
    # Write WAV data to a byte buffer
    buf = io.BytesIO()
    wavfile.write(buf, final_sr, wav_data)
    wav_bytes = buf.getvalue()
    # Encode as Base64 data URL
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"


# Function to predict gender using the loaded Transformers model
def predict_gender(y, sr):
    """Predicts gender from audio using the Hugging Face model."""
    if not GENDER_MODEL_LOADED:
        return "Gender detection model not available."
    try:
        # Resample to 16kHz if necessary (model's expected sample rate)
        if sr != 16000:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_resampled = y

        # Extract features using the model's feature extractor
        inputs = gender_feature_extractor(y_resampled, sampling_rate=16000, return_tensors="pt")
        # Make prediction
        with torch.no_grad():
            logits = gender_model(inputs.input_values).logits

        # Get the predicted label
        pred_id = torch.argmax(logits, dim=-1).item()
        predicted_label = gender_model.config.id2label[pred_id].capitalize()
        return f"Predicted Gender: {predicted_label}"
    except Exception as e:
        print(f"❌ Error during gender prediction: {e}")
        return "Could not predict gender from audio."


# --- UI Component Creation Functions ---

# Function to create the HTML audio player elements using Bootstrap Columns
def create_audio_players(original_data, downsampled_data, reconstructed_data):
    """Generates dbc.Row containing audio players for different audio versions."""
    cols = []
    # Original Audio Player
    cols.append(dbc.Col([
        html.H5("1. Original Audio", className="text-primary"),
        html.Audio(controls=True, style={'width': '100%'},
                   src=create_audio_data(original_data['waveform'], original_data['sample_rate']))
    ], width=12, lg=4, className="mb-3 mb-lg-0")) # Added bottom margin for small screens

    # Downsampled Audio Player
    if downsampled_data:
        cols.append(dbc.Col([
            html.H5("2. Downsampled (Aliased) Audio", className="text-danger"),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(downsampled_data['waveform'], downsampled_data['sample_rate']))
        ], width=12, lg=4, className="mb-3 mb-lg-0")) # Added bottom margin for small screens

    # Reconstructed Audio Player
    if reconstructed_data:
        reconstruction_title = "3. AI Reconstructed Audio" if MODEL_LOADED else "3. Upsampled Audio (Fallback)"
        reconstruction_color = 'text-success' if MODEL_LOADED else 'text-warning'
        cols.append(dbc.Col([
            html.H5(reconstruction_title, className=reconstruction_color),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(reconstructed_data['waveform'],
                                             reconstructed_data['sample_rate']))
        ], width=12, lg=4))
    # Return players wrapped in a Bootstrap Row
    return dbc.Row(cols, className="mt-3 g-3") # Added gutter class g-3 for spacing


# --- App Layout ---
# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Audio Analysis Dashboard"
server = app.server # Expose server for deployment

# Define the main layout of the application using Bootstrap components
app.layout = dbc.Container([ # Changed fluid=True to fluid=False (default)
    # Header Section
    dbc.Row(
        dbc.Col(
            html.Div([
                html.H1("🎧 Interactive Audio Sampling & Analysis", className="text-center mb-3"),
                html.P("Upload an audio file to downsample, reconstruct, and test the effects on an AI model.",
                       className="text-center text-muted"),
            ], className="bg-light p-4 rounded mb-4 shadow-sm")
        )
    ),

    # Hidden stores to hold audio data between callbacks
    dcc.Store(id='original-audio-data'),
    dcc.Store(id='downsampled-audio-data'),
    dcc.Store(id='reconstructed-audio-data'),

    # Section 1: Audio Upload Card
    dbc.Card(
        dbc.CardBody([
            html.H3("1. Load Audio", className="card-title mb-3"),
            dcc.Upload(id='upload-audio', children=html.Div(['Drag and Drop or ', html.A('Select Audio File')]),
                       style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                              'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'},
                       className="mb-3"),
            html.Div(id='upload-status', className="mt-3") # Displays status messages after upload
        ]), className="mb-4 shadow-sm"
    ),

    # Section 2: Audio Processing Controls Card
    dbc.Card(
        dbc.CardBody([
            html.H3("2. Process Audio", className="card-title mb-3"),
            html.Div([
                html.H4("Downsample Frequency", className="text-danger mb-2"),
                # Slider to select the target downsampling frequency
                dcc.Slider(id='target-sr-slider', min=500, max=16000, step=500, value=8000,
                           marks={
                               500: '0.5k', # Mark for 500 Hz
                               **{i: f'{i // 1000}k' for i in range(2000, 17000, 2000)} # Marks from 2k to 16k
                           },
                           tooltip={"placement": "bottom", "always_visible": True},
                           className="mb-3"), # Added bottom margin
            ], className="mb-4"),
            html.Div([
                # Button to trigger AI reconstruction
                dbc.Button('Apply AI Reconstruction', id='model-reconstruct-btn', n_clicks=0,
                           color="success", className="me-3"),
                # Displays status of loaded AI models
                html.Span(id='model-status', style={'fontWeight': 'bold'})
            ], className="text-center mt-3"), # Added top margin
        ]), className="mb-4 shadow-sm"
    ),

    # Section 3: AI Model Testing Card
    dbc.Card(
        dbc.CardBody([
            html.H3("3. Test Audio on Model", className="card-title mb-3"),
            html.P("Select an audio signal and test its effect on the model's prediction.", className="text-center mb-3"),
            # Dropdown to select which audio version to test
            dcc.Dropdown(
                id='test-audio-selector',
                options=[
                    {'label': 'Original Audio', 'value': 'original'},
                    {'label': 'Downsampled (Aliased) Audio', 'value': 'downsampled'},
                    {'label': 'AI Reconstructed Audio', 'value': 'reconstructed'}
                ],
                value='original',
                clearable=False,
                className="mb-3"
            ),
            # Button to run the gender detection test
            dbc.Button('Run Gender Detection Test', id='run-test-btn', n_clicks=0,
                       color="primary", className="w-100 mb-3"),
            # Loading indicator and output for the test result
            dcc.Loading(
                id="loading-test-result", type="circle",
                children=[html.Div(id='test-result-output', className="mt-3")] # Wrap output in Div for spinner placement
            )
        ]), className="mb-4 shadow-sm"
    ),

    # Section 4: Listening and Analysis Card
    dbc.Card(
        dbc.CardBody([
            html.H3("4. Listen & Analyze", className="card-title mb-3"),
            # Output area for the initial gender prediction on the uploaded file
            html.Div(id='gender-prediction-output', className="text-center mt-2 mb-4 h4", style={'color': '#0d6efd'}), # Using Bootstrap primary color

            # Controls for graph display
            dbc.Row([
                dbc.Col([
                    html.Label("Select Signals to Display:", className="fw-bold d-block mb-2"), # Make label block
                    # Checkboxes to select which signals
                    dbc.Checklist(
                        id='signal-selector-checklist',
                        options=[
                            {'label': 'Original', 'value': 'original'},
                            {'label': 'Downsampled', 'value': 'downsampled'},
                            {'label': 'Reconstructed', 'value': 'reconstructed'},
                        ],
                        value=['original', 'downsampled', 'reconstructed'], # Default selected values
                        inline=True,
                        className="mb-2 mb-md-0"
                    ),
                ], width=12, md=6),
                dbc.Col([
                    html.Label("Graph View Mode:", className="fw-bold d-block mb-2"), # Make label block
                    # Radio buttons to choose view mode
                    dbc.RadioItems(
                        id='graph-view-selector',
                        options=[
                            {'label': 'Overlap', 'value': 'overlap'},
                            {'label': 'Separate', 'value': 'separate'},
                        ],
                        value='overlap', # Default view mode
                        inline=True
                    ),
                ], width=12, md=6),
            ], className="mb-4 text-center"),

            # Loading indicator for the main output area (players and graphs)
            dcc.Loading(
                id="loading-main-output", type="default",
                children=html.Div([
                    # Container for audio players
                    html.Div(id='audio-players', className="mb-4"), # Added bottom margin
                    # Container for the spectrum plot(s)
                    html.Div(id='spectrum-plot-container', className="mt-4")
                ])
            )
        ]), className="shadow-sm mb-4" # Added bottom margin
    ),
# Set fluid=False (default) for standard container width, add background
], fluid=False, className="py-4 bg-light")


# --- Callbacks ---

# Callback to handle audio file uploads
@app.callback(
    Output('original-audio-data', 'data'),      # Store the analyzed original audio data
    Output('upload-status', 'children'),         # Display upload status message
    Output('model-status', 'children'),          # Display AI model load status
    Output('gender-prediction-output', 'children'),# Display initial gender prediction
    Input('upload-audio', 'contents'),          # Triggered when a file is uploaded
    State('upload-audio', 'filename')           # Get the filename
)
def upload_audio_file(contents, filename):
    """Processes uploaded audio files, analyzes them, and performs initial gender prediction."""
    # Display model loading status using Bootstrap Alerts for better visibility
    recon_alert_color = "success" if MODEL_LOADED else "danger"
    gender_alert_color = "success" if GENDER_MODEL_LOADED else "danger"
    recon_alert = dbc.Badge(f"Reconstruction AI: {'OK' if MODEL_LOADED else 'FAIL'}", color=recon_alert_color, className="me-2")
    gender_alert = dbc.Badge(f"Gender AI: {'OK' if GENDER_MODEL_LOADED else 'FAIL'}", color=gender_alert_color)
    model_status_div = html.Div([recon_alert, gender_alert])


    if contents is None:
        # No file uploaded
        return None, dbc.Alert("Please upload a WAV, MP3, or FLAC file.", color="info"), model_status_div, ""
    try:
        # Decode the base64 encoded file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Use soundfile to read audio data (supports various formats)
        y, sr = sf.read(io.BytesIO(decoded), dtype='float32')
        # Convert to mono if stereo
        if y.ndim > 1: y = y.mean(axis=1)
        # Limit audio length to 20 seconds
        if len(y) > sr * 20: y = y[:sr * 20]

        # Perform initial gender prediction
        gender_text = "Initial Upload: " + predict_gender(y, sr) # Added prefix
        status_msg = dbc.Alert(f"✅ Loaded: {filename} ({len(y) / sr:.2f}s)", color="success")
        # Analyze the audio and store data
        return analyze_audio(y, sr), status_msg, model_status_div, gender_text
    except Exception as e:
        # Handle errors during file processing
        return None, dbc.Alert(f"❌ Error loading file: {e}", color="danger"), model_status_div, ""


# Callback to process audio (downsampling and optional reconstruction)
@app.callback(
    Output('downsampled-audio-data', 'data'),    # Store analyzed downsampled audio
    Output('reconstructed-audio-data', 'data'),  # Store analyzed reconstructed audio
    Input('target-sr-slider', 'value'),         # Triggered by slider change
    Input('model-reconstruct-btn', 'n_clicks'), # Triggered by reconstruction button click
    State('original-audio-data', 'data')        # Get the original audio data
)
def process_audio(target_sr, model_clicks, original_data):
    """Downsamples audio based on slider and optionally applies AI reconstruction."""
    if original_data is None: return None, None # Exit if no original audio
    # Identify which input triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered_id if ctx.triggered else 'No-ID'

    # Get original audio data
    y_original = np.array(original_data['waveform'])
    sr_original = original_data['sample_rate']

    # Always perform downsampling based on the slider value
    print(f"Downsampling to {target_sr} Hz...") # Add print statement
    y_downsampled = librosa.resample(y_original, orig_sr=sr_original, target_sr=target_sr)
    current_downsampled_data = analyze_audio(y_downsampled, target_sr)
    print("Downsampling complete.") # Add print statement

    # Apply AI reconstruction only if the button was clicked
    if triggered_id == 'model-reconstruct-btn' and model_clicks > 0:
        print("Applying AI reconstruction...") # Add print statement
        y_input = np.array(current_downsampled_data['waveform'])
        sr_input = current_downsampled_data['sample_rate']
        y_reconstructed = apply_model_reconstruction(y_input, sr_input)
        reconstructed_data = analyze_audio(y_reconstructed, 16000) # Model outputs at 16kHz
        print("AI reconstruction complete.") # Add print statement
        return current_downsampled_data, reconstructed_data
    # If only slider changed, update downsampled but keep reconstructed as is (or None if not yet created)
    # If button wasn't clicked on initial load (or slider moved after reconstruction), keep existing or None
    print("Slider changed or initial load without button click.") # Add print statement
    # Check if this callback was triggered ONLY by the slider
    if triggered_id == 'target-sr-slider':
         # If slider was the ONLY trigger, don't change reconstruction data
         return current_downsampled_data, dash.no_update
    else:
         # If triggered by button (but clicks <=0) or initial load, set recon to None
         return current_downsampled_data, None


# Callback to update the UI elements (players and plots)
@app.callback(
    Output('audio-players', 'children'),        # Update the audio player section
    Output('spectrum-plot-container', 'children'), # Update the plot section
    Input('original-audio-data', 'data'),       # Trigger on original data change (upload)
    Input('downsampled-audio-data', 'data'),    # Trigger on downsampled data change
    Input('reconstructed-audio-data', 'data'),  # Trigger on reconstructed data change
    Input('graph-view-selector', 'value'),      # Trigger on view mode change (overlap/separate)
    Input('signal-selector-checklist', 'value') # Trigger on signal selection change
)
def update_ui(original_data, downsampled_data, reconstructed_data, view_mode, selected_signals):
    """Updates the audio players and spectrum plots based on available data and user selections."""
    if original_data is None:
        # Show message if no audio is loaded
        return dbc.Alert("Upload an audio file to begin.", color="info"), None

    # Create audio players
    players = create_audio_players(original_data, downsampled_data, reconstructed_data)

    # Prepare data for plotting based on selected signals
    all_signals = {
        'original': {'data': original_data, 'name': 'Original', 'color': '#0d6efd'}, # Bootstrap primary
        'downsampled': {'data': downsampled_data, 'name': 'Downsampled', 'color': '#dc3545'}, # Bootstrap danger
        'reconstructed': {
            'data': reconstructed_data,
            'name': "AI Reconstructed" if MODEL_LOADED else "Upsampled (Fallback)",
            'color': '#198754' if MODEL_LOADED else '#ffc107' # Bootstrap success / warning
        }
    }

    # Filter signals based on user checklist selection and data availability
    signals_to_plot = {key: val for key, val in all_signals.items() if
                       key in selected_signals and val['data'] is not None}

    if not signals_to_plot:
        # No signals selected or available to plot
        return players, dbc.Alert("Select a signal to display its graph.", color="info")

    # Generate plot(s) based on view mode
    if view_mode == 'overlap':
        # Create a single figure with overlapping plots
        fig = go.Figure()
        nyquist_freq = None # Initialize Nyquist freq
        for key, sig_info in signals_to_plot.items():
            data = sig_info['data']
            fig.add_trace(go.Scattergl(x=data['frequencies'], y=data['magnitude'],
                                     name=f"{sig_info['name']} ({data['sample_rate']/1000:.1f} kHz)", # Show kHz
                                     line=dict(color=sig_info['color'], width=2)))
            # Store nyquist frequency if this is the downsampled signal
            if key == 'downsampled':
                 nyquist_freq = data['nyquist_freq']

        # Add Nyquist line if the downsampled signal was plotted
        if nyquist_freq is not None:
            fig.add_vline(x=nyquist_freq, line_dash="dash", line_color="#dc3545", # Bootstrap danger
                          annotation_text=f"Nyquist: {nyquist_freq/1000:.1f} kHz", # Show kHz
                          annotation_position="top left")

        fig.update_layout(
            title='Frequency Spectrum Comparison',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Normalized Magnitude',
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white", # Use a clean template
            xaxis_rangemode='tozero', # Start x-axis at 0
            yaxis_rangemode='tozero'  # Start y-axis at 0
        )
        # Wrap graph in a Card for consistent styling
        graph_card = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
        return players, graph_card

    else:  # 'separate' view
        # Create multiple separate figures in Bootstrap columns
        graph_cols = []
        num_signals = len(signals_to_plot)
        # Adjust column width based on number of plots for better layout
        col_md = 12 // num_signals if num_signals > 0 and num_signals <= 3 else 4
        col_lg = 12 // num_signals if num_signals > 0 and num_signals <= 3 else 4

        for key, sig_info in signals_to_plot.items():
            data = sig_info['data']
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=data['frequencies'], y=data['magnitude'],
                                     name=sig_info['name'], line=dict(color=sig_info['color'])))
            title = f"{sig_info['name']} ({data['sample_rate']/1000:.1f} kHz)" # Show kHz
            nyquist_freq_sep = None
            # Add Nyquist line only to the downsampled plot
            if key == 'downsampled':
                nyquist_freq_sep = data['nyquist_freq']
                fig.add_vline(x=nyquist_freq_sep, line_dash="dash", line_color="#dc3545", # Bootstrap danger
                              annotation_text=f"Nyquist: {nyquist_freq_sep/1000:.1f} kHz", # Show kHz
                              annotation_position="top left")

            fig.update_layout(
                title=title,
                xaxis_title='Frequency (Hz)',
                yaxis_title='Normalized Magnitude',
                height=350,
                margin=dict(t=50, b=40, l=40, r=20),
                template="plotly_white",
                xaxis_rangemode='tozero',
                yaxis_rangemode='tozero'
            )
             # Wrap each graph in a Card and Column
            graph_cols.append(dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig))), width=12, md=col_md, lg=col_lg, className="mb-3")) # Added bottom margin
        # Return players and the row containing graph columns
        return players, dbc.Row(graph_cols, className="g-3") # Add gutter spacing


# Callback to run the gender detection model test
@app.callback(
    Output('test-result-output', 'children'),     # Output the prediction result
    Input('run-test-btn', 'n_clicks'),           # Triggered by the test button
    State('test-audio-selector', 'value'),       # Get the selected audio type (original, downsampled, etc.)
    State('original-audio-data', 'data'),        # Get original audio data
    State('downsampled-audio-data', 'data'),     # Get downsampled audio data
    State('reconstructed-audio-data', 'data'),   # Get reconstructed audio data
    prevent_initial_call=True                    # Don't run on initial load
)
def run_model_test(n_clicks, selected_audio_type, original_data, downsampled_data, reconstructed_data):
    """Runs the gender detection model on the selected audio version."""
    if n_clicks == 0: return dash.no_update # Prevent running on initial load if button wasn't clicked

    if not GENDER_MODEL_LOADED:
        return dbc.Alert("Gender model is not available for testing.", color="warning")

    # Map the dropdown value to the actual data stored in dcc.Store
    data_map = {'original': original_data, 'downsampled': downsampled_data, 'reconstructed': reconstructed_data}
    selected_data = data_map.get(selected_audio_type)

    # Check if the selected audio data exists
    if selected_data is None:
        # Provide more specific feedback if reconstruction hasn't happened yet
        if selected_audio_type == 'reconstructed' and reconstructed_data is None:
             msg = "Please click 'Apply AI Reconstruction' first to generate reconstructed audio."
        elif selected_audio_type == 'downsampled' and downsampled_data is None:
             msg = "Please select a downsample frequency first."
        else:
             msg = f"'{selected_audio_type.capitalize()}' audio data not available."
        return dbc.Alert(msg, color="warning")


    # Get waveform and sample rate
    try:
        y = np.array(selected_data['waveform'])
        sr = selected_data['sample_rate']
        # Perform prediction
        print(f"Running gender prediction on '{selected_audio_type}' audio...") # Add print
        prediction = predict_gender(y, sr)
        print(f"Prediction result: {prediction}") # Add print
        # Return result wrapped in a Bootstrap Alert
        alert_color = "primary" if "Predicted" in prediction else "danger" # Use primary color for success
        return dbc.Alert(f"Test Result ({selected_audio_type.capitalize()}): {prediction}", color=alert_color)
    except Exception as e:
        print(f"❌ Error during gender test execution: {e}")
        return dbc.Alert(f"Error during test: {e}", color="danger")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the Dash app server
    app.run(debug=True, port=8054) # Run on port 8054

