# --- SEARCHABLE COMMENT: Imports ---
import io
import base64
import numpy as np
import soundfile as sf
import plotly.graph_objs as go
import traceback # For printing detailed error information

# Dash imports
from dash import Input, Output, State, html, no_update, callback_context
from dash.exceptions import PreventUpdate

# Local module imports
import config # Import style constants
import model_utils # Import model loading and prediction
import audio_utils # Import audio processing functions
import plotting    # Import plotting functions

# ============================================
# --- SEARCHABLE COMMENT: Register Callbacks Function ---
# Encapsulates all callback definitions.
# ============================================
def register_callbacks(app):
    """Registers all callbacks for the Dash app."""

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

        new_style = {**config.BUTTON_STYLE_PRIMARY} # Start with the default primary style
        if is_disabled:
            new_style.update(config.BUTTON_DISABLED_STYLE) # Apply disabled style if needed

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
            raise PreventUpdate

        # --- SEARCHABLE COMMENT: Base64 Decoding ---
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        err_msg = "" # Initialize error message variable

        try:
            # --- SEARCHABLE COMMENT: Audio File Reading ---
            audio_data, sr = sf.read(io.BytesIO(decoded))
            # --- SEARCHABLE COMMENT: Mono Conversion ---
            if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)

            # --- SEARCHABLE COMMENT: Audio Normalization ---
            if not np.issubdtype(audio_data.dtype, np.floating):
                max_val = np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else 32767.0
                if max_val != 0: audio_data = audio_data.astype(np.float32) / max_val
                else: audio_data = audio_data.astype(np.float32)
            elif np.max(np.abs(audio_data)) > 1.5:
                 print("Warning: Float audio data exceeds [-1, 1] range. Normalizing.")
                 max_abs = np.max(np.abs(audio_data))
                 if max_abs > 0: audio_data /= max_abs

        except Exception as e:
            # --- SEARCHABLE COMMENT: File Read Error Handling ---
            err_msg = f"⚠️ Error reading audio file '{filename}': Please ensure it's a valid audio format. ({e})"
            print(err_msg)
            return ({'display': 'none'}, "", go.Figure(), None, None, "",
                    {'display': 'none'}, {'display': 'none'}, err_msg,
                    "", no_update, "", {'display':'none'})

        # --- SEARCHABLE COMMENT: Initial Plot Creation ---
        fig = plotting.create_initial_figure(audio_data)
        # --- SEARCHABLE COMMENT: Storing Audio Data ---
        store_data = {'original_audio': audio_data.tolist(), 'original_sr': sr, 'filename': filename}
        initial_text = html.Span("File loaded. Click 'Analyze Original Audio' to classify.", style={'color': config.SUBTLE_TEXT_COLOR, 'fontStyle': 'italic', 'fontSize': '16px'})

        sampled_result_style = {
            "fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
            "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
            "border": f"1px dashed {config.BORDER_COLOR}", 'backgroundColor': config.BACKGROUND_COLOR, 'minHeight': '60px',
            "color": config.SUBTLE_TEXT_COLOR
        }

        # --- SEARCHABLE COMMENT: Update UI after Upload ---
        return (
            {**config.CARD_STYLE, 'display': 'block'},
            f"File: {filename}",
            fig,
            contents,
            store_data,
            initial_text,
            {'display': 'none'}, {'display': 'none'}, "",
            "Prediction result for sampled audio will appear here.", sampled_result_style,
            "", {'display':'none'}
        )


    # --- SEARCHABLE COMMENT: Analyze Original Audio Callback ---
    @app.callback(
        Output("classification-result", "children", allow_duplicate=True),
        Output("classification-result", "style", allow_duplicate=True),
        Output("show-sampling-btn", "style", allow_duplicate=True),
        Output("sampling-freq-slider", "max"),
        Output("sampling-freq-slider", "value"),
        Output("sampling-freq-slider", "marks"),
        Output("nyquist-info", "children"),
        Output("audio-data-store", "data", allow_duplicate=True),
        Input("classify-btn", "n_clicks"),
        State("audio-data-store", "data"),
        prevent_initial_call=True,
    )
    def analyze_original_audio(n_clicks, stored_data):
        """
        Runs AI classification on the original audio and calculates Nyquist info.
        """
        if not n_clicks or not stored_data or 'original_audio' not in stored_data:
            raise PreventUpdate

        try:
            audio_data = np.array(stored_data['original_audio'], dtype=np.float32)
            sr = stored_data['original_sr']
            filename = stored_data.get('filename', 'Unknown file')

            # --- SEARCHABLE COMMENT: AI Prediction (Original Audio) ---
            print(f"Analyzing original audio: {filename} ({len(audio_data)} samples @ {sr} Hz)")
            processor, model, device, load_err = model_utils.ensure_model()
            classification = ""
            top_pred_label = "N/A"

            if load_err:
                classification = html.Span(f"⚠️ Model Load Error: {load_err}", style={'color': config.ERROR_COLOR})
                top_pred_label = "Error"
            else:
                # --- SEARCHABLE COMMENT: Calling Prediction Function ---
                preds = model_utils.predict_with_local_model(processor, model, device, audio_data.copy(), sr)

                # --- SEARCHABLE COMMENT: Processing Prediction Results ---
                if preds and isinstance(preds, list) and len(preds) > 0:
                    if any("error" in p for p in preds):
                        error_msg = next((p['error'] for p in preds if 'error' in p), 'Unknown prediction error')
                        classification = html.Span(f"⚠️ Prediction Error: {error_msg}", style={'color': config.ERROR_COLOR})
                        top_pred_label = "Error"
                    elif "label" in preds[0] and "score" in preds[0]:
                        valid_preds = [p for p in preds if "error" not in p and "label" in p]
                        if valid_preds:
                            top_pred = max(valid_preds, key=lambda x: x.get('score', 0))
                            pred_color = config.PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else config.ERROR_COLOR
                            classification = html.Div([
                                html.Span("Original Prediction: "),
                                html.Strong(f"{top_pred['label']}", style={'color': pred_color}),
                                html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':config.SUBTLE_TEXT_COLOR})
                            ])
                            top_pred_label = top_pred['label']
                        else:
                            classification = html.Span("⚠️ Prediction error occurred in all chunks.", style={'color': config.WARNING_COLOR})
                            top_pred_label = "Error"
                    elif "message" in preds[0]:
                         classification = html.Span(f"⚠️ {preds[0]['message']}", style={'color': config.WARNING_COLOR})
                         top_pred_label = "N/A"
                    else:
                        classification = html.Span("⚠️ Prediction format unexpected.", style={'color': config.ERROR_COLOR})
                        top_pred_label = "Error"
                else:
                     classification = html.Span("⚠️ No prediction returned.", style={'color': config.WARNING_COLOR})
                     top_pred_label = "N/A"

            # --- SEARCHABLE COMMENT: Nyquist Rate Calculation ---
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
                    print(f"Warning: FFT analysis failed: {fft_err}")
                    if n > 0 and 'xf' in locals() and xf is not None and len(xf) > 0: max_freq = xf[-1]

            max_freq = max(500, max_freq)
            nyquist_rate = 2 * max_freq
            slider_default = max(1000, np.ceil(nyquist_rate / 100.0) * 100)
            initial_sampling_rate = int(min(sr + 500, slider_default + 500, sr))

            # --- SEARCHABLE COMMENT: Dynamic UI Styling (Classification Result) ---
            result_style = {
                "fontSize": "22px", "fontWeight": "700", "textAlign": "center", "marginBottom": "20px",
                "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                "border": "2px solid"
            }
            if 'drone' in top_pred_label.lower():
                result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': config.ERROR_COLOR, 'color': config.ERROR_COLOR})
            elif top_pred_label not in ["N/A", "Error"]:
                 result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': config.PRIMARY_COLOR, 'color': config.PRIMARY_COLOR})
            else:
                 result_style.update({'backgroundColor': config.BACKGROUND_COLOR, 'borderColor': config.BORDER_COLOR, 'color': config.TEXT_COLOR})

            # --- SEARCHABLE COMMENT: Update Stored Data ---
            stored_data['max_freq'] = max_freq

            # --- SEARCHABLE COMMENT: Slider Marks Configuration ---
            slider_max = max(sr + 1000, 48000)
            slider_marks = {
                 500: {'label': '0.5 kHz', 'style': {'fontSize': '11px'}},
                 int(nyquist_rate): {'label': f'Nyquist ({nyquist_rate / 1000:.1f} kHz)', 'style': {'color': config.PRIMARY_COLOR, 'fontWeight': 'bold', 'fontSize': '11px', 'whiteSpace':'nowrap'}},
                 int(slider_max): {'label': f'{int(slider_max/1000)} kHz', 'style': {'fontSize': '11px'}}
            }
            if abs(500 - nyquist_rate) < slider_max * 0.05: del slider_marks[500]
            if abs(slider_max - nyquist_rate) < slider_max * 0.05: del slider_marks[int(slider_max)]

            # --- SEARCHABLE COMMENT: Nyquist Info Text ---
            nyquist_text = [
                html.Div([html.Strong("Original SR:"), html.Span(f" {sr} Hz")]),
                html.Div([html.Strong("Est. Max Freq:"), html.Span(f" {max_freq:.0f} Hz")]),
                html.Div([html.Strong("Nyquist Rate:"), html.Span(f" {nyquist_rate:.0f} Hz")], style={'fontWeight': 'bold', 'color': config.PRIMARY_COLOR})
            ]

            # --- SEARCHABLE COMMENT: Show Sampling Button ---
            show_sampling_btn_style = {**config.BUTTON_STYLE_SECONDARY, 'width': '100%', 'marginTop': '15px', 'display': 'block'}

            # --- SEARCHABLE COMMENT: Return Values for Analyze Callback ---
            return (classification, result_style, show_sampling_btn_style, slider_max, initial_sampling_rate,
                    slider_marks, nyquist_text, stored_data)

        except Exception as e:
            # --- SEARCHABLE COMMENT: Analysis Error Handling ---
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            error_text = html.Div([html.Strong("⚠️ Analysis Error: "), f"{e}"])
            error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                           "padding": "15px", "borderRadius": "8px", 'backgroundColor': config.ERROR_COLOR,
                           "border": f"1px solid {config.ERROR_COLOR}"}
            return error_text, error_style, {'display':'none'}, no_update, no_update, no_update, no_update, no_update


    # --- SEARCHABLE COMMENT: Toggle Sampling Controls Callback ---
    @app.callback(
        Output("sampling-controls", "style", allow_duplicate=True),
        Input("show-sampling-btn", "n_clicks"),
        State("sampling-controls", "style"),
        prevent_initial_call=True,
    )
    def toggle_sampling_controls(n_clicks, current_style):
        """Toggles the visibility of the sampling controls div."""
        if n_clicks % 2 == 1:
            return {'display': 'block', 'marginTop': '25px'}
        else:
            return {'display': 'none', 'marginTop': '25px'}


    # --- SEARCHABLE COMMENT: Update Waveform on Slider Callback ---
    # --- SEARCHABLE COMMENT: Aliasing Plot Update ---
    @app.callback(
        Output("waveform-plot", "figure", allow_duplicate=True),
        Output("playback-warning", "children"),
        Input("sampling-freq-slider", "value"),
        State("audio-data-store", "data"),
        State("sampling-controls", "style"),
        prevent_initial_call=True,
    )
    def update_waveform_on_sample(new_sr, stored_data, sampling_style):
        """Updates the waveform plot when the slider changes and controls are visible."""
        # --- SEARCHABLE COMMENT: Prevent Plot Update if Hidden ---
        if not stored_data or 'max_freq' not in stored_data or sampling_style.get('display') == 'none':
            raise PreventUpdate

        try:
            audio_data = np.array(stored_data['original_audio'], dtype=np.float32)
            # --- SEARCHABLE COMMENT: Calling Resampled Plot Function ---
            fig = plotting.create_resampled_figure(audio_data, stored_data['original_sr'], new_sr, stored_data['max_freq'])

            # --- SEARCHABLE COMMENT: Aliasing/Playback Warning ---
            warning_text = ""
            is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2
            if new_sr < 3000:
                warning_text = html.Span("⚠️ Playback may fail or be distorted below ~3 kHz.", style={'color':config.WARNING_COLOR, 'fontWeight':'500'})
            elif is_aliasing:
                 warning_text = html.Span(["📉 Aliasing likely: ", html.B(f"Fs ({new_sr} Hz)"), f" < Nyquist ({stored_data.get('max_freq', 0) * 2:.0f} Hz)."], style={'color':config.WARNING_COLOR, 'fontWeight':'500'})

            return fig, warning_text
        except Exception as e:
            print(f"Error updating waveform plot: {e}")
            return no_update, html.Span(f"Error plotting: {e}", style={'color':config.ERROR_COLOR})


    # --- SEARCHABLE COMMENT: Play Sampled Audio Callback ---
    @app.callback(
        Output("sampled-audio-player", "src", allow_duplicate=True),
        Output("sampled-audio-player", "style", allow_duplicate=True),
        Input("play-sampled-btn", "n_clicks"),
        State("audio-data-store", "data"),
        State("sampling-freq-slider", "value"),
        prevent_initial_call=True,
    )
    def play_resampled_audio(n_clicks, stored_data, new_sr):
        """Resamples audio using decimation/sinc for playback."""
        if not n_clicks or not stored_data:
            return no_update, {'display': 'none'}

        try:
            original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
            original_sr = stored_data['original_sr']

            print(f"Resampling audio from {original_sr} Hz to {new_sr} Hz for playback (using decimation/sinc).")
            # --- SEARCHABLE COMMENT: Calling Playback Resampling Function ---
            processed_audio = audio_utils.resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

            # --- SEARCHABLE COMMENT: WAV Encoding ---
            buffer = io.BytesIO()
            if len(processed_audio) == 0:
                print("Warning: Resampled audio for playback is empty.")
                return "", {'display': 'none'}

            max_abs_val = np.max(np.abs(processed_audio)) if len(processed_audio) > 0 else 0
            processed_audio_int = np.zeros_like(processed_audio, dtype=np.int16)
            if max_abs_val > 0:
                processed_audio_int = np.int16(np.clip(processed_audio / max_abs_val * 32767, -32767, 32767))

            sf.write(buffer, processed_audio_int, int(new_sr), format='WAV', subtype='PCM_16')
            buffer.seek(0)

            # --- SEARCHABLE COMMENT: Data URI Creation ---
            encoded_sound = base64.b64encode(buffer.read()).decode()
            data_uri = f"data:audio/wav;base64,{encoded_sound}"

            return data_uri, {"width": "100%", "marginTop": "15px", 'display': 'block'}
        except Exception as e:
            # --- SEARCHABLE COMMENT: Playback Error Handling ---
            print(f"Error resampling audio for playback: {e}")
            traceback.print_exc()
            return "", {'display': 'none'}


    # --- SEARCHABLE COMMENT: Predict Sampled Audio Callback ---
    @app.callback(
        Output("sampled-classification-result", "children", allow_duplicate=True),
        Output("sampled-classification-result", "style", allow_duplicate=True),
        Input("predict-sampled-btn", "n_clicks"),
        State("audio-data-store", "data"),
        State("sampling-freq-slider", "value"),
        prevent_initial_call=True,
    )
    def predict_sampled_audio(n_clicks, stored_data, new_sr):
        """Resamples audio using decimation/sinc and runs prediction."""
        # Default message and style
        default_style = {
            "fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
            "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
            "border": f"1px dashed {config.BORDER_COLOR}", 'backgroundColor': config.BACKGROUND_COLOR, 'minHeight': '60px',
            "color": config.SUBTLE_TEXT_COLOR
        }
        if not n_clicks or not stored_data or 'original_audio' not in stored_data:
            return "Prediction result for sampled audio will appear here.", default_style

        # --- SEARCHABLE COMMENT: Loading State for Sampled Prediction ---
        loading_text = "🧠 Analyzing sampled audio..."
        loading_style = {**default_style, "border": f"1px solid {config.BORDER_COLOR}"} # Solid border while loading

        try:
            original_audio = np.array(stored_data['original_audio'], dtype=np.float32)
            original_sr = stored_data['original_sr']
            filename = stored_data.get('filename', 'Audio File')

            print(f"Resampling {filename} from {original_sr} Hz to {new_sr} Hz for prediction (using decimation/sinc).")
            # --- SEARCHABLE COMMENT: Resampling for Sampled Prediction ---
            resampled_audio_for_pred = audio_utils.resample_audio_decimation(original_audio.copy(), original_sr, new_sr)

            # --- SEARCHABLE COMMENT: AI Prediction (Sampled Audio) ---
            print(f"Predicting on resampled audio ({len(resampled_audio_for_pred)} samples @ {new_sr} Hz)...")
            processor, model, device, load_err = model_utils.ensure_model()
            classification_html = ""
            top_pred_label = "N/A"

            if load_err:
                classification_html = html.Span(f"⚠️ Model Load Error: {load_err}", style={'color': config.ERROR_COLOR})
                top_pred_label = "Error"
            else:
                # --- SEARCHABLE COMMENT: Calling Prediction on Sampled Data ---
                preds = model_utils.predict_with_local_model(processor, model, device, resampled_audio_for_pred, new_sr)

                # --- Process prediction results ---
                if preds and isinstance(preds, list) and len(preds) > 0:
                     if any("error" in p for p in preds):
                        error_msg = next((p['error'] for p in preds if 'error' in p), 'Unknown error')
                        classification_html = html.Span(f"⚠️ Error: {error_msg}", style={'color': config.ERROR_COLOR})
                        top_pred_label = "Error"
                     elif "label" in preds[0] and "score" in preds[0]:
                        valid_preds = [p for p in preds if "error" not in p and "label" in p]
                        if valid_preds:
                            top_pred = max(valid_preds, key=lambda x: x.get('score', 0))
                            pred_color = config.PRIMARY_COLOR if top_pred['label'].lower() != 'drone' else config.ERROR_COLOR
                            classification_html = html.Div([
                                html.Span(f"Sampled Prediction ({new_sr} Hz): "),
                                html.Strong(f"{top_pred['label']}", style={'color': pred_color}),
                                html.Span(f" ({top_pred['score'] * 100:.1f}%)", style={'fontSize':'0.9em', 'color':config.SUBTLE_TEXT_COLOR})
                            ])
                            top_pred_label = top_pred['label']
                        else:
                            classification_html = html.Span("⚠️ Prediction failed.", style={'color': config.WARNING_COLOR})
                            top_pred_label = "Error"
                     elif "message" in preds[0]:
                         classification_html = html.Span(f"⚠️ {preds[0]['message']}", style={'color': config.WARNING_COLOR})
                         top_pred_label = "N/A"
                     else:
                         classification_html = html.Span("⚠️ Unexpected format.", style={'color': config.ERROR_COLOR})
                         top_pred_label = "Error"
                else:
                     classification_html = html.Span(f"⚠️ No prediction returned.", style={'color': config.WARNING_COLOR})
                     top_pred_label = "N/A"

            # --- SEARCHABLE COMMENT: Dynamic UI Styling (Sampled Classification Result) ---
            result_style = {
                "fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px",
                "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
                "border": "2px solid", 'minHeight': '60px'
            }
            is_aliasing = new_sr < stored_data.get('max_freq', 0) * 2

            if 'drone' in top_pred_label.lower():
                result_style.update({'backgroundColor': '#FFF1F0', 'borderColor': config.ERROR_COLOR, 'color': config.ERROR_COLOR})
            elif top_pred_label not in ["N/A", "Error"]:
                 result_style.update({'backgroundColor': '#EBF8FF', 'borderColor': config.PRIMARY_COLOR, 'color': config.PRIMARY_COLOR})
            else:
                 result_style.update({'backgroundColor': config.BACKGROUND_COLOR, 'borderColor': config.BORDER_COLOR, 'color': config.TEXT_COLOR})

            # --- SEARCHABLE COMMENT: Aliasing Note ---
            aliasing_note_html = ""
            if is_aliasing:
                 aliasing_note_html = html.P(
                     "Note: Audio sampled below Nyquist rate (aliasing occurred).",
                     style={ 'fontSize': '12px', 'color': config.SUBTLE_TEXT_COLOR, 'marginTop': '10px',
                             'marginBottom': '0', 'fontStyle': 'italic' }
                 )

            return html.Div([classification_html, aliasing_note_html]), result_style

        except Exception as e:
            # --- SEARCHABLE COMMENT: Sampled Prediction Error Handling ---
            print(f"Error during sampled prediction: {e}")
            traceback.print_exc()
            error_text = html.Div([html.Strong("⚠️ Sampled Prediction Error: "), f"{e}"])
            error_style = {"fontSize": "18px", "fontWeight": "bold", "textAlign": "center", "color": "white",
                           "padding": "15px", "borderRadius": "8px", 'backgroundColor': config.ERROR_COLOR,
                           "border": f"1px solid {config.ERROR_COLOR}", 'minHeight': '60px', "marginTop": "25px"}
            return error_text, error_style
