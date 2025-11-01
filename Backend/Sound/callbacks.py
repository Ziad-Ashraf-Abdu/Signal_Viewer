import sys
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots # <-- ADD THIS LINE
import traceback

# Correct path setup - shared is in Backend/shared/
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_path = os.path.join(current_dir, '..', 'shared')

if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

import io
import base64
import numpy as np
import plotly.graph_objs as go
import traceback
from dash import Input, Output, State, html, no_update, callback_context
from dash.exceptions import PreventUpdate

# Import from shared modules
from shared.config import *
from shared.audio_processing import AudioProcessor
from shared.analysis_utils import AnalysisUtils
from shared.plotting_utils import PlottingUtils
from shared.model_utils import model_manager, ensure_model_loaded
from shared.ui_components import create_button

# Initialize shared utilities
audio_processor = AudioProcessor()
analysis_utils = AnalysisUtils()
plotting_utils = PlottingUtils()

def register_callbacks(app):
    """Registers all callbacks for the Sound app using shared components."""

    @app.callback(
        Output("classify-btn", "disabled"),
        Output("classify-btn", "style"),
        Input("upload-audio", "contents"),
        State("classify-btn", "disabled"),
    )
    def update_button_state(contents, current_disabled_state):
        """Enable/disable the 'Analyze Original Audio' button based on file upload."""
        is_disabled = not contents
        if is_disabled == current_disabled_state:
            raise PreventUpdate

        new_style = get_button_style("primary")
        if is_disabled:
            new_style.update(BUTTON_DISABLED_STYLE)

        return is_disabled, new_style

    @app.callback(
        Output("results-card", "style", allow_duplicate=True),
        Output("file-name", "children"),
        Output("waveform-plot", "figure", allow_duplicate=True),
        Output("audio-player", "src"),
        Output("audio-data-store", "data"),
        Output("classification-result", "children", allow_duplicate=True),
        Output("show-sampling-btn", "style", allow_duplicate=True),
        Output("sampling-controls", "style", allow_duplicate=True),
        Output("upload-error-output", "children"),
        Output("sampled-classification-result", "children", allow_duplicate=True),
        Output("sampled-classification-result", "style", allow_duplicate=True),
        Output("sampled-audio-player", "src", allow_duplicate=True),
        Output("sampled-audio-player", "style", allow_duplicate=True),
        Input("upload-audio", "contents"),
        State("upload-audio", "filename"),
        prevent_initial_call=True,
    )
    def display_uploaded_audio(contents, filename):
        """Handles file uploads using shared audio processor."""
        if not contents:
            raise PreventUpdate

        # Default values for reset
        default_result_style = {
            "fontSize": "18px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
            "padding": "20px", "borderRadius": "10px", 'transition': 'all 0.3s ease-in-out',
            "border": f"1px dashed {BORDER_COLOR}", 'backgroundColor': BACKGROUND_COLOR, 'minHeight': '60px',
            "color": SUBTLE_TEXT_COLOR
        }

        try:
            # Use shared audio processor
            audio_data, sr, error_msg = audio_processor.load_audio_from_base64(contents, filename)
            
            if error_msg:
                return (
                    {'display': 'none'}, "", go.Figure(), None, None, "",
                    {'display': 'none'}, {'display': 'none'}, error_msg,
                    "Prediction result for sampled audio will appear here.", default_result_style,
                    "", {'display':'none'}
                )

            # Create initial plot using shared plotting
            fig = plotting_utils.create_waveform_plot(audio_data, sr, "Waveform Preview")
            
            # Store audio data
            store_data = {
                'original_audio': audio_data.tolist(), 
                'original_sr': sr, 
                'filename': filename
            }
            
            # Create playable audio
            audio_src = audio_processor.make_playable_wav(audio_data, sr)
            
            initial_text = html.Span(
                "File loaded. Click 'Analyze Original Audio' to classify.", 
                style={'color': SUBTLE_TEXT_COLOR, 'fontStyle': 'italic', 'fontSize': '16px'}
            )

            return (
                {**CARD_STYLE, 'display': 'block'},
                f"File: {filename}",
                fig,
                audio_src,
                store_data,
                initial_text,
                {'display': 'none'}, {'display': 'none'}, "",
                "Prediction result for sampled audio will appear here.", default_result_style,
                "", {'display':'none'}
            )

        except Exception as e:
            error_msg = f"⚠️ Error reading audio file '{filename}': {str(e)}"
            return (
                {'display': 'none'}, "", go.Figure(), None, None, "",
                {'display': 'none'}, {'display': 'none'}, error_msg,
                "Prediction result for sampled audio will appear here.", default_result_style,
                "", {'display':'none'}
            )

    @app.callback(
        Output("classification-result", "children", allow_duplicate=True),
        Output("classification-result", "style"),
        Output("show-sampling-btn", "style"),
        Input("classify-btn", "n_clicks"),
        State("audio-data-store", "data"),
        prevent_initial_call=True,
    )
    def classify_original_audio(n_clicks, store_data):
        """Runs AI classification on the original, stored audio data."""
        if not n_clicks or not store_data:
            raise PreventUpdate

        try:
            # --- 1. Load the AI Model ---
            # Use the correct model key from model_utils.py
            model_name = "drone_detection" 
            ensure_model_loaded(model_name)
            
            # --- 2. Get Audio from Store ---
            audio_data = np.array(store_data.get('original_audio'))
            sr = store_data.get('original_sr')

            if audio_data is None or sr is None:
                return "Error: Could not find audio data. Please re-upload.", {}, {'display': 'none'}

            # --- 3. Get Prediction ---
            # Call predict_with_transformers directly.
            # This method handles its own preprocessing.
            results = model_manager.predict_with_transformers(
                model_key=model_name,
                audio_data=audio_data,
                sr=sr
            )
            
            if not results or "error" in results[0]:
                error_msg = results[0]["error"] if results else "No prediction returned"
                raise Exception(f"Prediction failed: {error_msg}")

            # --- 4. Parse the Output ---
            # Aggregate results by taking the prediction with the highest score
            # from all chunks.
            best_result = max(results, key=lambda x: x.get('score', 0))
            prediction = best_result.get('label', 'UNKNOWN')
            confidence = best_result.get('score', 0)
            confidence_pct = confidence * 100
            
            # --- 5. Format the Output ---
            if prediction == "drone":
                icon = "🚁"
                color = SUCCESS_COLOR
                result_text = f"Drone Detected"
                border_style = f"2px solid {color}"
            elif prediction == "not_drone":
                icon = "🌳"
                color = WARNING_COLOR
                result_text = f"Background Noise"
                border_style = f"2px solid {color}"

            result_style = {
                "fontSize": "22px", "fontWeight": "700", "textAlign": "center", "marginBottom": "20px",
                "padding": "20px", "borderRadius": "10px", 'color': color,
                'border': border_style, 'backgroundColor': f'{color}15' 
            }

            formatted_output = html.Div([
                html.Span(icon, style={'fontSize': '32px'}),
                html.H3(result_text, style={'margin': '10px 0 5px 0'}),
                html.P(f"Confidence: {confidence_pct:.2f}%", style={'fontSize': '16px', 'color': SUBTLE_TEXT_COLOR, 'margin':'0'})
            ])
            
            # --- 6. Show the next button ---
            # We need to get the base style and update it
            show_sampling_btn_style = get_button_style("secondary")
            show_sampling_btn_style.update({'display': 'block', 'width': '100%', 'marginTop': '15px'})

            return formatted_output, result_style, show_sampling_btn_style

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error during classification: {e}\n{tb_str}")
            error_style = {
                "fontSize": "16px", "fontWeight": "500", "textAlign": "center", "marginBottom": "20px",
                "padding": "20px", "borderRadius": "10px", 'color': ERROR_COLOR,
                'border': f'2px solid {ERROR_COLOR}', 'backgroundColor': f'{ERROR_COLOR}15'
            }
            return f"⚠️ Analysis Failed: {str(e)}", error_style, {'display': 'none'}
        
    @app.callback(
        Output("sampling-controls", "style", allow_duplicate=True),
        Output("nyquist-info", "children"),
        Output("sampling-freq-slider", "max"),
        Output("sampling-freq-slider", "value"),
        Output("sampling-freq-slider", "marks"),
        Input("show-sampling-btn", "n_clicks"),
        State("audio-data-store", "data"),
        prevent_initial_call=True,
    )
    def show_sampling_controls(n_clicks, store_data):
        """Displays the interactive sampling controls and calculates Nyquist info."""
        if not n_clicks or not store_data:
            raise PreventUpdate

        try:
            audio_data = np.array(store_data.get('original_audio'))
            sr = store_data.get('original_sr')

            if audio_data is None or sr is None:
                return {'display': 'none'}, "Error: No audio data.", no_update, no_update, no_update

            # --- 1. Analyze Audio for Frequencies ---
            # Use the shared audio processor function
            nyquist_info = audio_processor.calculate_nyquist_info(audio_data, sr)
            
            max_freq = nyquist_info.get('max_freq', 0)
            nyquist_rate = nyquist_info.get('nyquist_rate', 0)
            original_sr = nyquist_info.get('original_sr', 0)

            # --- 2. Create Info Display ---
            info_style = {'textAlign': 'left', 'paddingLeft': '10px', 'borderLeft': f'3px solid {BORDER_COLOR}'}
            label_style = {'fontSize': '12px', 'color': SUBTLE_TEXT_COLOR, 'display': 'block', 'marginBottom': '3px'}
            value_style = {'fontSize': '16px', 'fontWeight': '600', 'color': TEXT_COLOR}

            info_children = [
                html.Div([
                    html.Span("Original SR (Fs)", style=label_style),
                    html.Span(f"{original_sr / 1000:.1f} kHz", style=value_style)
                ], style={**info_style, 'borderLeftColor': '#667eea'}),
                html.Div([
                    html.Span("Max. Content Freq (Fmax)", style=label_style),
                    html.Span(f"{max_freq / 1000:.1f} kHz", style=value_style)
                ], style={**info_style, 'borderLeftColor': '#764ba2'}),
                html.Div([
                    html.Span("Nyquist Rate (2*Fmax)", style=label_style),
                    html.Span(f"{nyquist_rate / 1000:.1f} kHz", style={**value_style, 'color': SUCCESS_COLOR})
                ], style={**info_style, 'borderLeftColor': SUCCESS_COLOR}),
            ]
            
            # --- 3. Configure Slider ---
            # Set slider max to the original SR (or 48k if higher)
            slider_max = max(original_sr, 48000) 
            # Default value from layout.py was 8000
            slider_value = min(8000, original_sr) 

            slider_marks = {
                500: '0.5k',
                int(nyquist_rate): {'label': 'Nyquist', 'style': {'color': SUCCESS_COLOR, 'fontWeight': 'bold'}},
                int(original_sr): {'label': f'Original ({original_sr/1000:.0f}k)', 'style': {'color': '#667eea'}},
                int(slider_max): f'{int(slider_max/1000)}k'
            }
            # Ensure no overlapping marks
            if abs(nyquist_rate - original_sr) < 2000:
                del slider_marks[int(original_sr)]

            # --- 4. Show Controls ---
            controls_style = {'display': 'block', 'marginTop': '25px'}

            return controls_style, info_children, slider_max, slider_value, slider_marks

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in show_sampling_controls: {e}\n{tb_str}")
            return {'display': 'none'}, f"Error: {str(e)}", no_update, no_update, no_update
    
    @app.callback(
        Output("waveform-plot", "figure", allow_duplicate=True),
        Output("playback-warning", "children"),
        Input("sampling-freq-slider", "value"),
        State("audio-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_waveform_on_slider(new_sr, store_data):
        """
        Generates a Plotly figure comparing a small segment of the original audio
        with the points that would be sampled at the new_sr.
        This logic is taken directly from the user's provided app.py.
        """
        if not new_sr or not store_data:
            raise PreventUpdate

        try:
            # --- 1. Get Data ---
            original_audio = np.array(store_data.get('original_audio'))
            original_sr = store_data.get('original_sr')

            if original_audio.size == 0:
                return no_update, no_update
            
            # --- 2. Calculate Aliasing Info ---
            nyquist_info = audio_processor.calculate_nyquist_info(original_audio, original_sr)
            max_freq = nyquist_info.get('max_freq', 500.0) # Default to 500 if fails
            nyquist_rate = 2 * max_freq
            
            fig = go.Figure()
            title_text = "Waveform Sampling (Zoomed View)"
            is_aliasing = new_sr < nyquist_rate
            warning_text = ""

            if is_aliasing:
                title_text = f"⚠ Aliasing Likely (Fs={new_sr} Hz < Nyquist={nyquist_rate:.0f} Hz)"
                title_font_color = ERROR_COLOR
                warning_text = f"Warning: Fs ({new_sr/1000:.1f}k) is below Nyquist ({nyquist_rate/1000:.1f}k). Audio will be aliased/distorted!"
            else:
                title_font_color = TEXT_COLOR
                warning_text = "" # No warning

            # --- 3. Get 50ms Segment ---
            display_duration_s = 0.05
            display_samples_orig = int(min(len(original_audio), original_sr * display_duration_s))

            if display_samples_orig < 2:
                fig.update_layout(title="Audio segment too short to display sampling visualization.")
                return fig, "Audio segment too short for visualization."

            original_audio_segment = original_audio[:display_samples_orig]
            time_original = np.linspace(0, display_samples_orig / original_sr, num=display_samples_orig)

            # --- 4. Plot Faint Original Signal ---
            fig.add_trace(go.Scattergl(
                x=time_original, 
                y=original_audio_segment, 
                mode="lines",
                line=dict(color='rgba(150, 150, 150, 0.5)', width=2), 
                name="Original Signal"
            ))

            # --- 5. Calculate Sample Points ---
            num_samples_new = int(display_duration_s * new_sr)
            if num_samples_new < 2:
                fig.update_layout(title=title_text + " - (Target SR too low for visualization)", title_font_color=title_font_color)
                return fig, warning_text

            max_index = display_samples_orig - 1
            sample_indices_orig = np.linspace(0, max_index, num=num_samples_new, dtype=int)
            sample_indices_orig = np.clip(sample_indices_orig, 0, max_index)

            sampled_audio = original_audio_segment[sample_indices_orig]
            time_sampled = sample_indices_orig / original_sr 

            # --- 6. Plot Sampled Signal (Lines + Markers) ---
            line_color = ERROR_COLOR if is_aliasing else PRIMARY_COLOR
            fig.add_trace(go.Scattergl(
                x=time_sampled, 
                y=sampled_audio, 
                mode="lines+markers",
                line=dict(color=line_color, width=1.5),
                marker=dict(color=line_color, size=7, symbol='circle-open'),
                name=f"Sampled at {new_sr} Hz"
            ))

            # --- 7. Apply Final Layout ---
            fig.update_layout(
                title=dict(text=title_text, font=dict(color=title_font_color, size=16)),
                margin=dict(l=50, r=30, t=60, b=50), 
                height=320,
                xaxis_title="Time (s)", 
                yaxis_title="Amplitude",
                plot_bgcolor=BACKGROUND_COLOR, # Use our config colors
                paper_bgcolor=BACKGROUND_COLOR,
                font=dict(color=TEXT_COLOR),
                xaxis=dict(gridcolor=BORDER_COLOR),
                yaxis=dict(gridcolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.7)')
            )
            
            return fig, warning_text

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in update_waveform_on_slider: {e}\n{tb_str}")
            return no_update, f"Error generating plot: {str(e)}"
    
    @app.callback(
        Output("sampled-audio-player", "src", allow_duplicate=True),
        Output("sampled-audio-player", "style", allow_duplicate=True),
        Input("play-sampled-btn", "n_clicks"),
        State("audio-data-store", "data"),
        State("sampling-freq-slider", "value"),
        prevent_initial_call=True,
    )
    def play_sampled_audio(n_clicks, store_data, target_sr):
        """
        Processes the original audio to simulate the target sampling frequency
        (with aliasing) and sends it to the sampled audio player.
        """
        if not n_clicks or not store_data or not target_sr:
            raise PreventUpdate

        try:
            # --- 1. Get Original Data ---
            original_audio = np.array(store_data.get('original_audio'))
            original_sr = store_data.get('original_sr')

            if original_audio.size == 0:
                return no_update, no_update

            # --- 2. Create Aliased/Sampled Audio ---
            # We use downsample_with_aliasing to intentionally create
            # the aliasing effect when target_sr < Nyquist rate.
            aliased_audio, new_sr = audio_processor.downsample_with_aliasing(
                original_audio,
                original_sr,
                target_sr
            )

            # --- 3. Make it Playable ---
            # make_playable_wav will handle upsampling this to a
            # browser-compatible rate (like 44.1k) while preserving
            # the aliased sound.
            audio_src = audio_processor.make_playable_wav(aliased_audio, new_sr)
            
            # --- 4. Return the audio source and make the player visible ---
            player_style = {"width": "100%", "marginTop": "15px", 'display': 'block'}
            
            return audio_src, player_style

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in play_sampled_audio: {e}\n{tb_str}")
            return no_update, no_update
        
    
    @app.callback(
        Output("sampled-classification-result", "children", allow_duplicate=True),
        Output("sampled-classification-result", "style", allow_duplicate=True),
        Input("predict-sampled-btn", "n_clicks"),
        State("audio-data-store", "data"),
        State("sampling-freq-slider", "value"),
        prevent_initial_call=True,
    )
    def predict_sampled_audio(n_clicks, store_data, target_sr):
        """
        Runs the AI classification on the newly sampled (and potentially
        aliased) audio data to show the effect of undersampling.
        """
        if not n_clicks or not store_data or not target_sr:
            raise PreventUpdate

        # --- Get the original classification style for comparison ---
        original_classification = callback_context.states.get('classification-result.children', {}).get('props', {}).get('children', [{}])[0].get('props', {}).get('children', 'UNKNOWN')

        try:
            # --- 1. Load the AI Model ---
            model_name = "drone_detection"
            ensure_model_loaded(model_name)

            # --- 2. Get Original Audio ---
            original_audio = np.array(store_data.get('original_audio'))
            original_sr = store_data.get('original_sr')

            if original_audio.size == 0:
                raise Exception("No audio data found.")

            # --- 3. Create Aliased/Sampled Audio ---
            aliased_audio, new_sr = audio_processor.downsample_with_aliasing(
                original_audio,
                original_sr,
                target_sr
            )

            if aliased_audio.size == 0:
                raise Exception("Aliased audio is empty, cannot predict.")

            # --- 4. Get Prediction on Aliased Audio ---
            results = model_manager.predict_with_transformers(
                model_key=model_name,
                audio_data=aliased_audio,
                sr=new_sr
            )

            if not results or "error" in results[0]:
                error_msg = results[0]["error"] if results else "No prediction returned"
                raise Exception(f"Prediction failed: {error_msg}")

            # --- 5. Parse and Format Output ---
            best_result = max(results, key=lambda x: x.get('score', 0))
            prediction = best_result.get('label', 'UNKNOWN')
            confidence = best_result.get('score', 0)
            confidence_pct = confidence * 100

            if prediction == "drone":
                icon = "🚁"
                color = SUCCESS_COLOR
                result_text = "Prediction: Drone"
            elif prediction == "not_drone":
                icon = "🌳"
                color = WARNING_COLOR
                result_text = "Prediction: Not_Drone"
                
            # --- 6. Style the Result Box ---
            result_style = {
                "fontSize": "18px", "fontWeight": "600", "textAlign": "center", "marginTop": "25px",
                "padding": "20px", "borderRadius": "10px", 'color': color,
                'border': f'2px solid {color}', 'backgroundColor': f'{color}15', 'minHeight': '60px'
            }
            
            formatted_output = html.Div([
                html.Span(icon, style={'fontSize': '28px'}),
                html.H4(result_text, style={'margin': '8px 0 3px 0'}),
                html.P(f"Confidence: {confidence_pct:.2f}%", style={'fontSize': '14px', 'color': SUBTLE_TEXT_COLOR, 'margin':'0'})
            ])

            return formatted_output, result_style

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error during sampled prediction: {e}\n{tb_str}")
            error_style = {
                "fontSize": "16px", "fontWeight": "500", "textAlign": "center", "marginTop": "25px",
                "padding": "20px", "borderRadius": "10px", 'color': ERROR_COLOR,
                'border': f'2px solid {ERROR_COLOR}', 'backgroundColor': f'{ERROR_COLOR}15', 'minHeight': '60px'
            }
            return f"⚠️ Sampled Analysis Failed: {str(e)}", error_style
            
    # Add other callbacks as needed...

    print("✅ Sound app callbacks registered successfully!")