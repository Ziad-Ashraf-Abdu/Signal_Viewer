import sys
import os

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

    # Add other callbacks as needed...

    print("✅ Sound app callbacks registered successfully!")