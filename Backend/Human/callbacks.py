"""
Callback functions for Human application
"""
import io
import base64
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, no_update, html, dcc

# Import shared modules
from shared.audio_processing import audio_processor
from shared.plotting_utils import plotting_utils

# Import local modules
from utils import analyze_audio, apply_model_reconstruction, predict_gender
from layout import create_audio_players

def register_callbacks(app):
    """Register all callbacks for the Human application"""
    
    @app.callback(
        Output('original-audio-data', 'data'),
        Output('upload-status', 'children'),
        Output('model-status', 'children'),
        Output('gender-prediction-output', 'children'),
        Input('upload-audio', 'contents'),
        State('upload-audio', 'filename')
    )
    def upload_audio_file(contents, filename):
        """Process uploaded audio files"""
        from models import MODEL_LOADED, GENDER_MODEL_LOADED
        
        # Model status display
        recon_alert_color = "success" if MODEL_LOADED else "danger"
        gender_alert_color = "success" if GENDER_MODEL_LOADED else "danger"
        recon_alert = dbc.Badge(f"Reconstruction AI: {'OK' if MODEL_LOADED else 'FAIL'}", 
                               color=recon_alert_color, className="me-2")
        gender_alert = dbc.Badge(f"Gender AI: {'OK' if GENDER_MODEL_LOADED else 'FAIL'}", 
                                color=gender_alert_color)
        model_status_div = html.Div([recon_alert, gender_alert])

        if contents is None:
            return None, dbc.Alert("Please upload a WAV, MP3, or FLAC file.", color="info"), model_status_div, ""

        try:
            # Use shared audio processor to load audio
            audio_data, sr, error = audio_processor.load_audio_from_base64(contents, filename)
            
            if audio_data is None:
                return None, dbc.Alert(f"❌ {error}", color="danger"), model_status_div, ""

            # Perform initial gender prediction
            gender_text = "Initial Upload: " + predict_gender(audio_data, sr)
            status_msg = dbc.Alert(f"✅ Loaded: {filename} ({len(audio_data) / sr:.2f}s)", color="success")
            
            # Analyze the audio and store data
            return analyze_audio(audio_data, sr), status_msg, model_status_div, gender_text
            
        except Exception as e:
            return None, dbc.Alert(f"❌ Error loading file: {e}", color="danger"), model_status_div, ""

    @app.callback(
        Output('downsampled-audio-data', 'data'),
        Output('reconstructed-audio-data', 'data'),
        Input('target-sr-slider', 'value'),
        Input('model-reconstruct-btn', 'n_clicks'),
        State('original-audio-data', 'data')
    )
    def process_audio(target_sr, model_clicks, original_data):
        """Downsample audio and apply reconstruction"""
        if original_data is None:
            return None, None
            
        ctx = callback_context
        triggered_id = ctx.triggered_id if ctx.triggered else 'No-ID'

        # Get original audio data
        y_original = np.array(original_data['waveform'])
        sr_original = original_data['sample_rate']

        # Downsampling logic (same as original)
        if target_sr >= sr_original:
            y_downsampled = y_original
            downsampled_sr = sr_original
        else:
            ratio = sr_original / target_sr
            new_len = int(len(y_original) / ratio)
            indices = (np.arange(new_len) * ratio).astype(int)
            indices = indices[indices < len(y_original)]
            y_downsampled = y_original[indices]
            downsampled_sr = target_sr

        current_downsampled_data = analyze_audio(y_downsampled, downsampled_sr)

        # Apply AI reconstruction only if button was clicked
        if triggered_id == 'model-reconstruct-btn' and model_clicks > 0:
            y_input = np.array(current_downsampled_data['waveform'])
            sr_input = current_downsampled_data['sample_rate']
            y_reconstructed = apply_model_reconstruction(y_input, sr_input)
            reconstructed_data = analyze_audio(y_reconstructed, 16000)
            return current_downsampled_data, reconstructed_data

        if triggered_id == 'target-sr-slider':
            return current_downsampled_data, no_update
        else:
            return current_downsampled_data, None

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
        """Update the audio players and spectrum plots"""
        if original_data is None:
            return dbc.Alert("Upload an audio file to begin.", color="info"), None

        # Create audio players
        players = create_audio_players(original_data, downsampled_data, reconstructed_data)

        # Prepare data for plotting based on selected signals
        all_signals = {
            'original': {'data': original_data, 'name': 'Original', 'color': '#0d6efd'},
            'downsampled': {'data': downsampled_data, 'name': 'Downsampled', 'color': '#dc3545'},
            'reconstructed': {
                'data': reconstructed_data,
                'name': "AI Reconstructed",
                'color': '#198754'
            }
        }

        # Filter signals based on user checklist selection and data availability
        signals_to_plot = {key: val for key, val in all_signals.items() if
                           key in selected_signals and val['data'] is not None}

        if not signals_to_plot:
            return players, dbc.Alert("Select a signal to display its graph.", color="info")

        # Generate plot(s) based on view mode
        if view_mode == 'overlap':
            fig = go.Figure()
            nyquist_freq = None
            
            for key, sig_info in signals_to_plot.items():
                data = sig_info['data']
                fig.add_trace(go.Scattergl(
                    x=data['frequencies'], 
                    y=data['magnitude'],
                    name=f"{sig_info['name']} ({data['sample_rate']/1000:.1f} kHz)",
                    line=dict(color=sig_info['color'], width=2)
                ))
                
                if key == 'downsampled':
                    nyquist_freq = data['nyquist_freq']

            if nyquist_freq is not None:
                fig.add_vline(
                    x=nyquist_freq, 
                    line_dash="dash", 
                    line_color="#dc3545",
                    annotation_text=f"Nyquist: {nyquist_freq/1000:.1f} kHz",
                    annotation_position="top left"
                )

            fig.update_layout(
                title='Frequency Spectrum Comparison',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Normalized Magnitude',
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                xaxis_rangemode='tozero',
                yaxis_rangemode='tozero'
            )
            
            graph_card = dbc.Card(dbc.CardBody(dcc.Graph(figure=fig)))
            return players, graph_card

        else:  # 'separate' view
            graph_cols = []
            num_signals = len(signals_to_plot)
            col_md = 12 // num_signals if num_signals > 0 and num_signals <= 3 else 4

            for key, sig_info in signals_to_plot.items():
                data = sig_info['data']
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=data['frequencies'], 
                    y=data['magnitude'],
                    name=sig_info['name'], 
                    line=dict(color=sig_info['color'])
                ))
                
                title = f"{sig_info['name']} ({data['sample_rate']/1000:.1f} kHz)"
                nyquist_freq_sep = None
                
                if key == 'downsampled':
                    nyquist_freq_sep = data['nyquist_freq']
                    fig.add_vline(
                        x=nyquist_freq_sep, 
                        line_dash="dash", 
                        line_color="#dc3545",
                        annotation_text=f"Nyquist: {nyquist_freq_sep/1000:.1f} kHz",
                        annotation_position="top left"
                    )

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
                
                graph_cols.append(
                    dbc.Col(
                        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig))), 
                        width=12, md=col_md, lg=col_md, className="mb-3"
                    )
                )
            
            return players, dbc.Row(graph_cols, className="g-3")

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
        """Run gender detection test on selected audio"""
        from models import GENDER_MODEL_LOADED
        
        if n_clicks == 0:
            return no_update

        if not GENDER_MODEL_LOADED:
            return dbc.Alert("Gender model is not available for testing.", color="warning")

        data_map = {
            'original': original_data, 
            'downsampled': downsampled_data, 
            'reconstructed': reconstructed_data
        }
        selected_data = data_map.get(selected_audio_type)

        if selected_data is None:
            if selected_audio_type == 'reconstructed' and reconstructed_data is None:
                msg = "Please click 'Apply AI Reconstruction' first to generate reconstructed audio."
            elif selected_audio_type == 'downsampled' and downsampled_data is None:
                msg = "Please select a downsample frequency first."
            else:
                msg = f"'{selected_audio_type.capitalize()}' audio data not available."
            return dbc.Alert(msg, color="warning")

        try:
            y = np.array(selected_data['waveform'])
            sr = selected_data['sample_rate']
            prediction = predict_gender(y, sr)
            alert_color = "primary" if "Predicted" in prediction else "danger"
            return dbc.Alert(f"Test Result ({selected_audio_type.capitalize()}): {prediction}", color=alert_color)
        except Exception as e:
            return dbc.Alert(f"Error during test: {e}", color="danger")