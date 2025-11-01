# callbacks.py
import base64
import io
import numpy as np
import dash
from dash import Input, Output, State, callback_context, no_update, html, dcc
import dash_bootstrap_components as dbc  # ✅ Critical import
import plotly.graph_objects as go
import soundfile as sf

from audio_processor import analyze_audio, create_audio_data
from models import MODEL_LOADED, GENDER_MODEL_LOADED, apply_model_reconstruction, predict_gender
from Backend.shared.downsample import downsample_without_anti_aliasing

def create_audio_players(original_data, downsampled_data, reconstructed_data):
    cols = []
    cols.append(dbc.Col([
        html.H5("1. Original Audio", className="text-primary"),
        html.Audio(controls=True, style={'width': '100%'},
                   src=create_audio_data(original_data['waveform'], original_data['sample_rate']))
    ], width=12, lg=4, className="mb-3 mb-lg-0"))

    if downsampled_data is not None:
        cols.append(dbc.Col([
            html.H5("2. Downsampled (Aliased) Audio", className="text-danger"),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(downsampled_data['waveform'], downsampled_data['sample_rate']))
        ], width=12, lg=4, className="mb-3 mb-lg-0"))

    if reconstructed_data is not None:
        title = "3. AI Reconstructed Audio" if MODEL_LOADED else "3. Upsampled Audio (Fallback)"
        color_class = 'text-success' if MODEL_LOADED else 'text-warning'
        cols.append(dbc.Col([
            html.H5(title, className=color_class),
            html.Audio(controls=True, style={'width': '100%'},
                       src=create_audio_data(reconstructed_data['waveform'], reconstructed_data['sample_rate']))
        ], width=12, lg=4))

    return dbc.Row(cols, className="mt-3 g-3")


def register_callbacks(app):

    @app.callback(
        Output('original-audio-data', 'data'),
        Output('upload-status', 'children'),
        Output('model-status', 'children'),
        Output('gender-prediction-output', 'children'),
        Input('upload-audio', 'contents'),
        State('upload-audio', 'filename')
    )
    def upload_audio_file(contents, filename):
        # Model status badges
        recon_alert = dbc.Badge(
            f"Reconstruction AI: {'OK' if MODEL_LOADED else 'FAIL'}",
            color="success" if MODEL_LOADED else "danger",
            className="me-2"
        )
        gender_alert = dbc.Badge(
            f"Gender AI: {'OK' if GENDER_MODEL_LOADED else 'FAIL'}",
            color="success" if GENDER_MODEL_LOADED else "danger"
        )
        model_status_div = html.Div([recon_alert, gender_alert])

        if contents is None:
            return None, dbc.Alert("Please upload a WAV, MP3, or FLAC file.", color="info"), model_status_div, ""

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            y, sr = sf.read(io.BytesIO(decoded), dtype='float32')

            # Handle stereo → mono
            if y.ndim > 1:
                y = y.mean(axis=1)

            # Validate audio
            if y.size == 0:
                raise ValueError("Audio file is empty.")

            # Limit to 20 seconds
            if len(y) > sr * 20:
                y = y[:int(sr * 20)]

            # Initial gender prediction
            gender_text = "Initial Upload: " + predict_gender(y, sr)
            status_msg = dbc.Alert(f"✅ Loaded: {filename} ({len(y) / sr:.2f}s)", color="success")
            original_analysis = analyze_audio(y, sr)

            return original_analysis, status_msg, model_status_div, gender_text

        except Exception as e:
            import traceback
            print("🔥 UPLOAD ERROR:")
            traceback.print_exc()
            return None, dbc.Alert(f"❌ Error loading file: {str(e)}", color="danger"), model_status_div, ""


    @app.callback(
        Output('downsampled-audio-data', 'data'),
        Output('reconstructed-audio-data', 'data'),
        Input('target-sr-slider', 'value'),
        Input('model-reconstruct-btn', 'n_clicks'),
        State('original-audio-data', 'data')
    )
    def process_audio(target_sr, model_clicks, original_data):
        if original_data is None:
            return None, None

        ctx = callback_context
        triggered_id = ctx.triggered_id if ctx.triggered else 'No-ID'

        y_original = np.array(original_data['waveform'])
        sr_original = original_data['sample_rate']

        y_down, ds_sr = downsample_without_anti_aliasing(y_original, sr_original, target_sr)
        current_downsampled_data = analyze_audio(y_down, ds_sr)

        if triggered_id == 'model-reconstruct-btn' and model_clicks > 0:
            y_recon = apply_model_reconstruction(y_down, ds_sr)
            reconstructed_data = analyze_audio(y_recon, 16000)
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
        if original_data is None:
            return dbc.Alert("Upload an audio file to begin.", color="info"), None

        players = create_audio_players(original_data, downsampled_data, reconstructed_data)

        all_signals = {
            'original': {'data': original_data, 'name': 'Original', 'color': '#0d6efd'},
            'downsampled': {'data': downsampled_data, 'name': 'Downsampled', 'color': '#dc3545'},
            'reconstructed': {
                'data': reconstructed_data,
                'name': "AI Reconstructed" if MODEL_LOADED else "Upsampled (Fallback)",
                'color': '#198754' if MODEL_LOADED else '#ffc107'
            }
        }

        signals_to_plot = {
            k: v for k, v in all_signals.items()
            if k in selected_signals and v['data'] is not None
        }

        if not signals_to_plot:
            return players, dbc.Alert("Select a signal to display its graph.", color="info")

        if view_mode == 'overlap':
            fig = go.Figure()
            nyquist_freq = None
            for key, sig in signals_to_plot.items():
                d = sig['data']
                fig.add_trace(go.Scattergl(
                    x=d['frequencies'],
                    y=d['magnitude'],
                    name=f"{sig['name']} ({d['sample_rate']/1000:.1f} kHz)",
                    line=dict(color=sig['color'], width=2)
                ))
                if key == 'downsampled':
                    nyquist_freq = d['nyquist_freq']
            if nyquist_freq is not None:
                fig.add_vline(
                    x=nyquist_freq,
                    line_dash="dash",
                    line_color="#dc3545",
                    annotation_text=f"Nyquist: {nyquist_freq/1000:.1f} kHz"
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
        else:
            graph_cols = []
            num = len(signals_to_plot)
            col_width = 12 // num if num <= 3 else 4
            for key, sig in signals_to_plot.items():
                d = sig['data']
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=d['frequencies'],
                    y=d['magnitude'],
                    name=sig['name'],
                    line=dict(color=sig['color'])
                ))
                title = f"{sig['name']} ({d['sample_rate']/1000:.1f} kHz)"
                if key == 'downsampled':
                    fig.add_vline(
                        x=d['nyquist_freq'],
                        line_dash="dash",
                        line_color="#dc3545",
                        annotation_text=f"Nyquist: {d['nyquist_freq']/1000:.1f} kHz"
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
                        width=12, md=col_width, lg=col_width,
                        className="mb-3"
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
            import traceback
            print("🔥 TEST ERROR:")
            traceback.print_exc()
            return dbc.Alert(f"Error during test: {str(e)}", color="danger")